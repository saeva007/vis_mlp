#!/bin/bash
#SBATCH -J lowvis_gate_select
#SBATCH -p kshdexclu01
# Single full-node post-processing job: one node, all 4 DCUs on that node.
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive
#
# Post-hoc threshold selection only: no model inference, no training, no checkpoint writes.
#
# Submit after both val and test diagnostics are available:
#   mkdir -p /public/home/putianshu/vis_mlp/train/logs
#   sbatch /public/home/putianshu/vis_mlp/train/sub_select_lowvis_gate_from_diagnostics.slurm
#
# Useful overrides:
#   sbatch --export=ALL,RUN_ID=exp_1778563813_pm10_more_temp_search_utc sub_select_lowvis_gate_from_diagnostics.slurm
#   sbatch --export=ALL,VAL_DIR=diagnostics/my_val_diag,TEST_DIR=diagnostics/my_test_diag sub_select_lowvis_gate_from_diagnostics.slurm
#   sbatch --export=ALL,DIAG_DIR=/public/home/putianshu/vis_mlp/diagnostics/exp_1778563813_pm10_more_temp_search_utc_test_lowvis_diag sub_select_lowvis_gate_from_diagnostics.slurm
#   sbatch --export=ALL,MAX_FPR=0.05,GOAL_MIST_RECALL=0.45 sub_select_lowvis_gate_from_diagnostics.slurm
#   sbatch --export=ALL,HARD_CONSTRAINTS=0 sub_select_lowvis_gate_from_diagnostics.slurm

set -euo pipefail

DCUS_PER_NODE=4
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Partition=${SLURM_JOB_PARTITION:-kshdexclu01} Nodes=${SLURM_NNODES:-1} CPUs=${CPUS_PER_TASK} DCUs/node=${DCUS_PER_NODE}"
if [ "${SLURM_NNODES:-1}" -ne 1 ]; then
    echo "Refusing to launch: this selector is intentionally single-node, but SLURM_NNODES=${SLURM_NNODES:-unset}."
    exit 1
fi

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"

# Use the same Python environment as the low-vis diagnostic/training launchers.
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

RUN_ID="${RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-/public/home/putianshu/vis_mlp/diagnostics}"
DIAG_DIR="${DIAG_DIR:-}"
VAL_DIR="${VAL_DIR:-${DIAGNOSTICS_ROOT}/${RUN_ID}_val_lowvis_diag}"
TEST_DIR="${TEST_DIR:-${DIAGNOSTICS_ROOT}/${RUN_ID}_test_lowvis_diag}"
OUT_DIR="${OUT_DIR:-${DIAGNOSTICS_ROOT}/${RUN_ID}_lowvis_gate_selection}"
DRY_RUN="${DRY_RUN:-0}"
HARD_CONSTRAINTS="${HARD_CONSTRAINTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GOAL_LOWVIS_RECALL="${GOAL_LOWVIS_RECALL:-0.75}"
GOAL_MIST_RECALL="${GOAL_MIST_RECALL:-0.40}"
GOAL_FOG_RECALL="${GOAL_FOG_RECALL:-0.50}"
GOAL_LOWVIS_CSI="${GOAL_LOWVIS_CSI:-0.22}"
GOAL_MIST_CSI="${GOAL_MIST_CSI:-0.10}"
GOAL_LOWVIS_PRECISION="${GOAL_LOWVIS_PRECISION:-0.20}"
MAX_FPR="${MAX_FPR:-0.06}"
MIN_LOWVIS_PRECISION="${MIN_LOWVIS_PRECISION:-0.12}"
MIN_MIST_PRECISION="${MIN_MIST_PRECISION:-0.08}"

W_LOWVIS_RECALL="${W_LOWVIS_RECALL:-0.35}"
W_MIST_RECALL="${W_MIST_RECALL:-0.25}"
W_FOG_RECALL="${W_FOG_RECALL:-0.20}"
W_LOWVIS_CSI="${W_LOWVIS_CSI:-0.10}"
W_MIST_CSI="${W_MIST_CSI:-0.05}"
W_PRECISION="${W_PRECISION:-0.05}"
W_FPR_PENALTY="${W_FPR_PENALTY:-0.35}"
W_PRECISION_PENALTY="${W_PRECISION_PENALTY:-0.20}"

mkdir -p /public/home/putianshu/vis_mlp/train/logs

echo "SCRIPT_DIR      : ${SCRIPT_DIR}"
echo "RUN_ID          : ${RUN_ID}"
echo "DIAGNOSTICS_ROOT: ${DIAGNOSTICS_ROOT}"
echo "DIAG_DIR        : ${DIAG_DIR:-<unused>}"
echo "VAL_DIR         : ${VAL_DIR}"
echo "TEST_DIR        : ${TEST_DIR}"
echo "OUT_DIR         : ${OUT_DIR}"
echo "DRY_RUN         : ${DRY_RUN}"
echo "HARD_CONSTRAINTS: ${HARD_CONSTRAINTS}"
echo "Goals           : lowvis_R=${GOAL_LOWVIS_RECALL}, mist_R=${GOAL_MIST_RECALL}, fog_R=${GOAL_FOG_RECALL}, max_FPR=${MAX_FPR}"
echo "Weights         : lowvis_R=${W_LOWVIS_RECALL}, mist_R=${W_MIST_RECALL}, fog_R=${W_FOG_RECALL}"

cd "${SCRIPT_DIR}" || exit 1

if [ -n "${DIAG_DIR}" ]; then
    if [ ! -f "${DIAG_DIR}/binary_gate_multiclass_sweep.csv" ]; then
        echo "Missing ${DIAG_DIR}/binary_gate_multiclass_sweep.csv"
        echo "DIAG_DIR mode is only for exploratory scoring of an existing diagnostic directory."
        exit 2
    fi
else
    if [ ! -f "${VAL_DIR}/binary_gate_multiclass_sweep.csv" ]; then
        echo "Missing ${VAL_DIR}/binary_gate_multiclass_sweep.csv"
        echo "Run validation diagnostics before gate selection:"
        echo "  sbatch --export=ALL,SPLIT=val,OUT_DIR=${VAL_DIR} sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm"
        exit 2
    fi
    if [ -n "${TEST_DIR}" ] && [ ! -f "${TEST_DIR}/binary_gate_multiclass_sweep.csv" ]; then
        echo "Missing ${TEST_DIR}/binary_gate_multiclass_sweep.csv"
        echo "Run test diagnostics before applying the validation-selected gate to test:"
        echo "  sbatch --export=ALL,SPLIT=test,OUT_DIR=${TEST_DIR} sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm"
        exit 2
    fi
fi

args=()
if [ -n "${DIAG_DIR}" ]; then
    args+=(--diag-dir "${DIAG_DIR}")
else
    args+=(--val-dir "${VAL_DIR}")
    if [ -n "${TEST_DIR}" ]; then
        args+=(--test-dir "${TEST_DIR}")
    fi
fi

args+=(--out-dir "${OUT_DIR}")
args+=(--experiment-id "${RUN_ID}")
args+=(--goal-lowvis-recall "${GOAL_LOWVIS_RECALL}")
args+=(--goal-mist-recall "${GOAL_MIST_RECALL}")
args+=(--goal-fog-recall "${GOAL_FOG_RECALL}")
args+=(--goal-lowvis-csi "${GOAL_LOWVIS_CSI}")
args+=(--goal-mist-csi "${GOAL_MIST_CSI}")
args+=(--goal-lowvis-precision "${GOAL_LOWVIS_PRECISION}")
args+=(--max-fpr "${MAX_FPR}")
args+=(--min-lowvis-precision "${MIN_LOWVIS_PRECISION}")
args+=(--min-mist-precision "${MIN_MIST_PRECISION}")
args+=(--w-lowvis-recall "${W_LOWVIS_RECALL}")
args+=(--w-mist-recall "${W_MIST_RECALL}")
args+=(--w-fog-recall "${W_FOG_RECALL}")
args+=(--w-lowvis-csi "${W_LOWVIS_CSI}")
args+=(--w-mist-csi "${W_MIST_CSI}")
args+=(--w-precision "${W_PRECISION}")
args+=(--w-fpr-penalty "${W_FPR_PENALTY}")
args+=(--w-precision-penalty "${W_PRECISION_PENALTY}")

if [[ "${HARD_CONSTRAINTS}" == "1" ]]; then
    args+=(--hard-constraints)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
fi

echo "Starting low-vis gate selection..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/select_lowvis_gate_from_diagnostics.py" "${args[@]}"
echo "Finished at $(date)"
