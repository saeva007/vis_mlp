#!/bin/bash
#SBATCH -J posthoc_lowvis_gate
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
# No-training post-hoc decision-rule selection.
# Reads diagnostic _rank_parts and evaluates:
#   - fine_low_mass gate: p_fog + p_mist >= threshold
#   - hybrid_or gate: binary low-vis head OR fine_low_mass gate
#
# Example:
#   cd /public/home/putianshu/vis_mlp/train
#   export BASE=/public/home/putianshu/vis_mlp
#   export REPO=/public/home/putianshu/vis_mlp/train
#   export RUN_ID=exp_1778563813_pm10_more_temp_search_utc
#   export DIAGNOSTICS_ROOT=${BASE}/diagnostics
#   sbatch --export=ALL,RUN_ID=${RUN_ID},DIAGNOSTICS_ROOT=${DIAGNOSTICS_ROOT},\
#   MAX_FPR=0.045,MIN_LOWVIS_PRECISION=0.12,MIN_MIST_PRECISION=0.06,\
#   OUT_DIR=${DIAGNOSTICS_ROOT}/${RUN_ID}_posthoc_lowvis_gate_selection \
#     ${REPO}/sub_select_posthoc_lowvis_gate_from_diagnostics.slurm

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
OUT_DIR="${OUT_DIR:-${DIAGNOSTICS_ROOT}/${RUN_ID}_posthoc_lowvis_gate_selection}"
DRY_RUN="${DRY_RUN:-0}"
HARD_CONSTRAINTS="${HARD_CONSTRAINTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BINARY_GRID="${BINARY_GRID:-0.01:0.99:0.01}"
FINE_GRID="${FINE_GRID:-0.01:0.99:0.01}"
TOP_K="${TOP_K:-25}"

GOAL_LOWVIS_RECALL="${GOAL_LOWVIS_RECALL:-0.75}"
GOAL_MIST_RECALL="${GOAL_MIST_RECALL:-0.40}"
GOAL_FOG_RECALL="${GOAL_FOG_RECALL:-0.50}"
GOAL_LOWVIS_CSI="${GOAL_LOWVIS_CSI:-0.22}"
GOAL_MIST_CSI="${GOAL_MIST_CSI:-0.10}"
GOAL_LOWVIS_PRECISION="${GOAL_LOWVIS_PRECISION:-0.20}"
MAX_FPR="${MAX_FPR:-0.045}"
MIN_LOWVIS_PRECISION="${MIN_LOWVIS_PRECISION:-0.12}"
MIN_MIST_PRECISION="${MIN_MIST_PRECISION:-0.06}"

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
echo "BINARY_GRID     : ${BINARY_GRID}"
echo "FINE_GRID       : ${FINE_GRID}"
echo "Goals           : lowvis_R=${GOAL_LOWVIS_RECALL}, mist_R=${GOAL_MIST_RECALL}, fog_R=${GOAL_FOG_RECALL}, max_FPR=${MAX_FPR}"

cd "${SCRIPT_DIR}" || exit 1

if [ -n "${DIAG_DIR}" ]; then
    if ! compgen -G "${DIAG_DIR}/_rank_parts/rank_*.npz" >/dev/null; then
        echo "Missing ${DIAG_DIR}/_rank_parts/rank_*.npz"
        echo "Run diagnostics with saved rank parts before post-hoc gate selection."
        exit 2
    fi
else
    if ! compgen -G "${VAL_DIR}/_rank_parts/rank_*.npz" >/dev/null; then
        echo "Missing ${VAL_DIR}/_rank_parts/rank_*.npz"
        echo "Run validation diagnostics before post-hoc gate selection."
        exit 2
    fi
    if [ -n "${TEST_DIR}" ] && ! compgen -G "${TEST_DIR}/_rank_parts/rank_*.npz" >/dev/null; then
        echo "Missing ${TEST_DIR}/_rank_parts/rank_*.npz"
        echo "Run test diagnostics before applying the validation-selected gate to test."
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
args+=(--binary-grid "${BINARY_GRID}")
args+=(--fine-grid "${FINE_GRID}")
args+=(--top-k "${TOP_K}")
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

echo "Starting post-hoc low-vis gate selection..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/select_posthoc_lowvis_gate_from_diagnostics.py" "${args[@]}"
echo "Finished at $(date)"
