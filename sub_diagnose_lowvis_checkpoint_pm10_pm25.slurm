#!/bin/bash
#SBATCH -J lowvis_diag_pm10pm25
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/log/lowvis_diag/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/log/lowvis_diag/%j_%x.err
#
# Post-hoc diagnostic only: no training, no checkpoint writes.
#
# Submit:
#   mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_diag
#   sbatch /public/home/putianshu/vis_mlp/train/sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm
#
# Useful overrides:
#   sbatch --export=ALL,SPLIT=val,SKIP_FINE_GRID=1 sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm
#   sbatch --export=ALL,LIMIT_SAMPLES=200000 sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm

set -euo pipefail

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Partition=${SLURM_JOB_PARTITION:-kshdexclu01} CPUs=${SLURM_CPUS_PER_TASK:-16}"

# Match the DCU PyTorch runtime used by the S2 training launcher.
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE_DIR="${BASE_DIR:-/public/home/putianshu/vis_mlp}"
RUN_ID="${RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
SPLIT="${SPLIT:-test}"
DATA_DIR="${DATA_DIR:-ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
CKPT_PATH="${CKPT_PATH:-checkpoints/${RUN_ID}_S2_PhaseB_best_score.pt}"
SCALER_PATH="${SCALER_PATH:-checkpoints/robust_scaler_${RUN_ID}_w12_dyn27_s2_48h_pm10.pkl}"
SEASON_TH_PATH="${SEASON_TH_PATH:-checkpoints/${RUN_ID}_season_thresholds.pt}"
OUT_DIR="${OUT_DIR:-diagnostics/${RUN_ID}_${SPLIT}_lowvis_diag}"
FOG_TH="${FOG_TH:-0.34}"
MIST_TH="${MIST_TH:-0.56}"
THRESHOLD_RULE="${THRESHOLD_RULE:-mutual}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
SKIP_FINE_GRID="${SKIP_FINE_GRID:-0}"
USE_CALIBRATION="${USE_CALIBRATION:-0}"

mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_diag
cd "${REPO_ROOT}"

args=(
  --base "${BASE_DIR}"
  --data-dir "${DATA_DIR}"
  --split "${SPLIT}"
  --ckpt-path "${CKPT_PATH}"
  --scaler-path "${SCALER_PATH}"
  --season-th-path "${SEASON_TH_PATH}"
  --out-dir "${OUT_DIR}"
  --fog-th "${FOG_TH}"
  --mist-th "${MIST_TH}"
  --threshold-rule "${THRESHOLD_RULE}"
  --batch-size "${BATCH_SIZE}"
  --limit-samples "${LIMIT_SAMPLES}"
  --device auto
)

if [[ "${SKIP_FINE_GRID}" == "1" ]]; then
  args+=(--skip-fine-grid)
fi
if [[ "${USE_CALIBRATION}" == "1" ]]; then
  args+=(--use-calibration)
fi

echo "Running low-visibility checkpoint diagnostics"
echo "RUN_ID=${RUN_ID} SPLIT=${SPLIT} DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"

python diagnose_lowvis_checkpoint_pm10_pm25.py "${args[@]}" \
  2>&1 | tee "/public/home/putianshu/vis_mlp/log/lowvis_diag/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"

echo "Finished at $(date)"
