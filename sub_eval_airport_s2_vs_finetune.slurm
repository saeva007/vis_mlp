#!/bin/bash
#SBATCH -J eval_airport_s2_ft
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
AIRPORT_RNN_DATA_DIR="${AIRPORT_RNN_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_static_rnn_airport_metar_2025_12h}"
AIRPORT_EVAL_SPLIT="${AIRPORT_EVAL_SPLIT:-auto}"
OUT_DIR="${AIRPORT_EVAL_OUT_DIR:-${AIRPORT_RNN_DATA_DIR}/eval_s2_vs_finetune_${AIRPORT_EVAL_SPLIT}_${SLURM_JOB_ID}}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "AIRPORT_RNN_DATA_DIR=${AIRPORT_RNN_DATA_DIR}"
echo "AIRPORT_EVAL_SPLIT=${AIRPORT_EVAL_SPLIT}"
echo "AIRPORT25_MAIN_RUN_ID=${AIRPORT25_MAIN_RUN_ID:-}"
echo "AIRPORT25_FT_RUN_ID=${AIRPORT25_FT_RUN_ID:-}"
echo "OUT_DIR=${OUT_DIR}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd "${SCRIPT_DIR}"
mkdir -p logs "${OUT_DIR}"

args=(
  --data-dir "${AIRPORT_RNN_DATA_DIR}"
  --split "${AIRPORT_EVAL_SPLIT}"
  --out-dir "${OUT_DIR}"
  --batch-size "${AIRPORT_EVAL_BATCH_SIZE:-4096}"
  --num-workers "${AIRPORT_EVAL_NUM_WORKERS:-0}"
)

if [ -n "${AIRPORT25_MAIN_RUN_ID:-}" ]; then
  args+=(--s2-run-id "${AIRPORT25_MAIN_RUN_ID}")
fi
if [ -n "${AIRPORT25_FT_RUN_ID:-}" ]; then
  args+=(--finetune-run-id "${AIRPORT25_FT_RUN_ID}")
fi
if [ -n "${AIRPORT25_S2_CKPT:-}" ]; then
  args+=(--s2-ckpt "${AIRPORT25_S2_CKPT}")
fi
if [ -n "${AIRPORT25_S2_SCALER:-}" ]; then
  args+=(--s2-scaler "${AIRPORT25_S2_SCALER}")
fi
if [ -n "${AIRPORT25_FT_CKPT:-}" ]; then
  args+=(--finetune-ckpt "${AIRPORT25_FT_CKPT}")
fi
if [ -n "${AIRPORT25_FT_SCALER:-}" ]; then
  args+=(--finetune-scaler "${AIRPORT25_FT_SCALER}")
fi

python evaluate_airport_s2_vs_finetune.py "${args[@]}"

echo "Finished at $(date)"
