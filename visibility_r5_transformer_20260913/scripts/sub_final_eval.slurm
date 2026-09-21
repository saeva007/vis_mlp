#!/bin/bash
#SBATCH -J r5_final_eval
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH -t 72:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
SEED=${2:?seed required}
BEST=${3:?checkpoint required}
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913
RUN=${ROOT}/runs/${ROUTE}/seed${SEED}/S2
source "${ROOT}/scripts/runtime.sh"
unset R5_X_DIR R5_QUICK_X
export MIOPEN_USER_DB_PATH="/tmp/miopen_r5_final_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/r5_pipeline.py" validate --route "${ROUTE}" --stage S2 --split val \
  --checkpoint "${BEST}" --draws 128 --output "${RUN}/validation/best_review_128.json"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/r5_pipeline.py" validate --route "${ROUTE}" --stage S2 --split test \
  --checkpoint "${BEST}" --draws 256 --output "${RUN}/test_metrics.json" --save-predictions
