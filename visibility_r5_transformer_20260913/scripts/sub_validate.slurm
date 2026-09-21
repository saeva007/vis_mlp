#!/bin/bash
#SBATCH -J r5_fullval
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH -t 48:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
STAGE=${2:?stage required}
SEED=${3:?seed required}
CHECKPOINT=${4:?checkpoint required}
OUTPUT=${5:?output required}
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913
test -f "${CHECKPOINT%.pt}.ready"
source "${ROOT}/scripts/runtime.sh"
unset R5_X_DIR R5_QUICK_X
export MIOPEN_USER_DB_PATH="/tmp/miopen_r5_val_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/r5_pipeline.py" validate --route "${ROUTE}" --stage "${STAGE}" --split val \
  --checkpoint "${CHECKPOINT}" --draws 64 --reverse-steps 20 --output "${OUTPUT}"
