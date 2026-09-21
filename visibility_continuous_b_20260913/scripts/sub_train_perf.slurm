#!/bin/bash
#SBATCH -J vcb_perf_train
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH -t 240:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
SEED=${2:?seed required}
NPROC=${3:-1}
BATCH=${4:-4096}
WORKERS=${5:-16}
AMP=${6:-none}
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913
source "${ROOT}/scripts/runtime.sh"
export MIOPEN_USER_DB_PATH="/tmp/miopen_vcb_${ROUTE}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Shared mmap/OS cache; do not duplicate or preload the complete dataset.
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  "${ROOT}/scripts/viscast_continuous_b_perf.py" train --route "${ROUTE}" --seed "${SEED}" \
  --batch-size "${BATCH}" --num-workers "${WORKERS}" --amp "${AMP}"

# Sampling-only evaluation uses all four allocated DCUs by simple sample sharding.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/viscast_continuous_b_perf.py" eval --route "${ROUTE}" --seed "${SEED}" \
  --batch-size "${BATCH}" --num-workers "${WORKERS}" --amp "${AMP}"
