#!/bin/bash
#SBATCH -J vcb_train
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH -t 240:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?R2b, R3b, or R4b required}
SEED=${2:?seed required}
NPROC=${3:?selected DCU count required}
BATCH=${4:?selected per-DCU batch required}
WORKERS=${5:?selected num_workers required}
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913
source ${ROOT}/scripts/runtime.sh
SOURCE_DATA=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
LOCAL_X=/tmp/viscast_b_${ROUTE}_s${SEED}_${SLURM_JOB_ID}
test "${LOCAL_X}" = "/tmp/viscast_b_${ROUTE}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_X}"
trap 'test "${LOCAL_X}" = "/tmp/viscast_b_${ROUTE}_s${SEED}_${SLURM_JOB_ID}" && rm -rf -- "${LOCAL_X}"' EXIT
cp "${SOURCE_DATA}/X_train.npy" "${LOCAL_X}/X_train.npy"
cp "${SOURCE_DATA}/X_val.npy" "${LOCAL_X}/X_val.npy"
export VISCAST_X_DIR="${LOCAL_X}"
export RUN_NODE="$(hostname)"
python -m torch.distributed.run --standalone --nproc_per_node=${NPROC} \
  "${ROOT}/scripts/viscast_continuous_b.py" train --route "${ROUTE}" --seed "${SEED}" \
  --batch-size "${BATCH}" --num-workers "${WORKERS}"
cp "${SOURCE_DATA}/X_test.npy" "${LOCAL_X}/X_test.npy"
python -m torch.distributed.run --standalone --nproc_per_node=${NPROC} \
  "${ROOT}/scripts/viscast_continuous_b.py" eval --route "${ROUTE}" --seed "${SEED}" \
  --batch-size "${BATCH}" --num-workers "${WORKERS}"
