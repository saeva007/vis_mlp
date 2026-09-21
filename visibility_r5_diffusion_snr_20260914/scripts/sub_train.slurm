#!/bin/bash
#SBATCH -J r5snr_train
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=110G
#SBATCH -t 240:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
STAGE=${2:?stage required}
SEED=${3:-1}
INIT=${4:-}
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914
source "${ROOT}/scripts/runtime.sh"
unset R5_X_DIR R5_QUICK_X
if [ "${STAGE}" = "S1" ]; then
  SOURCE=/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25
else
  SOURCE=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
fi
LOCAL_X=/tmp/r5_${STAGE}_${SEED}_${SLURM_JOB_ID}
test "${LOCAL_X}" = "/tmp/r5_${STAGE}_${SEED}_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_X}"
trap 'test "${LOCAL_X}" = "/tmp/r5_${STAGE}_${SEED}_${SLURM_JOB_ID}" && rm -rf -- "${LOCAL_X}"' EXIT
if [ "${STAGE}" = "S1" ]; then
  python "${ROOT}/scripts/stage_quick_subset.py" --stage S1 --output "${LOCAL_X}/X_quick_val.npy"
  export R5_QUICK_X="${LOCAL_X}/X_quick_val.npy"
else
  cp "${SOURCE}/X_train.npy" "${LOCAL_X}/X_train.npy"
  cp "${SOURCE}/X_val.npy" "${LOCAL_X}/X_val.npy"
  export R5_X_DIR="${LOCAL_X}"
fi
export MIOPEN_USER_DB_PATH="/tmp/miopen_r5_${STAGE}_${SEED}_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
ARGS=(train --route "${ROUTE}" --stage "${STAGE}" --seed "${SEED}")
if [ -n "${INIT}" ]; then ARGS+=(--init-checkpoint "${INIT}"); fi
HIP_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
  "${ROOT}/scripts/r5_pipeline.py" "${ARGS[@]}"
