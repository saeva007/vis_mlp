#!/bin/bash
#SBATCH -J r5flow_sanity
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/logs/sanity_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/logs/sanity_%j.err
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915
source "${ROOT}/scripts/runtime.sh"
python -m py_compile "${ROOT}/scripts/r5_model.py" "${ROOT}/scripts/r5_pipeline.py" "${ROOT}/scripts/prepare_contracts.py" "${ROOT}/scripts/r5_sanity.py"
python "${ROOT}/scripts/prepare_contracts.py"
python "${ROOT}/scripts/r5_pipeline.py" describe
TMP_ROOT=/tmp/r5_flow_sanity_${SLURM_JOB_ID}
mkdir -p "${TMP_ROOT}"
trap 'rm -rf -- "${TMP_ROOT}"' EXIT
python "${ROOT}/scripts/stage_quick_subset.py" --stage S1 --output "${TMP_ROOT}/X_quick_val.npy"
export R5_QUICK_X="${TMP_ROOT}/X_quick_val.npy"
export MIOPEN_USER_DB_PATH="/tmp/miopen_r5flow_sanity_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
HIP_VISIBLE_DEVICES=0 python "${ROOT}/scripts/r5_sanity.py"
