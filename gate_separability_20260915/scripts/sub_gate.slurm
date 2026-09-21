#!/bin/bash
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=110G
#SBATCH -t 24:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/gate_separability_20260915/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/gate_separability_20260915/logs/%x_%j.err

set -euo pipefail
EXPERIMENT="$1"
ROOT=/public/home/putianshu/vis_mlp/gate_separability_20260915
DATA=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
LOCAL_DATA=/tmp/gate_separability_${EXPERIMENT}_${SLURM_JOB_ID}

source "${ROOT}/scripts/runtime.sh"
mkdir -p "${LOCAL_DATA}"
cp "${DATA}/X_train.npy" "${LOCAL_DATA}/X_train.npy"
cp "${DATA}/X_val.npy" "${LOCAL_DATA}/X_val.npy"
cp "${DATA}/X_test.npy" "${LOCAL_DATA}/X_test.npy"
export GATE_X_DIR="${LOCAL_DATA}"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  "${ROOT}/scripts/gate_study.py" --experiment "${EXPERIMENT}"
