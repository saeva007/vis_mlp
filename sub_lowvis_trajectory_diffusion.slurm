#!/bin/bash
#SBATCH -J lowvis_traj_diff
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 168:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail
SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_TRAJ_BASE:-/public/home/putianshu/vis_mlp}"
DATA_DIR="${LOWVIS_TRAJ_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_trajectory_0_48h_pm10_pm25_v1}"
MODEL_TYPE="${LOWVIS_TRAJ_MODEL_TYPE:-diffusion}"
RUN_ID="${LOWVIS_TRAJ_RUN_ID:-exp_${SLURM_JOB_ID}_lowvis_trajectory_${MODEL_TYPE}}"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
export LOWVIS_DIFFUSION_USE_DCU=1
export LOWVIS_DIFFUSION_TORCH_ENV="${LOWVIS_DIFFUSION_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
source "${SCRIPT_DIR}/activate_lowvis_diffusion_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

torchrun --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/train_lowvis_trajectory_diffusion.py" \
    --model-type "${MODEL_TYPE}" \
    --data-dir "${DATA_DIR}" \
    --run-id "${RUN_ID}" \
    --checkpoint-dir "${LOWVIS_TRAJ_CHECKPOINT_DIR:-${BASE}/checkpoints}" \
    --batch-size "${LOWVIS_TRAJ_BATCH_SIZE:-128}" \
    --max-steps "${LOWVIS_TRAJ_MAX_STEPS:-100000}" \
    --val-interval "${LOWVIS_TRAJ_VAL_INTERVAL:-2000}" \
    ${LOWVIS_TRAJ_TRAIN_EXTRA_ARGS:-}

