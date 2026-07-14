#!/bin/bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}}"
export LOWVIS_DIFFUSION_USE_DCU=1
export LOWVIS_DIFFUSION_TORCH_ENV="${LOWVIS_TRAJ_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
source "${SCRIPT_DIR}/activate_lowvis_diffusion_runtime.sh"
cd "${SCRIPT_DIR}"

export MIOPEN_USER_DB_PATH="/tmp/miopen_lowvis_diffusion_${SLURM_JOB_ID}_${SLURM_NODEID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

extra_args=()
if [[ -n "${LOWVIS_TRAJ_TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${LOWVIS_TRAJ_TRAIN_EXTRA_ARGS}"
fi

torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${LOWVIS_TRAJ_DCU_PER_NODE}" \
    --node_rank="${SLURM_NODEID}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --max_restarts=0 \
    "${SCRIPT_DIR}/train_lowvis_trajectory_diffusion.py" \
    --model-type "${LOWVIS_TRAJ_MODEL_TYPE}" \
    --data-dir "${LOWVIS_TRAJ_DATA_DIR}" \
    --run-id "${LOWVIS_TRAJ_RUN_ID}" \
    --checkpoint-dir "${LOWVIS_TRAJ_CHECKPOINT_DIR}" \
    --batch-size "${LOWVIS_TRAJ_BATCH_SIZE}" \
    --num-workers "${LOWVIS_TRAJ_NUM_WORKERS}" \
    --max-steps "${LOWVIS_TRAJ_MAX_STEPS}" \
    --val-interval "${LOWVIS_TRAJ_VAL_INTERVAL}" \
    --d-model "${LOWVIS_TRAJ_D_MODEL}" \
    --nhead "${LOWVIS_TRAJ_NHEAD}" \
    --condition-layers "${LOWVIS_TRAJ_CONDITION_LAYERS}" \
    --decoder-layers "${LOWVIS_TRAJ_DECODER_LAYERS}" \
    --dropout "${LOWVIS_TRAJ_DROPOUT}" \
    --condition-token-version 2 \
    --learning-rate "${LOWVIS_TRAJ_LEARNING_RATE}" \
    --warmup-steps "${LOWVIS_TRAJ_WARMUP_STEPS}" \
    --ema-decay "${LOWVIS_TRAJ_EMA_DECAY}" \
    --min-snr-gamma "${LOWVIS_TRAJ_MIN_SNR_GAMMA}" \
    --val-monitor-size "${LOWVIS_TRAJ_VAL_MONITOR_SIZE}" \
    --val-members "${LOWVIS_TRAJ_VAL_MEMBERS}" \
    --val-ddim-steps "${LOWVIS_TRAJ_VAL_DDIM_STEPS}" \
    --patience "${LOWVIS_TRAJ_PATIENCE}" \
    --lowvis-selection-weight "${LOWVIS_TRAJ_LOWVIS_SELECTION_WEIGHT}" \
    "${extra_args[@]}"
