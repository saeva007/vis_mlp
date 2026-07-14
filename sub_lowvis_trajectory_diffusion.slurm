#!/bin/bash
#SBATCH -J lowvis_traj_diff_5n
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_TRAJ_BASE:-/public/home/putianshu/vis_mlp}"
export SCRIPT_DIR BASE
export LOWVIS_TRAJ_DATA_DIR="${LOWVIS_TRAJ_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_trajectory_0_48h_pm10_pm25_v1}"
export LOWVIS_TRAJ_MODEL_TYPE="${LOWVIS_TRAJ_MODEL_TYPE:-diffusion}"
export LOWVIS_TRAJ_RUN_ID="${LOWVIS_TRAJ_RUN_ID:-exp_${SLURM_JOB_ID}_lowvis_trajectory_${LOWVIS_TRAJ_MODEL_TYPE}_5n}"
export LOWVIS_TRAJ_CHECKPOINT_DIR="${LOWVIS_TRAJ_CHECKPOINT_DIR:-${BASE}/checkpoints}"
export LOWVIS_TRAJ_DCU_PER_NODE="${LOWVIS_TRAJ_DCU_PER_NODE:-4}"

# Five nodes × four ranks.  Per-rank batch 32 keeps the global batch at 640,
# close to the original 512 instead of accidentally increasing it to 2560.
export LOWVIS_TRAJ_BATCH_SIZE="${LOWVIS_TRAJ_BATCH_SIZE:-32}"
export LOWVIS_TRAJ_NUM_WORKERS="${LOWVIS_TRAJ_NUM_WORKERS:-2}"
export LOWVIS_TRAJ_MAX_STEPS="${LOWVIS_TRAJ_MAX_STEPS:-80000}"
export LOWVIS_TRAJ_VAL_INTERVAL="${LOWVIS_TRAJ_VAL_INTERVAL:-1000}"
export LOWVIS_TRAJ_D_MODEL="${LOWVIS_TRAJ_D_MODEL:-192}"
export LOWVIS_TRAJ_NHEAD="${LOWVIS_TRAJ_NHEAD:-8}"
export LOWVIS_TRAJ_CONDITION_LAYERS="${LOWVIS_TRAJ_CONDITION_LAYERS:-5}"
export LOWVIS_TRAJ_DECODER_LAYERS="${LOWVIS_TRAJ_DECODER_LAYERS:-6}"
export LOWVIS_TRAJ_DROPOUT="${LOWVIS_TRAJ_DROPOUT:-0.10}"
export LOWVIS_TRAJ_LEARNING_RATE="${LOWVIS_TRAJ_LEARNING_RATE:-2e-4}"
export LOWVIS_TRAJ_WARMUP_STEPS="${LOWVIS_TRAJ_WARMUP_STEPS:-2000}"
export LOWVIS_TRAJ_EMA_DECAY="${LOWVIS_TRAJ_EMA_DECAY:-0.9999}"
export LOWVIS_TRAJ_MIN_SNR_GAMMA="${LOWVIS_TRAJ_MIN_SNR_GAMMA:-5.0}"
export LOWVIS_TRAJ_VAL_MONITOR_SIZE="${LOWVIS_TRAJ_VAL_MONITOR_SIZE:-768}"
export LOWVIS_TRAJ_VAL_MEMBERS="${LOWVIS_TRAJ_VAL_MEMBERS:-20}"
export LOWVIS_TRAJ_VAL_DDIM_STEPS="${LOWVIS_TRAJ_VAL_DDIM_STEPS:-25}"
export LOWVIS_TRAJ_PATIENCE="${LOWVIS_TRAJ_PATIENCE:-15}"
export LOWVIS_TRAJ_LOWVIS_SELECTION_WEIGHT="${LOWVIS_TRAJ_LOWVIS_SELECTION_WEIGHT:-0.60}"
export LOWVIS_TRAJ_TORCH_ENV="${LOWVIS_TRAJ_TORCH_ENV:-${LOWVIS_DIFFUSION_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}}"

mkdir -p "${SCRIPT_DIR}/logs" "${LOWVIS_TRAJ_CHECKPOINT_DIR}"

# Fail once on the batch host before spawning 20 workers.  A missing config is
# the normal signal that the CPU trajectory-builder job has not finished.
required_dataset_files=("${LOWVIS_TRAJ_DATA_DIR}/dataset_build_config.json")
for split in train val test; do
    for stem in dynamic static_cont veg_id time_features visibility target_mask; do
        required_dataset_files+=("${LOWVIS_TRAJ_DATA_DIR}/${stem}_${split}.npy")
    done
    required_dataset_files+=("${LOWVIS_TRAJ_DATA_DIR}/meta_${split}.csv")
done
missing_dataset_files=()
missing_dataset_count=0
for path in "${required_dataset_files[@]}"; do
    if [[ ! -s "${path}" ]]; then
        missing_dataset_files+=("${path}")
        missing_dataset_count=$((missing_dataset_count + 1))
    fi
done
if (( missing_dataset_count > 0 )); then
    echo "ERROR: trajectory dataset is missing or incomplete: ${LOWVIS_TRAJ_DATA_DIR}" >&2
    printf '  missing: %s\n' "${missing_dataset_files[@]}" >&2
    echo "Build it first with:" >&2
    echo "  sbatch --export=ALL,LOWVIS_TRAJ_DATA_DIR=${LOWVIS_TRAJ_DATA_DIR} ${SCRIPT_DIR}/sub_build_lowvis_trajectory_dataset.slurm" >&2
    exit 2
fi

# Conservative RCCL settings already used by the stable S2 five-node jobs.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export LOWVIS_TRAJ_DIST_TIMEOUT=3600

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
head_node="${nodes[0]}"
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${head_node}" bash -c '
  for c in /sbin/ip /usr/sbin/ip; do
    if [ -x "$c" ]; then
      v=$("$c" -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
      if [ -n "$v" ]; then echo "$v"; exit 0; fi
    fi
  done
  h=$(hostname -I 2>/dev/null | awk "{print \$1}")
  [ -n "$h" ] && echo "$h" && exit 0
  exit 1
' 2>/dev/null | head -1 | tr -d "\r") || true
if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR="${head_node}"
fi

BASE_PORT=$(expr 10000 + $(echo -n "${SLURM_JOB_ID}" | tail -c 4))
export MASTER_ADDR
export RDZV_PORT="${LOWVIS_TRAJ_RDZV_PORT:-${BASE_PORT}}"

echo "[launch] nodes=${SLURM_NNODES} ranks=$((SLURM_NNODES * LOWVIS_TRAJ_DCU_PER_NODE))"
echo "[launch] master=${MASTER_ADDR}:${RDZV_PORT}"
echo "[launch] model=${LOWVIS_TRAJ_MODEL_TYPE} run_id=${LOWVIS_TRAJ_RUN_ID}"
echo "[launch] per_rank_batch=${LOWVIS_TRAJ_BATCH_SIZE} global_batch=$((SLURM_NNODES * LOWVIS_TRAJ_DCU_PER_NODE * LOWVIS_TRAJ_BATCH_SIZE))"

srun --nodes="${SLURM_NNODES}" \
     --ntasks="${SLURM_NNODES}" \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     --kill-on-bad-exit=1 \
     bash "${SCRIPT_DIR}/launch_lowvis_trajectory_distributed.sh"
