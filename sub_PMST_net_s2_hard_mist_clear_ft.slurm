#!/bin/bash
#SBATCH -J PMST_s2_hard_mist_clear_ft
#SBATCH -p kshdexclu01
# Full-node DCU training job: every allocated node uses all 4 DCUs.
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

DCUS_PER_NODE=4
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Partition=${SLURM_JOB_PARTITION:-kshdexclu01} Nodes=${SLURM_NNODES} CPUs=${CPUS_PER_TASK} DCUs/node=${DCUS_PER_NODE}"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# Hard Mist/Clear fine-tune defaults. Override with:
#   sbatch --export=ALL,HARD_FT_STEPS=6000,HARD_FT_HARD_CLEAR_RATIO=0.28 ...
export INIT_RUN_ID="${INIT_RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
export HARD_FT_RUN_ID="${HARD_FT_RUN_ID:-${INIT_RUN_ID}_hard_mist_clear_ft}"
export INIT_CKPT_PATH="${INIT_CKPT_PATH:-/public/home/putianshu/vis_mlp/checkpoints/${INIT_RUN_ID}_S2_PhaseB_best_score.pt}"
export HARD_POOL_DIR="${HARD_POOL_DIR:-/public/home/putianshu/vis_mlp/hard_samples/${INIT_RUN_ID}_train_hard_samples}"
export LOWVIS_GATE_TH="${LOWVIS_GATE_TH:-0.81}"
export TARGET_MAX_FPR="${TARGET_MAX_FPR:-0.045}"

export HARD_FT_STEPS="${HARD_FT_STEPS:-12000}"
export HARD_FT_VAL_INTERVAL="${HARD_FT_VAL_INTERVAL:-500}"
export HARD_FT_PATIENCE="${HARD_FT_PATIENCE:-8}"
export HARD_FT_BATCH_SIZE="${HARD_FT_BATCH_SIZE:-512}"
export HARD_FT_GRAD_ACCUM="${HARD_FT_GRAD_ACCUM:-2}"
export HARD_FT_NUM_WORKERS="${HARD_FT_NUM_WORKERS:-0}"

export HARD_FT_LR_BACKBONE="${HARD_FT_LR_BACKBONE:-8.0e-7}"
export HARD_FT_LR_FUSION="${HARD_FT_LR_FUSION:-6.0e-6}"
export HARD_FT_LR_HEAD="${HARD_FT_LR_HEAD:-2.5e-5}"
export HARD_FT_L2SP="${HARD_FT_L2SP:-2.0e-5}"

export HARD_FT_FOG_RATIO="${HARD_FT_FOG_RATIO:-0.16}"
export HARD_FT_MIST_BG_RATIO="${HARD_FT_MIST_BG_RATIO:-0.18}"
export HARD_FT_HARD_MIST_RATIO="${HARD_FT_HARD_MIST_RATIO:-0.20}"
export HARD_FT_HARD_CLEAR_RATIO="${HARD_FT_HARD_CLEAR_RATIO:-0.22}"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c '
  for c in /sbin/ip /usr/sbin/ip; do
    if [ -x "$c" ]; then
      v=$("$c" -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
      if [ -n "$v" ]; then echo "$v"; exit 0; fi
    fi
  done
  if command -v ip >/dev/null 2>&1; then
    v=$(ip -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
    if [ -n "$v" ]; then echo "$v"; exit 0; fi
  fi
  h=$(hostname -I 2>/dev/null | awk "{print \$1}")
  [ -n "$h" ] && echo "$h" && exit 0
  exit 1
' 2>/dev/null | head -1 | tr -d "\r") || true

if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=${head_node}
fi

BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
RDZV_PORT=${BASE_PORT}
DIST_STORE_PORT=$(expr ${BASE_PORT} + 1)

export MASTER_ADDR
export DIST_STORE_PORT

mkdir -p /public/home/putianshu/vis_mlp/train/logs

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}       (torchrun rendezvous)"
echo "DIST_STORE_PORT : ${DIST_STORE_PORT} (Python dist.TCPStore)"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * DCUS_PER_NODE))"
echo "HARD_FT_RUN_ID  : ${HARD_FT_RUN_ID}"
echo "INIT_CKPT_PATH  : ${INIT_CKPT_PATH}"
echo "HARD_POOL_DIR   : ${HARD_POOL_DIR}"
echo "LOWVIS_GATE_TH  : ${LOWVIS_GATE_TH}"
echo "TARGET_MAX_FPR  : ${TARGET_MAX_FPR}"
echo "HARD_FT_STEPS   : ${HARD_FT_STEPS}"
echo "Ratios          : fog=${HARD_FT_FOG_RATIO}, mist_bg=${HARD_FT_MIST_BG_RATIO}, hard_mist=${HARD_FT_HARD_MIST_RATIO}, hard_clear=${HARD_FT_HARD_CLEAR_RATIO}"
echo "LRs             : backbone=${HARD_FT_LR_BACKBONE}, fusion=${HARD_FT_LR_FUSION}, head=${HARD_FT_LR_HEAD}"

echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*hard_mist_clear_ft* 2>/dev/null; \
              echo \"\$(hostname) SHM Cleaned\""

echo "Starting hard Mist/Clear fine-tuning on ${SLURM_NNODES} nodes (${SLURM_NNODES}x${DCUS_PER_NODE} DCUs)..."

srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=${CPUS_PER_TASK} \
     --gres=dcu:${DCUS_PER_NODE} \
     --kill-on-bad-exit=1 \
     bash -c "
         cd \"${SCRIPT_DIR}\" || exit 1
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\${MIOPEN_USER_DB_PATH}
         mkdir -p \${MIOPEN_USER_DB_PATH}

         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=${DCUS_PER_NODE} \
             --node_rank=\${SLURM_NODEID} \
             --rdzv_backend=static \
             --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --max_restarts=0 \
             \"${SCRIPT_DIR}/PMST_net_test_11_s2_pm10_hard_mist_clear_ft.py\"
     "

echo "Job finished at: $(date)"
