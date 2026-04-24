#!/bin/bash
#SBATCH -J PMST_net_test_s2_pm10
#SBATCH -p kshdexclu01
# 与 sub_s1.slurm 对齐为 5 节点：在 9 节点 (36 rank) 下，本脚本与 s1 相同的首次 GPU all_reduce(safe_barrier)
# 曾在作业 111693678 中于 init 成功后挂死（日志无 “[Main] Initial barrier passed”）。
# 需更大规模时：先在此节点数下跑通，再逐步加节点并观察首 collective。
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive

echo "Job started at: $(date)"

MAX_STABLE_NNODES=10
if [ "${SLURM_NNODES}" -gt "${MAX_STABLE_NNODES}" ]; then
    echo "Refusing to launch on ${SLURM_NNODES} nodes: this job is only validated up to ${MAX_STABLE_NNODES} nodes on the current RCCL/DCU cluster, and larger runs can hang on the first collective."
    exit 1
fi

# 训练脚本目录（srun 在各节点上需显式 cd，不能依赖 sbatch 提交时的 cwd）
SCRIPT_DIR="/public/home/putianshu/vis_mlp/train"

# ==========================================
# 1. Environment
# ==========================================
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

# ==========================================
# 2. RCCL / NCCL Communication (AMD DCU cluster)
# ==========================================
# Disable SHM to avoid RCCL mutex crash; disable IB for socket fallback
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
# Primary: ib0 (IPoIB). Fallback: eth0 if cluster uses Ethernet for TCP
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
# 与 sub_s1.slurm 一致：显式 Ring/Simple 在部分 DCU/RCCL 组合上更易在首 collective 挂死，交由 RCCL 自选算法
# Default WARN: NCCL_DEBUG=INFO multiplies stderr volume (36 ranks × 9 nodes) and can add
# many minutes of wall time at DDP/RCCL bootstrap — looks like a hang but is log I/O + connect.
export NCCL_DEBUG=WARN
# For one-off hangs, temporarily: export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# ==========================================
# 3. Node Discovery and IP Resolution
# ==========================================
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

# ==========================================
# PORT ASSIGNMENT — THE ROOT CAUSE FIX
# ==========================================
#
# PROBLEM: torchrun --rdzv_backend=static creates a TCPStore server on the
# --rdzv_endpoint port BEFORE your Python script starts. When your Python
# init_distributed() then tries to create another TCPStore server on the
# identical MASTER_PORT (which torchrun also exports into each worker's
# environment as MASTER_PORT), you get "Address already in use". Rank 0
# dies, all other ranks see "Connection reset by peer", and the job fails
# immediately without reaching the training loop.
#
# FIX: Two separate ports.
#   RDZV_PORT      — used exclusively by torchrun's --rdzv_endpoint.
#                    torchrun binds its rendezvous TCPStore here.
#   DIST_STORE_PORT — used exclusively by the Python dist.TCPStore inside
#                    init_distributed(). torchrun does NOT override custom
#                    env vars, so DIST_STORE_PORT passes through to every
#                    worker process untouched.
#
# Using the last 4 digits of SLURM_JOBID ensures jobs on the same cluster
# do not collide with each other. The +1 guarantees the two ports are
# always distinct even if the base port wraps around.
# ==========================================
BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
RDZV_PORT=${BASE_PORT}
DIST_STORE_PORT=$(expr ${BASE_PORT} + 1)

export MASTER_ADDR
export DIST_STORE_PORT   # Read by Python init_distributed(); NOT overridden by torchrun

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}       (torchrun rendezvous)"
echo "DIST_STORE_PORT : ${DIST_STORE_PORT} (Python dist.TCPStore)"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * 4))"

# ==========================================
# 4. Clean /dev/shm on all nodes
# ==========================================
echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*X_train* /dev/shm/*y_train* 2>/dev/null; \
              echo \"\$(hostname) SHM Cleaned\""

# ==========================================
# 5. Launch Training
# ==========================================
echo "Starting Training on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."

srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     --kill-on-bad-exit=1 \
     bash -c "
         cd \"${SCRIPT_DIR}\" || exit 1
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\${MIOPEN_USER_DB_PATH}
         mkdir -p \${MIOPEN_USER_DB_PATH}

         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=4 \
             --node_rank=\${SLURM_NODEID} \
             --rdzv_backend=static \
             --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --max_restarts=0 \
             \"${SCRIPT_DIR}/PMST_net_test_11_s2_pm10.py\"
     "
echo "Job finished at: $(date)"