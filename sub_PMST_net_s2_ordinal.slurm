#!/bin/bash
#SBATCH -J PMST_s2_ord
#SBATCH -p kshdexclu01
# 与当前 S2 主实验保持一致；若 RCCL 仍不稳，可先改为 4–8 节点做对照。
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
    echo "Refusing to launch on ${SLURM_NNODES} nodes: validated only up to ${MAX_STABLE_NNODES} nodes on the current RCCL/DCU cluster."
    exit 1
fi

SCRIPT_DIR="/public/home/putianshu/vis_mlp/train"
TRAIN_SCRIPT="${SCRIPT_DIR}/PMST_net_test_13_s2_pm10_ordinal.py"

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
# 2. RCCL / NCCL Communication
# ==========================================
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# ==========================================
# 3. Node Discovery and Port Setup
# ==========================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
    ip -4 -o addr show ib0 | awk '{print $4}' | cut -d/ -f1)

if [ -z "$MASTER_ADDR" ]; then
    echo "Warning: Could not find ib0 IP for $head_node, falling back to hostname"
    MASTER_ADDR=${head_node}
fi

BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
RDZV_PORT=${BASE_PORT}
DIST_STORE_PORT=$(expr ${BASE_PORT} + 1)

export MASTER_ADDR
export DIST_STORE_PORT

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}       (torchrun rendezvous)"
echo "DIST_STORE_PORT : ${DIST_STORE_PORT} (Python dist.TCPStore)"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * 4))"
echo "Train Script    : ${TRAIN_SCRIPT}"

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
echo "Starting ordinal S2 training on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."

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
             \"${TRAIN_SCRIPT}\"
     "

echo "Job finished at: $(date)"
