#!/bin/bash
#SBATCH -J PMST_net_5
#SBATCH -p kshdexclu01
#SBATCH -N 20
#SBATCH --ntasks-per-node=1   # Launch 1 manager task per node (torchrun handles the 4 GPUs)
#SBATCH --gres=dcu:4
#SBATCH --mem=0               # Use all node memory
#SBATCH --exclusive           # Exclusive access
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "Job started at: `date`"

# ==========================================
# 1. Environment Setup
# ==========================================
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1

# Load DCU/DTK environment
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh

# User specific paths (Keep these if they are required for your specific libs)
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

# Activate Conda
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch

# ==========================================
# 2. Performance & Cache Optimization
# ==========================================
export PYTHONUNBUFFERED=1

# [CPU Threading] 
# CRITICAL: Set to 1 because your DataLoader uses separate workers. 
# Prevents CPU contention between PyTorch OMP threads and DataLoaders.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# [MIOpen Cache - Local Performance]
# Use /tmp on each node to avoid shared filesystem locking/slowness
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# FIX: Use srun to create directory on ALL nodes, not just the head node
srun --ntasks-per-node=1 mkdir -p ${MIOPEN_USER_DB_PATH}

# ==========================================
# 3. DCU/NCCL Stability Settings
# ==========================================
# [Communication Backend]
export NCCL_P2P_DISABLE=1 
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0,vnet0

# [Debugging & Timeout]
# OPTIMIZATION: Changed INFO to WARN. INFO on 20 nodes produces too much log spam.
# Use INFO only if the job hangs or crashes.
export NCCL_DEBUG=VERSION
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_COMM_ID_TIMEOUT=1800

# Remove core dumps if they exist
rm -f core.*

# ==========================================
# 4. Distributed Setup (Rendezvous)
# ==========================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node: $head_node, IP: $head_node_ip"

# Generate a unique port based on Job ID to avoid collisions
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

# ==========================================
# 5. Execution
# ==========================================
# srun launches 'torchrun' once on every node.
# torchrun then spawns 4 processes (nproc_per_node) on that node.

srun -u --ntasks-per-node=1 torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    main_5.py 

echo "Job finished at: `date`"

# ==========================================
# 6. Cleanup
# ==========================================
# Clean up the temporary MIOpen cache on all nodes
srun -u --ntasks-per-node=1 rm -rf ${MIOPEN_USER_DB_PATH}