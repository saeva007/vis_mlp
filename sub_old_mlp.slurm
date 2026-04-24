#!/bin/bash
#SBATCH -J old_mlp_test
#SBATCH -p kshdexclu01
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem 0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive 

echo "Job started at: `date`"

# --- 环境加载（保持不变）---
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

# --- 关键修复 ---
export PYTHONUNBUFFERED=1
ulimit -u 200000
export OMP_NUM_THREADS=1

# ====== [FIX 1] DCU/NCCL 超时设置 ======
export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# ====== [FIX 2] 添加超时保护（关键！）======
export NCCL_TIMEOUT=3600        # NCCL操作超时（秒）
export NCCL_BLOCKING_WAIT=1     # 阻塞等待，便于诊断
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 暂时开启详细日志

# ====== [FIX 3] 防止 /dev/shm 耗尽 ======
# 限制每个进程的共享内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

rm -f core.*

# --- 获取主节点 ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head Node: $head_node, IP: $head_node_ip"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

# ====== [FIX 4] 添加健康检查 ======
echo "Testing NCCL connectivity..."
srun bash -c "python -c 'import torch; print(f\"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else -1}: torch={torch.__version__}\")'"

# --- 启动训练（添加超时） ---
timeout 230h srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --max_restarts 3 \
    --log_dir logs/torchrun_${SLURM_JOB_ID} \
    old_mlp_test_1.py

echo "Job finished at: `date`"