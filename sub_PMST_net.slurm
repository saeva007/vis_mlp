#!/bin/bash
#SBATCH -J PMST_net_test_2
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

# Added --exclusive: 确保独占节点，防止其他用户的任务干扰内存或CPU

echo "Job started at: `date`"

# --- 1. 环境加载 ---
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1

source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
echo "MIOPEN_FIND_MODE: ${MIOPEN_FIND_MODE}"

export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch

export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

# --- 2. 关键环境变量设置 ---

# [重要] 允许 Python 实时输出日志，否则 OOM 时看不到最后一行报错
export PYTHONUNBUFFERED=1

# 进程限制与线程设置
ulimit -u 200000
export OMP_NUM_THREADS=1

# [DCU/NCCL 特定设置]
# 必须禁用 P2P，否则在部分 DCU 集群上必卡死 (保持原样)
export NCCL_P2P_DISABLE=1 
export HSA_FORCE_FINE_GRAIN_PCIE=1

# 网络设置 (如果确定是 ib0 则保持，否则可用 ^lo,docker0 排查)
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# [日志级别优化] 
# 改为 WARN，防止 80 个 GPU 的 INFO 日志刷屏掩盖报错
export NCCL_DEBUG=WARN
# 只有在调试具体死锁问题时才开启 INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO 
# export NCCL_ASYNC_ERROR_HANDLING=1

# 清理残留文件
rm -f core.*

# --- 3. 获取主节点 IP ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head Node: $head_node, IP: $head_node_ip"

# 动态端口
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

# --- 4. 启动命令 ---
# 注意：PyTorch DDP 推荐使用 torchrun
# srun 负责在每个节点启动一个 torchrun 代理
# torchrun 负责在每个节点内启动 4 个 Python 进程 (对应 nproc_per_node)

srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    PMST_net_test_2.py