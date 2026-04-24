#!/bin/bash
#SBATCH -J Ablation
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem 0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive 

# ------------------------------------------------------------------
# 0. 参数解析 (放在最前面，清晰明了)
# ------------------------------------------------------------------
# 默认值设置：如果命令行没传参数，则使用默认值
MODEL_TYPE=${1:-ours}       # 第1个参数: 模型 (默认 ours)
ABLATION_TYPE=${2:-none}    # 第2个参数: 消融类型 (默认 none)
WINDOW_SIZE=${3:-9}         # 第3个参数: 窗口大小 (默认 9)

echo "Job started at: `date`"
echo "Configuration: Model=${MODEL_TYPE}, Ablation=${ABLATION_TYPE}, Window=${WINDOW_SIZE}h"

# 检查日志目录是否存在，不存在则创建
mkdir -p logs

# ------------------------------------------------------------------
# 1. 环境加载
# ------------------------------------------------------------------
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1

source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
# 显式重置参数，防止 source activate 读取到脚本的 $1 $2 (解决 ArgumentError 的潜在根源)
shift $# 

export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH

# 激活 Conda
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

# ------------------------------------------------------------------
# 2. 环境变量设置 (PyTorch DDP & DCU)
# ------------------------------------------------------------------
export PYTHONUNBUFFERED=1
ulimit -u 200000
export OMP_NUM_THREADS=1

# DCU 必须设置
export NCCL_P2P_DISABLE=1 
export HSA_FORCE_FINE_GRAIN_PCIE=1

# 根据集群实际情况，如果不确定 ib0 可先注释掉，让 nccl 自动探测
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

export NCCL_DEBUG=WARN
rm -f core.*

# ------------------------------------------------------------------
# 3. 节点通信配置
# ------------------------------------------------------------------
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head Node: $head_node, IP: $head_node_ip"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

# ------------------------------------------------------------------
# 4. 启动命令
# ------------------------------------------------------------------
# 你的 Python 脚本名是 Ablation.py 还是 train.py? 请根据实际情况修改文件名
# 这里假设是 Ablation.py
# 注意：传入之前解析好的变量

srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    Ablation.py \
    --model ${MODEL_TYPE} \
    --ablation ${ABLATION_TYPE} \
    --window_size ${WINDOW_SIZE}