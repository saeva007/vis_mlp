#!/bin/bash
#SBATCH -J gbm_vis      # 作业名
#SBATCH -p kshdexclu01   # 队列
#SBATCH -N 5             # 节点数
#SBATCH --ntasks-per-node=1  # 每节点任务数
#SBATCH --cpus-per-task=32   # 每任务CPU数
#SBATCH --gres=dcu:4         # DCU资源（虽然树模型主要用CPU）
#SBATCH --mem 0              # 内存
#SBATCH -t 120:00:00         # 时间限制（树模型训练快，2小时足够）
#SBATCH -o logs/%j_gbm.out   # 输出日志
#SBATCH -e logs/%j_gbm.err   # 错误日志
#SBATCH --exclusive          # 独占节点

echo "=========================================="
echo "  Gradient Boosting - Distributed Training"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Started at: $(date)"
echo "=========================================="

# ==========================================
# 1. 环境设置（与你的配置一致）
# ==========================================

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh

export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH

# 进程限制
ulimit -u 200000
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# DCU设置（虽然树模型主要用CPU，保留兼容性）
export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# ==========================================
# 2. Dask分布式设置
# ==========================================

# 获取主节点信息
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"

# 动态端口
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
DASK_SCHEDULER_PORT=$(expr 8786 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master Port: $MASTER_PORT"
echo "Dask Scheduler Port: $DASK_SCHEDULER_PORT"

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$MASTER_PORT

# ==========================================
# 3. 启动Dask Scheduler（在主节点）
# ==========================================

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Starting Dask Scheduler on $head_node_ip:$DASK_SCHEDULER_PORT..."
    
    # 后台启动scheduler
    dask-scheduler \
        --host $head_node_ip \
        --port $DASK_SCHEDULER_PORT \
        --dashboard-address :8787 &
    
    SCHEDULER_PID=$!
    echo "Dask Scheduler PID: $SCHEDULER_PID"
    
    # 等待scheduler启动
    sleep 10
fi

# 同步所有节点
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "echo 'Node $(hostname) ready'"

# ==========================================
# 4. 启动Dask Workers（所有节点）
# ==========================================

echo "Starting Dask Workers on all nodes..."

# 每个节点启动一个worker（可根据需要调整nthreads和nprocs）
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
    dask-worker $head_node_ip:$DASK_SCHEDULER_PORT \
        --nthreads 16 \
        --nprocs 2 \
        --memory-limit 100GB \
        --death-timeout 120 &
    
    echo 'Dask Worker started on' \$(hostname)
" &

# 等待workers启动
sleep 15

# ==========================================
# 5. 训练配置选择
# ==========================================

# 默认配置（可通过环境变量覆盖）
MODEL_TYPE=${MODEL_TYPE:-xgboost}
WINDOW_SIZE=${WINDOW_SIZE:-12}
IMBALANCE_METHOD=${IMBALANCE_METHOD:-focal}
OPTIMIZE_METRIC=${OPTIMIZE_METRIC:-f2}

echo "=========================================="
echo "  Training Configuration"
echo "=========================================="
echo "  Model Type:        $MODEL_TYPE"
echo "  Window Size:       ${WINDOW_SIZE}h"
echo "  Imbalance Method:  $IMBALANCE_METHOD"
echo "  Optimize Metric:   $OPTIMIZE_METRIC"
echo "  Total Nodes:       $SLURM_NNODES"
echo "  Total Tasks:       $SLURM_NTASKS"
echo "=========================================="

# ==========================================
# 6. 启动训练（仅在主节点运行Python脚本）
# ==========================================

# 注意：树模型的分布式训练由Dask管理，不需要像PyTorch那样每个节点运行
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Starting training on master node..."
    
    python visibility_forecast_distributed.py \
        --model $MODEL_TYPE \
        --strategy mixed \
        --window $WINDOW_SIZE \
        --imbalance $IMBALANCE_METHOD \
        --optimize $OPTIMIZE_METRIC \
        --use-dask \
        --min-precision 0.20
    
    TRAIN_EXIT_CODE=$?
    
    echo "Training exit code: $TRAIN_EXIT_CODE"
    
    # 清理Dask集群
    echo "Cleaning up Dask cluster..."
    pkill -f dask-scheduler
    pkill -f dask-worker
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed successfully!"
    else
        echo "✗ Training failed with code $TRAIN_EXIT_CODE"
    fi
fi

echo "=========================================="
echo "  Job finished at: $(date)"
echo "=========================================="