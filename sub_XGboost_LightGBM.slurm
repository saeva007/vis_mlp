#!/bin/bash
#SBATCH -J ML_Baseline
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem 0
#SBATCH -t 48:00:00
#SBATCH -o logs/ml_%j.out
#SBATCH -e logs/ml_%j.err
#SBATCH --exclusive 

# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
MODEL_TYPE=${1:-xgboost}    # xgboost 或 lightgbm
WINDOW_SIZE=${2:-12}        # 窗口大小 (默认 12)

echo "Job started at: `date`"
echo "Configuration: Model=${MODEL_TYPE}, Window=${WINDOW_SIZE}h"

# 创建目录
mkdir -p logs


# ------------------------------------------------------------------
# 环境加载
# ------------------------------------------------------------------
module purge
module load compiler/devtoolset/7.3.1

# 加载 DCU 环境 (为了 XGBoost GPU 支持)
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
shift $# 

export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH

# 激活 Conda
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

# ------------------------------------------------------------------
# 运行配置
# ------------------------------------------------------------------
# 优化 CPU 线程设置
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

echo "Running on node: $(hostname)"
echo "Local storage check (/dev/shm):"
df -h /dev/shm

# ------------------------------------------------------------------
# 启动训练
# ------------------------------------------------------------------
python -u XGboost_Ablation.py \
    --model ${MODEL_TYPE} \
    --window_size ${WINDOW_SIZE} \
    --save_dir "./checkpoints"

echo "Job finished at: `date`"