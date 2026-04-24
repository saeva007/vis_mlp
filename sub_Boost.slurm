#!/bin/bash
#SBATCH -J gbdt_vis_xgboost
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem 0
#SBATCH -t 24:00:00
#SBATCH -o logs/%j_gbdt.out
#SBATCH -e logs/%j_gbdt.err
#SBATCH --exclusive 

echo "Job started at: `date`"

module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch 

# 确保安装了 GBDT 库 (如果没有，请取消注释安装)
# pip install xgboost lightgbm catboost scikit-learn

# 设置线程数
export OMP_NUM_THREADS=32

# 运行 Python 脚本
# 可以通过 --model_type 切换架构: xgboost, lightgbm, catboost
# 可以通过 --window_size 切换窗口: 3, 6, 9
python Boost.py \
    --model_type xgboost \
    --window_size 12 \
    --num_threads 32 \
    --base_path "/public/home/putianshu/vis_mlp" \
    #--only_stage2 \
    --max_depth 15

echo "Job finished at: `date`"