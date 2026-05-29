#!/bin/bash
#SBATCH -J s2_data_monthgroup
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "S2 month-group data build started at: $(date)"
echo "JobId=${SLURM_JOB_ID} Node=$(hostname)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/train
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

export S2_SPLIT_POLICY="${S2_SPLIT_POLICY:-month_group}"
export S2_SPLIT_NAME="${S2_SPLIT_NAME:-monthgroup_train13579_val11_test24681012}"
export S2_TRAIN_MONTHS="${S2_TRAIN_MONTHS:-1,3,5,7,9}"
export S2_VAL_MONTHS="${S2_VAL_MONTHS:-11}"
export S2_TEST_MONTHS="${S2_TEST_MONTHS:-2,4,6,8,10,12}"
export S2_DROP_CROSS_SPLIT_WINDOWS="${S2_DROP_CROSS_SPLIT_WINDOWS:-1}"
export S2_OUTPUT_DATASET_DIR="${S2_OUTPUT_DATASET_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthgroup_train13579_val11_test24681012}"

echo "S2_SPLIT_POLICY=${S2_SPLIT_POLICY}"
echo "S2_SPLIT_NAME=${S2_SPLIT_NAME}"
echo "S2_TRAIN_MONTHS=${S2_TRAIN_MONTHS}"
echo "S2_VAL_MONTHS=${S2_VAL_MONTHS}"
echo "S2_TEST_MONTHS=${S2_TEST_MONTHS}"
echo "S2_DROP_CROSS_SPLIT_WINDOWS=${S2_DROP_CROSS_SPLIT_WINDOWS}"
echo "S2_OUTPUT_DATASET_DIR=${S2_OUTPUT_DATASET_DIR}"

python s2_data_aerosol.py

echo "S2 month-group data build finished at: $(date)"
