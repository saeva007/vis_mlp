#!/bin/bash
#SBATCH -J s2_data_vera
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

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/train

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "Running Stage 2 VERA data build..."
python s2_data_aerosol_vera.py

echo "Finished at $(date)"
