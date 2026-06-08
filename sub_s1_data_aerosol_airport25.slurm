#!/bin/bash
#SBATCH -J s1_airport25_data
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 120:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
S1_OUTPUT_DATASET_DIR="${S1_OUTPUT_DATASET_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_airport25}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "S1_OUTPUT_DATASET_DIR=${S1_OUTPUT_DATASET_DIR}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd "${SCRIPT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

export LOWVIS_INCLUDE_PM=0
export S1_OUTPUT_DATASET_DIR

python s1_data_aerosol.py

echo "Finished at $(date)"
