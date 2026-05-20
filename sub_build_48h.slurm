#!/bin/bash
#SBATCH -J build_48h_pm10pm25
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/public/home/putianshu/vis_mlp/train}"
OUT_DIR="${OUT_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime}"

echo "48h PM10+PM2.5 data build started at: $(date)"
echo "Node      : $(hostname)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "OUT_DIR   : ${OUT_DIR}"
echo "SPLITS    : ${SPLITS_TO_WRITE:-test}"

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OUTPUT_DATASET_DIR="${OUT_DIR}"
export SPLITS_TO_WRITE="${SPLITS_TO_WRITE:-test}"

python "${SCRIPT_DIR}/PMST_s2_data_48h_pm10.py"

echo "48h PM10+PM2.5 data build finished at: $(date)"
