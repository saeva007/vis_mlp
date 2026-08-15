#!/bin/bash
#SBATCH -J spcv_logistic
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --array=0-4%1
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.err
#SBATCH --exclusive

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
FOLD="${SLURM_ARRAY_TASK_ID}"
SEED="${MODEL_SEED:-20260815}"
OUTPUT_DIR="${RESULT_ROOT}/results/logistic/fold_${FOLD}"

module purge
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

test -s "${RESULT_ROOT}/folds/fold_manifest.json"
test -s "${RESULT_ROOT}/folds/fold_${FOLD}/s2_train_indices.npy"
test -s "${RESULT_ROOT}/folds/fold_${FOLD}/s2_val_indices.npy"
test -s "${RESULT_ROOT}/folds/fold_${FOLD}/s2_test_indices.npy"
echo "[preflight] logistic fold=${FOLD} indices=OK output=${OUTPUT_DIR}"

python -u "${REPO_ROOT}/spatial_mapping_cv.py" train-logistic \
    --data-dir "${DATA_DIR}" \
    --fold-dir "${RESULT_ROOT}/folds/fold_${FOLD}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --fog-ratio 0.18 \
    --mist-ratio 0.22 \
    --max-epochs 20 \
    --patience 4

echo "[done] logistic fold=${FOLD} at $(date)"
