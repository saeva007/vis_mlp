#!/bin/bash
#SBATCH -J spcv_prepare
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
FOLD_SEED="${FOLD_SEED:-20260815}"
BUFFER_KM="${BUFFER_KM:-50}"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-}"

export LOWVIS_DIFFUSION_USE_DCU=0
export LOWVIS_DIFFUSION_LOAD_HIP_RUNTIME=1
source "${REPO_ROOT}/activate_lowvis_diffusion_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${REPO_ROOT}/logs" "${RESULT_ROOT}/folds"
cd "${REPO_ROOT}"

echo "[preflight] repo=${REPO_ROOT}"
echo "[preflight] data=${DATA_DIR}"
echo "[preflight] output=${RESULT_ROOT}/folds"
for split in train val test; do
    for stem in X y meta; do
        suffix=npy
        if [ "${stem}" = meta ]; then suffix=csv; fi
        path="${DATA_DIR}/${stem}_${split}.${suffix}"
        test -s "${path}"
        echo "[preflight] ${path}=OK"
    done
done

python -u "${REPO_ROOT}/spatial_mapping_cv.py" prepare \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RESULT_ROOT}/folds" \
    --n-folds 5 \
    --seed "${FOLD_SEED}" \
    --buffer-km "${BUFFER_KM}" \
    --min-fold-stations 100

if [ -n "${IFS_PER_SAMPLE_CSV}" ]; then
    test -s "${IFS_PER_SAMPLE_CSV}"
    python -u "${REPO_ROOT}/spatial_mapping_cv.py" validate-ifs-baseline \
        --data-dir "${DATA_DIR}" \
        --ifs-csv "${IFS_PER_SAMPLE_CSV}" \
        --output-json "${RESULT_ROOT}/folds/ifs_baseline_compatibility.json"
    echo "[preflight] IFS baseline exact test-row alignment=OK"
fi

echo "[done] spatial folds prepared at $(date)"
