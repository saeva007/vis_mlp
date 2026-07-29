#!/bin/bash
#SBATCH -J lowvis_tree_prep
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_TREE_BASE:-/public/home/putianshu/vis_mlp}"
DATA_DIR="${LOWVIS_TREE_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
CACHE_DIR="${LOWVIS_TREE_CACHE_DIR:-${BASE}/tree_baseline_cache/s2_monthtail_w12_v1}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${SCRIPT_DIR}/logs" "${CACHE_DIR}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/activate_lowvis_data_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

required=()
for split in train val test; do
    required+=("${DATA_DIR}/X_${split}.npy" "${DATA_DIR}/y_${split}.npy")
done
missing=()
for path in "${required[@]}"; do
    [[ -s "${path}" ]] || missing+=("${path}")
done
if (( ${#missing[@]} > 0 )); then
    echo "ERROR: incomplete mainline dataset: ${DATA_DIR}" >&2
    printf '  missing: %s\n' "${missing[@]}" >&2
    exit 2
fi

command=(
    python -u "${SCRIPT_DIR}/mainline_tree_baselines.py" prepare
    --data-dir "${DATA_DIR}"
    --cache-dir "${CACHE_DIR}"
    --window-size "${LOWVIS_TREE_WINDOW_SIZE:-12}"
    --chunk-rows "${LOWVIS_TREE_PREP_CHUNK_ROWS:-32768}"
    --median-sample-rows "${LOWVIS_TREE_MEDIAN_SAMPLE_ROWS:-200000}"
    --row-limit "${LOWVIS_TREE_PREP_ROW_LIMIT:-0}"
    --seed "${LOWVIS_TREE_SEED:-20260729}"
)
if [[ "${LOWVIS_TREE_PREP_FORCE:-0}" == "1" ]]; then
    command+=(--force)
fi

echo "[tree-prep] data=${DATA_DIR}"
echo "[tree-prep] cache=${CACHE_DIR}"
echo "[tree-prep] row_limit=${LOWVIS_TREE_PREP_ROW_LIMIT:-0}"
"${command[@]}"
echo "[tree-prep] finished=$(date --iso-8601=seconds)"
