#!/bin/bash
#SBATCH -J lowvis_tree
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 120:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.err
#SBATCH --exclusive
#SBATCH --array=0-2

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_TREE_BASE:-/public/home/putianshu/vis_mlp}"
CACHE_DIR="${LOWVIS_TREE_CACHE_DIR:-${BASE}/tree_baseline_cache/s2_monthtail_w12_v1}"
RUN_ROOT="${LOWVIS_TREE_RUN_ROOT:-${BASE}/tree_baseline_runs}"
RUN_ID="${LOWVIS_TREE_RUN_ID:-exp_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_mainline_tree}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"

models=(rf xgboost lightgbm)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#models[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} is outside 0-2" >&2
    exit 2
fi
MODEL="${LOWVIS_TREE_MODEL:-${models[$task_id]}}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}/${MODEL}"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/activate_lowvis_data_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

if [[ ! -s "${CACHE_DIR}/preprocess_config.json" ]]; then
    echo "ERROR: tree cache is missing: ${CACHE_DIR}/preprocess_config.json" >&2
    echo "Submit sub_prepare_mainline_tree_features.slurm first." >&2
    exit 2
fi
for split in train val test; do
    for stem in X y_cls y_raw; do
        if [[ ! -s "${CACHE_DIR}/${stem}_${split}.npy" ]]; then
            echo "ERROR: incomplete tree cache: ${CACHE_DIR}/${stem}_${split}.npy" >&2
            exit 2
        fi
    done
done

case "${MODEL}" in
    rf)
        python -c "import numpy, sklearn; print('[tree-env] numpy=' + numpy.__version__ + ' sklearn=' + sklearn.__version__)"
        ;;
    xgboost)
        python -c "import numpy, xgboost; print('[tree-env] numpy=' + numpy.__version__ + ' xgboost=' + xgboost.__version__)"
        ;;
    lightgbm)
        python -c "import numpy, lightgbm; print('[tree-env] numpy=' + numpy.__version__ + ' lightgbm=' + lightgbm.__version__)"
        ;;
    *)
        echo "ERROR: unsupported LOWVIS_TREE_MODEL=${MODEL}" >&2
        exit 2
        ;;
esac

command=(
    python -u "${SCRIPT_DIR}/mainline_tree_baselines.py" train
    --model "${MODEL}"
    --cache-dir "${CACHE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --threads "${THREADS}"
    --seed "${LOWVIS_TREE_SEED:-20260729}"
    --class-weight-mode "${LOWVIS_TREE_CLASS_WEIGHT_MODE:-mainline}"
    --rf-trees "${LOWVIS_TREE_RF_TREES:-400}"
    --num-boost-round "${LOWVIS_TREE_NUM_BOOST_ROUND:-2500}"
    --early-stopping-rounds "${LOWVIS_TREE_EARLY_STOPPING_ROUNDS:-100}"
    --predict-chunk-rows "${LOWVIS_TREE_PREDICT_CHUNK_ROWS:-100000}"
)
if [[ "${LOWVIS_TREE_FORCE:-0}" == "1" ]]; then
    command+=(--force)
fi

echo "[tree-train] model=${MODEL}"
echo "[tree-train] cache=${CACHE_DIR}"
echo "[tree-train] output=${OUTPUT_DIR}"
"${command[@]}"
echo "[tree-train] finished=$(date --iso-8601=seconds)"
