#!/bin/bash
#SBATCH -J mapping_cv_plot
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT:?set SPATIAL_RESULT_ROOT}"
TEMPORAL_RESULT_ROOT="${TEMPORAL_RESULT_ROOT:?set TEMPORAL_RESULT_ROOT}"
FIGURE_OUTPUT_DIR="${FIGURE_OUTPUT_DIR:-${TEMPORAL_RESULT_ROOT}/figures}"
PLOT_ENV="${PLOT_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
PLOT_PYTHON="${PLOT_PYTHON:-${PLOT_ENV}/bin/python}"
FIGURE_STEM="${FIGURE_STEM:-mapping_operator_spatiotemporal_cv}"
MAPPING_GRU_LABEL="${MAPPING_GRU_LABEL:-Static-MLP + GRU (12 h)}"

# This script imports NumPy/pandas/matplotlib only.  Do not initialize the
# HIP-linked Torch runtime: that emits irrelevant HyHAL warnings on CPU nodes.
if [ ! -x "${PLOT_PYTHON}" ]; then
    echo "ERROR: plotting Python is missing or not executable: ${PLOT_PYTHON}" >&2
    exit 2
fi
unset LD_PRELOAD
export PATH="${PLOT_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PLOT_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${FIGURE_OUTPUT_DIR}/.matplotlib"

mkdir -p "${REPO_ROOT}/logs" "${FIGURE_OUTPUT_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[preflight] python=${PLOT_PYTHON}"
"${PLOT_PYTHON}" -c \
    "import matplotlib, numpy, pandas; print('[preflight] plotting imports=OK')"
echo "[preflight] spatial_result_root=${SPATIAL_RESULT_ROOT}"
echo "[preflight] temporal_result_root=${TEMPORAL_RESULT_ROOT}"

missing_input=0
for path in \
    "${SPATIAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
    "${SPATIAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
    "${SPATIAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
    "${SPATIAL_RESULT_ROOT}/folds/fold_manifest.json" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
    "${TEMPORAL_RESULT_ROOT}/folds/fold_manifest.json"; do
    if [ -s "${path}" ]; then
        echo "[preflight] ${path}=OK"
    else
        echo "ERROR: required plotting input is missing or empty: ${path}" >&2
        missing_input=1
    fi
done
if [ "${missing_input}" -ne 0 ]; then
    exit 3
fi

"${PLOT_PYTHON}" -u "${REPO_ROOT}/plot_mapping_cv_folds.py" \
    --spatial-result-root "${SPATIAL_RESULT_ROOT}" \
    --temporal-result-root "${TEMPORAL_RESULT_ROOT}" \
    --output-dir "${FIGURE_OUTPUT_DIR}" \
    --stem "${FIGURE_STEM}" \
    --gru-label "${MAPPING_GRU_LABEL}" \
    --formats svg,pdf,png,tiff \
    --dpi 600

echo "[done] mapping CV figure=${FIGURE_OUTPUT_DIR} at $(date)"
