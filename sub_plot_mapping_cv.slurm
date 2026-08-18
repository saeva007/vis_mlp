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

export LOWVIS_DIFFUSION_USE_DCU=0
export LOWVIS_DIFFUSION_LOAD_HIP_RUNTIME=1
source "${REPO_ROOT}/activate_lowvis_diffusion_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${FIGURE_OUTPUT_DIR}/.matplotlib"

mkdir -p "${REPO_ROOT}/logs" "${FIGURE_OUTPUT_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

for path in \
    "${SPATIAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
    "${SPATIAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
    "${SPATIAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
    "${SPATIAL_RESULT_ROOT}/folds/fold_manifest.json" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
    "${TEMPORAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
    "${TEMPORAL_RESULT_ROOT}/folds/fold_manifest.json"; do
    test -s "${path}"
    echo "[preflight] ${path}=OK"
done

python -u "${REPO_ROOT}/plot_mapping_cv_folds.py" \
    --spatial-result-root "${SPATIAL_RESULT_ROOT}" \
    --temporal-result-root "${TEMPORAL_RESULT_ROOT}" \
    --output-dir "${FIGURE_OUTPUT_DIR}" \
    --formats svg,pdf,png,tiff \
    --dpi 600

echo "[done] mapping CV figure=${FIGURE_OUTPUT_DIR} at $(date)"
