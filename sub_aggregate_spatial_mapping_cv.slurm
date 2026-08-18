#!/bin/bash
#SBATCH -J spcv_aggregate
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
RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-}"
MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE:-argmax}"

case "${MAPPING_DECISION_RULE}" in
    argmax|val_search) ;;
    *) echo "[preflight] invalid MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}" >&2; exit 2 ;;
esac

export LOWVIS_DIFFUSION_USE_DCU=0
export LOWVIS_DIFFUSION_LOAD_HIP_RUNTIME=1
source "${REPO_ROOT}/activate_lowvis_diffusion_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${REPO_ROOT}/logs" "${RESULT_ROOT}/aggregate"
cd "${REPO_ROOT}"

command=(
    python -u "${REPO_ROOT}/spatial_mapping_cv.py" aggregate
    --folds-dir "${RESULT_ROOT}/folds"
    --results-dir "${RESULT_ROOT}/results"
    --output-dir "${RESULT_ROOT}/aggregate"
    --decision-rule "${MAPPING_DECISION_RULE}"
)

# Bash 4.2 on the cluster treats an empty-array expansion as an unbound
# variable under `set -u`. Keep the command array non-empty and append the
# optional IFS arguments only when the baseline file actually exists.
if [ -n "${IFS_PER_SAMPLE_CSV}" ] && [ -s "${IFS_PER_SAMPLE_CSV}" ]; then
    command+=(--ifs-csv "${IFS_PER_SAMPLE_CSV}")
    echo "[preflight] IFS per-sample baseline=${IFS_PER_SAMPLE_CSV}"
else
    echo "[preflight] IFS per-sample baseline unavailable; learned operators only"
fi

"${command[@]}"

echo "[done] aggregate=${RESULT_ROOT}/aggregate at $(date)"
