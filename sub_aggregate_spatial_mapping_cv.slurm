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

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
unset LD_PRELOAD
export LOWVIS_DIFFUSION_USE_DCU=0
source "${REPO_ROOT}/activate_lowvis_diffusion_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${REPO_ROOT}/logs" "${RESULT_ROOT}/aggregate"
cd "${REPO_ROOT}"

IFS_ARGS=()
if [ -n "${IFS_PER_SAMPLE_CSV}" ] && [ -s "${IFS_PER_SAMPLE_CSV}" ]; then
    IFS_ARGS=(--ifs-csv "${IFS_PER_SAMPLE_CSV}")
    echo "[preflight] IFS per-sample baseline=${IFS_PER_SAMPLE_CSV}"
else
    echo "[preflight] IFS per-sample baseline unavailable; learned operators only"
fi

python -u "${REPO_ROOT}/spatial_mapping_cv.py" aggregate \
    --folds-dir "${RESULT_ROOT}/folds" \
    --results-dir "${RESULT_ROOT}/results" \
    --output-dir "${RESULT_ROOT}/aggregate" \
    "${IFS_ARGS[@]}"

echo "[done] aggregate=${RESULT_ROOT}/aggregate at $(date)"
