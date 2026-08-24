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
DEFAULT_IFS_PER_SAMPLE_CSV="/public/home/putianshu/vis_mlp/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal/per_sample_eval.csv"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-${DEFAULT_IFS_PER_SAMPLE_CSV}}"
REQUIRE_IFS_BASELINE="${REQUIRE_IFS_BASELINE:-1}"
MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE:-argmax}"
MAPPING_MODELS="${MAPPING_MODELS:-logistic,mlp,gru}"
MAPPING_CV_SEEDS="${MAPPING_CV_SEEDS:-}"

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
    --models "${MAPPING_MODELS}"
    --decision-rule "${MAPPING_DECISION_RULE}"
)

if [ -n "${MAPPING_CV_SEEDS}" ]; then
    if [ "${MAPPING_DECISION_RULE}" != "argmax" ]; then
        echo "ERROR: P13 seed aggregation requires MAPPING_DECISION_RULE=argmax" >&2
        exit 2
    fi
    command+=(--seeds "${MAPPING_CV_SEEDS}")
    echo "[preflight] P13 seed ensemble=${MAPPING_CV_SEEDS}"
fi

# The formal mapping figure requires the native IFS diagnostic on the same
# frozen-test rows.  Fail closed rather than silently producing learned-only
# tables that the plotting step cannot use.
if [ -s "${IFS_PER_SAMPLE_CSV}" ]; then
    command+=(--ifs-csv "${IFS_PER_SAMPLE_CSV}")
    echo "[preflight] IFS per-sample baseline=${IFS_PER_SAMPLE_CSV}"
elif [ "${REQUIRE_IFS_BASELINE}" = "1" ]; then
    echo "ERROR: required IFS per-sample baseline is missing or empty: ${IFS_PER_SAMPLE_CSV}" >&2
    exit 3
else
    echo "[preflight] IFS baseline disabled explicitly; learned operators only"
fi

echo "[preflight] learned_models=${MAPPING_MODELS}"

"${command[@]}"

echo "[done] aggregate=${RESULT_ROOT}/aggregate at $(date)"
