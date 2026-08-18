#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
BUNDLE_ID="${BUNDLE_ID:-temporal_mapping_cv_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/public/home/putianshu/vis_mlp/temporal_mapping_cv}"
RESULT_ROOT="${RESULT_ROOT:-${RESULT_BASE}/${BUNDLE_ID}}"
SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT:-}"
SPATIAL_AGGREGATE_JOB_ID="${SPATIAL_AGGREGATE_JOB_ID:-}"
TEMPORAL_EMBARGO_HOURS="${TEMPORAL_EMBARGO_HOURS:-24}"
INPUT_WINDOW_HOURS="${INPUT_WINDOW_HOURS:-12}"
MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE:-argmax}"
MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY:-5}"
MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY:-4}"
MAPPING_MODELS="${MAPPING_MODELS:-logistic,mlp,gru}"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-/public/home/putianshu/vis_mlp/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal/per_sample_eval.csv}"
STATE_FILE="${STATE_FILE:-${RESULT_ROOT}/submission_state.sh}"
SUBMIT_LOG="${SUBMIT_LOG:-${RESULT_ROOT}/submission.log}"

write_state() {
    local temporary="${STATE_FILE}.tmp"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'RESULT_ROOT=%q\n' "${RESULT_ROOT}"
        printf 'DATA_DIR=%q\n' "${DATA_DIR}"
        printf 'SPATIAL_RESULT_ROOT=%q\n' "${SPATIAL_RESULT_ROOT}"
        printf 'SPATIAL_AGGREGATE_JOB_ID=%q\n' "${SPATIAL_AGGREGATE_JOB_ID}"
        printf 'TEMPORAL_EMBARGO_HOURS=%q\n' "${TEMPORAL_EMBARGO_HOURS}"
        printf 'INPUT_WINDOW_HOURS=%q\n' "${INPUT_WINDOW_HOURS}"
        printf 'MAPPING_DECISION_RULE=%q\n' "${MAPPING_DECISION_RULE}"
        printf 'MAPPING_LOGISTIC_CONCURRENCY=%q\n' "${MAPPING_LOGISTIC_CONCURRENCY}"
        printf 'MAPPING_NEURAL_CONCURRENCY=%q\n' "${MAPPING_NEURAL_CONCURRENCY}"
        printf 'MAPPING_MODELS=%q\n' "${MAPPING_MODELS}"
        printf 'IFS_PER_SAMPLE_CSV=%q\n' "${IFS_PER_SAMPLE_CSV}"
        printf 'PREP_JOB_ID=%q\n' "${PREP_JOB_ID:-}"
        printf 'LOGISTIC_JOB_ID=%q\n' "${LOGISTIC_JOB_ID:-}"
        printf 'NEURAL_JOB_ID=%q\n' "${NEURAL_JOB_ID:-}"
        printf 'AGGREGATE_JOB_ID=%q\n' "${AGGREGATE_JOB_ID:-}"
        printf 'PLOT_JOB_ID=%q\n' "${PLOT_JOB_ID:-}"
        printf 'SUBMISSION_STATUS=%q\n' "${SUBMISSION_STATUS:-initializing}"
    } > "${temporary}"
    mv "${temporary}" "${STATE_FILE}"
}

if [ "${1:-}" != "--worker" ]; then
    mkdir -p "${RESULT_ROOT}"
    echo "[launcher] bundle=${BUNDLE_ID}"
    echo "[launcher] result_root=${RESULT_ROOT}"
    echo "[launcher] spatial_result_root=${SPATIAL_RESULT_ROOT:-not_set}"
    echo "[launcher] state_file=${STATE_FILE}"
    echo "[launcher] submit_log=${SUBMIT_LOG}"
    REPO_ROOT="${REPO_ROOT}" BUNDLE_ID="${BUNDLE_ID}" RESULT_ROOT="${RESULT_ROOT}" \
        DATA_DIR="${DATA_DIR}" SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT}" \
        SPATIAL_AGGREGATE_JOB_ID="${SPATIAL_AGGREGATE_JOB_ID}" \
        TEMPORAL_EMBARGO_HOURS="${TEMPORAL_EMBARGO_HOURS}" \
        INPUT_WINDOW_HOURS="${INPUT_WINDOW_HOURS}" \
        MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE}" \
        MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY}" \
        MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY}" \
        MAPPING_MODELS="${MAPPING_MODELS}" \
        IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV}" \
        STATE_FILE="${STATE_FILE}" \
        SUBMIT_LOG="${SUBMIT_LOG}" \
        nohup bash "${REPO_ROOT}/submit_temporal_mapping_cv_chain.sh" --worker \
        </dev/null >"${SUBMIT_LOG}" 2>&1 &
    echo "[launcher] detached_pid=$!"
    exit 0
fi

mkdir -p "${REPO_ROOT}/logs" "${RESULT_ROOT}"
cd "${REPO_ROOT}"

echo "[worker] started=$(date --iso-8601=seconds)"
echo "[worker] repo=${REPO_ROOT}"
echo "[worker] data=${DATA_DIR}"
echo "[worker] result_root=${RESULT_ROOT}"
echo "[worker] spatial_result_root=${SPATIAL_RESULT_ROOT:-not_set}"
echo "[worker] embargo=${TEMPORAL_EMBARGO_HOURS}h window=${INPUT_WINDOW_HOURS}h"
echo "[worker] decision_rule=${MAPPING_DECISION_RULE} models=${MAPPING_MODELS} logistic_concurrency=${MAPPING_LOGISTIC_CONCURRENCY} neural_concurrency=${MAPPING_NEURAL_CONCURRENCY}"
echo "[worker] ifs_per_sample=${IFS_PER_SAMPLE_CSV}"

case "${MAPPING_DECISION_RULE}" in
    argmax|val_search) ;;
    *) echo "[preflight] invalid MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}" >&2; exit 2 ;;
esac
case "${MAPPING_LOGISTIC_CONCURRENCY}:${MAPPING_NEURAL_CONCURRENCY}" in
    *[!0-9:]*|0:*|*:0) echo "[preflight] concurrency values must be positive integers" >&2; exit 2 ;;
esac

IFS=',' read -r -a REQUESTED_MODELS <<< "${MAPPING_MODELS}"
declare -A MODEL_SEEN=()
for model in "${REQUESTED_MODELS[@]}"; do
    case "${model}" in
        logistic|mlp|gru) ;;
        *) echo "[preflight] invalid model in MAPPING_MODELS=${MAPPING_MODELS}: ${model}" >&2; exit 2 ;;
    esac
    if [ -n "${MODEL_SEEN[${model}]:-}" ]; then
        echo "[preflight] duplicate model in MAPPING_MODELS=${MAPPING_MODELS}: ${model}" >&2
        exit 2
    fi
    MODEL_SEEN[${model}]=1
done
if [ "${#REQUESTED_MODELS[@]}" -eq 0 ]; then
    echo "[preflight] MAPPING_MODELS must not be empty" >&2
    exit 2
fi
export MAPPING_MODELS

for required in \
    temporal_mapping_cv.py \
    spatial_mapping_cv.py \
    train_static_rnn_lowvis.py \
    prepare_static_rnn_local_cache.sh \
    plot_mapping_cv_folds.py \
    sub_prepare_temporal_mapping_cv.slurm \
    sub_spatial_mapping_logistic_cv.slurm \
    sub_spatial_mapping_neural_cv.slurm \
    sub_aggregate_spatial_mapping_cv.slurm \
    sub_plot_mapping_cv.slurm; do
    test -s "${REPO_ROOT}/${required}"
    echo "[preflight] ${required}=OK"
done
for split in train val test; do
    for stem in X y meta; do
        suffix=npy
        if [ "${stem}" = meta ]; then suffix=csv; fi
        test -s "${DATA_DIR}/${stem}_${split}.${suffix}"
        echo "[preflight] ${stem}_${split}.${suffix}=OK"
    done
done
test -s "${IFS_PER_SAMPLE_CSV}"
echo "[preflight] IFS per-sample baseline=OK"

PREP_JOB_ID=$(sbatch --parsable \
    --job-name=tpcv_prepare \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATA_DIR=${DATA_DIR},RESULT_ROOT=${RESULT_ROOT},TEMPORAL_EMBARGO_HOURS=${TEMPORAL_EMBARGO_HOURS},INPUT_WINDOW_HOURS=${INPUT_WINDOW_HOURS},IFS_PER_SAMPLE_CSV=${IFS_PER_SAMPLE_CSV}" \
    "${REPO_ROOT}/sub_prepare_temporal_mapping_cv.slurm")
PREP_JOB_ID="${PREP_JOB_ID%%;*}"
SUBMISSION_STATUS=prepare_submitted
write_state
echo "[submit] prepare_job=${PREP_JOB_ID}"

LOGISTIC_JOB_ID=""
if [ -n "${MODEL_SEEN[logistic]:-}" ]; then
    LOGISTIC_JOB_ID=$(sbatch --parsable \
        --job-name=tpcv_logistic \
        --array="0-4%${MAPPING_LOGISTIC_CONCURRENCY}" \
        --dependency="afterok:${PREP_JOB_ID}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},DATA_DIR=${DATA_DIR},RESULT_ROOT=${RESULT_ROOT},MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}" \
        "${REPO_ROOT}/sub_spatial_mapping_logistic_cv.slurm")
    LOGISTIC_JOB_ID="${LOGISTIC_JOB_ID%%;*}"
    echo "[submit] logistic_array_job=${LOGISTIC_JOB_ID}"
else
    echo "[submit] logistic_array_job=SKIPPED"
fi
SUBMISSION_STATUS=logistic_stage_complete
write_state

NEURAL_JOB_ID=""
NEURAL_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},DATA_DIR=${DATA_DIR},RESULT_ROOT=${RESULT_ROOT},BUNDLE_ID=${BUNDLE_ID},CV_PREFIX=tpcv,MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}"
if [ -n "${MODEL_SEEN[mlp]:-}" ] && [ -n "${MODEL_SEEN[gru]:-}" ]; then
    NEURAL_ARRAY="0-9%${MAPPING_NEURAL_CONCURRENCY}"
elif [ -n "${MODEL_SEEN[mlp]:-}" ]; then
    NEURAL_ARRAY="0-4%${MAPPING_NEURAL_CONCURRENCY}"
    NEURAL_EXPORT="${NEURAL_EXPORT},MAPPING_NEURAL_MODEL=mlp"
elif [ -n "${MODEL_SEEN[gru]:-}" ]; then
    NEURAL_ARRAY="0-4%${MAPPING_NEURAL_CONCURRENCY}"
    NEURAL_EXPORT="${NEURAL_EXPORT},MAPPING_NEURAL_MODEL=gru"
else
    NEURAL_ARRAY=""
fi
if [ -n "${NEURAL_ARRAY}" ]; then
    NEURAL_JOB_ID=$(sbatch --parsable \
        --job-name=tpcv_neural \
        --array="${NEURAL_ARRAY}" \
        --dependency="afterok:${PREP_JOB_ID}" \
        --export="${NEURAL_EXPORT}" \
        "${REPO_ROOT}/sub_spatial_mapping_neural_cv.slurm")
    NEURAL_JOB_ID="${NEURAL_JOB_ID%%;*}"
    echo "[submit] neural_array_job=${NEURAL_JOB_ID}"
else
    echo "[submit] neural_array_job=SKIPPED"
fi
SUBMISSION_STATUS=training_submitted
write_state

AGGREGATE_DEPENDENCY="afterok"
for job_id in "${LOGISTIC_JOB_ID}" "${NEURAL_JOB_ID}"; do
    if [ -n "${job_id}" ]; then AGGREGATE_DEPENDENCY="${AGGREGATE_DEPENDENCY}:${job_id}"; fi
done
if [ "${AGGREGATE_DEPENDENCY}" = "afterok" ]; then
    echo "[preflight] no model jobs were submitted" >&2
    exit 2
fi
AGGREGATE_JOB_ID=$(sbatch --parsable \
    --job-name=tpcv_aggregate \
    --dependency="${AGGREGATE_DEPENDENCY}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},RESULT_ROOT=${RESULT_ROOT},IFS_PER_SAMPLE_CSV=${IFS_PER_SAMPLE_CSV},REQUIRE_IFS_BASELINE=1,MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}" \
    "${REPO_ROOT}/sub_aggregate_spatial_mapping_cv.slurm")
AGGREGATE_JOB_ID="${AGGREGATE_JOB_ID%%;*}"
SUBMISSION_STATUS=aggregate_submitted
write_state
echo "[submit] aggregate_job=${AGGREGATE_JOB_ID}"

PLOT_JOB_ID=""
if [ -n "${SPATIAL_RESULT_ROOT}" ]; then
    PLOT_DEPENDENCY="afterok:${AGGREGATE_JOB_ID}"
    if [ -n "${SPATIAL_AGGREGATE_JOB_ID}" ]; then
        PLOT_DEPENDENCY="${PLOT_DEPENDENCY}:${SPATIAL_AGGREGATE_JOB_ID}"
    fi
    PLOT_JOB_ID=$(sbatch --parsable \
        --job-name=mapping_cv_plot \
        --dependency="${PLOT_DEPENDENCY}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},SPATIAL_RESULT_ROOT=${SPATIAL_RESULT_ROOT},TEMPORAL_RESULT_ROOT=${RESULT_ROOT}" \
        "${REPO_ROOT}/sub_plot_mapping_cv.slurm")
    PLOT_JOB_ID="${PLOT_JOB_ID%%;*}"
    echo "[submit] plot_job=${PLOT_JOB_ID}"
    SUBMISSION_STATUS=complete_with_plot
else
    echo "[submit] plot_job=SKIPPED (set SPATIAL_RESULT_ROOT to enable combined figure)"
    SUBMISSION_STATUS=complete_without_plot
fi
write_state
echo "[worker] state_file=${STATE_FILE}"
echo "[worker] completed=$(date --iso-8601=seconds)"
