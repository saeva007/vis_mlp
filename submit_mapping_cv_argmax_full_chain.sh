#!/bin/bash
set -euo pipefail

# One-shot formal rerun in which Logistic, MLP, and GRU all use argmax for
# validation checkpoint selection and frozen-test classification. Spatial and
# temporal training are submitted together; the combined figure waits for both
# aggregate jobs.

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
RUN_ID="${RUN_ID:-mapping_cv_argmax_$(date +%Y%m%d_%H%M%S)}"
SPATIAL_BUNDLE_ID="${SPATIAL_BUNDLE_ID:-${RUN_ID}_spatial}"
TEMPORAL_BUNDLE_ID="${TEMPORAL_BUNDLE_ID:-${RUN_ID}_temporal}"
SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT:-/public/home/putianshu/vis_mlp/spatial_mapping_cv/${SPATIAL_BUNDLE_ID}}"
TEMPORAL_RESULT_ROOT="${TEMPORAL_RESULT_ROOT:-/public/home/putianshu/vis_mlp/temporal_mapping_cv/${TEMPORAL_BUNDLE_ID}}"
MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY:-5}"
MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY:-4}"
MASTER_STATE_FILE="${MASTER_STATE_FILE:-${TEMPORAL_RESULT_ROOT}/argmax_full_chain_state.sh}"
MASTER_LOG="${MASTER_LOG:-${TEMPORAL_RESULT_ROOT}/argmax_full_chain_submission.log}"

case "${MAPPING_LOGISTIC_CONCURRENCY}:${MAPPING_NEURAL_CONCURRENCY}" in
    *[!0-9:]*|0:*|*:0) echo "[preflight] concurrency values must be positive integers" >&2; exit 2 ;;
esac

write_master_state() {
    local temporary="${MASTER_STATE_FILE}.tmp"
    {
        printf 'RUN_ID=%q\n' "${RUN_ID}"
        printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
        printf 'DATA_DIR=%q\n' "${DATA_DIR}"
        printf 'MAPPING_DECISION_RULE=%q\n' "argmax"
        printf 'MAPPING_LOGISTIC_CONCURRENCY=%q\n' "${MAPPING_LOGISTIC_CONCURRENCY}"
        printf 'MAPPING_NEURAL_CONCURRENCY=%q\n' "${MAPPING_NEURAL_CONCURRENCY}"
        printf 'SPATIAL_BUNDLE_ID=%q\n' "${SPATIAL_BUNDLE_ID}"
        printf 'TEMPORAL_BUNDLE_ID=%q\n' "${TEMPORAL_BUNDLE_ID}"
        printf 'SPATIAL_RESULT_ROOT=%q\n' "${SPATIAL_RESULT_ROOT}"
        printf 'TEMPORAL_RESULT_ROOT=%q\n' "${TEMPORAL_RESULT_ROOT}"
        printf 'SPATIAL_STATE_FILE=%q\n' "${SPATIAL_STATE_FILE:-}"
        printf 'TEMPORAL_STATE_FILE=%q\n' "${TEMPORAL_STATE_FILE:-}"
        printf 'SPATIAL_AGGREGATE_JOB_ID=%q\n' "${SPATIAL_AGGREGATE_JOB_ID:-}"
        printf 'TEMPORAL_AGGREGATE_JOB_ID=%q\n' "${TEMPORAL_AGGREGATE_JOB_ID:-}"
        printf 'PLOT_JOB_ID=%q\n' "${PLOT_JOB_ID:-}"
        printf 'SUBMISSION_STATUS=%q\n' "${SUBMISSION_STATUS:-initializing}"
    } > "${temporary}"
    mv "${temporary}" "${MASTER_STATE_FILE}"
}

state_value() {
    local state_file="$1"
    local variable="$2"
    bash -c 'set -euo pipefail; source "$1"; printf "%s" "${!2}"' _ "${state_file}" "${variable}"
}

if [ "${1:-}" != "--worker" ]; then
    mkdir -p "${SPATIAL_RESULT_ROOT}" "${TEMPORAL_RESULT_ROOT}"
    echo "[launcher] run_id=${RUN_ID}"
    echo "[launcher] spatial_result_root=${SPATIAL_RESULT_ROOT}"
    echo "[launcher] temporal_result_root=${TEMPORAL_RESULT_ROOT}"
    echo "[launcher] master_state=${MASTER_STATE_FILE}"
    echo "[launcher] master_log=${MASTER_LOG}"
    REPO_ROOT="${REPO_ROOT}" DATA_DIR="${DATA_DIR}" RUN_ID="${RUN_ID}" \
        SPATIAL_BUNDLE_ID="${SPATIAL_BUNDLE_ID}" \
        TEMPORAL_BUNDLE_ID="${TEMPORAL_BUNDLE_ID}" \
        SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT}" \
        TEMPORAL_RESULT_ROOT="${TEMPORAL_RESULT_ROOT}" \
        MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY}" \
        MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY}" \
        MASTER_STATE_FILE="${MASTER_STATE_FILE}" MASTER_LOG="${MASTER_LOG}" \
        nohup bash "${REPO_ROOT}/submit_mapping_cv_argmax_full_chain.sh" --worker \
        </dev/null >"${MASTER_LOG}" 2>&1 &
    echo "[launcher] detached_pid=$!"
    exit 0
fi

mkdir -p "${SPATIAL_RESULT_ROOT}" "${TEMPORAL_RESULT_ROOT}"
for required in \
    submit_spatial_mapping_cv_chain.sh \
    submit_temporal_mapping_cv_chain.sh \
    sub_spatial_mapping_logistic_cv.slurm \
    sub_spatial_mapping_neural_cv.slurm \
    sub_aggregate_spatial_mapping_cv.slurm \
    sub_plot_mapping_cv.slurm; do
    test -s "${REPO_ROOT}/${required}"
    echo "[preflight] ${required}=OK"
done

SPATIAL_STATE_FILE="${SPATIAL_RESULT_ROOT}/submission_state.sh"
TEMPORAL_STATE_FILE="${TEMPORAL_RESULT_ROOT}/submission_state.sh"
SUBMISSION_STATUS=submitting_spatial
write_master_state

REPO_ROOT="${REPO_ROOT}" DATA_DIR="${DATA_DIR}" \
    BUNDLE_ID="${SPATIAL_BUNDLE_ID}" RESULT_ROOT="${SPATIAL_RESULT_ROOT}" \
    STATE_FILE="${SPATIAL_STATE_FILE}" MAPPING_DECISION_RULE=argmax \
    MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY}" \
    MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY}" \
    bash "${REPO_ROOT}/submit_spatial_mapping_cv_chain.sh" --worker

SPATIAL_AGGREGATE_JOB_ID="$(state_value "${SPATIAL_STATE_FILE}" AGGREGATE_JOB_ID)"
test -n "${SPATIAL_AGGREGATE_JOB_ID}"
SUBMISSION_STATUS=submitting_temporal
write_master_state

REPO_ROOT="${REPO_ROOT}" DATA_DIR="${DATA_DIR}" \
    BUNDLE_ID="${TEMPORAL_BUNDLE_ID}" RESULT_ROOT="${TEMPORAL_RESULT_ROOT}" \
    STATE_FILE="${TEMPORAL_STATE_FILE}" MAPPING_DECISION_RULE=argmax \
    MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY}" \
    MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY}" \
    SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT}" \
    SPATIAL_AGGREGATE_JOB_ID="${SPATIAL_AGGREGATE_JOB_ID}" \
    bash "${REPO_ROOT}/submit_temporal_mapping_cv_chain.sh" --worker

TEMPORAL_AGGREGATE_JOB_ID="$(state_value "${TEMPORAL_STATE_FILE}" AGGREGATE_JOB_ID)"
PLOT_JOB_ID="$(state_value "${TEMPORAL_STATE_FILE}" PLOT_JOB_ID)"
test -n "${TEMPORAL_AGGREGATE_JOB_ID}"
test -n "${PLOT_JOB_ID}"
SUBMISSION_STATUS=complete
write_master_state

echo "[submit] spatial_aggregate=${SPATIAL_AGGREGATE_JOB_ID}"
echo "[submit] temporal_aggregate=${TEMPORAL_AGGREGATE_JOB_ID}"
echo "[submit] combined_plot=${PLOT_JOB_ID}"
echo "[worker] master_state=${MASTER_STATE_FILE}"
