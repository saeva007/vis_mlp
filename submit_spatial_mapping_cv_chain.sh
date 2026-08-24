#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
BUNDLE_ID="${BUNDLE_ID:-spatial_mapping_cv_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/public/home/putianshu/vis_mlp/spatial_mapping_cv}"
RESULT_ROOT="${RESULT_ROOT:-${RESULT_BASE}/${BUNDLE_ID}}"
MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE:-argmax}"
MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY:-5}"
MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY:-4}"
MAPPING_MODELS="${MAPPING_MODELS:-logistic,mlp,gru}"
MAPPING_P13="${MAPPING_P13:-0}"
MAPPING_CV_SEEDS="${MAPPING_CV_SEEDS:-}"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-/public/home/putianshu/vis_mlp/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal/per_sample_eval.csv}"
STATE_FILE="${STATE_FILE:-${RESULT_ROOT}/submission_state.sh}"
SUBMIT_LOG="${SUBMIT_LOG:-${RESULT_ROOT}/submission.log}"

write_state() {
    local temporary="${STATE_FILE}.tmp"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'RESULT_ROOT=%q\n' "${RESULT_ROOT}"
        printf 'DATA_DIR=%q\n' "${DATA_DIR}"
        printf 'MAPPING_DECISION_RULE=%q\n' "${MAPPING_DECISION_RULE}"
        printf 'MAPPING_LOGISTIC_CONCURRENCY=%q\n' "${MAPPING_LOGISTIC_CONCURRENCY}"
        printf 'MAPPING_NEURAL_CONCURRENCY=%q\n' "${MAPPING_NEURAL_CONCURRENCY}"
        printf 'MAPPING_MODELS=%q\n' "${MAPPING_MODELS}"
        printf 'MAPPING_P13=%q\n' "${MAPPING_P13}"
        printf 'MAPPING_CV_SEEDS=%q\n' "${MAPPING_CV_SEEDS}"
        printf 'IFS_PER_SAMPLE_CSV=%q\n' "${IFS_PER_SAMPLE_CSV}"
        printf 'PREP_JOB_ID=%q\n' "${PREP_JOB_ID:-}"
        printf 'LOGISTIC_JOB_ID=%q\n' "${LOGISTIC_JOB_ID:-}"
        printf 'NEURAL_JOB_ID=%q\n' "${NEURAL_JOB_ID:-}"
        printf 'AGGREGATE_JOB_ID=%q\n' "${AGGREGATE_JOB_ID:-}"
        printf 'SUBMISSION_STATUS=%q\n' "${SUBMISSION_STATUS:-initializing}"
    } > "${temporary}"
    mv "${temporary}" "${STATE_FILE}"
}

if [ "${1:-}" != "--worker" ]; then
    mkdir -p "${RESULT_ROOT}"
    echo "[launcher] bundle=${BUNDLE_ID}"
    echo "[launcher] result_root=${RESULT_ROOT}"
    echo "[launcher] state_file=${STATE_FILE}"
    echo "[launcher] submit_log=${SUBMIT_LOG}"
    BUNDLE_ID="${BUNDLE_ID}" RESULT_ROOT="${RESULT_ROOT}" DATA_DIR="${DATA_DIR}" \
        MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE}" \
        MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY}" \
        MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY}" \
        MAPPING_MODELS="${MAPPING_MODELS}" \
        MAPPING_P13="${MAPPING_P13}" MAPPING_CV_SEEDS="${MAPPING_CV_SEEDS}" \
        IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV}" \
        STATE_FILE="${STATE_FILE}" SUBMIT_LOG="${SUBMIT_LOG}" \
        nohup bash "${REPO_ROOT}/submit_spatial_mapping_cv_chain.sh" --worker \
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
echo "[worker] decision_rule=${MAPPING_DECISION_RULE} models=${MAPPING_MODELS} logistic_concurrency=${MAPPING_LOGISTIC_CONCURRENCY} neural_concurrency=${MAPPING_NEURAL_CONCURRENCY}"
echo "[worker] p13=${MAPPING_P13} cv_seeds=${MAPPING_CV_SEEDS:-single_seed}"
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

if [ -n "${MAPPING_CV_SEEDS}" ]; then
    if [ "${MAPPING_P13}" != "1" ] || [ "${MAPPING_DECISION_RULE}" != "argmax" ]; then
        echo "[preflight] seed-ensemble CV requires MAPPING_P13=1 and MAPPING_DECISION_RULE=argmax" >&2
        exit 2
    fi
    if [ "${#REQUESTED_MODELS[@]}" -ne 1 ] || [ "${REQUESTED_MODELS[0]}" != "gru" ]; then
        echo "[preflight] seed-ensemble CV is defined for the GRU mapping model only" >&2
        exit 2
    fi
    MAPPING_CV_SEED_COUNT="$(printf '%s' "${MAPPING_CV_SEEDS}" | tr ':,' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
    case "${MAPPING_CV_SEED_COUNT}" in
        ''|0|*[!0-9]*) echo "[preflight] invalid MAPPING_CV_SEEDS=${MAPPING_CV_SEEDS}" >&2; exit 2 ;;
    esac
else
    MAPPING_CV_SEED_COUNT=0
fi

for script in \
    activate_lowvis_diffusion_runtime.sh \
    spatial_mapping_cv.py \
    train_static_rnn_lowvis.py \
    prepare_static_rnn_local_cache.sh \
    sub_prepare_spatial_mapping_cv.slurm \
    sub_spatial_mapping_logistic_cv.slurm \
    sub_spatial_mapping_neural_cv.slurm \
    sub_aggregate_spatial_mapping_cv.slurm; do
    test -s "${REPO_ROOT}/${script}"
    echo "[preflight] ${script}=OK"
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

SUBMISSION_STATUS=preflight_ok
write_state

PREP_JOB_ID=$(sbatch --parsable \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATA_DIR=${DATA_DIR},RESULT_ROOT=${RESULT_ROOT},IFS_PER_SAMPLE_CSV=${IFS_PER_SAMPLE_CSV}" \
    "${REPO_ROOT}/sub_prepare_spatial_mapping_cv.slurm")
PREP_JOB_ID="${PREP_JOB_ID%%;*}"
SUBMISSION_STATUS=prep_submitted
write_state
echo "[submit] prep_job=${PREP_JOB_ID}"

LOGISTIC_JOB_ID=""
if [ -n "${MODEL_SEEN[logistic]:-}" ]; then
    LOGISTIC_JOB_ID=$(sbatch --parsable \
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
NEURAL_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},DATA_DIR=${DATA_DIR},RESULT_ROOT=${RESULT_ROOT},BUNDLE_ID=${BUNDLE_ID},MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}"
if [ "${MAPPING_CV_SEED_COUNT}" -gt 0 ]; then
    NEURAL_ARRAY="0-$(( MAPPING_CV_SEED_COUNT * 5 - 1 ))%${MAPPING_NEURAL_CONCURRENCY}"
    NEURAL_EXPORT="${NEURAL_EXPORT},MAPPING_NEURAL_MODEL=gru,MAPPING_P13=1,MAPPING_CV_SEEDS=${MAPPING_CV_SEEDS}"
elif [ -n "${MODEL_SEEN[mlp]:-}" ] && [ -n "${MODEL_SEEN[gru]:-}" ]; then
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
AGGREGATE_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},RESULT_ROOT=${RESULT_ROOT},MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE},IFS_PER_SAMPLE_CSV=${IFS_PER_SAMPLE_CSV},REQUIRE_IFS_BASELINE=1,MAPPING_CV_SEEDS=${MAPPING_CV_SEEDS}"
AGGREGATE_JOB_ID=$(sbatch --parsable \
    --dependency="${AGGREGATE_DEPENDENCY}" \
    --export="${AGGREGATE_EXPORT}" \
    "${REPO_ROOT}/sub_aggregate_spatial_mapping_cv.slurm")
AGGREGATE_JOB_ID="${AGGREGATE_JOB_ID%%;*}"
SUBMISSION_STATUS=complete
write_state
echo "[submit] aggregate_job=${AGGREGATE_JOB_ID}"
echo "[worker] state_file=${STATE_FILE}"
echo "[worker] completed=$(date --iso-8601=seconds)"
