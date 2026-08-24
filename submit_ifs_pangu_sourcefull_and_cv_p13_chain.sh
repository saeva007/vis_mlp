#!/bin/bash
# Submit the complete IFS/Pangu source-full P13 and IFS blocked-CV P13 DAG.
#
# One invocation submits every upstream and downstream Slurm job.  The worker
# only calls sbatch and persists a sourceable state file; training/evaluation
# starts exclusively through afterok dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TRAIN_DIR="${TRAIN_DIR:-${SCRIPT_DIR}}"
IFS_BASELINE_DIR="${IFS_BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}"
RUN_TAG="${RUN_TAG:-p13_ifs_pangu_sourcefull_cv_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/public/home/putianshu/vis_mlp/p13_sourcefull_and_cv/${RUN_TAG}}"
STATE_FILE="${STATE_FILE:-${RUN_ROOT}/submission_state.sh}"
SUBMIT_LOG="${SUBMIT_LOG:-${RUN_ROOT}/submission.log}"
JOB_MANIFEST="${JOB_MANIFEST:-${RUN_ROOT}/jobs.tsv}"
P13_SEEDS="${P13_SEEDS:-42:314:2718}"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/${RUN_TAG}}"
SOURCE_SEED_EVAL_ROOT="${SOURCE_SEED_EVAL_ROOT:-${RUN_ROOT}/source_seed_eval}"
SOURCE_AGGREGATE_ROOT="${SOURCE_AGGREGATE_ROOT:-${RUN_ROOT}/source_p13_aggregate}"
P13_MAINLINE_PER_SAMPLE="${P13_MAINLINE_PER_SAMPLE:-/public/home/putianshu/vis_mlp/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal/per_sample_eval.csv}"
IFS_SOURCE_S1_DATA="${IFS_SOURCE_S1_DATA:-${IFS_BASELINE_DIR}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_ifs}"
IFS_SOURCE_S2_DATA="${IFS_SOURCE_S2_DATA:-${IFS_BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
PANGU_SOURCE_S1_DATA="${PANGU_SOURCE_S1_DATA:-${IFS_BASELINE_DIR}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_pangu2025}"
PANGU_SOURCE_S2_DATA="${PANGU_SOURCE_S2_DATA:-${IFS_BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full}"
CV_RUN_ID="${CV_RUN_ID:-${RUN_TAG}_ifs_gru}"
CV_SPATIAL_ROOT="${CV_SPATIAL_ROOT:-/public/home/putianshu/vis_mlp/spatial_mapping_cv_ifs_p13/${CV_RUN_ID}_spatial}"
CV_TEMPORAL_ROOT="${CV_TEMPORAL_ROOT:-/public/home/putianshu/vis_mlp/temporal_mapping_cv_ifs_p13/${CV_RUN_ID}_temporal}"
CV_MASTER_STATE="${CV_MASTER_STATE:-${CV_TEMPORAL_ROOT}/p13_full_chain_state.sh}"
CV_MASTER_LOG="${CV_MASTER_LOG:-${CV_TEMPORAL_ROOT}/p13_full_chain_submission.log}"

# Exact P13 candidate settings.  The source-full S1/S2 pipeline uses the same
# arguments, while fold-local CV trains each held-out fold from scratch and then
# applies Phase D; it never reuses a mainline S1 checkpoint.
P13_TRAIN_ARGS="--threshold-mode argmax --loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --sampler-mode stratified_balanced --fog-ratio-s2 0.18 --mist-ratio-s2 0.22 --s2-phase-d-steps 3000 --s2-lr-head-d 2e-5 --phase-d-natural-mix 0.30 --phase-d-ramp-start 0.50 --phase-d-event-batch-ratio 0.50 --phase-d-min-fog-count 40 --phase-d-selection-metric sampling_csi --phase-d-min-low-vis-csi 0.195 --phase-d-max-fpr 0.025 --phase-d-min-event-mean-csi 0.235 --phase-d-min-event-mean-recall 0.45 --phase-d-min-event-recall 0.20 --phase-d-min-event-area-ratio-mean 0.80 --phase-d-max-event-area-ratio-mean 1.80 --phase-d-max-event-area-ratio 2.20"

parse_seeds() {
    local normalized="${P13_SEEDS//:/,}"
    IFS=',' read -r -a P13_SEED_VALUES <<< "${normalized}"
    if [ "${#P13_SEED_VALUES[@]}" -ne 3 ]; then
        echo "ERROR: P13_SEEDS must contain exactly three seeds, got ${P13_SEEDS}" >&2
        exit 2
    fi
    local seen=""
    local seed
    for seed in "${P13_SEED_VALUES[@]}"; do
        seed="$(echo "${seed}" | tr -d '[:space:]')"
        case "${seed}" in
            ''|*[!0-9]*) echo "ERROR: invalid P13 seed list: ${P13_SEEDS}" >&2; exit 2 ;;
        esac
        if [[ ":${seen}:" == *":${seed}:"* ]]; then
            echo "ERROR: duplicate P13 seed: ${seed}" >&2
            exit 2
        fi
        seen="${seen:+${seen}:}${seed}"
    done
    P13_SEED_VALUES=( ${seen//:/ } )
    if [ "${P13_SEED_VALUES[*]}" != "42 314 2718" ]; then
        echo "ERROR: formal P13 seed contract is 42:314:2718, got ${P13_SEEDS}" >&2
        exit 2
    fi
}

write_state() {
    local temporary="${STATE_FILE}.tmp"
    {
        printf 'RUN_TAG=%q\n' "${RUN_TAG}"
        printf 'TRAIN_DIR=%q\n' "${TRAIN_DIR}"
        printf 'IFS_BASELINE_DIR=%q\n' "${IFS_BASELINE_DIR}"
        printf 'P13_SEEDS=%q\n' "${P13_SEEDS}"
        printf 'SOURCE_CKPT_DIR=%q\n' "${SOURCE_CKPT_DIR}"
        printf 'SOURCE_SEED_EVAL_ROOT=%q\n' "${SOURCE_SEED_EVAL_ROOT}"
        printf 'SOURCE_AGGREGATE_ROOT=%q\n' "${SOURCE_AGGREGATE_ROOT}"
        printf 'P13_MAINLINE_PER_SAMPLE=%q\n' "${P13_MAINLINE_PER_SAMPLE}"
        printf 'CV_RUN_ID=%q\n' "${CV_RUN_ID}"
        printf 'CV_SPATIAL_ROOT=%q\n' "${CV_SPATIAL_ROOT}"
        printf 'CV_TEMPORAL_ROOT=%q\n' "${CV_TEMPORAL_ROOT}"
        printf 'CV_MASTER_STATE=%q\n' "${CV_MASTER_STATE}"
        printf 'CV_MASTER_LOG=%q\n' "${CV_MASTER_LOG}"
        printf 'SOURCE_S1_JOB_IDS=%q\n' "${SOURCE_S1_JOB_IDS:-}"
        printf 'SOURCE_S2_JOB_IDS=%q\n' "${SOURCE_S2_JOB_IDS:-}"
        printf 'SOURCE_EVAL_JOB_IDS=%q\n' "${SOURCE_EVAL_JOB_IDS:-}"
        printf 'SOURCE_AGGREGATE_JOB_ID=%q\n' "${SOURCE_AGGREGATE_JOB_ID:-}"
        printf 'SUBMISSION_STATUS=%q\n' "${SUBMISSION_STATUS:-initializing}"
    } > "${temporary}"
    mv "${temporary}" "${STATE_FILE}"
}

record_job() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${JOB_MANIFEST}"
}

fail_state() {
    local code="$?"
    SUBMISSION_STATUS="failed"
    write_state || true
    exit "${code}"
}

if [ "${1:-}" != "--worker" ]; then
    if [ -e "${RUN_ROOT}" ]; then
        echo "ERROR: RUN_ROOT already exists; choose a fresh RUN_TAG/RUN_ROOT: ${RUN_ROOT}" >&2
        exit 3
    fi
    mkdir -p "${RUN_ROOT}"
    echo "[launcher] run_tag=${RUN_TAG}"
    echo "[launcher] state=${STATE_FILE}"
    echo "[launcher] log=${SUBMIT_LOG}"
    RUN_TAG="${RUN_TAG}" RUN_ROOT="${RUN_ROOT}" STATE_FILE="${STATE_FILE}" \
        SUBMIT_LOG="${SUBMIT_LOG}" JOB_MANIFEST="${JOB_MANIFEST}" \
        TRAIN_DIR="${TRAIN_DIR}" IFS_BASELINE_DIR="${IFS_BASELINE_DIR}" \
        P13_SEEDS="${P13_SEEDS}" SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR}" \
        SOURCE_SEED_EVAL_ROOT="${SOURCE_SEED_EVAL_ROOT}" \
        SOURCE_AGGREGATE_ROOT="${SOURCE_AGGREGATE_ROOT}" \
        P13_MAINLINE_PER_SAMPLE="${P13_MAINLINE_PER_SAMPLE}" \
        IFS_SOURCE_S1_DATA="${IFS_SOURCE_S1_DATA}" IFS_SOURCE_S2_DATA="${IFS_SOURCE_S2_DATA}" \
        PANGU_SOURCE_S1_DATA="${PANGU_SOURCE_S1_DATA}" PANGU_SOURCE_S2_DATA="${PANGU_SOURCE_S2_DATA}" \
        CV_RUN_ID="${CV_RUN_ID}" CV_SPATIAL_ROOT="${CV_SPATIAL_ROOT}" \
        CV_TEMPORAL_ROOT="${CV_TEMPORAL_ROOT}" CV_MASTER_STATE="${CV_MASTER_STATE}" \
        CV_MASTER_LOG="${CV_MASTER_LOG}" \
        nohup bash "${SCRIPT_DIR}/submit_ifs_pangu_sourcefull_and_cv_p13_chain.sh" --worker \
        </dev/null >"${SUBMIT_LOG}" 2>&1 &
    echo "[launcher] detached_pid=$!"
    exit 0
fi

trap fail_state ERR
parse_seeds
mkdir -p "${RUN_ROOT}" "${SOURCE_CKPT_DIR}" "${SOURCE_SEED_EVAL_ROOT}"
printf 'stage\tsource\tseed\tjob_id\tdependency\trun_id\toutput\n' > "${JOB_MANIFEST}"
SOURCE_S1_JOB_IDS=""
SOURCE_S2_JOB_IDS=""
SOURCE_EVAL_JOB_IDS=""
SOURCE_AGGREGATE_JOB_ID=""
SUBMISSION_STATUS="preflight"
write_state

for required in \
    "${TRAIN_DIR}/submit_ifs_gru_mapping_cv_chain.sh" \
    "${TRAIN_DIR}/submit_mapping_cv_argmax_full_chain.sh" \
    "${IFS_BASELINE_DIR}/sub_ifs_overlap_baseline.slurm" \
    "${IFS_BASELINE_DIR}/sub_source_full_p13_seed_eval.slurm" \
    "${IFS_BASELINE_DIR}/sub_aggregate_source_full_p13.slurm" \
    "${IFS_BASELINE_DIR}/aggregate_source_full_p13.py" \
    "${P13_MAINLINE_PER_SAMPLE}"; do
    test -s "${required}"
    echo "[preflight] ${required}=OK"
done
for required_dir in \
    "${IFS_SOURCE_S1_DATA}" "${IFS_SOURCE_S2_DATA}" \
    "${PANGU_SOURCE_S1_DATA}" "${PANGU_SOURCE_S2_DATA}"; do
    test -d "${required_dir}"
    echo "[preflight] ${required_dir}=OK"
done

submit_source_s1() {
    local source="$1"
    local seed="$2"
    local experiment="$3"
    local run_id="$4"
    local cache_id="${RUN_TAG}_${source}_seed${seed}_s1"
    local job
    job=$(sbatch --parsable \
        --job-name="p13sf_${source}_s1_${seed}" \
        --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_CKPT_DIR=${SOURCE_CKPT_DIR},LOWVIS_RNN_EXTRA_ARGS=${P13_TRAIN_ARGS} --seed ${seed},LOWVIS_RNN_S1_STEPS=15000,LOWVIS_RNN_S2_A_STEPS=8000,LOWVIS_RNN_S2_B_STEPS=22000,LOWVIS_RNN_VAL_INTERVAL=500,LOWVIS_RNN_BATCH_SIZE=512,LOWVIS_RNN_GRAD_ACCUM=2,LOWVIS_RNN_NUM_WORKERS=0,LOWVIS_RNN_PATIENCE=10,LOWVIS_RNN_LOCAL_CACHE_ID=${cache_id}" \
        "${IFS_BASELINE_DIR}/sub_ifs_overlap_baseline.slurm")
    printf '%s' "${job%%;*}"
}

submit_source_s2() {
    local source="$1"
    local seed="$2"
    local experiment="$3"
    local run_id="$4"
    local s1_checkpoint="$5"
    local s1_job="$6"
    local cache_id="${RUN_TAG}_${source}_seed${seed}_s2"
    local job
    job=$(sbatch --parsable \
        --job-name="p13sf_${source}_s2_${seed}" \
        --dependency="afterok:${s1_job}" \
        --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_CKPT_DIR=${SOURCE_CKPT_DIR},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_checkpoint},LOWVIS_RNN_EXTRA_ARGS=${P13_TRAIN_ARGS} --seed ${seed},LOWVIS_RNN_S1_STEPS=15000,LOWVIS_RNN_S2_A_STEPS=8000,LOWVIS_RNN_S2_B_STEPS=22000,LOWVIS_RNN_VAL_INTERVAL=500,LOWVIS_RNN_BATCH_SIZE=512,LOWVIS_RNN_GRAD_ACCUM=2,LOWVIS_RNN_NUM_WORKERS=0,LOWVIS_RNN_PATIENCE=10,LOWVIS_RNN_LOCAL_CACHE_ID=${cache_id}" \
        "${IFS_BASELINE_DIR}/sub_ifs_overlap_baseline.slurm")
    printf '%s' "${job%%;*}"
}

submit_seed_eval() {
    local source="$1"
    local seed="$2"
    local data_dir="$3"
    local checkpoint="$4"
    local s2_job="$5"
    local out_dir="${SOURCE_SEED_EVAL_ROOT}/${source}/seed_${seed}"
    local job
    job=$(sbatch --parsable \
        --job-name="p13sf_${source}_eval_${seed}" \
        --dependency="afterok:${s2_job}" \
        --export="ALL,SOURCE_TAG=${source},SOURCE_SEED=${seed},SOURCE_DATA_DIR=${data_dir},SOURCE_CKPT=${checkpoint},OUT_DIR=${out_dir},BASELINE_DIR=${IFS_BASELINE_DIR},TRAIN_DIR=${TRAIN_DIR}" \
        "${IFS_BASELINE_DIR}/sub_source_full_p13_seed_eval.slurm")
    printf '%s' "${job%%;*}"
}

for source in ifs pangu2025_source_full; do
    case "${source}" in
        ifs)
            s1_experiment="s1_source_full_ifs"
            s2_experiment="s2_ifs_source_full"
            source_s2_data="${IFS_SOURCE_S2_DATA}"
            ;;
        pangu2025_source_full)
            s1_experiment="s1_source_full_pangu2025"
            s2_experiment="s2_pangu2025_source_full"
            source_s2_data="${PANGU_SOURCE_S2_DATA}"
            ;;
    esac
    for seed in "${P13_SEED_VALUES[@]}"; do
        run_id="${RUN_TAG}_${source}_seed${seed}"
        s1_checkpoint="${SOURCE_CKPT_DIR}/${run_id}_S1_best_score.pt"
        s2_checkpoint="${SOURCE_CKPT_DIR}/${run_id}_S2_PhaseD_best_score.pt"
        s1_job="$(submit_source_s1 "${source}" "${seed}" "${s1_experiment}" "${run_id}")"
        SOURCE_S1_JOB_IDS="${SOURCE_S1_JOB_IDS:+${SOURCE_S1_JOB_IDS}:}${s1_job}"
        record_job "source_s1" "${source}" "${seed}" "${s1_job}" "" "${run_id}" "${s1_checkpoint}"
        write_state
        s2_job="$(submit_source_s2 "${source}" "${seed}" "${s2_experiment}" "${run_id}" "${s1_checkpoint}" "${s1_job}")"
        SOURCE_S2_JOB_IDS="${SOURCE_S2_JOB_IDS:+${SOURCE_S2_JOB_IDS}:}${s2_job}"
        record_job "source_s2" "${source}" "${seed}" "${s2_job}" "afterok:${s1_job}" "${run_id}" "${s2_checkpoint}"
        write_state
        eval_job="$(submit_seed_eval "${source}" "${seed}" "${source_s2_data}" "${s2_checkpoint}" "${s2_job}")"
        SOURCE_EVAL_JOB_IDS="${SOURCE_EVAL_JOB_IDS:+${SOURCE_EVAL_JOB_IDS}:}${eval_job}"
        record_job "source_seed_eval" "${source}" "${seed}" "${eval_job}" "afterok:${s2_job}" "${run_id}" "${SOURCE_SEED_EVAL_ROOT}/${source}/seed_${seed}"
        write_state
    done
done

SOURCE_AGGREGATE_DEPENDENCY="afterok:${SOURCE_EVAL_JOB_IDS}"
SOURCE_AGGREGATE_JOB_ID=$(sbatch --parsable \
    --job-name="p13sf_aggregate" \
    --dependency="${SOURCE_AGGREGATE_DEPENDENCY}" \
    --export="ALL,BASELINE_DIR=${IFS_BASELINE_DIR},SEED_EVAL_ROOT=${SOURCE_SEED_EVAL_ROOT},OUT_ROOT=${SOURCE_AGGREGATE_ROOT},P13_SEEDS=${P13_SEEDS},MAINLINE_P13_PER_SAMPLE=${P13_MAINLINE_PER_SAMPLE}" \
    "${IFS_BASELINE_DIR}/sub_aggregate_source_full_p13.slurm")
SOURCE_AGGREGATE_JOB_ID="${SOURCE_AGGREGATE_JOB_ID%%;*}"
record_job "source_p13_aggregate" "ifs+pangu2025" "${P13_SEEDS}" "${SOURCE_AGGREGATE_JOB_ID}" "${SOURCE_AGGREGATE_DEPENDENCY}" "${RUN_TAG}" "${SOURCE_AGGREGATE_ROOT}"
SUBMISSION_STATUS="source_dag_submitted"
write_state

IFS_CV_REPO_ROOT="${TRAIN_DIR}" \
    DATA_DIR="${IFS_SOURCE_S2_DATA}" \
    RUN_ID="${CV_RUN_ID}" \
    SPATIAL_BUNDLE_ID="${CV_RUN_ID}_spatial" \
    TEMPORAL_BUNDLE_ID="${CV_RUN_ID}_temporal" \
    SPATIAL_RESULT_ROOT="${CV_SPATIAL_ROOT}" \
    TEMPORAL_RESULT_ROOT="${CV_TEMPORAL_ROOT}" \
    MASTER_STATE_FILE="${CV_MASTER_STATE}" \
    MASTER_LOG="${CV_MASTER_LOG}" \
    MAPPING_MODELS=gru \
    MAPPING_P13=1 \
    MAPPING_CV_SEEDS="${P13_SEEDS}" \
    MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY:-5}" \
    MAPPING_LOGISTIC_CONCURRENCY=1 \
    MAPPING_GRU_LABEL="IFS-input P13 Static-MLP + GRU (12 h)" \
    IFS_PER_SAMPLE_CSV="${P13_MAINLINE_PER_SAMPLE}" \
    bash "${TRAIN_DIR}/submit_ifs_gru_mapping_cv_chain.sh" --worker

record_job "ifs_p13_cv_dag" "ifs" "${P13_SEEDS}" "state_file" "internal_afterok" "${CV_RUN_ID}" "${CV_MASTER_STATE}"
SUBMISSION_STATUS="complete"
write_state
echo "[done] state=${STATE_FILE}"
echo "[done] jobs=${JOB_MANIFEST}"
