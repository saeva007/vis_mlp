#!/bin/bash
# Train only the missing 5/15/25/35/45% S2 sampling points, evaluate them,
# then merge them with the completed 0/10/20/30/40/50% evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/sub_static_rnn_lowvis_sampling_matrix.slurm}"
EVAL_SBATCH_SCRIPT="${EVAL_SBATCH_SCRIPT:-${BASE}/paper_eval/sub_static_rnn_lowvis_sampling_eval.slurm}"
MERGE_SBATCH_SCRIPT="${MERGE_SBATCH_SCRIPT:-${BASE}/paper_eval/sub_merge_static_rnn_sampling_curve.slurm}"
REDRAW_SBATCH_SCRIPT="${REDRAW_SBATCH_SCRIPT:-${BASE}/paper_eval/sub_plot_viscast_manuscript_composites.slurm}"
PRETRAINED_CKPT="${LOWVIS_RNN_PRETRAINED_CKPT:-${BASE}/checkpoints/exp_114287869_static_mlp_gru_main_S1_best_score.pt}"
SEED="${LOWVIS_RNN_SEED:-42}"
EXPERIMENTS="11 13 15 17 19"

if [[ "${1:-}" != "--worker" ]]; then
    test -f "${SBATCH_SCRIPT}" || { echo "ERROR: missing ${SBATCH_SCRIPT}" >&2; exit 2; }
    test -f "${EVAL_SBATCH_SCRIPT}" || { echo "ERROR: missing ${EVAL_SBATCH_SCRIPT}" >&2; exit 2; }
    test -f "${MERGE_SBATCH_SCRIPT}" || { echo "ERROR: missing ${MERGE_SBATCH_SCRIPT}" >&2; exit 2; }
    test -f "${REDRAW_SBATCH_SCRIPT}" || { echo "ERROR: missing ${REDRAW_SBATCH_SCRIPT}" >&2; exit 2; }
    test -s "${PRETRAINED_CKPT}" || { echo "ERROR: missing or empty ${PRETRAINED_CKPT}" >&2; exit 2; }
    if [[ "${LEGACY_EVAL_DIR:-auto}" != "auto" ]]; then
        test -s "${LEGACY_EVAL_DIR}/sampling_ablation_overall_metrics.csv" || {
            echo "ERROR: missing legacy sampling metrics under ${LEGACY_EVAL_DIR}" >&2
            exit 2
        }
    fi

    bundle_id="sampling_curve_fill5_$(date +%Y%m%d_%H%M%S)"
    run_prefix="exp_$(date +%Y%m%d_%H%M%S)_static_rnn_sampling_curve_fill5"
    state_file="${SCRIPT_DIR}/logs/${bundle_id}.state"
    launcher_log="${SCRIPT_DIR}/logs/${bundle_id}.launcher.log"
    eval_out_dir="paper_eval_results_${bundle_id}_new_points"
    merged_out_dir="paper_eval_results_${bundle_id}_complete_curve"
    manuscript_out_dir="paper_eval_results_pm10_pm25_journal/manuscript_main_figures_${bundle_id}"
    mkdir -p "${SCRIPT_DIR}/logs"

    echo "checkpoint=OK ${PRETRAINED_CKPT}"
    echo "training_script=OK ${SBATCH_SCRIPT}"
    echo "evaluation_script=OK ${EVAL_SBATCH_SCRIPT}"
    echo "merge_script=OK ${MERGE_SBATCH_SCRIPT}"
    echo "redraw_script=OK ${REDRAW_SBATCH_SCRIPT}"
    echo "training_only=5,15,25,35,45 percent; existing 0,10,20,30,40,50 percent are reused"

    nohup env \
        BASE="${BASE}" \
        SBATCH_SCRIPT="${SBATCH_SCRIPT}" \
        EVAL_SBATCH_SCRIPT="${EVAL_SBATCH_SCRIPT}" \
        MERGE_SBATCH_SCRIPT="${MERGE_SBATCH_SCRIPT}" \
        REDRAW_SBATCH_SCRIPT="${REDRAW_SBATCH_SCRIPT}" \
        LOWVIS_RNN_PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
        LOWVIS_RNN_SEED="${SEED}" \
        LEGACY_EVAL_DIR="${LEGACY_EVAL_DIR:-auto}" \
        BUNDLE_ID="${bundle_id}" \
        LOWVIS_RNN_RUN_PREFIX="${run_prefix}" \
        STATE_FILE="${state_file}" \
        EVAL_OUT_DIR="${eval_out_dir}" \
        MERGED_OUT_DIR="${merged_out_dir}" \
        MANUSCRIPT_OUT_DIR="${manuscript_out_dir}" \
        bash "${BASH_SOURCE[0]}" --worker </dev/null >"${launcher_log}" 2>&1 &
    launcher_pid=$!

    echo "launcher_pid=${launcher_pid}"
    echo "launcher_log=${launcher_log}"
    echo "state_file=${state_file}"
    echo "The launcher is detached; disconnecting SSH will not cancel submission."
    exit 0
fi

BUNDLE_ID="${BUNDLE_ID:?BUNDLE_ID is required}"
RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:?LOWVIS_RNN_RUN_PREFIX is required}"
STATE_FILE="${STATE_FILE:?STATE_FILE is required}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:?EVAL_OUT_DIR is required}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:?MERGED_OUT_DIR is required}"
MANUSCRIPT_OUT_DIR="${MANUSCRIPT_OUT_DIR:?MANUSCRIPT_OUT_DIR is required}"
LEGACY_EVAL_DIR="${LEGACY_EVAL_DIR:-auto}"
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-${BASE}/checkpoints}"
LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
THRESHOLD_SOURCE="${EVAL_THRESHOLD_SOURCE:-argmax}"

declare -a TRAIN_JOB_IDS=()
EVAL_JOB_ID=""
MERGE_JOB_ID=""
REDRAW_JOB_ID=""
STATUS="submitting"

write_state() {
    local tmp="${STATE_FILE}.tmp.$$"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'STATUS=%q\n' "${STATUS}"
        printf 'RUN_PREFIX=%q\n' "${RUN_PREFIX}"
        printf 'PRETRAINED_CKPT=%q\n' "${PRETRAINED_CKPT}"
        printf 'EXPERIMENTS=%q\n' "${EXPERIMENTS}"
        printf 'LEGACY_EVAL_DIR=%q\n' "${LEGACY_EVAL_DIR}"
        printf 'EVAL_OUT_DIR=%q\n' "${BASE}/${EVAL_OUT_DIR}"
        printf 'MERGED_OUT_DIR=%q\n' "${BASE}/${MERGED_OUT_DIR}"
        printf 'MANUSCRIPT_OUT_DIR=%q\n' "${BASE}/${MANUSCRIPT_OUT_DIR}"
        printf 'TRAIN_JOB_IDS=%q\n' "${TRAIN_JOB_IDS[*]:-}"
        printf 'EVAL_JOB_ID=%q\n' "${EVAL_JOB_ID}"
        printf 'MERGE_JOB_ID=%q\n' "${MERGE_JOB_ID}"
        printf 'REDRAW_JOB_ID=%q\n' "${REDRAW_JOB_ID}"
    } >"${tmp}"
    mv "${tmp}" "${STATE_FILE}"
}

on_error() {
    STATUS="failed"
    write_state
}
trap on_error ERR

mkdir -p "${SCRIPT_DIR}/logs"
write_state

for exp_id in ${EXPERIMENTS}; do
    job_id="$(
        sbatch --parsable \
            --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_SAMPLING_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_PREFIX}_${exp_id},LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_CKPT_DIR=${CKPT_DIR},LOWVIS_RNN_PRETRAINED_CKPT=${PRETRAINED_CKPT},LOWVIS_RNN_SEED=${SEED} \
            "${SBATCH_SCRIPT}"
    )"
    TRAIN_JOB_IDS+=("${job_id}")
    echo "submitted missing ratio experiment ${exp_id}: job ${job_id}"
    write_state
done

dependency="$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"
experiment_ids="${EXPERIMENTS// /:}"
EVAL_JOB_ID="$(
    sbatch --parsable \
        --dependency=afterok:${dependency} \
        --export=ALL,SAMPLING_RUN_PREFIX=${RUN_PREFIX},EXPERIMENTS=${experiment_ids},OUT_DIR=${EVAL_OUT_DIR},THRESHOLD_SOURCE=${THRESHOLD_SOURCE},DEVICE=cpu \
        "${EVAL_SBATCH_SCRIPT}"
)"
write_state

MERGE_JOB_ID="$(
    sbatch --parsable \
        --dependency=afterok:${EVAL_JOB_ID} \
        --export=ALL,BASE=${BASE},LEGACY_EVAL_DIR=${LEGACY_EVAL_DIR},FILL_EVAL_DIR=${BASE}/${EVAL_OUT_DIR},OUT_DIR=${BASE}/${MERGED_OUT_DIR} \
        "${MERGE_SBATCH_SCRIPT}"
)"

REDRAW_JOB_ID="$(
    sbatch --parsable \
        --dependency=afterok:${MERGE_JOB_ID} \
        --export=ALL,BASE=${BASE},MAIN_EVAL_DIR=${BASE}/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal,SAMPLING_CSV=${BASE}/${MERGED_OUT_DIR}/sampling_ablation_overall_metrics.csv,QCORE_ANALYSIS_DIR=${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/qcore_t925_mhtpw_formal_v2_20260728/analysis,OUT_DIR=${BASE}/${MANUSCRIPT_OUT_DIR} \
        "${REDRAW_SBATCH_SCRIPT}"
)"

STATUS="submitted"
write_state
trap - ERR

echo "evaluation job ${EVAL_JOB_ID} depends on ${dependency}"
echo "merge job ${MERGE_JOB_ID} depends on ${EVAL_JOB_ID}"
echo "redraw job ${REDRAW_JOB_ID} depends on ${MERGE_JOB_ID}"
echo "state_file=${STATE_FILE}"
echo "complete_curve_dir=${BASE}/${MERGED_OUT_DIR}"
echo "manuscript_figure_dir=${BASE}/${MANUSCRIPT_OUT_DIR}"
