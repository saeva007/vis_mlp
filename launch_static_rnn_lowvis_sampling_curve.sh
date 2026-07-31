#!/bin/bash
# Submit an SSH-resilient S2-only Low-vis sampling curve (0% to 50%, step 10%).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/sub_static_rnn_lowvis_sampling_matrix.slurm}"
EVAL_SBATCH_SCRIPT="${EVAL_SBATCH_SCRIPT:-${BASE}/paper_eval/sub_static_rnn_lowvis_sampling_eval.slurm}"
PRETRAINED_CKPT="${LOWVIS_RNN_PRETRAINED_CKPT:-${BASE}/checkpoints/exp_114287869_static_mlp_gru_main_S1_best_score.pt}"
SEED="${LOWVIS_RNN_SEED:-42}"
EXPERIMENTS="10 11 12 13 14 15"

if [[ "${1:-}" != "--worker" ]]; then
    test -f "${SBATCH_SCRIPT}" || { echo "ERROR: missing ${SBATCH_SCRIPT}" >&2; exit 2; }
    test -f "${EVAL_SBATCH_SCRIPT}" || { echo "ERROR: missing ${EVAL_SBATCH_SCRIPT}" >&2; exit 2; }
    test -s "${PRETRAINED_CKPT}" || { echo "ERROR: missing or empty ${PRETRAINED_CKPT}" >&2; exit 2; }

    bundle_id="sampling_curve_$(date +%Y%m%d_%H%M%S)"
    run_prefix="exp_$(date +%Y%m%d_%H%M%S)_static_rnn_sampling_curve"
    state_file="${SCRIPT_DIR}/logs/${bundle_id}.state"
    launcher_log="${SCRIPT_DIR}/logs/${bundle_id}.launcher.log"
    eval_out_dir="paper_eval_results_${bundle_id}"
    mkdir -p "${SCRIPT_DIR}/logs"

    echo "checkpoint=OK ${PRETRAINED_CKPT}"
    echo "training_script=OK ${SBATCH_SCRIPT}"
    echo "evaluation_script=OK ${EVAL_SBATCH_SCRIPT}"
    echo "curve=0,10,20,30,40,50 percent; S1 checkpoint fixed; S2 sampling only"

    nohup env \
        BASE="${BASE}" \
        SBATCH_SCRIPT="${SBATCH_SCRIPT}" \
        EVAL_SBATCH_SCRIPT="${EVAL_SBATCH_SCRIPT}" \
        LOWVIS_RNN_PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
        LOWVIS_RNN_SEED="${SEED}" \
        BUNDLE_ID="${bundle_id}" \
        LOWVIS_RNN_RUN_PREFIX="${run_prefix}" \
        STATE_FILE="${state_file}" \
        LAUNCHER_LOG="${launcher_log}" \
        EVAL_OUT_DIR="${eval_out_dir}" \
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
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-${BASE}/checkpoints}"
CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${RUN_PREFIX}_shared_data}"
LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
THRESHOLD_SOURCE="${EVAL_THRESHOLD_SOURCE:-argmax}"

declare -a TRAIN_JOB_IDS=()
EVAL_JOB_ID=""
STATUS="submitting"

write_state() {
    local tmp="${STATE_FILE}.tmp.$$"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'STATUS=%q\n' "${STATUS}"
        printf 'RUN_PREFIX=%q\n' "${RUN_PREFIX}"
        printf 'PRETRAINED_CKPT=%q\n' "${PRETRAINED_CKPT}"
        printf 'SEED=%q\n' "${SEED}"
        printf 'EXPERIMENTS=%q\n' "${EXPERIMENTS}"
        printf 'EVAL_OUT_DIR=%q\n' "${EVAL_OUT_DIR}"
        printf 'TRAIN_JOB_IDS=%q\n' "${TRAIN_JOB_IDS[*]:-}"
        printf 'EVAL_JOB_ID=%q\n' "${EVAL_JOB_ID}"
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
            --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_SAMPLING_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID},LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_CKPT_DIR=${CKPT_DIR},LOWVIS_RNN_PRETRAINED_CKPT=${PRETRAINED_CKPT},LOWVIS_RNN_SEED=${SEED} \
            "${SBATCH_SCRIPT}"
    )"
    TRAIN_JOB_IDS+=("${job_id}")
    echo "submitted ratio experiment ${exp_id}: job ${job_id}"
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

STATUS="submitted"
write_state
trap - ERR

echo "evaluation job ${EVAL_JOB_ID} depends on ${dependency}"
echo "state_file=${STATE_FILE}"
echo "output_dir=${BASE}/${EVAL_OUT_DIR}"
