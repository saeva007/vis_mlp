#!/bin/bash
#
# Submit the sampling-method ablation as independent S1 jobs plus dependent S2
# jobs. Optionally submit one CPU evaluation job after every S2 job succeeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-exp_$(date +%Y%m%d_%H%M%S)_static_rnn_sampling_matrix}"
CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${RUN_PREFIX}_shared_data}"
LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
CLEAN_LOCAL_CACHE="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
EXPERIMENTS_RAW="${LOWVIS_RNN_SAMPLING_EXPERIMENTS:-0 1}"
EXPERIMENT_LIST="${EXPERIMENTS_RAW//,/ }"
EXPERIMENT_LIST="${EXPERIMENT_LIST//:/ }"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_static_rnn_lowvis_sampling_matrix.slurm}"
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-${BASE}/checkpoints}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
EVAL_SBATCH_SCRIPT="${EVAL_SBATCH_SCRIPT:-${BASE}/paper_eval/sub_static_rnn_lowvis_sampling_eval.slurm}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-static_rnn_sampling_eval_results}"
EVAL_THRESHOLD_SOURCE="${EVAL_THRESHOLD_SOURCE:-argmax}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

case "${RUN_PREFIX}${CACHE_ID}${LOCAL_CACHE_DIR}${CLEAN_LOCAL_CACHE}${CKPT_DIR}${BASE}" in
    *","*|*" "*)
        echo "ERROR: BASE, RUN_PREFIX, CACHE_ID, LOCAL_CACHE_DIR, CLEAN_LOCAL_CACHE, and CKPT_DIR must not contain spaces or commas for sbatch --export." >&2
        exit 2
        ;;
esac

sampling_experiment_name() {
    case "$1" in
        0) echo "natural_shuffle" ;;
        1) echo "current_stratified" ;;
        2) echo "light_lowvis_oversample" ;;
        3) echo "heavy_lowvis_oversample" ;;
        *)
            echo "ERROR: unknown sampling experiment id: $1" >&2
            exit 2
            ;;
    esac
}

join_by_colon() {
    local out=""
    for item in "$@"; do
        if [ -z "${out}" ]; then
            out="${item}"
        else
            out="${out}:${item}"
        fi
    done
    echo "${out}"
}

echo "Submitting Static-RNN sampling matrix as split S1 -> S2 jobs"
echo "BASE=${BASE}"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "CACHE_ID=${CACHE_ID}"
echo "LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR}"
echo "CLEAN_LOCAL_CACHE=${CLEAN_LOCAL_CACHE}"
echo "EXPERIMENTS=${EXPERIMENT_LIST}"
echo "SBATCH_SCRIPT=${SBATCH_SCRIPT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "SUBMIT_EVAL=${SUBMIT_EVAL}"

s2_jobs=()
eval_experiments=()

for exp_id in ${EXPERIMENT_LIST}; do
    exp_name="$(sampling_experiment_name "${exp_id}")"
    run_id="${RUN_PREFIX}_${exp_id}_${exp_name}"
    s1_ckpt="${CKPT_DIR}/${run_id}_S1_best_score.pt"

    s1_job="$(
        sbatch --parsable \
            --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_SAMPLING_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID},LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=${CLEAN_LOCAL_CACHE},LOWVIS_RNN_CKPT_DIR=${CKPT_DIR} \
            "${SBATCH_SCRIPT}"
    )"

    s2_job="$(
        sbatch --parsable \
            --dependency=afterok:${s1_job} \
            --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_SAMPLING_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID},LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=${CLEAN_LOCAL_CACHE},LOWVIS_RNN_CKPT_DIR=${CKPT_DIR},LOWVIS_RNN_PRETRAINED_CKPT=${s1_ckpt} \
            "${SBATCH_SCRIPT}"
    )"

    s2_jobs+=("${s2_job}")
    eval_experiments+=("${exp_id}")
    echo "exp ${exp_id} ${exp_name}: S1 job ${s1_job} -> S2 job ${s2_job}"
    echo "  S1 checkpoint: ${s1_ckpt}"
done

if [ "${SUBMIT_EVAL}" = "1" ]; then
    if [ ! -f "${EVAL_SBATCH_SCRIPT}" ]; then
        echo "ERROR: eval sbatch script not found: ${EVAL_SBATCH_SCRIPT}" >&2
        echo "Set SUBMIT_EVAL=0 to submit training only." >&2
        exit 2
    fi
    dep="$(join_by_colon "${s2_jobs[@]}")"
    eval_exp="$(join_by_colon "${eval_experiments[@]}")"
    eval_job="$(
        sbatch --parsable \
            --dependency=afterok:${dep} \
            --export=ALL,SAMPLING_RUN_PREFIX=${RUN_PREFIX},EXPERIMENTS=${eval_exp},OUT_DIR=${EVAL_OUT_DIR},THRESHOLD_SOURCE=${EVAL_THRESHOLD_SOURCE},DEVICE=${EVAL_DEVICE} \
            "${EVAL_SBATCH_SCRIPT}"
    )"
    echo "sampling eval: afterok:${dep} -> job ${eval_job}"
fi

echo "Submitted sampling matrix. Default comparison is no Low-vis event oversampling vs current oversampling (0 1)."
