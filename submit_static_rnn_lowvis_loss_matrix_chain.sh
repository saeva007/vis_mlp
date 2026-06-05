#!/bin/bash
#
# Submit the loss-function matrix as independent S1 jobs plus dependent S2 jobs.
#
# Each job still uses sub_static_rnn_lowvis_loss_matrix.slurm, but only runs one
# experiment and one stage. This avoids a single long allocation running
# exp0-S1/S2 -> exp1-S1/S2 -> exp2-S1/S2 in sequence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-exp_$(date +%Y%m%d_%H%M%S)_static_rnn_loss_matrix}"
CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${RUN_PREFIX}_shared_data}"
EXPERIMENTS_RAW="${LOWVIS_RNN_EXPERIMENTS:-0 1 2}"
EXPERIMENT_LIST="${EXPERIMENTS_RAW//,/ }"
EXPERIMENT_LIST="${EXPERIMENT_LIST//:/ }"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_static_rnn_lowvis_loss_matrix.slurm}"
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-/public/home/putianshu/vis_mlp/checkpoints}"

case "${RUN_PREFIX}${CACHE_ID}${CKPT_DIR}" in
    *","*|*" "*)
        echo "ERROR: RUN_PREFIX, CACHE_ID, and CKPT_DIR must not contain spaces or commas for sbatch --export." >&2
        exit 2
        ;;
esac

loss_experiment_name() {
    case "$1" in
        0) echo "simple_ce_classification" ;;
        1) echo "simple_logvis_regression" ;;
        2) echo "proposed_rare_event_focal" ;;
        3) echo "plain_focal_loss" ;;
        *)
            echo "ERROR: unknown loss experiment id: $1" >&2
            exit 2
            ;;
    esac
}

echo "Submitting Static-RNN loss matrix as split S1 -> S2 jobs"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "CACHE_ID=${CACHE_ID}"
echo "EXPERIMENTS=${EXPERIMENT_LIST}"
echo "SBATCH_SCRIPT=${SBATCH_SCRIPT}"

for exp_id in ${EXPERIMENT_LIST}; do
    exp_name="$(loss_experiment_name "${exp_id}")"
    run_id="${RUN_PREFIX}_${exp_id}_${exp_name}"
    s1_ckpt="${CKPT_DIR}/${run_id}_S1_best_score.pt"

    s1_job="$(
        sbatch --parsable \
            --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID} \
            "${SBATCH_SCRIPT}"
    )"

    s2_job="$(
        sbatch --parsable \
            --dependency=afterok:${s1_job} \
            --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_EXPERIMENTS=${exp_id},LOWVIS_RNN_RUN_PREFIX=${RUN_PREFIX},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID},LOWVIS_RNN_PRETRAINED_CKPT=${s1_ckpt} \
            "${SBATCH_SCRIPT}"
    )"

    echo "exp ${exp_id} ${exp_name}: S1 job ${s1_job} -> S2 job ${s2_job}"
    echo "  S1 checkpoint: ${s1_ckpt}"
done

echo "Submitted split loss-matrix chain. With 10 nodes and 5 nodes/job, Slurm can run two jobs concurrently."
