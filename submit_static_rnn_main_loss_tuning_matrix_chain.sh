#!/bin/bash
#
# Submit a compact hyperparameter matrix for the main Static MLP + GRU loss.
#
# The script reuses sub_static_rnn_lowvis_loss_matrix.slurm with experiment 2
# (proposed_rare_event_focal), but gives each variant a different run prefix so
# checkpoints never collide. Each variant is submitted as S1 followed by a
# dependent S2 job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE_RUN_PREFIX="${LOWVIS_RNN_TUNE_RUN_PREFIX:-${LOWVIS_RNN_RUN_PREFIX:-exp_$(date +%Y%m%d_%H%M%S)_main_loss_tuning}}"
CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${BASE_RUN_PREFIX}_shared_data}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_static_rnn_lowvis_loss_matrix.slurm}"
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-/public/home/putianshu/vis_mlp/checkpoints}"
THRESHOLD_MODE="${LOWVIS_RNN_THRESHOLD_MODE:-argmax}"
DEFAULT_VARIANTS="v00_base v01_recall_plus v02_ultra_focus v03_precision_guard v04_boundary_soft"
VARIANTS_RAW="${LOWVIS_RNN_TUNE_VARIANTS:-${DEFAULT_VARIANTS}}"
MANIFEST="${LOWVIS_RNN_TUNE_MANIFEST:-logs/${BASE_RUN_PREFIX}_main_loss_tuning_manifest.tsv}"

COMMON_EXTRA_ARGS="${LOWVIS_RNN_TUNE_COMMON_EXTRA_ARGS:-}"
if [ -z "${COMMON_EXTRA_ARGS}" ]; then
    COMMON_EXTRA_ARGS="--threshold-mode ${THRESHOLD_MODE}"
fi

if [ "${VARIANTS_RAW}" = "all" ]; then
    VARIANTS_RAW="${DEFAULT_VARIANTS}"
fi
VARIANT_LIST="${VARIANTS_RAW//,/ }"
VARIANT_LIST="${VARIANT_LIST//:/ }"

case "${BASE_RUN_PREFIX}${CACHE_ID}${CKPT_DIR}" in
    *","*|*" "*)
        echo "ERROR: BASE_RUN_PREFIX, CACHE_ID, and CKPT_DIR must not contain spaces or commas for sbatch --export." >&2
        exit 2
        ;;
esac

variant_label() {
    case "$1" in
        v00_base) echo "current_main_defaults" ;;
        v01_recall_plus) echo "modest_low_vis_recall_boost" ;;
        v02_ultra_focus) echo "stronger_ultra_low_weighting" ;;
        v03_precision_guard) echo "clear_false_alarm_guard" ;;
        v04_boundary_soft) echo "visibility_boundary_soft_weight" ;;
        *)
            echo "ERROR: unknown tuning variant: $1" >&2
            exit 2
            ;;
    esac
}

variant_extra_args() {
    case "$1" in
        v00_base)
            echo ""
            ;;
        v01_recall_plus)
            echo "--class-weight-fog 2.2 --class-weight-mist 2.2 --class-weight-clear 0.8 --focal-gamma-fog 2.7 --focal-gamma-mist 3.2 --focal-gamma-clear 0.5 --alpha-clear-fp 2.0 --alpha-recall-boost 0.30"
            ;;
        v02_ultra_focus)
            echo "--class-weight-fog 2.6 --class-weight-mist 1.8 --class-weight-clear 0.8 --focal-gamma-fog 2.8 --focal-gamma-mist 2.6 --focal-gamma-clear 0.5 --alpha-clear-fp 2.0 --alpha-recall-boost 0.25"
            ;;
        v03_precision_guard)
            echo "--class-weight-fog 2.0 --class-weight-mist 1.6 --class-weight-clear 1.0 --focal-gamma-fog 2.4 --focal-gamma-mist 2.6 --focal-gamma-clear 0.5 --alpha-clear-fp 2.8 --alpha-recall-boost 0.18"
            ;;
        v04_boundary_soft)
            echo "--boundary-weight 0.5 --boundary-fog-sigma 100 --boundary-mist-sigma 200 --alpha-clear-fp 2.0 --alpha-recall-boost 0.20"
            ;;
        *)
            echo "ERROR: unknown tuning variant: $1" >&2
            exit 2
            ;;
    esac
}

printf "variant_id\tvariant_label\trun_prefix\trun_id\textra_args\ts1_job\ts2_job\ts1_checkpoint\ts2_checkpoint\n" > "${MANIFEST}"

echo "Submitting main-loss tuning matrix as split S1 -> S2 jobs"
echo "BASE_RUN_PREFIX=${BASE_RUN_PREFIX}"
echo "CACHE_ID=${CACHE_ID}"
echo "VARIANTS=${VARIANT_LIST}"
echo "SBATCH_SCRIPT=${SBATCH_SCRIPT}"
echo "COMMON_EXTRA_ARGS=${COMMON_EXTRA_ARGS}"
echo "MANIFEST=${MANIFEST}"

for variant_id in ${VARIANT_LIST}; do
    label="$(variant_label "${variant_id}")"
    variant_prefix="${BASE_RUN_PREFIX}_${variant_id}"
    run_id="${variant_prefix}_2_proposed_rare_event_focal"
    s1_ckpt="${CKPT_DIR}/${run_id}_S1_best_score.pt"
    s2_ckpt="${CKPT_DIR}/${run_id}_S2_PhaseB_best_score.pt"
    extra_args="${COMMON_EXTRA_ARGS} $(variant_extra_args "${variant_id}")"
    extra_args="$(echo "${extra_args}" | xargs)"

    s1_job="$(
        LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
        sbatch --parsable \
            --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_EXPERIMENTS=2,LOWVIS_RNN_RUN_PREFIX=${variant_prefix},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID} \
            "${SBATCH_SCRIPT}"
    )"

    s2_job="$(
        LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
        sbatch --parsable \
            --dependency=afterok:${s1_job} \
            --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_EXPERIMENTS=2,LOWVIS_RNN_RUN_PREFIX=${variant_prefix},LOWVIS_RNN_LOCAL_CACHE_ID=${CACHE_ID},LOWVIS_RNN_PRETRAINED_CKPT=${s1_ckpt} \
            "${SBATCH_SCRIPT}"
    )"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${variant_id}" "${label}" "${variant_prefix}" "${run_id}" "${extra_args}" \
        "${s1_job}" "${s2_job}" "${s1_ckpt}" "${s2_ckpt}" >> "${MANIFEST}"

    echo "${variant_id} ${label}: S1 job ${s1_job} -> S2 job ${s2_job}"
    echo "  RUN_ID: ${run_id}"
    echo "  EXTRA_ARGS: ${extra_args}"
    echo "  S1 checkpoint: ${s1_ckpt}"
done

echo "Submitted main-loss tuning matrix."
echo "With 10 nodes and 5 nodes/job, Slurm can run two jobs concurrently."
echo "After all S2 jobs finish, summarize validation metrics with:"
echo "  python summarize_static_rnn_main_loss_tuning.py --manifest ${MANIFEST}"
