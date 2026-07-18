#!/bin/bash
# Submit the pre-registered low-false-alarm Static-RNN loss experiment.
# This workflow creates candidate checkpoints only. It never changes the
# current mainline run id, paper-eval config, or deployment configuration.
#
# Stages:
#   screen: reuse one S1 checkpoint and run equal-step S2 jobs.
#   full:   train each selected candidate as an independent S1 -> S2 chain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE_PREFIX="${LOWVIS_RNN_PRECISION_RUN_PREFIX:-exp_$(date +%Y%m%d_%H%M%S)_precision_loss}"
STAGE="${LOWVIS_RNN_PRECISION_STAGE:-screen}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_static_rnn_lowvis_loss_matrix.slurm}"
CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-/public/home/putianshu/vis_mlp/checkpoints}"
CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${BASE_PREFIX}_shared_data}"
PRETRAINED_S1="${LOWVIS_RNN_PRETRAINED_CKPT:-}"
SEEDS_RAW="${LOWVIS_RNN_PRECISION_SEEDS:-42}"
DEFAULT_CANDIDATES="p0 p1 p2 p3"
CANDIDATES_RAW="${LOWVIS_RNN_PRECISION_CANDIDATES:-${DEFAULT_CANDIDATES}}"
ENABLE_NARROW_SOFT="${LOWVIS_RNN_ENABLE_NARROW_SOFT:-0}"
MANIFEST="${LOWVIS_RNN_PRECISION_MANIFEST:-logs/${BASE_PREFIX}_precision_loss_manifest.tsv}"
COMMON_ARGS="${LOWVIS_RNN_PRECISION_COMMON_ARGS:---threshold-mode argmax}"
RETRY_EXPORTS="LOWVIS_RNN_S1_STEPS=${LOWVIS_RNN_S1_STEPS:-15000};LOWVIS_RNN_S2_A_STEPS=${LOWVIS_RNN_S2_A_STEPS:-8000};LOWVIS_RNN_S2_B_STEPS=${LOWVIS_RNN_S2_B_STEPS:-22000};LOWVIS_RNN_VAL_INTERVAL=${LOWVIS_RNN_VAL_INTERVAL:-500};LOWVIS_RNN_BATCH_SIZE=${LOWVIS_RNN_BATCH_SIZE:-512};LOWVIS_RNN_GRAD_ACCUM=${LOWVIS_RNN_GRAD_ACCUM:-2};LOWVIS_RNN_NUM_WORKERS=${LOWVIS_RNN_NUM_WORKERS:-0};LOWVIS_RNN_PATIENCE=${LOWVIS_RNN_PATIENCE:-10}"

if [[ "${STAGE}" != "screen" && "${STAGE}" != "full" ]]; then
    echo "ERROR: LOWVIS_RNN_PRECISION_STAGE must be screen or full." >&2
    exit 2
fi
if [[ "${STAGE}" == "screen" && -z "${PRETRAINED_S1}" ]]; then
    echo "ERROR: screen stage requires LOWVIS_RNN_PRETRAINED_CKPT=<shared S1 checkpoint>." >&2
    exit 2
fi
if [[ "${STAGE}" == "screen" && ! -f "${PRETRAINED_S1}" ]]; then
    echo "ERROR: screen stage S1 checkpoint not found: ${PRETRAINED_S1}" >&2
    echo "Set MAIN_RUN_ID to a real run id before deriving LOWVIS_RNN_PRETRAINED_CKPT." >&2
    exit 2
fi

normalize_list() {
    local value="$1"
    value="${value//,/ }"
    value="${value//:/ }"
    echo "${value}"
}

candidate_label() {
    case "$1" in
        p0) echo "current_loss_reproduction" ;;
        p1) echo "hard_clear_lite" ;;
        p2) echo "conditional_rate_guard" ;;
        p3) echo "combined_lite" ;;
        p4) echo "combined_lite_narrow_soft" ;;
        p5) echo "conditional_recall_guard_030" ;;
        p6) echo "conditional_recall_guard_050" ;;
        p7) echo "conditional_balanced_recall_125_050" ;;
        p8) echo "clear_ultra_pair_lite" ;;
        p9) echo "clear_ultra_pair_balanced" ;;
        p10) echo "clear_ultra_pair_strong" ;;
        p11) echo "p3_event_footprint_beta050" ;;
        p12) echo "p3_event_footprint_beta075" ;;
        p13) echo "p3_sampling_calibration_eta030" ;;
        p14) echo "p3_sampling_calibration_eta045" ;;
        p15) echo "p3_sampling_calibration_eta060" ;;
        *) echo "ERROR: unknown precision-loss candidate: $1" >&2; exit 2 ;;
    esac
}

candidate_args() {
    case "$1" in
        p0)
            echo "--loss-mode designed_focal"
            ;;
        p1)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25"
            ;;
        p2)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10"
            ;;
        p3)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25"
            ;;
        p4)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --soft-mist-clear-low 850 --soft-mist-clear-high 1050"
            ;;
        p5)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --event-loss-normalization conditional --event-fp-weight 1.0 --event-fn-weight 0.30"
            ;;
        p6)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --event-loss-normalization conditional --event-fp-weight 1.0 --event-fn-weight 0.50"
            ;;
        p7)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --event-loss-normalization conditional --event-fp-weight 1.25 --event-fn-weight 0.50"
            ;;
        p8)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --selection-metric csi --clear-pair-vis-min 3000 --clear-to-fog-weight 0.5 --clear-to-mist-weight 0.10 --moderate-fn-weight 0.20"
            ;;
        p9)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --selection-metric csi --clear-pair-vis-min 3000 --clear-to-fog-weight 1.0 --clear-to-mist-weight 0.10 --moderate-fn-weight 0.25"
            ;;
        p10)
            echo "--loss-mode designed_focal --focal-gamma-clear 0.5 --selection-metric csi --clear-pair-vis-min 3000 --clear-to-fog-weight 1.5 --clear-to-mist-weight 0.15 --moderate-fn-weight 0.35"
            ;;
        p11)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --s2-phase-c-steps 5000 --s2-lr-head-c 2e-5 --phase-c-prior-beta 0.50 --event-footprint-csi-weight 0.50 --event-footprint-area-ratio-cap 1.50 --event-footprint-area-slack 0.005 --event-footprint-min-recall 0.50 --event-footprint-min-fog-count 40 --event-footprint-event-batch-ratio 0.50 --event-footprint-dual-init 1.0 --event-footprint-dual-rho 5.0 --event-footprint-dual-lr 0.05 --event-footprint-dual-max 20.0 --phase-c-selection-metric footprint_csi --phase-c-min-low-vis-recall 0.55 --phase-c-max-fpr 0.03 --phase-c-min-event-mean-recall 0.55 --phase-c-min-event-recall 0.40 --phase-c-max-event-area-ratio-mean 1.80 --phase-c-max-event-area-ratio 2.20"
            ;;
        p12)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --s2-phase-c-steps 5000 --s2-lr-head-c 2e-5 --phase-c-prior-beta 0.75 --event-footprint-csi-weight 0.50 --event-footprint-area-ratio-cap 1.50 --event-footprint-area-slack 0.005 --event-footprint-min-recall 0.50 --event-footprint-min-fog-count 40 --event-footprint-event-batch-ratio 0.50 --event-footprint-dual-init 1.0 --event-footprint-dual-rho 5.0 --event-footprint-dual-lr 0.05 --event-footprint-dual-max 20.0 --phase-c-selection-metric footprint_csi --phase-c-min-low-vis-recall 0.55 --phase-c-max-fpr 0.03 --phase-c-min-event-mean-recall 0.55 --phase-c-min-event-recall 0.40 --phase-c-max-event-area-ratio-mean 1.80 --phase-c-max-event-area-ratio 2.20"
            ;;
        p13)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --s2-phase-d-steps 3000 --s2-lr-head-d 2e-5 --phase-d-natural-mix 0.30 --phase-d-ramp-start 0.50 --phase-d-event-batch-ratio 0.50 --phase-d-min-fog-count 40 --phase-d-selection-metric sampling_csi --phase-d-min-low-vis-csi 0.195 --phase-d-max-fpr 0.025 --phase-d-min-event-mean-csi 0.235 --phase-d-min-event-mean-recall 0.45 --phase-d-min-event-recall 0.20 --phase-d-min-event-area-ratio-mean 0.80 --phase-d-max-event-area-ratio-mean 1.80 --phase-d-max-event-area-ratio 2.20"
            ;;
        p14)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --s2-phase-d-steps 3000 --s2-lr-head-d 2e-5 --phase-d-natural-mix 0.45 --phase-d-ramp-start 0.50 --phase-d-event-batch-ratio 0.50 --phase-d-min-fog-count 40 --phase-d-selection-metric sampling_csi --phase-d-min-low-vis-csi 0.195 --phase-d-max-fpr 0.025 --phase-d-min-event-mean-csi 0.235 --phase-d-min-event-mean-recall 0.45 --phase-d-min-event-recall 0.20 --phase-d-min-event-area-ratio-mean 0.80 --phase-d-max-event-area-ratio-mean 1.80 --phase-d-max-event-area-ratio 2.20"
            ;;
        p15)
            echo "--loss-mode designed_focal --focal-gamma-clear 1.0 --event-loss-normalization conditional --event-fp-weight 1.5 --event-fn-weight 0.10 --physical-hard-weight 0.5 --aerosol-hard-weight 0.25 --s2-phase-d-steps 3000 --s2-lr-head-d 2e-5 --phase-d-natural-mix 0.60 --phase-d-ramp-start 0.50 --phase-d-event-batch-ratio 0.50 --phase-d-min-fog-count 40 --phase-d-selection-metric sampling_csi --phase-d-min-low-vis-csi 0.195 --phase-d-max-fpr 0.025 --phase-d-min-event-mean-csi 0.235 --phase-d-min-event-mean-recall 0.45 --phase-d-min-event-recall 0.20 --phase-d-min-event-area-ratio-mean 0.80 --phase-d-max-event-area-ratio-mean 1.80 --phase-d-max-event-area-ratio 2.20"
            ;;
    esac
}

CANDIDATES="$(normalize_list "${CANDIDATES_RAW}")"
SEEDS="$(normalize_list "${SEEDS_RAW}")"
if [[ "${ENABLE_NARROW_SOFT}" == "1" && ! " ${CANDIDATES} " =~ " p4 " ]]; then
    CANDIDATES="${CANDIDATES} p4"
fi
if [[ " ${CANDIDATES} " =~ " p4 " && "${ENABLE_NARROW_SOFT}" != "1" ]]; then
    echo "ERROR: p4 is conditional. Set LOWVIS_RNN_ENABLE_NARROW_SOFT=1 only after the >=35% diagnostic fires." >&2
    exit 2
fi

printf "candidate_id\tcandidate_label\texperiment_status\treplaces_mainline\tseed\tstage\trun_prefix\trun_id\textra_args\ts1_job\ts2_job\ts1_checkpoint\ts2_checkpoint\tretry_exports\n" > "${MANIFEST}"

echo "Submitting precision-loss candidates"
echo "STAGE=${STAGE} CANDIDATES=${CANDIDATES} SEEDS=${SEEDS}"
echo "MANIFEST=${MANIFEST}"

for seed in ${SEEDS}; do
    for candidate_id in ${CANDIDATES}; do
        label="$(candidate_label "${candidate_id}")"
        candidate_prefix="${BASE_PREFIX}_${candidate_id}_seed${seed}"
        run_id="${candidate_prefix}_2_proposed_rare_event_focal"
        s1_ckpt="${CKPT_DIR}/${run_id}_S1_best_score.pt"
        s2_tag="S2_PhaseB"
        if [[ "${candidate_id}" = "p11" || "${candidate_id}" = "p12" ]]; then
            s2_tag="S2_PhaseC"
        elif [[ "${candidate_id}" = "p13" || "${candidate_id}" = "p14" || "${candidate_id}" = "p15" ]]; then
            s2_tag="S2_PhaseD"
        fi
        s2_ckpt="${CKPT_DIR}/${run_id}_${s2_tag}_best_score.pt"
        extra_args="${COMMON_ARGS} --seed ${seed} $(candidate_args "${candidate_id}")"
        extra_args="$(echo "${extra_args}" | xargs)"
        s1_job=""
        candidate_cache_id="${CACHE_ID}"
        if [[ "${candidate_id}" = "p13" || "${candidate_id}" = "p14" || "${candidate_id}" = "p15" ]]; then
            candidate_cache_id="${CACHE_ID}_${candidate_id}_seed${seed}"
        fi

        if [[ "${STAGE}" == "full" ]]; then
            s1_job="$(
                LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                sbatch --parsable \
                    --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_EXPERIMENTS=2,LOWVIS_RNN_RUN_PREFIX=${candidate_prefix},LOWVIS_RNN_LOCAL_CACHE_ID=${candidate_cache_id} \
                    "${SBATCH_SCRIPT}"
            )"
            s2_job="$(
                LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                sbatch --parsable \
                    --dependency=afterok:${s1_job} \
                    --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_EXPERIMENTS=2,LOWVIS_RNN_RUN_PREFIX=${candidate_prefix},LOWVIS_RNN_LOCAL_CACHE_ID=${candidate_cache_id},LOWVIS_RNN_PRETRAINED_CKPT=${s1_ckpt} \
                    "${SBATCH_SCRIPT}"
            )"
        else
            s1_ckpt="${PRETRAINED_S1}"
            s2_job="$(
                LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                sbatch --parsable \
                    --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_EXPERIMENTS=2,LOWVIS_RNN_RUN_PREFIX=${candidate_prefix},LOWVIS_RNN_LOCAL_CACHE_ID=${candidate_cache_id},LOWVIS_RNN_PRETRAINED_CKPT=${PRETRAINED_S1} \
                    "${SBATCH_SCRIPT}"
            )"
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${candidate_id}" "${label}" "candidate_only" "false" "${seed}" "${STAGE}" "${candidate_prefix}" "${run_id}" \
            "${extra_args}" "${s1_job}" "${s2_job}" "${s1_ckpt}" "${s2_ckpt}" "${RETRY_EXPORTS}" >> "${MANIFEST}"
        echo "${candidate_id} seed=${seed}: ${s1_job:-shared-S1} -> ${s2_job}"
    done
done

echo "Submitted candidate-only precision-loss experiment."
echo "No mainline run id, paper-eval config, or deployment setting was changed."
