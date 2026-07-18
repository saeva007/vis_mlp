#!/bin/bash
# Cancel only the active 17 July P13 full-training chains recorded in manifests,
# then submit a fresh three-seed P13 run and its CPU watchdog.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

OLD_RUN_PATTERN="${OLD_RUN_PATTERN:-^exp_20260717_[0-9]+_p13_sampling_calibration_full_p13_seed(42|314|2718)$}"
CONFIRM_CANCEL="${CONFIRM_CANCEL:-NO}"

shopt -s nullglob
MANIFEST_POOL=(logs/*_precision_loss_manifest.tsv)
if [ "${#MANIFEST_POOL[@]}" -eq 0 ]; then
    echo "ERROR: no precision-loss manifests found under ${SCRIPT_DIR}/logs." >&2
    exit 2
fi

mapfile -t OLD_ROWS < <(
    awk -F'\t' -v pattern="${OLD_RUN_PATTERN}" '
        FNR > 1 && $1 == "p13" && ($5 == "42" || $5 == "314" || $5 == "2718") &&
        $6 == "full" && $7 ~ pattern {
            print $5 "\t" $7 "\t" $10 "\t" $11
        }
    ' "${MANIFEST_POOL[@]}" | sort -u
)

if [ "${#OLD_ROWS[@]}" -eq 0 ]; then
    echo "ERROR: no old P13 rows matched OLD_RUN_PATTERN=${OLD_RUN_PATTERN}" >&2
    echo "Set OLD_RUN_PATTERN to the exact old run-prefix pattern and retry." >&2
    exit 2
fi

declare -A ACTIVE_JOB_IDS=()
while IFS= read -r job_id; do
    job_id="${job_id%%;*}"
    if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        ACTIVE_JOB_IDS["${job_id}"]=1
    fi
done < <(squeue -h -u "${USER}" -o "%i")

declare -A CANCEL_JOB_IDS=()
declare -A ACTIVE_PREFIX_BY_SEED=()

echo "Matched old manifest rows:"
for row in "${OLD_ROWS[@]}"; do
    IFS=$'\t' read -r seed run_prefix s1_job s2_job <<< "${row}"
    printf '  seed=%s run_prefix=%s s1=%s s2=%s\n' \
        "${seed}" "${run_prefix}" "${s1_job:-none}" "${s2_job:-none}"

    row_is_active=0
    for raw_job_id in "${s1_job:-}" "${s2_job:-}"; do
        job_id="${raw_job_id%%;*}"
        if [[ "${job_id}" =~ ^[0-9]+$ ]] && [ "${ACTIVE_JOB_IDS[${job_id}]:-0}" = "1" ]; then
            CANCEL_JOB_IDS["${job_id}"]=1
            row_is_active=1
        fi
    done

    if [ "${row_is_active}" -eq 1 ]; then
        if [ -n "${ACTIVE_PREFIX_BY_SEED[${seed}]:-}" ] && \
           [ "${ACTIVE_PREFIX_BY_SEED[${seed}]}" != "${run_prefix}" ]; then
            echo "ERROR: seed ${seed} has more than one active matched run prefix:" >&2
            echo "  ${ACTIVE_PREFIX_BY_SEED[${seed}]}" >&2
            echo "  ${run_prefix}" >&2
            echo "Narrow OLD_RUN_PATTERN before allowing cancellation." >&2
            exit 2
        fi
        ACTIVE_PREFIX_BY_SEED["${seed}"]="${run_prefix}"
    fi
done

mapfile -t CANCEL_LIST < <(printf '%s\n' "${!CANCEL_JOB_IDS[@]}" | sed '/^$/d' | sort -n)
if [ "${#CANCEL_LIST[@]}" -gt 0 ]; then
    CANCEL_CSV="$(IFS=,; echo "${CANCEL_LIST[*]}")"
    echo
    echo "Active old-chain jobs selected for exact cancellation: ${CANCEL_CSV}"
    squeue -h -j "${CANCEL_CSV}" \
        -o '  job=%i state=%T node=%N name=%j dependency=%E' || true
else
    echo
    echo "No still-active JobID from the matched old manifests was found."
fi

if [ "${CONFIRM_CANCEL}" != "YES" ]; then
    echo
    echo "DRY RUN ONLY: no job was cancelled and no new job was submitted."
    echo "After checking the rows and JobIDs above, run:"
    echo "  CONFIRM_CANCEL=YES bash ${SCRIPT_DIR}/restart_p13_three_seed_with_watchdog.sh"
    exit 0
fi

if [ "${#CANCEL_LIST[@]}" -gt 0 ]; then
    # Refresh immediately before the destructive action and keep only exact IDs
    # that are still present in the user's queue.
    declare -A STILL_ACTIVE=()
    while IFS= read -r job_id; do
        job_id="${job_id%%;*}"
        if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
            STILL_ACTIVE["${job_id}"]=1
        fi
    done < <(squeue -h -u "${USER}" -o "%i")

    CANCEL_NOW=()
    for job_id in "${CANCEL_LIST[@]}"; do
        if [ "${STILL_ACTIVE[${job_id}]:-0}" = "1" ]; then
            CANCEL_NOW+=("${job_id}")
        fi
    done

    if [ "${#CANCEL_NOW[@]}" -gt 0 ]; then
        echo "Cancelling exact old-chain JobIDs: ${CANCEL_NOW[*]}"
        scancel "${CANCEL_NOW[@]}"

        for _ in $(seq 1 24); do
            declare -A QUEUED_NOW=()
            while IFS= read -r job_id; do
                job_id="${job_id%%;*}"
                if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
                    QUEUED_NOW["${job_id}"]=1
                fi
            done < <(squeue -h -u "${USER}" -o "%i")

            remaining=0
            for job_id in "${CANCEL_NOW[@]}"; do
                if [ "${QUEUED_NOW[${job_id}]:-0}" = "1" ]; then
                    remaining=1
                    break
                fi
            done
            if [ "${remaining}" -eq 0 ]; then
                break
            fi
            sleep 5
        done

        if [ "${remaining:-0}" -ne 0 ]; then
            echo "ERROR: an old-chain job is still present after 120 seconds; refusing new submission." >&2
            exit 2
        fi
    fi
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
FULL_PREFIX="${NEW_FULL_PREFIX:-exp_${RUN_STAMP}_p13_sampling_calibration_full}"
FULL_MANIFEST="${NEW_FULL_MANIFEST:-${SCRIPT_DIR}/logs/${FULL_PREFIX}_precision_loss_manifest.tsv}"
RESOLVED_MANIFEST="${NEW_RESOLVED_MANIFEST:-${FULL_MANIFEST%.tsv}_resolved.tsv}"

if [ -e "${FULL_MANIFEST}" ] || [ -e "${RESOLVED_MANIFEST}" ]; then
    echo "ERROR: refusing to overwrite an existing new-run manifest:" >&2
    echo "  ${FULL_MANIFEST}" >&2
    echo "  ${RESOLVED_MANIFEST}" >&2
    exit 2
fi

export LOWVIS_RNN_PRECISION_RUN_PREFIX="${FULL_PREFIX}"
export LOWVIS_RNN_PRECISION_STAGE=full
export LOWVIS_RNN_PRECISION_CANDIDATES=p13
export LOWVIS_RNN_PRECISION_SEEDS=42:314:2718
export LOWVIS_RNN_PRECISION_MANIFEST="${FULL_MANIFEST}"
export LOWVIS_RNN_LOCAL_CACHE_ID="${FULL_PREFIX}_shared_data"
export LOWVIS_RNN_LOCAL_CACHE_DIR=/tmp
export LOWVIS_RNN_CACHE_BASE_DIR=/tmp
export LOWVIS_RNN_CLEAN_LOCAL_CACHE=0
export LOWVIS_RNN_CLEAN_LEGACY_CACHE=0
export LOWVIS_RNN_PRECISION_COMMON_ARGS='--threshold-mode argmax'
export LOWVIS_RNN_S1_STEPS="${LOWVIS_RNN_S1_STEPS:-15000}"
export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-8000}"
export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-22000}"
export LOWVIS_RNN_VAL_INTERVAL="${LOWVIS_RNN_VAL_INTERVAL:-500}"
export LOWVIS_RNN_BATCH_SIZE="${LOWVIS_RNN_BATCH_SIZE:-512}"
export LOWVIS_RNN_GRAD_ACCUM="${LOWVIS_RNN_GRAD_ACCUM:-2}"
export LOWVIS_RNN_NUM_WORKERS="${LOWVIS_RNN_NUM_WORKERS:-0}"
export LOWVIS_RNN_PATIENCE="${LOWVIS_RNN_PATIENCE:-10}"

echo
echo "Submitting fresh P13 full-training chains for seeds 42, 314, and 2718."
bash submit_static_rnn_precision_loss_candidates_chain.sh
test -s "${FULL_MANIFEST}"

WATCH_ONLY_ROWS='p13:42;p13:314;p13:2718'
WATCH_SUBMISSION="$(
    sbatch --parsable \
        --export=ALL,WATCH_MANIFEST=${FULL_MANIFEST},WATCH_RESOLVED_MANIFEST=${RESOLVED_MANIFEST},WATCH_ONLY_ROWS=${WATCH_ONLY_ROWS},WATCH_AUTO_RETRY=1,WATCH_POLL_SECONDS=120,WATCH_STARTUP_STALE_MINUTES=30,WATCH_DATA_STALE_MINUTES=45,WATCH_TRAIN_STALE_MINUTES=30,WATCH_VALIDATION_STALE_MINUTES=60,WATCH_CONFIRMATIONS=2,WATCH_MAX_RETRIES=2,WATCH_EXCLUDE_FAILED_NODES=1 \
        sub_static_rnn_loss_watchdog.slurm
)"
WATCH_JOB="${WATCH_SUBMISSION%%;*}"

STATE_FILE="${SCRIPT_DIR}/logs/${FULL_PREFIX}_watchdog_run.env"
printf 'FULL_PREFIX=%q\nFULL_MANIFEST=%q\nRESOLVED_MANIFEST=%q\nWATCH_JOB=%q\n' \
    "${FULL_PREFIX}" "${FULL_MANIFEST}" "${RESOLVED_MANIFEST}" "${WATCH_JOB}" > "${STATE_FILE}"

echo
echo "Fresh training and CPU watchdog submitted successfully."
echo "FULL_PREFIX=${FULL_PREFIX}"
echo "FULL_MANIFEST=${FULL_MANIFEST}"
echo "RESOLVED_MANIFEST=${RESOLVED_MANIFEST}"
echo "WATCH_JOB=${WATCH_JOB}"
echo "STATE_FILE=${STATE_FILE}"
echo "Monitor: tail -f ${SCRIPT_DIR}/logs/${WATCH_JOB}_lowvis_loss_watch.out"
echo "Queue:   squeue -j ${WATCH_JOB}"
echo "IMPORTANT: after any automatic retry, use RESOLVED_MANIFEST for validation and plotting."
