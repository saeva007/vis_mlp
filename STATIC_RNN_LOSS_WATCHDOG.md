# Static-RNN loss-chain watchdog

This watchdog runs as one small job in the CPU partition and manages only the
candidate/seed rows explicitly allowlisted from one or more precision-loss
manifests. It never scans or modifies unrelated jobs owned by the same user.

## Diagnosis of the 17–18 July 2026 stalls

The two pasted logs represent different failure stages:

- P13 seed 314 is an S2 job on `e20r4n15-19`. The process group was created,
  but no data-copy, model, or training marker followed. This is consistent
  with a hang in the first RCCL collective. Recovery should reuse the completed
  S1 checkpoint and retry S2 only.
- P13 seed 42 is an S1 job on `e15r3n02-06`. It reached
  `step=2600/15000` and then stopped making semantic progress. Its original
  dependent S2 must be cancelled together with S1, followed by a new S1 -> S2
  chain.
- Seed 42 also reported insufficient `/tmp` space and fell back to NFS. This
  can make memory-mapped training I/O slow or unstable. It is a separate risk
  from the RCCL startup hang.

The NUMA and missing `iommu=pt` RCCL warnings appear in both logs. They are
cluster configuration warnings, not standalone watchdog triggers. Send the
affected JobIDs and NodeLists to the cluster administrator for a node-health,
IB/RCCL, NUMA-balancing, and IOMMU review.

## Safety model

The watcher records semantic progress rather than arbitrary stdout/stderr
growth. Trusted markers include the RCCL data-copy barrier, individual copy
completion, dataset/scaler/model initialization, the latest training step,
explicit validation start and score, and latest/best checkpoint updates.

Only `RUNNING` jobs are subject to silence timeouts. The conservative defaults
are 120 min for startup, 180 min for data work, 180 min for training, and 360
min for validation, with two consecutive stale checks five minutes apart.
`PENDING`, `CONFIGURING`, `COMPLETING`, query errors, and accounting lag never
trigger cancellation.

An automatic recovery follows this transaction:

1. observe the same semantic-progress token beyond the phase limit;
2. wait 15 seconds and confirm that authoritative `squeue` state is still
   `RUNNING` with the same token;
3. immediately before cancellation, make the same authoritative check again;
4. for S1, verify that the dependent S2 is still waiting on the exact
   `afterok:<S1 JobID>` dependency;
5. cancel the affected stage/chain;
6. wait until `squeue` has lost the jobs and `sacct` reports every cancelled
   job as `CANCELLED`;
7. submit the replacement with a new run prefix, cache ID, job name, and Slurm
   comment.

Any state change, Slurm query failure, unexpected dependency, or terminal state
stops the transaction before `sbatch`. Automatic recovery is deliberately
limited to confirmed stale `RUNNING` jobs. `NODE_FAIL`, `FAILED`, `TIMEOUT`,
OOM, and manual cancellation are reported for manual handling, so normal
RCCL/NCCL version and configuration warnings can never cause a retry.

A completed stage is accepted only after its expected checkpoint exists, is
non-empty, and passes a 30-minute filesystem grace period. Automatic retry is
capped at two attempts.

## Repair and monitor the current P13 jobs

Run this from the remote repository root after syncing the updated watcher and
training scripts. Do not manually cancel the stalled jobs first; the watchdog needs
their live JobIDs, NodeLists, and dependencies.

```bash
set -euo pipefail

export REPO=/public/home/putianshu/vis_mlp/train
cd ${REPO}

shopt -s nullglob
MANIFEST_POOL=(logs/*_precision_loss_manifest.tsv)

unique_manifest_for_row() {
  local candidate="$1" seed="$2" run_pattern="$3"
  local file
  local -a hits=()
  for file in "${MANIFEST_POOL[@]}"; do
    if awk -F'\t' -v candidate="${candidate}" -v seed="${seed}" -v pattern="${run_pattern}" \
      'NR>1 && $1==candidate && $5==seed && $6=="full" && $7 ~ pattern {found=1} END {exit !found}' \
      "${file}"; then
      hits+=("${file}")
    fi
  done
  if [ "${#hits[@]}" -ne 1 ]; then
    printf 'Expected exactly one manifest for %s:%s; found %s\n' \
      "${candidate}" "${seed}" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]}" >&2
    return 2
  fi
  printf '%s\n' "${hits[0]}"
}

M42=$(unique_manifest_for_row p13 42 '^exp_20260717_203305_p13_sampling_calibration_full_p13_seed42$')
M314=$(unique_manifest_for_row p13 314 '^exp_20260717_203259_p13_sampling_calibration_full_p13_seed314$')
M2718=$(unique_manifest_for_row p13 2718 '_p13_sampling_calibration_full_p13_seed2718$')
mapfile -t WATCH_FILES < <(printf '%s\n' "${M42}" "${M314}" "${M2718}" | sort -u)

printf 'Candidate manifests: %s\n' "${WATCH_FILES[@]}"
awk -F'\t' 'FNR==1 {next} $1=="p13" {
  print "candidate=" $1, "seed=" $5, "stage=" $6,
        "run_id=" $8, "s1_job=" $10, "s2_job=" $11
}' "${WATCH_FILES[@]}"

export WATCH_MANIFESTS=$(IFS=:; echo "${WATCH_FILES[*]}")
export WATCH_ONLY_ROWS='p13:42;p13:314;p13:2718'
export WATCH_COMBINED_RESOLVED_MANIFEST=${REPO}/logs/p13_three_seed_watchdog_resolved.tsv
export WATCH_RETRY_EXPORTS='LOWVIS_RNN_S1_STEPS=15000;LOWVIS_RNN_S2_A_STEPS=8000;LOWVIS_RNN_S2_B_STEPS=22000;LOWVIS_RNN_VAL_INTERVAL=500;LOWVIS_RNN_BATCH_SIZE=512;LOWVIS_RNN_GRAD_ACCUM=2;LOWVIS_RNN_NUM_WORKERS=0;LOWVIS_RNN_PATIENCE=10'

WATCH_JOB=$(sbatch --parsable \
  --export=ALL,WATCH_MANIFESTS=${WATCH_MANIFESTS},WATCH_ONLY_ROWS=${WATCH_ONLY_ROWS},WATCH_RETRY_EXPORTS=${WATCH_RETRY_EXPORTS},WATCH_COMBINED_RESOLVED_MANIFEST=${WATCH_COMBINED_RESOLVED_MANIFEST},WATCH_AUTO_RETRY=1,WATCH_POLL_SECONDS=300,WATCH_STARTUP_STALE_MINUTES=120,WATCH_DATA_STALE_MINUTES=180,WATCH_TRAIN_STALE_MINUTES=180,WATCH_VALIDATION_STALE_MINUTES=360,WATCH_CONFIRMATIONS=2,WATCH_MAX_RETRIES=2,WATCH_EXCLUDE_FAILED_NODES=1 \
  sub_static_rnn_loss_watchdog.slurm)

echo "WATCH_JOB=${WATCH_JOB}"
echo "RESOLVED_MANIFEST=${WATCH_COMBINED_RESOLVED_MANIFEST}"
echo "Watch with: tail -f logs/${WATCH_JOB}_lowvis_loss_watch.out"
```

The command requires exactly one matching manifest for each logical row and
fails before any cancellation if a row is missing or historically duplicated.
If the seed-2718 lookup reports multiple files, replace its suffix pattern with
the exact run prefix printed by the intended full-training manifest. If the
third seed differs from `2718`, update both that lookup and `WATCH_ONLY_ROWS`.
P14/P15 rows in a matched manifest remain untouched and are omitted from the
combined resolved manifest.

The explicit retry recipe matches the current loss wrapper defaults; the
seed-42 log independently confirms 15,000 S1 steps. If the original launch
overrode a listed value, replace it with that original value before enabling
automatic retry. Newly generated manifests record this recipe in a
`retry_exports` column, so future watchdog runs do not need
`WATCH_RETRY_EXPORTS`.

The original manifest is never edited. Each input gets an adjacent
`*_resolved.tsv`, `*_watchdog_state.json`, append-only
`*_watchdog_actions.tsv`, and a single-instance lock. The combined resolved
manifest is the three-row file to use for validation, mean-probability/argmax,
and final evaluation after all members complete.

If either stalled job was already cancelled manually, this automatic command
will stop without reviving it. Inspect its checkpoint and dependency state,
then use a candidate-specific manual recovery command; a user cancellation is
intentionally distinguishable from a watchdog-confirmed cancellation.

## Monitor-only dry run

Report-only is the default and is the recommended first run for a new
experiment family:

```bash
: "${FULL_MANIFEST:?export FULL_MANIFEST=/absolute/path/to/the/exact_manifest.tsv}"
test -f "${FULL_MANIFEST}"
WATCH_JOB=$(sbatch --parsable \
  --export=ALL,WATCH_MANIFEST=${FULL_MANIFEST},WATCH_AUTO_RETRY=0 \
  sub_static_rnn_loss_watchdog.slurm)
echo "WATCH_JOB=${WATCH_JOB}"
```

Inspect `logs/${WATCH_JOB}_lowvis_loss_watch.out` and the generated action TSV.
Stopping the watchdog does not cancel its managed training jobs:

```bash
scancel ${WATCH_JOB}
```

Before enabling `WATCH_AUTO_RETRY=1`, set an explicit `WATCH_ONLY_ROWS`
allowlist and provide an exact retry recipe when using a legacy manifest.

## Cache and node handling

The loss wrapper now prints disk capacity and inode usage on every allocated
node. Each retry writes into a unique
`<WATCH_LOCAL_CACHE_DIR>/<retry_cache_id>` directory. The wrapper removes that
verified, user-owned scoped directory at job exit; if cancellation prevents an
exit-time cleanup step, the next exclusive retry allocation removes older
user-owned watchdog cache directories under `/tmp`. It also removes only
user-owned legacy Static-RNN `X/y_*.npy` files at the top level of `/tmp`.
These checks address the seed-42 space failure without touching shared datasets
or other users' files.

If the cluster provides an administrator-approved node-local SSD/scratch path,
set it as `WATCH_LOCAL_CACHE_DIR`. Do not point this setting at a shared dataset
directory.

Failed NodeLists accumulate only within one candidate/seed's two-attempt
budget. S1 recovery creates a new S1 checkpoint and submits S2 with
`afterok:<new S1 JobID>`; S2 recovery reuses the completed S1 checkpoint. The
new checkpoint paths are derived from the original manifest rather than from a
hard-coded candidate phase.

If retry limits, terminal failures, query uncertainty, dependency mismatch, or
artifact checks block recovery, the watchdog records the reason and exits
without guessing. An in-flight intent plus the unique Slurm comment protects
against duplicate submission after a process crash. Inspect the recorded
intent and the matching job comment in `squeue`/`sacct` before any manual
takeover; do not simply delete watchdog state.
