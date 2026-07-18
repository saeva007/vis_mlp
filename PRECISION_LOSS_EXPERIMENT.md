# Static-RNN Low-False-Alarm Candidate Experiment

## Status boundary

This is an isolated candidate experiment. It does **not** replace the current
`Static MLP + GRU` mainline, does not edit `paper_eval_config.json`, and does
not change any deployment configuration. Candidate checkpoints use independent
run prefixes and may be promoted only through a separate, explicit decision
after validation and frozen-test acceptance.

The default behavior of `train_static_rnn_lowvis.py` remains the historical
mainline behavior:

- `event_loss_normalization=legacy_batch`;
- Mist/Clear soft-label window `800–1200 m`;
- current sampling ratios;
- no candidate gate is loaded or applied.

## Remote repository mapping

- Training repository: `saeva007/vis_mlp` →
  `/public/home/putianshu/vis_mlp/train`
- Evaluation repository: `saeva007/vis_eval` →
  `/public/home/putianshu/vis_mlp/paper_eval`

All executable files for this experiment live in one of those two repositories.
Do not place experiment scripts in the non-repository workspace root.

## Training candidates

`submit_static_rnn_precision_loss_candidates_chain.sh` creates only isolated
candidate runs:

- P0: historical loss reproduction;
- P1: light hard-Clear weighting;
- P2: sampler-invariant conditional FP/FN loss;
- P3: P1 + P2;
- P4: P3 with a narrower Mist/Clear soft window, enabled only when the
  validation false-alarm diagnostic reports at least 35% of Clear→Moderate
  false alarms in the 1000–1200 m band.

If P0–P3 and their validation-only gates have no feasible solution, a separate
recall-protection screen may be launched explicitly (these are not default
candidates):

- P5: conditional FP/FN weights `1.0/0.30`;
- P6: conditional FP/FN weights `1.0/0.50`;
- P7: conditional FP/FN weights `1.25/0.50`.

P5–P7 retain the original sampling, class weights, Fog/Mist focal gamma, and
soft-label windows. They do not use physical/aerosol hard-negative weights.
The screen is intended to test whether the weakest validation event can recover
recall before any full multi-seed training is considered.

If the operational objective changes from recall recovery to reducing visible
event-footprint overprediction, use the separate pair-specific screen:

- P8: light Clear→Ultra penalty on observed visibility >=3000 m;
- P9: balanced Clear→Ultra penalty plus Moderate false-negative guard;
- P10: stronger Clear→Ultra penalty and Moderate guard.

P8–P10 keep the original sampler and soft-label windows. Unlike P1–P3, their
extra loss does not suppress the combined Low-vis probability for every Clear
sample. It penalizes Ultra/Moderate probability separately on unambiguously
high-visibility Clear samples and explicitly protects Moderate-low recall.
They use validation CSI checkpoint selection and remain candidate-only.

P8–P10 must not be promoted when Clear→Ultra errors merely move to
Clear→Moderate. For the fixed-argmax event-footprint objective, use the
P3-derived Phase-C candidates instead:

- P11: P3 loss plus Phase-C focal prior correction with `beta=0.50`;
- P12: the same objective with stronger prior correction `beta=0.75`.

P11/P12 keep the P3 conditional total Low-vis FP/FN loss and physical/aerosol
hard-negative weights. They do not use the P8–P10 pair-specific penalties or
the Moderate recall guard. Phase C freezes the GRU and static encoder, keeps a
stratified batch for the prior-corrected focal term, and adds a second natural
station batch drawn from one valid time for the event-footprint term. The
footprint objective combines differentiable Low-vis CSI with augmented-
Lagrangian constraints on predicted area and widespread-event recall. All
reported decisions remain three-class `argmax`; this is not threshold tuning.

The event-footprint screen should report argmax metrics and should prefer a
candidate only when all of the following validation diagnostics improve over
P0: global Clear FPR, Clear→Ultra count, event-mean predicted/observed area
ratio, event footprint FAR, and predicted-count overshoot. Suggested screening
targets are event-mean area ratio <=1.8, each event recall >=0.55, Ultra recall
>=0.55, and Moderate recall >=0.25. These are screening targets rather than an
automatic mainline promotion rule.

Screening reuses one Stage-1 checkpoint and trains equal-step Stage-2 jobs.
Only the validation-selected top two configurations proceed to full S1→S2
training with seeds 42, 314, and 2718.

## Evaluation boundary

- Candidate selection uses validation outputs only.
- Paper-comparison output remains three-class `argmax`.
- The optional temperature-scaled Low-vis gate is an offline secondary
  experiment. Its JSON is marked `deployment_approved=false`.
- Test data are evaluated once after the configuration and gate are frozen.
- Passing the acceptance checks produces evidence for a future promotion
  decision; it does not perform that promotion.

## Entry points

Training repository:

- `submit_static_rnn_precision_loss_candidates_chain.sh`
- `select_static_rnn_precision_candidates.py`
- optional arguments in `train_static_rnn_lowvis.py`

Evaluation repository:

- `run_static_rnn_precision_candidate_eval.py`
- `sub_static_rnn_precision_candidate_eval.slurm`
- `diagnose_static_rnn_false_alarms.py`
- `fit_static_rnn_lowvis_gate.py`
- `apply_static_rnn_lowvis_gate.py`
- `bootstrap_static_rnn_metric_deltas.py`

## Minimal execution sequence

On the cluster, keep the current mainline id read-only and submit a separate
screening prefix:

```bash
export BASE=/public/home/putianshu/vis_mlp
export MAIN_RUN_ID=exp_114287869_static_mlp_gru_main
export MAIN_S1=${BASE}/checkpoints/${MAIN_RUN_ID}_S1_best_score.pt
export SCREEN_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_precision_loss_screen
export SCREEN_MANIFEST=${BASE}/train/logs/${SCREEN_PREFIX}_precision_loss_manifest.tsv

cd ${BASE}/train
LOWVIS_RNN_PRECISION_RUN_PREFIX=${SCREEN_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=screen \
LOWVIS_RNN_PRETRAINED_CKPT=${MAIN_S1} \
LOWVIS_RNN_PRECISION_CANDIDATES=p0:p1:p2:p3 \
LOWVIS_RNN_PRECISION_SEEDS=42 \
LOWVIS_RNN_PRECISION_MANIFEST=${SCREEN_MANIFEST} \
bash submit_static_rnn_precision_loss_candidates_chain.sh
```

After the Stage-2 jobs finish, evaluate validation data from the evaluation
repository:

```bash
export SCREEN_VAL_DIR=${BASE}/static_rnn_precision_candidate_eval/${SCREEN_PREFIX}_val
cd ${BASE}/paper_eval
sbatch --export=ALL,MANIFEST=${SCREEN_MANIFEST},SPLIT=val,RUN_EVENT_EVAL=1,OUT_DIR=${SCREEN_VAL_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm
```

Select two candidates for full experimental replication:

```bash
cd ${BASE}/train
python select_static_rnn_precision_candidates.py \
  --validation-summary-csv ${SCREEN_VAL_DIR}/precision_candidates_val_overall_metrics.csv \
  --validation-event-csv ${SCREEN_VAL_DIR}/precision_candidates_val_event_metrics.csv \
  --top-k 2 \
  --out-csv ${SCREEN_VAL_DIR}/constraint_ranking.csv \
  --out-json ${SCREEN_VAL_DIR}/constraint_ranking.json
```

The selector does not fill `top-k` with infeasible runs by default. When
`n_feasible=0`, `selected` is empty and no full training should be submitted.
The optional `--allow-infeasible-fallback` flag is diagnostic-only.

Then submit only those two candidate ids with three fixed seeds:

```bash
export FULL_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_precision_loss_full
export FULL_MANIFEST=${BASE}/train/logs/${FULL_PREFIX}_precision_loss_manifest.tsv
export SELECTED_CANDIDATES=p1:p3  # replace from validation ranking

LOWVIS_RNN_PRECISION_RUN_PREFIX=${FULL_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=full \
LOWVIS_RNN_PRECISION_CANDIDATES=${SELECTED_CANDIDATES} \
LOWVIS_RNN_PRECISION_SEEDS=42:314:2718 \
LOWVIS_RNN_PRECISION_MANIFEST=${FULL_MANIFEST} \
bash submit_static_rnn_precision_loss_candidates_chain.sh
```

Use a unique `exp_<timestamp>_precision_loss_*` prefix throughout. Do not edit
the current mainline run id, `paper_eval_config.json`, or deployment settings.

## P11/P12 calibration-only screen from P3

The first P11/P12 screen should reuse the validated P3 S2 checkpoint and run
only Phase C. This isolates footprint calibration from representation training
and avoids repeating P3 Phase A/B:

```bash
export BASE=/public/home/putianshu/vis_mlp
export P3_RUN_ID=exp_20260627_194302_precision_loss_screen_p3_seed42_2_proposed_rare_event_focal
export P3_CKPT=${BASE}/checkpoints/${P3_RUN_ID}_S2_PhaseB_best_score.pt
export FOOTPRINT_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_p3_event_footprint_screen
export FOOTPRINT_MANIFEST=${BASE}/train/logs/${FOOTPRINT_PREFIX}_precision_loss_manifest.tsv

cd ${BASE}/train
LOWVIS_RNN_PRECISION_RUN_PREFIX=${FOOTPRINT_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=screen \
LOWVIS_RNN_PRETRAINED_CKPT=${P3_CKPT} \
LOWVIS_RNN_PRECISION_CANDIDATES=p11:p12 \
LOWVIS_RNN_PRECISION_SEEDS=42 \
LOWVIS_RNN_PRECISION_MANIFEST=${FOOTPRINT_MANIFEST} \
LOWVIS_RNN_PRECISION_COMMON_ARGS="--threshold-mode argmax --s2-phase-a-steps 0 --s2-phase-b-steps 0" \
bash submit_static_rnn_precision_loss_candidates_chain.sh
```

The manifest automatically points P11/P12 to
`*_S2_PhaseC_best_score.pt`. After both jobs finish, run the normal validation
candidate evaluator with `RUN_EVENT_EVAL=1`, then apply the footprint-aware
selector:

```bash
export FOOTPRINT_VAL_DIR=${BASE}/static_rnn_precision_candidate_eval/${FOOTPRINT_PREFIX}_val
cd ${BASE}/paper_eval
sbatch --export=ALL,MANIFEST=${FOOTPRINT_MANIFEST},SPLIT=val,RUN_EVENT_EVAL=1,OUT_DIR=${FOOTPRINT_VAL_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm

cd ${BASE}/train
python select_static_rnn_precision_candidates.py \
  --validation-summary-csv ${FOOTPRINT_VAL_DIR}/precision_candidates_val_overall_metrics.csv \
  --validation-event-csv ${FOOTPRINT_VAL_DIR}/precision_candidates_val_event_metrics.csv \
  --max-fpr 0.030 \
  --min-low-vis-csi 0.190 \
  --min-low-vis-recall 0.55 \
  --min-event-recall 0.40 \
  --min-mean-event-recall 0.55 \
  --min-mean-event-csi 0.24 \
  --max-mean-event-area-ratio 1.80 \
  --max-event-area-ratio 2.20 \
  --min-ultra-recall 0.40 \
  --min-moderate-recall 0.20 \
  --min-moderate-csi 0.06 \
  --top-k 1 \
  --out-csv ${FOOTPRINT_VAL_DIR}/footprint_constraint_ranking.csv \
  --out-json ${FOOTPRINT_VAL_DIR}/footprint_constraint_ranking.json
```

Do not submit frozen test evaluation when `n_feasible=0`. If one beta is
feasible, submit only that candidate in `LOWVIS_RNN_PRECISION_STAGE=full` with
seeds `42:314:2718`; full mode runs the normal S1 and S2 Phase A/B training and
then appends Phase C.

## P13-P15 dual-stream sampling calibration

P13-P15 replace Phase-C prior correction and footprint penalties with a
decoupled Phase D. The balanced stream keeps the P3 designed-focal objective;
the natural stream uses unweighted cross entropy on complete valid-time station
snapshots. Half of the natural snapshots come from widespread-event times and
half from background times. A cosine schedule introduces the natural stream
only after 50% of Phase D, with final natural-loss fractions 0.30, 0.45, and
0.60 for P13, P14, and P15 respectively.

The initial screen reuses the verified P3 Phase-B checkpoint and trains only
Phase D:

```bash
export BASE=/public/home/putianshu/vis_mlp
export P3_RUN_ID=exp_20260627_194302_precision_loss_screen_p3_seed42_2_proposed_rare_event_focal
export P3_CKPT=${BASE}/checkpoints/${P3_RUN_ID}_S2_PhaseB_best_score.pt
export SAMPLING_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_p3_sampling_calibration_screen
export SAMPLING_MANIFEST=${BASE}/train/logs/${SAMPLING_PREFIX}_precision_loss_manifest.tsv

test -f "${P3_CKPT}" || { echo "Missing ${P3_CKPT}"; exit 1; }
cd ${BASE}/train
LOWVIS_RNN_PRECISION_RUN_PREFIX=${SAMPLING_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=screen \
LOWVIS_RNN_PRETRAINED_CKPT=${P3_CKPT} \
LOWVIS_RNN_PRECISION_CANDIDATES=p13:p14:p15 \
LOWVIS_RNN_PRECISION_SEEDS=42 \
LOWVIS_RNN_PRECISION_MANIFEST=${SAMPLING_MANIFEST} \
LOWVIS_RNN_LOCAL_CACHE_ID=${SAMPLING_PREFIX}_shared_data \
LOWVIS_RNN_PRECISION_COMMON_ARGS="--threshold-mode argmax --s2-phase-a-steps 0 --s2-phase-b-steps 0 --s2-phase-c-steps 0" \
bash submit_static_rnn_precision_loss_candidates_chain.sh
```

After all three Phase-D jobs complete, run fixed-argmax validation with event
evaluation and select one eta:

```bash
export SAMPLING_VAL_DIR=${BASE}/static_rnn_precision_candidate_eval/${SAMPLING_PREFIX}_val
cd ${BASE}/paper_eval
sbatch --export=ALL,MANIFEST=${SAMPLING_MANIFEST},SPLIT=val,RUN_EVENT_EVAL=1,OUT_DIR=${SAMPLING_VAL_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm

cd ${BASE}/train
python select_static_rnn_precision_candidates.py \
  --validation-summary-csv ${SAMPLING_VAL_DIR}/precision_candidates_val_overall_metrics.csv \
  --validation-event-csv ${SAMPLING_VAL_DIR}/precision_candidates_val_event_metrics.csv \
  --max-fpr 0.025 \
  --min-low-vis-csi 0.195 \
  --min-low-vis-recall 0.0 \
  --min-event-recall 0.20 \
  --min-mean-event-recall 0.45 \
  --min-mean-event-csi 0.235 \
  --min-mean-event-area-ratio 0.80 \
  --max-mean-event-area-ratio 1.80 \
  --max-event-area-ratio 2.20 \
  --min-ultra-recall 0.40 \
  --min-moderate-recall 0.20 \
  --min-moderate-csi 0.06 \
  --top-k 1 \
  --out-csv ${SAMPLING_VAL_DIR}/sampling_constraint_ranking.csv \
  --out-json ${SAMPLING_VAL_DIR}/sampling_constraint_ranking.json
```

Do not use test data when `n_feasible=0`. When one eta is feasible, submit
that candidate in `full` mode with seeds `42:314:2718`. Full mode trains S1,
S2 Phase A/B, and then appends the same Phase D strategy. Evaluate the complete
three-seed manifest first on validation and then once on frozen test:

```bash
export FULL_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_sampling_calibration_full
export FULL_MANIFEST=${BASE}/train/logs/${FULL_PREFIX}_precision_loss_manifest.tsv
export SELECTED_CANDIDATE=p14  # replace with the validation-selected candidate id

cd ${BASE}/train
LOWVIS_RNN_PRECISION_RUN_PREFIX=${FULL_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=full \
LOWVIS_RNN_PRECISION_CANDIDATES=${SELECTED_CANDIDATE} \
LOWVIS_RNN_PRECISION_SEEDS=42:314:2718 \
LOWVIS_RNN_PRECISION_MANIFEST=${FULL_MANIFEST} \
bash submit_static_rnn_precision_loss_candidates_chain.sh

export FULL_TEST_DIR=${BASE}/static_rnn_precision_candidate_eval/${FULL_PREFIX}_test
cd ${BASE}/paper_eval
sbatch --export=ALL,MANIFEST=${FULL_MANIFEST},SPLIT=test,RUN_EVENT_EVAL=1,OUT_DIR=${FULL_TEST_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm
```

The frozen test job writes overall metrics, per-class precision/recall/CSI,
confusion counts, and event metrics to
`precision_candidates_test_overall_metrics.csv`,
`precision_candidates_test_per_class_metrics.csv`,
`precision_candidates_test_confusion_counts.csv`, and
`precision_candidates_test_event_metrics.csv`.

## Formal P13 three-seed probability ensemble

For long-running Slurm monitoring and candidate-specific recovery from RCCL,
node, or silent-progress stalls, use `STATIC_RNN_LOSS_WATCHDOG.md`.  The CPU
watchdog preserves the original submission manifest and writes a resolved
manifest that must be used after any retry.

P13 was the only feasible eta in the fixed-argmax validation screen. Its formal
run is therefore a new full S1 -> S2 Phase A/B -> Phase D training for each of
the pre-registered seeds `42`, `314`, and `2718`. Do not reuse the screen
checkpoint in the formal result and do not tune a threshold after the full
training.

The ensemble decision is defined before opening the frozen test result:

```text
p_mean(i, c) = [p_42(i, c) + p_314(i, c) + p_2718(i, c)] / 3
y_hat(i) = argmax_c p_mean(i, c)
```

This is a mean of post-softmax class probabilities, followed by one argmax. It
is not a majority vote over three argmax labels and it is not a mean of three
reported scores.

### 1. Submit the three independent full training chains

Run from the remote repository root. The launcher submits three S1 jobs and
three corresponding S2 jobs, with each S2 depending on its own S1.

```bash
set -euo pipefail

export BASE=/public/home/putianshu/vis_mlp
export FULL_PREFIX=exp_$(date +%Y%m%d_%H%M%S)_p13_sampling_calibration_full
export FULL_MANIFEST=${BASE}/train/logs/${FULL_PREFIX}_precision_loss_manifest.tsv

cd ${BASE}/train
LOWVIS_RNN_PRECISION_RUN_PREFIX=${FULL_PREFIX} \
LOWVIS_RNN_PRECISION_STAGE=full \
LOWVIS_RNN_PRECISION_CANDIDATES=p13 \
LOWVIS_RNN_PRECISION_SEEDS=42:314:2718 \
LOWVIS_RNN_PRECISION_MANIFEST=${FULL_MANIFEST} \
LOWVIS_RNN_LOCAL_CACHE_ID=${FULL_PREFIX}_shared_data \
LOWVIS_RNN_PRECISION_COMMON_ARGS="--threshold-mode argmax" \
bash submit_static_rnn_precision_loss_candidates_chain.sh

echo "FULL_PREFIX=${FULL_PREFIX}"
echo "FULL_MANIFEST=${FULL_MANIFEST}"
```

The expected final checkpoints are the three manifest entries in column 13.
After all six training jobs finish, verify them before evaluation:

```bash
awk -F'\t' 'NR>1 && $1=="p13" {print "seed=" $5, "run_id=" $8, "ckpt=" $13}' \
  "${FULL_MANIFEST}"

awk -F'\t' 'NR>1 && $1=="p13" {print $13}' "${FULL_MANIFEST}" | \
while IFS= read -r ckpt; do
  test -s "${ckpt}" || { echo "MISSING: ${ckpt}"; exit 1; }
  echo "OK: ${ckpt}"
done
```

### 2. Recheck the three seeds on validation

This is a stability and pre-registration check only. Do not change P13, the
argmax decision, the event definitions, or any threshold after inspecting it.

```bash
export FULL_VAL_DIR=${BASE}/static_rnn_precision_candidate_eval/${FULL_PREFIX}_val
test ! -e "${FULL_VAL_DIR}"
unset EXTRA_ARGS

cd ${BASE}/paper_eval
VAL_JOB=$(sbatch --parsable \
  --export=ALL,MANIFEST=${FULL_MANIFEST},SPLIT=val,RUN_EVENT_EVAL=1,OUT_DIR=${FULL_VAL_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm)
echo "VAL_JOB=${VAL_JOB}"
```

Only after this validation audit is accepted should the frozen test jobs below
be submitted.

### 3. Produce three-member probabilities on the main and 48 h test sets

```bash
export FULL_TEST_DIR=${BASE}/static_rnn_precision_candidate_eval/${FULL_PREFIX}_test
export FULL_48H_DIR=${BASE}/static_rnn_precision_candidate_eval/${FULL_PREFIX}_48h_members
export ENSEMBLE_REUSE_DIR=${BASE}/static_rnn_precision_candidate_eval/${FULL_PREFIX}_p13_seed_mean_argmax

test ! -e "${FULL_TEST_DIR}"
test ! -e "${FULL_48H_DIR}"
test ! -e "${ENSEMBLE_REUSE_DIR}"
unset EXTRA_ARGS

cd ${BASE}/paper_eval
MAIN_TEST_JOB=$(sbatch --parsable \
  --export=ALL,MANIFEST=${FULL_MANIFEST},SPLIT=test,RUN_EVENT_EVAL=0,OUT_DIR=${FULL_TEST_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm)

export EXTRA_ARGS="--data_dir ${BASE}/ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime"
H48_TEST_JOB=$(sbatch --parsable \
  --export=ALL,MANIFEST=${FULL_MANIFEST},SPLIT=test,RUN_EVENT_EVAL=0,OUT_DIR=${FULL_48H_DIR},DEVICE=cpu \
  sub_static_rnn_precision_candidate_eval.slurm)
unset EXTRA_ARGS

echo "MAIN_TEST_JOB=${MAIN_TEST_JOB}"
echo "H48_TEST_JOB=${H48_TEST_JOB}"
```

The member jobs deliberately use `RUN_EVENT_EVAL=0`. Event metrics and figures
are calculated once from the frozen ensemble, rather than three times from
members that are not the paper prediction.

### 4. Align samples, average probabilities, and create reusable 12 h/48 h products

The preparation job checks the exact candidate and seed set, full-stage
manifest, member run IDs, station/time/duplicate-index identities, observed
labels and visibility, probability normalization, dataset row order, and file
hashes. A mismatch aborts instead of silently averaging arrays by length.

```bash
cd ${BASE}/paper_eval
MEAN_JOB=$(sbatch --parsable \
  --dependency=afterok:${MAIN_TEST_JOB}:${H48_TEST_JOB} \
  --export=ALL,MANIFEST=${FULL_MANIFEST},MAIN_EVAL_DIR=${FULL_TEST_DIR},FORECAST48_EVAL_DIR=${FULL_48H_DIR},OUT_DIR=${ENSEMBLE_REUSE_DIR} \
  sub_prepare_static_rnn_seed_mean_for_eval.slurm)
echo "MEAN_JOB=${MEAN_JOB}"
```

The reusable directory contains `probs.npy`, `pred_argmax.npy`, member and
ensemble metric tables, strict provenance in `run_config.json`, and 48 h lead
tables generated from the same three-seed probability mean.

### 5. Redraw the complete paper-evaluation figures from the frozen ensemble

The seed-42 checkpoint below is only the representative architecture/scaler
loader required by the existing evaluator. Main-test and 48 h predictions both
come from `ENSEMBLE_REUSE_DIR`; they must be described as the three-seed P13
ensemble, not as seed 42.

```bash
export SEED42_RUN_ID=$(awk -F'\t' 'NR>1 && $1=="p13" && $5=="42" {print $8; exit}' "${FULL_MANIFEST}")
export SEED42_CKPT=$(awk -F'\t' 'NR>1 && $1=="p13" && $5=="42" {print $13; exit}' "${FULL_MANIFEST}")
export PAPER_OUT=${BASE}/static_rnn_eval_results/${FULL_PREFIX}_p13_seed_mean_argmax

test -n "${SEED42_RUN_ID}"
test -s "${SEED42_CKPT}"
test ! -e "${PAPER_OUT}"

export EXTRA_ARGS="--threshold_source argmax --skip_feature_importance --skip_variable_quality --skip_overlap_source_comparison"
cd ${BASE}/paper_eval
PLOT_JOB=$(sbatch --parsable \
  --dependency=afterok:${MEAN_JOB} \
  --export=ALL,CONFIG_JSON=${BASE}/paper_eval/paper_eval_config.json,MODE=main,DATA_DIR=ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2,OUT_DIR=${PAPER_OUT},MAIN_RUN_ID=${SEED42_RUN_ID},MAIN_CKPT=${SEED42_CKPT},STAGE_TAG=S2_PhaseD,REUSE_INFERENCE_DIR=${ENSEMBLE_REUSE_DIR},PLOTS=all,DEVICE=cpu \
  sub_static_rnn_lowvis_eval.slurm)
unset EXTRA_ARGS
echo "PLOT_JOB=${PLOT_JOB}"
```

Do not add `--skip_static_48h` when the preparation job has completed. The
reuse directory supplies the 48 h ensemble tables. If the 48 h member job was
not run, then `--skip_static_48h` is mandatory to prevent a mixed ensemble and
single-seed figure set.

### 6. Redraw only the three event figures after layout changes

This command reuses the completed ensemble evaluation and does not rerun model
inference:

```bash
export ENSEMBLE_EVAL_DIR=${PAPER_OUT}/${SEED42_RUN_ID}
export EVENT_OUT=${ENSEMBLE_EVAL_DIR}/event_rerun_mean_argmax
test -s "${ENSEMBLE_EVAL_DIR}/per_sample_eval.csv"
test ! -e "${EVENT_OUT}"

cd ${BASE}/paper_eval
EVENT_JOB=$(sbatch --parsable \
  --export=ALL,EVAL_DIR=${ENSEMBLE_EVAL_DIR},OUT_DIR=${EVENT_OUT},EVENT_ENV_SOURCE=grid,EVENT_ENV_MAX_EVENTS=3,WINDOW_HOURS=3 \
  sub_rerun_static_rnn_event_figures.slurm)
echo "EVENT_JOB=${EVENT_JOB}"
```

Report the ensemble scores as the primary result and the three member scores as
mean +/- SD only as a training-stability diagnostic. Three seeds are not a
test-sample uncertainty interval; confidence intervals for the primary claim
should still use date-block or event-block bootstrap on the frozen ensemble.
