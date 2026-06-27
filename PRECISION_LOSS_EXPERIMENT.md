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
