# Formal spatial mapping-operator experiment

## Fixed scientific question

Does the learned forecast-to-station visibility mapping retain its advantage at
stations excluded from all model fitting, and how much of that advantage comes
from nonlinear instantaneous mapping versus 12-hour temporal context?

## Frozen design

- Dataset: the existing nationwide Tianji S2 train/validation/test arrays.
- Temporal boundary: the existing train, validation, and frozen test files are
  preserved unchanged.
- Spatial boundary: station coordinates alone define five balanced recursive
  spherical-coordinate blocks (`seed=20260815` is used only to break exact
  projection ties); visibility labels never enter fold construction. Held-out
  station counts differ by at most one across folds.
- Buffer: training and validation stations within 50 km of a held-out station
  are excluded for that fold.
- Training rows: existing training times at non-held-out, non-buffer stations.
- Validation rows: existing validation times at the same eligible stations.
- Final rows: existing frozen test times at held-out stations only.
- Preprocessing: fitted independently from each fold's training rows.
- Decision rule: learned operators use three-class argmax; the same rule is
  used for validation checkpoint selection and is frozen before test labels
  are loaded.

## Operators

1. `logistic`: multinomial linear logistic mapping trained with log loss.
2. `mlp`: nonlinear instantaneous mapping using the final dynamic state.
3. `gru`: the same static branch plus the full 12-hour dynamic sequence.

All learned operators omit the separate engineered-feature block so that the
MLP cannot receive hidden 12-hour summaries. They use the same dynamic/static
information contract, three visibility classes, training-only target class
proportions (Fog 0.18, Mist 0.22, Clear 0.60), and argmax checkpoint-selection policy.
The neural baselines are trained from scratch on S2 so neither receives a
pretraining advantage unavailable to logistic regression.

The native IFS diagnostic is required during formal aggregation as a
fitting-free operational reference. It is exactly aligned to the frozen test
metadata by UTC time, station ID, and within-key occurrence, must cover the
entire frozen test set before validity masking, and is reclassified directly
from native visibility at 500 m and 1000 m. All four plotted operators are
restricted to the same IFS-valid rows within each fold. IFS is not treated as
an architecture rung and has no AP because it does not provide class
probabilities.

## Paper-facing endpoints

- Primary: pooled out-of-fold low-visibility CSI and recall on the common
  IFS-diagnostic-matched test rows.
- Secondary: argmax low-visibility precision and FPR, plus learned-operator AP.
- Robustness display: the five fold values and station-level metrics.
- Figure exports: the complete operator ladder and a companion direct comparison
  containing only VisCast (`gru`) and native IFS diagnostic visibility.

## Submission

From `/public/home/putianshu/vis_mlp/train`:

```bash
bash submit_spatial_mapping_cv_chain.sh
```

The launcher detaches its worker, stores every JobID in
`$RESULT_ROOT/submission_state.sh`, and submits preparation, Logistic and neural
arrays, then aggregation through `afterok` dependencies.

## IFS-input mainline-only transfer run

`submit_ifs_gru_mapping_cv_chain.sh` repeats both the balanced spatial blocked
five-fold experiment and the embargoed calendar-blocked temporal five-fold
experiment with the IFS `source_full` S2 dataset. It submits only the mainline
`gru` operator: no Logistic or instantaneous MLP jobs are created. The two
five-task neural arrays are independent and can run concurrently.

Each fold is trained from scratch. Reusing the existing full-data IFS S1/S2
checkpoint would expose held-out stations to the spatial-CV model through
pretraining, so it is not valid for this transferability endpoint. Fold-local
scalers, argmax checkpoint selection, and the frozen-test rule are unchanged.

Before either neural array can start, each preparation job verifies that the
native IFS diagnostic CSV covers every frozen IFS test row exactly once by UTC
time, station ID, and within-key occurrence, and that the observed class labels
match. Aggregation is restricted to `gru,ifs_native`; the plotter emits a
two-operator CSI/recall figure.

Default data and output roots:

```text
DATA_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full
SPATIAL_RESULT_ROOT=/public/home/putianshu/vis_mlp/spatial_mapping_cv_ifs/<run>_spatial
TEMPORAL_RESULT_ROOT=/public/home/putianshu/vis_mlp/temporal_mapping_cv_ifs/<run>_temporal
```

Submit from the synchronized repository root:

```bash
bash submit_ifs_gru_mapping_cv_chain.sh
```

This `source_full` run measures source-specific operational transfer. It must
not be described as a controlled Tianji-versus-IFS data-quality attribution,
because the available predictor inventories differ. For that stricter claim,
override `DATA_DIR` with the agreed common-variable IFS dataset and pair it with
the corresponding Tianji common-variable run.
