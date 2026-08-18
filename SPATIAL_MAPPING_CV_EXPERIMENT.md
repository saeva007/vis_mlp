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
