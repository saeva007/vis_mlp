# Mainline architecture-replacement baselines

This experiment replaces only the current Static-MLP+GRU architecture with
Random Forest, XGBoost, or LightGBM. It does not change or rebuild the
established S2 Tianji train/validation/test split.

## Scientific contract

- Source dataset:
  `ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2`.
- Input information is matched to the neural mainline: all 12-hour dynamic
  fields, five continuous station attributes, vegetation, and all engineered
  features. The sequence is flattened for tree models and vegetation is
  one-hot encoded.
- The vegetation vocabulary, skewed-variable transforms, and missing-value
  medians are fitted on training data only.
- Fog, Mist, and Clear retain the physical thresholds `<500 m`,
  `500-1000 m`, and `>=1000 m`.
- Mainline class weights `2.0 / 2.0 / 0.8` are applied consistently to all
  three tree families. Both weighted raw probabilities and analytical
  prior-corrected probabilities are reported.
- Boosting early stopping sees validation only. Test is never used for fitting,
  early stopping, parameter choice, probability correction, or thresholds.
- The decision rule is argmax. This experiment performs no validation or test
  threshold search.

The old `XGboost_Ablation.py`, `LightGBM_Ablation.py`, and
`sub_XGboost_LightGBM.slurm` are retained as historical files, but are not valid
entrypoints for this experiment: they point to obsolete data, drop the wrong
feature column, omit Random Forest, and do not perform independent test
evaluation.

## Interpretation boundary

All three tree families are trained on S2 only. This is intentional: Random
Forest has no operation symmetric to neural S1 pretraining and S2 fine-tuning,
while continuing XGBoost or LightGBM trees from S1 would create an asymmetric
comparison. Therefore:

1. use an S2-from-scratch Static-MLP+GRU as the architecture-isolated neural
   control;
2. report the established S1-to-S2 neural mainline separately as the complete
   operational-system reference.

The matching S2-from-scratch control can reuse the current launcher:

```bash
S2_DATA=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
S2_SCRATCH_ID=exp_static_gru_s2_scratch_$(date +%Y%m%d_%H%M%S)

sbatch --export="ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_S2_DATA_DIR=${S2_DATA},LOWVIS_RNN_RUN_ID=${S2_SCRATCH_ID},LOWVIS_RNN_PRETRAINED_CKPT=,LOWVIS_RNN_S2_A_STEPS=0,LOWVIS_RNN_S2_B_STEPS=30000" \
  sub_static_rnn_lowvis_main.slurm
```

## Why these fixed parameters

Recent visibility work continues to use RF, XGBoost, and LightGBM as strong
tabular baselines and generally tunes them on validation data rather than
randomly splitting station-times. The fixed first-pass preset here is
deliberately conservative for a multi-million-row, high-dimensional dataset:

- RF uses 400 bootstrapped trees, sqrt feature subsampling, 0.65 row
  subsampling, depth 22, and minimum leaf size 20.
- XGBoost uses histogram trees, depth 8, learning rate 0.03, row/feature
  fractions 0.8, conservative child and update constraints, L1/L2
  regularization, and validation early stopping.
- LightGBM uses 63 leaves, depth 10, minimum 500 rows per leaf, row/feature
  fractions 0.8, L1/L2 regularization, and validation early stopping. The large
  leaf minimum follows LightGBM guidance for large datasets and limits
  over-specific leaf-wise growth.

These are fixed architecture-comparison anchors, not claims of globally optimal
hyperparameters. If the first run is competitive, a small validation-only
search and multiple seeds can be added later; the test set must remain frozen.

Literature and implementation references used for this first-pass design:

- Schütz et al. (2026), *Atmospheric Research*,
  [doi:10.1016/j.atmosres.2025.108395](https://doi.org/10.1016/j.atmosres.2025.108395):
  recent XGBoost visibility forecasting with explicit attention to fog
  formation/dissipation and imbalance.
- Chen et al. (2024), *Scientific Reports*,
  [doi:10.1038/s41598-024-61572-8](https://doi.org/10.1038/s41598-024-61572-8):
  RF/XGBoost/LightGBM visibility baselines optimized on validation data.
- Kim (2021), *Journal of Korea Water Resources Association*,
  [doi:10.3741/JKWRA.2021.54.12.1255](https://doi.org/10.3741/JKWRA.2021.54.12.1255):
  direct regional fog comparison showing the sensitivity/false-alarm
  trade-off among RF, XGBoost, and LightGBM.
- Lee and Chun (2025), *npj Climate and Atmospheric Science*,
  [doi:10.1038/s41612-025-01260-0](https://doi.org/10.1038/s41612-025-01260-0):
  a multi-million-sample operational aviation comparison of the same three
  tree families and their rare-event trade-offs.
- The current
  [XGBoost parameter guide](https://xgboost.readthedocs.io/en/stable/parameter.html),
  [LightGBM tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html),
  and
  [scikit-learn RandomForestClassifier guide](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  provide the implementation-specific regularization and sampling constraints.

## Full run

Build the shared cache once:

```bash
cd /public/home/putianshu/vis_mlp/train

DATA_DIR=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
CACHE_DIR=/public/home/putianshu/vis_mlp/tree_baseline_cache/s2_monthtail_w12_v1

PREP_JOB=$(sbatch --parsable \
  --export="ALL,LOWVIS_TREE_DATA_DIR=${DATA_DIR},LOWVIS_TREE_CACHE_DIR=${CACHE_DIR}" \
  sub_prepare_mainline_tree_features.slurm)
echo "${PREP_JOB}"
```

Start all three baselines only after the cache succeeds:

```bash
RUN_ID=exp_mainline_tree_$(date +%Y%m%d_%H%M%S)

TREE_JOB=$(sbatch --parsable \
  --dependency=afterok:${PREP_JOB} \
  --export="ALL,LOWVIS_TREE_CACHE_DIR=${CACHE_DIR},LOWVIS_TREE_RUN_ID=${RUN_ID}" \
  sub_mainline_tree_baselines.slurm)
echo "${TREE_JOB}"
```

Array mapping is `0=RF`, `1=XGBoost`, and `2=LightGBM`. To rerun only one
model, add `--array=1` (for example) to the second submission.

Completed runs write:

- model serialization;
- `probs_val.npy` and `probs_test.npy` after fixed prior correction;
- `probs_*_weighted_raw.npy`;
- top-level `probs.npy` as the test probability compatibility artifact;
- `metrics.json`, `metrics.csv`, `feature_importance.csv`, and
  `run_config.json`.

Summarize the array after it finishes:

```bash
python mainline_tree_baselines.py summarize \
  --run-root "/public/home/putianshu/vis_mlp/tree_baseline_runs/${RUN_ID}" \
  --output-csv "/public/home/putianshu/vis_mlp/tree_baseline_runs/${RUN_ID}/comparison.csv"
```

## Small CPU smoke test

Use fresh smoke directories; do not reuse them for scientific evaluation:

```bash
SMOKE_CACHE=/public/home/putianshu/vis_mlp/tree_baseline_cache/smoke_${USER}_$(date +%s)
SMOKE_RUN=smoke_tree_$(date +%s)

PREP_JOB=$(sbatch --parsable \
  --export="ALL,LOWVIS_TREE_CACHE_DIR=${SMOKE_CACHE},LOWVIS_TREE_PREP_ROW_LIMIT=20000" \
  sub_prepare_mainline_tree_features.slurm)

sbatch --dependency=afterok:${PREP_JOB} --array=0-2 \
  --export="ALL,LOWVIS_TREE_CACHE_DIR=${SMOKE_CACHE},LOWVIS_TREE_RUN_ID=${SMOKE_RUN},LOWVIS_TREE_RF_TREES=10,LOWVIS_TREE_NUM_BOOST_ROUND=10,LOWVIS_TREE_EARLY_STOPPING_ROUNDS=3" \
  sub_mainline_tree_baselines.slurm
```
