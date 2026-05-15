# Feature Engineering Ablation Runbook

This runbook covers the PMST feature-engineering ablation experiment after the
S2 validation-split, aerosol-routing, and UTC fixes.

## Scope

- Training script: `PMST_net_test_11_s2_pm10_fe_ablation.py`
- Submit script: `sub_ablation.slurm`
- Base S2 script imported by the ablation wrapper: `PMST_net_test_11_s2_pm10.py`
- Main S2 data: `/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2`

The FE ablation has no separate data-build step. It masks the precomputed FE
block from the corrected S2 PM10+PM2.5 month-tail dataset and otherwise uses the
same Stage-2 training protocol as `PMST_net_test_11_s2_pm10.py`.

## Required Order

1. Sync the latest `vis_mlp/train` code on the cluster.
2. Confirm the corrected S1 checkpoint referenced by the base S2 script exists:
   `/public/home/putianshu/vis_mlp/checkpoints/exp_1778563813_S1_best_score.pt`
3. Confirm the corrected S2 PM10+PM2.5 month-tail dataset exists.
4. Train each FE ablation variant.
5. Evaluate and plot FE ablation results from `paper_eval`.

## Train FE Ablation Variants

Run from the remote checkout root:

```bash
cd /public/home/putianshu/vis_mlp/train
mkdir -p logs
```

Minimal paper-critical rerun:

```bash
sbatch --export=ALL,ABLATION_TYPE=no_fe_all sub_ablation.slurm
```

Optional group ablations:

```bash
for mode in no_core_physics no_temporal_stats no_empirical_flags no_boundary_layer no_time_cyc; do
  sbatch --export=ALL,ABLATION_TYPE="${mode}" sub_ablation.slurm
done
```

Optional keep-only variants:

```bash
for mode in only_core_physics only_temporal_stats only_time_cyc; do
  sbatch --export=ALL,ABLATION_TYPE="${mode}" sub_ablation.slurm
done
```

Use `TRAIN_EXTRA_ARGS` only for explicit smoke tests or controlled reruns, for
example:

```bash
sbatch --export=ALL,ABLATION_TYPE=no_fe_all,TRAIN_EXTRA_ARGS="--steps_scale 0.05 --run_suffix_prefix smoke" sub_ablation.slurm
```

## Evaluate FE Ablation Results

The evaluation job lives in the paper-eval repository, not this training
repository:

```bash
cd /public/home/putianshu/vis_mlp/paper_eval
mkdir -p logs
sbatch --export=ALL,CONFIG_JSON=/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json,REFRESH_SPEC=1 sub_run_ablation_eval.slurm
```

If additional ablation variants are trained, update or regenerate the spec CSV
before plotting. The evaluator checks checkpoint temporal-input width to avoid
mixing legacy PM10-only checkpoints with current PM10+PM2.5 data.

## Notes

- `PMST_net_test_11_s2_pm10_fe_ablation.py` now refuses to run unless the
  imported base S2 script exposes the current explicit split and layout helpers.
- Reusing old FE ablation checkpoints from before the S2 fixes is not valid for
  the paper.
