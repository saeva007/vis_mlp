# Low-Visibility Trajectory Diffusion Candidate

This workflow is candidate-only. It does not replace, import, or post-process
the paper-facing Static-MLP+GRU predictions.

## Scientific contract

- Condition: one Tianji initialization, one station, hourly 0-48 h forecast
  trajectory, 27 dynamic variables, five continuous station descriptors,
  vegetation category, and initialization-time encoding.
- Response: absolute observed `log1p(visibility)` at leads 12-48 h.
- No Static-RNN logits, residual target, auxiliary classifier, or 36-variable
  hand-engineered feature branch is used.
- Fog is `<500 m`, Mist is `500-1000 m`, and Clear is `>=1000 m`.
- Candidate probabilities are ensemble frequencies within those fixed physical
  intervals. Decision thresholds are fitted on validation only and frozen for
  test.

## Tracked locations and remote roots

- Training/data code is tracked in `vis_mlp` and deployed under
  `/public/home/putianshu/vis_mlp/train`.
- Evaluation code is tracked in `vis_eval` and deployed under
  `/public/home/putianshu/vis_mlp/paper_eval`.
- Generated datasets, checkpoints, logs, and results live under
  `/public/home/putianshu/vis_mlp`; do not add them to either repository.

## 1. Build the trajectory dataset (CPU)

The builder fails if PM10/PM2.5 sources are missing, if the canonical PM policy
is unavailable/stale, or if an output dataset already exists. It never silently
falls back to zero PM. To intentionally rebuild, use a new output tag; use
`--allow-overwrite` only after archiving the prior dataset.

The CPU builder and DCU trainer use the same validated Jarvis torch runtime by
default (`/public/home/jarvis226/miniconda3/envs/torch`). The activation helper
also supplies the cluster OpenSSL 1.1 and compatible HIPNN libraries on CPU
nodes. The effective loader order is Jarvis lib, OpenSSL 1.1, compatible HIPNN,
then any DTK/inherited libraries. This prevents both `libssl.so.1.1` failures
and the incompatible `libgalaxyhip.so.5` / `hipThreadExchangeStreamCaptureMode`
symbol error. An explicit `LOWVIS_TRAJ_TORCH_ENV` still takes precedence.

```bash
cd /public/home/putianshu/vis_mlp/train
sbatch sub_build_lowvis_trajectory_dataset.slurm
```

Recommended explicit version tag:

```bash
sbatch --export=ALL,LOWVIS_TRAJ_DATA_DIR=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_trajectory_0_48h_pm10_pm25_v1 \
  sub_build_lowvis_trajectory_dataset.slurm
```

Before training, verify that `dataset_build_config.json` reports non-zero
train/val/test counts, `dynamic_feature_order` has 27 entries, and the policy
versions are:

- `pmst_canonical_units_v2_20260630`
- `pm_explicit_legacy_scale_then_train_median_qc_v2_20260701`

The five-node launcher also checks the config, all split arrays and metadata
before starting `torchrun`; an incomplete build exits once with the exact
missing-file list instead of producing the same traceback on all 20 ranks.

## 2. Train the Gaussian benchmark, then diffusion (five nodes, 20 DCUs)

The Gaussian model is a separate distributional baseline, not a component of
the diffusion model.  The five-node profile uses 20 ranks with a per-rank
batch of 32 (global batch 640), rather than multiplying the original per-rank
batch 128 to an overly large global batch.  It also enables:

- separate station/static, vegetation, initialization-time and meteorological
  condition tokens;
- `d_model=192`, five condition layers and six denoising layers;
- EMA inference weights, 2,000-step warmup and cosine learning-rate decay;
- Min-SNR (`gamma=5`) diffusion-timestep weighting and stratified timestep
  draws; these are not class or event weights;
- checkpoint selection on fixed-noise validation ensembles, weighted 40% for
  overall CRPS and 60% for low-visibility conditional CRPS; validation is
  sharded across all 20 ranks rather than leaving 19 DCUs idle.

The same architecture and checkpoint-selection profile is used for the
Gaussian comparison where applicable. Test data are never used for training,
early stopping, threshold fitting or checkpoint selection.

```bash
cd /public/home/putianshu/vis_mlp/train

sbatch --export=ALL,LOWVIS_TRAJ_MODEL_TYPE=gaussian,LOWVIS_TRAJ_RUN_ID=exp_lowvis_traj_gaussian_5n_v2 \
  sub_lowvis_trajectory_diffusion.slurm

sbatch --export=ALL,LOWVIS_TRAJ_MODEL_TYPE=diffusion,LOWVIS_TRAJ_RUN_ID=exp_lowvis_traj_diffusion_5n_v2 \
  sub_lowvis_trajectory_diffusion.slurm
```

Do not submit the second command until the Gaussian job has at least reached
its first successful validation. This keeps cluster failures separate from
model comparisons; the jobs do not share checkpoints.

For a one-node DCU smoke run, override the node count and use a fresh run id:

```bash
sbatch -N 1 --export=ALL,LOWVIS_TRAJ_MODEL_TYPE=diffusion,LOWVIS_TRAJ_RUN_ID=smoke_lowvis_traj_diffusion_5n_v2,LOWVIS_TRAJ_MAX_STEPS=2,LOWVIS_TRAJ_VAL_INTERVAL=1,LOWVIS_TRAJ_BATCH_SIZE=2,LOWVIS_TRAJ_VAL_MONITOR_SIZE=4,LOWVIS_TRAJ_VAL_MEMBERS=2,LOWVIS_TRAJ_VAL_DDIM_STEPS=2 \
  sub_lowvis_trajectory_diffusion.slurm
```

Do not reuse a smoke checkpoint for paper evaluation.

## 3. Evaluate with frozen validation thresholds

Evaluate Gaussian first, then diffusion and point the latter to the Gaussian
result directory for the comparison table.

```bash
cd /public/home/putianshu/vis_mlp/paper_eval

sbatch --export=ALL,LOWVIS_TRAJ_CHECKPOINT=/public/home/putianshu/vis_mlp/checkpoints/exp_lowvis_traj_gaussian_5n_v2_best.pt,LOWVIS_TRAJ_EVAL_RUN_ID=exp_lowvis_traj_gaussian_5n_v2 \
  sub_lowvis_trajectory_diffusion_eval.slurm

sbatch --export=ALL,LOWVIS_TRAJ_CHECKPOINT=/public/home/putianshu/vis_mlp/checkpoints/exp_lowvis_traj_diffusion_5n_v2_best.pt,LOWVIS_TRAJ_EVAL_RUN_ID=exp_lowvis_traj_diffusion_5n_v2,LOWVIS_TRAJ_EVAL_EXTRA_ARGS="--compare-result-dir /public/home/putianshu/vis_mlp/trajectory_diffusion_results/exp_lowvis_traj_gaussian_5n_v2" \
  sub_lowvis_trajectory_diffusion_eval.slurm
```

Optional exact Static-RNN comparison requires both arrays and exact
`init_time, station_id, lead_hour` metadata. No nearest-time or station
approximation is allowed:

```bash
LOWVIS_TRAJ_EVAL_EXTRA_ARGS="--mainline-probs /path/to/probs.npy --mainline-meta /path/to/meta.csv --mainline-thresholds /path/to/validation_thresholds.json"
```

## Required review before any promotion discussion

- Diffusion must outperform the Gaussian baseline on test CRPS and joint
  trajectory scores.
- Review Fog/Mist Brier and reliability alongside CSI, recall, precision, and
  false-positive rate.
- Review event occurrence, onset, duration, and minimum-visibility errors; only
  complete observed trajectories enter these joint event metrics.
- The candidate remains separate if any pre-declared event has recall below
  0.60. No script in this workflow promotes or rewires the mainline.
