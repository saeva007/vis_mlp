# Static RNN Low-Visibility Experiment Runbook

This runbook covers the compact paper-oriented neural model:

- dynamic meteorological sequence encoder: GRU or LSTM;
- static branch: one MLP over latitude/longitude/topography/static fields plus an embedding for vegetation type;
- optional FE branch: compact MLP for engineered physical features;
- fusion head: dynamic, static, and optional FE embeddings merged into a three-class classifier;
- optional auxiliary log-visibility regression head.

The implementation is intentionally not a replacement for the current complex
PMST model. It is a cleaner candidate main model and ablation driver for paper
experiments.

## Files

- `train_static_rnn_lowvis.py`: shared training script for the main GRU model,
  the LSTM comparison, and neural ablations.
- `sub_static_rnn_lowvis_main.slurm`: priority main-model launch script.
- `sub_static_rnn_lowvis_matrix.slurm`: Slurm array for the main ablation
  matrix.

## Data Assumptions

The script uses the current 12-hour stage datasets by default:

- Stage 1: `/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25`
- Stage 2: `/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2`

It does not rebuild `.npy` files. If data processing changes, rebuild with the
existing Stage 1 and Stage 2 data scripts first, then run this model script.

The default label thresholds are kept inside the existing dataset labels. The
training script reads labels from the saved arrays and does not redefine fog,
mist, or clear classes from scratch.

## Main Run

Submit the priority main experiment:

```bash
sbatch sub_static_rnn_lowvis_main.slurm
```

Useful overrides:

```bash
sbatch --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_RUN_ID=exp_static_gru_s1 sub_static_rnn_lowvis_main.slurm
sbatch --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_PRETRAINED_CKPT=/public/home/putianshu/vis_mlp/checkpoints/exp_static_gru_s1_S1_best_score.pt sub_static_rnn_lowvis_main.slurm
sbatch --export=ALL,LOWVIS_RNN_BATCH_SIZE=384,LOWVIS_RNN_GRAD_ACCUM=3 sub_static_rnn_lowvis_main.slurm
```

Default main architecture:

- `Static MLP + GRU`
- one GRU layer;
- hidden dimension 256;
- mean pooling over the 12-hour sequence;
- static hidden dimension 96;
- fusion hidden dimension 256;
- dropout 0.2;
- Focal Loss with class weights and recall-aware validation selection;
- threshold search is validation-only post-processing.

## Ablation Matrix

After the main model script is uploaded and a Stage 1 checkpoint is available,
submit the matrix:

```bash
sbatch --export=ALL,LOWVIS_RNN_PRETRAINED_CKPT=/public/home/putianshu/vis_mlp/checkpoints/<run_id>_S1_best_score.pt sub_static_rnn_lowvis_matrix.slurm
```

The array contains:

| Task ID | Experiment | Purpose |
|---:|---|---|
| 0 | `static_mlp_gru_main` | Main paper candidate |
| 1 | `static_mlp_lstm` | GRU vs LSTM dynamic encoder |
| 2 | `static_mlp_gru_no_fe` | Remove engineered physical features |
| 3 | `static_mlp_gru_no_pm` | Remove PM/aerosol information while preserving layout |
| 4 | `static_mlp_gru_aux_reg` | Test auxiliary log-visibility regression |
| 5 | `static_mlp_gru_attention` | Mean pooling vs attention pooling |
| 6 | `static_mlp_bigru` | One-direction vs bidirectional sequence context |
| 7 | `static_mlp_gru_csi_select` | Recall-oriented vs CSI-oriented validation selection |

The matrix defaults to `LOWVIS_RNN_MODE=s2`, because its intended use is
Stage 2 fine-tuning from a common Stage 1 checkpoint.

## Outputs and Naming

All outputs go to the checkpoint directory, default:

```text
/public/home/putianshu/vis_mlp/checkpoints
```

Files use the run id and stage tag:

- `<run_id>_S1_best_score.pt`
- `<run_id>_S1_latest.pt`
- `<run_id>_S1_history.json`
- `<run_id>_S2_phaseA_best_score.pt`
- `<run_id>_S2_phaseA_latest.pt`
- `<run_id>_S2_phaseA_history.json`
- `<run_id>_S2_phaseB_best_score.pt`
- `<run_id>_S2_phaseB_latest.pt`
- `<run_id>_S2_phaseB_history.json`
- `<run_id>_config.json`

Scaler files are stage- and layout-specific:

```text
robust_scaler_<run_id>_<stage>_w<window>_dyn<dynamic_dim>_<pm|nopm>.pkl
```

This naming keeps pretraining, fine-tuning, and ablations separable without
overwriting older stable runs.

## Stable Training Settings

The Slurm scripts follow the stable PMST S2/DDP pattern:

- 5 nodes by default;
- 4 DCUs per node;
- separate ports for `torchrun` rendezvous and the Python `TCPStore`;
- `NCCL_SHM_DISABLE=1`;
- `/dev/shm` cleanup before launch;
- `LOWVIS_RNN_NUM_WORKERS=0` by default to avoid mmap/NFS worker instability;
- warmup plus cosine learning-rate decay;
- gradient clipping;
- Stage 2 phase A trains only head/fusion/normalization/attention-style
  parameters, then phase B unfreezes all parameters with lower backbone LR;
- Stage 2 phase A uses DDP unused-parameter detection and Phase B rewraps DDP
  after unfreezing, matching the staged-freeze safety pattern in the stable
  PMST training stack;
- when `--aux-reg-weight 0` is used, the auxiliary regression head is kept in
  the zero-weight autograd path so DDP does not treat it as an abandoned branch;
- L2-SP regularization is available in both Stage 2 phases.

## Validation and Threshold Search

Training uses a stable classification loss. Operational metrics are used in
validation:

- Focal Loss plus class weights drives long-tail learning;
- optional recall boost encourages fog/mist probabilities without making
  recall itself the differentiable training target;
- threshold search is validation-only post-processing;
- `recall_csi` selection rewards fog/mist recall while retaining precision,
  clear recall, and CSI constraints.

The script stores the selected fog and mist thresholds in checkpoint metadata
and in the stage history JSON.

## Tree Baselines

XGBoost and LightGBM should stay as independent tabular baselines, not as
neural encoders. They should be run through the existing tree-model path after
the dataset paths and feature schema are checked against the current Stage 1
and Stage 2 data products.
