# Low-Visibility Hard Mist/Clear Workflow

Run commands from the remote repository checkout:

```bash
cd /public/home/putianshu/vis_mlp/train
export BASE=/public/home/putianshu/vis_mlp
export REPO=/public/home/putianshu/vis_mlp/train
export RUN_ID=exp_1778563813_pm10_more_temp_search_utc
export DATA_DIR=ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
export DIAGNOSTICS_ROOT=${BASE}/diagnostics
```

This workflow does not modify `PMST_net_test_11_s2_pm10.py`. It exports hard
cases from an existing checkpoint, fine-tunes through a new entry point, and
then selects a validation-derived binary gate.

## 1. Select The Baseline Gate

Run this after validation and test diagnostics both contain
`binary_gate_multiclass_sweep.csv`.

```bash
sbatch --export=ALL,RUN_ID=${RUN_ID},DIAGNOSTICS_ROOT=${DIAGNOSTICS_ROOT},\
HARD_CONSTRAINTS=1,MAX_FPR=0.045,MIN_LOWVIS_PRECISION=0.12,MIN_MIST_PRECISION=0.08,\
OUT_DIR=${DIAGNOSTICS_ROOT}/${RUN_ID}_lowvis_gate_selection \
  ${REPO}/sub_select_lowvis_gate_from_diagnostics.slurm
```

The selector writes the experiment id, selected threshold, validation row, and
test row to:

```bash
${DIAGNOSTICS_ROOT}/${RUN_ID}_lowvis_gate_selection/selected_lowvis_gate.json
```

## 2. Export Hard Samples

Export the train split for hard fine-tuning. The job is inference-only and uses
one full node with all 4 DCUs on that node.

```bash
sbatch --export=ALL,RUN_ID=${RUN_ID},SPLIT=train,DATA_DIR=${DATA_DIR},\
LOWVIS_GATE_TH=0.79,MAX_PER_POOL=0 \
  ${REPO}/sub_export_lowvis_hard_samples_pm10_pm25.slurm
```

Default output:

```bash
${BASE}/hard_samples/${RUN_ID}_train_hard_samples/
```

Key files:

```bash
hard_sample_pools.csv
hard_sample_pools.npz
hard_sample_pool_counts.csv
hard_sample_export_summary.json
```

The JSON summary records `run_id`, split, paths, thresholds, feature layout, and
pool counts.

## 3. Hard Mist/Clear Fine-Tune

Use the exported train hard-sample directory. This is a 5-node training job;
each node uses all 4 DCUs.

```bash
sbatch --export=ALL,INIT_RUN_ID=${RUN_ID},\
HARD_POOL_DIR=${BASE}/hard_samples/${RUN_ID}_train_hard_samples,\
LOWVIS_GATE_TH=0.79,TARGET_MAX_FPR=0.045,\
HARD_FT_RUN_ID=${RUN_ID}_hard_mist_clear_ft \
  ${REPO}/sub_PMST_net_s2_hard_mist_clear_ft.slurm
```

The fine-tune writes checkpoint metadata to:

```bash
${BASE}/checkpoints/${RUN_ID}_hard_mist_clear_ft_meta.json
```

That JSON records the fine-tune run id, initialization checkpoint, scaler, hard
pool summary, temperature, and the active hard-fine-tune config subset.

## 4. Diagnose Fine-Tuned Checkpoints

Use the new run id and the best fixed-gate checkpoint:

```bash
export FT_RUN_ID=${RUN_ID}_hard_mist_clear_ft

sbatch --export=ALL,RUN_ID=${FT_RUN_ID},SPLIT=val,DATA_DIR=${DATA_DIR},\
CKPT_PATH=checkpoints/${FT_RUN_ID}_S2_HardMistClearFT_best_score.pt,\
OUT_DIR=${DIAGNOSTICS_ROOT}/${FT_RUN_ID}_val_lowvis_diag,SKIP_FINE_GRID=1 \
  ${REPO}/sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm

sbatch --export=ALL,RUN_ID=${FT_RUN_ID},SPLIT=test,DATA_DIR=${DATA_DIR},\
CKPT_PATH=checkpoints/${FT_RUN_ID}_S2_HardMistClearFT_best_score.pt,\
OUT_DIR=${DIAGNOSTICS_ROOT}/${FT_RUN_ID}_test_lowvis_diag,SKIP_FINE_GRID=1 \
  ${REPO}/sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm
```

Then repeat the gate selection on `${FT_RUN_ID}` so the paper evaluation uses a
validation-selected threshold for the fine-tuned checkpoint.

## 5. No-Training Post-Hoc Gate Diagnosis

Use this when validation/test diagnostics already contain `_rank_parts`. It
does not run inference or training. It selects `fine_low_mass` and `hybrid_or`
rules on validation, then applies the same rule to test:

```bash
sbatch --export=ALL,RUN_ID=${RUN_ID},DIAGNOSTICS_ROOT=${DIAGNOSTICS_ROOT},\
MAX_FPR=0.045,MIN_LOWVIS_PRECISION=0.12,MIN_MIST_PRECISION=0.06,\
OUT_DIR=${DIAGNOSTICS_ROOT}/${RUN_ID}_posthoc_lowvis_gate_selection \
  ${REPO}/sub_select_posthoc_lowvis_gate_from_diagnostics.slurm
```

Key output:

```bash
${DIAGNOSTICS_ROOT}/${RUN_ID}_posthoc_lowvis_gate_selection/selected_posthoc_lowvis_gate.json
```

This README is a runbook only; performance numbers must come from the generated
diagnostic CSV/JSON files.
