# VisCast operating rules

This file defines the working rules for continuing the VisCast continuous/probabilistic visibility experiments.

## Sources of truth

1. Human-readable history and interpretation: `R5_EXPERIMENT_LOG.md`.
2. Machine-readable experiment registry: `R5_EXPERIMENT_REGISTRY.json`.
3. Actual experiment state: the config, checkpoint, log, result JSON/CSV and Slurm job under the paths recorded in the registry.
4. When a local `run_summary.json` conflicts with current cluster files, record the discrepancy in the ledger; do not silently rewrite or delete the old file.

## Preservation rules

- Never move, rename, overwrite or delete an existing experiment directory, config, checkpoint, prediction or result.
- New scientific definitions must use a new experiment directory and a new canonical display name.
- Never reuse one result file as a shared write target for concurrent jobs.
- Each run/seed must have independent logs, checkpoints, output directories and random seeds.
- Do not commit model checkpoints, datasets, caches, virtual environments, node modules or generated binary bundles to GitHub.

## Cluster operation

- The cluster is available through the configured SSH host `viscast-hpccube`.
- For a status request, take one bounded `squeue`/`sacct` snapshot. Do not start sleep loops, repeated polling or continuous log monitoring unless explicitly requested.
- Do not stop, resubmit or alter a running formal job unless explicitly authorized.
- A main training run owns one node. Resource allocation may request four DCUs without requiring all four to be used.
- Do not run hashes, whole-storage scans, historical Git audits or broad data rechecks for routine experiment work.
- Missing metrics may be recomputed only from already saved predictions when explicitly needed; do not rerun inference or training by default.

## Validation framework

- Training nodes perform forward/backward, optimizer, EMA and atomic checkpoint saves.
- Quick validation is fixed-subset, lightweight and synchronous; it monitors training only.
- Full validation is checkpoint-based, inference-only and asynchronous on another job/node.
- Validators must not run backward, update optimizer/EMA/normalization, or modify checkpoints.
- Distinct validators write distinct result files; aggregation happens after files are complete.
- S1→S2 handoff uses finite, non-sampling forward validation objective and does not wait for diffusion sampling validation.
- Full-validation/test sampling must batch or chunk the draw dimension; do not use a Python loop over individual draws.

## Scientific definitions that must remain distinct

- 28.9-km Gate: `P(V >= 28.9 km | X)`; a frequency-turning-point diagnostic label.
- 30-km Cap probability: `P(Y = 30 km | X)`; an observed point-mass probability used as a soft mixture weight.
- 30-km censored likelihood: for `Y=30`, only latent `V>=30` is known; there is no independent Gate/Cap head.
- Extinction is `E=3.912/V_km`.
- Event classification uses the unified three probabilities with argmax; threshold search is prohibited unless a new explicitly authorized experiment changes the protocol.
- `V=0` is excluded from continuous likelihood/loss but remains in event evaluation, with the existing excluding-`V=0` sensitivity protocol.

## Experiment-record update protocol

- Every new actual run is appended to both ledger files; do not create a competing experiment-record system.
- Record canonical display name, real paths, status, research question, baseline, input, backbone, output distribution, high-vis treatment, continuous transform, loss, exact changes, training protocol, results, conclusion and Slurm job IDs.
- Running/queued experiments must not be assigned guessed results or marked finished.
- Quick-validation metrics must be labeled as monitoring values and never presented as formal test results.
- Preserve failed/canceled attempts as operational history, but do not count retries or collision directories as new scientific experiments.

## Bootstrap prompt for another Codex computer

After cloning or pulling the repository, tell Codex to read `R5_EXPERIMENT_LOG.md`, `R5_EXPERIMENT_REGISTRY.json` and this file first; then verify only the explicitly relevant cluster paths and take at most one current Slurm snapshot before answering or acting.
