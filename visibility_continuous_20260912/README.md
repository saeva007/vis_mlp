# VisCast continuous visibility ablation

Independent R0–R4 implementation using the frozen p13 data/split contract and the formal p13 encoder architecture/initialization. It does not modify any original p13 artifact.

- `configs/experiment.json`: fixed data, baseline, optimizer, and route settings.
- `configs/data_contract.json` and `configs/masks_*.npz`: generated provenance and explicit cap/continuous/zero masks.
- `models/R*/`: best checkpoints and histories.
- `eval/R*/`: metrics and compressed test predictions.
- `logs/job_chain.tsv`: strict R0/sanity → R1 → R2 → R3 → R4 Slurm chain.
