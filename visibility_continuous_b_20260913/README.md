# VisCast physically constrained continuous visibility experiments

This directory contains the isolated R2b/R3b/R4b experiment family.  It reads
the existing p13 inputs and split contract but never writes to the completed
R0-R4 result directory.

- `E_min = 3.912 / 30 = 0.1304`
- R2b: Gate + deterministic bounded extinction.
- R3b: Gate + heteroscedastic Gaussian in log-offset extinction space.
- R4b: Gate + conditional DDPM in log-offset extinction space with EMA.
- Every formal run writes only below `runs/<route>/seed<seed>/`.
- Checkpoint selection is mean validation event AP, with mean CSI as tie-break.

Formal training is launched only after the independent 1/2/4-DCU throughput
benchmark selects the efficient device count and batch size.
