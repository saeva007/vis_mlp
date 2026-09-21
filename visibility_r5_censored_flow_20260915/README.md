# VisCast R5-CensoredFlow

This independent experiment keeps the current R5 inputs and shared Transformer backbone, then replaces every Gate/Cap/diffusion/classification branch with one conditional scalar rational-quadratic spline flow.

- Target for uncensored observations: `z = log(3.912 / V_km)`, for `0 < V < 30 km`.
- Right-censoring in visibility becomes left-censoring in z: an observed 30 km contributes `-log F(z30|X)` with `z30=log(3.912/30)`.
- V=0 contributes no likelihood but remains in event evaluation.
- One 16-bin monotonic RQS with identity linear tails maps `u~N(0,1)` to z.
- The stable censored term uses `torch.special.log_ndtr` after analytic spline inversion; it does not use an epsilon repair.
- Event probabilities are analytic CDF intervals from the same flow; there is no sampling or threshold search.
- The only loss is the censored NLL. There is no Gate/Cap BCE, boundary/interval loss, focal loss, weighting, oversampling, or auxiliary classification loss.

Formal training is ERA5 S1 followed by Tianji S2. Each training run uses one node, 32 CPUs, four DCUs with DDP, batch 1024 per DCU (effective batch 4096), and 16 DataLoader workers in total. Checkpoints and validators are isolated under `runs/R5-CensoredFlow/seed1`.
