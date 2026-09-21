# VisCast R5-Cap: 30 km point-mass mixture

This directory is independent of p13 and R2/R3/R4. The first round uses only the existing point, static, PM, and engineered inputs.

- `DynamicTokenizer`: 12 point-weather tokens with temporal position embeddings.
- `DynamicEncoder`: 6-layer, 8-head pre-LN Transformer preserving `[B,12,256]`.
- `ContextEncoder`: separate station/static and aerosol/PM tokens.
- `TaskQuery`: independent Cap and continuous cross-attention queries.
- `CapHead`: LayerNorm, 256→128, SiLU, 128→1 for `P(V=30 km|X)`.
- `DiffusionHead`: four cross-attention residual blocks, cosine DDPM, v-prediction, EMA.
- `PatchDynamicTokenizer`, `AlphaEarthEncoder`, and `GaussianHead`: reserved token/head interfaces only.

The continuous target is standardized `z=log(3.912/V-3.912/30)` for `0<V<30 km`. Exact 30 km observations are handled by an unweighted Cap BCE over all samples and excluded from the continuous loss. Boundary supervision at 0.5 and 1 km comes from the same recovered clean latent. Positive/negative boundary BCE terms are balanced separately and weighted per sample by `alpha_bar(t)=SNR/(1+SNR)`. `lambda_cap=lambda_boundary=1`.

Training performs fixed-subset, forward-only quick validation every 1000 steps and atomic checkpoints every 2500 steps. S1 hands off immediately to S2 using the ready EMA checkpoint with the lowest finite forward validation objective; S1 never waits for sampling validation. S2 alone submits asynchronous full validation. Final event probabilities use the Cap probability only as a soft mixture weight. Metrics include the three event levels, Cap diagnostics, CRPS/continuous error, the true 500–1000 m probability/argmax breakdown, and physical-support violation counts.
