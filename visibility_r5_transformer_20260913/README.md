# VisCast R5 modular Transformer baseline

This directory is independent of p13 and R2/R3/R4. The first round uses only the existing point, static, PM, and engineered inputs.

- `PointDynamicEncoder`: 12 weather tokens, 6-layer pre-LN Transformer.
- `ContextEncoder`: separate station/static and aerosol/PM tokens.
- `TaskQuery`: independent Gate and continuous cross-attention queries.
- `GaussianContinuousHead`: heteroscedastic Gaussian in standardized supported z-space.
- `DiffusionContinuousHead`: four cross-attention decoder blocks, cosine DDPM, v-prediction, EMA.
- `PatchDynamicEncoder` and `AlphaEarthEncoder`: reserved token interfaces only; no patch or AlphaEarth data is used.

Training performs fixed-subset, forward-only quick validation every 1000 steps. It records Gate BCE, Gaussian NLL or diffusion denoising loss, interval/boundary loss, and the total validation objective; it never invokes reverse diffusion. Atomic checkpoints are saved every 2500 steps. S1 hands off immediately to S2 using the ready checkpoint with the lowest finite forward validation objective, loading EMA weights when present. S1 does not wait for sampling validation. S2 alone submits independent four-DCU full-screening validation jobs on the complete validation split with 64 draws and a 20-step DDIM sampler; candidate review and final test retain the configured high-precision path.
