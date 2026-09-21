# Gate separability study

Fixed task: `gate = 1` for visibility at or above 28.9 km. The study uses the formal
R5-Diffusion S2 EMA encoder checkpoint and unchanged Tianji train/validation/test splits.

- A: frozen encoder plus linear sigmoid probe.
- B: frozen encoder plus `256 -> 64 -> 1` MLP sigmoid probe.
- C: the same MLP with end-to-end encoder fine-tuning.

All class metrics use the fixed probability cutoff 0.5. Test margins are evaluation-only.
