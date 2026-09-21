#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r5_model import R5Model
from r5_pipeline import VisibilityDataset, config, contract, cosine_schedule, prediction_batch, train_loss

ROOT = Path(__file__).resolve().parents[1]
cfg, stats = config(), contract("S2")
device = torch.device("cuda:0")
idx = np.load(ROOT / "contracts" / "quick_indices_S2.npy")[:32]
ds = VisibilityDataset(cfg, "S2", "val", idx)
x, v, _ = next(iter(DataLoader(ds, batch_size=32, num_workers=0)))
x, v = x.to(device), v.to(device)
schedule = cosine_schedule(cfg["diffusion_steps"], device)
result = {}
for route in ("R5-Gaussian", "R5-Diffusion"):
    model = R5Model(cfg, route).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, parts = train_loss(model, model, x, v, stats, cfg, schedule)
    loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        probs, p_gate, v_point, v_samples, e_samples, _ = prediction_batch(model, x, 4, stats, cfg, schedule)
    result[route] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss), "parts": {k: float(value) for k, value in parts.items()},
        "loss_finite": bool(torch.isfinite(loss)), "probability_sum_error": float((probs.sum(1) - 1).abs().max()),
        "min_E": float(e_samples.min()), "max_V": float(v_samples.max()),
        "E_min": cfg["extinction_constant"] / cfg["gate_vc_km"], "Vc": cfg["gate_vc_km"],
    }
    del model
    torch.cuda.empty_cache()
(ROOT / "sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not all(v["loss_finite"] and v["probability_sum_error"] < 1e-8 for v in result.values()):
    raise SystemExit(2)
