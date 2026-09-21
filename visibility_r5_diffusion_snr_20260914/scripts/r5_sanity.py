#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r5_model import R5Model
from r5_pipeline import VisibilityDataset, config, contract, cosine_schedule, prediction_batch, train_loss

ROOT = Path(__file__).resolve().parents[1]
cfg, stats = config(), contract("S1")
device = torch.device("cuda:0")
indices = np.load(ROOT / "contracts" / "quick_indices_S1.npy")
dataset = VisibilityDataset(cfg, "S1", "val", indices, x_path_override=Path(os.environ["R5_QUICK_X"]), local_rows=True)
loader = DataLoader(dataset, batch_size=cfg["quick_eval_batch_size"], num_workers=min(cfg["num_workers"], 4),
                    pin_memory=True, persistent_workers=True)
schedule = cosine_schedule(cfg["diffusion_steps"], device)
model = R5Model(cfg, "R5-Diffusion").to(device)
model.train()

sums = {"samples": 0, "total": 0.0, "gate": 0.0, "diffusion": 0.0,
        "boundary": 0.0, "boundary_snr": 0.0, "boundary_weight": 0.0}
first_x = None
for x, v, _ in loader:
    x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
    loss, parts = train_loss(model, model, x, v, stats, cfg, schedule)
    if first_x is None:
        first_x = x[:32].detach()
        loss.backward()
        model.zero_grad(set_to_none=True)
    count = len(v)
    sums["samples"] += count
    sums["total"] += float(loss) * count
    for key in parts:
        sums[key] += float(parts[key]) * count

count = sums.pop("samples")
means = {key: value / count for key, value in sums.items()}
scale_ratio = means["diffusion"] / max(means["boundary_snr"], 1e-12)
model.eval()
with torch.inference_mode():
    probs, _, _, v_samples, e_samples, _ = prediction_batch(model, first_x, 4, stats, cfg, schedule, reverse_steps=20)

modules = {
    "DynamicTokenizer": model.dynamic_tokenizer,
    "DynamicEncoder": model.dynamic_encoder,
    "ContextEncoder": model.context_encoder,
    "GateQuery": model.gate_query,
    "GateHead": model.gate_head,
    "ContinuousQuery": model.continuous_query,
    "DiffusionHead": model.continuous_head,
}
result = {
    "route": "R5-Diffusion",
    "parameters": sum(p.numel() for p in model.parameters()),
    "module_parameters": {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()},
    "objective_scale_check": {"samples": count, "means": means, "diffusion_to_snr_boundary_ratio": scale_ratio,
                              "within_one_order_of_magnitude": 0.1 <= scale_ratio <= 10.0,
                              "lambda_gate": cfg["lambda_gate"], "lambda_boundary": cfg["lambda_boundary"]},
    "loss_finite": all(np.isfinite(value) for value in means.values()),
    "probability_sum_error": float((probs.sum(1) - 1).abs().max()),
    "physical_support": {
        "E_min": cfg["extinction_constant"] / cfg["gate_vc_km"],
        "Vc": cfg["gate_vc_km"],
        "min_generated_E": float(e_samples.min()),
        "max_generated_V": float(v_samples.max()),
        "E_le_Emin_count": int((e_samples <= cfg["extinction_constant"] / cfg["gate_vc_km"]).sum()),
        "V_ge_Vc_count": int((v_samples >= cfg["gate_vc_km"]).sum()),
    },
}
(ROOT / "sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["loss_finite"] or result["probability_sum_error"] >= 1e-8:
    raise SystemExit(2)
if not result["objective_scale_check"]["within_one_order_of_magnitude"]:
    raise SystemExit(3)
if result["physical_support"]["E_le_Emin_count"] or result["physical_support"]["V_ge_Vc_count"]:
    raise SystemExit(4)
