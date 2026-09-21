#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r5_model import R5CensoredFlowModel
from r5_pipeline import VisibilityDataset, config, contract, inference_batch, likelihood_from_output

ROOT = Path(__file__).resolve().parents[1]
cfg = config()
stats = contract("S1")
device = torch.device("cuda:0")
indices = np.load(ROOT / "contracts" / "quick_indices_S1.npy")[:2048]
dataset = VisibilityDataset(
    cfg, "S1", "val", indices, x_path_override=Path(os.environ["R5_QUICK_X"]), local_rows=True
)
loader = DataLoader(dataset, batch_size=1024, num_workers=4, pin_memory=True, persistent_workers=True)
model = R5CensoredFlowModel(cfg).to(device)
model.train()

losses = []
first = None
for x, v, _ in loader:
    x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
    output = model(x)
    loss, parts, _ = likelihood_from_output(model, output, v, cfg)
    losses.append({key: float(value) for key, value in {"total": loss, **parts}.items()})
    if first is None:
        first = (x[:64], v[:64])
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        model.zero_grad(set_to_none=True)

model.eval()
with torch.inference_mode():
    x, v = first
    output = model(x)
    params = output["flow_params"]
    base = torch.linspace(-11, 11, len(x), device=device)
    z, forward_logdet = model.flow.forward_transform(base, params)
    reconstructed, inverse_logdet = model.flow.inverse_transform(z, params)
    probs, v_point, pit, censor_cdf = inference_batch(model, output, v, cfg)
    z1 = torch.full_like(v, math.log(cfg["extinction_constant"] / 1.0))
    z05 = torch.full_like(v, math.log(cfg["extinction_constant"] / 0.5))
    f1 = model.flow.cdf(z1, params)
    f05 = model.flow.cdf(z05, params)
    z30 = torch.full_like(v, math.log(cfg["extinction_constant"] / cfg["censoring_limit_km"]))
    log_cdf30 = model.flow.log_cdf(z30, params)

module_names = [name.lower() for name, _ in model.named_modules()]
result = {
    "route": "R5-CensoredFlow",
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "architecture": {
        "spline_bins": cfg["spline_bins"],
        "spline_blocks": 1,
        "linear_tails": True,
        "gate_head": any("gate" in name for name in module_names),
        "cap_head": any("cap" in name for name in module_names),
        "diffusion": any("diffusion" in name for name in module_names),
        "classification_head": False,
    },
    "target": {
        "definition": "z=log(3.912/V_km)",
        "censoring_limit_km": cfg["censoring_limit_km"],
        "z30": stats["z_30km"],
        "V0_excluded_from_likelihood": True,
    },
    "losses": losses,
    "losses_finite": all(np.isfinite(value) for row in losses for value in row.values()),
    "gradients_finite": bool(gradients_finite),
    "rqs_inverse": {
        "max_reconstruction_error": float((reconstructed - base).abs().max()),
        "max_logdet_cancellation_error": float((forward_logdet + inverse_logdet).abs().max()),
        "linear_tail_error": float((z[(base.abs() >= cfg["spline_tail_bound"])] - base[(base.abs() >= cfg["spline_tail_bound"])]).abs().max()),
    },
    "cdf": {
        "log_cdf30_finite": bool(torch.isfinite(log_cdf30).all()),
        "F_z1_le_F_z0p5": bool((f1 <= f05).all()),
        "censor_probability_min": float(censor_cdf.min()),
        "censor_probability_max": float(censor_cdf.max()),
    },
    "event_probabilities": {
        "minimum": float(probs.min()),
        "maximum": float(probs.max()),
        "max_sum_error": float((probs.sum(1) - 1).abs().max()),
        "all_finite": bool(torch.isfinite(probs).all()),
    },
    "observed_visibility_median": {
        "minimum_km": float(v_point.min()),
        "maximum_km": float(v_point.max()),
        "all_finite": bool(torch.isfinite(v_point).all()),
    },
}
(ROOT / "sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

failed = (
    result["architecture"]["gate_head"] or result["architecture"]["cap_head"] or result["architecture"]["diffusion"]
    or not result["losses_finite"] or not result["gradients_finite"]
    or result["rqs_inverse"]["max_reconstruction_error"] > 1e-4
    or result["rqs_inverse"]["max_logdet_cancellation_error"] > 1e-4
    or result["rqs_inverse"]["linear_tail_error"] > 1e-7
    or not result["cdf"]["log_cdf30_finite"] or not result["cdf"]["F_z1_le_F_z0p5"]
    or result["event_probabilities"]["minimum"] < -1e-10
    or result["event_probabilities"]["max_sum_error"] > 1e-10
    or not result["event_probabilities"]["all_finite"]
    or not result["observed_visibility_median"]["all_finite"]
)
if failed:
    raise SystemExit(2)
