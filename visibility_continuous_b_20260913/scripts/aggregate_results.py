#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ROOT = Path("/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913")
OLD = Path("/public/home/putianshu/vis_mlp/visibility_continuous_20260912")
routes = {"R2b": (1, 2, 3), "R3b": (1, 2, 3), "R4b": (1, 2, 3, 4)}
records = {"R0": json.load(open(OLD / "eval" / "R0" / "metrics.json")), "routes": {}}

def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items(): flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
        out[prefix] = float(value)

for route, seeds in routes.items():
    per_seed = []
    for seed in seeds:
        metrics_path = ROOT / "runs" / route / f"seed{seed}" / "eval" / "metrics.json"
        summary_path = ROOT / "runs" / route / f"seed{seed}" / "training_summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            continue
        per_seed.append({"seed": seed, "metrics": json.load(open(metrics_path)), "training": json.load(open(summary_path))})
    flat = []
    for row in per_seed:
        item = {}; flatten("", row["metrics"], item); flat.append(item)
    common = sorted(set.intersection(*(set(x) for x in flat))) if flat else []
    aggregate = {k: {"mean": float(np.mean([x[k] for x in flat])), "std": float(np.std([x[k] for x in flat], ddof=1)) if len(flat) > 1 else 0.0} for k in common}
    records["routes"][route] = {"per_seed": per_seed, "aggregate": aggregate, "complete_seed_count": len(per_seed)}
with open(ROOT / "eval" / "multiseed_summary.json", "w") as f:
    json.dump(records, f, indent=2)
print(json.dumps({r: v["complete_seed_count"] for r, v in records["routes"].items()}))
