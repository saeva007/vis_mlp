#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for path in root.glob("dcu*_w*_b*.json"):
    try:
        row = json.load(open(path))
        if row.get("status") == "ok":
            rows[row["tag"]] = row
    except Exception:
        pass

required = ("dcu1_w16_b512", "dcu2_w16_b512", "dcu4_w16_b512")
if not all(k in rows for k in required):
    result = {"status": "incomplete", "missing": [k for k in required if k not in rows]}
else:
    one, two, four = (rows[k] for k in required)
    scale_2_over_1 = two["samples_per_second"] / one["samples_per_second"]
    scale_4_over_2 = four["samples_per_second"] / two["samples_per_second"]
    if scale_2_over_1 < 1.5:
        dcu = 1
    elif scale_4_over_2 < 1.2:
        dcu = 2
    else:
        dcu = 4
    worker_candidates = [rows[k] for k in ("dcu1_w12_b512", "dcu1_w16_b512") if k in rows]
    workers = max(worker_candidates, key=lambda x: x["samples_per_second"])["num_workers_total"]
    if dcu == 1:
        batch_candidates = [rows[k] for k in ("dcu1_w16_b512", "dcu1_w16_b1024", "dcu1_w16_b2048") if k in rows]
        batch_candidates.sort(key=lambda x: x["batch_size_per_rank"])
        chosen = batch_candidates[0]
        for candidate in batch_candidates[1:]:
            if candidate["samples_per_second"] >= chosen["samples_per_second"] * 1.05:
                chosen = candidate
        batch = chosen["batch_size_per_rank"]
    else:
        batch = 512
    result = {
        "status": "selected", "dcu_count": dcu, "num_workers_total": workers,
        "batch_size_per_rank": batch, "effective_batch_size": dcu * batch,
        "scale_2_over_1": scale_2_over_1, "scale_4_over_2": scale_4_over_2,
        "selection_rules": {"minimum_2_over_1": 1.5, "minimum_4_over_2": 1.2, "minimum_batch_gain": 1.05},
        "benchmark_rows": rows,
    }
with open(root / "selected_resources.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
