#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

roots = {
    "S1": Path("/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25"),
    "S2": Path("/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"),
}
out = {}
for stage, root in roots.items():
    out[stage] = {}
    for split in ("train", "val", "test"):
        xp, yp = root / f"X_{split}.npy", root / f"y_{split}.npy"
        out[stage][split] = {
            "X_shape": list(np.load(xp, mmap_mode="r").shape) if xp.exists() else None,
            "y_shape": list(np.load(yp, mmap_mode="r").shape) if yp.exists() else None,
        }
print(json.dumps(out, indent=2))
