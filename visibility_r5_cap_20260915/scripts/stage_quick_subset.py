#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "configs" / "r5_cap.json").read_text(encoding="utf-8"))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=("S1", "S2"), required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()
indices = np.load(ROOT / "contracts" / f"quick_indices_{args.stage}.npy")
x = np.load(Path(cfg["data"][args.stage]) / "X_val.npy", mmap_mode="r")
np.save(args.output, np.asarray(x[indices]))
