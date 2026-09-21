#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "r5.json").read_text(encoding="utf-8"))


def visibility(path):
    v = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if len(v) and float(np.nanmax(v)) >= 100.0:
        v = v / 1000.0
    return v


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    out = ROOT / "contracts"
    out.mkdir(parents=True, exist_ok=True)
    breakpoint = json.loads(Path(CFG["gate_breakpoint_source"]).read_text(encoding="utf-8"))
    vc, c = float(breakpoint["Vc_km"]), CFG["extinction_constant"]
    if vc != float(CFG["gate_vc_km"]):
        raise RuntimeError(f"configured Vc={CFG['gate_vc_km']} differs from gate_breakpoint Vc={vc}")
    emin = c / vc
    for stage in ("S1", "S2"):
        root = Path(CFG["data"][stage])
        train = visibility(root / "y_train.npy")
        cont = (train > 0) & (train < vc)
        z = np.log(c / train[cont].astype(np.float64) - emin)
        contract = {
            "stage": stage,
            "data_dir": str(root),
            "train_samples": int(len(train)),
            "val_samples": int(len(visibility(root / "y_val.npy"))),
            "test_samples": int(len(visibility(root / "y_test.npy"))) if (root / "y_test.npy").exists() else 0,
            "gate_vc_km": vc,
            "extinction_constant": c,
            "e_min": emin,
            "z_mean": float(z.mean(dtype=np.float64)),
            "z_std": float(z.std(dtype=np.float64)),
            "z_0p5km": float(np.log(c / 0.5 - emin)),
            "z_1km": float(np.log(c / 1.0 - emin)),
            "z_norm_0p5km": float((np.log(c / 0.5 - emin) - z.mean(dtype=np.float64)) / z.std(dtype=np.float64)),
            "z_norm_1km": float((np.log(c / 1.0 - emin) - z.mean(dtype=np.float64)) / z.std(dtype=np.float64)),
            "gate_breakpoint_source": CFG["gate_breakpoint_source"],
            "gate_region_mean_visibility_km": float(train[train >= vc].mean()),
            "continuous_train_samples": int(cont.sum()),
            "zero_visibility_train_samples": int((train == 0).sum()),
        }
        atomic_json(out / f"{stage}.json", contract)

        val = visibility(root / "y_val.npy")
        strata = [val < 0.5, (val >= 0.5) & (val < 1.0), (val >= 1.0) & (val < vc), val >= vc]
        rng = np.random.default_rng(20260913)
        per = CFG["quick_val_samples"] // len(strata)
        chosen = []
        counts = []
        for mask in strata:
            pool = np.flatnonzero(mask)
            take = min(per, len(pool))
            picked = rng.choice(pool, size=take, replace=False)
            chosen.append(picked)
            counts.append(int(take))
        indices = np.sort(np.concatenate(chosen).astype(np.int64))
        tmp = out / f"quick_indices_{stage}.tmp.npy"
        np.save(tmp, indices)
        os.replace(tmp, out / f"quick_indices_{stage}.npy")
        contract["quick_validation"] = {"samples": int(len(indices)), "stratum_counts": counts, "seed": 20260913}
        atomic_json(out / f"{stage}.json", contract)
        print(json.dumps(contract))


if __name__ == "__main__":
    main()
