#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "r5_censored_flow.json").read_text(encoding="utf-8"))


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
    limit, c = float(CFG["censoring_limit_km"]), CFG["extinction_constant"]
    for stage in ("S1", "S2"):
        root = Path(CFG["data"][stage])
        train = visibility(root / "y_train.npy")
        if np.any(train < 0) or np.any(train > limit):
            raise RuntimeError(f"{stage} visibility is outside the expected [0,{limit}] km observation support")
        uncensored = (train > 0) & (train < limit)
        censored = train == limit
        z = np.log(c / train[uncensored].astype(np.float64))
        contract = {
            "stage": stage,
            "data_dir": str(root),
            "train_samples": int(len(train)),
            "val_samples": int(len(visibility(root / "y_val.npy"))),
            "test_samples": int(len(visibility(root / "y_test.npy"))) if (root / "y_test.npy").exists() else 0,
            "censoring_limit_km": limit,
            "extinction_constant": c,
            "z_mean": float(z.mean(dtype=np.float64)),
            "z_std": float(z.std(dtype=np.float64)),
            "z_30km": float(np.log(c / limit)),
            "z_0p5km": float(np.log(c / 0.5)),
            "z_1km": float(np.log(c / 1.0)),
            "frequency_reversal_diagnostic_km": CFG["frequency_reversal_diagnostic_km"],
            "uncensored_train_samples": int(uncensored.sum()),
            "censored_30km_train_samples": int(censored.sum()),
            "zero_visibility_train_samples": int((train == 0).sum()),
        }
        atomic_json(out / f"{stage}.json", contract)

        val = visibility(root / "y_val.npy")
        strata = [val < 0.5, (val >= 0.5) & (val < 1.0), (val >= 1.0) & (val < limit), val == limit]
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
