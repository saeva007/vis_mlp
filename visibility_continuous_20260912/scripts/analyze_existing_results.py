#!/usr/bin/env python3
"""Read-only diagnostics from saved test predictions; never imports/runs a model."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/public/home/putianshu/vis_mlp/visibility_continuous_20260912")
DATA = Path("/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2")
v = np.asarray(np.load(DATA / "y_test.npy", mmap_mode="r"), dtype=np.float64).reshape(-1)
if np.nanmax(v) >= 100:
    v /= 1000.0
n = len(v)
cap_true = np.isclose(v, 30.0, atol=1e-6)


def finite_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "finite_fraction": float(np.isfinite(x).mean()),
        "min": float(np.nanmin(x)),
        "p01": float(np.nanquantile(x, 0.01)),
        "median": float(np.nanmedian(x)),
        "mean": float(np.nanmean(x)),
        "p99": float(np.nanquantile(x, 0.99)),
        "max": float(np.nanmax(x)),
    }


visibility_bins = [
    ("0_0p5", lambda z: (z >= 0) & (z < 0.5)),
    ("0p5_1", lambda z: (z >= 0.5) & (z < 1)),
    ("1_2", lambda z: (z >= 1) & (z < 2)),
    ("2_5", lambda z: (z >= 2) & (z < 5)),
    ("5_10", lambda z: (z >= 5) & (z < 10)),
    ("10_20", lambda z: (z >= 10) & (z < 20)),
    ("20_30", lambda z: (z >= 20) & (z < 30)),
    ("30_cap", lambda z: np.isclose(z, 30.0, atol=1e-6)),
]
e_bins = [
    ("lt0p5", lambda z: (z > 0) & (z < 0.5)),
    ("0p5_1", lambda z: (z >= 0.5) & (z < 1)),
    ("1_5", lambda z: (z >= 1) & (z < 5)),
    ("5_15", lambda z: (z >= 5) & (z < 15)),
    ("15_30", lambda z: (z >= 15) & (z < 30)),
]

out = {
    "n": n,
    "y_test_sha256_expected": "cdfdfffa6904453445af34db7d755f083d43e7e9fe18791bc75a7a29fdd58ef7",
    "routes": {},
}

reference_index = None
for route in ("R1", "R2", "R3", "R4"):
    path = ROOT / "eval" / route / "test_predictions.npz"
    with np.load(path) as pred:
        keys = sorted(pred.files)
        index = np.asarray(pred["index"], dtype=np.int64)
        probs = np.asarray(pred["probs"], dtype=np.float64)
        pcap = np.asarray(pred["p_cap"], dtype=np.float64)
        vpoint = np.asarray(pred["v_point"], dtype=np.float64)
        emean = np.asarray(pred["e_mean"], dtype=np.float64)
        emedian = np.asarray(pred["e_median"], dtype=np.float64)
        if reference_index is None:
            reference_index = index.copy()
        same_index = bool(np.array_equal(index, reference_index))
        ordered_full = bool(np.array_equal(index, np.arange(n, dtype=np.int64)))
        route_out = {
            "keys": keys,
            "shape": {"index": list(index.shape), "probs": list(probs.shape)},
            "same_index_as_R1": same_index,
            "index_is_exact_arange": ordered_full,
            "probability_sum_max_abs_error": float(np.max(np.abs(probs.sum(axis=1) - 1.0))),
            "argmax_class_counts": np.bincount(probs.argmax(axis=1), minlength=3).astype(int).tolist(),
            "v_point_global": finite_stats(vpoint),
            "e_mean_global": finite_stats(emean),
            "visibility_bins": {},
            "extinction_bins": {},
        }
        if route != "R1":
            pred_cap = pcap >= 0.5
            tp = int(np.sum(pred_cap & cap_true)); fp = int(np.sum(pred_cap & ~cap_true)); fn = int(np.sum(~pred_cap & cap_true)); tn = int(np.sum(~pred_cap & ~cap_true))
            route_out["gate"] = {
                "accuracy_at_0p5": (tp + tn) / n,
                "precision_at_0p5": tp / (tp + fp) if tp + fp else 0.0,
                "recall_at_0p5": tp / (tp + fn) if tp + fn else 0.0,
                "AUROC": float(roc_auc_score(cap_true, pcap)),
                "AP": float(average_precision_score(cap_true, pcap)),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "p_cap_true_cap_mean": float(pcap[cap_true].mean()),
                "p_cap_noncap_mean": float(pcap[~cap_true].mean()),
            }
        for name, fn in visibility_bins:
            mask = fn(v)
            err = vpoint[mask] - v[mask]
            row = {
                "n": int(mask.sum()),
                "truth_mean": float(v[mask].mean()),
                "prediction_mean": float(vpoint[mask].mean()),
                "prediction_median": float(np.median(vpoint[mask])),
                "MAE": float(np.abs(err).mean()),
                "bias": float(err.mean()),
            }
            if route == "R4":
                eq05 = np.asarray(pred["e_q05"], dtype=np.float64)[mask]
                eq95 = np.asarray(pred["e_q95"], dtype=np.float64)[mask]
                e50 = emedian[mask]
                vlo = 3.912 / eq95
                vhi = 3.912 / eq05
                truth = v[mask]
                row.update({
                    "conditional_visibility_median": float(np.median(3.912 / e50)),
                    "conditional_interval_q05_q95_lower_median": float(np.median(vlo)),
                    "conditional_interval_q05_q95_upper_median": float(np.median(vhi)),
                    "conditional_interval_coverage": float(np.mean((truth >= vlo) & (truth <= vhi))),
                })
            route_out["visibility_bins"][name] = row
        for name, fn in e_bins:
            mask = fn(v)
            true_e = 3.912 / v[mask]
            pred_e = emean[mask]
            route_out["extinction_bins"][name] = {
                "n": int(mask.sum()),
                "true_E_mean": float(true_e.mean()),
                "true_E_median": float(np.median(true_e)),
                "pred_E_mean": float(pred_e.mean()),
                "pred_E_median": float(np.median(pred_e)),
                "E_MAE": float(np.abs(pred_e - true_e).mean()),
                "E_bias": float((pred_e - true_e).mean()),
                "pred_over_true_mean_ratio": float(pred_e.mean() / true_e.mean()),
                "pred_E_p99": float(np.quantile(pred_e, 0.99)),
                "pred_E_max": float(pred_e.max()),
            }
        metrics = json.load(open(ROOT / "eval" / route / "metrics.json"))
        draws = int(metrics["draws"])
        total_draws = n * draws
        route_out["sampling"] = {
            "draws": draws,
            "rejection_counter": int(metrics["positive_resample_rejections"]),
            "rejection_counter_over_initial_draws": float(metrics["positive_resample_rejections"] / total_draws),
            "fallback_count": int(metrics["positive_fallback_count"]),
            "fallback_over_initial_draws": float(metrics["positive_fallback_count"] / total_draws),
        }
        out["routes"][route] = route_out

print(json.dumps(out, indent=2, allow_nan=False))
