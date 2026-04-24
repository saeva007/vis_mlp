#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for per-init S2 dataset: train vs val distribution of lead_hours and valid hour,
and Mist / Fog / Clear fractions (aligned with PMST_net_test_12_s2_pm10_48h_timewin._y_raw_and_cls).

Usage:
  python diag_s2_per_init_mist_distribution.py
  python diag_s2_per_init_mist_distribution.py --data-dir /path/to/ml_dataset_fe_12h_48h_pm10_per_init_lead_split_uid27798

Environment:
  S2_DATA_DIR overrides default data directory.

Do not run on the login node for full scans (large meta CSV). Use:
  sbatch /public/home/putianshu/vis_mlp/train/sub_diag_s2_mist_distribution.slurm
(partition kshcexclu04).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get(
    "S2_DATA_DIR",
    "/public/home/putianshu/vis_mlp/ml_dataset_fe_12h_48h_pm10_validtime_monthtail",
)

# Match train script: Fog=0, Mist=1, Clear=2
FOG, MIST, CLEAR = 0, 1, 2


def vis_to_cls(y_raw):
    y = y_raw.astype(np.float64).copy()
    looks_km = bool(np.nanmax(y) < 100.0)
    if looks_km:
        y = y * 1000.0
    cls = np.zeros(len(y), dtype=np.int64)
    cls[y >= 500.0] = 1
    cls[y >= 1000.0] = 2
    return cls


def accumulate_split(
    y_path,
    meta_path,
    tag,
    chunksize,
):
    y_all = np.load(y_path, mmap_mode="r")
    n_y = len(y_all)
    # Aggregates
    total = 0
    n_fog = n_mist = n_clear = 0
    # lead bins: [0,12), [12,24), [24,36), [36,49)
    lead_edges = np.array([0.0, 12.0, 24.0, 36.0, 49.0])
    lead_counts = np.zeros((4, 3), dtype=np.int64)  # bin x class
    mist_in_lead = np.zeros(4, dtype=np.int64)
    total_in_lead = np.zeros(4, dtype=np.int64)
    # hour bins 0-23
    hour_counts = np.zeros((24, 3), dtype=np.int64)
    mist_in_hour = np.zeros(24, dtype=np.int64)
    total_in_hour = np.zeros(24, dtype=np.int64)
    sum_lead = 0.0
    sum_lead_mist = 0.0
    n_mist_rows = 0

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)

    offset = 0
    reader = pd.read_csv(
        meta_path,
        usecols=["time", "lead_hours"],
        chunksize=chunksize,
        dtype={"lead_hours": np.float32},
        parse_dates=["time"],
    )
    for chunk in reader:
        n = len(chunk)
        if offset + n > n_y:
            raise RuntimeError(
                f"{tag}: meta rows exceed y length: offset+n={offset+n} > n_y={n_y}"
            )
        y_chunk = np.asarray(y_all[offset : offset + n])
        cls = vis_to_cls(y_chunk)

        t = pd.to_datetime(chunk["time"], utc=False, errors="coerce")
        hour = t.dt.hour.fillna(-1).astype(np.int64).values
        lead = np.asarray(chunk["lead_hours"], dtype=np.float64)

        bc = np.bincount(cls, minlength=3)
        n_fog += int(bc[FOG])
        n_mist += int(bc[MIST])
        n_clear += int(bc[CLEAR])
        total += n
        sum_lead += float(np.sum(lead))
        m_mist = cls == MIST
        n_mist_rows += int(np.sum(m_mist))
        if m_mist.any():
            sum_lead_mist += float(np.sum(lead[m_mist]))

        lb = np.minimum((lead // 12.0).astype(np.int64), 3)
        for b in range(4):
            m = lb == b
            if not np.any(m):
                continue
            lead_counts[b] += np.bincount(cls[m], minlength=3)
            total_in_lead[b] += int(np.sum(m))
            mist_in_lead[b] += int(np.sum(m & m_mist))

        valid_h = (hour >= 0) & (hour < 24)
        if np.any(valid_h):
            hv = hour[valid_h]
            clv = cls[valid_h]
            idx = hv * 3 + clv
            part = np.bincount(idx, minlength=72).reshape(24, 3)
            hour_counts += part
            total_in_hour += np.bincount(hv, minlength=24)
            hv_m = hv[clv == MIST]
            if hv_m.size:
                mist_in_hour += np.bincount(hv_m, minlength=24)

        offset += n

    if offset != n_y:
        print(
            f"[WARN] {tag}: meta rows read={offset} != len(y)={n_y}",
            flush=True,
            file=sys.stderr,
        )

    mist_rate = n_mist / max(total, 1)
    mean_lead = sum_lead / max(total, 1)
    mean_lead_mist = sum_lead_mist / max(n_mist_rows, 1)

    return {
        "tag": tag,
        "total": total,
        "n_fog": n_fog,
        "n_mist": n_mist,
        "n_clear": n_clear,
        "mist_rate": mist_rate,
        "mean_lead_all": mean_lead,
        "mean_lead_mist_only": mean_lead_mist,
        "lead_edges": lead_edges,
        "lead_counts": lead_counts,
        "mist_in_lead": mist_in_lead,
        "total_in_lead": total_in_lead,
        "hour_counts": hour_counts,
        "mist_in_hour": mist_in_hour,
        "total_in_hour": total_in_hour,
    }


def _safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b.astype(np.float64), np.nan)


def print_report(tr, va):
    print("=" * 72)
    print("S2 per-init dataset: train vs val (meta + y alignment)")
    print("=" * 72)
    for d in (tr, va):
        print(
            f"\n[{d['tag']}] N={d['total']:,}  "
            f"Fog={d['n_fog']:,} ({100*d['n_fog']/d['total']:.3f}%)  "
            f"Mist={d['n_mist']:,} ({100*d['mist_rate']:.3f}%)  "
            f"Clear={d['n_clear']:,} ({100*d['n_clear']/d['total']:.3f}%)"
        )
        print(
            f"  mean lead (all samples): {d['mean_lead_all']:.3f} h  |  "
            f"mean lead (Mist only): {d['mean_lead_mist_only']:.3f} h"
        )

    print("\n--- Mist rate by lead bin [h] ---")
    edges = tr["lead_edges"]
    nonempty_tr = int(np.sum(tr["total_in_lead"] > 0))
    nonempty_va = int(np.sum(va["total_in_lead"] > 0))
    if nonempty_tr > 1 or nonempty_va > 1:
        print(
            "  NOTE: train/val samples fall in different lead bins (within-run split). "
            "Compare Mist% only where both N>0; disjoint bins imply strong covariate shift.",
            flush=True,
        )
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        dt = int(tr["total_in_lead"][i])
        dv = int(va["total_in_lead"][i])
        rt = _safe_div(
            np.array([tr["mist_in_lead"][i]], dtype=np.float64),
            np.array([max(tr["total_in_lead"][i], 1)], dtype=np.float64),
        )[0]
        rv = _safe_div(
            np.array([va["mist_in_lead"][i]], dtype=np.float64),
            np.array([max(va["total_in_lead"][i], 1)], dtype=np.float64),
        )[0]
        delta = ""
        if dt > 0 and dv > 0 and np.isfinite(rt) and np.isfinite(rv):
            delta = "  |  same-bin Δppt={:+.4f}".format(100.0 * (rv - rt))
        print(
            "  [{:4.0f},{:4.0f})  train Mist%={} (N={:,})  |  val Mist%={} (N={:,}){}".format(
                lo,
                hi,
                "{:.4f}".format(100.0 * rt) if dt > 0 else "   —   ",
                dt,
                "{:.4f}".format(100.0 * rv) if dv > 0 else "   —   ",
                dv,
                delta,
            ),
            flush=True,
        )

    print("\n--- Hourly Mist rate (valid local hour, top 8 by |Δppt| where both splits have support) ---")
    tr_mr = _safe_div(tr["mist_in_hour"].astype(np.float64), tr["total_in_hour"].astype(np.float64))
    va_mr = _safe_div(va["mist_in_hour"].astype(np.float64), va["total_in_hour"].astype(np.float64))
    diff = np.abs(va_mr - tr_mr)
    support = (tr["total_in_hour"] > 500) & (va["total_in_hour"] > 500)
    diff = np.where(support, diff, np.nan)
    order = np.argsort(np.nan_to_num(-diff, nan=-1.0))
    shown = 0
    for h in order:
        if shown >= 8:
            break
        if not np.isfinite(diff[h]):
            continue
        print(
            "  hour {:2d}  train Mist%={:.4f}  val Mist%={:.4f}  Δppt={:+.4f}".format(
                h,
                100.0 * tr_mr[h],
                100.0 * va_mr[h],
                100.0 * (va_mr[h] - tr_mr[h]),
            ),
            flush=True,
        )
        shown += 1
    if shown == 0:
        print(
            "  (No clock hour with >500 samples in BOTH train and val — "
            "likely disjoint valid-time hours under within-run split; compare lead bins above.)",
            flush=True,
        )

    print("\n--- Train vs val: overall Mist rate delta ---")
    delta_mist = va["mist_rate"] - tr["mist_rate"]
    print(f"  val_mist% - train_mist% = {100*delta_mist:+.4f} percentage points")

    print_split_recommendations(tr, va)


def print_split_recommendations(tr, va):
    """Heuristic recommendations after inspecting distributions (plan todo: decide-split-strategy)."""
    print("\n" + "=" * 72)
    print("Split-strategy notes (feasibility)")
    print("=" * 72)
    tr_mr = _safe_div(tr["mist_in_hour"].astype(np.float64), tr["total_in_hour"].astype(np.float64))
    va_mr = _safe_div(va["mist_in_hour"].astype(np.float64), va["total_in_hour"].astype(np.float64))
    lead_tr = _safe_div(tr["mist_in_lead"].astype(np.float64), tr["total_in_lead"].astype(np.float64))
    lead_va = _safe_div(va["mist_in_lead"].astype(np.float64), va["total_in_lead"].astype(np.float64))

    support = (tr["total_in_hour"] > 500) & (va["total_in_hour"] > 500)
    d_h = np.abs(va_mr - tr_mr)
    d_h = d_h[support]
    d_h = d_h[np.isfinite(d_h)]
    max_hour_diff = float(np.max(d_h)) if d_h.size else 0.0

    both = (tr["total_in_lead"] > 0) & (va["total_in_lead"] > 0)
    if np.any(both):
        d_l = np.abs(lead_va[both] - lead_tr[both])
        max_lead_diff = float(np.max(d_l)) if d_l.size else 0.0
    else:
        max_lead_diff = 0.0

    tr_bins = np.where(tr["total_in_lead"] > 0)[0]
    va_bins = np.where(va["total_in_lead"] > 0)[0]
    disjoint_lead = (
        tr_bins.size > 0
        and va_bins.size > 0
        and np.intersect1d(tr_bins, va_bins).size == 0
    )

    print(
        "If per-init *within-run* split creates eval shift, consider:\n"
        "  (1) Global split by valid `time` (month-tail or rolling blocks), like monthtail notebook — "
        "train and val both see mixed lead_hours and mixed local hours.\n"
        "  (2) Stratified holdout: stratify val on (init_hour mod 12, lead_bin) so val matches train.\n"
        "  (3) Keep 48h files but merge 00Z+12Z into longer valid-time series before windowing "
        "(recover 'two inits per day' diversity without label leakage).\n"
        "Quantitative hints from this run:"
    )
    print(
        "  max_bin |Δ Mist_rate| (only bins with BOTH train and val mass): {:.4f} ppt".format(
            100.0 * max_lead_diff
        )
    )
    print(
        "  max_hour |Δ Mist_rate| (hours with >500 samples in both): {:.4f} ppt".format(
            100.0 * max_hour_diff
        )
    )
    if disjoint_lead:
        print(
            "  => Train and val occupy DISJOINT lead bins — covariate shift; "
            "val Mist/Fog metrics are not comparable to training distribution on lead.",
            flush=True,
        )
    elif max_lead_diff > 0.002 or max_hour_diff > 0.005:
        print(
            "  => Non-negligible Mist rate differences across overlapping lead/hour bins — "
            "consider aligning val to train or using a global valid-time split.",
            flush=True,
        )
    else:
        print(
            "  => Overlapping bins look similar; if Mist_R still collapses on val, "
            "inspect Fog↔Mist confusion and calibration.",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description="Per-init S2 Mist/lead/hour diagnostics")
    ap.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory with y_train.npy, y_val.npy, meta_train.csv, meta_val.csv",
    )
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    tr = accumulate_split(
        os.path.join(data_dir, "y_train.npy"),
        os.path.join(data_dir, "meta_train.csv"),
        "train",
        args.chunksize,
    )
    va = accumulate_split(
        os.path.join(data_dir, "y_val.npy"),
        os.path.join(data_dir, "meta_val.csv"),
        "val",
        args.chunksize,
    )
    print_report(tr, va)


if __name__ == "__main__":
    main()
