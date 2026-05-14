#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a low-visibility binary gate from diagnostic sweep outputs.

This is a post-processing utility. It does not run model inference and does
not modify checkpoints. The intended workflow is:

  1. Run diagnose_lowvis_checkpoint_pm10_pm25.py on val and test.
  2. Select the binary gate threshold on val with a recall-heavy score.
  3. Apply the same threshold to the test sweep for an unbiased estimate.

The gate rule evaluated by diagnose_lowvis_checkpoint_pm10_pm25.py is:
  if sigmoid(low_vis_detector) > binary_th:
      predict Fog/Mist from fine-head argmax within classes 0/1
  else:
      predict Clear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_GATE_COLUMNS = {
    "binary_th",
    "Fog_R",
    "Mist_R",
    "Mist_P",
    "Mist_CSI",
    "low_vis_recall",
    "low_vis_precision",
    "low_vis_csi",
    "false_positive_rate",
}


def resolve_diag_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "binary_gate_multiclass_sweep.csv").exists():
        return path

    nested = path / path.name
    if (nested / "binary_gate_multiclass_sweep.csv").exists():
        return nested

    matches = list(path.glob("**/binary_gate_multiclass_sweep.csv"))
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise FileNotFoundError(f"No binary_gate_multiclass_sweep.csv found under {path}")
    raise RuntimeError(f"Multiple diagnostic directories found under {path}; pass one explicitly.")


def load_gate_sweep(diag_dir: Path) -> pd.DataFrame:
    path = diag_dir / "binary_gate_multiclass_sweep.csv"
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_GATE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df.copy()


def bounded_ratio(value: float, goal: float) -> float:
    if goal <= 0:
        return float(value)
    return float(min(max(value, 0.0) / goal, 1.0))


def score_gate_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["recall_heavy_score"] = (
        args.w_lowvis_recall * out["low_vis_recall"].map(lambda v: bounded_ratio(v, args.goal_lowvis_recall))
        + args.w_mist_recall * out["Mist_R"].map(lambda v: bounded_ratio(v, args.goal_mist_recall))
        + args.w_fog_recall * out["Fog_R"].map(lambda v: bounded_ratio(v, args.goal_fog_recall))
        + args.w_lowvis_csi * out["low_vis_csi"].map(lambda v: bounded_ratio(v, args.goal_lowvis_csi))
        + args.w_mist_csi * out["Mist_CSI"].map(lambda v: bounded_ratio(v, args.goal_mist_csi))
        + args.w_precision * out["low_vis_precision"].map(lambda v: bounded_ratio(v, args.goal_lowvis_precision))
    )

    fpr_excess = np.maximum(out["false_positive_rate"].to_numpy(dtype=float) - args.max_fpr, 0.0)
    out["fpr_penalty"] = args.w_fpr_penalty * fpr_excess / max(args.max_fpr, 1e-6)

    lv_prec_short = np.maximum(args.min_lowvis_precision - out["low_vis_precision"].to_numpy(dtype=float), 0.0)
    out["lowvis_precision_penalty"] = (
        args.w_precision_penalty * lv_prec_short / max(args.min_lowvis_precision, 1e-6)
    )

    mist_prec_short = np.maximum(args.min_mist_precision - out["Mist_P"].to_numpy(dtype=float), 0.0)
    out["mist_precision_penalty"] = (
        args.w_precision_penalty * mist_prec_short / max(args.min_mist_precision, 1e-6)
    )

    out["selection_score"] = (
        out["recall_heavy_score"]
        - out["fpr_penalty"]
        - out["lowvis_precision_penalty"]
        - out["mist_precision_penalty"]
    )
    return out.sort_values(
        ["selection_score", "low_vis_recall", "Mist_R", "low_vis_csi"],
        ascending=[False, False, False, False],
    )


def nearest_threshold_row(df: pd.DataFrame, threshold: float) -> pd.Series:
    idx = (df["binary_th"].astype(float) - float(threshold)).abs().idxmin()
    return df.loc[idx]


def row_to_dict(row: pd.Series) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select recall-heavy low-vis binary gate from diagnostics.")
    ap.add_argument("--val-dir", type=Path, help="Validation diagnostic directory.")
    ap.add_argument("--test-dir", type=Path, default=None, help="Optional test diagnostic directory.")
    ap.add_argument("--diag-dir", type=Path, default=None, help="Single diagnostic directory for exploratory scoring.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to val/test diag parent.")
    ap.add_argument("--dry-run", action="store_true", help="Print the selection without writing files.")

    ap.add_argument("--goal-lowvis-recall", type=float, default=0.75)
    ap.add_argument("--goal-mist-recall", type=float, default=0.40)
    ap.add_argument("--goal-fog-recall", type=float, default=0.50)
    ap.add_argument("--goal-lowvis-csi", type=float, default=0.22)
    ap.add_argument("--goal-mist-csi", type=float, default=0.10)
    ap.add_argument("--goal-lowvis-precision", type=float, default=0.20)
    ap.add_argument("--max-fpr", type=float, default=0.06)
    ap.add_argument("--min-lowvis-precision", type=float, default=0.12)
    ap.add_argument("--min-mist-precision", type=float, default=0.08)

    ap.add_argument("--w-lowvis-recall", type=float, default=0.35)
    ap.add_argument("--w-mist-recall", type=float, default=0.25)
    ap.add_argument("--w-fog-recall", type=float, default=0.20)
    ap.add_argument("--w-lowvis-csi", type=float, default=0.10)
    ap.add_argument("--w-mist-csi", type=float, default=0.05)
    ap.add_argument("--w-precision", type=float, default=0.05)
    ap.add_argument("--w-fpr-penalty", type=float, default=0.35)
    ap.add_argument("--w-precision-penalty", type=float, default=0.20)
    args = ap.parse_args()

    if args.diag_dir is not None:
        if args.val_dir is not None or args.test_dir is not None:
            raise ValueError("Use either --diag-dir or --val-dir/--test-dir, not both.")
        args.val_dir = args.diag_dir
    if args.val_dir is None:
        raise ValueError("Pass --val-dir or --diag-dir.")
    return args


def main() -> None:
    args = parse_args()
    val_dir = resolve_diag_dir(args.val_dir)
    test_dir: Optional[Path] = resolve_diag_dir(args.test_dir) if args.test_dir else None

    val_scored = score_gate_rows(load_gate_sweep(val_dir), args)
    selected_val = val_scored.iloc[0]
    selected_th = float(selected_val["binary_th"])

    result = {
        "selection_source": str(val_dir),
        "selected_binary_threshold": selected_th,
        "selection_objective": {
            "goal_lowvis_recall": args.goal_lowvis_recall,
            "goal_mist_recall": args.goal_mist_recall,
            "goal_fog_recall": args.goal_fog_recall,
            "max_fpr": args.max_fpr,
            "min_lowvis_precision": args.min_lowvis_precision,
            "min_mist_precision": args.min_mist_precision,
            "weights": {
                "lowvis_recall": args.w_lowvis_recall,
                "mist_recall": args.w_mist_recall,
                "fog_recall": args.w_fog_recall,
                "lowvis_csi": args.w_lowvis_csi,
                "mist_csi": args.w_mist_csi,
                "precision": args.w_precision,
                "fpr_penalty": args.w_fpr_penalty,
                "precision_penalty": args.w_precision_penalty,
            },
        },
        "selected_val_row": row_to_dict(selected_val),
    }

    rows = [selected_val]
    if test_dir is not None:
        test_df = load_gate_sweep(test_dir)
        test_row = nearest_threshold_row(test_df, selected_th)
        result["test_source"] = str(test_dir)
        result["applied_test_row"] = row_to_dict(test_row)
        rows.append(test_row)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    elif test_dir is not None:
        out_dir = test_dir / "lowvis_gate_selection"
    else:
        out_dir = val_dir / "lowvis_gate_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_scored.to_csv(out_dir / "val_gate_score_table.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "selected_gate_rows.csv", index=False)
    with open(out_dir / "selected_lowvis_gate.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote gate selection to {out_dir}")


if __name__ == "__main__":
    main()
