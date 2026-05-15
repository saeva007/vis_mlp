#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select no-training low-visibility gates from saved diagnostic predictions.

This utility reads the `_rank_parts/rank_*.npz` files written by
diagnose_lowvis_checkpoint_pm10_pm25.py. It does not run model inference,
does not train, and does not touch checkpoints.

Rules evaluated:

  fine_low_mass:
      low = p_fog + p_mist >= fine_low_mass_th
      if low: predict Fog/Mist by max(p_fog, p_mist), else Clear

  hybrid_or:
      low = sigmoid(low_vis_detector) >= binary_th
            OR p_fog + p_mist >= fine_low_mass_th
      if low: predict Fog/Mist by max(p_fog, p_mist), else Clear

Thresholds are selected on validation diagnostics, then the exact same rule is
applied to the optional test diagnostics for an unbiased estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CLASS_NAMES = ("Fog", "Mist", "Clear")
METHOD_FINE = "fine_low_mass"
METHOD_HYBRID = "hybrid_or"


def resolve_diag_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "_rank_parts").is_dir():
        return path

    nested = path / path.name
    if (nested / "_rank_parts").is_dir():
        return nested

    matches = sorted(path.glob("**/_rank_parts/rank_*.npz"))
    parents = sorted({p.parent.parent for p in matches})
    if len(parents) == 1:
        return parents[0]
    if not parents:
        raise FileNotFoundError(
            f"No _rank_parts/rank_*.npz files found under {path}. "
            "Run diagnose_lowvis_checkpoint_pm10_pm25.py first; this selector "
            "needs saved probabilities, not only CSV metric tables."
        )
    raise RuntimeError(f"Multiple diagnostic directories found under {path}; pass one explicitly.")


def load_diag_predictions(diag_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    part_dir = diag_dir / "_rank_parts"
    files = sorted(part_dir.glob("rank_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rank_*.npz files in {part_dir}")

    probs_l: List[np.ndarray] = []
    low_l: List[np.ndarray] = []
    y_l: List[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            for key in ("probs", "low_prob", "y_cls"):
                if key not in z.files:
                    raise KeyError(f"{path} missing required array `{key}`")
            probs_l.append(np.asarray(z["probs"], dtype=np.float32))
            low_l.append(np.asarray(z["low_prob"], dtype=np.float32).reshape(-1))
            y_l.append(np.asarray(z["y_cls"], dtype=np.int64).reshape(-1))

    probs = np.concatenate(probs_l, axis=0)
    low_prob = np.concatenate(low_l, axis=0)
    y_cls = np.concatenate(y_l, axis=0)
    valid = (y_cls >= 0) & (y_cls <= 2)
    probs = probs[valid]
    low_prob = low_prob[valid]
    y_cls = y_cls[valid]

    summary = {
        "diag_dir": str(diag_dir),
        "rank_parts_dir": str(part_dir),
        "rank_files": [str(p) for p in files],
        "n": int(y_cls.size),
        "class_counts": {name: int((y_cls == idx).sum()) for idx, name in enumerate(CLASS_NAMES)},
    }
    return probs, low_prob, y_cls, summary


def parse_grid(value: str, default: np.ndarray) -> np.ndarray:
    text = (value or "").strip()
    if not text:
        return default
    if ":" in text:
        parts = [float(x) for x in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Grid range must be start:stop:step, got {value!r}")
        start, stop, step = parts
        if step <= 0:
            raise ValueError(f"Grid step must be positive, got {value!r}")
        grid = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    else:
        grid = np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float64)
    grid = np.unique(np.round(grid, 6))
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]
    if grid.size == 0:
        raise ValueError(f"Empty grid from {value!r}")
    return grid


def default_grid() -> np.ndarray:
    return np.round(np.linspace(0.01, 0.99, 99), 4)


def shifted_threshold_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # 0 means "passes no threshold"; 1..len(grid) means highest passed
    # threshold index + 1.
    idx = np.searchsorted(grid, values, side="right").astype(np.int32)
    return np.clip(idx, 0, len(grid))


def class_support(y_true: np.ndarray) -> np.ndarray:
    return np.bincount(y_true.astype(np.int64), minlength=3).astype(np.int64)


def confusion_from_selected(selected_by_true_lowclass: np.ndarray, support: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[:, :2] = np.rint(selected_by_true_lowclass).astype(np.int64)
    low_selected = cm[:, :2].sum(axis=1)
    cm[:, 2] = support - low_selected
    return cm


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    n = float(cm.sum())
    out: Dict[str, float] = {"n": n, "accuracy": safe_div(float(np.trace(cm)), n)}
    for cid, name in enumerate(CLASS_NAMES):
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        out[f"{name}_P"] = safe_div(tp, tp + fp)
        out[f"{name}_R"] = safe_div(tp, tp + fn)
        out[f"{name}_CSI"] = safe_div(tp, tp + fp + fn)
        out[f"{name}_FAR"] = safe_div(fp, tp + fp)
        out[f"{name}_support"] = float(cm[cid, :].sum())
        out[f"pred_{name.lower()}"] = float(cm[:, cid].sum())

    low_tp = float(cm[:2, :2].sum())
    low_fp = float(cm[2, :2].sum())
    low_fn = float(cm[:2, 2].sum())
    clear_support = float(cm[2, :].sum())
    out["low_vis_precision"] = safe_div(low_tp, low_tp + low_fp)
    out["low_vis_recall"] = safe_div(low_tp, low_tp + low_fn)
    out["low_vis_csi"] = safe_div(low_tp, low_tp + low_fp + low_fn)
    out["false_positive_rate"] = safe_div(low_fp, clear_support)
    out["balanced_acc"] = float(np.mean([out["Fog_R"], out["Mist_R"], out["Clear_R"]]))
    out["mist_to_clear_rate"] = safe_div(float(cm[1, 2]), float(cm[1, :].sum()))
    return out


def target_achievement(row: Dict[str, float]) -> float:
    return float(
        min(row["Fog_R"] / 0.65, 1.0) * 0.30
        + min(row["Mist_R"] / 0.75, 1.0) * 0.30
        + min(row["accuracy"] / 0.95, 1.0) * 0.25
        + min(row["low_vis_precision"] / 0.20, 1.0) * 0.10
        + min((1.0 - row["false_positive_rate"]) / 0.60, 1.0) * 0.05
    )


def add_selection_scores(row: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    def bounded(value: float, goal: float) -> float:
        return min(max(float(value), 0.0) / max(float(goal), 1e-6), 1.0)

    score = (
        args.w_lowvis_recall * bounded(float(row["low_vis_recall"]), args.goal_lowvis_recall)
        + args.w_mist_recall * bounded(float(row["Mist_R"]), args.goal_mist_recall)
        + args.w_fog_recall * bounded(float(row["Fog_R"]), args.goal_fog_recall)
        + args.w_lowvis_csi * bounded(float(row["low_vis_csi"]), args.goal_lowvis_csi)
        + args.w_mist_csi * bounded(float(row["Mist_CSI"]), args.goal_mist_csi)
        + args.w_precision * bounded(float(row["low_vis_precision"]), args.goal_lowvis_precision)
    )
    fpr_excess = max(0.0, float(row["false_positive_rate"]) - args.max_fpr)
    low_p_short = max(0.0, args.min_lowvis_precision - float(row["low_vis_precision"]))
    mist_p_short = max(0.0, args.min_mist_precision - float(row["Mist_P"]))

    row["target_achievement"] = target_achievement(row)  # type: ignore[arg-type]
    row["recall_heavy_score"] = float(score)
    row["fpr_penalty"] = float(args.w_fpr_penalty * fpr_excess / max(args.max_fpr, 1e-6))
    row["lowvis_precision_penalty"] = float(
        args.w_precision_penalty * low_p_short / max(args.min_lowvis_precision, 1e-6)
    )
    row["mist_precision_penalty"] = float(
        args.w_precision_penalty * mist_p_short / max(args.min_mist_precision, 1e-6)
    )
    row["selection_score"] = float(
        row["recall_heavy_score"]
        - row["fpr_penalty"]
        - row["lowvis_precision_penalty"]
        - row["mist_precision_penalty"]
    )
    row["constraint_fpr_ok"] = bool(float(row["false_positive_rate"]) <= args.max_fpr)
    row["constraint_lowvis_precision_ok"] = bool(
        float(row["low_vis_precision"]) >= args.min_lowvis_precision
    )
    row["constraint_mist_precision_ok"] = bool(float(row["Mist_P"]) >= args.min_mist_precision)
    row["constraint_satisfied"] = bool(
        row["constraint_fpr_ok"] and row["constraint_lowvis_precision_ok"] and row["constraint_mist_precision_ok"]
    )
    return row


def row_sort_key(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    return (
        float(row["selection_score"]),
        float(row["low_vis_recall"]),
        float(row["Mist_R"]),
        float(row["low_vis_csi"]),
    )


def select_row(rows: List[Dict[str, object]], args: argparse.Namespace) -> Tuple[Dict[str, object], str, bool, Optional[str]]:
    if args.hard_constraints:
        constrained = [r for r in rows if bool(r["constraint_satisfied"])]
        if constrained:
            return max(constrained, key=row_sort_key), "hard_constraints", True, None
        fallback = (
            "no validation rows satisfied hard constraints: "
            f"false_positive_rate <= {args.max_fpr}, "
            f"low_vis_precision >= {args.min_lowvis_precision}, "
            f"Mist_P >= {args.min_mist_precision}; selected by soft-penalty score"
        )
        return max(rows, key=row_sort_key), "soft_penalty_fallback", False, fallback
    selected = max(rows, key=row_sort_key)
    return selected, "soft_penalty", bool(selected["constraint_satisfied"]), None


def histogram_1d(shifted_idx: np.ndarray, y_true: np.ndarray, low_class: np.ndarray, n_grid: int) -> np.ndarray:
    cat = y_true.astype(np.int64) * 2 + low_class.astype(np.int64)
    flat = cat * (n_grid + 1) + shifted_idx.astype(np.int64)
    counts = np.bincount(flat, minlength=6 * (n_grid + 1)).reshape(3, 2, n_grid + 1)
    return counts.astype(np.int64)


def rows_from_1d_gate(
    *,
    method: str,
    score_name: str,
    score_values: np.ndarray,
    grid: np.ndarray,
    y_true: np.ndarray,
    low_class: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    support = class_support(y_true)
    shifted = shifted_threshold_index(score_values, grid)
    counts = histogram_1d(shifted, y_true, low_class, len(grid))
    ge = counts[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]

    rows: List[Dict[str, object]] = []
    for i, th in enumerate(grid):
        selected = ge[:, :, i + 1]
        cm = confusion_from_selected(selected, support)
        row: Dict[str, object] = {
            "method": method,
            "name": score_name,
            "fine_low_mass_th": float(th) if method == METHOD_FINE else None,
            "binary_th": None,
            "hybrid_rule": None,
        }
        row.update(metrics_from_cm(cm))
        rows.append(add_selection_scores(row, args))
    return rows


def rows_from_hybrid_or(
    *,
    low_prob: np.ndarray,
    fine_low_mass: np.ndarray,
    binary_grid: np.ndarray,
    fine_grid: np.ndarray,
    y_true: np.ndarray,
    low_class: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    support = class_support(y_true)
    bin_shift = shifted_threshold_index(low_prob, binary_grid)
    fine_shift = shifted_threshold_index(fine_low_mass, fine_grid)
    n_bin = len(binary_grid)
    n_fine = len(fine_grid)

    bin_counts = histogram_1d(bin_shift, y_true, low_class, n_bin)
    fine_counts = histogram_1d(fine_shift, y_true, low_class, n_fine)
    bin_ge = bin_counts[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]
    fine_ge = fine_counts[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]

    cat = y_true.astype(np.int64) * 2 + low_class.astype(np.int64)
    flat = ((cat * (n_bin + 1) + bin_shift.astype(np.int64)) * (n_fine + 1)) + fine_shift.astype(np.int64)
    both_counts = np.bincount(
        flat, minlength=6 * (n_bin + 1) * (n_fine + 1)
    ).reshape(3, 2, n_bin + 1, n_fine + 1)
    both_ge = both_counts[:, :, ::-1, ::-1].cumsum(axis=2).cumsum(axis=3)[:, :, ::-1, ::-1]

    rows: List[Dict[str, object]] = []
    for i, b_th in enumerate(binary_grid):
        for j, f_th in enumerate(fine_grid):
            selected = bin_ge[:, :, i + 1] + fine_ge[:, :, j + 1] - both_ge[:, :, i + 1, j + 1]
            cm = confusion_from_selected(selected, support)
            row: Dict[str, object] = {
                "method": METHOD_HYBRID,
                "name": "hybrid_or_binary_or_fine_low_mass",
                "binary_th": float(b_th),
                "fine_low_mass_th": float(f_th),
                "hybrid_rule": "binary_low_prob>=binary_th OR p_fog+p_mist>=fine_low_mass_th",
            }
            row.update(metrics_from_cm(cm))
            rows.append(add_selection_scores(row, args))
    return rows


def build_candidate_rows(
    probs: np.ndarray,
    low_prob: np.ndarray,
    y_true: np.ndarray,
    binary_grid: np.ndarray,
    fine_grid: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    fine_low_mass = np.clip(probs[:, 0].astype(np.float64) + probs[:, 1].astype(np.float64), 0.0, 1.0)
    low_class = np.where(probs[:, 0] >= probs[:, 1], 0, 1).astype(np.int64)
    rows = rows_from_1d_gate(
        method=METHOD_FINE,
        score_name="fine_low_mass_gate",
        score_values=fine_low_mass,
        grid=fine_grid,
        y_true=y_true,
        low_class=low_class,
        args=args,
    )
    rows.extend(
        rows_from_hybrid_or(
            low_prob=low_prob,
            fine_low_mass=fine_low_mass,
            binary_grid=binary_grid,
            fine_grid=fine_grid,
            y_true=y_true,
            low_class=low_class,
            args=args,
        )
    )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def nearest_test_row(rows: List[Dict[str, object]], selected: Dict[str, object]) -> Dict[str, object]:
    method = str(selected["method"])
    candidates = [r for r in rows if r["method"] == method]
    if not candidates:
        raise ValueError(f"No test rows for method {method}")
    if method == METHOD_FINE:
        f_th = float(selected["fine_low_mass_th"])
        return min(candidates, key=lambda r: abs(float(r["fine_low_mass_th"]) - f_th))
    b_th = float(selected["binary_th"])
    f_th = float(selected["fine_low_mass_th"])
    return min(
        candidates,
        key=lambda r: (
            abs(float(r["binary_th"]) - b_th),
            abs(float(r["fine_low_mass_th"]) - f_th),
        ),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None:
        return None
    return value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select no-training fine_low_mass and hybrid low-vis gates.")
    ap.add_argument("--val-dir", type=Path, help="Validation diagnostic directory with _rank_parts.")
    ap.add_argument("--test-dir", type=Path, default=None, help="Optional test diagnostic directory.")
    ap.add_argument("--diag-dir", type=Path, default=None, help="Single diagnostic directory for exploratory scoring.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    ap.add_argument("--experiment-id", default="", help="Experiment id written into JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Print selections without writing files.")
    ap.add_argument("--hard-constraints", action="store_true", help="Select only rows satisfying constraints when possible.")

    ap.add_argument("--binary-grid", default="", help="Hybrid binary threshold grid: comma list or start:stop:step.")
    ap.add_argument("--fine-grid", default="", help="Fine low-mass threshold grid: comma list or start:stop:step.")
    ap.add_argument("--top-k", type=int, default=25, help="Rows per method written to top_val_posthoc_gate_rows.csv.")

    ap.add_argument("--goal-lowvis-recall", type=float, default=0.75)
    ap.add_argument("--goal-mist-recall", type=float, default=0.40)
    ap.add_argument("--goal-fog-recall", type=float, default=0.50)
    ap.add_argument("--goal-lowvis-csi", type=float, default=0.22)
    ap.add_argument("--goal-mist-csi", type=float, default=0.10)
    ap.add_argument("--goal-lowvis-precision", type=float, default=0.20)
    ap.add_argument("--max-fpr", type=float, default=0.045)
    ap.add_argument("--min-lowvis-precision", type=float, default=0.12)
    ap.add_argument("--min-mist-precision", type=float, default=0.06)

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
    binary_grid = parse_grid(args.binary_grid, default_grid())
    fine_grid = parse_grid(args.fine_grid, default_grid())
    val_dir = resolve_diag_dir(args.val_dir)
    test_dir = resolve_diag_dir(args.test_dir) if args.test_dir else None

    val_probs, val_low, val_y, val_summary = load_diag_predictions(val_dir)
    val_rows = build_candidate_rows(val_probs, val_low, val_y, binary_grid, fine_grid, args)
    val_rows_sorted = sorted(val_rows, key=row_sort_key, reverse=True)

    selected_by_method: Dict[str, Dict[str, object]] = {}
    for method in (METHOD_FINE, METHOD_HYBRID):
        method_rows = [r for r in val_rows if r["method"] == method]
        selected, mode, ok, fallback = select_row(method_rows, args)
        selected_by_method[method] = {
            "selection_mode": mode,
            "constraint_satisfied": ok,
            "fallback_reason": fallback,
            "selected_val_row": selected,
            "applied_test_row": None,
        }

    recommended, rec_mode, rec_ok, rec_fallback = select_row(val_rows, args)
    result: Dict[str, object] = {
        "experiment_id": args.experiment_id or None,
        "no_training": True,
        "selection_source": str(val_dir),
        "selection_input_summary": val_summary,
        "selection_objective": {
            "hard_constraints": args.hard_constraints,
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
        "threshold_grids": {
            "binary_grid_min": float(binary_grid.min()),
            "binary_grid_max": float(binary_grid.max()),
            "binary_grid_n": int(binary_grid.size),
            "fine_grid_min": float(fine_grid.min()),
            "fine_grid_max": float(fine_grid.max()),
            "fine_grid_n": int(fine_grid.size),
        },
        "recommended": {
            "selection_mode": rec_mode,
            "constraint_satisfied": rec_ok,
            "fallback_reason": rec_fallback,
            "selected_val_row": recommended,
            "applied_test_row": None,
        },
        "methods": selected_by_method,
    }

    test_rows: Optional[List[Dict[str, object]]] = None
    if test_dir is not None:
        test_probs, test_low, test_y, test_summary = load_diag_predictions(test_dir)
        test_rows = build_candidate_rows(test_probs, test_low, test_y, binary_grid, fine_grid, args)
        result["test_source"] = str(test_dir)
        result["test_input_summary"] = test_summary
        result["recommended"]["applied_test_row"] = nearest_test_row(test_rows, recommended)  # type: ignore[index]
        for method, payload in selected_by_method.items():
            payload["applied_test_row"] = nearest_test_row(test_rows, payload["selected_val_row"])  # type: ignore[arg-type]

    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    elif test_dir is not None:
        out_dir = test_dir / "posthoc_lowvis_gate_selection"
    else:
        out_dir = val_dir / "posthoc_lowvis_gate_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "val_posthoc_gate_score_table.csv", val_rows_sorted)
    if test_rows is not None:
        write_csv(out_dir / "test_posthoc_gate_score_table.csv", sorted(test_rows, key=row_sort_key, reverse=True))

    selected_rows: List[Dict[str, object]] = []
    selected_rows.append({"selection": "recommended", "split": "val", **recommended})
    if test_rows is not None:
        selected_rows.append(
            {"selection": "recommended", "split": "test", **nearest_test_row(test_rows, recommended)}
        )
    for method, payload in selected_by_method.items():
        selected_rows.append({"selection": method, "split": "val", **payload["selected_val_row"]})  # type: ignore[dict-item]
        if payload["applied_test_row"] is not None:
            selected_rows.append({"selection": method, "split": "test", **payload["applied_test_row"]})  # type: ignore[dict-item]
    write_csv(out_dir / "selected_posthoc_gate_rows.csv", selected_rows)

    top_rows: List[Dict[str, object]] = []
    for method in (METHOD_FINE, METHOD_HYBRID):
        method_rows = [r for r in val_rows_sorted if r["method"] == method]
        top_rows.extend(method_rows[: max(1, int(args.top_k))])
    write_csv(out_dir / "top_val_posthoc_gate_rows.csv", top_rows)

    with (out_dir / "selected_posthoc_lowvis_gate.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2, ensure_ascii=False)
    print(f"[done] wrote posthoc low-vis gate selection to {out_dir}")


if __name__ == "__main__":
    main()
