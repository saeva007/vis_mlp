#!/usr/bin/env python3
"""Constraint-first validation selector for low-false-alarm Static-RNN runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-summary-csv", required=True)
    p.add_argument("--validation-event-csv", default="")
    p.add_argument("--candidate-key", default="run_id")
    p.add_argument("--event-recall-column", default="pmst_low_vis_recall_mean")
    p.add_argument("--max-fpr", type=float, default=0.04)
    p.add_argument("--min-event-recall", type=float, default=0.60)
    p.add_argument("--min-ultra-recall", type=float, default=0.55)
    p.add_argument("--min-moderate-recall", type=float, default=0.30)
    p.add_argument("--min-moderate-csi", type=float, default=0.09)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--allow-missing-events", action="store_true")
    p.add_argument(
        "--allow-infeasible-fallback",
        action="store_true",
        help="Fill top-k with constraint-violating candidates for diagnostics only.",
    )
    p.add_argument("--out-csv", default="")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    raise KeyError(f"None of the required columns exist: {list(candidates)}")


def numeric(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.validation_summary_csv)
    summary = pd.read_csv(summary_path)
    if args.candidate_key not in summary:
        raise KeyError(f"Missing candidate key {args.candidate_key!r} in {summary_path}")

    fpr_col = first_existing(summary.columns, ["FPR", "false_positive_rate"])
    ultra_r_col = first_existing(summary.columns, ["Ultra_low_R", "Fog_R"])
    moderate_r_col = first_existing(summary.columns, ["Moderate_low_R", "Mist_R"])
    moderate_csi_col = first_existing(summary.columns, ["Moderate_low_CSI", "Mist_CSI"])
    low_csi_col = first_existing(summary.columns, ["Low_vis_CSI", "low_vis_csi"])

    summary = summary.copy()
    summary["selection_fpr"] = numeric(summary, fpr_col)
    summary["selection_ultra_recall"] = numeric(summary, ultra_r_col)
    summary["selection_moderate_recall"] = numeric(summary, moderate_r_col)
    summary["selection_moderate_csi"] = numeric(summary, moderate_csi_col)
    summary["selection_low_vis_csi"] = numeric(summary, low_csi_col)

    if args.validation_event_csv:
        events = pd.read_csv(args.validation_event_csv)
        if args.candidate_key not in events:
            raise KeyError(f"Missing candidate key {args.candidate_key!r} in event table")
        recall_col = first_existing(
            events.columns,
            [args.event_recall_column, "pmst_low_vis_recall_mean", "low_vis_recall"],
        )
        events[recall_col] = pd.to_numeric(events[recall_col], errors="coerce")
        event_agg = (
            events.groupby(args.candidate_key, dropna=False)[recall_col]
            .agg(validation_event_count="count", min_validation_event_recall="min", mean_validation_event_recall="mean")
            .reset_index()
        )
        summary = summary.merge(event_agg, on=args.candidate_key, how="left")
    else:
        if not args.allow_missing_events:
            raise ValueError("--validation-event-csv is required unless --allow-missing-events is set")
        summary["validation_event_count"] = 0
        summary["min_validation_event_recall"] = np.nan
        summary["mean_validation_event_recall"] = np.nan

    event_ok = (
        summary["min_validation_event_recall"].ge(args.min_event_recall)
        if args.validation_event_csv
        else pd.Series(True, index=summary.index)
    )
    summary["passes_event_recall"] = event_ok
    summary["passes_fpr"] = summary["selection_fpr"].le(args.max_fpr)
    summary["passes_ultra_recall"] = summary["selection_ultra_recall"].ge(args.min_ultra_recall)
    summary["passes_moderate_recall"] = summary["selection_moderate_recall"].ge(args.min_moderate_recall)
    summary["passes_moderate_csi"] = summary["selection_moderate_csi"].ge(args.min_moderate_csi)
    pass_cols = [
        "passes_event_recall",
        "passes_fpr",
        "passes_ultra_recall",
        "passes_moderate_recall",
        "passes_moderate_csi",
    ]
    summary["feasible"] = summary[pass_cols].all(axis=1)

    summary["constraint_shortfall"] = (
        (summary["selection_fpr"] - args.max_fpr).clip(lower=0.0).fillna(1.0)
        + (args.min_ultra_recall - summary["selection_ultra_recall"]).clip(lower=0.0).fillna(1.0)
        + (args.min_moderate_recall - summary["selection_moderate_recall"]).clip(lower=0.0).fillna(1.0)
        + (args.min_moderate_csi - summary["selection_moderate_csi"]).clip(lower=0.0).fillna(1.0)
    )
    if args.validation_event_csv:
        summary["constraint_shortfall"] += (
            args.min_event_recall - summary["min_validation_event_recall"]
        ).clip(lower=0.0).fillna(1.0)

    ranked = summary.sort_values(
        ["feasible", "constraint_shortfall", "selection_fpr", "selection_low_vis_csi", "selection_moderate_csi"],
        ascending=[False, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked["selected_for_full_training"] = False
    feasible_idx = ranked.index[ranked["feasible"]].tolist()
    chosen_idx = feasible_idx[: max(0, args.top_k)]
    if args.allow_infeasible_fallback and len(chosen_idx) < max(0, args.top_k):
        for idx in ranked.index:
            if idx not in chosen_idx:
                chosen_idx.append(idx)
            if len(chosen_idx) >= args.top_k:
                break
    ranked.loc[chosen_idx, "selected_for_full_training"] = True

    out_csv = Path(args.out_csv) if args.out_csv else summary_path.with_name(summary_path.stem + "_constraint_ranking.csv")
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    ranked.to_csv(out_csv, index=False, float_format="%.8f")

    selected = ranked.loc[ranked["selected_for_full_training"]]
    if selected.empty:
        selection_status = "no_feasible_candidates"
    elif bool((~selected["feasible"]).any()):
        selection_status = "infeasible_fallback_selected"
    else:
        selection_status = "feasible_candidates_selected"
    payload = {
        "experiment_status": "candidate_only",
        "replaces_mainline": False,
        "selection_source": "validation_only",
        "constraints": {
            "max_fpr": args.max_fpr,
            "min_event_recall": args.min_event_recall,
            "min_ultra_recall": args.min_ultra_recall,
            "min_moderate_recall": args.min_moderate_recall,
            "min_moderate_csi": args.min_moderate_csi,
        },
        "n_candidates": int(len(ranked)),
        "n_feasible": int(ranked["feasible"].sum()),
        "selection_status": selection_status,
        "allow_infeasible_fallback": bool(args.allow_infeasible_fallback),
        "selected": selected[[args.candidate_key, "selection_rank", "feasible"]].to_dict("records"),
        "ranking_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ranked[["selection_rank", args.candidate_key, "feasible", "selection_fpr", "min_validation_event_recall", "selection_low_vis_csi"]].to_string(index=False))
    if selected.empty:
        print("[warn] No feasible candidates; no run was selected for full training.")
    print(f"[table] {out_csv}")
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
