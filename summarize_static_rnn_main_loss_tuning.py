#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize validation metrics from main-loss tuning checkpoints.

This script reads the checkpoint metadata saved by train_static_rnn_lowvis.py.
It is intentionally validation-only: use it to choose a loss setting before
running the selected model on the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


DEFAULT_VARIANTS = (
    "v00_base",
    "v01_recall_plus",
    "v02_ultra_focus",
    "v03_precision_guard",
    "v04_boundary_soft",
)

PREFERRED_FIELDS = (
    "variant_id",
    "variant_label",
    "run_id",
    "checkpoint",
    "status",
    "score",
    "step",
    "selection_metric",
    "threshold_mode",
    "thresholds",
    "Ultra_low_CSI",
    "Ultra_low_R",
    "Ultra_low_P",
    "Moderate_low_CSI",
    "Moderate_low_R",
    "Moderate_low_P",
    "Low_vis_CSI",
    "Low_vis_R",
    "Low_vis_P",
    "FPR",
    "accuracy",
    "balanced_acc",
    "mcc",
    "class_weight_fog_internal",
    "class_weight_mist_internal",
    "class_weight_clear",
    "focal_gamma_fog_internal",
    "focal_gamma_mist_internal",
    "focal_gamma_clear",
    "alpha_clear_fp",
    "alpha_recall_boost",
    "boundary_weight",
    "extra_args",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize validation metrics for main-loss tuning checkpoints.")
    p.add_argument("--manifest", default="", help="TSV manifest written by submit_static_rnn_main_loss_tuning_matrix_chain.sh.")
    p.add_argument("--base_run_prefix", default="", help="Base run prefix used when no manifest is provided.")
    p.add_argument("--checkpoint_dir", default="/public/home/putianshu/vis_mlp/checkpoints")
    p.add_argument("--stage_tag", default="S2_PhaseB")
    p.add_argument("--variants", default=":".join(DEFAULT_VARIANTS), help="Variant ids separated by colon/comma/space.")
    p.add_argument("--out_csv", default="", help="Output CSV path. Defaults next to the manifest when possible.")
    p.add_argument("--out_json", default="", help="Optional JSON copy of the same summary rows.")
    p.add_argument("--allow_missing", action="store_true")
    return p.parse_args()


def split_tokens(value: str) -> List[str]:
    return [tok for tok in value.replace(",", ":").replace(" ", ":").split(":") if tok]


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def rows_from_prefix(base_run_prefix: str, variants: Iterable[str], checkpoint_dir: Path, stage_tag: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for variant_id in variants:
        variant_prefix = f"{base_run_prefix}_{variant_id}"
        run_id = f"{variant_prefix}_2_proposed_rare_event_focal"
        rows.append(
            {
                "variant_id": variant_id,
                "variant_label": variant_id,
                "run_prefix": variant_prefix,
                "run_id": run_id,
                "extra_args": "",
                "s2_checkpoint": str(checkpoint_dir / f"{run_id}_{stage_tag}_best_score.pt"),
            }
        )
    return rows


def load_checkpoint_metadata(path: Path) -> Mapping[str, object]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        meta = payload.get("metadata", {})
        if isinstance(meta, Mapping):
            return meta
    return {}


def fnum(value: object) -> object:
    if value is None:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def metric(metrics: Mapping[str, object], key: str) -> object:
    return fnum(metrics.get(key, ""))


def summarize_row(row: Mapping[str, str], checkpoint_dir: Path, stage_tag: str, allow_missing: bool) -> Optional[Dict[str, object]]:
    ckpt_raw = row.get("s2_checkpoint", "").strip()
    run_id = row.get("run_id", "").strip()
    if ckpt_raw:
        ckpt = Path(ckpt_raw)
    elif run_id:
        ckpt = checkpoint_dir / f"{run_id}_{stage_tag}_best_score.pt"
    else:
        return None
    if not ckpt.is_absolute():
        ckpt = checkpoint_dir / ckpt
    if not ckpt.exists():
        if allow_missing:
            return {
                "variant_id": row.get("variant_id", ""),
                "variant_label": row.get("variant_label", ""),
                "run_id": run_id,
                "checkpoint": str(ckpt),
                "status": "missing",
            }
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    meta = load_checkpoint_metadata(ckpt)
    metrics_obj = meta.get("metrics", {}) if isinstance(meta, Mapping) else {}
    metrics = metrics_obj if isinstance(metrics_obj, Mapping) else {}
    loss_obj = meta.get("loss_terms", {}) if isinstance(meta, Mapping) else {}
    loss_terms = loss_obj if isinstance(loss_obj, Mapping) else {}
    thresholds = meta.get("thresholds", {}) if isinstance(meta, Mapping) else {}

    out: Dict[str, object] = {
        "variant_id": row.get("variant_id", ""),
        "variant_label": row.get("variant_label", ""),
        "run_id": meta.get("run_id", run_id),
        "checkpoint": str(ckpt),
        "status": "ok",
        "score": fnum(meta.get("score", "")),
        "step": fnum(meta.get("step", "")),
        "selection_metric": meta.get("selection_metric", ""),
        "threshold_mode": meta.get("threshold_mode", ""),
        "thresholds": json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
        "Ultra_low_CSI": metric(metrics, "Fog_CSI"),
        "Ultra_low_R": metric(metrics, "Fog_R"),
        "Ultra_low_P": metric(metrics, "Fog_P"),
        "Moderate_low_CSI": metric(metrics, "Mist_CSI"),
        "Moderate_low_R": metric(metrics, "Mist_R"),
        "Moderate_low_P": metric(metrics, "Mist_P"),
        "Low_vis_CSI": metric(metrics, "low_vis_csi"),
        "Low_vis_R": metric(metrics, "low_vis_recall"),
        "Low_vis_P": metric(metrics, "low_vis_precision"),
        "FPR": metric(metrics, "false_positive_rate"),
        "accuracy": metric(metrics, "accuracy"),
        "balanced_acc": metric(metrics, "balanced_acc"),
        "mcc": metric(metrics, "mcc"),
        "class_weight_fog_internal": fnum(loss_terms.get("class_weight_fog", "")),
        "class_weight_mist_internal": fnum(loss_terms.get("class_weight_mist", "")),
        "class_weight_clear": fnum(loss_terms.get("class_weight_clear", "")),
        "focal_gamma_fog_internal": fnum(loss_terms.get("focal_gamma_fog", "")),
        "focal_gamma_mist_internal": fnum(loss_terms.get("focal_gamma_mist", "")),
        "focal_gamma_clear": fnum(loss_terms.get("focal_gamma_clear", "")),
        "alpha_clear_fp": fnum(loss_terms.get("alpha_clear_fp", "")),
        "alpha_recall_boost": fnum(loss_terms.get("alpha_recall_boost", "")),
        "boundary_weight": fnum(loss_terms.get("boundary_weight", "")),
        "extra_args": row.get("extra_args", ""),
    }
    return out


def sort_key(row: Mapping[str, object]) -> float:
    try:
        return float(row.get("score", "-inf"))
    except (TypeError, ValueError):
        return float("-inf")


def print_table(rows: List[Mapping[str, object]]) -> None:
    cols = (
        "variant_id",
        "score",
        "Ultra_low_CSI",
        "Ultra_low_R",
        "Moderate_low_CSI",
        "Moderate_low_R",
        "Low_vis_CSI",
        "Low_vis_P",
        "FPR",
        "step",
    )
    print("\t".join(cols))
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        print("\t".join(vals))


def collect_fieldnames(rows: List[Mapping[str, object]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for key in PREFERRED_FIELDS:
        if any(key in row for row in rows):
            names.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names or [
        "variant_id",
        "variant_label",
        "run_id",
        "checkpoint",
        "status",
    ]


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if args.manifest:
        manifest = Path(args.manifest).expanduser()
        rows_in = load_manifest(manifest)
        default_csv = manifest.with_name(manifest.stem.replace("_manifest", "") + "_validation_summary.csv")
    else:
        if not args.base_run_prefix:
            raise SystemExit("Provide --manifest or --base_run_prefix.")
        rows_in = rows_from_prefix(args.base_run_prefix, split_tokens(args.variants), checkpoint_dir, args.stage_tag)
        default_csv = Path("main_loss_tuning_validation_summary.csv")

    rows: List[Dict[str, object]] = []
    for row in rows_in:
        out = summarize_row(row, checkpoint_dir, args.stage_tag, args.allow_missing)
        if out is not None:
            rows.append(out)
    rows.sort(key=sort_key, reverse=True)

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else default_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(rows)
    print(f"\nWrote: {out_csv}")
    if args.out_json:
        print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
