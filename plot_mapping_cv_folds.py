#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot temporal and spatial mapping-operator cross-validation results.

The figure follows the manuscript's existing visual language while using the
four-panel logic of temporal (top) and spatial (bottom) fold comparisons.  It
reports binary low-visibility CSI and recall for Logistic, instantaneous
Static-MLP, and 12 h Static-MLP + GRU operators.
"""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("logistic", "mlp", "gru")
MODEL_LABELS = {
    "logistic": "Logistic",
    "mlp": "Static-MLP",
    "gru": "Static-MLP + GRU (12 h)",
}
MODEL_COLORS = {
    "logistic": "#8A8F98",
    "mlp": "#D09A3A",
    "gru": "#2E5A87",
}
MODEL_MARKERS = {
    "logistic": "D",
    "mlp": "o",
    "gru": "s",
}
INK = "#25282B"
GRID = "#E5E7EB"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_cv_result(
    result_root: Path, expected_kind: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object]]:
    aggregate_dir = result_root / "aggregate"
    folds_dir = result_root / "folds"
    fold_path = aggregate_dir / "fold_metrics.csv"
    pooled_path = aggregate_dir / "pooled_metrics.csv"
    coverage_path = aggregate_dir / "coverage_manifest.json"
    manifest_path = folds_dir / "fold_manifest.json"
    for path in (fold_path, pooled_path, coverage_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    folds = pd.read_csv(fold_path)
    pooled = pd.read_csv(pooled_path)
    coverage = _load_json(coverage_path)
    manifest = _load_json(manifest_path)
    actual_kind = str(manifest.get("cv_kind", "spatial"))
    if actual_kind != expected_kind:
        raise ValueError(f"Expected {expected_kind} CV at {result_root}, found {actual_kind}")
    if str(coverage.get("analysis_decision_rule", "")) != "argmax":
        raise ValueError(f"{coverage_path} is not an argmax primary analysis")

    required = {"model", "fold", "decision_rule", "low_vis_csi", "low_vis_recall"}
    if not required.issubset(folds.columns):
        raise ValueError(f"{fold_path} lacks columns: {sorted(required - set(folds.columns))}")
    if not {"model", "decision_rule", "low_vis_csi", "low_vis_recall"}.issubset(pooled.columns):
        raise ValueError(f"{pooled_path} lacks pooled low-visibility metrics")
    folds = folds.loc[folds["model"].isin(MODEL_ORDER)].copy()
    pooled = pooled.loc[pooled["model"].isin(MODEL_ORDER)].copy()
    fold_rules = set(folds["decision_rule"].astype(str))
    pooled_rules = set(pooled["decision_rule"].astype(str))
    if fold_rules != {"argmax"} or pooled_rules != {"argmax"}:
        raise ValueError(
            f"{expected_kind} plotted metrics must use argmax; "
            f"fold_rules={sorted(fold_rules)} pooled_rules={sorted(pooled_rules)}"
        )
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    for metric in ("low_vis_csi", "low_vis_recall"):
        folds[metric] = pd.to_numeric(folds[metric], errors="raise")
        pooled[metric] = pd.to_numeric(pooled[metric], errors="raise")
        values = folds[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{expected_kind} {metric} values must be finite and within [0, 1]")
    n_folds = int(manifest["n_folds"])
    expected_folds = list(range(n_folds))
    for model in MODEL_ORDER:
        model_rows = folds.loc[folds["model"] == model]
        observed = sorted(model_rows["fold"].tolist())
        if observed != expected_folds:
            raise ValueError(f"{expected_kind} {model} folds are {observed}; expected {expected_folds}")
        if int((pooled["model"] == model).sum()) != 1:
            raise ValueError(f"{expected_kind} pooled metrics must contain exactly one {model} row")
    return folds, pooled, manifest, coverage


def _temporal_tick_labels(manifest: Mapping[str, object]) -> List[str]:
    fold_months = manifest.get("fold_months")
    n_folds = int(manifest["n_folds"])
    if not isinstance(fold_months, dict):
        return [f"T{fold + 1}" for fold in range(n_folds)]
    labels: List[str] = []
    for fold in range(n_folds):
        months = [str(value) for value in fold_months[str(fold)]]
        first = pd.Period(months[0], freq="M")
        last = pd.Period(months[-1], freq="M")
        if first.year == last.year:
            span = calendar.month_abbr[first.month]
            if first != last:
                span += f"–{calendar.month_abbr[last.month]}"
        else:
            span = f"{first.strftime('%Y-%m')}–{last.strftime('%Y-%m')}"
        labels.append(f"T{fold + 1}\n{span}")
    return labels


def _spatial_tick_labels(manifest: Mapping[str, object]) -> List[str]:
    return [f"S{fold + 1}" for fold in range(int(manifest["n_folds"]))]


def _panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    tick_labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    letter: str,
) -> None:
    x = np.arange(len(tick_labels), dtype=float)
    for model in MODEL_ORDER:
        rows = table.loc[table["model"] == model].sort_values("fold")
        ax.plot(
            x,
            rows[metric].to_numpy(dtype=float),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.55,
            label=MODEL_LABELS[model],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.25, len(tick_labels) - 0.75)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(length=2.8, width=0.7, color=INK, pad=2.2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)
    ax.text(
        -0.13,
        1.07,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )


def _source_table(
    temporal: pd.DataFrame,
    spatial: pd.DataFrame,
    temporal_labels: Sequence[str],
    spatial_labels: Sequence[str],
) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for kind, table, labels in (
        ("temporal", temporal, temporal_labels),
        ("spatial", spatial, spatial_labels),
    ):
        part = table.loc[table["model"].isin(MODEL_ORDER)].copy()
        part.insert(0, "cv_type", kind)
        part["fold_label"] = part["fold"].map({index: label.replace("\n", " ") for index, label in enumerate(labels)})
        part["model_label"] = part["model"].map(MODEL_LABELS)
        pieces.append(
            part[
                [
                    "cv_type",
                    "fold",
                    "fold_label",
                    "model",
                    "model_label",
                    "decision_rule",
                    "low_vis_csi",
                    "low_vis_recall",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _summary_table(
    temporal_folds: pd.DataFrame,
    temporal_pooled: pd.DataFrame,
    spatial_folds: pd.DataFrame,
    spatial_pooled: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for kind, folds, pooled in (
        ("temporal", temporal_folds, temporal_pooled),
        ("spatial", spatial_folds, spatial_pooled),
    ):
        for model in MODEL_ORDER:
            fold_rows = folds.loc[folds["model"] == model]
            pooled_row = pooled.loc[pooled["model"] == model].iloc[0]
            for metric in ("low_vis_csi", "low_vis_recall"):
                values = fold_rows[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "cv_type": kind,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "decision_rule": "argmax",
                        "metric": metric,
                        "fold_mean": float(np.mean(values)),
                        "fold_sd": float(np.std(values, ddof=1)),
                        "fold_min": float(np.min(values)),
                        "fold_max": float(np.max(values)),
                        "pooled_oof": float(pooled_row[metric]),
                        "n_folds": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def plot_mapping_cv(
    spatial_result_root: Path,
    temporal_result_root: Path,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> List[Path]:
    setup_style()
    spatial_folds, spatial_pooled, spatial_manifest, spatial_coverage = _load_cv_result(
        spatial_result_root.resolve(), "spatial"
    )
    temporal_folds, temporal_pooled, temporal_manifest, temporal_coverage = _load_cv_result(
        temporal_result_root.resolve(), "temporal"
    )
    if int(spatial_manifest["n_folds"]) != int(temporal_manifest["n_folds"]):
        raise ValueError("Spatial and temporal experiments must use the same fold count")
    temporal_labels = _temporal_tick_labels(temporal_manifest)
    spatial_labels = _spatial_tick_labels(spatial_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_table(temporal_folds, spatial_folds, temporal_labels, spatial_labels)
    source.to_csv(output_dir / f"{stem}_source_data.csv", index=False)
    summary = _summary_table(
        temporal_folds, temporal_pooled, spatial_folds, spatial_pooled
    )
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)

    width_in = 183.0 / 25.4
    height_in = 132.0 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width_in, height_in), sharey="col")
    _panel(
        axes[0, 0],
        temporal_folds,
        "low_vis_csi",
        temporal_labels,
        "Temporal transfer: low-visibility CSI",
        "Held-out temporal block",
        "CSI",
        "a",
    )
    _panel(
        axes[0, 1],
        temporal_folds,
        "low_vis_recall",
        temporal_labels,
        "Temporal transfer: low-visibility recall",
        "Held-out temporal block",
        "Recall",
        "b",
    )
    _panel(
        axes[1, 0],
        spatial_folds,
        "low_vis_csi",
        spatial_labels,
        "Spatial transfer: low-visibility CSI",
        "Held-out spatial block (west to east)",
        "CSI",
        "c",
    )
    _panel(
        axes[1, 1],
        spatial_folds,
        "low_vis_recall",
        spatial_labels,
        "Spatial transfer: low-visibility recall",
        "Held-out spatial block (west to east)",
        "Recall",
        "d",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        handlelength=1.8,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.91, wspace=0.25, hspace=0.43)

    written: List[Path] = []
    for raw_format in formats:
        fmt = raw_format.strip().lower()
        if not fmt:
            continue
        if fmt not in {"svg", "pdf", "png", "tiff"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs: Dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt in {"png", "tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)

    figure_manifest = {
        "schema_version": 1,
        "core_conclusion": (
            "Tests whether nonlinear and temporal mapping-operator gains persist across both "
            "held-out calendar blocks and geographically held-out station blocks."
        ),
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "final_size_mm": {"width": 183.0, "height": 132.0},
        "panels": {
            "a": "temporal-fold low-visibility CSI",
            "b": "temporal-fold low-visibility recall",
            "c": "spatial-fold low-visibility CSI",
            "d": "spatial-fold low-visibility recall",
        },
        "models": [MODEL_LABELS[model] for model in MODEL_ORDER],
        "metric_definition": "binary low visibility is visibility < 1000 m",
        "variability_definition": "five held-out fold values; no row-level error bars",
        "decision_rule": "argmax",
        "checkpoint_selection": {
            "spatial": {
                "rules": spatial_coverage.get("checkpoint_selection_rules", []),
                "all_argmax": bool(spatial_coverage.get("all_checkpoints_selected_with_argmax", False)),
            },
            "temporal": {
                "rules": temporal_coverage.get("checkpoint_selection_rules", []),
                "all_argmax": bool(temporal_coverage.get("all_checkpoints_selected_with_argmax", False)),
            },
        },
        "threshold_policy": (
            "argmax of the three class probabilities; no model- or fold-specific "
            "probability thresholds are used in the plotted predictions"
        ),
        "input_artifacts": {
            "spatial": "aggregate/fold_metrics.csv and aggregate/pooled_metrics.csv",
            "temporal": "aggregate/fold_metrics.csv and aggregate/pooled_metrics.csv",
        },
        "source_data": f"{stem}_source_data.csv",
        "summary_data": f"{stem}_summary.csv",
        "outputs": [path.name for path in written],
    }
    with (output_dir / f"{stem}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(figure_manifest, handle, indent=2, ensure_ascii=False)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-result-root", required=True)
    parser.add_argument("--temporal-result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="mapping_operator_spatiotemporal_cv")
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = plot_mapping_cv(
        Path(args.spatial_result_root),
        Path(args.temporal_result_root),
        Path(args.output_dir),
        args.stem,
        args.formats.split(","),
        int(args.dpi),
    )
    for path in written:
        print(f"[figure] {path}", flush=True)


if __name__ == "__main__":
    main()
