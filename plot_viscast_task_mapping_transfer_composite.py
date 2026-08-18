#!/usr/bin/env python3
"""Compose Fig. 2 from existing operator and mapping-CV panel functions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plot_mapping_cv_folds as mapping_plot


FIGURE_WIDTH = 7.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifs-comparison-dir", type=Path, required=True)
    parser.add_argument("--spatial-result-root", type=Path, required=True)
    parser.add_argument("--temporal-result-root", type=Path, required=True)
    parser.add_argument("--controlled-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="fig2_viscast_mapping_transferability")
    parser.add_argument("--gru-label", default=mapping_plot.MODEL_LABELS["gru"])
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_module(path: Path, name: str) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if str(resolved.parent) not in sys.path:
        sys.path.insert(0, str(resolved.parent))
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_mapping_models(
    spatial_folds: pd.DataFrame,
    temporal_folds: pd.DataFrame,
    gru_label: str,
) -> tuple[Sequence[str], Dict[str, str]]:
    spatial_models = tuple(
        model
        for model in mapping_plot.MODEL_ORDER
        if model in set(spatial_folds["model"].astype(str))
    )
    temporal_models = tuple(
        model
        for model in mapping_plot.MODEL_ORDER
        if model in set(temporal_folds["model"].astype(str))
    )
    if spatial_models != temporal_models:
        raise ValueError(
            "Spatial and temporal metric tables contain different models: "
            f"spatial={spatial_models}, temporal={temporal_models}"
        )
    if spatial_models == mapping_plot.MODEL_ORDER:
        labels = dict(mapping_plot.MODEL_LABELS)
        labels["gru"] = str(gru_label)
        return mapping_plot.MODEL_ORDER, labels
    if set(spatial_models) == set(mapping_plot.DIRECT_MODEL_ORDER):
        labels = dict(mapping_plot.DIRECT_MODEL_LABELS)
        labels["gru"] = str(gru_label)
        return mapping_plot.DIRECT_MODEL_ORDER, labels
    raise ValueError(f"Unsupported mapping-CV model set: {spatial_models}")


def export(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for raw in formats:
        fmt = raw.strip().lower()
        if not fmt:
            continue
        if fmt not in {"svg", "pdf", "png", "tiff"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        path = output_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    if not outputs:
        raise ValueError("At least one output format is required")
    return outputs


def main() -> None:
    args = parse_args()
    controlled = load_module(args.controlled_script, "viscast_controlled_panels")
    ifs_dir = args.ifs_comparison_dir.expanduser().resolve()
    matched_path = ifs_dir / "ifs_diagnostic_matched_metrics.csv"
    if not matched_path.is_file():
        raise FileNotFoundError(matched_path)
    matched = pd.read_csv(matched_path)
    operator_metrics = [
        "fog_csi",
        "fog_pod",
        "fog_precision",
        "mist_csi",
        "mist_pod",
        "mist_precision",
        "low_vis_csi",
        "low_vis_recall",
        "low_vis_precision",
    ]
    controlled.require_columns(matched, ["source", *operator_metrics], matched_path)

    spatial_folds, spatial_pooled, spatial_manifest, spatial_coverage = mapping_plot._load_cv_result(
        args.spatial_result_root.expanduser().resolve(), "spatial"
    )
    temporal_folds, temporal_pooled, temporal_manifest, temporal_coverage = mapping_plot._load_cv_result(
        args.temporal_result_root.expanduser().resolve(), "temporal"
    )
    model_order, model_labels = select_mapping_models(
        spatial_folds,
        temporal_folds,
        args.gru_label,
    )
    temporal_labels = mapping_plot._temporal_tick_labels(temporal_manifest)
    spatial_labels = mapping_plot._spatial_tick_labels(spatial_manifest)

    controlled.setup_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH, 9.05))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.86, 1.08, 1.08],
        left=0.09,
        right=0.985,
        top=0.965,
        bottom=0.065,
        hspace=0.55,
    )
    operator_grid = outer[0].subgridspec(1, 3, wspace=0.50)
    operator_axes = [fig.add_subplot(operator_grid[0, index]) for index in range(3)]
    operator_panels = [
        (
            "Ultra-low",
            [("fog_csi", "CSI"), ("fog_pod", "Recall"), ("fog_precision", "Precision")],
        ),
        (
            "Moderate-low",
            [("mist_csi", "CSI"), ("mist_pod", "Recall"), ("mist_precision", "Precision")],
        ),
        (
            "Low-vis event",
            [
                ("low_vis_csi", "CSI"),
                ("low_vis_recall", "Recall"),
                ("low_vis_precision", "Precision"),
            ],
        ),
    ]
    source_frames = []
    for index, (axis, (title, specs)) in enumerate(zip(operator_axes, operator_panels)):
        source_frames.append(
            controlled.draw_operator_panel(
                axis,
                matched,
                title,
                specs,
                show_ylabel=(index == 0),
                show_legend=(index == 0),
            )
        )
    for letter, axis in zip("abc", operator_axes):
        controlled.panel_label(axis, letter, x=-0.24)

    mapping_plot.setup_style()
    temporal_grid = outer[1].subgridspec(1, 2, wspace=0.25)
    spatial_grid = outer[2].subgridspec(1, 2, wspace=0.25)
    temporal_axes = [fig.add_subplot(temporal_grid[0, index]) for index in range(2)]
    spatial_axes = [fig.add_subplot(spatial_grid[0, index]) for index in range(2)]
    panel_specs = (
        (
            temporal_axes[0],
            temporal_folds,
            "low_vis_csi",
            temporal_labels,
            "Temporal transfer: low-visibility CSI",
            "Held-out temporal block",
            "CSI",
            "d",
        ),
        (
            temporal_axes[1],
            temporal_folds,
            "low_vis_recall",
            temporal_labels,
            "Temporal transfer: low-visibility recall",
            "Held-out temporal block",
            "Recall",
            "e",
        ),
        (
            spatial_axes[0],
            spatial_folds,
            "low_vis_csi",
            spatial_labels,
            "Spatial transfer: low-visibility CSI",
            "Held-out spatial block (west to east)",
            "CSI",
            "f",
        ),
        (
            spatial_axes[1],
            spatial_folds,
            "low_vis_recall",
            spatial_labels,
            "Spatial transfer: low-visibility recall",
            "Held-out spatial block (west to east)",
            "Recall",
            "g",
        ),
    )
    for spec in panel_specs:
        mapping_plot._panel(
            *spec,
            model_order=model_order,
            model_labels=model_labels,
        )
    handles, labels = temporal_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.650),
        ncol=len(model_order),
        handlelength=1.8,
        columnspacing=1.5,
    )

    output_dir = args.output_dir.expanduser().resolve()
    outputs = export(
        fig,
        output_dir,
        args.stem,
        args.formats.split(","),
        args.dpi,
    )
    plt.close(fig)
    mapping_source = mapping_plot._source_table(
        temporal_folds,
        spatial_folds,
        temporal_labels,
        spatial_labels,
        model_order,
        model_labels,
    )
    mapping_source.insert(0, "figure_section", "transferability")
    operator_source = pd.concat(source_frames, ignore_index=True, sort=False)
    operator_source.insert(0, "figure_section", "task_specific_mapping")
    pd.concat([operator_source, mapping_source], ignore_index=True, sort=False).to_csv(
        output_dir / f"{args.stem}_source_data.csv",
        index=False,
        float_format="%.8f",
    )
    manifest = {
        "core_conclusion": (
            "Replacing the fixed IFS visibility diagnostic with VisCast improves task-specific mapping, "
            "and the advantage persists across held-out temporal and spatial blocks."
        ),
        "panels": {
            "a-c": "IFS diagnostic visibility versus IFS-driven VisCast",
            "d-e": "five-fold temporal transfer",
            "f-g": "five-fold spatial transfer",
        },
        "inputs": {
            "ifs_comparison_dir": str(ifs_dir),
            "spatial_result_root": str(args.spatial_result_root.expanduser().resolve()),
            "temporal_result_root": str(args.temporal_result_root.expanduser().resolve()),
        },
        "outputs": outputs,
        "rendering": "existing panel functions redrawn on a 3+2+2 composite canvas",
        "spatial_coverage": spatial_coverage,
        "temporal_coverage": temporal_coverage,
        "spatial_pooled_rows": int(len(spatial_pooled)),
        "temporal_pooled_rows": int(len(temporal_pooled)),
    }
    (output_dir / f"{args.stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_dir / f"{args.stem}.png")


if __name__ == "__main__":
    main()
