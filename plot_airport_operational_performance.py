#!/usr/bin/env python3
"""Plot direct performance of the airport METAR-tuned operational model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DEFAULT_RESULT_DIR = Path(
    r"C:\Users\saeva\Downloads\eval_s2_vs_finetune_test_115190956"
)
DEFAULT_OUT_DIR = Path(
    r"C:\vis_code\vis_mlp\figures\airport_operational_performance_115190956"
)
MODEL_NAME = "finetune"

INK = "#222222"
MUTED = "#66717C"
GRID = "#D8DEE3"
TEAL = "#2F6F73"
BLUE = "#4C78A8"
GOLD = "#D39C3F"
CORAL = "#C05A45"
PALE = "#F5F7F8"


def configure_matplotlib() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def pct(value: float) -> float:
    return float(value) * 100.0


def fmt_pct(value: float) -> str:
    return f"{pct(value):.1f}%"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.09,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def read_inputs(result_dir: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    summary_path = result_dir / "comparison_summary.csv"
    station_path = result_dir / "station_metrics.csv"
    cm_path = result_dir / "finetune_confusion_matrix.csv"
    for path in (summary_path, station_path, cm_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    summary = pd.read_csv(summary_path)
    row = summary.loc[summary["model"] == MODEL_NAME]
    if row.empty:
        raise ValueError(f"No model={MODEL_NAME!r} row found in {summary_path}")

    stations = pd.read_csv(station_path)
    stations = stations.loc[stations["model"] == MODEL_NAME].copy()
    cm = pd.read_csv(cm_path, index_col=0)
    return row.iloc[0], stations, cm


def export_source_data(
    summary: pd.Series,
    stations: pd.DataFrame,
    cm: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows = []
    for metric in (
        "low_vis_precision",
        "low_vis_recall",
        "low_vis_csi",
        "false_positive_rate",
        "accuracy",
        "Fog_P",
        "Fog_R",
        "Fog_CSI",
        "Mist_P",
        "Mist_R",
        "Mist_CSI",
    ):
        rows.append(
            {
                "panel": "overall_metrics",
                "metric": metric,
                "value": summary[metric],
                "value_percent": pct(summary[metric]),
                "n": int(summary["n"]),
                "true_fog": int(summary["true_fog"]),
                "true_mist": int(summary["true_mist"]),
                "true_clear": int(summary["true_clear"]),
            }
        )

    event_sites = stations.loc[stations["low_true"] > 0].copy()
    for _, row in event_sites.iterrows():
        for metric in ("low_vis_precision", "low_vis_recall", "accuracy"):
            rows.append(
                {
                    "panel": "station_event_sites",
                    "station": row["station"],
                    "metric": metric,
                    "value": row[metric],
                    "value_percent": pct(row[metric]),
                    "n": int(row["n"]),
                    "low_true": int(row["low_true"]),
                    "low_pred": int(row["low_pred"]),
                }
            )

    for true_label, true_row in cm.iterrows():
        for pred_label, value in true_row.items():
            rows.append(
                {
                    "panel": "confusion_matrix_counts",
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "count": int(value),
                }
            )
    pd.DataFrame(rows).to_csv(
        out_dir / "airport_operational_performance_source_data.csv",
        index=False,
    )


def plot_kpis(ax: plt.Axes, summary: pd.Series) -> None:
    metrics = [
        ("总体准确率", "accuracy", BLUE),
        ("低能见度事件召回率", "low_vis_recall", TEAL),
        ("低能见度事件精确率", "low_vis_precision", CORAL),
        ("低能见度事件CSI", "low_vis_csi", GOLD),
        ("误报率", "false_positive_rate", MUTED),
    ]
    y = np.arange(len(metrics))[::-1]
    ax.barh(
        y,
        [pct(summary[key]) for _, key, _ in metrics],
        height=0.46,
        color=[color for _, _, color in metrics],
        alpha=0.9,
    )
    for yi, (label, key, color) in zip(y, metrics):
        value = pct(summary[key])
        ax.text(
            value + 1.4,
            yi,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=6.6,
            color=color,
            fontweight="bold" if key in {"low_vis_recall", "low_vis_precision"} else "normal",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in metrics])
    ax.set_xlim(0, 104)
    ax.set_xlabel("指标值 (%)")
    ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.8)
    ax.set_title("业务测试集关键指标", loc="left", fontsize=7.5, fontweight="bold", pad=5)
    panel_label(ax, "a")


def plot_event_metrics(ax: plt.Axes, summary: pd.Series) -> None:
    groups = [
        ("低能见度事件\n(<1000 m)", "low_vis_precision", "low_vis_recall", "low_vis_csi"),
        ("超低能见度\n(<500 m)", "Fog_P", "Fog_R", "Fog_CSI"),
        ("中度低能见度\n(500-1000 m)", "Mist_P", "Mist_R", "Mist_CSI"),
    ]
    metric_labels = ["精确率", "召回率", "CSI"]
    colors = [CORAL, TEAL, GOLD]
    x = np.arange(len(groups))
    width = 0.22
    offsets = np.array([-width, 0.0, width])
    for j, (metric_label, color) in enumerate(zip(metric_labels, colors)):
        vals = [pct(summary[group[j + 1]]) for group in groups]
        ax.bar(x + offsets[j], vals, width=width, color=color, alpha=0.92, label=metric_label)
        for xi, val in zip(x + offsets[j], vals):
            ax.text(xi, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=5.6, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([group[0] for group in groups])
    ax.set_ylim(0, 58)
    ax.set_ylabel("指标值 (%)")
    ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.8)
    ax.legend(loc="upper right", ncol=1, handlelength=1.1, handletextpad=0.4)
    ax.set_title("低能见度事件识别能力", loc="left", fontsize=7.5, fontweight="bold", pad=5)
    panel_label(ax, "b")


def plot_station_scatter(ax: plt.Axes, summary: pd.Series, stations: pd.DataFrame) -> None:
    event_sites = stations.loc[stations["low_true"] > 0].copy()
    event_sites["precision_pct"] = event_sites["low_vis_precision"] * 100.0
    event_sites["recall_pct"] = event_sites["low_vis_recall"] * 100.0
    max_events = max(float(event_sites["low_true"].max()), 1.0)
    sizes = 18.0 + 82.0 * np.sqrt(event_sites["low_true"].astype(float) / max_events)

    ax.scatter(
        event_sites["precision_pct"],
        event_sites["recall_pct"],
        s=sizes,
        c=event_sites["low_true"],
        cmap=mpl.colors.LinearSegmentedColormap.from_list("event_load", ["#D7E6E7", TEAL]),
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.scatter(
        pct(summary["low_vis_precision"]),
        pct(summary["low_vis_recall"]),
        marker="D",
        s=56,
        color=CORAL,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax.text(
        pct(summary["low_vis_precision"]) + 2.3,
        pct(summary["low_vis_recall"]) + 1.0,
        "总体",
        fontsize=6.2,
        color=CORAL,
        ha="left",
        va="center",
    )
    median_p = np.nanmedian(event_sites["precision_pct"])
    median_r = np.nanmedian(event_sites["recall_pct"])
    ax.axvline(median_p, color="#AAB3BB", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axhline(median_r, color="#AAB3BB", lw=0.7, ls=(0, (3, 2)), zorder=0)

    top = event_sites.sort_values("low_true", ascending=False).head(5)
    for _, row in top.iterrows():
        ax.text(
            row["precision_pct"] + 1.4,
            row["recall_pct"] + 1.4,
            str(row["station"]),
            fontsize=5.3,
            color="#3D464F",
            ha="left",
            va="bottom",
        )

    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 104)
    ax.set_xlabel("站点低能见度事件精确率 (%)")
    ax.set_ylabel("站点低能见度事件召回率 (%)")
    ax.grid(color=GRID, lw=0.45, alpha=0.8)
    ax.set_title("机场站点表现", loc="left", fontsize=7.5, fontweight="bold", pad=5)
    ax.text(
        1,
        100,
        f"有低能见度事件样本的站点: n={event_sites['station'].nunique()}",
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED,
    )
    panel_label(ax, "c")


def plot_confusion(ax: plt.Axes, cm: pd.DataFrame) -> None:
    labels = ["超低能见度", "中度低能见度", "正常"]
    arr = cm.to_numpy(dtype=float)
    row_sum = np.maximum(arr.sum(axis=1, keepdims=True), 1.0)
    norm = arr / row_sum * 100.0
    im = ax.imshow(norm, cmap=mpl.colors.LinearSegmentedColormap.from_list("cm", ["#FFFFFF", "#D7E6E7", TEAL]), vmin=0, vmax=100)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            color = "white" if norm[i, j] > 55 else INK
            ax.text(
                j,
                i,
                f"{norm[i, j]:.1f}%\n{int(arr[i, j]):,}",
                ha="center",
                va="center",
                fontsize=5.7,
                color=color,
            )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("观测类别")
    ax.set_title("按观测类别归一化的混淆矩阵", loc="left", fontsize=7.5, fontweight="bold", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.set_label("行占比 (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    panel_label(ax, "d")


def write_qa_notes(
    summary: pd.Series,
    stations: pd.DataFrame,
    out_dir: Path,
) -> None:
    event_sites = stations.loc[stations["low_true"] > 0].copy()
    notes = [
        "# 机场业务模型性能图 QA",
        "",
        "- 后端: 仅使用 Python/matplotlib。",
        "- 图型: 定量网格。",
        "- 核心结论: 直接汇总机场 METAR 微调业务模型在测试集上的表现，不展示微调前对比。",
        "- 数据划分: test。",
        "- 判别规则: 最大概率类别。",
        f"- 样本数: n={int(summary['n'])}; 真实超低能见度={int(summary['true_fog'])}; 真实中度低能见度={int(summary['true_mist'])}; 真实正常={int(summary['true_clear'])}。",
        f"- 低能见度事件精确率: {fmt_pct(summary['low_vis_precision'])}。",
        f"- 低能见度事件召回率: {fmt_pct(summary['low_vis_recall'])}。",
        f"- 低能见度事件 CSI: {fmt_pct(summary['low_vis_csi'])}。",
        f"- 误报率: {fmt_pct(summary['false_positive_rate'])}。",
        f"- 总体准确率: {fmt_pct(summary['accuracy'])}。",
        f"- 站点面板仅纳入有低能见度事件观测样本的站点: n={event_sites['station'].nunique()}。",
        "- 导出检查: SVG/PDF 保留可编辑文本; TIFF 为 600 dpi; PNG 预览为 300 dpi。",
        "- 源数据: airport_operational_performance_source_data.csv。",
    ]
    (out_dir / "airport_operational_performance_QA.md").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8",
    )


def build_figure(result_dir: Path, out_dir: Path) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, stations, cm = read_inputs(result_dir)
    export_source_data(summary, stations, cm, out_dir)

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 1.2],
        height_ratios=[1.0, 1.08],
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.855,
        wspace=0.34,
        hspace=0.48,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    plot_kpis(ax0, summary)
    plot_event_metrics(ax1, summary)
    plot_station_scatter(ax2, summary, stations)
    plot_confusion(ax3, cm)

    low_events = int(summary["true_fog"] + summary["true_mist"])
    fig.suptitle(
        "机场 METAR 微调能见度模型：业务测试集直接性能",
        x=0.08,
        y=0.972,
        ha="left",
        va="top",
        fontsize=8.7,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.923,
        (
            f"静态 MLP + GRU，12 小时窗口；测试集 n={int(summary['n']):,}；"
            f"观测低能见度事件={low_events:,}；最大概率判别规则"
        ),
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )

    base = out_dir / "airport_operational_performance"
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_qa_notes(summary, stations, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_figure(args.result_dir, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
