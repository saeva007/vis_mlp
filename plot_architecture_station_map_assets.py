#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create China station-map assets for the PMST low-visibility architecture figure.

This script is intended to run on the remote project machine, not locally.  It
uses the S2 month-tail test set and the existing China shapefile to create
mostly annotation-free PNG layers for an architecture schematic:

  - compact dynamic-input station stacks at selected lags, default t-11/t-7/t-3/t.
  - an angled 12-h RH2M window stack for the dynamic sequence icon.
  - one static/feature-engineering station stack.
  - one gridded China orography map clipped to the national boundary.
  - several three-class output maps selected for adequate low-visibility count
    and high low-visibility precision/recall when evaluation CSV is available.
  - a small categorical output colorbar for the three visibility classes.

The script avoids importing the training model; it only visualizes data already
saved by s2_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


BASE_PATH = Path("/public/home/putianshu/vis_mlp")
DEFAULT_DATA_DIR = BASE_PATH / "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_SHP = Path("/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
DEFAULT_ORO = Path("/public/home/putianshu/vis_cnn/data_orography.nc")
DEFAULT_OUT = BASE_PATH / "architecture_map_assets"

WINDOW_SIZE = 12
CHINA_EXTENT = (73.0, 136.0, 3.0, 54.8)
MAP_BACKGROUND = "#FFFFFF"
HUMIDITY_CMAP = "YlGnBu"
FEATURE_ENGINEERING_CMAP = LinearSegmentedColormap.from_list(
    "pmst_feature_warm",
    ["#FFF7E6", "#FEE8B0", "#FDB863", "#E36C3D", "#9E2F1C"],
    N=256,
)
TOPOGRAPHY_CMAP = LinearSegmentedColormap.from_list(
    "pmst_topography",
    ["#1A9850", "#66BD63", "#D9EF8B", "#C8A96A", "#8C510A", "#F7F7F2"],
    N=256,
)
TERRAIN_ANOMALY_CMAP = "BrBG"
TERRAIN_STD_CMAP = "YlOrBr"

BASE_DYN_NAMES = [
    "RH2M",
    "T2M",
    "PRECIP",
    "MSLP",
    "SW_RAD",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "CAPE",
    "LCC",
    "T_925",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "W_925",
    "W_1000",
    "DPD",
    "INVERSION",
]

FE_NAMES = [
    "near_saturation_proxy",
    "wind_favorability",
    "inversion_weak_wind_stability",
    "night_clear_sky_cooling",
    "rh2m_minus_rh925_proxy",
    "fog_potential_index",
    "rh2m_delta_3h",
    "rh2m_delta_6h",
    "rh2m_std_12h",
    "rh2m_range_12h",
    "t2m_delta_3h",
    "t2m_delta_6h",
    "t2m_std_12h",
    "t2m_range_12h",
    "wspd10_delta_3h",
    "wspd10_delta_6h",
    "wspd10_std_12h",
    "wspd10_range_12h",
    "rh_acceleration",
    "humid_cold_proxy",
    "night_low_cloud_proxy",
    "cold_humid_weak_wind_flag",
    "rh_cloud_ratio",
    "rh2m_quadratic",
    "low_level_wind_shear",
    "wind_direction_turning",
    "convective_wet_proxy",
    "daytime_mixing_proxy",
    "ventilation_proxy",
    "moisture_stratification",
    "omega_contrast",
    "warm_instability_proxy",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]

CLASS_COLORS = ["#7C1737", "#F0A737", "#D7DEE8"]
CLASS_NAMES = ["0-500 m", "500-1000 m", ">=1000 m"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLASS_CMAP.N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw stackable China station-map assets for the PMST architecture figure."
    )
    parser.add_argument("--base", default=str(BASE_PATH))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--shp", default=str(DEFAULT_SHP))
    parser.add_argument("--oro", default=str(DEFAULT_ORO))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--time", default="", help="UTC valid time, e.g. 2025-10-31 15:00")
    parser.add_argument(
        "--dynamic_vars",
        default="RH2M,T2M,PM10,WSPD10,LCC",
        help="Comma-separated representative dynamic variables to draw, e.g. RH2M,T2M,PM10,WSPD10,LCC.",
    )
    parser.add_argument(
        "--dynamic_var",
        default="",
        help="Backward-compatible single dynamic variable option; appended to --dynamic_vars if set.",
    )
    parser.add_argument(
        "--dynamic_lags",
        default="-11,-7,-3,0",
        help="Comma-separated window lags to draw for dynamic stacks. Default: -11,-7,-3,0.",
    )
    parser.add_argument(
        "--static_vars",
        default="terrain_h,terrain_anomaly,terrain_std,veg",
        help="Comma-separated static station inputs to draw: terrain_h,terrain_anomaly,terrain_std,lat,lon,veg.",
    )
    parser.add_argument(
        "--fe_vars",
        default="near_saturation_proxy,ventilation_proxy,fog_potential_index",
        help="Comma-separated FE variables to include in the static/FE stack.",
    )
    parser.add_argument(
        "--output_times",
        type=int,
        default=3,
        help="Number of output class maps to draw.",
    )
    parser.add_argument(
        "--output_min_gap_hours",
        type=float,
        default=24.0,
        help="Preferred minimum temporal gap between selected output maps.",
    )
    parser.add_argument(
        "--allow_same_day_outputs",
        action="store_true",
        help="Allow output maps from the same UTC date; off by default for visual diversity.",
    )
    parser.add_argument(
        "--output_min_lowvis",
        type=int,
        default=40,
        help="Preferred minimum observed low-visibility station count when selecting output times.",
    )
    parser.add_argument(
        "--output_min_precision",
        type=float,
        default=0.22,
        help="Preferred minimum low-visibility precision when selecting output times from eval_csv.",
    )
    parser.add_argument(
        "--output_min_recall",
        type=float,
        default=0.35,
        help="Preferred minimum low-visibility recall when selecting output times from eval_csv.",
    )
    parser.add_argument(
        "--eval_csv",
        default="",
        help="Optional per-sample evaluation CSV used to choose and draw high-skill PMST output maps.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional cap for station points per map; 0 keeps all stations at the selected time.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--asset_width", type=float, default=3.0)
    parser.add_argument("--asset_height", type=float, default=3.05)
    parser.add_argument(
        "--draw_grid_orography",
        action="store_true",
        help="Draw a gridded China terrain map. Kept for backward compatibility; this is on by default.",
    )
    parser.add_argument(
        "--skip_grid_orography",
        dest="draw_grid_orography",
        action="store_false",
        help="Skip the default gridded China terrain map.",
    )
    parser.set_defaults(draw_grid_orography=True)
    parser.add_argument(
        "--write_sheets",
        action="store_true",
        help="Also write unlabeled contact sheets for quick inspection.",
    )
    return parser.parse_args()


def resolve_layout(total_dim: int, window: int) -> Tuple[int, int]:
    rest = int(total_dim) - 6
    if rest <= 0:
        raise ValueError(f"X column count is too small: D={total_dim}")
    for dyn in (27, 26, 25, 24):
        fe = rest - dyn * int(window)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot resolve X layout: total_dim={total_dim}, window={window}")


def dynamic_names(dyn_count: int) -> List[str]:
    names = list(BASE_DYN_NAMES)
    if dyn_count >= 25:
        names.append("ZENITH")
    if dyn_count == 26:
        names.append("PM10")
    elif dyn_count >= 27:
        names.extend(["PM10", "PM2.5"])
    while len(names) < dyn_count:
        names.append(f"DYN_{len(names)}")
    return names[:dyn_count]


def parse_dynamic_lags(spec: str, window: int) -> List[int]:
    """Convert lag strings such as -11,-7,-3,0 into zero-based window indices."""
    out: List[int] = []
    for raw in str(spec).split(","):
        s = raw.strip().lower().replace("t", "")
        if not s:
            continue
        lag = int(s)
        idx = window - 1 + lag if lag <= 0 else lag
        if idx < 0 or idx >= window:
            raise ValueError(f"Dynamic lag/index {raw!r} is outside a {window}-hour window")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("No valid dynamic lags were provided")
    return out


def split_items(spec: str) -> List[str]:
    return [x.strip() for x in str(spec).replace(";", ",").split(",") if x.strip()]


def read_shp_geometry(shp_path: Path) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    try:
        data = shp_path.read_bytes()
        if len(data) < 100:
            return {"segments": [], "polygons": []}
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        polygons: List[Tuple[np.ndarray, np.ndarray]] = []
        pos = 100
        while pos + 8 <= len(data):
            content_bytes = int(struct.unpack(">i", data[pos + 4 : pos + 8])[0]) * 2
            pos += 8
            rec = data[pos : pos + content_bytes]
            pos += content_bytes
            if len(rec) < 44:
                continue
            shape_type = int(struct.unpack("<i", rec[:4])[0])
            if shape_type == 0:
                continue
            if shape_type not in (3, 5, 13, 15):
                continue
            num_parts, num_points = struct.unpack("<2i", rec[36:44])
            parts_start = 44
            points_start = parts_start + 4 * int(num_parts)
            points_bytes = 16 * int(num_points)
            if num_parts <= 0 or num_points <= 0 or len(rec) < points_start + points_bytes:
                continue
            parts = list(struct.unpack(f"<{int(num_parts)}i", rec[parts_start:points_start]))
            parts.append(int(num_points))
            points = np.frombuffer(
                rec, dtype="<f8", count=int(num_points) * 2, offset=points_start
            ).reshape(int(num_points), 2)
            for i in range(len(parts) - 1):
                a, b = int(parts[i]), int(parts[i + 1])
                if b > a:
                    seg = points[a:b].copy()
                    segments.append((seg[:, 0], seg[:, 1]))
                    if shape_type in (5, 15):
                        polygons.append((seg[:, 0], seg[:, 1]))
        return {"segments": segments, "polygons": polygons}
    except Exception as exc:
        print(f"[WARN] Pure-Python shapefile reader failed for {shp_path}: {exc}", flush=True)
        return {"segments": [], "polygons": []}


def load_boundary(shp_path: Path):
    if not shp_path.exists():
        print(f"[WARN] Shapefile not found: {shp_path}; drawing without boundary.", flush=True)
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(str(shp_path))
    except Exception as exc:
        print(f"[WARN] geopandas could not read {shp_path}: {exc}", flush=True)
    return read_shp_geometry(shp_path)


def fill_boundary_area(ax, boundary, color: str = MAP_BACKGROUND, zorder: int = 0) -> None:
    if boundary is None:
        return
    if hasattr(boundary, "plot"):
        try:
            boundary.plot(ax=ax, facecolor=color, edgecolor="none", linewidth=0, zorder=zorder)
            return
        except Exception as exc:
            print(f"[WARN] Could not fill boundary polygons with geopandas: {exc}", flush=True)
    for xs, ys in boundary.get("polygons") or []:
        ax.fill(xs, ys, facecolor=color, edgecolor="none", linewidth=0, zorder=zorder)


def make_boundary_clip_patch(ax, boundary) -> Optional[PathPatch]:
    if boundary is None:
        return None

    vertices: List[Tuple[float, float]] = []
    codes: List[int] = []

    def add_ring(coords: Sequence[Tuple[float, float]]) -> None:
        pts = np.asarray(coords, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return
        pts = pts[:, :2]
        if not np.all(np.isfinite(pts)):
            pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 3:
            return
        vertices.extend(map(tuple, pts))
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (pts.shape[0] - 1))
        vertices.append(tuple(pts[0]))
        codes.append(MplPath.CLOSEPOLY)

    if hasattr(boundary, "geometry"):
        for geom in boundary.geometry:
            for poly in getattr(geom, "geoms", [geom]):
                exterior = getattr(poly, "exterior", None)
                if exterior is not None:
                    add_ring(list(exterior.coords))
    else:
        for xs, ys in boundary.get("polygons") or []:
            add_ring(list(zip(xs, ys)))

    if not vertices:
        return None
    return PathPatch(MplPath(vertices, codes), transform=ax.transData)


def draw_boundary(ax, boundary, color: str = "#212529", lw: float = 0.55, zorder: int = 5) -> None:
    if boundary is None:
        return
    if hasattr(boundary, "boundary"):
        boundary.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=zorder)
        return
    for xs, ys in boundary.get("segments") or []:
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder)


def style_map_ax(ax, boundary, compact: bool = True) -> None:
    ax.set_xlim(CHINA_EXTENT[0], CHINA_EXTENT[1])
    ax.set_ylim(CHINA_EXTENT[2], CHINA_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("none")
    fill_boundary_area(ax, boundary, zorder=0)
    draw_boundary(ax, boundary, color="#242424", lw=0.45, zorder=6)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def robust_limits(values: np.ndarray, symmetric: bool = False) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    if symmetric:
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    return float(lo), float(hi)


def draw_station_value_map(
    df: pd.DataFrame,
    value: np.ndarray,
    out_path: Path,
    boundary,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 220,
    figsize: Tuple[float, float] = (3.0, 2.45),
    annotate: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    style_map_ax(ax, boundary, compact=True)
    vals = np.asarray(value, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(df["lon"].to_numpy(float)) & np.isfinite(df["lat"].to_numpy(float))
    if valid.any():
        ax.scatter(
            df.loc[valid, "lon"],
            df.loc[valid, "lat"],
            c=vals[valid],
            s=7,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.92,
            zorder=4,
        )
    fig.savefig(
        out_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def draw_class_map(
    df: pd.DataFrame,
    classes: np.ndarray,
    out_path: Path,
    boundary,
    title: str = "",
    dpi: int = 220,
    figsize: Tuple[float, float] = (3.0, 2.45),
    annotate: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    style_map_ax(ax, boundary, compact=True)
    cls = np.asarray(classes, dtype=float)
    valid = np.isfinite(cls)
    ax.scatter(
        df.loc[valid, "lon"],
        df.loc[valid, "lat"],
        c=cls[valid],
        s=7,
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        linewidths=0,
        alpha=0.96,
        zorder=4,
    )
    fig.savefig(
        out_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def compose_stack_canvas(
    image_paths: Sequence[Path],
    offset: Tuple[int, int] = (28, 18),
    shadow: bool = True,
):
    from PIL import Image, ImageFilter

    if not image_paths:
        return None
    imgs = [Image.open(p).convert("RGBA") for p in image_paths]
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    canvas_w = max_w + offset[0] * (len(imgs) - 1)
    canvas_h = max_h + offset[1] * (len(imgs) - 1)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    for i, im in enumerate(imgs):
        x = offset[0] * i
        y = offset[1] * (len(imgs) - 1 - i)
        if shadow:
            alpha = im.split()[-1]
            sh = Image.new("RGBA", im.size, (25, 25, 25, 56))
            sh.putalpha(alpha.filter(ImageFilter.GaussianBlur(2.5)))
            canvas.alpha_composite(sh, (x + 5, y + 5))
        canvas.alpha_composite(im, (x, y))
    return canvas


def compose_stack(
    image_paths: Sequence[Path],
    out_path: Path,
    offset: Tuple[int, int] = (28, 18),
    shadow: bool = True,
) -> None:
    canvas = compose_stack_canvas(image_paths, offset=offset, shadow=shadow)
    if canvas is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_output_class_colorbar(out_path: Path, dpi: int = 220) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.05, 0.82))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    box = FancyBboxPatch(
        (0.035, 0.075),
        0.93,
        0.85,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        transform=ax.transAxes,
        facecolor=(1.0, 1.0, 1.0, 0.94),
        edgecolor="#9AA0A6",
        linewidth=0.75,
    )
    ax.add_patch(box)
    ys = [0.70, 0.50, 0.30]
    for y, color, label in zip(ys, CLASS_COLORS, CLASS_NAMES):
        ax.scatter([0.18], [y], s=34, c=[color], edgecolors="none", transform=ax.transAxes, zorder=3)
        ax.text(
            0.30,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.8,
            fontfamily="DejaVu Serif",
            color="#1F2933",
        )
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def compose_stack_with_colorbar(
    image_paths: Sequence[Path],
    colorbar_path: Path,
    out_path: Path,
    offset: Tuple[int, int] = (28, 18),
    shadow: bool = True,
) -> None:
    from PIL import Image

    stack = compose_stack_canvas(image_paths, offset=offset, shadow=shadow)
    if stack is None:
        return
    bar = Image.open(colorbar_path).convert("RGBA")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    max_bar_w = max(1, int(stack.width * 0.62))
    if bar.width > max_bar_w:
        new_h = max(1, int(round(bar.height * max_bar_w / bar.width)))
        bar = bar.resize((max_bar_w, new_h), resample)
    pad = max(10, int(round(stack.width * 0.045)))
    gap = max(8, int(round(stack.height * 0.035)))
    canvas_w = max(stack.width, bar.width + pad * 2)
    canvas_h = stack.height + gap + bar.height
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    stack_x = (canvas_w - stack.width) // 2
    canvas.alpha_composite(stack, (stack_x, 0))
    bar_x = min(canvas_w - bar.width - pad, max(pad, stack_x + int(round(stack.width * 0.08))))
    canvas.alpha_composite(bar, (bar_x, stack.height + gap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def compose_angled_window_stack(
    image_paths: Sequence[Path],
    out_path: Path,
    panel_width: int = 150,
    x_step: int = 10,
    y_step: int = 3,
    shear: float = 0.18,
    rotation: float = -5.0,
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    if not image_paths:
        return
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    affine = getattr(getattr(Image, "Transform", Image), "AFFINE")
    panels = []
    for p in image_paths:
        im = Image.open(p).convert("RGBA")
        bbox = im.getbbox()
        if bbox:
            im = im.crop(bbox)
        scale = panel_width / max(1, im.width)
        im = im.resize((panel_width, max(1, int(round(im.height * scale)))), resampling)
        pad = max(4, int(round(panel_width * 0.035)))
        card = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(card)
        rect = [1, 1, card.width - 2, card.height - 2]
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(rect, radius=4, fill=(255, 255, 255, 245), outline=(68, 135, 195, 215))
            draw.rounded_rectangle([3, 3, card.width - 4, card.height - 4], radius=3, outline=(68, 135, 195, 135))
        else:
            draw.rectangle(rect, fill=(255, 255, 255, 245), outline=(68, 135, 195, 215))
            draw.rectangle([3, 3, card.width - 4, card.height - 4], outline=(68, 135, 195, 135))
        card.alpha_composite(im, (pad, pad))
        skew_w = int(round(card.width + abs(shear) * card.height))
        slanted = card.transform((skew_w, card.height), affine, (1, -shear, 0, 0, 1, 0), resample=bicubic)
        slanted = slanted.rotate(rotation, resample=bicubic, expand=True)
        panels.append(slanted)

    max_w = max(panel.width for panel in panels)
    max_h = max(panel.height for panel in panels)
    margin = 10
    canvas_w = max_w + x_step * (len(panels) - 1) + margin * 2
    canvas_h = max_h + y_step * (len(panels) - 1) + margin * 2
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    for i, panel in enumerate(panels):
        x = margin + x_step * i
        y = margin + y_step * (len(panels) - 1 - i)
        alpha = panel.split()[-1]
        shadow = Image.new("RGBA", panel.size, (18, 38, 64, 44))
        shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(2.0)))
        canvas.alpha_composite(shadow, (x + 3, y + 4))
        canvas.alpha_composite(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_contact_sheet(
    image_paths: Sequence[Path],
    out_path: Path,
    ncols: int = 4,
    label_prefix: str = "",
) -> None:
    from PIL import Image

    if not image_paths:
        return
    imgs = [Image.open(p).convert("RGBA") for p in image_paths]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    nrows = int(math.ceil(len(imgs) / ncols))
    sheet = Image.new("RGBA", (ncols * w, nrows * h), (255, 255, 255, 0))
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        x = c * w
        y = r * h
        sheet.alpha_composite(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def load_test_data(data_dir: Path, window: int) -> Tuple[np.memmap, np.ndarray, pd.DataFrame, int, int]:
    x_path = data_dir / "X_test.npy"
    y_path = data_dir / "y_test.npy"
    meta_path = data_dir / "meta_test.csv"
    missing = [str(p) for p in (x_path, y_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required test-set files: " + "; ".join(missing))
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    meta = pd.read_csv(meta_path)
    meta["time"] = pd.to_datetime(meta["time"], errors="coerce")
    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, meta={len(meta)}")
    dyn_count, fe_dim = resolve_layout(X.shape[1], window)
    return X, y, meta, dyn_count, fe_dim


def labels_from_visibility(y_raw: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw, dtype=float).copy()
    if np.nanmax(y) < 100:
        y *= 1000.0
    cls = np.zeros(len(y), dtype=np.int16)
    cls[y >= 500.0] = 1
    cls[y >= 1000.0] = 2
    return cls


def choose_time(meta: pd.DataFrame, y_cls: np.ndarray, requested: str = "") -> pd.Timestamp:
    valid_time = meta["time"].notna().to_numpy()
    if requested:
        t_req = pd.Timestamp(requested)
        available = pd.DatetimeIndex(meta.loc[valid_time, "time"].unique())
        if t_req not in set(available):
            nearest = available[np.argmin(np.abs(available - t_req))]
            print(f"[WARN] Requested time not found; using nearest available time {nearest}.", flush=True)
            return pd.Timestamp(nearest)
        return t_req

    df = pd.DataFrame({"time": meta["time"], "y": y_cls})
    df = df[df["time"].notna()]
    counts = (
        df.assign(fog=(df["y"] == 0).astype(int), mist=(df["y"] == 1).astype(int))
        .groupby("time", sort=True)[["fog", "mist"]]
        .sum()
        .sort_values(["fog", "mist"], ascending=False)
    )
    if counts.empty:
        raise ValueError("No valid test times found in meta_test.csv")
    return pd.Timestamp(counts.index[0])


def time_slice(meta: pd.DataFrame, target_time: pd.Timestamp, max_points: int = 0) -> np.ndarray:
    idx = np.flatnonzero(meta["time"].to_numpy(dtype="datetime64[ns]") == np.datetime64(target_time))
    if max_points and len(idx) > max_points:
        rng = np.random.default_rng(20260514)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    if len(idx) == 0:
        raise ValueError(f"No rows found for time {target_time}")
    return idx


def draw_grid_orography(oro_path: Path, out_path: Path, boundary, dpi: int) -> bool:
    if not oro_path.exists():
        print(f"[WARN] Orography file not found: {oro_path}", flush=True)
        return False
    try:
        import xarray as xr
    except Exception as exc:
        print(f"[WARN] xarray unavailable; skip gridded orography: {exc}", flush=True)
        return False
    try:
        ds = xr.open_dataset(oro_path)
        if "h" not in ds:
            print(f"[WARN] Orography variable 'h' not found in {oro_path}", flush=True)
            return False
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        da = ds["h"]
        lats = np.asarray(ds[lat_name].values)
        lons = np.asarray(ds[lon_name].values)
        lons_plot = ((lons + 180.0) % 360.0) - 180.0
        order = np.argsort(lons_plot)
        lons_plot = lons_plot[order]
        vals = np.asarray(da.values)
        if da.dims.index(lon_name) == 1:
            vals = vals[:, order]
        else:
            vals = vals[order, :]
        lat_mask = (lats >= CHINA_EXTENT[2] - 2) & (lats <= CHINA_EXTENT[3] + 2)
        lon_mask = (lons_plot >= CHINA_EXTENT[0] - 2) & (lons_plot <= CHINA_EXTENT[1] + 2)
        vals = vals[np.ix_(lat_mask, lon_mask)]
        lats = lats[lat_mask]
        lons_plot = lons_plot[lon_mask]
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        fig.patch.set_alpha(0)
        style_map_ax(ax, boundary, compact=True)
        lo, hi = robust_limits(vals)
        mesh = ax.pcolormesh(
            lons_plot,
            lats,
            vals,
            cmap=TOPOGRAPHY_CMAP,
            shading="auto",
            vmin=lo,
            vmax=hi,
            zorder=1,
        )
        clip_patch = make_boundary_clip_patch(ax, boundary)
        if clip_patch is not None:
            mesh.set_clip_path(clip_patch)
        draw_boundary(ax, boundary, color="#202020", lw=0.5, zorder=6)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to draw gridded orography: {exc}", flush=True)
        return False


def load_eval_csv(eval_csv: str) -> Optional[pd.DataFrame]:
    if not eval_csv:
        return None
    p = Path(eval_csv)
    if not p.exists():
        print(f"[WARN] eval_csv not found: {p}", flush=True)
        return None
    df = pd.read_csv(p)
    if "time" not in df:
        print(f"[WARN] eval_csv has no time column: {p}", flush=True)
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "station_id" in df:
        df["station_id"] = df["station_id"].astype(str)
    return df


def attach_meta_to_eval(eval_df: Optional[pd.DataFrame], meta: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Add station coordinates to eval rows when the eval CSV omits them."""
    if eval_df is None:
        return None
    out = eval_df.copy()
    meta_ref = meta.copy()
    if "station_id" in meta_ref:
        meta_ref["station_id"] = meta_ref["station_id"].astype(str)
    if len(out) == len(meta_ref):
        for col in ("station_id", "lat", "lon"):
            if col not in out and col in meta_ref:
                out[col] = meta_ref[col].to_numpy()
        if "time" not in out and "time" in meta_ref:
            out["time"] = meta_ref["time"].to_numpy()
    if "time" in out:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if "station_id" in out:
        out["station_id"] = out["station_id"].astype(str)

    if {"lat", "lon"}.issubset(out.columns):
        return out
    if {"time", "station_id"}.issubset(out.columns) and {"time", "station_id", "lat", "lon"}.issubset(meta_ref.columns):
        coords = meta_ref[["time", "station_id", "lat", "lon"]].drop_duplicates(["time", "station_id"])
        out = out.merge(coords, on=["time", "station_id"], how="left")
    elif "station_id" in out and {"station_id", "lat", "lon"}.issubset(meta_ref.columns):
        coords = meta_ref[["station_id", "lat", "lon"]].drop_duplicates("station_id")
        out = out.merge(coords, on="station_id", how="left")
    return out


def eval_class_columns(eval_df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str]]:
    if eval_df is None:
        return None, None
    true_candidates = ("y_true", "target", "target_class", "label", "obs_class")
    pred_candidates = ("pmst_pred", "pred_current", "pred_class", "prediction", "argmax_fine")
    true_col = next((c for c in true_candidates if c in eval_df.columns), None)
    pred_col = next((c for c in pred_candidates if c in eval_df.columns), None)
    return true_col, pred_col


def select_diverse_time_rows(
    candidates: pd.DataFrame,
    n: int,
    score_columns: Sequence[str],
    min_gap_hours: float,
    prefer_unique_dates: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.head(0).copy()
    score_columns = [c for c in score_columns if c in candidates.columns]
    sort_by = score_columns or ["time"]
    sort_ascending = [False] * len(sort_by) if score_columns else [True]
    ranked = candidates.sort_values(sort_by, ascending=sort_ascending)
    ranked = ranked.reset_index(drop=True)

    chosen: List[int] = []
    chosen_times: List[pd.Timestamp] = []
    chosen_dates = set()
    min_gap_seconds = max(0.0, float(min_gap_hours)) * 3600.0

    def can_take(row: pd.Series, require_new_date: bool, require_gap: bool) -> bool:
        t = pd.Timestamp(row["time"])
        if require_new_date and t.date() in chosen_dates:
            return False
        if require_gap and any(abs((t - old).total_seconds()) < min_gap_seconds for old in chosen_times):
            return False
        return True

    def take_from_ranked(require_new_date: bool, require_gap: bool) -> None:
        for idx, row in ranked.iterrows():
            if len(chosen) >= n:
                return
            if idx in chosen:
                continue
            if not can_take(row, require_new_date=require_new_date, require_gap=require_gap):
                continue
            chosen.append(int(idx))
            t = pd.Timestamp(row["time"])
            chosen_times.append(t)
            chosen_dates.add(t.date())

    take_from_ranked(require_new_date=prefer_unique_dates, require_gap=min_gap_seconds > 0)
    if len(chosen) < n and prefer_unique_dates:
        take_from_ranked(require_new_date=True, require_gap=False)
    if len(chosen) < n and min_gap_seconds > 0:
        take_from_ranked(require_new_date=False, require_gap=True)
    if len(chosen) < n:
        take_from_ranked(require_new_date=False, require_gap=False)

    selected = ranked.iloc[chosen[:n]].copy()
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["selection_date_utc"] = selected["time"].map(lambda t: str(pd.Timestamp(t).date()))
    return selected


def choose_output_times(
    meta: pd.DataFrame,
    y_cls: np.ndarray,
    n: int,
    eval_df: Optional[pd.DataFrame],
    min_lowvis: int,
    min_precision: float,
    min_recall: float,
    min_gap_hours: float,
    prefer_unique_dates: bool,
) -> Tuple[List[pd.Timestamp], pd.DataFrame]:
    """Pick visually useful output times, preferring good low-vis PMST skill."""
    n = max(1, int(n))
    true_col, pred_col = eval_class_columns(eval_df)
    if eval_df is not None and true_col and pred_col and "time" in eval_df.columns:
        tmp = eval_df[["time", true_col, pred_col]].copy()
        tmp = tmp[tmp["time"].notna()]
        tmp["true_low"] = tmp[true_col].astype(float) <= 1
        tmp["pred_low"] = tmp[pred_col].astype(float) <= 1
        tmp["hit_low"] = tmp["true_low"] & tmp["pred_low"]
        grouped = tmp.groupby("time", sort=True).agg(
            n_total=(true_col, "size"),
            lowvis_count=("true_low", "sum"),
            pred_lowvis_count=("pred_low", "sum"),
            lowvis_hits=("hit_low", "sum"),
        )
        grouped["lowvis_precision"] = np.divide(
            grouped["lowvis_hits"],
            grouped["pred_lowvis_count"],
            out=np.zeros(len(grouped), dtype=float),
            where=grouped["pred_lowvis_count"].to_numpy() > 0,
        )
        grouped["lowvis_recall"] = np.divide(
            grouped["lowvis_hits"],
            grouped["lowvis_count"],
            out=np.zeros(len(grouped), dtype=float),
            where=grouped["lowvis_count"].to_numpy() > 0,
        )
        p = grouped["lowvis_precision"].to_numpy(float)
        r = grouped["lowvis_recall"].to_numpy(float)
        grouped["lowvis_f1"] = np.divide(
            2 * p * r,
            p + r,
            out=np.zeros(len(grouped), dtype=float),
            where=(p + r) > 0,
        )
        max_low = max(float(grouped["lowvis_count"].max()), 1.0)
        grouped["count_score"] = np.sqrt(grouped["lowvis_count"].clip(lower=0).astype(float) / max_low)
        grouped["selection_score"] = 0.78 * grouped["lowvis_f1"] + 0.22 * grouped["count_score"]
        grouped = grouped.reset_index()

        preferred = grouped[
            (grouped["lowvis_count"] >= int(min_lowvis))
            & (grouped["lowvis_precision"] >= float(min_precision))
            & (grouped["lowvis_recall"] >= float(min_recall))
        ]
        if len(preferred) < n:
            relaxed = grouped[
                (grouped["lowvis_count"] >= max(5, int(round(min_lowvis * 0.5))))
                & (grouped["lowvis_precision"] >= float(min_precision) * 0.75)
                & (grouped["lowvis_recall"] >= float(min_recall) * 0.75)
            ]
            preferred = pd.concat([preferred, relaxed], ignore_index=True).drop_duplicates("time")
        if len(preferred) < n:
            low_count_pool = grouped[grouped["lowvis_count"] >= max(1, int(round(min_lowvis * 0.25)))]
            preferred = pd.concat([preferred, low_count_pool], ignore_index=True).drop_duplicates("time")
        if len(preferred) < n:
            preferred = pd.concat([preferred, grouped], ignore_index=True).drop_duplicates("time")

        selected = select_diverse_time_rows(
            preferred,
            n,
            ["selection_score", "lowvis_count", "lowvis_precision", "lowvis_recall"],
            min_gap_hours=min_gap_hours,
            prefer_unique_dates=prefer_unique_dates,
        ).head(n)
        selected = selected.assign(selection_source="eval_csv", prediction_column=pred_col, truth_column=true_col)
        return [pd.Timestamp(t) for t in selected["time"]], selected.reset_index(drop=True)

    fallback = pd.DataFrame({"time": meta["time"], "y_true": y_cls})
    fallback = fallback[fallback["time"].notna()]
    grouped = (
        fallback.assign(lowvis=(fallback["y_true"].astype(float) <= 1))
        .groupby("time", sort=True)
        .agg(n_total=("y_true", "size"), lowvis_count=("lowvis", "sum"))
        .reset_index()
    )
    grouped["selection_score"] = grouped["lowvis_count"].astype(float)
    selected = select_diverse_time_rows(
        grouped,
        n,
        ["lowvis_count", "selection_score"],
        min_gap_hours=min_gap_hours,
        prefer_unique_dates=prefer_unique_dates,
    ).head(n)
    selected = selected.assign(selection_source="observed_class_fallback")
    return [pd.Timestamp(t) for t in selected["time"]], selected.reset_index(drop=True)


def eval_slice_for_time(
    eval_df: Optional[pd.DataFrame], target_time: pd.Timestamp, max_points: int = 0
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    _, pred_col = eval_class_columns(eval_df)
    if eval_df is None or not pred_col or "time" not in eval_df:
        return None, None
    sub = eval_df[eval_df["time"].to_numpy(dtype="datetime64[ns]") == np.datetime64(target_time)].copy()
    if sub.empty or not {"lat", "lon"}.issubset(sub.columns):
        return None, None
    valid = np.isfinite(sub["lat"].to_numpy(float)) & np.isfinite(sub["lon"].to_numpy(float))
    sub = sub.loc[valid].copy()
    if sub.empty:
        return None, None
    if max_points and len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=20260514).sort_index()
    return sub, pred_col


def dataframe_records_for_json(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    out = df.copy()
    if "time" in out:
        out["time"] = out["time"].astype(str)
    return json.loads(out.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    shp_path = Path(args.shp)
    oro_path = Path(args.oro)
    out_dir = Path(args.out_dir)
    layer_dir = out_dir / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary = load_boundary(shp_path)
    X, y_raw, meta, dyn_count, fe_dim = load_test_data(data_dir, args.window)
    dyn_names = dynamic_names(dyn_count)
    y_cls = labels_from_visibility(y_raw)
    eval_df = attach_meta_to_eval(load_eval_csv(args.eval_csv), meta)
    selected_time = choose_time(meta, y_cls, args.time)
    rows = time_slice(meta, selected_time, args.max_points)
    df_t = meta.iloc[rows].copy()
    X_t = np.asarray(X[rows], dtype=np.float32)

    split_dyn = args.window * dyn_count
    split_static = split_dyn + 5
    dyn_seq = X_t[:, :split_dyn].reshape(len(rows), args.window, dyn_count)
    static_cont = X_t[:, split_dyn:split_static]
    fe = X_t[:, split_static + 1 : split_static + 1 + fe_dim]
    dynamic_indices = parse_dynamic_lags(args.dynamic_lags, args.window)

    dyn_lookup = {name.lower(): i for i, name in enumerate(dyn_names)}
    requested_dyn = split_items(args.dynamic_vars)
    if args.dynamic_var.strip():
        requested_dyn.append(args.dynamic_var.strip())
    seen_dyn = set()
    requested_dyn = [x for x in requested_dyn if not (x.lower() in seen_dyn or seen_dyn.add(x.lower()))]

    dynamic_outputs: Dict[str, Dict[str, object]] = {}
    for dyn_var in requested_dyn:
        if dyn_var.lower() not in dyn_lookup:
            raise ValueError(f"Unknown dynamic var {dyn_var!r}. Available: {', '.join(dyn_names)}")
        dyn_idx = dyn_lookup[dyn_var.lower()]
        dyn_values = dyn_seq[:, :, dyn_idx]
        dyn_vmin, dyn_vmax = robust_limits(dyn_values)
        if dyn_names[dyn_idx] in ("DPD", "INVERSION"):
            cmap = "RdBu_r"
        elif dyn_names[dyn_idx] in ("PM10", "PM2.5", "PRECIP"):
            cmap = "YlOrRd"
        elif dyn_names[dyn_idx] in ("T2M", "T_925"):
            cmap = "coolwarm"
        else:
            cmap = HUMIDITY_CMAP
        dynamic_paths: List[Path] = []
        for t in dynamic_indices:
            lag = t - (args.window - 1)
            safe_name = dyn_names[dyn_idx].replace(".", "p")
            lag_name = f"m{abs(lag):02d}" if lag < 0 else "t00"
            p = layer_dir / "dynamic" / safe_name / f"dynamic_{safe_name}_{lag_name}.png"
            draw_station_value_map(
                df_t,
                dyn_values[:, t],
                p,
                boundary,
                cmap=cmap,
                vmin=dyn_vmin,
                vmax=dyn_vmax,
                dpi=args.dpi,
                figsize=(args.asset_width, args.asset_height),
            )
            dynamic_paths.append(p)
        stack_path = out_dir / f"stack_dynamic_{safe_name}.png"
        sheet_path = out_dir / f"sheet_dynamic_{safe_name}.png"
        compose_stack(dynamic_paths, stack_path)
        if args.write_sheets:
            draw_contact_sheet(dynamic_paths, sheet_path, ncols=4)
        dynamic_outputs[dyn_names[dyn_idx]] = {
            "layers": [str(p) for p in dynamic_paths],
            "stack": str(stack_path),
        }
        if args.write_sheets:
            dynamic_outputs[dyn_names[dyn_idx]]["sheet"] = str(sheet_path)

    rh2m_window_assets: Dict[str, object] = {}
    if "rh2m" in dyn_lookup:
        rh_idx = dyn_lookup["rh2m"]
        rh_values = dyn_seq[:, :, rh_idx]
        rh_vmin, rh_vmax = robust_limits(rh_values)
        rh_window_paths: List[Path] = []
        for t in range(args.window):
            lag = t - (args.window - 1)
            lag_name = f"m{abs(lag):02d}" if lag < 0 else "t00"
            p = layer_dir / "dynamic" / "RH2M_12h_window" / f"rh2m_window_{lag_name}.png"
            draw_station_value_map(
                df_t,
                rh_values[:, t],
                p,
                boundary,
                cmap=HUMIDITY_CMAP,
                vmin=rh_vmin,
                vmax=rh_vmax,
                dpi=args.dpi,
                figsize=(args.asset_width, args.asset_height),
            )
            rh_window_paths.append(p)
        rh_window_stack = out_dir / "stack_dynamic_RH2M_12h_angled.png"
        compose_angled_window_stack(rh_window_paths, rh_window_stack)
        rh2m_window_assets = {
            "layers": [str(p) for p in rh_window_paths],
            "angled_stack": str(rh_window_stack),
        }
    else:
        print("[WARN] RH2M is unavailable; skip RH2M 12-h angled stack.", flush=True)

    static_lookup = {
        "lat": (static_cont[:, 0], HUMIDITY_CMAP),
        "lon": (static_cont[:, 1], HUMIDITY_CMAP),
        "terrain_h": (static_cont[:, 2], TOPOGRAPHY_CMAP),
        "terrain_anomaly": (static_cont[:, 3], TERRAIN_ANOMALY_CMAP),
        "terrain_std": (static_cont[:, 4], TERRAIN_STD_CMAP),
        "veg": (X_t[:, split_static], "tab20"),
    }
    requested_static = split_items(args.static_vars)
    static_paths: List[Path] = []
    for name in requested_static:
        key = name.lower()
        if key not in static_lookup:
            print(f"[WARN] Skip unknown static variable {name!r}", flush=True)
            continue
        vals, cmap = static_lookup[key]
        if key == "veg":
            finite_vals = np.asarray(vals, dtype=float)
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if finite_vals.size:
                lo = float(np.floor(finite_vals.min()) - 0.5)
                hi = float(np.ceil(finite_vals.max()) + 0.5)
            else:
                lo, hi = -0.5, 0.5
        else:
            lo, hi = robust_limits(vals, symmetric=(key == "terrain_anomaly"))
        p = layer_dir / "static" / f"static_{key}.png"
        draw_station_value_map(
            df_t,
            vals,
            p,
            boundary,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            dpi=args.dpi,
            figsize=(args.asset_width, args.asset_height),
        )
        static_paths.append(p)

    grid_oro_path = out_dir / "static_orography_china_grid.png"
    grid_oro_written = False
    if args.draw_grid_orography:
        grid_oro_written = draw_grid_orography(oro_path, grid_oro_path, boundary, args.dpi)

    requested_fe = split_items(args.fe_vars)
    fe_lookup = {name.lower(): i for i, name in enumerate(FE_NAMES[:fe_dim])}
    fe_paths: List[Path] = []
    for name in requested_fe:
        if name.lower() not in fe_lookup:
            print(f"[WARN] Skip unknown FE variable {name!r}", flush=True)
            continue
        idx = fe_lookup[name.lower()]
        vals = fe[:, idx]
        lo, hi = robust_limits(vals)
        p = layer_dir / "feature_engineering" / f"fe_{idx:02d}_{FE_NAMES[idx]}.png"
        draw_station_value_map(
            df_t,
            vals,
            p,
            boundary,
            cmap=FEATURE_ENGINEERING_CMAP,
            vmin=lo,
            vmax=hi,
            dpi=args.dpi,
            figsize=(args.asset_width, args.asset_height),
        )
        fe_paths.append(p)
    static_fe_paths = static_paths + fe_paths
    compose_stack(static_fe_paths, out_dir / "stack_static_feature_inputs.png")
    if args.write_sheets:
        draw_contact_sheet(static_fe_paths, out_dir / "sheet_static_feature_inputs.png", ncols=max(1, min(4, len(static_fe_paths))))

    output_times, output_selection = choose_output_times(
        meta,
        y_cls,
        args.output_times,
        eval_df,
        args.output_min_lowvis,
        args.output_min_precision,
        args.output_min_recall,
        args.output_min_gap_hours,
        not args.allow_same_day_outputs,
    )
    output_selection_csv = out_dir / "selected_output_times.csv"
    output_selection.to_csv(output_selection_csv, index=False)
    output_paths: List[Path] = []
    used_prediction_output = False
    for t in output_times:
        eval_sub, pred_col = eval_slice_for_time(eval_df, t, args.max_points)
        if eval_sub is not None and pred_col:
            p = layer_dir / "outputs" / f"pmst_output_class_{t:%Y%m%d_%H%M}.png"
            draw_class_map(
                eval_sub,
                eval_sub[pred_col].to_numpy(),
                p,
                boundary,
                dpi=args.dpi,
                figsize=(args.asset_width, args.asset_height),
            )
            used_prediction_output = True
        else:
            idx = time_slice(meta, t, args.max_points)
            df_o = meta.iloc[idx].copy()
            p = layer_dir / "outputs" / f"observed_output_class_{t:%Y%m%d_%H%M}.png"
            draw_class_map(
                df_o,
                y_cls[idx],
                p,
                boundary,
                dpi=args.dpi,
                figsize=(args.asset_width, args.asset_height),
            )
        output_paths.append(p)
    output_colorbar_path = out_dir / "output_visibility_class_colorbar.png"
    draw_output_class_colorbar(output_colorbar_path, args.dpi)
    output_stack_maps_only = out_dir / "stack_output_classes_maps_only.png"
    output_stack_with_colorbar = out_dir / "stack_output_classes.png"
    compose_stack(output_paths, output_stack_maps_only)
    compose_stack_with_colorbar(output_paths, output_colorbar_path, output_stack_with_colorbar)
    if args.write_sheets:
        draw_contact_sheet(output_paths, out_dir / "sheet_output_classes.png", ncols=max(1, min(4, len(output_paths))))
    output_source = "pmst_prediction" if used_prediction_output else "observed_class_fallback"

    station_overview = meta[["station_id", "lat", "lon"]].drop_duplicates("station_id").copy()
    draw_station_value_map(
        station_overview,
        np.ones(len(station_overview)),
        out_dir / "station_distribution_china.png",
        boundary,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        dpi=args.dpi,
        figsize=(3.35, 2.75),
    )

    manifest = {
        "selected_time_utc": str(selected_time),
        "data_dir": str(data_dir),
        "x_path": str(data_dir / "X_test.npy"),
        "y_path": str(data_dir / "y_test.npy"),
        "meta_path": str(data_dir / "meta_test.csv"),
        "shp_path": str(shp_path),
        "orography_path": str(oro_path),
        "window_size": int(args.window),
        "dyn_vars_count": int(dyn_count),
        "fe_dim": int(fe_dim),
        "dynamic_vars": list(dynamic_outputs.keys()),
        "dynamic_lags": [int(i - (args.window - 1)) for i in dynamic_indices],
        "output_time_selection": {
            "requested_maps": int(args.output_times),
            "preferred_unique_dates": bool(not args.allow_same_day_outputs),
            "preferred_min_gap_hours": float(args.output_min_gap_hours),
        },
        "static_vars": [p.stem for p in static_paths],
        "feature_engineering_vars": [p.stem for p in fe_paths],
        "style": {
            "map_layers_annotation_free": True,
            "input_colorbars": False,
            "output_class_colorbar": True,
            "axis_frame": False,
            "opaque_china_fill": True,
            "transparent_outside_boundary": True,
            "china_fill_color": MAP_BACKGROUND,
            "extent": list(CHINA_EXTENT),
            "colormaps": {
                "relative_humidity": HUMIDITY_CMAP,
                "feature_engineering": "pmst_feature_warm",
                "terrain_h": "pmst_topography",
                "terrain_anomaly": TERRAIN_ANOMALY_CMAP,
                "terrain_std": TERRAIN_STD_CMAP,
            },
        },
        "outputs": {
            "dynamic": dynamic_outputs,
            "rh2m_12h_window": rh2m_window_assets,
            "static_layers": [str(p) for p in static_paths],
            "feature_layers": [str(p) for p in fe_paths],
            "static_feature_stack": str(out_dir / "stack_static_feature_inputs.png"),
            "output_source": output_source,
            "output_layers": [str(p) for p in output_paths],
            "output_class_colorbar": str(output_colorbar_path),
            "output_stack_maps_only": str(output_stack_maps_only),
            "output_stack": str(output_stack_with_colorbar),
            "output_selection_csv": str(output_selection_csv),
            "output_selection": dataframe_records_for_json(output_selection),
            "station_distribution": str(out_dir / "station_distribution_china.png"),
        },
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    if grid_oro_written:
        manifest["outputs"]["static_orography_grid"] = str(grid_oro_path)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
