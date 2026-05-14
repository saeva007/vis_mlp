#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create China station-map assets for the PMST low-visibility architecture figure.

This script is intended to run on the remote project machine, not locally.  It
uses the S2 month-tail test set and the existing China shapefile to create
stackable PNG layers similar in spirit to the NeuralGCM architecture schematic:

  - 12 dynamic-input station maps for selected dynamic variables.
  - static terrain assets from both the gridded orography file and the model's
    station-level static terrain input.
  - several feature-engineering station maps.
  - output station maps for observed target classes, and optional PMST
    prediction/probability maps if a per-sample evaluation CSV is supplied.

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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize


BASE_PATH = Path("/public/home/putianshu/vis_mlp")
DEFAULT_DATA_DIR = BASE_PATH / "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_SHP = Path("/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
DEFAULT_ORO = Path("/public/home/putianshu/vis_cnn/data_orography.nc")
DEFAULT_OUT = BASE_PATH / "architecture_map_assets"

WINDOW_SIZE = 12
CHINA_EXTENT = (73.0, 136.0, 17.0, 54.5)

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
        default="RH2M,T2M,PM10",
        help="Comma-separated dynamic variables to draw as 12-layer input stacks, e.g. RH2M,T2M,PM10.",
    )
    parser.add_argument(
        "--dynamic_var",
        default="",
        help="Backward-compatible single dynamic variable option; appended to --dynamic_vars if set.",
    )
    parser.add_argument(
        "--fe_vars",
        default="near_saturation_proxy,night_clear_sky_cooling,fog_potential_index,ventilation_proxy",
        help="Comma-separated FE variables to draw as a stack.",
    )
    parser.add_argument(
        "--output_times",
        type=int,
        default=4,
        help="Number of consecutive available test valid times to draw for output class stack.",
    )
    parser.add_argument(
        "--eval_csv",
        default="",
        help="Optional per_sample_eval.csv. If supplied, also draw PMST prediction/probability output maps.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional cap for station points per map; 0 keeps all stations at the selected time.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--asset_width", type=float, default=3.0)
    parser.add_argument("--asset_height", type=float, default=2.45)
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


def read_shp_segments(shp_path: Path) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    try:
        data = shp_path.read_bytes()
        if len(data) < 100:
            return None
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
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
        return segments if segments else None
    except Exception as exc:
        print(f"[WARN] Pure-Python shapefile reader failed for {shp_path}: {exc}", flush=True)
        return None


def load_boundary(shp_path: Path):
    if not shp_path.exists():
        print(f"[WARN] Shapefile not found: {shp_path}; drawing without boundary.", flush=True)
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(str(shp_path))
    except Exception as exc:
        print(f"[WARN] geopandas could not read {shp_path}: {exc}", flush=True)
    return {"segments": read_shp_segments(shp_path)}


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
    ax.set_facecolor("#F5F6F3")
    draw_boundary(ax, boundary, color="#242424", lw=0.45, zorder=6)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, color="#D9DED8", linewidth=0.35, zorder=0)


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
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 220,
    figsize: Tuple[float, float] = (3.0, 2.45),
    colorbar: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    style_map_ax(ax, boundary, compact=True)
    vals = np.asarray(value, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(df["lon"].to_numpy(float)) & np.isfinite(df["lat"].to_numpy(float))
    if valid.any():
        sc = ax.scatter(
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
        if colorbar:
            cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, pad=2)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_class_map(
    df: pd.DataFrame,
    classes: np.ndarray,
    out_path: Path,
    boundary,
    title: str,
    dpi: int = 220,
    figsize: Tuple[float, float] = (3.0, 2.45),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.set_title(title, fontsize=8, pad=2)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compose_stack(
    image_paths: Sequence[Path],
    out_path: Path,
    offset: Tuple[int, int] = (28, 18),
    shadow: bool = True,
) -> None:
    from PIL import Image, ImageFilter

    if not image_paths:
        return
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_contact_sheet(
    image_paths: Sequence[Path],
    out_path: Path,
    ncols: int = 4,
    label_prefix: str = "",
) -> None:
    from PIL import Image, ImageDraw

    if not image_paths:
        return
    imgs = [Image.open(p).convert("RGBA") for p in image_paths]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs) + 26
    nrows = int(math.ceil(len(imgs) / ncols))
    sheet = Image.new("RGBA", (ncols * w, nrows * h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        x = c * w
        y = r * h
        sheet.alpha_composite(im, (x, y))
        draw.text((x + 8, y + im.height + 2), f"{label_prefix}{i + 1}", fill=(30, 30, 30, 255))
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


def next_available_times(meta: pd.DataFrame, target_time: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    times = pd.DatetimeIndex(np.sort(meta["time"].dropna().unique()))
    pos = int(np.searchsorted(times.values, np.datetime64(target_time)))
    out = [pd.Timestamp(t) for t in times[pos : pos + max(n, 1)]]
    return out


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
        fig, ax = plt.subplots(figsize=(3.3, 2.65))
        style_map_ax(ax, boundary, compact=True)
        lo, hi = robust_limits(vals)
        pc = ax.pcolormesh(lons_plot, lats, vals, cmap="terrain", shading="auto", vmin=lo, vmax=hi, zorder=1)
        draw_boundary(ax, boundary, color="#202020", lw=0.5, zorder=6)
        cb = fig.colorbar(pc, ax=ax, shrink=0.72, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_title("Static terrain (gridded h)", fontsize=8, pad=2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to draw gridded orography: {exc}", flush=True)
        return False


def maybe_load_eval_csv(eval_csv: str, target_time: pd.Timestamp) -> Optional[pd.DataFrame]:
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
    return df[df["time"] == target_time].copy()


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
    selected_time = choose_time(meta, y_cls, args.time)
    rows = time_slice(meta, selected_time, args.max_points)
    df_t = meta.iloc[rows].copy()
    X_t = np.asarray(X[rows], dtype=np.float32)

    split_dyn = args.window * dyn_count
    split_static = split_dyn + 5
    dyn_seq = X_t[:, :split_dyn].reshape(len(rows), args.window, dyn_count)
    static_cont = X_t[:, split_dyn:split_static]
    fe = X_t[:, split_static + 1 : split_static + 1 + fe_dim]

    dyn_lookup = {name.lower(): i for i, name in enumerate(dyn_names)}
    requested_dyn = [x.strip() for x in args.dynamic_vars.split(",") if x.strip()]
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
            cmap = "YlGnBu"
        dynamic_paths: List[Path] = []
        for t in range(args.window):
            lag = t - (args.window - 1)
            safe_name = dyn_names[dyn_idx].replace(".", "p")
            p = layer_dir / "dynamic" / safe_name / f"dynamic_{safe_name}_t{t:02d}.png"
            draw_station_value_map(
                df_t,
                dyn_values[:, t],
                p,
                boundary,
                title=f"{dyn_names[dyn_idx]}  {lag:+d} h",
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
        draw_contact_sheet(dynamic_paths, sheet_path, ncols=4, label_prefix="t")
        dynamic_outputs[dyn_names[dyn_idx]] = {
            "layers": [str(p) for p in dynamic_paths],
            "stack": str(stack_path),
            "sheet": str(sheet_path),
        }

    static_station_h = static_cont[:, 2]
    static_lo, static_hi = robust_limits(static_station_h)
    static_station_path = layer_dir / "static" / "static_station_terrain_h.png"
    draw_station_value_map(
        df_t,
        static_station_h,
        static_station_path,
        boundary,
        title="Static station terrain h",
        cmap="gist_earth",
        vmin=static_lo,
        vmax=static_hi,
        dpi=args.dpi,
        figsize=(args.asset_width, args.asset_height),
    )
    draw_grid_orography(oro_path, out_dir / "static_orography_china_grid.png", boundary, args.dpi)

    requested_fe = [x.strip() for x in args.fe_vars.split(",") if x.strip()]
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
            title=FE_NAMES[idx],
            cmap="magma",
            vmin=lo,
            vmax=hi,
            dpi=args.dpi,
            figsize=(args.asset_width, args.asset_height),
        )
        fe_paths.append(p)
    compose_stack(fe_paths, out_dir / "stack_feature_engineering.png")
    draw_contact_sheet(fe_paths, out_dir / "sheet_feature_engineering.png", ncols=max(1, min(4, len(fe_paths))), label_prefix="FE")

    output_paths: List[Path] = []
    output_times = next_available_times(meta, selected_time, args.output_times)
    for t in output_times:
        idx = time_slice(meta, t, args.max_points)
        df_o = meta.iloc[idx].copy()
        p = layer_dir / "outputs" / f"observed_visibility_class_{t:%Y%m%d_%H%M}.png"
        draw_class_map(
            df_o,
            y_cls[idx],
            p,
            boundary,
            title=f"Observed class  {t:%Y-%m-%d %H:%M} UTC",
            dpi=args.dpi,
            figsize=(args.asset_width, args.asset_height),
        )
        output_paths.append(p)
    compose_stack(output_paths, out_dir / "stack_output_observed_classes.png")
    draw_contact_sheet(output_paths, out_dir / "sheet_output_observed_classes.png", ncols=max(1, min(4, len(output_paths))), label_prefix="Y")

    eval_sub = maybe_load_eval_csv(args.eval_csv, selected_time)
    pred_paths: List[Path] = []
    if eval_sub is not None and not eval_sub.empty:
        for col, cmap in [("pmst_p_fog", "rocket_r"), ("pmst_p_mist", "flare"), ("pmst_pred", "")]:
            if col not in eval_sub:
                continue
            p = layer_dir / "outputs" / f"{col}_{selected_time:%Y%m%d_%H%M}.png"
            if col == "pmst_pred":
                draw_class_map(
                    eval_sub,
                    eval_sub[col].to_numpy(),
                    p,
                    boundary,
                    title=f"PMST predicted class  {selected_time:%Y-%m-%d %H:%M} UTC",
                    dpi=args.dpi,
                    figsize=(args.asset_width, args.asset_height),
                )
            else:
                draw_station_value_map(
                    eval_sub,
                    eval_sub[col].to_numpy(),
                    p,
                    boundary,
                    title=col,
                    cmap="YlOrRd" if col == "pmst_p_fog" else "PuRd",
                    vmin=0.0,
                    vmax=1.0,
                    dpi=args.dpi,
                    figsize=(args.asset_width, args.asset_height),
                )
            pred_paths.append(p)
        compose_stack(pred_paths, out_dir / "stack_output_pmst_prediction_optional.png")

    station_overview = meta[["station_id", "lat", "lon"]].drop_duplicates("station_id").copy()
    draw_station_value_map(
        station_overview,
        np.ones(len(station_overview)),
        out_dir / "station_distribution_china.png",
        boundary,
        title="Test-set stations",
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
        "feature_engineering_vars": [p.stem for p in fe_paths],
        "outputs": {
            "dynamic": dynamic_outputs,
            "static_station_terrain": str(static_station_path),
            "static_orography_grid": str(out_dir / "static_orography_china_grid.png"),
            "feature_layers": [str(p) for p in fe_paths],
            "feature_stack": str(out_dir / "stack_feature_engineering.png"),
            "observed_output_layers": [str(p) for p in output_paths],
            "observed_output_stack": str(out_dir / "stack_output_observed_classes.png"),
            "station_distribution": str(out_dir / "station_distribution_china.png"),
        },
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    if pred_paths:
        manifest["outputs"]["optional_prediction_layers"] = [str(p) for p in pred_paths]
        manifest["outputs"]["optional_prediction_stack"] = str(out_dir / "stack_output_pmst_prediction_optional.png")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
