#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Pangu-Weather 24 h ONNX inference for China-region 2025 fields.

The script expects global 0.25 degree pressure-level and surface GRIB initial
fields. It keeps the model input global, then crops the model output to the
China domain and writes canonical variables used by the low-visibility
source-comparison pipeline.
"""

from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from onnx2torch import convert


UPPER_VARS = ("z", "q", "t", "u", "v")
SURFACE_VARS = ("msl", "u10", "v10", "t2m")
PRESSURE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
ERA5_ROOT_DEFAULT = "/sharedata/dataset/realtime/SD005-ERA5/0p25"

ERA5_PRESSURE_CONFIG = {
    "z": ("geopotential", ("z", "geopotential")),
    "q": ("specific_humidity", ("q", "specific_humidity")),
    "t": ("temperature", ("t", "temperature")),
    "u": ("u_component_of_wind", ("u", "u_component_of_wind")),
    "v": ("v_component_of_wind", ("v", "v_component_of_wind")),
}

ERA5_SURFACE_CONFIG = {
    "msl": ("mean_sea_level_pressure", ("msl", "mean_sea_level_pressure")),
    "u10": ("10m_u_component_of_wind", ("u10", "10u", "10m_u_component_of_wind")),
    "v10": ("10m_v_component_of_wind", ("v10", "10v", "10m_v_component_of_wind")),
    "t2m": ("2m_temperature", ("t2m", "2t", "2m_temperature")),
}

OUTPUT_ORDER = (
    "T2M",
    "MSLP",
    "U10",
    "V10",
    "WSPD10",
    "WDIR10",
    "T_1000",
    "RH_1000",
    "Q_1000",
    "DP_1000",
    "U_1000",
    "V_1000",
    "WSPD1000",
    "WDIR1000",
    "T_925",
    "RH_925",
    "Q_925",
    "DP_925",
    "U_925",
    "V_925",
    "WSPD925",
    "WDIR925",
    "INVERSION",
)


@dataclass(frozen=True)
class GribPair:
    upper_path: str
    surface_path: str
    init_time: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class Era5Init:
    init_time: pd.Timestamp


def log(message: str) -> None:
    print(message, flush=True)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_steps(value: str) -> List[int]:
    out: List[int] = []
    for raw in str(value or "").replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        step = int(raw)
        if step <= 0:
            raise ValueError("--save-steps must contain positive integers.")
        if step not in out:
            out.append(step)
    return sorted(out or [1])


def parse_init_hours(value: str) -> List[int]:
    raw_value = str(value or "").strip().lower()
    if raw_value in {"all", "*"}:
        return list(range(24))
    hours: List[int] = []
    for raw in raw_value.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        hour = int(raw)
        if hour < 0 or hour > 23:
            raise ValueError("--init-hours values must be between 0 and 23, or 'all'.")
        if hour not in hours:
            hours.append(hour)
    return sorted(hours or [0])


def infer_coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    lower = {str(name).lower(): str(name) for name in list(ds.coords) + list(ds.dims)}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Cannot infer coordinate from {candidates}; available={list(ds.coords) + list(ds.dims)}")


def open_grib(path: str, cfgrib_indexpath: str) -> xr.Dataset:
    backend_kwargs = {}
    if cfgrib_indexpath != "AUTO":
        backend_kwargs["indexpath"] = cfgrib_indexpath
    kwargs = {"engine": "cfgrib"}
    if backend_kwargs:
        kwargs["backend_kwargs"] = backend_kwargs
    return xr.open_dataset(path, **kwargs)


def discover_files(directory: str, pattern: str) -> List[str]:
    root = Path(directory).expanduser()
    files = sorted(glob.glob(str(root / pattern)))
    return [str(Path(p).resolve()) for p in files]


def load_pair_file(path: str) -> List[GribPair]:
    pairs: List[GribPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"[\s,]+", line) if p]
            if len(parts) == 2:
                pairs.append(GribPair(parts[0], parts[1], None))
            elif len(parts) >= 3:
                pairs.append(GribPair(parts[1], parts[2], pd.Timestamp(parts[0])))
            else:
                raise ValueError(f"{path}:{line_no}: expected upper surface or init_time upper surface")
    return pairs


def build_pairs(args: argparse.Namespace) -> List[GribPair]:
    if args.pair_file:
        pairs = load_pair_file(args.pair_file)
    else:
        upper = discover_files(args.upper_dir, args.upper_glob)
        surface = discover_files(args.surface_dir, args.surface_glob)
        if not upper:
            raise FileNotFoundError(f"No upper-air GRIB files found: {args.upper_dir}/{args.upper_glob}")
        if len(upper) != len(surface):
            raise ValueError(
                f"Upper/surface file counts differ: upper={len(upper)} surface={len(surface)}. "
                "Use --pair-file if filename sorting is not one-to-one."
            )
        pairs = [GribPair(u, s, None) for u, s in zip(upper, surface)]
    return pairs


def maybe_sort_lat(ds: xr.Dataset, lat_name: str) -> xr.Dataset:
    lat = np.asarray(ds[lat_name].values)
    if lat.ndim == 1 and len(lat) > 1 and float(lat[0]) < float(lat[-1]):
        return ds.sortby(lat_name, ascending=False)
    return ds


def infer_init_time(ds: xr.Dataset) -> pd.Timestamp:
    for name in ("time", "valid_time"):
        if name in ds.coords:
            values = np.asarray(ds[name].values).reshape(-1)
            if len(values):
                return pd.Timestamp(values[0])
    raise KeyError("Cannot infer init time from GRIB coordinates; use --pair-file with an init_time column.")


def validate_global_grid(lat: np.ndarray, lon: np.ndarray, require_global: bool) -> None:
    if not require_global:
        return
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Pangu model input must use 1D global latitude/longitude coordinates.")
    if len(lat) != 721 or len(lon) != 1440:
        raise ValueError(
            f"Pangu 0.25 degree model expects 721x1440 global grid; got {len(lat)}x{len(lon)}. "
            "Do not feed a China-cropped GRIB into the model; crop only after inference."
        )
    if abs(float(lon[0])) > 1.0e-5 or abs(float(lon[-1]) - 359.75) > 1.0e-3:
        raise ValueError(
            f"Pangu expects longitudes ordered from 0 to 359.75; got first={lon[0]} last={lon[-1]}."
        )


def read_initial_state(
    pair: GribPair,
    levels: Sequence[int],
    cfgrib_indexpath: str,
    require_global_grid: bool,
) -> Tuple[pd.Timestamp, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    ds_upper = open_grib(pair.upper_path, cfgrib_indexpath)
    ds_surface = open_grib(pair.surface_path, cfgrib_indexpath)
    try:
        init_time = pair.init_time if pair.init_time is not None else infer_init_time(ds_upper)

        lat_name = infer_coord_name(ds_upper, ("latitude", "lat"))
        lon_name = infer_coord_name(ds_upper, ("longitude", "lon"))
        lev_name = infer_coord_name(ds_upper, ("isobaricInhPa", "level", "plev"))
        s_lat_name = infer_coord_name(ds_surface, ("latitude", "lat"))
        s_lon_name = infer_coord_name(ds_surface, ("longitude", "lon"))

        ds_upper = maybe_sort_lat(ds_upper, lat_name)
        ds_surface = maybe_sort_lat(ds_surface, s_lat_name)

        missing_upper = [name for name in UPPER_VARS if name not in ds_upper]
        missing_surface = [name for name in SURFACE_VARS if name not in ds_surface]
        if missing_upper:
            raise KeyError(f"{pair.upper_path} is missing upper-air variables: {missing_upper}")
        if missing_surface:
            raise KeyError(f"{pair.surface_path} is missing surface variables: {missing_surface}")

        data_upper = ds_upper[list(UPPER_VARS)].sel({lev_name: list(levels)}).to_array()
        data_upper = data_upper.transpose("variable", lev_name, lat_name, lon_name)
        data_surface = ds_surface[list(SURFACE_VARS)].to_array()
        data_surface = data_surface.transpose("variable", s_lat_name, s_lon_name)

        lat = np.asarray(data_upper[lat_name].values, dtype=np.float64)
        lon = np.asarray(data_upper[lon_name].values, dtype=np.float64)
        s_lat = np.asarray(data_surface[s_lat_name].values, dtype=np.float64)
        s_lon = np.asarray(data_surface[s_lon_name].values, dtype=np.float64)
        if len(lat) != len(s_lat) or len(lon) != len(s_lon) or not np.allclose(lat, s_lat) or not np.allclose(lon, s_lon):
            raise ValueError("Upper-air and surface GRIB grids do not match after latitude sorting.")
        validate_global_grid(lat, lon, require_global_grid)

        upper_np = np.asarray(data_upper.values, dtype=np.float32)
        surface_np = np.asarray(data_surface.values, dtype=np.float32)
        return pd.Timestamp(init_time), torch.from_numpy(upper_np), torch.from_numpy(surface_np), lat, lon
    finally:
        ds_upper.close()
        ds_surface.close()


def date_span(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current = current + timedelta(days=1)


def build_era5_init_times(args: argparse.Namespace, save_steps: Sequence[int]) -> List[Era5Init]:
    valid_start = pd.Timestamp(parse_date(args.start_date))
    valid_end = pd.Timestamp(datetime.combine(parse_date(args.end_date), datetime.max.time()))
    lead_base = int(args.lead_hours)
    init_start_day = (valid_start - pd.Timedelta(hours=lead_base * max(save_steps))).date()
    init_end_day = (valid_end - pd.Timedelta(hours=lead_base * min(save_steps))).date()
    init_hours = parse_init_hours(args.init_hours)

    items: List[Era5Init] = []
    for day in date_span(init_start_day, init_end_day):
        for hour in init_hours:
            init_time = pd.Timestamp(datetime(day.year, day.month, day.day, hour))
            possible_valids = [init_time + pd.Timedelta(hours=lead_base * step) for step in save_steps]
            if any(valid_start <= vt <= valid_end for vt in possible_valids):
                items.append(Era5Init(init_time=init_time))
    return items


def era5_file_path(root: str, dataset_type: str, folder: str, init_time: pd.Timestamp) -> str:
    ts = pd.Timestamp(init_time)
    date_str = ts.strftime("%Y%m%d")
    return os.path.join(
        str(root),
        dataset_type,
        folder,
        ts.strftime("%Y"),
        ts.strftime("%m"),
        f"era5_hourly_{folder}_{date_str}_global.grib",
    )


def select_var_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower = {str(name).lower(): str(name) for name in ds.data_vars}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if len(ds.data_vars) == 1:
        return str(next(iter(ds.data_vars)))
    raise KeyError(f"Cannot find any variable from {candidates}; available={list(ds.data_vars)}")


def select_init_time(ds: xr.Dataset, init_time: pd.Timestamp) -> xr.Dataset:
    if "time" not in ds.coords and "time" not in ds.dims:
        return ds
    target = np.datetime64(pd.Timestamp(init_time).to_datetime64())
    try:
        return ds.sel(time=target)
    except Exception as exc:
        available = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds["time"].values).reshape(-1)))
        preview = ", ".join(str(v) for v in available[:4])
        raise KeyError(f"Init time {init_time} not found in ERA5 file; first available times: {preview}") from exc


def read_era5_var(
    root: str,
    dataset_type: str,
    folder: str,
    candidates: Sequence[str],
    init_time: pd.Timestamp,
    cfgrib_indexpath: str,
    levels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = era5_file_path(root, dataset_type, folder, init_time)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ds = open_grib(path, cfgrib_indexpath)
    try:
        var_name = select_var_name(ds, candidates)
        lat_name = infer_coord_name(ds, ("latitude", "lat"))
        lon_name = infer_coord_name(ds, ("longitude", "lon"))
        ds = maybe_sort_lat(ds, lat_name)
        ds = select_init_time(ds, init_time)
        da = ds[var_name]
        if levels is not None:
            lev_name = infer_coord_name(da, ("isobaricInhPa", "level", "plev"))
            da = da.sel({lev_name: list(levels)})
            da = da.transpose(lev_name, lat_name, lon_name)
        else:
            da = da.transpose(lat_name, lon_name)
        lat = np.asarray(da[lat_name].values, dtype=np.float64)
        lon = np.asarray(da[lon_name].values, dtype=np.float64)
        values = np.asarray(da.values, dtype=np.float32)
        return values, lat, lon
    finally:
        ds.close()


def read_era5_initial_state(
    item: Era5Init,
    args: argparse.Namespace,
    levels: Sequence[int],
) -> Tuple[pd.Timestamp, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    init_time = pd.Timestamp(item.init_time)
    root = str(args.era5_root)

    upper_parts: List[np.ndarray] = []
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    for var_name in UPPER_VARS:
        folder, candidates = ERA5_PRESSURE_CONFIG[var_name]
        values, lat, lon = read_era5_var(
            root,
            "pressure-hourly",
            folder,
            candidates,
            init_time,
            args.cfgrib_indexpath,
            levels=levels,
        )
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
        elif len(lat) != len(ref_lat) or len(lon) != len(ref_lon) or not np.allclose(lat, ref_lat) or not np.allclose(lon, ref_lon):
            raise ValueError(f"Grid mismatch for ERA5 pressure variable {var_name}.")
        upper_parts.append(values)

    surface_parts: List[np.ndarray] = []
    for var_name in SURFACE_VARS:
        folder, candidates = ERA5_SURFACE_CONFIG[var_name]
        values, lat, lon = read_era5_var(
            root,
            "single-hourly",
            folder,
            candidates,
            init_time,
            args.cfgrib_indexpath,
            levels=None,
        )
        if ref_lat is None or ref_lon is None:
            ref_lat, ref_lon = lat, lon
        elif len(lat) != len(ref_lat) or len(lon) != len(ref_lon) or not np.allclose(lat, ref_lat) or not np.allclose(lon, ref_lon):
            raise ValueError(f"Grid mismatch for ERA5 surface variable {var_name}.")
        surface_parts.append(values)

    if ref_lat is None or ref_lon is None:
        raise RuntimeError("No ERA5 variables were loaded.")
    validate_global_grid(ref_lat, ref_lon, not args.no_require_global_grid)
    upper_np = np.stack(upper_parts, axis=0).astype(np.float32, copy=False)
    surface_np = np.stack(surface_parts, axis=0).astype(np.float32, copy=False)
    return init_time, torch.from_numpy(upper_np), torch.from_numpy(surface_np), ref_lat, ref_lon


def calc_rh_from_q(temperature_k, specific_humidity, pressure_hpa):
    temp_c = temperature_k - 273.15
    es = 6.112 * np.exp((17.67 * temp_c) / np.clip(temp_c + 243.5, 1.0e-6, None))
    q = np.clip(specific_humidity, 1.0e-8, 0.08)
    e = (q * pressure_hpa) / (0.622 + 0.378 * q)
    return np.clip((e / np.clip(es, 1.0e-6, None)) * 100.0, 0.0, 100.0).astype(np.float32)


def calc_dewpoint_from_q(specific_humidity, pressure_hpa):
    q = np.clip(specific_humidity, 1.0e-8, 0.08)
    e_hpa = (q * pressure_hpa) / np.clip(0.622 + 0.378 * q, 1.0e-8, None)
    ln_ratio = np.log(np.clip(e_hpa, 1.0e-6, None) / 6.112)
    td_c = (243.5 * ln_ratio) / np.clip(17.67 - ln_ratio, 1.0e-6, None)
    return (td_c + 273.15).astype(np.float32)


def calc_dewpoint_from_rh(temperature_k, rh_percent):
    temp_c = temperature_k - 273.15
    rh_frac = np.clip(rh_percent / 100.0, 1.0e-4, 1.0)
    gamma = np.log(rh_frac) + (17.67 * temp_c) / np.clip(243.5 + temp_c, 1.0e-6, None)
    td_c = (243.5 * gamma) / np.clip(17.67 - gamma, 1.0e-6, None)
    return (td_c + 273.15).astype(np.float32)


def calc_wind_speed_dir(u, v):
    speed = np.sqrt(u * u + v * v).astype(np.float32)
    direction = ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(np.float32)
    return speed, direction


def crop_indices(lat: np.ndarray, lon: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    lat_mask = (lat >= float(args.lat_min)) & (lat <= float(args.lat_max))
    lon_mask = (lon >= float(args.lon_min)) & (lon <= float(args.lon_max))
    lat_idx = np.flatnonzero(lat_mask)
    lon_idx = np.flatnonzero(lon_mask)
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("China crop produced an empty latitude/longitude subset.")
    return lat_idx, lon_idx


def contiguous_slice(indices: np.ndarray, name: str) -> slice:
    if len(indices) == 0:
        raise ValueError(f"Empty {name} crop indices.")
    if len(indices) > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError(f"{name} crop is not contiguous; adjust crop bounds or handle wrap-around explicitly.")
    return slice(int(indices[0]), int(indices[-1]) + 1)


def as_dataarray(values: np.ndarray, time_value: pd.Timestamp, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values[np.newaxis, :, :].astype(np.float32, copy=False),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [np.datetime64(time_value.to_datetime64())],
            "latitude": lat.astype(np.float32),
            "longitude": lon.astype(np.float32),
        },
    )


def build_output_dataset(
    pressure: np.ndarray,
    surface: np.ndarray,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
    lead_hours: int,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_idx: Optional[np.ndarray],
    lon_idx: Optional[np.ndarray],
    already_cropped: bool = False,
) -> xr.Dataset:
    if already_cropped:
        out_lat = lat
        out_lon = lon
    else:
        if lat_idx is None or lon_idx is None:
            raise ValueError("lat_idx/lon_idx are required when pressure and surface are not pre-cropped.")
        pressure = pressure[:, :, lat_idx, :][:, :, :, lon_idx]
        surface = surface[:, lat_idx, :][:, :, lon_idx]
        out_lat = lat[lat_idx]
        out_lon = lon[lon_idx]

    fields: Dict[str, np.ndarray] = {}
    fields["MSLP"] = surface[0]
    fields["U10"] = surface[1]
    fields["V10"] = surface[2]
    fields["T2M"] = surface[3]
    fields["WSPD10"], fields["WDIR10"] = calc_wind_speed_dir(fields["U10"], fields["V10"])

    level_index = {int(level): i for i, level in enumerate(PRESSURE_LEVELS)}
    for level in (1000, 925):
        i = level_index[level]
        fields[f"Q_{level}"] = pressure[1, i]
        fields[f"T_{level}"] = pressure[2, i]
        fields[f"U_{level}"] = pressure[3, i]
        fields[f"V_{level}"] = pressure[4, i]
        fields[f"RH_{level}"] = calc_rh_from_q(fields[f"T_{level}"], fields[f"Q_{level}"], float(level))
        fields[f"DP_{level}"] = calc_dewpoint_from_q(fields[f"Q_{level}"], float(level))
        fields[f"WSPD{level}"], fields[f"WDIR{level}"] = calc_wind_speed_dir(fields[f"U_{level}"], fields[f"V_{level}"])

    fields["INVERSION"] = (fields["T_925"] - fields["T2M"]).astype(np.float32)

    ds = xr.Dataset({name: as_dataarray(fields[name], valid_time, out_lat, out_lon) for name in OUTPUT_ORDER if name in fields})
    ds = ds.assign_coords(init_time=("time", [np.datetime64(init_time.to_datetime64())]))
    ds.attrs.update(
        {
            "source": "Pangu-Weather ONNX local inference",
            "forecast_lead_hours": int(lead_hours),
            "time_coordinate": "valid_time",
            "surface_input_order": ",".join(SURFACE_VARS),
            "pressure_input_order": ",".join(UPPER_VARS),
            "pressure_levels_hpa": ",".join(str(v) for v in PRESSURE_LEVELS),
            "note": (
                "RH2M, D2M, and DPD are intentionally not generated from 1000 hPa humidity. "
                "Pangu ONNX does not provide native RH2M, D2M, PRECIP, SW_RAD, CAPE, LCC, W_925, or W_1000."
            ),
        }
    )
    return ds


def netcdf_encoding(ds: xr.Dataset, compress_level: int) -> Dict[str, Dict[str, object]]:
    encoding: Dict[str, Dict[str, object]] = {}
    for name in ds.data_vars:
        encoding[name] = {"dtype": "float32"}
        if compress_level > 0:
            encoding[name].update({"zlib": True, "complevel": int(compress_level), "shuffle": True})
    return encoding


def month_label(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}{ts.month:02d}"


def flush_output_group(
    parts: List[xr.Dataset],
    output_dir: Path,
    label: str,
    lead_hours: int,
    compress_level: int,
    overwrite: bool,
    output_label_suffix: str = "",
) -> str:
    if not parts:
        return ""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"pangu_china_{label}{output_label_suffix}_lead{int(lead_hours):03d}h.nc"
    tmp_file = output_dir / f"{out_file.stem}.tmp.nc"
    if out_file.exists() and not overwrite:
        log(f"[SKIP] {out_file} exists; pass --overwrite to replace.")
        return str(out_file)
    if tmp_file.exists():
        tmp_file.unlink()

    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
    ds = ds.sortby("time")
    unique, keep_idx = np.unique(pd.DatetimeIndex(ds["time"].values).values, return_index=True)
    if len(unique) != ds.sizes["time"]:
        ds = ds.isel(time=np.sort(keep_idx))
    ds.attrs["time_count"] = int(ds.sizes["time"])
    ds.to_netcdf(tmp_file, engine="h5netcdf", encoding=netcdf_encoding(ds, compress_level))
    tmp_file.replace(out_file)
    ds.close()
    for part in parts:
        part.close()
    log(f"[OK] wrote {out_file}")
    return str(out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pangu-Weather 24 h ONNX and export China-region canonical NetCDF files."
    )
    parser.add_argument("--model-path", required=True, help="Path to pangu_weather_24.onnx.")
    parser.add_argument(
        "--input-mode",
        choices=["era5-dirs", "pair"],
        default="era5-dirs",
        help="era5-dirs reads the SD005-ERA5 variable-folder layout; pair reads pre-merged upper/surface GRIBs.",
    )
    parser.add_argument(
        "--era5-root",
        default=ERA5_ROOT_DEFAULT,
        help="Root of SD005-ERA5 0p25 data, containing single-hourly and pressure-hourly.",
    )
    parser.add_argument(
        "--init-hours",
        default="0",
        help="Comma-separated initialization hours in UTC for era5-dirs mode, e.g. 0,12 or all.",
    )
    parser.add_argument("--upper-dir", default="", help="Directory containing pressure-level GRIB files.")
    parser.add_argument("--surface-dir", default="", help="Directory containing surface GRIB files.")
    parser.add_argument("--upper-glob", default="*.grib*", help="Glob under --upper-dir.")
    parser.add_argument("--surface-glob", default="*.grib*", help="Glob under --surface-dir.")
    parser.add_argument(
        "--pair-file",
        default="",
        help="Optional text/CSV file: either 'upper surface' or 'init_time upper surface' per line.",
    )
    parser.add_argument("--start-date", default="2025-01-01", help="Output valid-date start, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="Output valid-date end, YYYY-MM-DD.")
    parser.add_argument("--lead-hours", type=int, default=24, help="Hours advanced by one ONNX model step.")
    parser.add_argument("--save-steps", default="1", help="Comma-separated iterative steps to save; 1 means lead24h.")
    parser.add_argument("--lat-min", type=float, default=18.0)
    parser.add_argument("--lat-max", type=float, default=54.0)
    parser.add_argument("--lon-min", type=float, default=73.0)
    parser.add_argument("--lon-max", type=float, default=135.0)
    parser.add_argument("--output-dir", default="/data2/share/chenxi/PuTS/mlp/pangu_2025_china_monthly")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--cfgrib-indexpath",
        default="",
        help="cfgrib indexpath. Empty disables sidecar .idx writes; use AUTO to let cfgrib choose.",
    )
    parser.add_argument("--compress-level", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--smoke-count", type=int, default=0, help="Process only N initial states for testing.")
    parser.add_argument("--manifest-name", default="pangu_onnx_manifest.json", help="Manifest JSON filename under --output-dir.")
    parser.add_argument("--output-label-suffix", default="", help="Suffix inserted after YYYYMM in output NetCDF filenames.")
    parser.add_argument("--skip-missing-init", action="store_true", help="Skip missing ERA5 initial states instead of failing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-global-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_steps = parse_steps(args.save_steps)
    max_step = max(save_steps)
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date.")

    if args.input_mode == "era5-dirs":
        input_items = build_era5_init_times(args, save_steps)
    else:
        input_items = build_pairs(args)
    if args.smoke_count > 0:
        input_items = input_items[: int(args.smoke_count)]
    if not input_items:
        raise RuntimeError("No initial states to process.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    log(f"[model] converting ONNX -> torch: {args.model_path}")
    model = convert(args.model_path).eval().to(device)

    output_dir = Path(args.output_dir)
    buffers: Dict[Tuple[str, int], List[xr.Dataset]] = defaultdict(list)
    written: List[str] = []
    processed = 0
    skipped = 0
    lat_idx = lon_idx = None
    lat_slice = lon_slice = None
    out_lat = out_lon = None
    cached_lat = cached_lon = None

    valid_start = pd.Timestamp(start)
    valid_end = pd.Timestamp(datetime.combine(end, datetime.max.time()))

    for item_no, item in enumerate(input_items, start=1):
        try:
            if isinstance(item, Era5Init):
                init_time, x, x_surface, lat, lon = read_era5_initial_state(item, args, PRESSURE_LEVELS)
            else:
                init_time, x, x_surface, lat, lon = read_initial_state(
                    item,
                    PRESSURE_LEVELS,
                    args.cfgrib_indexpath,
                    not args.no_require_global_grid,
                )
        except (FileNotFoundError, KeyError) as exc:
            if args.skip_missing_init:
                log(f"[SKIP] missing init state for item {item_no}: {exc}")
                skipped += 1
                continue
            raise
        possible_valids = [init_time + pd.Timedelta(hours=int(args.lead_hours) * step) for step in save_steps]
        if not any(valid_start <= vt <= valid_end for vt in possible_valids):
            skipped += 1
            continue

        if lat_idx is None or cached_lat is None or cached_lon is None:
            lat_idx, lon_idx = crop_indices(lat, lon, args)
            lat_slice = contiguous_slice(lat_idx, "latitude")
            lon_slice = contiguous_slice(lon_idx, "longitude")
            out_lat = lat[lat_idx]
            out_lon = lon[lon_idx]
            cached_lat, cached_lon = lat.copy(), lon.copy()
            log(
                "[grid] input={}x{} crop={}x{} lat {:.2f}-{:.2f} lon {:.2f}-{:.2f}".format(
                    len(lat),
                    len(lon),
                    len(lat_idx),
                    len(lon_idx),
                    float(lat[lat_idx].min()),
                    float(lat[lat_idx].max()),
                    float(lon[lon_idx].min()),
                    float(lon[lon_idx].max()),
                )
            )
        elif len(lat) != len(cached_lat) or len(lon) != len(cached_lon) or not np.allclose(lat, cached_lat) or not np.allclose(lon, cached_lon):
            raise ValueError("Grid changed between GRIB pairs; this script expects a fixed Pangu grid.")
        if lat_slice is None or lon_slice is None or out_lat is None or out_lon is None:
            raise RuntimeError("Crop slices were not initialized.")

        if isinstance(item, Era5Init):
            log(f"[{item_no}/{len(input_items)}] init={init_time} era5_root={args.era5_root}")
        else:
            log(f"[{item_no}/{len(input_items)}] init={init_time} upper={Path(item.upper_path).name} surface={Path(item.surface_path).name}")
        state = x.to(device=device, dtype=torch.float32, non_blocking=True)
        state_surface = x_surface.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.inference_mode():
            for step in range(1, max_step + 1):
                state, state_surface = model(state, state_surface)
                if step not in save_steps:
                    continue
                valid_time = init_time + pd.Timedelta(hours=int(args.lead_hours) * step)
                if not (valid_start <= valid_time <= valid_end):
                    continue
                pressure_np = state[:, :, lat_slice, lon_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                surface_np = state_surface[:, lat_slice, lon_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                lead_hours = int(args.lead_hours) * step
                ds_out = build_output_dataset(
                    pressure_np,
                    surface_np,
                    init_time,
                    valid_time,
                    lead_hours,
                    out_lat,
                    out_lon,
                    None,
                    None,
                    already_cropped=True,
                )
                key = (month_label(valid_time), lead_hours)
                buffers[key].append(ds_out)
                processed += 1

        del state, state_surface, x, x_surface
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for (label, lead_hours), parts in sorted(buffers.items()):
        written_path = flush_output_group(
            parts,
            output_dir,
            label,
            lead_hours,
            args.compress_level,
            args.overwrite,
            args.output_label_suffix,
        )
        if written_path:
            written.append(written_path)

    manifest = {
        "model_path": str(Path(args.model_path).expanduser()),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "lead_hours_per_step": int(args.lead_hours),
        "save_steps": save_steps,
        "processed_outputs": int(processed),
        "skipped_pairs": int(skipped),
        "input_mode": args.input_mode,
        "input_count": len(input_items),
        "era5_root": args.era5_root if args.input_mode == "era5-dirs" else "",
        "init_hours": parse_init_hours(args.init_hours) if args.input_mode == "era5-dirs" else [],
        "output_dir": str(output_dir),
        "output_label_suffix": args.output_label_suffix,
        "outputs": written,
        "surface_input_order": list(SURFACE_VARS),
        "pressure_input_order": list(UPPER_VARS),
        "pressure_levels_hpa": list(PRESSURE_LEVELS),
        "missing_overlap_fields_in_pangu_onnx": [
            "RH2M",
            "D2M",
            "DPD",
            "PRECIP",
            "SW_RAD",
            "CAPE",
            "LCC",
            "W_925",
            "W_1000",
        ],
        "rh2m_note": "RH2M is not generated from 1000 hPa humidity in this 2025 ONNX workflow.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[DONE] processed_outputs={processed} skipped_pairs={skipped} manifest={manifest_path}")


if __name__ == "__main__":
    main()
