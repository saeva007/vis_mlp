#!/usr/bin/env python3

import argparse
import gc
import json
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from airport_visibility_common import (
    DYNAMIC_FEATURE_ORDER,
    EXTRA_FEATURE_DIM,
    LOCAL_TIME_OFFSET_HOURS,
    MAX_VISIBILITY_M,
    WINDOW_SIZE,
    compute_fill_values,
    compute_fog_features,
    extract_dynamic_cube,
    fill_dynamic_cube,
    maybe_convert_visibility_to_meters,
    time_cyclical_features,
)


BASE_PATH = "/public/home/putianshu/vis_mlp"
DEFAULT_VISIBILITY_FILE = os.path.join(BASE_PATH, "metar_visibility_2025_fixed.nc")
DEFAULT_WEATHER_FILE = os.path.join(
    BASE_PATH, "merged_station_weather_data_20250101_20251231.nc"
)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, "ml_dataset_airport_metar_2025_12h")


def open_dataset_auto(path: str) -> xr.Dataset:
    errors = []
    for engine in (None, "h5netcdf", "netcdf4"):
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {exc}")
    raise OSError(f"Cannot open {path}. Tried engines: {' | '.join(errors)}")


def get_common_time_station(
    weather_ds: xr.Dataset,
    vis_ds: xr.Dataset,
) -> Tuple[pd.DatetimeIndex, list]:
    if "time" not in weather_ds.coords or "time" not in vis_ds.coords:
        raise ValueError("Both weather and visibility files must have a time coordinate")
    if "station" not in weather_ds.coords or "station" not in vis_ds.coords:
        raise ValueError("Both weather and visibility files must have a station coordinate")

    weather_times = pd.DatetimeIndex(pd.to_datetime(weather_ds["time"].values))
    vis_times = pd.DatetimeIndex(pd.to_datetime(vis_ds["time"].values))
    vis_time_set = set(vis_times.values)
    common_times = pd.DatetimeIndex([t for t in weather_times if t.to_datetime64() in vis_time_set])

    vis_station_set = {str(s) for s in vis_ds["station"].values}
    common_stations = [s for s in weather_ds["station"].values if str(s) in vis_station_set]

    if len(common_times) < WINDOW_SIZE:
        raise ValueError(
            f"Only {len(common_times)} common times found; need at least {WINDOW_SIZE}"
        )
    if not common_stations:
        raise ValueError("No common stations found between weather and visibility files")

    return common_times, common_stations


def write_dataset(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Load] weather:    {args.weather_file}", flush=True)
    print(f"[Load] visibility: {args.visibility_file}", flush=True)
    weather_ds = open_dataset_auto(args.weather_file)
    vis_ds = open_dataset_auto(args.visibility_file)

    visibility_var = args.visibility_var
    if visibility_var not in vis_ds.data_vars:
        if len(vis_ds.data_vars) != 1:
            raise KeyError(
                f"Visibility variable {visibility_var!r} not found and file has "
                f"multiple variables: {list(vis_ds.data_vars)}"
            )
        visibility_var = list(vis_ds.data_vars)[0]
        print(f"[Load] using visibility variable: {visibility_var}", flush=True)

    common_times, common_stations = get_common_time_station(weather_ds, vis_ds)
    print(
        f"[Align] common times={len(common_times)}, stations={len(common_stations)}",
        flush=True,
    )

    weather_ds = weather_ds.sel(time=common_times, station=common_stations)
    vis_da = vis_ds[visibility_var].sel(time=common_times, station=common_stations)

    print("[Feature] extracting dynamic weather cube...", flush=True)
    cube, times, stations = extract_dynamic_cube(
        weather_ds,
        local_time_offset_hours=args.local_time_offset_hours,
        use_source_zenith=False,
    )
    fill_values = compute_fill_values(cube)
    cube = fill_dynamic_cube(cube, fill_values)
    nt, ns, dyn_vars = cube.shape
    print(f"[Feature] dynamic cube shape={cube.shape}", flush=True)

    visibility_m, visibility_unit = maybe_convert_visibility_to_meters(vis_da.values)
    visibility_m = visibility_m.astype(np.float32)
    visibility_m = np.where(visibility_m <= args.max_visibility_m, visibility_m, np.nan)

    n_wins = (nt - args.window_size) // args.step_size + 1
    if n_wins <= 0:
        raise ValueError(
            f"Cannot build windows: nt={nt}, window_size={args.window_size}, "
            f"step_size={args.step_size}"
        )

    sample_times = pd.DatetimeIndex(times[args.window_size - 1 :: args.step_size])
    sample_y = visibility_m[args.window_size - 1 :: args.step_size, :]
    y_flat = sample_y.reshape(-1)
    valid_mask = (
        np.isfinite(y_flat)
        & (y_flat >= 0.0)
        & (y_flat <= float(args.max_visibility_m))
    )
    n_valid = int(valid_mask.sum())
    n_total = int(n_wins * ns)
    print(
        f"[Window] n_wins={n_wins}, total rows={n_total}, valid rows={n_valid} "
        f"({n_valid / max(n_total, 1):.1%})",
        flush=True,
    )
    if n_valid == 0:
        raise ValueError("No valid visibility samples after filtering")

    split_dyn = args.window_size * dyn_vars
    total_dim = split_dyn + 1 + EXTRA_FEATURE_DIM
    x_path = os.path.join(args.output_dir, "X_train.npy")
    y_path = os.path.join(args.output_dir, "y_train.npy")
    meta_path = os.path.join(args.output_dir, "meta_train.csv")
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")

    x_mm = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype="float32",
        shape=(n_valid, total_dim),
    )

    station_order = [str(s) for s in stations]
    station_indices = np.arange(ns, dtype=np.float32)
    write_pos = 0

    print(
        f"[Write] X_train shape=({n_valid}, {total_dim}), "
        f"chunk_wins={args.time_chunk_wins}",
        flush=True,
    )
    for w0 in tqdm(range(0, n_wins, args.time_chunk_wins), desc="windows"):
        w1 = min(w0 + args.time_chunk_wins, n_wins)
        t_start = w0 * args.step_size
        len_sl = (w1 - 1 - w0) * args.step_size + args.window_size
        t_end = t_start + len_sl

        cube_sl = np.ascontiguousarray(cube[t_start:t_end, :, :], dtype=np.float32)
        raw = sliding_window_view(cube_sl, args.window_size, axis=0)
        local_count = w1 - w0
        idx_loc = (np.arange(local_count) * args.step_size).astype(np.intp)
        wins = raw[idx_loc].transpose(0, 1, 3, 2)
        x_chunk_win = wins.reshape(-1, args.window_size, dyn_vars)
        dyn_flat = x_chunk_win.reshape(x_chunk_win.shape[0], -1)
        fog_features = compute_fog_features(x_chunk_win)
        cyc = time_cyclical_features(
            sample_times[w0:w1],
            ns,
            local_time_offset_hours=args.local_time_offset_hours,
        )
        extra = np.concatenate([fog_features, cyc], axis=1).astype(np.float32)
        station_col = np.tile(station_indices, local_count).reshape(-1, 1)
        raw_rows = np.concatenate([dyn_flat, station_col, extra], axis=1).astype(np.float32)

        row0, row1 = w0 * ns, w1 * ns
        keep = valid_mask[row0:row1]
        n_keep = int(keep.sum())
        if n_keep:
            x_mm[write_pos : write_pos + n_keep] = raw_rows[keep]
            write_pos += n_keep
            x_mm.flush()

        del cube_sl, raw, wins, x_chunk_win, dyn_flat, fog_features, cyc, extra, raw_rows
        gc.collect()

    del x_mm
    if write_pos != n_valid:
        raise RuntimeError(f"write_pos={write_pos} does not match n_valid={n_valid}")

    np.save(y_path, y_flat[valid_mask].astype(np.float32))

    all_times = np.repeat(sample_times.to_numpy(), ns)
    all_stations = np.tile(np.asarray(station_order, dtype=object), n_wins)
    all_station_idx = np.tile(np.arange(ns, dtype=np.int32), n_wins)
    meta_df = pd.DataFrame(
        {
            "time": all_times[valid_mask],
            "station": all_stations[valid_mask],
            "station_idx": all_station_idx[valid_mask],
        }
    )
    meta_df.to_csv(meta_path, index=False)

    metadata = {
        "dataset_type": "airport_metar_visibility",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "visibility_file": args.visibility_file,
        "weather_file": args.weather_file,
        "visibility_variable": visibility_var,
        "visibility_unit_handling": visibility_unit,
        "output_dir": args.output_dir,
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "local_time_offset_hours": float(args.local_time_offset_hours),
        "max_visibility_m": float(args.max_visibility_m),
        "dynamic_feature_order": DYNAMIC_FEATURE_ORDER,
        "dyn_vars_count": int(dyn_vars),
        "extra_feature_dim": int(EXTRA_FEATURE_DIM),
        "layout": {
            "X": "dyn_window_flat, station_idx, engineered_features",
            "dyn_window_flat_dim": int(split_dyn),
            "station_idx_dim": 1,
            "engineered_feature_dim": int(EXTRA_FEATURE_DIM),
            "total_dim": int(total_dim),
        },
        "fill_values": fill_values,
        "station_order": station_order,
        "n_times_aligned": int(nt),
        "n_stations": int(ns),
        "n_windows": int(n_wins),
        "n_total_rows_before_label_filter": int(n_total),
        "n_train_rows": int(n_valid),
        "time_start": str(pd.Timestamp(times[0])),
        "time_end": str(pd.Timestamp(times[-1])),
        "outputs": {
            "X_train": x_path,
            "y_train": y_path,
            "meta_train": meta_path,
            "dataset_metadata": metadata_path,
        },
        "note": "No test split is written. Training scripts may make an internal validation split from X_train.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    weather_ds.close()
    vis_ds.close()
    print(f"[Done] dataset written to {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build airport METAR visibility training arrays from station weather only."
    )
    parser.add_argument("--visibility-file", default=DEFAULT_VISIBILITY_FILE)
    parser.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visibility-var", default="visibility")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--time-chunk-wins", type=int, default=96)
    parser.add_argument("--local-time-offset-hours", type=float, default=LOCAL_TIME_OFFSET_HOURS)
    parser.add_argument("--max-visibility-m", type=float, default=MAX_VISIBILITY_M)
    return parser.parse_args()


def main():
    args = parse_args()
    write_dataset(args)


if __name__ == "__main__":
    main()
