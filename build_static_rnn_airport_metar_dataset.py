#!/usr/bin/env python3
"""Build airport METAR visibility data for train_static_rnn_lowvis.py.

The older airport-METAR builder writes only ``X_train.npy`` and lets the PMST
training script split validation internally.  The current Static MLP + GRU
trainer expects explicit train/validation arrays, so this script keeps the
same airport source data but emits the standard repository layout:

    X_train.npy, y_train.npy, meta_train.csv
    X_val.npy, y_val.npy, meta_val.csv

The airport link has no PM10/PM2.5 inputs, so the default output keeps the
true airport dynamic feature count.  A legacy flag can still append zero PM
slots, but the recommended Static-RNN path should train with the real input
dimension recorded in ``dataset_build_config.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
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
from s2_data_airport_metar import (
    BASE_PATH,
    DEFAULT_ORO_FILE,
    DEFAULT_STATION_FILE,
    DEFAULT_VEG_FILE,
    DEFAULT_VISIBILITY_FILE,
    DEFAULT_WEATHER_FILE,
    build_static_features,
    calculate_zenith_angle,
    get_common_time_station,
    load_station_table,
    open_dataset_auto,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_PATH, "ml_dataset_static_rnn_airport_metar_2025_12h"
)
ZERO_PM_FEATURES = ("PM10_ZERO", "PM25_ZERO")


def shift_time_if_needed(ds, shift_hours: float, label: str):
    if abs(float(shift_hours)) < 1e-9:
        return ds
    if "time" not in ds.coords:
        raise ValueError(f"{label} has no time coordinate to shift")
    print(f"[Time] shifting {label} time by {shift_hours:+.2f} hours", flush=True)
    return ds.assign_coords(time=ds.time + pd.Timedelta(hours=float(shift_hours)))


def infer_gap_windows(sample_times: pd.DatetimeIndex, gap_hours: float) -> int:
    if gap_hours <= 0 or len(sample_times) < 2:
        return 0
    deltas = np.diff(sample_times.values).astype("timedelta64[s]").astype(np.float64)
    finite = deltas[np.isfinite(deltas) & (deltas > 0)]
    step_seconds = float(np.median(finite)) if finite.size else 3600.0
    gap_seconds = float(gap_hours) * 3600.0
    return int(np.ceil(gap_seconds / max(step_seconds, 1.0)))


def build_time_split_masks(
    sample_times: pd.DatetimeIndex,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gap_hours: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    n_wins = len(sample_times)
    if n_wins < 3:
        raise ValueError(f"Too few windows to split: n_wins={n_wins}")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be positive; test_ratio cannot be negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0 or total_ratio > 1.000001:
        raise ValueError(f"Split ratios must sum to <= 1.0, got {total_ratio}")

    gap_wins = infer_gap_windows(sample_times, gap_hours)
    train_end = max(1, min(n_wins - 1, int(np.floor(n_wins * train_ratio))))
    val_start = min(n_wins, train_end + gap_wins)

    if test_ratio > 0:
        val_end = max(val_start + 1, int(np.floor(n_wins * (train_ratio + val_ratio))))
        val_end = min(n_wins, val_end)
        test_start = min(n_wins, val_end + gap_wins)
    else:
        val_end = n_wins
        test_start = n_wins

    masks: Dict[str, np.ndarray] = {}
    train_mask = np.zeros(n_wins, dtype=bool)
    train_mask[:train_end] = True
    val_mask = np.zeros(n_wins, dtype=bool)
    val_mask[val_start:val_end] = True
    masks["train"] = train_mask
    masks["val"] = val_mask

    if test_ratio > 0:
        test_mask = np.zeros(n_wins, dtype=bool)
        test_mask[test_start:] = True
        masks["test"] = test_mask

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(
                f"{name} split is empty. n_wins={n_wins}, gap_wins={gap_wins}, "
                f"ratios=({train_ratio}, {val_ratio}, {test_ratio})"
            )

    diagnostics: Dict[str, object] = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "gap_hours": float(gap_hours),
        "gap_windows": int(gap_wins),
        "n_windows": int(n_wins),
    }
    for name, mask in masks.items():
        idx = np.where(mask)[0]
        diagnostics[name] = {
            "n_windows": int(mask.sum()),
            "time_start": str(pd.Timestamp(sample_times[idx[0]])),
            "time_end": str(pd.Timestamp(sample_times[idx[-1]])),
        }
    return masks, diagnostics


def append_zero_pm_slots(cube: np.ndarray) -> np.ndarray:
    nt, ns, _ = cube.shape
    zeros = np.zeros((nt, ns, len(ZERO_PM_FEATURES)), dtype=np.float32)
    return np.concatenate([cube, zeros], axis=-1).astype(np.float32)


def count_rows_by_split(
    split_window_masks: Mapping[str, np.ndarray],
    valid_row_mask: np.ndarray,
    n_stations: int,
) -> Dict[str, int]:
    counts = {}
    for split_name, win_mask in split_window_masks.items():
        row_mask = np.repeat(win_mask, n_stations) & valid_row_mask
        counts[split_name] = int(row_mask.sum())
    return counts


def write_split_targets_and_meta(
    output_dir: str,
    split_window_masks: Mapping[str, np.ndarray],
    valid_row_mask: np.ndarray,
    y_flat: np.ndarray,
    sample_times: pd.DatetimeIndex,
    station_order: Iterable[str],
    station_idx: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> Dict[str, Dict[str, str]]:
    n_wins = len(sample_times)
    station_order_arr = np.asarray(list(station_order), dtype=object)
    n_stations = len(station_order_arr)

    all_times = np.repeat(sample_times.to_numpy(), n_stations)
    all_stations = np.tile(station_order_arr, n_wins)
    all_station_idx = np.tile(station_idx.astype(np.int32), n_wins)
    all_lats = np.tile(latitudes.astype(np.float32), n_wins)
    all_lons = np.tile(longitudes.astype(np.float32), n_wins)

    outputs: Dict[str, Dict[str, str]] = {}
    for split_name, win_mask in split_window_masks.items():
        row_mask = np.repeat(win_mask, n_stations) & valid_row_mask
        y_path = os.path.join(output_dir, f"y_{split_name}.npy")
        meta_path = os.path.join(output_dir, f"meta_{split_name}.csv")
        np.save(y_path, y_flat[row_mask].astype(np.float32))
        pd.DataFrame(
            {
                "time": all_times[row_mask],
                "station": all_stations[row_mask],
                "station_idx": all_station_idx[row_mask],
                "lat": all_lats[row_mask],
                "lon": all_lons[row_mask],
            }
        ).to_csv(meta_path, index=False)
        outputs[split_name] = {
            "X": os.path.join(output_dir, f"X_{split_name}.npy"),
            "y": y_path,
            "meta": meta_path,
        }
    return outputs


def write_dataset(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Load] weather:    {args.weather_file}", flush=True)
    print(f"[Load] visibility: {args.visibility_file}", flush=True)
    weather_ds = open_dataset_auto(args.weather_file)
    vis_ds = open_dataset_auto(args.visibility_file)
    weather_ds = shift_time_if_needed(weather_ds, args.weather_time_shift_hours, "weather")
    vis_ds = shift_time_if_needed(vis_ds, args.visibility_time_shift_hours, "visibility")

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

    station_table = load_station_table(args.station_file, common_stations)
    static_cont, veg_idx, latitudes, longitudes = build_static_features(
        station_table, args.vegetation_file, args.orography_file
    )
    if "num_station" in station_table.columns:
        station_idx = station_table["num_station"].to_numpy(dtype=np.int32)
    else:
        station_idx = np.arange(len(station_table), dtype=np.int32)

    zenith = calculate_zenith_angle(latitudes, longitudes, common_times)
    weather_ds = weather_ds.assign(
        ZENITH_PROXY=(("time", "station"), zenith.astype(np.float32))
    )

    print("[Feature] extracting dynamic weather cube...", flush=True)
    cube_base, times, stations = extract_dynamic_cube(
        weather_ds,
        local_time_offset_hours=args.local_time_offset_hours,
        use_source_zenith=True,
    )
    fill_values = compute_fill_values(cube_base)
    cube_base = fill_dynamic_cube(cube_base, fill_values)
    if args.append_zero_pm_slots:
        cube = append_zero_pm_slots(cube_base)
        dynamic_feature_order = list(DYNAMIC_FEATURE_ORDER) + list(ZERO_PM_FEATURES)
        for name in ZERO_PM_FEATURES:
            fill_values[name] = 0.0
    else:
        cube = cube_base
        dynamic_feature_order = list(DYNAMIC_FEATURE_ORDER)

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

    sample_times = pd.DatetimeIndex(times[args.window_size - 1 :: args.step_size])[:n_wins]
    sample_y = visibility_m[args.window_size - 1 :: args.step_size, :][:n_wins]
    y_flat = sample_y.reshape(-1)
    valid_row_mask = (
        np.isfinite(y_flat)
        & (y_flat >= 0.0)
        & (y_flat <= float(args.max_visibility_m))
    )

    split_window_masks, split_diagnostics = build_time_split_masks(
        sample_times,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.gap_hours,
    )
    row_counts = count_rows_by_split(split_window_masks, valid_row_mask, ns)
    for split_name, n_rows in row_counts.items():
        print(f"[Split] {split_name}: rows={n_rows}", flush=True)
        if n_rows <= 0:
            raise ValueError(f"{split_name} split has no valid rows after label filtering")

    split_dyn = args.window_size * dyn_vars
    static_dim = static_cont.shape[1]
    total_dim = split_dyn + static_dim + 1 + EXTRA_FEATURE_DIM
    x_mmaps = {
        split_name: np.lib.format.open_memmap(
            os.path.join(args.output_dir, f"X_{split_name}.npy"),
            mode="w+",
            dtype="float32",
            shape=(n_rows, total_dim),
        )
        for split_name, n_rows in row_counts.items()
    }
    write_pos = {split_name: 0 for split_name in row_counts}

    print(
        f"[Write] total_dim={total_dim}, dyn_vars={dyn_vars}, "
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
        fog_features = compute_fog_features(x_chunk_win[:, :, : len(DYNAMIC_FEATURE_ORDER)])
        cyc = time_cyclical_features(
            sample_times[w0:w1],
            ns,
            local_time_offset_hours=args.local_time_offset_hours,
        )
        extra = np.concatenate([fog_features, cyc], axis=1).astype(np.float32)
        static_tile = np.tile(static_cont, (local_count, 1))
        veg_col = np.tile(veg_idx, local_count).reshape(-1, 1)
        raw_rows = np.concatenate([dyn_flat, static_tile, veg_col, extra], axis=1).astype(np.float32)

        row0, row1 = w0 * ns, w1 * ns
        valid_chunk = valid_row_mask[row0:row1]
        for split_name, win_mask in split_window_masks.items():
            keep = np.repeat(win_mask[w0:w1], ns) & valid_chunk
            n_keep = int(keep.sum())
            if n_keep:
                pos = write_pos[split_name]
                x_mmaps[split_name][pos : pos + n_keep] = raw_rows[keep]
                write_pos[split_name] = pos + n_keep
                x_mmaps[split_name].flush()

        del cube_sl, raw, wins, x_chunk_win, dyn_flat, fog_features
        del cyc, extra, static_tile, veg_col, raw_rows
        gc.collect()

    for split_name in list(x_mmaps):
        x_mmaps[split_name].flush()
        if hasattr(x_mmaps[split_name], "_mmap"):
            x_mmaps[split_name]._mmap.close()
        del x_mmaps[split_name]
        if write_pos[split_name] != row_counts[split_name]:
            raise RuntimeError(
                f"{split_name} write_pos={write_pos[split_name]} "
                f"does not match expected rows={row_counts[split_name]}"
            )

    outputs = write_split_targets_and_meta(
        args.output_dir,
        split_window_masks,
        valid_row_mask,
        y_flat,
        sample_times,
        [str(s) for s in stations],
        station_idx,
        latitudes,
        longitudes,
    )

    dataset_config = {
        "dataset_type": "static_rnn_airport_metar_visibility",
        "feature_set": "airport_metar_true_dynamic_dim",
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "dyn_vars": int(dyn_vars),
        "dyn_vars_count": int(dyn_vars),
        "dynamic_feature_order": dynamic_feature_order,
        "base_airport_dynamic_feature_order": list(DYNAMIC_FEATURE_ORDER),
        "zero_pm_slots": bool(args.append_zero_pm_slots),
        "fe_dim": int(EXTRA_FEATURE_DIM),
        "extra_feature_dim": int(EXTRA_FEATURE_DIM),
        "static_continuous_dim": int(static_dim),
        "vegetation_index_dim": 1,
        "total_dim": int(total_dim),
        "layout": {
            "X": "dyn_window_flat, static_continuous_5, vegetation_index, engineered_features",
            "dyn_window_flat_dim": int(split_dyn),
            "static_continuous_dim": int(static_dim),
            "vegetation_index_dim": 1,
            "engineered_feature_dim": int(EXTRA_FEATURE_DIM),
            "total_dim": int(total_dim),
        },
    }

    config_path = os.path.join(args.output_dir, "dataset_build_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    metadata = {
        **dataset_config,
        "dataset_type": "static_rnn_airport_metar_visibility",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_notebook": "xiahang_train_data_prepare.ipynb",
        "visibility_file": args.visibility_file,
        "weather_file": args.weather_file,
        "station_file": args.station_file,
        "vegetation_file": args.vegetation_file,
        "orography_file": args.orography_file,
        "visibility_variable": visibility_var,
        "visibility_unit_handling": visibility_unit,
        "weather_time_shift_hours": float(args.weather_time_shift_hours),
        "visibility_time_shift_hours": float(args.visibility_time_shift_hours),
        "output_dir": args.output_dir,
        "local_time_offset_hours": float(args.local_time_offset_hours),
        "max_visibility_m": float(args.max_visibility_m),
        "fill_values": fill_values,
        "station_order": [str(s) for s in stations],
        "station_static": {
            str(row.station_name): {
                "num_station": int(station_idx[i]),
                "lat": float(row.station_lat),
                "lon": float(row.station_lon),
                "static_continuous": [float(v) for v in static_cont[i]],
                "vegetation_index": int(veg_idx[i]),
            }
            for i, row in station_table.iterrows()
        },
        "n_times_aligned": int(nt),
        "n_stations": int(ns),
        "n_windows": int(n_wins),
        "n_total_rows_before_label_filter": int(n_wins * ns),
        "n_valid_rows": int(valid_row_mask.sum()),
        "row_counts": row_counts,
        "split_policy": split_diagnostics,
        "time_start": str(pd.Timestamp(times[0])),
        "time_end": str(pd.Timestamp(times[-1])),
        "outputs": outputs,
        "training_entry": {
            "script": "train_static_rnn_lowvis.py",
            "recommended_mode": "s1",
            "recommended_arg": f"--s1-data-dir {args.output_dir}",
            "note": "Do not pass --no-pm for the recommended true-dimension airport dataset.",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    weather_ds.close()
    vis_ds.close()
    print(f"[Done] dataset written to {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Static-RNN-ready airport METAR visibility arrays."
    )
    parser.add_argument("--visibility-file", default=DEFAULT_VISIBILITY_FILE)
    parser.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station-file", default=DEFAULT_STATION_FILE)
    parser.add_argument("--vegetation-file", default=DEFAULT_VEG_FILE)
    parser.add_argument("--orography-file", default=DEFAULT_ORO_FILE)
    parser.add_argument("--visibility-var", default="visibility")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--time-chunk-wins", type=int, default=96)
    parser.add_argument("--local-time-offset-hours", type=float, default=LOCAL_TIME_OFFSET_HOURS)
    parser.add_argument("--weather-time-shift-hours", type=float, default=0.0)
    parser.add_argument("--visibility-time-shift-hours", type=float, default=0.0)
    parser.add_argument("--max-visibility-m", type=float, default=MAX_VISIBILITY_M)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--gap-hours", type=float, default=24.0)
    parser.add_argument(
        "--append-zero-pm-slots",
        action="store_true",
        dest="append_zero_pm_slots",
        help="Legacy compatibility only: append zero PM10/PM2.5 channels.",
    )
    parser.add_argument(
        "--no-zero-pm-slots",
        action="store_false",
        dest="append_zero_pm_slots",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(append_zero_pm_slots=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_dataset(args)


if __name__ == "__main__":
    main()
