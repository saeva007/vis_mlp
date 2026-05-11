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
DEFAULT_STATION_FILE = "/public/home/putianshu/vis_diffusion/test_data/178_stations_renumbered.csv"
DEFAULT_VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
DEFAULT_ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"

UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])


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


def load_station_table(path: str, station_order) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"station_name", "station_lon", "station_lat"}
    missing_cols = required - set(table.columns)
    if missing_cols:
        raise ValueError(f"Station table is missing columns: {sorted(missing_cols)}")
    table["station_name"] = table["station_name"].astype(str)
    table = table.set_index("station_name", drop=False)

    order = [str(s) for s in station_order]
    missing = [s for s in order if s not in table.index]
    if missing:
        raise ValueError(f"Station table missing {len(missing)} stations: {missing[:10]}")
    return table.loc[order].reset_index(drop=True)


def calculate_zenith_angle(latitudes, longitudes, times):
    try:
        import pvlib

        times_pd = pd.DatetimeIndex(pd.to_datetime(times))
        if times_pd.tz is None:
            times_pd = times_pd.tz_localize("UTC")
        else:
            times_pd = times_pd.tz_convert("UTC")
        n_times, n_stations = len(times_pd), len(latitudes)
        b_times = np.repeat(times_pd, n_stations)
        b_lats = np.tile(latitudes, n_times)
        b_lons = np.tile(longitudes, n_times)
        sp = pvlib.solarposition.get_solarposition(b_times, b_lats, b_lons)
        return sp["apparent_zenith"].values.reshape(n_times, n_stations).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] zenith calculation failed ({exc}); using local-time proxy.", flush=True)
        from airport_visibility_common import zenith_proxy_from_time

        return zenith_proxy_from_time(times, len(latitudes))


def get_nearest_veg_indices(latitudes, longitudes, veg_ds):
    veg = veg_ds.sortby("latitude").sortby("longitude")
    veg_raw = veg.sel(
        latitude=xr.DataArray(latitudes, dims="station"),
        longitude=xr.DataArray(longitudes, dims="station"),
        method="nearest",
    )["htcc"].values
    type_to_idx = {v: i for i, v in enumerate(UNIQUE_VEG_IDS)}
    return np.array([type_to_idx.get(v, 0) for v in veg_raw], dtype=np.float32)


def extract_terrain_features(latitudes, longitudes, oro_ds, r=2):
    h = oro_ds["h"].values
    lat_values = oro_ds.latitude.values
    lon_values = oro_ds.longitude.values
    lats_idx = np.abs(lat_values[:, None] - latitudes).argmin(axis=0)
    lons_idx = np.abs(lon_values[:, None] - (longitudes % 360.0)).argmin(axis=0)
    max_r, max_c = h.shape
    feats = []
    for r_idx, c_idx in zip(lats_idx, lons_idx):
        window = h[
            max(0, r_idx - r) : min(max_r, r_idx + r + 1),
            max(0, c_idx - r) : min(max_c, c_idx + r + 1),
        ]
        center = h[r_idx, c_idx]
        feats.append([center, center - np.nanmean(window), np.nanstd(window)])
    return np.asarray(feats, dtype=np.float32)


def build_static_features(station_table: pd.DataFrame, veg_file: str, oro_file: str):
    latitudes = station_table["station_lat"].to_numpy(dtype=np.float32)
    longitudes = station_table["station_lon"].to_numpy(dtype=np.float32)
    print(f"[Static] loading vegetation: {veg_file}", flush=True)
    print(f"[Static] loading orography:  {oro_file}", flush=True)
    veg_ds = open_dataset_auto(veg_file)
    oro_ds = open_dataset_auto(oro_file)
    veg_idx = get_nearest_veg_indices(latitudes, longitudes, veg_ds)
    terrain = extract_terrain_features(latitudes, longitudes, oro_ds)
    veg_ds.close()
    oro_ds.close()
    static_cont = np.concatenate(
        [
            (latitudes[:, None] / 90.0).astype(np.float32),
            (longitudes[:, None] / 180.0).astype(np.float32),
            terrain,
        ],
        axis=1,
    )
    return static_cont.astype(np.float32), veg_idx.astype(np.float32), latitudes, longitudes


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
    station_table = load_station_table(args.station_file, common_stations)
    static_cont, veg_idx, latitudes, longitudes = build_static_features(
        station_table, args.vegetation_file, args.orography_file
    )

    zenith = calculate_zenith_angle(latitudes, longitudes, common_times)
    weather_ds = weather_ds.assign(
        ZENITH_PROXY=(("time", "station"), zenith.astype(np.float32))
    )

    print("[Feature] extracting dynamic weather cube...", flush=True)
    cube, times, stations = extract_dynamic_cube(
        weather_ds,
        local_time_offset_hours=args.local_time_offset_hours,
        use_source_zenith=True,
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
    static_dim = static_cont.shape[1]
    total_dim = split_dyn + static_dim + 1 + EXTRA_FEATURE_DIM
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
        static_tile = np.tile(static_cont, (local_count, 1))
        veg_col = np.tile(veg_idx, local_count).reshape(-1, 1)
        raw_rows = np.concatenate([dyn_flat, static_tile, veg_col, extra], axis=1).astype(np.float32)

        row0, row1 = w0 * ns, w1 * ns
        keep = valid_mask[row0:row1]
        n_keep = int(keep.sum())
        if n_keep:
            x_mm[write_pos : write_pos + n_keep] = raw_rows[keep]
            write_pos += n_keep
            x_mm.flush()

        del cube_sl, raw, wins, x_chunk_win, dyn_flat, fog_features, cyc, extra, static_tile, veg_col, raw_rows
        gc.collect()

    del x_mm
    if write_pos != n_valid:
        raise RuntimeError(f"write_pos={write_pos} does not match n_valid={n_valid}")

    np.save(y_path, y_flat[valid_mask].astype(np.float32))

    all_times = np.repeat(sample_times.to_numpy(), ns)
    all_stations = np.tile(np.asarray(station_order, dtype=object), n_wins)
    all_station_idx = np.tile(station_table["num_station"].to_numpy(dtype=np.int32), n_wins) if "num_station" in station_table else np.tile(np.arange(ns, dtype=np.int32), n_wins)
    all_lats = np.tile(latitudes, n_wins)
    all_lons = np.tile(longitudes, n_wins)
    meta_df = pd.DataFrame(
        {
            "time": all_times[valid_mask],
            "station": all_stations[valid_mask],
            "station_idx": all_station_idx[valid_mask],
            "lat": all_lats[valid_mask],
            "lon": all_lons[valid_mask],
        }
    )
    meta_df.to_csv(meta_path, index=False)

    metadata = {
        "dataset_type": "airport_metar_visibility",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "visibility_file": args.visibility_file,
        "weather_file": args.weather_file,
        "station_file": args.station_file,
        "vegetation_file": args.vegetation_file,
        "orography_file": args.orography_file,
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
            "X": "dyn_window_flat, static_continuous_5, vegetation_index, engineered_features",
            "dyn_window_flat_dim": int(split_dyn),
            "static_continuous_dim": int(static_dim),
            "vegetation_index_dim": 1,
            "engineered_feature_dim": int(EXTRA_FEATURE_DIM),
            "total_dim": int(total_dim),
        },
        "fill_values": fill_values,
        "station_order": station_order,
        "station_static": {
            str(row.station_name): {
                "num_station": int(row.num_station) if "num_station" in station_table.columns else int(i),
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
    parser.add_argument("--station-file", default=DEFAULT_STATION_FILE)
    parser.add_argument("--vegetation-file", default=DEFAULT_VEG_FILE)
    parser.add_argument("--orography-file", default=DEFAULT_ORO_FILE)
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
