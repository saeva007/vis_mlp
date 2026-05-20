#!/usr/bin/env python3
"""
基于 current_48h 构建 ML 数据集 (s2, 2025)：
- 每次起报后 0–48 小时内单独划分 12h 时间窗口，不与其他起报时间合并；
- 气象变量顺序与 stage1 (PMST_s1_data_pm10) 一致：FINAL_FEATURE_ORDER 同序（ERA5 命名不同但变量一一对应）。
- 预报特有：lead_time、周期时间编码(月/时)、以及 PMST_s2_data 的 compute_fog_features 中多出的 7 维见下方注释。
- 当前默认输出与 PM10+PM2.5 主模型评估配置一致：27 dyn + 36 FE，raw time 按 UTC 处理。

输出目录:
  vis_mlp/ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime

--- Stage2 有而 Stage1 没有的特征工程（仅列出，不删除）---
1) 周期时间编码：fe 中增加 4 维 [sin(2π*month/12), cos(2π*month/12), sin(2π*hour/24), cos(2π*hour/24)]（观测时刻）。
2) 预报时效 lead_hour 写入 meta CSV，不进入 FE（与 PMST_net 输入维一致）。
3) compute_fog_features：PMST_s2_data 在 stage1 的 24 维 fog 特征之后多 8 维：
   shear_mag, dir_turning, convective_wet, daytime_mixing, ventilation, moisture_strat, omega_contrast, warm_instability。
   即 stage2 的 fog 部分为 32 维（24+8），stage1 为 24 维。
4) PM10 和 PM2.5 作为动态变量追加到 dyn 末两维（训练/评估端统一 log1p）。FE 总维数固定为 32+4 = 36。
"""

import json
import os
import re
import gc
import struct
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from tqdm import tqdm

# 与 stage1 气象变量顺序一致（FINAL_FEATURE_ORDER 同序）；预报侧特征工程仍用 PMST_s2_data
from PMST_s2_data import (
    VAR_MAPPING,
    FINAL_FEATURE_ORDER,
    WINDOW_SIZE,
    STEP_SIZE,
    UNIQUE_VEG_IDS,
    MAX_VIS_THRESHOLD,
    compute_fog_features,
    calculate_dewpoint_from_rh,
    calculate_zenith_angle,
    get_nearest_veg,
    extract_terrain,
)
from s2_data_aerosol import (
    GAP_HOURS,
    TEST_LAST_DAYS,
    VAL_LAST_DAYS,
    get_monthly_split_mask_last_days,
)

BASE_PATH = "/public/home/putianshu/vis_mlp"
CURRENT_48H_DIR = os.path.join(BASE_PATH, "tianji_auto_station", "current_48h")
VIS_SOURCE_NC = os.path.join(BASE_PATH, "tianji_auto_station", "merged_final_all_vars.nc")
VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"
OUTPUT_DATASET_DIR = os.environ.get(
    "OUTPUT_DATASET_DIR",
    os.path.join(BASE_PATH, "ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime"),
)

PM10_S2_FILE = os.environ.get("PM10_S2_FILE", os.path.join(BASE_PATH, "pm10_station", "pm10_station_s2_2025.nc"))
PM10_DIR = os.environ.get("PM10_DIR", os.path.join(BASE_PATH, "pm10_station"))
PM25_S2_FILE = os.environ.get("PM25_S2_FILE", os.path.join(BASE_PATH, "pm2.5_station", "pm2p5_station_s2_2025.nc"))
PM25_DIR = os.environ.get("PM25_DIR", os.path.join(BASE_PATH, "pm2.5_station"))
SPLITS_TO_WRITE_ENV = os.environ.get("SPLITS_TO_WRITE", "test")

# current_48h 目录中 2m 温度为 TMP2m（无 t2mz），与 VAR_MAPPING 一致
VARIABLES_48H = [
    "rh2m", "PRATEsfc", "slp", "DSWRFsfc",
    "UGRD10m", "VGRD10m", "cape", "cldl", "t925", "rh925",
    "u925", "v925", "dp1000", "dp925", "q1000", "q925", "omg925",
    "omg1000", "gust", "TMP2m",
]


def parse_splits_to_write(value):
    if value is None:
        return ("test",)
    norm = str(value).strip().lower()
    if norm in {"", "testonly", "test_only"}:
        return ("test",)
    if norm == "all":
        return ("train", "val", "test")
    aliases = {"validation": "val", "valid": "val"}
    splits = []
    for item in re.split(r"[,;\s]+", norm):
        if not item:
            continue
        tag = aliases.get(item, item)
        if tag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split in SPLITS_TO_WRITE={value!r}: {item!r}")
        if tag not in splits:
            splits.append(tag)
    return tuple(splits or ["test"])


def clear_existing_dataset_outputs(out_dir):
    for tag in ("train", "val", "test"):
        for name in (f"X_{tag}.npy", f"y_{tag}.npy", f"meta_{tag}.csv"):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                os.remove(path)
    config_path = os.path.join(out_dir, "dataset_build_config.json")
    if os.path.exists(config_path):
        os.remove(config_path)


def get_run_list_from_current_48h():
    pattern = os.path.join(CURRENT_48H_DIR, f"{VARIABLES_48H[0]}_*_0-48h_IDW.nc")
    files = glob(pattern)
    runs = []
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"^[^_]+_(\d{10})_0-48h_IDW\.nc$", base)
        if m:
            runs.append(m.group(1))
    return sorted(set(runs))


def _open_station_nc(path):
    """
    HDF5/NetCDF4 (h5netcdf), NetCDF4 classic (netcdf4), NetCDF3 (scipy).
    Returns (ds, None) or (None, last_exception) if file is corrupt / not NetCDF.
    """
    last_err = None
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng), None
        except Exception as e:
            last_err = e
            continue
    return None, last_err


def _load_station_latlon():
    import pandas as pd
    station_df = pd.read_csv(os.path.join(BASE_PATH, "tianji_auto_station", "station_info.csv"))
    station_df = station_df[
        (station_df["station_lon"] >= 65) & (station_df["station_lon"] <= 145)
        & (station_df["station_lat"] >= 10) & (station_df["station_lat"] <= 60)
    ]
    return station_df.set_index("num_station")


def load_station_pm_dataarray(file_path, dir_path, var_candidates, label):
    """返回 (time, station_id) 的 PM DataArray；缺失时返回 None 并由调用处补零。"""
    das = []
    if os.path.isfile(file_path):
        ds = xr.open_dataset(file_path, engine="h5netcdf")
        var_name = next((v for v in var_candidates if v in ds), None) or list(ds.data_vars)[0]
        da = ds[var_name].load()
        ds.close()
        das.append(da)
    elif os.path.isdir(dir_path):
        for fp in sorted(glob(os.path.join(dir_path, "*.nc"))):
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
                if "station_id" not in ds.coords and "station_id" not in ds.data_vars:
                    for alias in ("num_station", "id", "station"):
                        if alias in ds.coords or alias in ds.data_vars:
                            ds = ds.rename({alias: "station_id"})
                            break
                var_name = next((v for v in var_candidates if v in ds), None) or list(ds.data_vars)[0]
                da = ds[var_name].load()
                ds.close()
                das.append(da)
            except Exception as exc:
                print(f"[WARN] skip {label} file {fp}: {exc}", flush=True)
    if not das:
        print(f"[WARN] No {label} station files found; {label} dynamic channel will be zeros.", flush=True)
        return None
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if "station_id" not in da.dims:
        raise ValueError(f"{label} DataArray must have station_id dimension")
    return da.transpose("time", "station_id")


def station_pm_to_ugm3_grid(pm_da, times, stations, label):
    """将站点 PM 对齐到 48h forecast valid time/station；kg m-3 转 ug m-3。"""
    nt_ds, ns_ds = len(times), len(stations)
    if pm_da is None:
        return np.zeros((nt_ds, ns_ds), dtype=np.float32)
    pm_grid_da = pm_da
    if set(pm_grid_da.dims) >= {"time", "station_id"}:
        pm_grid_da = pm_grid_da.transpose("time", "station_id")
    else:
        raise ValueError(f"{label} dims must contain 'time' and 'station_id', got {pm_grid_da.dims}")
    pm_grid_da = pm_grid_da.load()
    time_vals = pm_grid_da["time"].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        time_index = pd.DatetimeIndex(time_vals)
    else:
        time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
    ds_times = pd.DatetimeIndex(times)
    sid_index = pd.Index(pm_grid_da["station_id"].values)
    sids = stations.astype(pm_grid_da["station_id"].dtype)
    time_pos = time_index.get_indexer(ds_times, method="nearest")
    sid_pos = sid_index.get_indexer(sids)

    _, ns_pm = pm_grid_da.shape
    pm_grid = np.full((nt_ds, ns_ds), np.nan, dtype=np.float32)
    base = np.asarray(pm_grid_da.values, dtype=np.float32).reshape(-1)
    linear_idx_grid = time_pos[:, None] * ns_pm + sid_pos[None, :]
    ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
    if ok_mask.any():
        pm_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
    pm_grid = np.maximum(pm_grid, 0.0)
    pm_ugm3 = pm_grid * 1e12
    median = np.nanmedian(pm_ugm3)
    if not np.isfinite(median):
        median = 0.0
    return np.where(np.isfinite(pm_ugm3), pm_ugm3, median).astype(np.float32)


def load_merged_run_ds(run_str, data_veg, data_oro):
    init_time = pd.to_datetime(run_str, format="%Y%m%d%H")
    ds_list = []
    for var in VARIABLES_48H:
        path = os.path.join(CURRENT_48H_DIR, f"{var}_{run_str}_0-48h_IDW.nc")
        if not os.path.exists(path):
            for d in ds_list:
                d.close()
            return None, None
        try:
            sz = os.path.getsize(path)
        except OSError:
            sz = 0
        if sz < 1:
            for d in ds_list:
                d.close()
            print(f"  [WARN] skip run {run_str}: empty file {path}", flush=True)
            return None, None
        ds, open_err = _open_station_nc(path)
        if ds is None:
            for d in ds_list:
                d.close()
            print(
                f"  [WARN] skip run {run_str}: unreadable {path} ({open_err!r})",
                flush=True,
            )
            return None, None
        ds_list.append(ds)

    ds = xr.merge(ds_list, join="inner")
    for d in ds_list:
        d.close()
    del ds_list
    gc.collect()

    if "num_station" in ds.dims:
        ds = ds.rename({"num_station": "station_id"})

    station_idx = _load_station_latlon()
    sids = ds.station_id.values
    # 仅保留 station_info.csv 中存在且经纬度落在中国区域的站点，避免 .loc KeyError
    try:
        sids_idx = pd.Index(sids)
    except Exception:
        sids_idx = pd.Index(np.asarray(sids))
    valid_sids = sids_idx.intersection(station_idx.index)
    if len(valid_sids) == 0:
        return None, None
    if len(valid_sids) < len(sids_idx):
        print(
            f"  [run {run_str}] drop stations not in station_info: {len(sids_idx) - len(valid_sids)}",
            flush=True,
        )
        ds = ds.sel(station_id=valid_sids.values)
        sids = ds.station_id.values
    lats = station_idx.loc[sids]["station_lat"].values
    lons = station_idx.loc[sids]["station_lon"].values
    ds = ds.assign_coords(lat=("station_id", lats), lon=("station_id", lons))

    times = pd.to_datetime(ds.time.values)
    lead_sec = (times - init_time).total_seconds()
    lead_hours = (lead_sec / 3600.0).astype(np.float32)
    ds["lead_time"] = (("time",), lead_hours)
    ds = ds.sortby("time")

    rename_map = {k: v for k, v in VAR_MAPPING.items() if k in ds}
    ds = ds.rename(rename_map)
    if "t2mz" in ds and "T2M" not in ds:
        ds = ds.rename({"t2mz": "T2M"})
    if "D2M" not in ds and "T2M" in ds and "RH2M" in ds:
        ds["D2M"] = calculate_dewpoint_from_rh(ds["T2M"], ds["RH2M"])
    if "DPD" not in ds:
        ds["DPD"] = ds["T2M"] - ds["D2M"]
    if "INVERSION" not in ds and "T_925" in ds and "T2M" in ds:
        ds["INVERSION"] = ds["T_925"] - ds["T2M"]

    if "WSPD10" not in ds and "U10" in ds and "V10" in ds:
        ds["WSPD10"] = np.sqrt(ds["U10"] ** 2 + ds["V10"] ** 2)
    if "WDIR10" not in ds and "U10" in ds and "V10" in ds:
        ds["WDIR10"] = np.degrees(np.arctan2(-ds["U10"], -ds["V10"])) % 360
    if "WSPD925" not in ds and "U_925" in ds and "V_925" in ds:
        ds["WSPD925"] = np.sqrt(ds["U_925"] ** 2 + ds["V_925"] ** 2)

    return ds, init_time


def build_windows_and_features_per_run(ds_run, init_time, data_veg, data_oro, vis_da, pm10_da, pm25_da, splits_to_write):
    """
    仅在当前 run 的 48h 时间序列内划分 12h 时间窗。

    为避免全年 48h 样本一次性拼接导致 OOM，本函数只物化需要写出的 split
    样本；默认只写 test split，供 48h 论文评估使用。
    """
    ds = ds_run
    lats = ds["lat"].values
    lons = ds["lon"].values
    times = pd.to_datetime(ds.time.values)
    stations = ds.station_id.values
    nt, ns = len(times), len(stations)

    var_list = [v for v in FINAL_FEATURE_ORDER if v in ds]
    if len(var_list) != len(FINAL_FEATURE_ORDER):
        missing = set(FINAL_FEATURE_ORDER) - set(var_list)
        raise RuntimeError(f"Missing vars for run: {missing}")
    X_met = ds[FINAL_FEATURE_ORDER].to_array(dim="v").transpose("time", "station_id", "v").values.astype(np.float32)
    zenith = calculate_zenith_angle(lats, lons, times)
    X_dyn = np.concatenate([X_met, zenith], axis=-1)
    del X_met, zenith
    gc.collect()

    # ========= 将 PM10 / PM2.5 作为动态变量追加到 dyn 的最后两维 =========
    # PM 文件单位为 kg/m^3，这里转换为 ug/m^3 后留给训练/评估脚本做 log1p。
    pm10_ugm3 = station_pm_to_ugm3_grid(pm10_da, times, stations, "PM10")
    pm25_ugm3 = station_pm_to_ugm3_grid(pm25_da, times, stations, "PM2.5")
    X_dyn = np.concatenate([X_dyn, pm10_ugm3[..., None], pm25_ugm3[..., None]], axis=-1)
    nv = X_dyn.shape[-1]
    del pm10_ugm3, pm25_ugm3

    lead_time = ds["lead_time"].values

    veg_raw = get_nearest_veg(lats, lons, data_veg)
    type_to_idx = {v: i for i, v in enumerate(UNIQUE_VEG_IDS)}
    feat_veg = np.array([type_to_idx.get(v, 0) for v in veg_raw])[:, None].astype(np.float32)
    feat_oro = extract_terrain(lats, lons, data_oro)
    X_stat = np.concatenate(
        [
            (lats[:, None] / 90.0).astype(np.float32),
            (lons[:, None] / 180.0).astype(np.float32),
            feat_oro,
            feat_veg,
        ],
        axis=1,
    )

    dims = {
        "dynamic_flat_dim": int(WINDOW_SIZE * nv),
        "static_dim": int(X_stat.shape[1]),
        "feature_engineering_dim": 36,
    }

    n_wins = (nt - WINDOW_SIZE) // STEP_SIZE + 1
    if n_wins <= 0:
        return {}, dims

    win_end_idx = (np.arange(n_wins) * STEP_SIZE + (WINDOW_SIZE - 1)).astype(int)
    win_end_times = times[win_end_idx]
    win_lead = lead_time[win_end_idx]

    if vis_da is not None:
        vis_sel = vis_da.sel(time=win_end_times, method="nearest")
        vis_sel = vis_sel.reindex(station_id=stations)
        y_flat = vis_sel.values.astype(np.float32).ravel()
    else:
        y_flat = np.full(n_wins * ns, np.nan, dtype=np.float32)

    m_t = np.repeat(win_end_times, ns)
    m_s = np.tile(stations, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)
    m_lead = np.repeat(win_lead, ns).astype(np.float32)

    valid_mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)
    valid_idxs = np.where(valid_mask)[0]
    if len(valid_idxs) == 0:
        return {}, dims

    tr_m, val_m, test_m = get_monthly_split_mask_last_days(
        pd.DatetimeIndex(m_t[valid_idxs]),
        gap_hours=GAP_HOURS,
        val_last_days=VAL_LAST_DAYS,
        test_last_days=TEST_LAST_DAYS,
    )
    split_masks = {"train": tr_m, "val": val_m, "test": test_m}
    split_indices = {
        tag: valid_idxs[split_masks[tag]]
        for tag in ("train", "val", "test")
        if tag in splits_to_write and split_masks[tag].any()
    }
    if not split_indices:
        return {}, dims

    selected_parts = []
    split_slices = {}
    cursor = 0
    for tag in ("train", "val", "test"):
        ix = split_indices.get(tag)
        if ix is None or len(ix) == 0:
            continue
        selected_parts.append(ix)
        split_slices[tag] = slice(cursor, cursor + len(ix))
        cursor += len(ix)
    selected_ix = np.concatenate(selected_parts, axis=0)

    from numpy.lib.stride_tricks import sliding_window_view

    X_wins = sliding_window_view(X_dyn, WINDOW_SIZE, axis=0)[::STEP_SIZE]
    # 须与 PMST_s2_data.py / s2_data_pm10_monthtail_cell 一致：(n_wins, ns, W, nv)，与 m_t=repeat(win,ns) 对齐
    X_wins = X_wins.transpose(0, 1, 3, 2).reshape(-1, WINDOW_SIZE, nv)[selected_ix]

    fe_flat = compute_fog_features(X_wins, WINDOW_SIZE, nv)

    sample_times = pd.DatetimeIndex(m_t[selected_ix])
    months = sample_times.month.values.astype(np.float32)
    hours = sample_times.hour.values.astype(np.float32)
    time_feat = np.column_stack(
        [
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
        ]
    ).astype(np.float32)
    fe_flat = np.concatenate([fe_flat, time_feat], axis=1)

    X_dyn_flat = X_wins.reshape(X_wins.shape[0], -1)
    station_pos = selected_ix % ns
    X_stat_flat = X_stat[station_pos].astype(np.float32, copy=False)
    y_sel = y_flat[selected_ix]
    meta_sel = (
        m_t[selected_ix],
        m_s[selected_ix],
        m_la[selected_ix],
        m_lo[selected_ix],
        m_lead[selected_ix],
    )

    chunks = {}
    for tag, slc in split_slices.items():
        chunks[tag] = {
            "X_dyn": X_dyn_flat[slc],
            "X_stat": X_stat_flat[slc],
            "fe": fe_flat[slc],
            "y": y_sel[slc],
            "meta": tuple(v[slc] for v in meta_sel),
        }

    dims.update(
        {
            "dynamic_flat_dim": int(X_dyn_flat.shape[1]),
            "static_dim": int(X_stat_flat.shape[1]),
            "feature_engineering_dim": int(fe_flat.shape[1]),
        }
    )
    return chunks, dims


def _fixed_npy_header(shape, dtype, header_len=1014):
    """Build a fixed-size NPY v1 header so append-only files can be finalized in place."""
    dtype = np.dtype(dtype)
    header = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (
        np.lib.format.dtype_to_descr(dtype),
        tuple(int(x) for x in shape),
    )
    header_bytes = header.encode("latin1")
    if len(header_bytes) + 1 > header_len:
        raise ValueError(f"NPY header too small for shape={shape}")
    payload = header_bytes + b" " * (header_len - len(header_bytes) - 1) + b"\n"
    return np.lib.format.magic(1, 0) + struct.pack("<H", header_len) + payload


class AppendableNpyWriter:
    """Append rows to a valid .npy file, then patch the final row count at close."""

    def __init__(self, path, dtype, sample_shape):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.sample_shape = tuple(sample_shape)
        self.n_rows = 0
        self.fh = open(path, "wb")
        self.fh.write(_fixed_npy_header((0,) + self.sample_shape, self.dtype))

    def write(self, arr):
        arr = np.asarray(arr, dtype=self.dtype)
        if self.sample_shape:
            arr = arr.reshape((-1,) + self.sample_shape)
        else:
            arr = arr.reshape(-1)
        if arr.shape[0] == 0:
            return
        arr = np.ascontiguousarray(arr)
        arr.tofile(self.fh)
        self.n_rows += int(arr.shape[0])

    def close(self):
        if self.fh.closed:
            return
        self.fh.flush()
        self.fh.seek(0)
        self.fh.write(_fixed_npy_header((self.n_rows,) + self.sample_shape, self.dtype))
        self.fh.close()


class Streaming48hWriter:
    def __init__(self, out_dir, splits_to_write, dims):
        self.out_dir = out_dir
        self.splits_to_write = tuple(splits_to_write)
        self.dims = dict(dims)
        self.dims["total_dim"] = (
            self.dims["dynamic_flat_dim"] + self.dims["static_dim"] + self.dims["feature_engineering_dim"]
        )
        self.x_writers = {}
        self.y_writers = {}
        self.counts = {tag: 0 for tag in self.splits_to_write}
        os.makedirs(out_dir, exist_ok=True)
        for tag in self.splits_to_write:
            self.x_writers[tag] = AppendableNpyWriter(
                os.path.join(out_dir, f"X_{tag}.npy"),
                "float32",
                (self.dims["total_dim"],),
            )
            self.y_writers[tag] = AppendableNpyWriter(
                os.path.join(out_dir, f"y_{tag}.npy"),
                "float32",
                (),
            )
            pd.DataFrame(columns=["time", "station_id", "lat", "lon", "lead_hour", "init_time"]).to_csv(
                os.path.join(out_dir, f"meta_{tag}.csv"),
                index=False,
            )

    def write_chunk(self, tag, chunk, init_time):
        if tag not in self.x_writers:
            return
        n = int(len(chunk["y"]))
        if n == 0:
            return
        x_full = np.empty((n, self.dims["total_dim"]), dtype=np.float32)
        d0 = self.dims["dynamic_flat_dim"]
        d1 = d0 + self.dims["static_dim"]
        x_full[:, :d0] = chunk["X_dyn"]
        x_full[:, d0:d1] = chunk["X_stat"]
        x_full[:, d1:] = chunk["fe"]
        self.x_writers[tag].write(x_full)
        self.y_writers[tag].write(chunk["y"])

        times, stats, lats, lons, leads = chunk["meta"]
        pd.DataFrame(
            {
                "time": times,
                "station_id": stats,
                "lat": lats,
                "lon": lons,
                "lead_hour": leads.astype(np.float32),
                "init_time": np.full(n, init_time, dtype=object),
            }
        ).to_csv(os.path.join(self.out_dir, f"meta_{tag}.csv"), mode="a", header=False, index=False)
        self.counts[tag] += n

    def close(self, run_count, skipped_count):
        for writer in self.x_writers.values():
            writer.close()
        for writer in self.y_writers.values():
            writer.close()
        config = {
            "builder": os.path.basename(__file__),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dataset_dir": self.out_dir,
            "source_current_48h_dir": CURRENT_48H_DIR,
            "visibility_source_nc": VIS_SOURCE_NC,
            "tianji_raw_time_alignment": "raw_utc_no_shift",
            "split": {
                "type": "monthly_tail_last_days",
                "val_last_days": VAL_LAST_DAYS,
                "test_last_days": TEST_LAST_DAYS,
                "gap_hours": GAP_HOURS,
                "splits_written": list(self.splits_to_write),
            },
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "dynamic_order": FINAL_FEATURE_ORDER + ["zenith", "PM10_ugm3", "PM25_ugm3"],
            "dynamic_dim": int(self.dims["dynamic_flat_dim"] // WINDOW_SIZE),
            "static_dim": int(self.dims["static_dim"]),
            "feature_engineering_dim": int(self.dims["feature_engineering_dim"]),
            "total_dim": int(self.dims["total_dim"]),
            "sample_counts": {k: int(v) for k, v in self.counts.items()},
            "runs_processed": int(run_count),
            "runs_skipped": int(skipped_count),
            "pm10_file": PM10_S2_FILE,
            "pm25_file": PM25_S2_FILE,
            "note": "Streaming 48h evaluation dataset aligned with current PM10+PM2.5 UTC main model layout.",
        }
        with open(os.path.join(self.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    import warnings

    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    clear_existing_dataset_outputs(OUTPUT_DATASET_DIR)

    print("Loading auxiliary data...", flush=True)
    data_veg = xr.open_dataset(VEG_FILE, engine="h5netcdf")
    data_oro = xr.open_dataset(ORO_FILE, engine="h5netcdf")

    if not os.path.exists(VIS_SOURCE_NC):
        print(f"Visibility source not found: {VIS_SOURCE_NC}", flush=True)
        vis_da = None
    else:
        ds_vis = xr.open_dataset(VIS_SOURCE_NC, engine="h5netcdf")
        if "vis" in ds_vis:
            ds_vis = ds_vis.rename({"vis": "visibility"})
        print("[Time Alignment] Visibility source raw time is UTC; no shift applied.", flush=True)
        vis_da = ds_vis["visibility"]

    pm10_da = load_station_pm_dataarray(PM10_S2_FILE, PM10_DIR, ("pm10", "PM10"), "PM10")
    pm25_da = load_station_pm_dataarray(PM25_S2_FILE, PM25_DIR, ("pm2p5", "pm25", "pm2_5", "PM2_5"), "PM2.5")

    run_list = get_run_list_from_current_48h()
    splits_to_write = parse_splits_to_write(SPLITS_TO_WRITE_ENV)
    print(
        f"Found {len(run_list)} runs in {CURRENT_48H_DIR}. "
        f"Building 12h windows within each 0–48h run only (no cross-run merge).",
        flush=True,
    )
    print(f"Streaming output splits: {', '.join(splits_to_write)}", flush=True)

    writer = None
    processed_runs = 0
    skipped_runs = 0
    for run_str in tqdm(run_list, desc="Runs"):
        ds_run, init_time = load_merged_run_ds(run_str, data_veg, data_oro)
        if ds_run is None:
            skipped_runs += 1
            continue
        if vis_da is None:
            vis_da_use = xr.DataArray(
                np.full((len(ds_run.time), len(ds_run.station_id)), np.nan, dtype=np.float32),
                dims=("time", "station_id"),
                coords={"time": ds_run.time, "station_id": ds_run.station_id},
            )
        else:
            vis_da_use = vis_da
        try:
            out = build_windows_and_features_per_run(
                ds_run,
                init_time,
                data_veg,
                data_oro,
                vis_da_use,
                pm10_da,
                pm25_da,
                splits_to_write,
            )
        except Exception as e:
            ds_run.close()
            print(f"  Run {run_str} skip: {e}", flush=True)
            skipped_runs += 1
            continue
        ds_run.close()
        gc.collect()
        if out is None:
            continue
        chunks, dims = out
        processed_runs += 1
        if chunks and writer is None:
            writer = Streaming48hWriter(OUTPUT_DATASET_DIR, splits_to_write, dims)
        if writer is not None:
            for tag, chunk in chunks.items():
                writer.write_chunk(tag, chunk, run_str)
        del chunks
        gc.collect()

    if writer is None:
        print("No selected 48h samples produced. Check SPLITS_TO_WRITE, current_48h, and visibility source.", flush=True)
        return

    writer.close(processed_runs, skipped_runs)
    print(f"Done. Output: {OUTPUT_DATASET_DIR}", flush=True)
    print(f"  Sample counts: {writer.counts}", flush=True)
    print(f"  Dyn dim: {writer.dims['dynamic_flat_dim'] // WINDOW_SIZE} (met+zenith+PM10+PM2.5)", flush=True)
    print(f"  FE dim (fog+cyclical): {writer.dims['feature_engineering_dim']}", flush=True)


if __name__ == "__main__":
    main()

