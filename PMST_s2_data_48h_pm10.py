#!/usr/bin/env python3
"""
基于 current_48h 构建 ML 数据集 (s2, 2025)：
- 每次起报后 0–48 小时内单独划分 12h 时间窗口，不与其他起报时间合并；
- 气象变量顺序与 stage1 (PMST_s1_data_pm10) 一致：FINAL_FEATURE_ORDER 同序（ERA5 命名不同但变量一一对应）。
- 预报特有：lead_time、周期时间编码(月/时)、以及 PMST_s2_data 的 compute_fog_features 中多出的 7 维见下方注释。

输出目录:
  vis_mlp/ml_dataset_fe_12h_48h_pm10

--- Stage2 有而 Stage1 没有的特征工程（仅列出，不删除）---
1) 周期时间编码：fe 中增加 4 维 [sin(2π*month/12), cos(2π*month/12), sin(2π*hour/24), cos(2π*hour/24)]（观测时刻）。
2) 预报时效 lead_hour 写入 meta CSV，不进入 FE（与 PMST_net 输入维一致）。
3) compute_fog_features：PMST_s2_data 在 stage1 的 24 维 fog 特征之后多 8 维：
   shear_mag, dir_turning, convective_wet, daytime_mixing, ventilation, moisture_strat, omega_contrast, warm_instability。
   即 stage2 的 fog 部分为 32 维（24+8），stage1 为 24 维。
4) 若提供 pm10 文件，将 pm10 作为动态变量追加到 dyn（并在训练端统一 log1p）。FE 总维数固定为 32+4 = 36。
"""

import os
import re
import gc
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from tqdm import tqdm

# 与 stage1 气象变量顺序一致（FINAL_FEATURE_ORDER 同序）；预报侧特征工程仍用 PMST_s2_data
from PMST_s2_data import (
    VAR_MAPPING,
    FINAL_FEATURE_ORDER,
    TRAIN_RATIO,
    VAL_RATIO,
    GAP_HOURS,
    WINDOW_SIZE,
    STEP_SIZE,
    UNIQUE_VEG_IDS,
    MAX_VIS_THRESHOLD,
    compute_fog_features,
    calculate_dewpoint_from_rh,
    calculate_zenith_angle,
    get_nearest_veg,
    extract_terrain,
    get_monthly_split_mask,
)

BASE_PATH = "/public/home/putianshu/vis_mlp"
CURRENT_48H_DIR = os.path.join(BASE_PATH, "tianji_auto_station", "current_48h")
VIS_SOURCE_NC = os.path.join(BASE_PATH, "tianji_auto_station", "merged_final_all_vars.nc")
VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"
OUTPUT_DATASET_DIR = os.path.join(BASE_PATH, "ml_dataset_fe_12h_48h_pm10")

PM10_S2_FILE = os.path.join(BASE_PATH, "pm10_station", "pm10_station_s2_2025.nc")

# current_48h 目录中 2m 温度为 TMP2m（无 t2mz），与 VAR_MAPPING 一致
VARIABLES_48H = [
    "rh2m", "PRATEsfc", "slp", "DSWRFsfc",
    "UGRD10m", "VGRD10m", "cape", "cldl", "t925", "rh925",
    "u925", "v925", "dp1000", "dp925", "q1000", "q925", "omg925",
    "omg1000", "gust", "TMP2m",
]


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


def build_windows_and_features_per_run(ds_run, init_time, data_veg, data_oro, vis_da, pm10_da):
    """
    仅在当前 run 的 48h 时间序列内划分 12h 窗口，不跨 run。
    返回: (X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta_tuple)
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
    nv = X_dyn.shape[-1]
    del X_met, zenith
    gc.collect()

    # ========= 将 pm10 作为动态变量追加到 dyn 的最后一维 =========
    # pm10 文件单位为 kg/m^3，这里转换为 ug/m^3 后留给训练脚本做 log1p。
    if pm10_da is not None:
        pm10_grid_da = pm10_da
        if set(pm10_grid_da.dims) >= {"time", "station_id"}:
            pm10_grid_da = pm10_grid_da.transpose("time", "station_id")
        else:
            raise ValueError(
                f"pm10_da dims must contain 'time' and 'station_id', got {pm10_grid_da.dims}"
            )
        pm10_grid_da = pm10_grid_da.load()

        time_vals = pm10_grid_da["time"].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_index = pd.DatetimeIndex(time_vals)
        else:
            time_index = pd.to_datetime(time_vals, unit="s", origin="unix")

        ds_times = pd.DatetimeIndex(times)
        sid_index = pd.Index(pm10_grid_da["station_id"].values)
        sids = stations.astype(pm10_grid_da["station_id"].dtype)

        time_pos = time_index.get_indexer(ds_times, method="nearest")
        sid_pos = sid_index.get_indexer(sids)

        nt_ds, ns_ds = len(ds_times), len(sids)
        nt_pm10, ns_pm10 = pm10_grid_da.shape
        pm10_grid = np.full((nt_ds, ns_ds), np.nan, dtype=np.float32)

        base = np.asarray(pm10_grid_da.values).reshape(-1)
        linear_idx_grid = time_pos[:, None] * ns_pm10 + sid_pos[None, :]
        ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
        if ok_mask.any():
            pm10_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)

        pm10_grid = np.maximum(pm10_grid, 0.0)
        pm10_ugm3 = pm10_grid * 1e12
        pm10_median = np.nanmedian(pm10_ugm3)
        if not np.isfinite(pm10_median):
            pm10_median = 0.0
        pm10_ugm3 = np.where(np.isfinite(pm10_ugm3), pm10_ugm3, pm10_median).astype(np.float32)

        X_dyn = np.concatenate([X_dyn, pm10_ugm3[..., None]], axis=-1)
        nv = X_dyn.shape[-1]

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

    n_wins = (nt - WINDOW_SIZE) // STEP_SIZE + 1
    if n_wins <= 0:
        return None

    from numpy.lib.stride_tricks import sliding_window_view

    X_wins = sliding_window_view(X_dyn, WINDOW_SIZE, axis=0)[::STEP_SIZE]
    # 须与 PMST_s2_data.py / s2_data_pm10_monthtail_cell 一致：(n_wins, ns, W, nv)，与 m_t=repeat(win,ns) 对齐
    X_wins = X_wins.transpose(0, 1, 3, 2).reshape(-1, WINDOW_SIZE, nv)

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
    meta = (m_t, m_s, m_la, m_lo, m_lead)

    fe_flat = compute_fog_features(X_wins, WINDOW_SIZE, nv)

    sample_times = pd.DatetimeIndex(m_t)
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

    # pm10 已作为动态变量进入 dyn，FE_flat 不再附加 pm10；lead 仅在 meta

    X_dyn_flat = X_wins.reshape(X_wins.shape[0], -1)
    X_stat_flat = np.tile(X_stat, (n_wins, 1))

    return X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta


def save_chunked_48h(X_dyn_flat, X_stat_flat, fe_flat, y_flat, mask, meta, out_dir):
    times_all, stats_all, lats_all, lons_all, lead_all, init_all = meta
    valid_idxs = np.where(mask)[0]
    n = len(valid_idxs)
    print(f"  Valid Samples: {n} ({n / len(mask):.1%})", flush=True)
    valid_times = pd.DatetimeIndex(times_all[valid_idxs])
    tr_m, val_m, test_m = get_monthly_split_mask(valid_times)
    splits = {"train": valid_idxs[tr_m], "val": valid_idxs[val_m], "test": valid_idxs[test_m]}
    dims = [X_dyn_flat.shape[1], X_stat_flat.shape[1], fe_flat.shape[1]]
    for tag, ix in splits.items():
        if len(ix) == 0:
            continue
        print(f"    Saving {tag} (N={len(ix)})", flush=True)
        np.save(os.path.join(out_dir, f"y_{tag}.npy"), y_flat[ix])
        pd.DataFrame(
            {
                "time": times_all[ix],
                "station_id": stats_all[ix],
                "lat": lats_all[ix],
                "lon": lons_all[ix],
                "lead_hour": lead_all[ix].astype(np.float32),
                "init_time": init_all[ix],
            }
        ).to_csv(os.path.join(out_dir, f"meta_{tag}.csv"), index=False)
        fp = np.lib.format.open_memmap(
            os.path.join(out_dir, f"X_{tag}.npy"), mode="w+", dtype="float32", shape=(len(ix), sum(dims))
        )
        bs = 100000
        for i in tqdm(range(0, len(ix), bs), desc=tag):
            bi = ix[i : i + bs]
            n_bi = len(bi)
            fp[i : i + n_bi, : dims[0]] = X_dyn_flat[bi]
            fp[i : i + n_bi, dims[0] : dims[0] + dims[1]] = X_stat_flat[bi]
            fp[i : i + n_bi, dims[0] + dims[1] :] = fe_flat[bi]
        del fp
        gc.collect()


def main():
    import warnings

    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

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

    if os.path.exists(PM10_S2_FILE):
        pm10_ds = xr.open_dataset(PM10_S2_FILE, engine="h5netcdf")
        pm10_da = pm10_ds["pm10"]
    else:
        print(f"[WARN] pm10 file for s2 not found: {PM10_S2_FILE}, pm10 will be NaN.", flush=True)
        pm10_da = None

    run_list = get_run_list_from_current_48h()
    print(
        f"Found {len(run_list)} runs in {CURRENT_48H_DIR}. "
        f"Building 12h windows within each 0–48h run only (no cross-run merge).",
        flush=True,
    )

    X_dyn_list, X_stat_list, fe_list, y_list = [], [], [], []
    meta_t_list, meta_s_list, meta_la_list, meta_lo_list = [], [], [], []
    meta_lead_list, meta_init_list = [], []

    for run_str in tqdm(run_list, desc="Runs"):
        ds_run, init_time = load_merged_run_ds(run_str, data_veg, data_oro)
        if ds_run is None:
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
            out = build_windows_and_features_per_run(ds_run, init_time, data_veg, data_oro, vis_da_use, pm10_da)
        except Exception as e:
            ds_run.close()
            print(f"  Run {run_str} skip: {e}", flush=True)
            continue
        ds_run.close()
        gc.collect()
        if out is None:
            continue
        X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta = out
        X_dyn_list.append(X_dyn_flat)
        X_stat_list.append(X_stat_flat)
        fe_list.append(fe_flat)
        y_list.append(y_flat)
        meta_t_list.append(meta[0])
        meta_s_list.append(meta[1])
        meta_la_list.append(meta[2])
        meta_lo_list.append(meta[3])
        meta_lead_list.append(meta[4])
        meta_init_list.append(
            np.full(len(meta[0]), run_str, dtype=object)
        )

    if not X_dyn_list:
        print("No samples produced. Check current_48h, vis source, and pm10 files.", flush=True)
        return

    X_dyn_flat = np.concatenate(X_dyn_list, axis=0)
    X_stat_flat = np.concatenate(X_stat_list, axis=0)
    fe_flat = np.concatenate(fe_list, axis=0)
    y_flat = np.concatenate(y_list, axis=0)
    m_t = np.concatenate(meta_t_list, axis=0)
    m_s = np.concatenate(meta_s_list, axis=0)
    m_la = np.concatenate(meta_la_list, axis=0)
    m_lo = np.concatenate(meta_lo_list, axis=0)
    m_lead = np.concatenate(meta_lead_list, axis=0)
    m_init = np.concatenate(meta_init_list, axis=0)
    del X_dyn_list, X_stat_list, fe_list, y_list, meta_t_list, meta_s_list, meta_la_list, meta_lo_list
    del meta_lead_list, meta_init_list
    gc.collect()

    mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)
    save_chunked_48h(
        X_dyn_flat, X_stat_flat, fe_flat, y_flat, mask,
        (m_t, m_s, m_la, m_lo, m_lead, m_init),
        OUTPUT_DATASET_DIR,
    )
    print(f"Done. Output: {OUTPUT_DATASET_DIR}", flush=True)
    print(f"  FE dim (fog+cyclical): {fe_flat.shape[1]}", flush=True)


if __name__ == "__main__":
    main()

