#!/usr/bin/env python3
"""
基于 current_48h 构建 ML 数据集：在每次起报后 48 小时内单独划分 12h 时间窗口，
不与其他起报时间合并，避免重叠与物理量跳跃；并加入 lead_time 特征供模型学习预报偏差。

数据流: tianji_auto_station/current_48h/*.nc -> 按 run 合并 -> 逐 run 内滑窗 -> 合并样本 -> 划分保存
"""
import os
import re
import gc
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from tqdm import tqdm

# 复用 PMST_s2_data 的配置与特征工程
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

# ================= 48h 专用配置 =================
BASE_PATH = "/public/home/putianshu/vis_mlp"
CURRENT_48H_DIR = os.path.join(BASE_PATH, "tianji_auto_station", "current_48h")
VIS_SOURCE_NC = os.path.join(BASE_PATH, "tianji_auto_station", "merged_final_all_vars.nc")  # 按 (time, station_id) 取 vis
VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"
OUTPUT_DATASET_DIR = os.path.join(BASE_PATH, "ml_dataset_fe_12h_48h")

# current_48h 中参与合并的变量（需与 PMST_s2_data 一致，且 t2mz -> TMP2m 等）
VARIABLES_48H = [
    "rh2m", "t2mz", "PRATEsfc", "slp", "DSWRFsfc",
    "UGRD10m", "VGRD10m", "cape", "cldl", "t925", "rh925",
    "u925", "v925", "dp1000", "dp925", "q1000", "q925", "omg925",
    "omg1000", "gust", "TMP2m",
]


def get_run_list_from_current_48h():
    """从 current_48h 目录解析出所有 run_str（如 2025010100, 2025010112）。"""
    # 任意一个变量即可得到全部 run
    pattern = os.path.join(CURRENT_48H_DIR, f"{VARIABLES_48H[0]}_*_0-48h_IDW.nc")
    files = glob(pattern)
    runs = []
    for f in files:
        base = os.path.basename(f)
        # rh2m_2025010100_0-48h_IDW.nc -> 2025010100
        m = re.match(r"^[^_]+_(\d{10})_0-48h_IDW\.nc$", base)
        if m:
            runs.append(m.group(1))
    return sorted(set(runs))


def _load_station_latlon():
    station_df = pd.read_csv("/public/home/putianshu/vis_mlp/tianji_auto_station/station_info.csv")
    station_df = station_df[
        (station_df["station_lon"] >= 65) & (station_df["station_lon"] <= 145)
        & (station_df["station_lat"] >= 10) & (station_df["station_lat"] <= 60)
    ]
    return station_df.set_index("num_station")


def load_merged_run_ds(run_str, data_veg, data_oro):
    """
    加载单个 run 的 current_48h 所有变量，合并为一个 Dataset；
    补充 D2M/DPD/INVERSION，并计算 lead_time（小时，从起报算起）。
    返回 (ds_merged, init_time) 或 (None, None)。ds 含 time, station_id, 各变量及 lead_time。
    """
    init_time = pd.to_datetime(run_str, format="%Y%m%d%H")
    ds_list = []
    for var in VARIABLES_48H:
        path = os.path.join(CURRENT_48H_DIR, f"{var}_{run_str}_0-48h_IDW.nc")
        if not os.path.exists(path):
            return None, None
        ds_list.append(xr.open_dataset(path))

    ds = xr.merge(ds_list, join="inner")
    for d in ds_list:
        d.close()
    del ds_list
    gc.collect()

    # 统一 station 维度名
    if "num_station" in ds.dims:
        ds = ds.rename({"num_station": "station_id"})
    # 站点 lat/lon（与 merge_parallel 一致）
    station_idx = _load_station_latlon()
    sids = ds.station_id.values
    lats = station_idx.loc[sids]["station_lat"].values
    lons = station_idx.loc[sids]["station_lon"].values
    ds = ds.assign_coords(lat=("station_id", lats), lon=("station_id", lons))

    times = pd.to_datetime(ds.time.values)
    lead_sec = (times - init_time).total_seconds()
    lead_hours = (lead_sec / 3600.0).astype(np.float32)
    ds["lead_time"] = (("time",), lead_hours)
    ds = ds.sortby("time")

    # 变量名映射到 FINAL_FEATURE_ORDER 所需
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

    # 风速/风向（若缺）
    if "WSPD10" not in ds and "U10" in ds and "V10" in ds:
        ds["WSPD10"] = np.sqrt(ds["U10"] ** 2 + ds["V10"] ** 2)
    if "WDIR10" not in ds and "U10" in ds and "V10" in ds:
        ds["WDIR10"] = np.degrees(np.arctan2(-ds["U10"], -ds["V10"])) % 360
    if "WSPD925" not in ds and "U_925" in ds and "V_925" in ds:
        ds["WSPD925"] = np.sqrt(ds["U_925"] ** 2 + ds["V_925"] ** 2)

    return ds, init_time


def build_windows_and_features_per_run(ds_run, init_time, data_veg, data_oro, vis_da):
    """
    仅在当前 run 的 48h 时间序列内划分 12h 窗口，不跨 run。
    返回: (X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta_tuple, lead_time_feature)
    """
    ds = ds_run
    lats = ds["lat"].values
    lons = ds["lon"].values
    times = pd.to_datetime(ds.time.values)
    stations = ds.station_id.values
    nt, ns = len(times), len(stations)

    # 动态特征矩阵 (time, station, var)
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

    lead_time = ds["lead_time"].values  # (nt,)

    # 静态特征
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

    # 仅在本 run 内滑窗：窗口数 = (nt - WINDOW_SIZE) // STEP_SIZE + 1，不与其他起报合并
    n_wins = (nt - WINDOW_SIZE) // STEP_SIZE + 1
    if n_wins <= 0:
        return None

    from numpy.lib.stride_tricks import sliding_window_view
    X_wins = sliding_window_view(X_dyn, WINDOW_SIZE, axis=0)[::STEP_SIZE]
    X_wins = X_wins.transpose(0, 1, 3, 2).reshape(-1, WINDOW_SIZE, nv)

    win_end_idx = (np.arange(n_wins) * STEP_SIZE + (WINDOW_SIZE - 1)).astype(int)
    win_end_times = times[win_end_idx]
    win_lead = lead_time[win_end_idx]

    # y: 从 vis 源按 (time, station_id) 向量化查
    if vis_da is not None and vis_da.dims >= 2:
        vis_sel = vis_da.sel(time=win_end_times, method="nearest")
        vis_sel = vis_sel.reindex(station_id=stations)
        y_flat = vis_sel.values.astype(np.float32).ravel()
    else:
        y_flat = np.full(n_wins * ns, np.nan, dtype=np.float32)

    # 元信息: 每个样本 (time, station_id, lat, lon)
    m_t = np.repeat(win_end_times, ns)
    m_s = np.tile(stations, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)
    meta = (m_t, m_s, m_la, m_lo)

    # 特征工程
    fe_flat = compute_fog_features(X_wins, WINDOW_SIZE, nv)

    # 周期时间编码
    sample_times = pd.DatetimeIndex(m_t)
    months = sample_times.month.values.astype(np.float32)
    hours = sample_times.hour.values.astype(np.float32)
    time_feat = np.column_stack([
        np.sin(2 * np.pi * months / 12),
        np.cos(2 * np.pi * months / 12),
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
    ]).astype(np.float32)

    # lead_time 特征：归一化到 0~1（48h 为 1）或保留小时/48
    lead_feat = np.repeat(win_lead, ns).astype(np.float32)[:, None] / 48.0
    fe_flat = np.concatenate([fe_flat, time_feat, lead_feat], axis=1)

    X_dyn_flat = X_wins.reshape(X_wins.shape[0], -1)
    X_stat_flat = np.tile(X_stat, (n_wins, 1))

    return X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta, lead_feat


def save_chunked_48h(X_dyn_flat, X_stat_flat, fe_flat, y_flat, mask, meta, out_dir):
    """与 PMST_s2_data 相同的月内划分保存逻辑。"""
    times_all, stats_all, lats_all, lons_all = meta
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
        pd.DataFrame({
            "time": times_all[ix],
            "station_id": stats_all[ix],
            "lat": lats_all[ix],
            "lon": lons_all[ix],
        }).to_csv(os.path.join(out_dir, f"meta_{tag}.csv"), index=False)
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

    # 可见度源：按 (time, station_id) 查
    if not os.path.exists(VIS_SOURCE_NC):
        print(f"Visibility source not found: {VIS_SOURCE_NC}", flush=True)
        print("Will use NaN for y; ensure current_48h valid times align with your vis file.", flush=True)
        vis_da = None
    else:
        ds_vis = xr.open_dataset(VIS_SOURCE_NC, engine="h5netcdf")
        if "vis" in ds_vis:
            ds_vis = ds_vis.rename({"vis": "visibility"})
        ds_vis = ds_vis.assign_coords(time=ds_vis.time - pd.Timedelta(hours=8))
        vis_da = ds_vis["visibility"]

    run_list = get_run_list_from_current_48h()
    print(f"Found {len(run_list)} runs in {CURRENT_48H_DIR}. Building windows per run only (no cross-run merge).", flush=True)

    X_dyn_list, X_stat_list, fe_list, y_list = [], [], [], []
    meta_t_list, meta_s_list, meta_la_list, meta_lo_list = [], [], [], []

    for run_str in tqdm(run_list, desc="Runs"):
        ds_run, init_time = load_merged_run_ds(run_str, data_veg, data_oro)
        if ds_run is None:
            continue
        if vis_da is None:
            vis_da_use = xr.DataArray(np.full((len(ds_run.time), len(ds_run.station_id)), np.nan), dims=("time", "station_id"), coords={"time": ds_run.time, "station_id": ds_run.station_id})
        else:
            vis_da_use = vis_da
        try:
            out = build_windows_and_features_per_run(ds_run, init_time, data_veg, data_oro, vis_da_use)
        except Exception as e:
            ds_run.close()
            print(f"  Run {run_str} skip: {e}", flush=True)
            continue
        ds_run.close()
        gc.collect()
        if out is None:
            continue
        X_dyn_flat, X_stat_flat, fe_flat, y_flat, meta, _ = out
        X_dyn_list.append(X_dyn_flat)
        X_stat_list.append(X_stat_flat)
        fe_list.append(fe_flat)
        y_list.append(y_flat)
        meta_t_list.append(meta[0])
        meta_s_list.append(meta[1])
        meta_la_list.append(meta[2])
        meta_lo_list.append(meta[3])

    if not X_dyn_list:
        print("No samples produced. Check current_48h and vis source.", flush=True)
        return

    X_dyn_flat = np.concatenate(X_dyn_list, axis=0)
    X_stat_flat = np.concatenate(X_stat_list, axis=0)
    fe_flat = np.concatenate(fe_list, axis=0)
    y_flat = np.concatenate(y_list, axis=0)
    m_t = np.concatenate(meta_t_list, axis=0)
    m_s = np.concatenate(meta_s_list, axis=0)
    m_la = np.concatenate(meta_la_list, axis=0)
    m_lo = np.concatenate(meta_lo_list, axis=0)
    del X_dyn_list, X_stat_list, fe_list, y_list, meta_t_list, meta_s_list, meta_la_list, meta_lo_list
    gc.collect()

    mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)
    save_chunked_48h(X_dyn_flat, X_stat_flat, fe_flat, y_flat, mask, (m_t, m_s, m_la, m_lo), OUTPUT_DATASET_DIR)
    print(f"Done. Output: {OUTPUT_DATASET_DIR}", flush=True)
    print(f"  FE dim (with lead_time): {fe_flat.shape[1]} (physics + 4 cyclical + 1 lead_time)", flush=True)


if __name__ == "__main__":
    main()
