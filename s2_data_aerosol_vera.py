import os
import glob
import gc
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

# ================= 配置（与 train/PMST_s2_data_48h_pm10 末尾单元格保持同一套“月末天数”语义） =================
BASE_PATH = "/public/home/putianshu/vis_mlp"
INPUT_FILE = os.path.join(BASE_PATH, "tianji_auto_station", "merged_final_all_vars.nc")
VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"
OUTPUT_DATASET_DIR = os.path.join(BASE_PATH, "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2_vera")

# PM10：优先单文件（与 48h 一致），否则尝试目录下多 nc 拼接
PM10_S2_FILE = os.path.join(BASE_PATH, "pm10_station", "pm10_station_s2_2025.nc")
PM10_DIR = os.path.join(BASE_PATH, "pm10_station")

# PM2.5 站点插值（pm25_station_IDW.py）
PM25_S2_FILE = os.path.join(BASE_PATH, "pm2.5_station", "pm2p5_station_s2_2025.nc")
PM25_DIR = os.path.join(BASE_PATH, "pm2.5_station")

VAR_MAPPING = {
    "rh2m": "RH2M",
    "TMP2m": "T2M",
    "PRATEsfc": "PRECIP",
    "slp": "MSLP",
    "DSWRFsfc": "SW_RAD",
    "UGRD10m": "U10",
    "VGRD10m": "V10",
    "wind_speed": "WSPD10",
    "wd10m": "WDIR10",
    "cape": "CAPE",
    "cldl": "LCC",
    "t925": "T_925",
    "rh925": "RH_925",
    "u925": "U_925",
    "v925": "V_925",
    "wind_speed_925": "WSPD925",
    "dp1000": "DP_1000",
    "dp925": "DP_925",
    "q1000": "Q_1000",
    "q925": "Q_925",
    "omg925": "W_925",
    "omg1000": "W_1000",
    "t1000": "T_1000",
}

FINAL_FEATURE_ORDER = [
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

# Monthly tail split with fixed retained val/test days plus 24h gaps
VAL_LAST_DAYS = 3
TEST_LAST_DAYS = 3
GAP_HOURS = 24

WINDOW_SIZE, STEP_SIZE = 12, 1
UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])
MAX_VIS_THRESHOLD = 30000
# 沿时间窗分批物化样本，避免一次性分配 (n_wins×n_st)×12×nv 的 X_samples（易 OOM）
TIME_CHUNK_WINS = 48

BASE_FOG_FEATURE_NAMES = [
    "sat_dpd", "wind_favourability", "stability_ri", "night_clear_radiation",
    "rh_surface_minus_925", "fog_potential",
    "rh2m_delta_3h", "rh2m_delta_6h", "rh2m_std_12h", "rh2m_range_12h",
    "t2m_delta_3h", "t2m_delta_6h", "t2m_std_12h", "t2m_range_12h",
    "wspd10_delta_3h", "wspd10_delta_6h", "wspd10_std_12h", "wspd10_range_12h",
    "rh2m_accel", "humid_cold_proxy", "night_low_cloud_proxy",
    "cold_humid_weak_wind_flag", "rh_low_cloud_ratio", "rh_squared",
    "low_level_shear", "wind_direction_turning", "convective_wet_proxy",
    "daytime_mixing_proxy", "ventilation_proxy", "moisture_stratification",
    "omega_contrast", "warm_instability_proxy",
]

VERA_FEATURE_NAMES = [
    "vera_pm25_fraction", "vera_coarse_pm_log", "vera_growth_fine_log",
    "vera_growth_coarse_log", "vera_hydrated_pm25", "vera_hydrated_coarse",
    "vera_aerosol_ext_proxy", "vera_near_saturation_activation",
    "vera_liquid_water_proxy", "vera_precip_ext_proxy", "vera_total_ext_proxy",
    "vera_inverse_visibility_proxy", "vera_rh_error_sensitivity",
    "vera_ext_std_12h", "vera_ext_trend_3h", "vera_ext_peak_12h",
]

TIME_FEATURE_NAMES = ["month_sin", "month_cos", "hour_sin", "hour_cos"]
FE_FEATURE_NAMES = BASE_FOG_FEATURE_NAMES + VERA_FEATURE_NAMES + TIME_FEATURE_NAMES


def get_monthly_split_mask_last_days(sample_times, gap_hours, val_last_days, test_last_days):
    """Month-tail split with explicit train/val/test day counts and hourly gaps."""
    times = pd.DatetimeIndex(sample_times)
    n = len(times)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    if gap_hours % 24 != 0:
        raise ValueError(f"gap_hours must be a multiple of 24 for day-based split, got {gap_hours}")
    gap_days = gap_hours // 24

    for month_period in times.to_period("M").unique():
        start = month_period.start_time
        end = month_period.end_time
        dim = month_period.days_in_month

        # Layout: [train] [gap_days] [val_last_days] [gap_days] [test_last_days]
        d_test0 = dim - test_last_days + 1
        d_val1 = d_test0 - gap_days - 1
        d_val0 = d_val1 - val_last_days + 1
        d_train_end = d_val0 - gap_days - 1

        if d_train_end < 1 or d_val0 < 1 or d_val1 < d_val0:
            print(
                f"  [WARN] split skip month {month_period}: "
                f"d_train_end={d_train_end}, d_val0={d_val0}, d_val1={d_val1}, d_test0={d_test0}",
                flush=True,
            )
            continue

        t_train_end = pd.Timestamp(month_period.year, month_period.month, d_train_end)
        t_val0 = pd.Timestamp(month_period.year, month_period.month, d_val0)
        t_val1 = pd.Timestamp(month_period.year, month_period.month, d_val1)
        t_test0 = pd.Timestamp(month_period.year, month_period.month, d_test0)

        msub = (times >= start) & (times <= end)
        train_mask |= msub & (times <= t_train_end + pd.Timedelta(hours=23, minutes=59, seconds=59))
        val_mask |= msub & (times >= t_val0) & (times <= t_val1 + pd.Timedelta(hours=23, minutes=59, seconds=59))
        test_mask |= msub & (times >= t_test0)

    return train_mask, val_mask, test_mask
def load_pm10_dataarray():
    """返回 (time, station_id) 的 pm10 DataArray，或 None。"""
    das = []
    if os.path.isfile(PM10_S2_FILE):
        ds = xr.open_dataset(PM10_S2_FILE, engine="h5netcdf")
        v = "pm10" if "pm10" in ds else list(ds.data_vars)[0]
        das.append(ds[v])
        ds.close()
    elif os.path.isdir(PM10_DIR):
        files = sorted(glob.glob(os.path.join(PM10_DIR, "*.nc")))
        for fp in files:
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
                if "station_id" not in ds.coords and "station_id" not in ds.data_vars:
                    for alias in ["num_station", "id", "station"]:
                        if alias in ds.coords or alias in ds.data_vars:
                            ds = ds.rename({alias: "station_id"})
                            break
                vn = list(ds.data_vars)[0]
                das.append(ds[vn])
                ds.close()
            except Exception as e:
                print(f"[WARN] skip pm10 file {fp}: {e}", flush=True)
    if not das:
        print("[WARN] No PM10 files found; pm10 column will be NaN.", flush=True)
        return None
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if "station_id" not in da.dims:
        raise ValueError("pm10 DataArray must have station_id dimension")
    return da.transpose("time", "station_id")


def load_pm25_dataarray():
    """返回 (time, station_id) 的 pm2p5 DataArray，或 None。"""
    das = []
    if os.path.isfile(PM25_S2_FILE):
        ds = xr.open_dataset(PM25_S2_FILE, engine="h5netcdf")
        v = "pm2p5" if "pm2p5" in ds else list(ds.data_vars)[0]
        das.append(ds[v])
        ds.close()
    elif os.path.isdir(PM25_DIR):
        files = sorted(glob.glob(os.path.join(PM25_DIR, "*.nc")))
        for fp in files:
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
                if "station_id" not in ds.coords and "station_id" not in ds.data_vars:
                    for alias in ["num_station", "id", "station"]:
                        if alias in ds.coords or alias in ds.data_vars:
                            ds = ds.rename({alias: "station_id"})
                            break
                if "pm2p5" in ds:
                    vn = "pm2p5"
                else:
                    vn = list(ds.data_vars)[0]
                das.append(ds[vn])
                ds.close()
            except Exception as e:
                print(f"[WARN] skip pm2.5 file {fp}: {e}", flush=True)
    if not das:
        print("[WARN] No PM2.5 station files found; pm2.5 channel will be zeros.", flush=True)
        return None
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if "station_id" not in da.dims:
        raise ValueError("pm2.5 DataArray must have station_id dimension")
    return da.transpose("time", "station_id")


def cams_station_pm_to_ugm3_grid(pm_da, times, stats):
    """将 CAMS 插值到站点的气溶胶对齐到 Tianji 时间/站点；×1e12 转为 µg/m³（与 pm10 单元格一致）。"""
    nt_ds, ns_ds = len(times), len(stats)
    if pm_da is None:
        return np.zeros((nt_ds, ns_ds), dtype=np.float32)
    pm_grid_da = pm_da
    if set(pm_grid_da.dims) >= {"time", "station_id"}:
        pm_grid_da = pm_grid_da.transpose("time", "station_id")
    else:
        raise ValueError(
            f"pm_da dims must contain 'time' and 'station_id', got {pm_grid_da.dims}"
        )
    pm_grid_da = pm_grid_da.load()
    time_vals = pm_grid_da["time"].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        time_index = pd.DatetimeIndex(time_vals)
    else:
        time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
    ds_times = pd.DatetimeIndex(times)
    sid_index = pd.Index(pm_grid_da["station_id"].values)
    sids = stats.astype(pm_grid_da["station_id"].dtype)
    time_pos = time_index.get_indexer(ds_times, method="nearest")
    sid_pos = sid_index.get_indexer(sids)
    _, ns_pm = pm_grid_da.shape
    pm_grid = np.full((nt_ds, ns_ds), np.nan, dtype=np.float32)
    base = np.asarray(pm_grid_da.values, dtype=np.float32).reshape(-1)
    del pm_grid_da
    gc.collect()
    linear_idx_grid = time_pos[:, None] * ns_pm + sid_pos[None, :]
    ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
    if ok_mask.any():
        pm_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
    pm_grid = np.maximum(pm_grid, 0.0)
    pm_ugm3 = pm_grid * 1e12
    del pm_grid, base, linear_idx_grid, ok_mask
    med = np.nanmedian(pm_ugm3)
    if not np.isfinite(med):
        med = 0.0
    out = np.where(np.isfinite(pm_ugm3), pm_ugm3, med).astype(np.float32)
    del pm_ugm3
    gc.collect()
    return out


def pm10_values_for_samples(pm10_da, sample_times, station_ids):
    """与 PMST_s1/s2_48h 一致：逐样本最近邻索引，避免 xarray 外积。"""
    if pm10_da is None:
        return np.full(len(sample_times), np.nan, dtype=np.float32)
    time_vals = pm10_da["time"].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        time_index = pd.DatetimeIndex(time_vals)
    else:
        time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
    sid_index = pd.Index(pm10_da["station_id"].values)
    st = pd.DatetimeIndex(sample_times)
    sids = station_ids.astype(pm10_da["station_id"].dtype)
    t_pos = time_index.get_indexer(st, method="nearest")
    s_pos = sid_index.get_indexer(sids)
    ok = (t_pos >= 0) & (s_pos >= 0)
    out = np.full(len(st), np.nan, dtype=np.float32)
    if ok.any():
        arr = np.asarray(pm10_da.values).reshape(len(time_index), len(sid_index))
        out[ok] = arr[t_pos[ok], s_pos[ok]].astype(np.float32)
    return out


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _aerosol_channels_from_window(X_dyn_window, dyn_vars):
    n = X_dyn_window.shape[0]
    if dyn_vars >= 27:
        pm10 = np.maximum(X_dyn_window[:, :, dyn_vars - 2], 0.0)
        pm25 = np.maximum(X_dyn_window[:, :, dyn_vars - 1], 0.0)
    elif dyn_vars >= 26:
        pm10 = np.maximum(X_dyn_window[:, :, dyn_vars - 1], 0.0)
        pm25 = np.zeros((n, X_dyn_window.shape[1]), dtype=np.float32)
    else:
        pm10 = np.zeros((n, X_dyn_window.shape[1]), dtype=np.float32)
        pm25 = np.zeros((n, X_dyn_window.shape[1]), dtype=np.float32)
    return pm10.astype(np.float32), pm25.astype(np.float32)


def compute_vera_features(X_dyn_window, dyn_vars):
    """VERA-inspired deterministic intermediates, not a full VERA diagnosis."""
    idx = {
        "rh2m": 0, "precip": 2, "wspd10": 6, "lcc": 10,
        "q1000": 18, "q925": 19, "dpd": 22, "inversion": 23,
    }
    X_current = X_dyn_window[:, -1, :]
    rh2m = X_current[:, idx["rh2m"]]
    dpd = X_current[:, idx["dpd"]]
    lcc = np.clip(X_current[:, idx["lcc"]], 0.0, 1.0)
    precip = np.maximum(X_current[:, idx["precip"]], 0.0)
    wspd = np.maximum(X_current[:, idx["wspd10"]], 0.0)
    inversion = X_current[:, idx["inversion"]]
    q1000 = X_current[:, idx["q1000"]]
    q925 = X_current[:, idx["q925"]]

    pm10_seq, pm25_seq = _aerosol_channels_from_window(X_dyn_window, dyn_vars)
    pm10 = pm10_seq[:, -1]
    pm25 = pm25_seq[:, -1]
    coarse = np.maximum(pm10 - pm25, 0.0)

    eps = 1e-6
    rh_frac = np.clip(rh2m / 100.0, 0.01, 0.995)
    one_minus_rh = np.maximum(1.0 - rh_frac, 0.005)
    g_fine = np.clip(one_minus_rh ** -0.85, 1.0, 10.0)
    g_coarse = np.clip(one_minus_rh ** -0.45, 1.0, 6.0)
    near_sat = _sigmoid_np((rh2m - 95.0) / 2.5) * _sigmoid_np(-dpd / 1.5)
    weak_wind = 1.0 / (1.0 + np.maximum(wspd, 0.0))
    stability = np.clip(inversion, 0.0, 6.0) / 6.0
    moisture_pool = np.clip((q1000 - q925) * 1500.0, -3.0, 3.0)

    pm25_fraction = pm25 / (pm10 + eps)
    coarse_pm_log = np.log1p(coarse)
    growth_fine_log = np.log1p(g_fine)
    growth_coarse_log = np.log1p(g_coarse)
    hydrated_pm25 = np.log1p(pm25) * growth_fine_log
    hydrated_coarse = np.log1p(coarse) * growth_coarse_log
    aerosol_ext = 0.80 * hydrated_pm25 + 0.25 * hydrated_coarse
    liquid_water_proxy = near_sat * (0.5 + 0.5 * lcc) * (1.0 + 0.25 * stability)
    precip_ext = np.log1p(precip) * (0.5 + 0.5 * near_sat)
    total_ext = aerosol_ext + 2.0 * liquid_water_proxy + 0.5 * precip_ext
    inv_vis = 1.0 / (0.10 + total_ext)
    rh_error_sens = np.log1p(np.clip((rh_frac / one_minus_rh) * g_fine, 0.0, 80.0))

    rh_seq = np.clip(X_dyn_window[:, :, idx["rh2m"]] / 100.0, 0.01, 0.995)
    one_minus_seq = np.maximum(1.0 - rh_seq, 0.005)
    g_fine_seq = np.clip(one_minus_seq ** -0.85, 1.0, 10.0)
    g_coarse_seq = np.clip(one_minus_seq ** -0.45, 1.0, 6.0)
    coarse_seq = np.maximum(pm10_seq - pm25_seq, 0.0)
    aerosol_seq = (
        0.80 * np.log1p(pm25_seq) * np.log1p(g_fine_seq)
        + 0.25 * np.log1p(coarse_seq) * np.log1p(g_coarse_seq)
    )
    ext_std = np.std(aerosol_seq, axis=1)
    ext_trend_3h = aerosol_seq[:, -1] - aerosol_seq[:, -4]
    ext_peak = np.max(aerosol_seq, axis=1)

    feats = [
        np.clip(pm25_fraction, 0.0, 1.5), coarse_pm_log, growth_fine_log,
        growth_coarse_log, hydrated_pm25, hydrated_coarse, aerosol_ext,
        near_sat, liquid_water_proxy, precip_ext, np.log1p(total_ext),
        np.clip(inv_vis, 0.0, 10.0), rh_error_sens, ext_std, ext_trend_3h,
        ext_peak + 0.1 * np.maximum(moisture_pool, 0.0) * weak_wind,
    ]
    return np.column_stack(feats).astype(np.float32)


def write_feature_layout(out_dir):
    layout = {
        "dynamic_order": FINAL_FEATURE_ORDER + ["zenith", "PM10_ugm3", "PM25_ugm3"],
        "static_order": ["lat_norm", "lon_norm", "orography", "orography_anom", "orography_std", "veg_type"],
        "fe_order": FE_FEATURE_NAMES,
        "note": "VERA-inspired features are deterministic proxies from PM10/PM2.5, RH, cloud, precip and temporal sensitivity; they are not a full VERA implementation.",
    }
    with open(os.path.join(out_dir, "feature_layout_vera.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)


def compute_fog_features(X_dyn_window, window_size=12, dyn_vars=25):
    idx = {
        "rh2m": 0,
        "t2m": 1,
        "precip": 2,
        "sw_rad": 4,
        "wspd10": 6,
        "cape": 9,
        "lcc": 10,
        "t925": 11,
        "rh925": 12,
        "dpd": 22,
        "inversion": 23,
        "zenith": 24,
    }
    params = {
        "optimal_wspd": 3.5,
        "wspd_sigma": 2.5,
        "dpd_threshold": 2.0,
        "stability_scale": 2.0,
        "lcc_threshold": 0.3,
        "rad_threshold": 800.0,
    }
    X_current = X_dyn_window[:, -1, :]
    rh2m, rh925 = X_current[:, idx["rh2m"]], X_current[:, idx["rh925"]]
    dpd, wspd = X_current[:, idx["dpd"]], X_current[:, idx["wspd10"]]
    inversion, sw_rad = X_current[:, idx["inversion"]], X_current[:, idx["sw_rad"]]
    lcc, zenith = X_current[:, idx["lcc"]], X_current[:, idx["zenith"]]
    t2m = X_current[:, idx["t2m"]]
    t2m_c = t2m - 273.15 if np.mean(t2m) > 200 else t2m
    feats = []
    rh_norm = np.clip(rh2m / 100.0, 0, 1)
    dpd_weight = 1.0 / (1.0 + np.exp(dpd / params["dpd_threshold"]))
    feats.append((rh_norm * dpd_weight).reshape(-1, 1))
    wind_fav = np.exp(-0.5 * ((wspd - params["optimal_wspd"]) / params["wspd_sigma"]) ** 2)
    feats.append(wind_fav.reshape(-1, 1))
    ri = inversion / (wspd**2 + 0.1)
    stability = np.tanh(ri / params["stability_scale"])
    feats.append(stability.reshape(-1, 1))
    is_night = (zenith > 90.0).astype(float)
    clear_sky = np.clip(1.0 - lcc / params["lcc_threshold"], 0, 1)
    rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0) / params["rad_threshold"], 0, 1)
    feats.append((is_night * clear_sky * rad_intensity).reshape(-1, 1))
    feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))
    fog_pot = (
        rh_norm * 0.4
        + wind_fav * 0.25
        + np.clip(stability, 0, 1) * 0.2
        + (is_night * clear_sky * rad_intensity) * 0.15
    )
    feats.append(fog_pot.reshape(-1, 1))
    for var_idx in [idx["rh2m"], idx["t2m"], idx["wspd10"]]:
        var_seq = X_dyn_window[:, :, var_idx]
        feats.append((var_seq[:, -1] - var_seq[:, -4]).reshape(-1, 1))
        feats.append((var_seq[:, -1] - var_seq[:, -7]).reshape(-1, 1))
        feats.append(np.std(var_seq, axis=1).reshape(-1, 1))
        feats.append((np.max(var_seq, axis=1) - np.min(var_seq, axis=1)).reshape(-1, 1))
    rh_seq = X_dyn_window[:, :, idx["rh2m"]]
    rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
    feats.append(rh_accel.reshape(-1, 1))
    feats.append((rh2m * np.exp(-t2m_c / 10.0)).reshape(-1, 1))
    feats.append((is_night * (1 - lcc)).reshape(-1, 1))
    feats.append(((rh2m > 90) & (t2m_c < 10) & (wspd < 4)).astype(float).reshape(-1, 1))
    feats.append((rh2m / (lcc * 100 + 1)).reshape(-1, 1))
    feats.append(((rh2m / 100.0) ** 2).reshape(-1, 1))
    u10 = X_current[:, 5]
    v10 = X_current[:, 7]
    u925 = X_current[:, 13]
    v925 = X_current[:, 15]
    precip = np.maximum(X_current[:, idx["precip"]], 0.0)
    cape = np.maximum(X_current[:, idx["cape"]], 0.0)
    q1000 = X_current[:, 18]
    q925 = X_current[:, 19]
    w925 = X_current[:, 20]
    w1000 = X_current[:, 21]
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    theta10 = np.arctan2(v10, u10)
    theta925 = np.arctan2(v925, u925)
    dir_turning = 0.5 * (1.0 - np.cos(theta925 - theta10))
    convective_wet = (1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))) * (
        1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5))
    )
    daytime_mixing = (1.0 / (1.0 + np.exp(-(np.maximum(sw_rad, 0.0) - 150.0) / 75.0))) * (
        1.0 / (1.0 + np.exp(-(wspd - 4.0) / 1.5))
    ) * (1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2)))
    ventilation = np.tanh((wspd * (1.0 + shear_mag)) / 12.0)
    moisture_strat = np.tanh((q1000 - q925) * 1500.0)
    omega_contrast = np.tanh((w925 - w1000) / 0.25)
    warm_instability = np.tanh((-inversion + np.maximum(t2m_c - 18.0, 0.0) * 0.25) / 3.0)
    feats.append(np.tanh(shear_mag / 8.0).reshape(-1, 1))
    feats.append(dir_turning.reshape(-1, 1))
    feats.append(convective_wet.reshape(-1, 1))
    feats.append(daytime_mixing.reshape(-1, 1))
    feats.append(ventilation.reshape(-1, 1))
    feats.append(moisture_strat.reshape(-1, 1))
    feats.append(omega_contrast.reshape(-1, 1))
    feats.append(warm_instability.reshape(-1, 1))
    base = np.concatenate(feats, axis=1).astype(np.float32)
    vera = compute_vera_features(X_dyn_window, dyn_vars)
    return np.concatenate([base, vera], axis=1).astype(np.float32)


def calculate_dewpoint_from_rh(t, rh):
    t_c = t - 273.15
    rh_frac = np.maximum(rh / 100.0, 0.001)
    b, c = 17.67, 243.5
    gamma = np.log(rh_frac) + (b * t_c) / (c + t_c)
    return (c * gamma) / (b - gamma) + 273.15


def calculate_zenith_angle(latitudes, longitudes, times):
    try:
        times_pd = pd.DatetimeIndex(times)
        if times_pd.tz is None:
            times_pd = times_pd.tz_localize("UTC")
    except Exception:
        times_pd = pd.DatetimeIndex(times).tz_localize("UTC")
    n_times, n_stations = len(times_pd), len(latitudes)
    b_times = np.repeat(times_pd, n_stations)
    b_lats = np.tile(latitudes, n_times)
    b_lons = np.tile(longitudes, n_times)
    try:
        sp = pvlib.solarposition.get_solarposition(b_times, b_lats, b_lons)
        return sp["apparent_zenith"].values.reshape(n_times, n_stations, 1).astype(np.float32)
    except Exception:
        return np.zeros((n_times, n_stations, 1), dtype=np.float32)


def get_nearest_veg(latitudes, longitudes, veg_data):
    return veg_data.sel(
        latitude=xr.DataArray(latitudes, dims="station"),
        longitude=xr.DataArray(longitudes, dims="station"),
        method="nearest",
    )["htcc"].values


def extract_terrain(latitudes, longitudes, oro_ds, r=2):
    h = oro_ds["h"].values
    lats_idx = np.abs(oro_ds.latitude.values[:, None] - latitudes).argmin(0)
    lons_idx = np.abs(oro_ds.longitude.values[:, None] - (longitudes % 360)).argmin(0)
    max_r, max_c = h.shape
    feats = []
    for r_idx, c_idx in zip(lats_idx, lons_idx):
        w = h[
            max(0, r_idx - r) : min(max_r, r_idx + r + 1),
            max(0, c_idx - r) : min(max_c, c_idx + r + 1),
        ]
        feats.append([h[r_idx, c_idx], h[r_idx, c_idx] - np.mean(w), np.std(w)])
    return np.array(feats, dtype=np.float32)


def prepare_raw_data(ds_in, data_veg, data_oro):
    print("\n--- Processing Data (Memory Optimized) ---", flush=True)
    ds = ds_in.copy(deep=False)
    if "vis" in ds:
        ds = ds.rename({"vis": "visibility"})
    ds = ds.rename({k: v for k, v in VAR_MAPPING.items() if k in ds})
    if "D2M" not in ds:
        ds["D2M"] = calculate_dewpoint_from_rh(ds["T2M"], ds["RH2M"])
    ds["DPD"] = ds["T2M"] - ds["D2M"]
    ds["INVERSION"] = ds["T_925"] - ds["T2M"]
    print(f"  Filtering Visibility > {MAX_VIS_THRESHOLD}...", flush=True)
    ds["visibility"] = ds["visibility"].where(ds["visibility"] <= MAX_VIS_THRESHOLD)
    if "lat" in ds:
        lats, lons = ds["lat"].values, ds["lon"].values
    elif "latitude" in ds:
        lats, lons = ds["latitude"].values, ds["longitude"].values
    else:
        raise AttributeError("Latitude/Longitude coordinates not found.")
    times, stations = pd.to_datetime(ds.time.values), ds.station_id.values
    print("  Building Dynamic Matrix...", flush=True)
    X_met = (
        ds[FINAL_FEATURE_ORDER]
        .to_array(dim="variable")
        .transpose("time", "station_id", "variable")
        .values.astype(np.float32)
    )
    del ds
    gc.collect()
    zenith = calculate_zenith_angle(lats, lons, times)
    X_dyn = np.concatenate([X_met, zenith], axis=-1)
    del X_met, zenith
    gc.collect()
    print("  Building Static Matrix...", flush=True)
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
    return X_dyn, X_stat, lats, lons, times, stations


def save_chunked(X_dyn_flat, X_stat_flat, fe_flat, y_flat, mask, meta, out_dir):
    times_all, stats_all, lats_all, lons_all = meta
    valid_idxs = np.where(mask)[0]
    n = len(valid_idxs)
    print(f"  Valid Samples: {n} ({n / len(mask):.1%})", flush=True)
    valid_times = pd.DatetimeIndex(times_all[valid_idxs])
    tr_m, val_m, test_m = get_monthly_split_mask_last_days(
        valid_times, GAP_HOURS, VAL_LAST_DAYS, TEST_LAST_DAYS
    )
    splits = {
        "train": valid_idxs[tr_m],
        "val": valid_idxs[val_m],
        "test": valid_idxs[test_m],
    }
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
            }
        ).to_csv(os.path.join(out_dir, f"meta_{tag}.csv"), index=False)
        fp = np.lib.format.open_memmap(
            os.path.join(out_dir, f"X_{tag}.npy"),
            mode="w+",
            dtype="float32",
            shape=(len(ix), sum(dims)),
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
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    print("Loading auxiliary data...", flush=True)
    data_veg = xr.open_dataset(VEG_FILE, engine="h5netcdf")
    data_oro = xr.open_dataset(ORO_FILE, engine="h5netcdf")
    ds_in = xr.open_dataset(INPUT_FILE, engine="h5netcdf")
    print("[Time Alignment] Tianji (BJT) -> UTC...", flush=True)
    ds_in = ds_in.assign_coords(time=ds_in.time - pd.Timedelta(hours=8))

    pm10_da = load_pm10_dataarray()
    pm25_da = load_pm25_dataarray()

    X_dyn, X_stat, lats, lons, times, stats = prepare_raw_data(ds_in, data_veg, data_oro)
    vis_key = "vis" if "vis" in ds_in else "visibility"

    # ========= pm10、pm2p5 作为动态变量末两维（×1e12 → µg/m³，与仅 pm10 单元格一致）=========
    pm10_ugm3 = cams_station_pm_to_ugm3_grid(pm10_da, times, stats)
    del pm10_da
    gc.collect()
    pm25_ugm3 = cams_station_pm_to_ugm3_grid(pm25_da, times, stats)
    del pm25_da
    gc.collect()
    X_dyn = np.concatenate(
        [X_dyn, pm10_ugm3[..., None], pm25_ugm3[..., None]], axis=-1
    )
    del pm10_ugm3, pm25_ugm3
    gc.collect()

    print(
        f"  Generating Windows ({WINDOW_SIZE}), chunk={TIME_CHUNK_WINS} wins (low-RAM)...",
        flush=True,
    )
    nt, ns, nv = X_dyn.shape
    n_wins = (nt - WINDOW_SIZE) // STEP_SIZE + 1
    n_total = n_wins * ns
    dyn_flat_dim = WINDOW_SIZE * nv

    y = ds_in[vis_key].values[WINDOW_SIZE - 1 :: STEP_SIZE].reshape(-1)
    m_t = np.repeat(times[WINDOW_SIZE - 1 :: STEP_SIZE], ns)
    m_s = np.tile(stats, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)

    sample_times = pd.DatetimeIndex(m_t)
    months = sample_times.month.values.astype(np.float32)
    hours = sample_times.hour.values.astype(np.float32)
    time_features = np.column_stack(
        [
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
        ]
    ).astype(np.float32)

    tmpd = os.path.join(OUTPUT_DATASET_DIR, "_tmp_build_mmap")
    os.makedirs(tmpd, exist_ok=True)
    path_x = os.path.join(tmpd, "X_dyn_flat.mmap")
    path_fe = os.path.join(tmpd, "fe_flat.mmap")
    mm_x = mm_fe = None
    fe_dim = None

    for w0 in range(0, n_wins, TIME_CHUNK_WINS):
        w1 = min(w0 + TIME_CHUNK_WINS, n_wins)
        t_start = w0 * STEP_SIZE
        len_sl = (w1 - 1 - w0) * STEP_SIZE + WINDOW_SIZE
        t_end = t_start + len_sl
        X_sl = np.ascontiguousarray(X_dyn[t_start:t_end, :, :], dtype=np.float32)
        raw = sliding_window_view(X_sl, WINDOW_SIZE, axis=0)
        n_chunk = w1 - w0
        idx_loc = (np.arange(n_chunk) * STEP_SIZE).astype(np.intp)
        X_wins = raw[idx_loc]
        X_wins = X_wins.transpose(0, 1, 3, 2)
        X_chunk = X_wins.reshape(-1, WINDOW_SIZE, nv)
        del X_sl, raw, X_wins

        fe_part = compute_fog_features(X_chunk, WINDOW_SIZE, nv)
        row0, row1 = w0 * ns, w1 * ns
        fe_part = np.concatenate([fe_part, time_features[row0:row1]], axis=1)
        dyn_flat_chunk = X_chunk.reshape(X_chunk.shape[0], -1)
        del X_chunk

        if fe_dim is None:
            fe_dim = fe_part.shape[1]
            if fe_dim != len(FE_FEATURE_NAMES):
                raise ValueError(f"FE dim mismatch: got {fe_dim}, expected {len(FE_FEATURE_NAMES)}.")
            mm_x = np.memmap(path_x, dtype=np.float32, mode="w+", shape=(n_total, dyn_flat_dim))
            mm_fe = np.memmap(path_fe, dtype=np.float32, mode="w+", shape=(n_total, fe_dim))
            print(f"  FE dim = {fe_dim} (32 fog + 16 VERA + 4 cyclical; pm10+pm2p5 in dyn)", flush=True)

        mm_x[row0:row1] = dyn_flat_chunk
        mm_fe[row0:row1] = fe_part
        mm_x.flush()
        mm_fe.flush()
        del fe_part, dyn_flat_chunk
        gc.collect()

    del time_features, X_dyn
    gc.collect()

    print("  Broadcasting Static...", flush=True)
    X_stat_flat = np.tile(X_stat, (n_wins, 1))

    print("  Saving (month-tail split)...", flush=True)
    mask = ~np.isnan(y) & (y >= 0) & (y <= MAX_VIS_THRESHOLD)
    save_chunked(mm_x, X_stat_flat, mm_fe, y, mask, (m_t, m_s, m_la, m_lo), OUTPUT_DATASET_DIR)

    del mm_x, mm_fe, X_stat_flat
    gc.collect()
    try:
        os.remove(path_x)
        os.remove(path_fe)
        os.rmdir(tmpd)
    except OSError:
        pass

    write_feature_layout(OUTPUT_DATASET_DIR)
    print(f"\nDone! -> {OUTPUT_DATASET_DIR}", flush=True)


if __name__ == "__main__":
    main()
