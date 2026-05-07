import os
import xarray as xr
import pandas as pd
import numpy as np
import pvlib
import warnings
import gc
import shutil
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.format import open_memmap

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_PATH = "/public/home/putianshu/vis_mlp"
FEATURE_DIR = os.path.join(BASE_PATH, 'station_data/station_data_merged')
OBS_PATH = os.path.join(BASE_PATH, 'CMA_visibility_2021_2023_GeoCoords_1.nc')
VEG_FILE = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE = "/public/home/putianshu/vis_cnn/data_orography.nc"
OUTPUT_DATASET_DIR = os.path.join(BASE_PATH, 'ml_dataset_pmst_v5_aligned_12h_pm10_pm25')
TEMP_DIR = os.path.join(OUTPUT_DATASET_DIR, 'temp_chunks')

VAR_CONFIG = {
    'T2M':    {'name': '2m_temperature', 'level': None},
    'D2M':    {'name': '2m_dewpoint_temperature', 'level': None},
    'PRECIP': {'name': 'total_precipitation', 'level': None},
    'MSLP':   {'name': 'mean_sea_level_pressure', 'level': None},
    'SW_RAD': {'name': 'mean_surface_downward_short_wave_radiation_flux', 'level': None},
    'U10':    {'name': '10m_u_component_of_wind', 'level': None},
    'V10':    {'name': '10m_v_component_of_wind', 'level': None},
    'CAPE':   {'name': 'convective_available_potential_energy', 'level': None},
    'LCC':    {'name': 'low_cloud_cover', 'level': None},
    'T_925':  {'name': 'temperature', 'level': 925},
    'T_1000': {'name': 'temperature', 'level': 1000},
    'RH_925': {'name': 'relative_humidity', 'level': 925},
    'RH_1000':{'name': 'relative_humidity', 'level': 1000},
    'U_925':  {'name': 'u_component_of_wind', 'level': 925},
    'V_925':  {'name': 'v_component_of_wind', 'level': 925},
    'W_925':  {'name': 'vertical_velocity', 'level': 925},
    'W_1000': {'name': 'vertical_velocity', 'level': 1000},
}

FINAL_FEATURE_ORDER = [
    'RH2M', 'T2M', 'PRECIP', 'MSLP', 'SW_RAD',
    'U10', 'WSPD10', 'V10', 'WDIR10', 'CAPE',
    'LCC', 'T_925', 'RH_925', 'U_925', 'WSPD925',
    'V_925', 'DP_1000', 'DP_925', 'Q_1000', 'Q_925', 'W_925',
    'W_1000', 'DPD', 'INVERSION'
]

YEARS_FOR_TRAIN_VAL = [2021, 2022, 2023]
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
GAP_HOURS = 24
WINDOW_SIZE = 12
STEP_SIZE = 1
UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])


def _stack_feature_rows(parts):
    """预分配一块大矩阵写入各站块，避免 np.concatenate(X_list) 与列表并存导致峰值翻倍。"""
    if not parts:
        return None
    total = sum(int(p.shape[0]) for p in parts)
    d = int(parts[0].shape[1])
    out = np.empty((total, d), dtype=np.float32)
    off = 0
    for p in parts:
        k = int(p.shape[0])
        out[off : off + k] = np.asarray(p, dtype=np.float32)
        off += k
    parts.clear()
    gc.collect()
    return out


def _stack_1d_rows(parts):
    if not parts:
        return None
    total = sum(int(p.shape[0]) for p in parts)
    out = np.empty((total,), dtype=np.float32)
    off = 0
    for p in parts:
        k = int(p.shape[0])
        out[off : off + k] = np.asarray(p, dtype=np.float32)
        off += k
    parts.clear()
    gc.collect()
    return out


# ================= 特征工程函数（对齐 s2 最新版本） =================
def compute_fog_features(X_dyn_window, window_size=12, dyn_vars=25):
    idx = {
        'rh2m': 0, 't2m': 1, 'precip': 2, 'sw_rad': 4,
        'wspd10': 6, 'cape': 9, 'lcc': 10, 't925': 11,
        'rh925': 12, 'dpd': 22, 'inversion': 23, 'zenith': 24
    }
    params = {
        'optimal_wspd': 3.5, 'wspd_sigma': 2.5, 'dpd_threshold': 2.0,
        'stability_scale': 2.0, 'lcc_threshold': 0.3, 'rad_threshold': 800.0
    }

    X_current = X_dyn_window[:, -1, :]
    rh2m, rh925 = X_current[:, idx['rh2m']], X_current[:, idx['rh925']]
    dpd, wspd = X_current[:, idx['dpd']], X_current[:, idx['wspd10']]
    inversion, sw_rad = X_current[:, idx['inversion']], X_current[:, idx['sw_rad']]
    lcc, zenith = X_current[:, idx['lcc']], X_current[:, idx['zenith']]
    t2m = X_current[:, idx['t2m']]

    t2m_c = t2m - 273.15 if np.mean(t2m) > 200 else t2m
    feats = []

    rh_norm = np.clip(rh2m / 100.0, 0, 1)
    dpd_weight = 1.0 / (1.0 + np.exp(dpd / params['dpd_threshold']))
    feats.append((rh_norm * dpd_weight).reshape(-1, 1))

    wind_fav = np.exp(-0.5 * ((wspd - params['optimal_wspd']) / params['wspd_sigma']) ** 2)
    feats.append(wind_fav.reshape(-1, 1))

    ri = inversion / (wspd ** 2 + 0.1)
    stability = np.tanh(ri / params['stability_scale'])
    feats.append(stability.reshape(-1, 1))

    is_night = (zenith > 90.0).astype(float)
    clear_sky = np.clip(1.0 - lcc / params['lcc_threshold'], 0, 1)
    rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0) / params['rad_threshold'], 0, 1)
    feats.append((is_night * clear_sky * rad_intensity).reshape(-1, 1))

    feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))

    fog_pot = (rh_norm * 0.4 + wind_fav * 0.25 + np.clip(stability, 0, 1) * 0.2 +
               (is_night * clear_sky * rad_intensity) * 0.15)
    feats.append(fog_pot.reshape(-1, 1))

    for var_idx in [idx['rh2m'], idx['t2m'], idx['wspd10']]:
        var_seq = X_dyn_window[:, :, var_idx]
        feats.append((var_seq[:, -1] - var_seq[:, -4]).reshape(-1, 1))
        feats.append((var_seq[:, -1] - var_seq[:, -7]).reshape(-1, 1))
        feats.append(np.std(var_seq, axis=1).reshape(-1, 1))
        feats.append((np.max(var_seq, axis=1) - np.min(var_seq, axis=1)).reshape(-1, 1))

    rh_seq = X_dyn_window[:, :, idx['rh2m']]
    rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
    feats.append(rh_accel.reshape(-1, 1))
    feats.append((rh2m * np.exp(-t2m_c / 10.0)).reshape(-1, 1))
    feats.append((is_night * (1 - lcc)).reshape(-1, 1))
    feats.append(((rh2m > 90) & (t2m_c < 10) & (wspd < 4)).astype(float).reshape(-1, 1))
    feats.append((rh2m / (lcc * 100 + 1)).reshape(-1, 1))
    feats.append(((rh2m / 100.0) ** 2).reshape(-1, 1))

    # 与 s2 对齐：Summer-regime 等 8 维（除起报时间外）
    u10 = X_current[:, 5]
    v10 = X_current[:, 7]
    u925 = X_current[:, 13]
    v925 = X_current[:, 15]
    precip = np.maximum(X_current[:, idx['precip']], 0.0)
    cape = np.maximum(X_current[:, idx['cape']], 0.0)
    q1000 = X_current[:, 18]
    q925 = X_current[:, 19]
    w925 = X_current[:, 20]
    w1000 = X_current[:, 21]
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    theta10 = np.arctan2(v10, u10)
    theta925 = np.arctan2(v925, u925)
    dir_turning = 0.5 * (1.0 - np.cos(theta925 - theta10))
    convective_wet = (1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))) * (1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5)))
    daytime_mixing = (1.0 / (1.0 + np.exp(-(np.maximum(sw_rad, 0.0) - 150.0) / 75.0))) * (1.0 / (1.0 + np.exp(-(wspd - 4.0) / 1.5))) * (1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2)))
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

    return np.concatenate(feats, axis=1).astype(np.float32)


# ================= 物理计算函数 =================
def calculate_rh_from_dewpoint(t2m, d2m):
    t_c = t2m - 273.15
    td_c = d2m - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip((e / es) * 100.0, 0, 100)


def calculate_dewpoint_from_rh(t, rh):
    t_c = t - 273.15
    rh_frac = np.maximum(rh / 100.0, 0.001)
    b, c = 17.67, 243.5
    gamma = np.log(rh_frac) + (b * t_c) / (c + t_c)
    return (c * gamma) / (b - gamma) + 273.15


def calculate_specific_humidity(t, rh, pressure_hpa):
    t_c = t - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    r = 0.622 * (es * (rh / 100.0)) / (pressure_hpa - es * (rh / 100.0))
    return r / (1 + r)


def calculate_wind_speed_dir(u, v):
    return np.sqrt(u ** 2 + v ** 2), (270 - np.degrees(np.arctan2(v, u))) % 360


def calculate_zenith_angle_vectorized(latitudes, longitudes, times):
    times_pd = pd.DatetimeIndex(times)
    if times_pd.tz is None:
        times_pd = times_pd.tz_localize('UTC')
    n_time, n_stat = len(times), len(latitudes)
    zenith_matrix = np.zeros((n_time, n_stat), dtype=np.float32)
    batch_size = 500
    for i in tqdm(range(0, n_stat, batch_size), desc="Zenith"):
        end = min(i + batch_size, n_stat)
        for j in range(i, end):
            solpos = pvlib.solarposition.get_solarposition(times_pd, latitudes[j], longitudes[j])
            zenith_matrix[:, j] = solpos['apparent_zenith'].values.astype(np.float32)
    return zenith_matrix[..., np.newaxis]


def extract_terrain_features_optimized(latitudes, longitudes, oro_ds, window_radius=2):
    h_data = oro_ds['h'].values.astype(np.float32)
    lat_indices = np.abs(oro_ds.latitude.values[:, None] - latitudes).argmin(axis=0)
    lon_indices = np.abs(oro_ds.longitude.values[:, None] - (longitudes % 360)).argmin(axis=0)
    max_lat, max_lon = h_data.shape
    lat_indices = np.clip(lat_indices, window_radius, max_lat - window_radius - 1)
    lon_indices = np.clip(lon_indices, window_radius, max_lon - window_radius - 1)
    n_samples = len(latitudes)
    features = np.zeros((n_samples, 3), dtype=np.float32)
    for i, (r, c) in enumerate(zip(lat_indices, lon_indices)):
        window = h_data[r - window_radius:r + window_radius + 1, c - window_radius:c + window_radius + 1]
        features[i] = [h_data[r, c], h_data[r, c] - np.mean(window), np.std(window)]
    return features


# ================= 数据处理核心 =================
def load_era5_single_year(year):
    datasets = []
    print(f"  -> Loading ERA5 raw files for {year}...")
    for var_short_name, conf in VAR_CONFIG.items():
        folder, level = conf['name'], conf['level']
        fname = f"{folder}_{year}_{level}hPa_merged.nc" if level else f"{folder}_{year}_merged.nc"
        fpath = os.path.join(FEATURE_DIR, fname)
        if os.path.exists(fpath):
            try:
                # 不使用 dask 分块，直接用 numpy 加载，避免 chunk manager 'dask' 依赖
                ds_var = xr.open_dataset(fpath, engine='h5netcdf')
                orig_name = list(ds_var.data_vars)[0]
                da = ds_var[orig_name].rename(var_short_name)
                for c in ['level', 'number', 'expver', 'metpy_crs']:
                    if c in da.coords:
                        da = da.drop_vars(c)
                datasets.append(da)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    if not datasets:
        return None
    ds = xr.merge(datasets)
    for alias in ['num_station', 'id', 'station']:
        if alias in ds.coords or alias in ds.data_vars:
            ds = ds.rename({alias: 'station_id'})
    return ds


def process_year_data_optimized(year, ds_obs_full, data_veg, data_oro, pm10_da_year, pm25_da_year=None):
    print(f"\n--- Processing Year: {year} ---")
    ds = load_era5_single_year(year)
    if ds is None:
        return None, None, None

    obs_year = ds_obs_full.sel(time=ds_obs_full.time.dt.year == year)
    ds['station_id'] = ds['station_id'].astype(np.int64)
    obs_year['station_id'] = obs_year['station_id'].astype(np.int64)

    common_s = np.intersect1d(ds.station_id.values, obs_year.station_id.values)
    common_t = np.intersect1d(ds.time.values, obs_year.time.values)
    print(f"  Intersect: {len(common_s)} stations, {len(common_t)} timesteps")
    if len(common_t) == 0:
        return None, None, None

    ds = ds.sel(station_id=common_s, time=common_t).load()
    obs_year = obs_year.sel(station_id=common_s, time=common_t).load()

    vis_data = np.where(obs_year['visibility'].values > 90000,
                        np.nan,
                        obs_year['visibility'].values.astype(np.float32))

    def get_coords(dataset):
        if 'lat' in dataset.coords:
            return dataset['lat'].values, dataset['lon'].values
        elif 'latitude' in dataset.coords:
            return dataset['latitude'].values, dataset['longitude'].values
        return None, None

    lats, lons = get_coords(obs_year)
    if lats is None:
        lats, lons = get_coords(ds)
        if lats is None:
            raise AttributeError(
                "Latitude/Longitude coordinates not found in obs or era5 dataset (checked 'lat' and 'latitude')."
            )

    times, station_ids = ds.time.values, ds.station_id.values
    del obs_year
    gc.collect()

    print("  Calculating derived variables...")
    ds['RH2M'] = calculate_rh_from_dewpoint(ds['T2M'], ds['D2M'])
    ds['WSPD10'], ds['WDIR10'] = calculate_wind_speed_dir(ds['U10'], ds['V10'])
    ds['WSPD925'], _ = calculate_wind_speed_dir(ds['U_925'], ds['V_925'])
    ds['DP_1000'] = calculate_dewpoint_from_rh(ds['T_1000'], ds['RH_1000'])
    ds['DP_925'] = calculate_dewpoint_from_rh(ds['T_925'], ds['RH_925'])
    ds['Q_1000'] = calculate_specific_humidity(ds['T_1000'], ds['RH_1000'], 1000.0)
    ds['Q_925'] = calculate_specific_humidity(ds['T_925'], ds['RH_925'], 925.0)
    ds['DPD'] = ds['T2M'] - ds['D2M']
    ds['INVERSION'] = ds['T_925'] - ds['T2M']

    print("  Preparing static features...")
    veg_raw = data_veg.sel(
        latitude=xr.DataArray(lats, dims="station"),
        longitude=xr.DataArray(lons, dims="station"),
        method="nearest"
    )['htcc'].values
    type_to_idx = {v: i for i, v in enumerate(UNIQUE_VEG_IDS)}
    feat_veg = np.array([type_to_idx.get(v, 0) for v in veg_raw], dtype=np.float32)[:, np.newaxis]

    feat_oro = extract_terrain_features_optimized(lats, lons, data_oro)
    X_stat_base = np.concatenate(
        [
            (lats[:, None] / 90.0).astype(np.float32),
            (lons[:, None] / 180.0).astype(np.float32),
            feat_oro,
            feat_veg,
        ],
        axis=1,
    )

    X_met = ds[FINAL_FEATURE_ORDER].to_array(dim='variable').transpose('time', 'station_id', 'variable').values.astype(
        np.float32
    )
    del ds
    gc.collect()

    zenith = calculate_zenith_angle_vectorized(lats, lons, times)
    X_dyn_base = np.concatenate([X_met, zenith], axis=-1)
    del X_met, zenith
    gc.collect()

    # ========= 将 pm10 作为动态变量追加到 dyn 的最后一维 =========
    # pm10 文件单位为 kg/m^3，这里转换为 ug/m^3 后留给训练脚本做 log1p。
    if pm10_da_year is not None:
        pm10_da = pm10_da_year

        # 确保维度顺序为 (time, station_id)
        if set(pm10_da.dims) >= {'time', 'station_id'}:
            pm10_da = pm10_da.transpose('time', 'station_id')
        else:
            raise ValueError(
                f'pm10_da_year dims must contain \'time\' and \'station_id\', got {pm10_da.dims}'
            )

        pm10_da = pm10_da.load()

        time_vals = pm10_da['time'].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_index = pd.DatetimeIndex(time_vals)
        else:
            # pm10 文件 time 是 float64 epoch 秒
            time_index = pd.to_datetime(time_vals, unit='s', origin='unix')

        ds_times = pd.DatetimeIndex(times)
        sid_index = pd.Index(pm10_da['station_id'].values)
        sids = station_ids.astype(pm10_da['station_id'].dtype)

        # 最近邻时间；站点按 station_id 精确匹配
        time_pos = time_index.get_indexer(ds_times, method='nearest')
        sid_pos = sid_index.get_indexer(sids)

        nt_ds, ns_ds = len(ds_times), len(sids)
        nt_pm10, ns_pm10 = pm10_da.shape

        pm10_grid = np.full((nt_ds, ns_ds), np.nan, dtype=np.float32)
        base = np.asarray(pm10_da.values).reshape(-1)
        linear_idx_grid = time_pos[:, None] * ns_pm10 + sid_pos[None, :]
        ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)

        if ok_mask.any():
            pm10_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)

        del base, linear_idx_grid, ok_mask
        gc.collect()

        pm10_grid = np.maximum(pm10_grid, 0.0)
        pm10_ugm3 = pm10_grid * 1e12
        del pm10_grid

        # 避免缺失导致 windows 被 mask 掉
        pm10_median = np.nanmedian(pm10_ugm3)
        if not np.isfinite(pm10_median):
            pm10_median = 0.0
        pm10_ugm3 = np.where(np.isfinite(pm10_ugm3), pm10_ugm3, pm10_median).astype(np.float32)

        X_dyn_base = np.concatenate([X_dyn_base, pm10_ugm3[..., None]], axis=-1)
        del pm10_ugm3, pm10_da
        gc.collect()

    # 若 pm10 缺失，仍需保证 dyn 维度一致（为训练脚本提供固定 dyn_vars=26）
    if pm10_da_year is None:
        pm10_ugm3 = np.zeros((len(times), len(station_ids)), dtype=np.float32)
        X_dyn_base = np.concatenate([X_dyn_base, pm10_ugm3[..., None]], axis=-1)

    # ========= PM2.5（pm2p5），与 pm10 同一对齐方式；×1e12 → µg/m³ =========
    if pm25_da_year is not None:
        pm25_da = pm25_da_year
        if set(pm25_da.dims) >= {"time", "station_id"}:
            pm25_da = pm25_da.transpose("time", "station_id")
        else:
            raise ValueError(
                f"pm25_da_year dims must contain 'time' and 'station_id', got {pm25_da.dims}"
            )
        pm25_da = pm25_da.load()
        time_vals = pm25_da["time"].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_index = pd.DatetimeIndex(time_vals)
        else:
            time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
        ds_times = pd.DatetimeIndex(times)
        sid_index = pd.Index(pm25_da["station_id"].values)
        sids = station_ids.astype(pm25_da["station_id"].dtype)
        time_pos = time_index.get_indexer(ds_times, method="nearest")
        sid_pos = sid_index.get_indexer(sids)
        nt_ds, ns_ds = len(ds_times), len(sids)
        _, ns_pm25 = pm25_da.shape
        pm25_grid = np.full((nt_ds, ns_ds), np.nan, dtype=np.float32)
        base = np.asarray(pm25_da.values, dtype=np.float32).reshape(-1)
        del pm25_da
        gc.collect()
        linear_idx_grid = time_pos[:, None] * ns_pm25 + sid_pos[None, :]
        ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
        if ok_mask.any():
            pm25_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
        del base, linear_idx_grid, ok_mask
        pm25_grid = np.maximum(pm25_grid, 0.0)
        pm25_ugm3 = pm25_grid * 1e12
        del pm25_grid
        pm25_median = np.nanmedian(pm25_ugm3)
        if not np.isfinite(pm25_median):
            pm25_median = 0.0
        pm25_ugm3 = np.where(np.isfinite(pm25_ugm3), pm25_ugm3, pm25_median).astype(np.float32)
        X_dyn_base = np.concatenate([X_dyn_base, pm25_ugm3[..., None]], axis=-1)
        del pm25_ugm3
        gc.collect()
    else:
        pm25_ugm3 = np.zeros((len(times), len(station_ids)), dtype=np.float32)
        X_dyn_base = np.concatenate([X_dyn_base, pm25_ugm3[..., None]], axis=-1)

    print("  Generating sliding windows (Station-wise)...")
    X_list, y_list = [], []
    meta_list = {k: [] for k in ['time', 'station_id', 'lat', 'lon']}

    n_time, n_stations, n_vars = X_dyn_base.shape
    n_dyn_vars = int(n_vars)
    target_indices = np.arange(WINDOW_SIZE - 1, n_time, STEP_SIZE)
    valid_times = times[target_indices]

    for s_idx in tqdm(range(n_stations), desc="Station Loop"):
        st_data_dyn = X_dyn_base[:, s_idx, :]
        st_vis = vis_data[:, s_idx]

        try:
            windows = sliding_window_view(st_data_dyn, window_shape=WINDOW_SIZE, axis=0)
            windows = windows.transpose(0, 2, 1)
        except ValueError:
            continue

        windows = windows[::STEP_SIZE]
        current_targets = st_vis[target_indices]
        mask_y = ~np.isnan(current_targets) & (current_targets >= 0)
        if mask_y.sum() == 0:
            continue

        windows_subset = windows[mask_y]
        mask_x = ~np.isnan(windows_subset).any(axis=(1, 2))
        if mask_x.sum() == 0:
            continue

        X_window_final = windows_subset[mask_x]
        fe_features = compute_fog_features(X_window_final, window_size=WINDOW_SIZE, dyn_vars=n_dyn_vars)

        # ===== 关键校验：S1 必须同步 S2 的特征工程维度拓展 =====
        # compute_fog_features: 32 维（含 Summer-regime 8 维）
        if fe_features.shape[1] != 32:
            raise ValueError(
                f"compute_fog_features dim mismatch: got {fe_features.shape[1]}, expected 32. "
                f"This indicates S1 feature engineering is NOT synced with S2."
            )

        # 周期时间编码（观测时刻，与 s2 一致，不含起报时间）
        valid_indices_rel = np.where(mask_y)[0][mask_x]
        sample_times = pd.DatetimeIndex(valid_times[valid_indices_rel])
        months = sample_times.month.values.astype(np.float32)
        hours = sample_times.hour.values.astype(np.float32)
        time_feat = np.column_stack([
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
        ]).astype(np.float32)
        fe_with_time = np.concatenate([fe_features, time_feat], axis=1)
        if fe_with_time.shape[1] != 36:
            raise ValueError(
                f"fe_with_time dim mismatch: got {fe_with_time.shape[1]}, expected 36 (32+4)."
            )

        X_dyn_flat = X_window_final.reshape(X_window_final.shape[0], -1)
        y_final = current_targets[mask_y][mask_x]

        st_stat_expanded = np.repeat(X_stat_base[s_idx:s_idx + 1, :], len(y_final), axis=0)
        X_final = np.concatenate([X_dyn_flat, st_stat_expanded, fe_with_time], axis=1)

        X_list.append(X_final)
        y_list.append(y_final)

        meta_list['time'].extend(valid_times[valid_indices_rel])
        meta_list['station_id'].extend([station_ids[s_idx]] * len(y_final))
        meta_list['lat'].extend([lats[s_idx]] * len(y_final))
        meta_list['lon'].extend([lons[s_idx]] * len(y_final))

        del (
            windows,
            windows_subset,
            X_window_final,
            fe_features,
            time_feat,
            fe_with_time,
            X_dyn_flat,
            st_stat_expanded,
            X_final,
            y_final,
            st_data_dyn,
            st_vis,
            mask_y,
            mask_x,
        )
        if (s_idx + 1) % 256 == 0:
            gc.collect()

    del X_dyn_base, vis_data
    gc.collect()
    if not X_list:
        return None, None, None

    X_yr = _stack_feature_rows(X_list)
    y_yr = _stack_1d_rows(y_list)
    meta_df = pd.DataFrame(
        {
            'time': meta_list['time'],
            'station_id': meta_list['station_id'],
            'lat': meta_list['lat'],
            'lon': meta_list['lon'],
        }
    )

    # pm10 已在生成 X_dyn_base 时作为 dyn 的最后一维追加（不再作为 extra 拼接）

    # ===== 关键校验：最终列布局必须满足训练脚本的拆分假设 =====
    expected_total_dim = WINDOW_SIZE * n_dyn_vars + 5 + 1 + 36
    if X_yr.shape[1] != expected_total_dim:
        raise ValueError(
            f"Final X dim mismatch: got {X_yr.shape[1]}, expected {expected_total_dim}. "
            f"(dyn={n_dyn_vars} per step, FE=36)"
        )

    return X_yr, y_yr, meta_df


def get_monthly_split_mask(meta_df):
    times = pd.to_datetime(meta_df['time'])
    n_samples = len(meta_df)
    train_mask, val_mask = np.zeros(n_samples, dtype=bool), np.zeros(n_samples, dtype=bool)

    for month_period in times.dt.to_period('M').unique():
        start, end = month_period.start_time, month_period.end_time
        split_time = start + pd.Timedelta(seconds=(end - start).total_seconds() * TRAIN_RATIO)

        mask_month = (times >= start) & (times <= end)
        train_mask |= mask_month & (times < split_time)
        val_mask |= mask_month & (times >= split_time + pd.Timedelta(hours=GAP_HOURS))

    return train_mask, val_mask


def merge_temp_files(temp_files, output_path_x, output_path_y):
    if not temp_files:
        return
    print(f"Merging {len(temp_files)} chunks into {output_path_x} ...")

    total_samples = 0
    feature_shape, dtype_x, dtype_y = None, None, None
    for f_x, f_y in temp_files:
        arr_x = np.load(f_x, mmap_mode='r')
        arr_y = np.load(f_y, mmap_mode='r')
        if feature_shape is None:
            feature_shape, dtype_x, dtype_y = arr_x.shape[1:], arr_x.dtype, arr_y.dtype
        total_samples += arr_x.shape[0]

    mmap_x = open_memmap(output_path_x, mode='w+', dtype=dtype_x, shape=(total_samples, *feature_shape))
    mmap_y = open_memmap(output_path_y, mode='w+', dtype=dtype_y, shape=(total_samples,))

    idx = 0
    for f_x, f_y in tqdm(temp_files, desc="Merging"):
        d_x, d_y = np.load(f_x), np.load(f_y)
        n = d_x.shape[0]
        mmap_x[idx:idx + n], mmap_y[idx:idx + n] = d_x, d_y
        idx += n
        os.remove(f_x)
        os.remove(f_y)
    print("Merge complete.")


def main():
    if not os.path.exists(OUTPUT_DATASET_DIR):
        os.makedirs(OUTPUT_DATASET_DIR)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print("Loading auxiliary data...")
    data_veg = xr.open_dataset(VEG_FILE, engine='h5netcdf')
    data_oro = xr.open_dataset(ORO_FILE, engine='h5netcdf')
    ds_obs = xr.open_dataset(OBS_PATH, engine='h5netcdf')

    if 'station_id' not in ds_obs.coords and 'station_id' not in ds_obs.data_vars:
        for alias in ['id', 'station']:
            if alias in ds_obs:
                ds_obs = ds_obs.rename({alias: 'station_id'})

    print("[Time Alignment] Converting Observation time from BJT to UTC...")
    ds_obs = ds_obs.assign_coords(time=ds_obs.time - pd.Timedelta(hours=8))

    temp_train, temp_val = [], []
    meta_train, meta_val = [], []

    print("\n================ Processing TRAIN/VAL Years ================")
    for year in YEARS_FOR_TRAIN_VAL:
        pm10_path_year = os.path.join(BASE_PATH, "pm10_station", f"pm10_station_s1_{year}.nc")
        if os.path.exists(pm10_path_year):
            pm10_ds_year = xr.open_dataset(pm10_path_year, engine='h5netcdf')
            pm10_da_year = pm10_ds_year["pm10"].load()
            pm10_ds_year.close()
            del pm10_ds_year
        else:
            print(f"[WARN] pm10 file not found for year {year}: {pm10_path_year}, pm10 will be NaN.", flush=True)
            pm10_da_year = None

        pm25_path_year = os.path.join(BASE_PATH, "pm2.5_station", f"pm2p5_station_s1_{year}.nc")
        if os.path.exists(pm25_path_year):
            pm25_ds_year = xr.open_dataset(pm25_path_year, engine='h5netcdf')
            if "pm2p5" in pm25_ds_year:
                pm25_da_year = pm25_ds_year["pm2p5"].load()
            else:
                pm25_da_year = pm25_ds_year[list(pm25_ds_year.data_vars)[0]].load()
            pm25_ds_year.close()
            del pm25_ds_year
        else:
            print(f"[WARN] pm2.5 file not found for year {year}: {pm25_path_year}, pm2.5 channel will be zeros.", flush=True)
            pm25_da_year = None

        gc.collect()
        X, y, meta = process_year_data_optimized(year, ds_obs, data_veg, data_oro, pm10_da_year, pm25_da_year)
        del pm10_da_year, pm25_da_year
        gc.collect()
        if X is not None:
            print(f"  Splitting Year {year}...")
            tr_m, val_m = get_monthly_split_mask(meta)

            for mask, f_list, m_list, tag in [
                (tr_m, temp_train, meta_train, 'train'),
                (val_m, temp_val, meta_val, 'val'),
            ]:
                if mask.sum() > 0:
                    p_x = os.path.join(TEMP_DIR, f'X_{tag}_{year}.npy')
                    p_y = os.path.join(TEMP_DIR, f'y_{tag}_{year}.npy')
                    np.save(p_x, X[mask])
                    np.save(p_y, y[mask])
                    f_list.append((p_x, p_y))
                    m_list.append(meta[mask])
            print(f"  -> Saved chunks for {year}.")
        del X, y, meta
        gc.collect()

    print("\n================ Merging Final Datasets ================")
    merge_temp_files(
        temp_train,
        os.path.join(OUTPUT_DATASET_DIR, 'X_train.npy'),
        os.path.join(OUTPUT_DATASET_DIR, 'y_train.npy'),
    )
    if meta_train:
        pd.concat(meta_train).to_csv(os.path.join(OUTPUT_DATASET_DIR, 'meta_train.csv'), index=False)

    merge_temp_files(
        temp_val,
        os.path.join(OUTPUT_DATASET_DIR, 'X_val.npy'),
        os.path.join(OUTPUT_DATASET_DIR, 'y_val.npy'),
    )
    if meta_val:
        pd.concat(meta_val).to_csv(os.path.join(OUTPUT_DATASET_DIR, 'meta_val.csv'), index=False)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()

