from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingTorchModule:
        Tensor = object

        def __getattr__(self, name):
            raise ImportError(
                "PyTorch is required for model construction/inference, but it could "
                f"not be imported: {_TORCH_IMPORT_ERROR}"
            ) from _TORCH_IMPORT_ERROR

    class _MissingNNModule:
        Module = object

        def __getattr__(self, name):
            raise ImportError(
                "PyTorch is required for model construction/inference, but it could "
                f"not be imported: {_TORCH_IMPORT_ERROR}"
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorchModule()
    nn = _MissingNNModule()
    F = _MissingTorchModule()


WINDOW_SIZE = 12
LOCAL_TIME_OFFSET_HOURS = 8
MAX_VISIBILITY_M = 30000.0


DYNAMIC_FEATURE_ORDER = [
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
    "ZENITH_PROXY",
]


WEATHER_ALIASES = {
    "RH2M": ("RH2M", "rh2m"),
    "T2M": ("T2M", "TMP2m", "t2m"),
    "PRECIP": ("PRECIP", "PRATEsfc", "prer", "precip"),
    "MSLP": ("MSLP", "slp", "SLP"),
    "SW_RAD": ("SW_RAD", "DSWRFsfc", "ssrd", "zenith_sw_rad"),
    "U10": ("U10", "UGRD10m", "u10"),
    "V10": ("V10", "VGRD10m", "v10"),
    "WSPD10": ("WSPD10", "wind_speed", "wspd10"),
    "WDIR10": ("WDIR10", "wd10m", "wdir10"),
    "CAPE": ("CAPE", "cape"),
    "LCC": ("LCC", "cldl", "lcc"),
    "T_925": ("T_925", "t925"),
    "RH_925": ("RH_925", "rh925"),
    "U_925": ("U_925", "u925"),
    "V_925": ("V_925", "v925"),
    "WSPD925": ("WSPD925", "wind_speed_925", "wspd925"),
    "DP_1000": ("DP_1000", "dp1000"),
    "DP_925": ("DP_925", "dp925"),
    "Q_1000": ("Q_1000", "q1000"),
    "Q_925": ("Q_925", "q925"),
    "W_925": ("W_925", "omg925"),
    "W_1000": ("W_1000", "omg1000"),
    "ZENITH_PROXY": ("ZENITH_PROXY", "zenith", "apparent_zenith"),
}


LOG_DYNAMIC_FEATURES = ("PRECIP", "SW_RAD", "CAPE")
FOG_FEATURE_DIM = 32
TIME_FEATURE_DIM = 4
EXTRA_FEATURE_DIM = FOG_FEATURE_DIM + TIME_FEATURE_DIM


def get_dynamic_log_indices(
    feature_order: Sequence[str] = DYNAMIC_FEATURE_ORDER,
) -> List[int]:
    names = set(LOG_DYNAMIC_FEATURES)
    return [i for i, name in enumerate(feature_order) if name in names]


def visibility_to_classes(visibility_m: np.ndarray) -> np.ndarray:
    y = np.asarray(visibility_m, dtype=np.float32)
    cls = np.zeros(y.shape, dtype=np.int64)
    cls[y >= 500.0] = 1
    cls[y >= 1000.0] = 2
    return cls


def maybe_convert_visibility_to_meters(values: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) <= 100.0:
        return arr * 1000.0, "km_to_m_auto"
    return arr, "m"


def calculate_dewpoint_from_rh(t_k: np.ndarray, rh: np.ndarray) -> np.ndarray:
    t_c = np.asarray(t_k, dtype=np.float32) - 273.15
    rh_frac = np.clip(np.asarray(rh, dtype=np.float32) / 100.0, 0.001, 1.0)
    b, c = 17.67, 243.5
    gamma = np.log(rh_frac) + (b * t_c) / (c + t_c)
    return (c * gamma) / (b - gamma) + 273.15


def _time_index_with_local_offset(
    times: Sequence,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    return idx + pd.Timedelta(hours=float(local_time_offset_hours))


def zenith_proxy_from_time(
    times: Sequence,
    n_stations: int,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
) -> np.ndarray:
    local_times = _time_index_with_local_offset(times, local_time_offset_hours)
    hours = local_times.hour.values + local_times.minute.values / 60.0
    # A station-location-free solar proxy: about 10 deg near local noon and
    # 170 deg near local midnight, enough for day/night fog physics terms.
    zenith = 90.0 - 80.0 * np.cos(2.0 * np.pi * (hours - 12.0) / 24.0)
    return np.repeat(zenith[:, None], int(n_stations), axis=1).astype(np.float32)


def time_cyclical_features(
    sample_times: Sequence,
    n_stations: int,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
) -> np.ndarray:
    local_times = _time_index_with_local_offset(sample_times, local_time_offset_hours)
    months = local_times.month.values.astype(np.float32)
    hours = (
        local_times.hour.values.astype(np.float32)
        + local_times.minute.values.astype(np.float32) / 60.0
    )
    per_time = np.column_stack(
        [
            np.sin(2.0 * np.pi * months / 12.0),
            np.cos(2.0 * np.pi * months / 12.0),
            np.sin(2.0 * np.pi * hours / 24.0),
            np.cos(2.0 * np.pi * hours / 24.0),
        ]
    ).astype(np.float32)
    return np.repeat(per_time, int(n_stations), axis=0)


XarrayLike = Union[xr.Dataset, xr.DataArray]


def _station_dim_name(obj: XarrayLike) -> str:
    for name in ("station", "station_id", "num_station", "station_name"):
        if name in obj.dims:
            return name
    raise ValueError(f"Cannot find station dimension in dims={obj.dims}")


def _get_times_and_stations(obj: XarrayLike) -> Tuple[np.ndarray, np.ndarray]:
    station_dim = _station_dim_name(obj)
    if "time" not in obj.coords:
        raise ValueError("Input must have a time coordinate")
    times = pd.to_datetime(obj["time"].values).to_numpy()
    if station_dim in obj.coords:
        stations = obj[station_dim].values
    else:
        stations = np.arange(obj.sizes[station_dim])
    return times, stations


def _dataarray_variable_names(da: xr.DataArray) -> List[str]:
    if "variable" not in da.dims and "variable" not in da.coords:
        return []
    return [str(v) for v in da["variable"].values]


def _select_data_var(
    source: XarrayLike,
    names: Iterable[str],
) -> Optional[xr.DataArray]:
    if isinstance(source, xr.Dataset):
        for name in names:
            if name in source.data_vars:
                return source[name]
        return None

    if "variable" in source.dims or "variable" in source.coords:
        present = set(_dataarray_variable_names(source))
        for name in names:
            if name in present:
                return source.sel(variable=name)
    return None


def _as_time_station(da: xr.DataArray) -> np.ndarray:
    station_dim = _station_dim_name(da)
    arr = da.transpose("time", station_dim).values
    return np.asarray(arr, dtype=np.float32)


def _get_var_array(
    source: XarrayLike,
    canonical_name: str,
    cache: MutableMapping[str, np.ndarray],
) -> Optional[np.ndarray]:
    if canonical_name in cache:
        return cache[canonical_name]
    da = _select_data_var(source, WEATHER_ALIASES.get(canonical_name, (canonical_name,)))
    if da is None:
        return None
    arr = _as_time_station(da)
    cache[canonical_name] = arr
    return arr


def _computed_weather_array(
    source: XarrayLike,
    canonical_name: str,
    cache: MutableMapping[str, np.ndarray],
    times: np.ndarray,
    n_stations: int,
    local_time_offset_hours: float,
    use_source_zenith: bool,
) -> np.ndarray:
    direct = _get_var_array(source, canonical_name, cache)
    if direct is not None and (canonical_name != "ZENITH_PROXY" or use_source_zenith):
        return direct

    if canonical_name == "WSPD10":
        u10 = _computed_weather_array(
            source, "U10", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        v10 = _computed_weather_array(
            source, "V10", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        return np.sqrt(u10 * u10 + v10 * v10).astype(np.float32)

    if canonical_name == "WDIR10":
        u10 = _computed_weather_array(
            source, "U10", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        v10 = _computed_weather_array(
            source, "V10", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        return ((270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0).astype(np.float32)

    if canonical_name == "WSPD925":
        u925 = _computed_weather_array(
            source, "U_925", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        v925 = _computed_weather_array(
            source, "V_925", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        return np.sqrt(u925 * u925 + v925 * v925).astype(np.float32)

    if canonical_name == "DPD":
        t2m = _computed_weather_array(
            source, "T2M", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        rh2m = _computed_weather_array(
            source, "RH2M", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        d2m = calculate_dewpoint_from_rh(t2m, rh2m)
        return (t2m - d2m).astype(np.float32)

    if canonical_name == "INVERSION":
        t925 = _computed_weather_array(
            source, "T_925", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        t2m = _computed_weather_array(
            source, "T2M", cache, times, n_stations, local_time_offset_hours, use_source_zenith
        )
        return (t925 - t2m).astype(np.float32)

    if canonical_name == "ZENITH_PROXY":
        return zenith_proxy_from_time(times, n_stations, local_time_offset_hours)

    raise KeyError(
        f"Missing required variable for {canonical_name}. "
        f"Accepted aliases: {WEATHER_ALIASES.get(canonical_name, (canonical_name,))}"
    )


def extract_dynamic_cube(
    source: XarrayLike,
    feature_order: Sequence[str] = DYNAMIC_FEATURE_ORDER,
    fill_values: Optional[Mapping[str, float]] = None,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
    use_source_zenith: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, stations = _get_times_and_stations(source)
    n_stations = len(stations)
    cache: Dict[str, np.ndarray] = {}
    arrays = []
    for name in feature_order:
        arr = _computed_weather_array(
            source,
            name,
            cache,
            times,
            n_stations,
            local_time_offset_hours,
            use_source_zenith,
        )
        arrays.append(np.asarray(arr, dtype=np.float32))
    cube = np.stack(arrays, axis=-1).astype(np.float32)
    if fill_values is not None:
        cube = fill_dynamic_cube(cube, fill_values, feature_order)
    return cube, times, stations


def compute_fill_values(
    cube: np.ndarray,
    feature_order: Sequence[str] = DYNAMIC_FEATURE_ORDER,
) -> Dict[str, float]:
    fill_values: Dict[str, float] = {}
    for i, name in enumerate(feature_order):
        vals = cube[..., i]
        finite = vals[np.isfinite(vals)]
        fill_values[name] = float(np.median(finite)) if finite.size else 0.0
    return fill_values


def fill_dynamic_cube(
    cube: np.ndarray,
    fill_values: Mapping[str, float],
    feature_order: Sequence[str] = DYNAMIC_FEATURE_ORDER,
) -> np.ndarray:
    out = np.asarray(cube, dtype=np.float32).copy()
    for i, name in enumerate(feature_order):
        fill = np.float32(fill_values.get(name, 0.0))
        vals = out[..., i]
        out[..., i] = np.where(np.isfinite(vals), vals, fill)
    return out


def compute_fog_features(
    x_dyn_window: np.ndarray,
    feature_order: Sequence[str] = DYNAMIC_FEATURE_ORDER,
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(feature_order)}
    x_current = np.asarray(x_dyn_window[:, -1, :], dtype=np.float32)

    rh2m = x_current[:, idx["RH2M"]]
    rh925 = x_current[:, idx["RH_925"]]
    dpd = x_current[:, idx["DPD"]]
    wspd = np.maximum(x_current[:, idx["WSPD10"]], 0.0)
    inversion = x_current[:, idx["INVERSION"]]
    sw_rad = x_current[:, idx["SW_RAD"]]
    lcc = x_current[:, idx["LCC"]]
    zenith = x_current[:, idx["ZENITH_PROXY"]]
    t2m = x_current[:, idx["T2M"]]
    t2m_c = t2m - 273.15 if np.nanmean(t2m) > 200 else t2m

    optimal_wspd = 3.5
    wspd_sigma = 2.5
    dpd_threshold = 2.0
    stability_scale = 2.0
    lcc_threshold = 0.3
    rad_threshold = 800.0

    feats = []
    rh_norm = np.clip(rh2m / 100.0, 0.0, 1.0)
    dpd_weight = 1.0 / (1.0 + np.exp(np.clip(dpd / dpd_threshold, -30.0, 30.0)))
    wind_fav = np.exp(-0.5 * ((wspd - optimal_wspd) / wspd_sigma) ** 2)
    ri = inversion / (wspd**2 + 0.1)
    stability = np.tanh(ri / stability_scale)
    is_night = (zenith > 90.0).astype(np.float32)
    clear_sky = np.clip(1.0 - lcc / lcc_threshold, 0.0, 1.0)
    rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0.0) / rad_threshold, 0.0, 1.0)

    feats.append((rh_norm * dpd_weight).reshape(-1, 1))
    feats.append(wind_fav.reshape(-1, 1))
    feats.append(stability.reshape(-1, 1))
    feats.append((is_night * clear_sky * rad_intensity).reshape(-1, 1))
    feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))
    fog_pot = (
        rh_norm * 0.4
        + wind_fav * 0.25
        + np.clip(stability, 0.0, 1.0) * 0.2
        + (is_night * clear_sky * rad_intensity) * 0.15
    )
    feats.append(fog_pot.reshape(-1, 1))

    for var_name in ("RH2M", "T2M", "WSPD10"):
        seq = x_dyn_window[:, :, idx[var_name]]
        feats.append((seq[:, -1] - seq[:, -4]).reshape(-1, 1))
        feats.append((seq[:, -1] - seq[:, -7]).reshape(-1, 1))
        feats.append(np.std(seq, axis=1).reshape(-1, 1))
        feats.append((np.max(seq, axis=1) - np.min(seq, axis=1)).reshape(-1, 1))

    rh_seq = x_dyn_window[:, :, idx["RH2M"]]
    rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
    feats.append(rh_accel.reshape(-1, 1))
    feats.append((rh2m * np.exp(np.clip(-t2m_c / 10.0, -20.0, 20.0))).reshape(-1, 1))
    feats.append((is_night * (1.0 - lcc)).reshape(-1, 1))
    feats.append(((rh2m > 90.0) & (t2m_c < 10.0) & (wspd < 4.0)).astype(np.float32).reshape(-1, 1))
    feats.append((rh2m / (lcc * 100.0 + 1.0)).reshape(-1, 1))
    feats.append(((rh2m / 100.0) ** 2).reshape(-1, 1))

    u10 = x_current[:, idx["U10"]]
    v10 = x_current[:, idx["V10"]]
    u925 = x_current[:, idx["U_925"]]
    v925 = x_current[:, idx["V_925"]]
    precip = np.maximum(x_current[:, idx["PRECIP"]], 0.0)
    cape = np.maximum(x_current[:, idx["CAPE"]], 0.0)
    q1000 = x_current[:, idx["Q_1000"]]
    q925 = x_current[:, idx["Q_925"]]
    w925 = x_current[:, idx["W_925"]]
    w1000 = x_current[:, idx["W_1000"]]

    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    theta10 = np.arctan2(v10, u10)
    theta925 = np.arctan2(v925, u925)
    dir_turning = 0.5 * (1.0 - np.cos(theta925 - theta10))
    convective_wet = (
        1.0 / (1.0 + np.exp(np.clip(-(np.log1p(cape) - np.log(200.0)) * 1.6, -30.0, 30.0)))
    ) * (
        1.0 / (1.0 + np.exp(np.clip(-(np.log1p(precip) - np.log(0.1)) * 2.5, -30.0, 30.0)))
    )
    daytime_mixing = (
        1.0 / (1.0 + np.exp(np.clip(-(np.maximum(sw_rad, 0.0) - 150.0) / 75.0, -30.0, 30.0)))
    ) * (
        1.0 / (1.0 + np.exp(np.clip(-(wspd - 4.0) / 1.5, -30.0, 30.0)))
    ) * (
        1.0 / (1.0 + np.exp(np.clip(-(-inversion + 0.5) / 1.2, -30.0, 30.0)))
    )
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

    out = np.concatenate(feats, axis=1).astype(np.float32)
    if out.shape[1] != FOG_FEATURE_DIM:
        raise RuntimeError(f"Fog feature dim mismatch: {out.shape[1]} != {FOG_FEATURE_DIM}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def raw_rows_to_continuous_matrix(
    rows: np.ndarray,
    window_size: int,
    dyn_vars_count: int,
    log_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    split_dyn = int(window_size) * int(dyn_vars_count)
    rows = np.asarray(rows, dtype=np.float32)
    dyn = rows[:, :split_dyn].astype(np.float32, copy=True)
    extra = rows[:, split_dyn + 1 :].astype(np.float32, copy=True)
    if log_indices is None:
        log_indices = get_dynamic_log_indices()
    if log_indices:
        mask = np.zeros(split_dyn, dtype=bool)
        for t in range(int(window_size)):
            for idx in log_indices:
                mask[t * int(dyn_vars_count) + int(idx)] = True
        dyn[:, mask] = np.log1p(np.maximum(dyn[:, mask], 0.0))
    return np.concatenate([dyn, extra], axis=1).astype(np.float32)


def transform_airport_rows(
    raw_rows: np.ndarray,
    scaler,
    window_size: int,
    dyn_vars_count: int,
    clip_value: float = 10.0,
) -> np.ndarray:
    raw_rows = np.asarray(raw_rows, dtype=np.float32)
    split_dyn = int(window_size) * int(dyn_vars_count)
    cont = raw_rows_to_continuous_matrix(raw_rows, window_size, dyn_vars_count)
    if scaler is not None:
        cont = (cont - scaler.center_) / (scaler.scale_ + 1e-6)
    cont = np.clip(cont, -clip_value, clip_value)
    dyn_scaled = cont[:, :split_dyn]
    extra_scaled = cont[:, split_dyn:]
    station_idx = raw_rows[:, split_dyn : split_dyn + 1]
    final = np.concatenate([dyn_scaled, station_idx, extra_scaled], axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=clip_value, neginf=-clip_value).astype(np.float32)


def transform_pmst_rows(
    raw_rows: np.ndarray,
    scaler,
    window_size: int,
    dyn_vars_count: int,
    clip_value: float = 10.0,
) -> np.ndarray:
    """Transform original PMST layout: dyn window + 5 static + veg + FE."""
    raw_rows = np.asarray(raw_rows, dtype=np.float32)
    split_dyn = int(window_size) * int(dyn_vars_count)
    continuous = raw_rows[:, : split_dyn + 5].astype(np.float32, copy=True)
    log_indices = get_dynamic_log_indices()
    if log_indices:
        mask = np.zeros(split_dyn, dtype=bool)
        for t in range(int(window_size)):
            for idx in log_indices:
                mask[t * int(dyn_vars_count) + int(idx)] = True
        continuous[:, :split_dyn] = np.where(
            mask,
            np.log1p(np.maximum(continuous[:, :split_dyn], 0.0)),
            continuous[:, :split_dyn],
        )
    if scaler is not None:
        continuous = (continuous - scaler.center_) / (scaler.scale_ + 1e-6)
    continuous = np.clip(continuous, -clip_value, clip_value)
    veg = raw_rows[:, split_dyn + 5 : split_dyn + 6]
    extra = np.clip(raw_rows[:, split_dyn + 6 :], -clip_value, clip_value)
    final = np.concatenate([continuous, veg, extra], axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=clip_value, neginf=-clip_value).astype(np.float32)


def build_inference_matrix(
    source: XarrayLike,
    station_names: Sequence,
    station_to_idx: Mapping[str, int],
    scaler,
    fill_values: Mapping[str, float],
    window_size: int = WINDOW_SIZE,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
    use_source_zenith: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    cube, times, source_stations = extract_dynamic_cube(
        source,
        fill_values=fill_values,
        local_time_offset_hours=local_time_offset_hours,
        use_source_zenith=use_source_zenith,
    )
    nt, ns, dyn_vars = cube.shape
    if station_names is None or len(station_names) != ns:
        station_names = [str(v) for v in source_stations]
    else:
        station_names = [str(v) for v in station_names]

    station_indices = np.zeros(ns, dtype=np.float32)
    unknown = []
    for i, name in enumerate(station_names):
        if name in station_to_idx:
            station_indices[i] = float(station_to_idx[name])
        else:
            unknown.append(name)
            station_indices[i] = 0.0

    split_dyn = int(window_size) * dyn_vars
    total_dim = split_dyn + 1 + EXTRA_FEATURE_DIM
    raw_rows = np.empty((nt * ns, total_dim), dtype=np.float32)
    write0 = 0
    for t in range(nt):
        idxs = np.arange(t - int(window_size) + 1, t + 1)
        idxs = np.clip(idxs, 0, nt - 1)
        wins = cube[idxs, :, :].transpose(1, 0, 2).astype(np.float32)
        dyn_flat = wins.reshape(ns, split_dyn)
        fog_features = compute_fog_features(wins)
        cyc = time_cyclical_features([times[t]], ns, local_time_offset_hours)
        extra = np.concatenate([fog_features, cyc], axis=1)
        write1 = write0 + ns
        raw_rows[write0:write1, :split_dyn] = dyn_flat
        raw_rows[write0:write1, split_dyn] = station_indices
        raw_rows[write0:write1, split_dyn + 1 :] = extra
        write0 = write1

    x = transform_airport_rows(raw_rows, scaler, window_size, dyn_vars)
    coords = {
        "time": np.asarray(times),
        "station_name": np.asarray(station_names, dtype=object),
    }
    return x, coords, unknown


def build_pmst_inference_matrix(
    source: XarrayLike,
    station_names: Sequence,
    station_static: Mapping[str, Mapping],
    scaler,
    fill_values: Mapping[str, float],
    window_size: int = WINDOW_SIZE,
    local_time_offset_hours: float = LOCAL_TIME_OFFSET_HOURS,
    use_source_zenith: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    cube, times, source_stations = extract_dynamic_cube(
        source,
        fill_values=fill_values,
        local_time_offset_hours=local_time_offset_hours,
        use_source_zenith=use_source_zenith,
    )
    nt, ns, dyn_vars = cube.shape
    if station_names is None or len(station_names) != ns:
        station_names = [str(v) for v in source_stations]
    else:
        station_names = [str(v) for v in station_names]

    static_cont = np.zeros((ns, 5), dtype=np.float32)
    veg_idx = np.zeros(ns, dtype=np.float32)
    unknown = []
    for i, name in enumerate(station_names):
        rec = station_static.get(str(name))
        if rec is None:
            unknown.append(str(name))
            continue
        static_cont[i] = np.asarray(rec.get("static_continuous", [0, 0, 0, 0, 0]), dtype=np.float32)
        veg_idx[i] = float(rec.get("vegetation_index", 0))

    split_dyn = int(window_size) * dyn_vars
    total_dim = split_dyn + 5 + 1 + EXTRA_FEATURE_DIM
    raw_rows = np.empty((nt * ns, total_dim), dtype=np.float32)
    write0 = 0
    for t in range(nt):
        idxs = np.arange(t - int(window_size) + 1, t + 1)
        idxs = np.clip(idxs, 0, nt - 1)
        wins = cube[idxs, :, :].transpose(1, 0, 2).astype(np.float32)
        dyn_flat = wins.reshape(ns, split_dyn)
        fog_features = compute_fog_features(wins)
        cyc = time_cyclical_features([times[t]], ns, local_time_offset_hours)
        extra = np.concatenate([fog_features, cyc], axis=1)
        write1 = write0 + ns
        raw_rows[write0:write1, :split_dyn] = dyn_flat
        raw_rows[write0:write1, split_dyn : split_dyn + 5] = static_cont
        raw_rows[write0:write1, split_dyn + 5] = veg_idx
        raw_rows[write0:write1, split_dyn + 6 :] = extra
        write0 = write1

    x = transform_pmst_rows(raw_rows, scaler, window_size, dyn_vars)
    coords = {
        "time": np.asarray(times),
        "station_name": np.asarray(station_names, dtype=object),
    }
    return x, coords, unknown


class StableChebyKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, degree: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.degree = int(degree)
        self.base = nn.Linear(self.input_dim, self.output_dim)
        self.cheby = nn.Parameter(
            torch.empty(self.degree, self.input_dim, self.output_dim)
        )
        nn.init.xavier_uniform_(self.cheby)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clip = torch.tanh(x)
        terms = []
        if self.degree >= 1:
            terms.append(x_clip)
        if self.degree >= 2:
            terms.append(2.0 * x_clip * x_clip - 1.0)
        for k in range(3, self.degree + 1):
            terms.append(2.0 * x_clip * terms[-1] - terms[-2])
        y = self.base(x)
        for k, term in enumerate(terms):
            y = y + torch.einsum("bi,io->bo", term, self.cheby[k])
        return self.norm(F.gelu(y))


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, int(channel) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class AirportPMSTNet(nn.Module):
    def __init__(
        self,
        dyn_vars_count: int = len(DYNAMIC_FEATURE_ORDER),
        window_size: int = WINDOW_SIZE,
        station_count: int = 178,
        station_emb_dim: int = 32,
        hidden_dim: int = 384,
        num_classes: int = 3,
        extra_feat_dim: int = EXTRA_FEATURE_DIM,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dyn_vars = int(dyn_vars_count)
        self.window = int(window_size)
        self.station_count = int(station_count)
        self.station_emb_dim = int(station_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.extra_feat_dim = int(extra_feat_dim)

        key_names = [
            "RH2M",
            "T2M",
            "PRECIP",
            "SW_RAD",
            "WSPD10",
            "LCC",
            "DPD",
            "INVERSION",
            "RH_925",
            "T_925",
            "U_925",
            "V_925",
            "DP_925",
            "Q_925",
            "ZENITH_PROXY",
        ]
        name_to_idx = {name: i for i, name in enumerate(DYNAMIC_FEATURE_ORDER)}
        self.temporal_var_indices = [
            name_to_idx[name] for name in key_names if name_to_idx[name] < self.dyn_vars
        ]

        self.station_embedding = nn.Embedding(self.station_count, self.station_emb_dim)
        self.station_encoder = nn.Sequential(
            nn.Linear(self.station_emb_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )

        self.physics_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 4),
        )

        self.temporal_input_proj = nn.Linear(len(self.temporal_var_indices), hidden_dim)
        self.temporal_stream = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.se_block = SEBlock(hidden_dim * 2)
        self.temporal_norm = nn.LayerNorm(hidden_dim * 2)

        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        fusion_dim = hidden_dim * 2 + hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 2
        self.fusion_kan = StableChebyKANLayer(fusion_dim, hidden_dim, degree=2)
        self.dropout = nn.Dropout(dropout)
        self.fine_classifier = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.low_vis_detector = nn.Linear(hidden_dim, 1)

    def _physics_features(self, x: torch.Tensor) -> torch.Tensor:
        rh2m = x[:, :, 0]
        t2m = x[:, :, 1]
        wspd = torch.clamp(x[:, :, 6], min=0.1)
        rh925 = x[:, :, 12]
        dpd = x[:, :, 22]
        inv = x[:, :, 23]
        lcc = x[:, :, 10]

        f1 = rh2m / 100.0 * torch.sigmoid(-2.0 * dpd)
        f2 = 1.0 / (wspd + 1.0)
        f3 = inv * f2
        f4 = (rh2m - rh925) / 100.0
        f5 = lcc
        f6 = torch.sigmoid((rh2m - 90.0) / 5.0)
        f7 = t2m[:, -1:].expand_as(t2m) - t2m[:, :1].expand_as(t2m)
        return torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=2).mean(dim=1)

    def forward(self, x: torch.Tensor):
        split_dyn = self.dyn_vars * self.window
        x_dyn = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        station_idx = x[:, split_dyn].long().clamp(0, self.station_count - 1)
        x_extra = x[:, split_dyn + 1 : split_dyn + 1 + self.extra_feat_dim]

        phy_feat = self.physics_encoder(self._physics_features(x_dyn))
        station_feat = self.station_encoder(self.station_embedding(station_idx))

        temporal_vars = x_dyn[:, :, self.temporal_var_indices]
        t_in = self.temporal_input_proj(temporal_vars)
        t_out, _ = self.temporal_stream(t_in)
        t_out_se = self.se_block(t_out.permute(0, 2, 1)).permute(0, 2, 1)
        t_feat = self.temporal_norm(t_out_se.mean(dim=1))
        extra_feat = self.extra_encoder(x_extra)

        emb = torch.cat([t_feat, station_feat, phy_feat, extra_feat], dim=1)
        emb = self.dropout(self.fusion_kan(emb))
        return self.fine_classifier(emb), self.reg_head(emb), self.low_vis_detector(emb)


class ImprovedDualStreamPMSTNet(nn.Module):
    def __init__(
        self,
        dyn_vars_count: int = len(DYNAMIC_FEATURE_ORDER),
        window_size: int = WINDOW_SIZE,
        static_cont_dim: int = 5,
        veg_num_classes: int = 21,
        hidden_dim: int = 512,
        num_classes: int = 3,
        extra_feat_dim: int = EXTRA_FEATURE_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dyn_vars = int(dyn_vars_count)
        self.window = int(window_size)
        self.temporal_var_indices = [
            0, 1, 2, 4, 6, 10, 22, 23, 12, 11, 13, 15, 14, self.dyn_vars - 1
        ]

        self.veg_embedding = nn.Embedding(veg_num_classes, 16)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_cont_dim + 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim // 4),
        )

        self.physics_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 4),
        )

        self.temporal_input_proj = nn.Linear(len(self.temporal_var_indices), hidden_dim)
        self.temporal_stream = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.se_block = SEBlock(hidden_dim * 2)
        self.temporal_norm = nn.LayerNorm(hidden_dim * 2)

        fusion_dim = (hidden_dim * 2) + (hidden_dim // 4) + (hidden_dim // 4)
        if extra_feat_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            fusion_dim += hidden_dim // 2
        else:
            self.extra_encoder = None

        self.fusion_kan = StableChebyKANLayer(fusion_dim, hidden_dim, degree=2)
        self.dropout = nn.Dropout(dropout)
        self.fine_classifier = nn.Linear(hidden_dim, num_classes)
        self.low_vis_detector = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def _physics_features(self, x: torch.Tensor) -> torch.Tensor:
        rh2m = x[:, :, 0]
        t2m = x[:, :, 1]
        wspd = torch.clamp(x[:, :, 6], min=0.1)
        dpd = x[:, :, 22]
        inv = x[:, :, 23]

        f1 = rh2m / 100.0 * torch.sigmoid(-2.0 * dpd)
        f2 = 1.0 / (wspd + 1.0)
        f3 = inv * f2
        f4 = (rh2m - x[:, :, 12]) / 100.0
        f5 = x[:, :, 10]
        f6 = torch.sigmoid((rh2m - 90.0) / 5.0)
        f7 = t2m[:, -1:].expand_as(t2m) - t2m[:, :1].expand_as(t2m)
        return torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=2).mean(dim=1)

    def forward(self, x: torch.Tensor):
        split_dyn = self.dyn_vars * self.window
        split_static = split_dyn + 5

        x_dyn = x[:, :split_dyn].view(-1, self.window, self.dyn_vars)
        x_stat = x[:, split_dyn:split_static]
        x_veg = x[:, split_static].long()
        x_extra = x[:, split_static + 1 :] if self.extra_encoder else None

        phy_feat = self.physics_encoder(self._physics_features(x_dyn))
        veg_emb = self.veg_embedding(torch.clamp(x_veg, 0, 20))
        stat_feat = self.static_encoder(torch.cat([x_stat, veg_emb], dim=1))

        imp_vars = x_dyn[:, :, self.temporal_var_indices]
        t_in = self.temporal_input_proj(imp_vars)
        t_out, _ = self.temporal_stream(t_in)
        t_out_se = self.se_block(t_out.permute(0, 2, 1)).permute(0, 2, 1)
        t_feat = self.temporal_norm(t_out_se.mean(dim=1))

        parts = [t_feat, stat_feat, phy_feat]
        if x_extra is not None and self.extra_encoder:
            parts.append(self.extra_encoder(x_extra))
        emb = torch.cat(parts, dim=1)
        emb = self.dropout(self.fusion_kan(emb))
        return self.fine_classifier(emb), self.reg_head(emb), self.low_vis_detector(emb)


def airport_model_config(
    station_count: int,
    window_size: int = WINDOW_SIZE,
    hidden_dim: int = 384,
    dropout: float = 0.25,
    station_emb_dim: int = 32,
) -> Dict[str, object]:
    return {
        "model_type": "airport_pmst",
        "dyn_vars_count": len(DYNAMIC_FEATURE_ORDER),
        "window_size": int(window_size),
        "station_count": int(station_count),
        "station_emb_dim": int(station_emb_dim),
        "hidden_dim": int(hidden_dim),
        "num_classes": 3,
        "extra_feat_dim": EXTRA_FEATURE_DIM,
        "dropout": float(dropout),
    }


def build_airport_model(config: Mapping) -> AirportPMSTNet:
    return AirportPMSTNet(
        dyn_vars_count=int(config.get("dyn_vars_count", len(DYNAMIC_FEATURE_ORDER))),
        window_size=int(config.get("window_size", WINDOW_SIZE)),
        station_count=int(config.get("station_count", 178)),
        station_emb_dim=int(config.get("station_emb_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 384)),
        num_classes=int(config.get("num_classes", 3)),
        extra_feat_dim=int(config.get("extra_feat_dim", EXTRA_FEATURE_DIM)),
        dropout=float(config.get("dropout", 0.25)),
    )


def build_improved_pmst_model(config: Mapping) -> ImprovedDualStreamPMSTNet:
    return ImprovedDualStreamPMSTNet(
        dyn_vars_count=int(config.get("dyn_vars_count", len(DYNAMIC_FEATURE_ORDER))),
        window_size=int(config.get("window_size", WINDOW_SIZE)),
        static_cont_dim=int(config.get("static_cont_dim", 5)),
        veg_num_classes=int(config.get("veg_num_classes", 21)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_classes=int(config.get("num_classes", 3)),
        extra_feat_dim=int(config.get("extra_feat_dim", EXTRA_FEATURE_DIM)),
        dropout=float(config.get("dropout", 0.3)),
    )


def predict_classes_from_probs(
    probabilities: np.ndarray,
    thresholds: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    th = thresholds or {}
    fog_th = float(th.get("fog", th.get("fog_threshold", 0.50)))
    mist_th = float(th.get("mist", th.get("mist_threshold", 0.50)))
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    pred[probs[:, 1] >= mist_th] = 1
    pred[probs[:, 0] >= fog_th] = 0
    fallback = np.argmax(probs, axis=1).astype(np.int64)
    low_conf = (probs[:, 0] < fog_th) & (probs[:, 1] < mist_th) & (fallback != 2)
    pred[low_conf] = fallback[low_conf]
    return pred
