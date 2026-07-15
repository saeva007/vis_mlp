#!/usr/bin/env python3
"""Torch-free data contract for low-visibility trajectory datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


CONDITION_LEADS = tuple(range(0, 49))
TARGET_LEADS = tuple(range(12, 49))
TARGET_LENGTH = len(TARGET_LEADS)
DYNAMIC_FEATURE_ORDER = (
    "RH2M", "T2M", "PRECIP", "MSLP", "SW_RAD", "U10", "WSPD10",
    "V10", "WDIR10", "CAPE", "LCC", "T_925", "RH_925", "U_925",
    "WSPD925", "V_925", "DP_1000", "DP_925", "Q_1000", "Q_925",
    "W_925", "W_1000", "DPD", "INVERSION", "ZENITH", "PM10_ugm3",
    "PM25_ugm3",
)
PM_UNIT_POLICY_VERSION = "pmst_canonical_units_v2_20260630"
PM_QC_POLICY_VERSION = "pm_explicit_legacy_scale_then_train_median_qc_v2_20260701"
MAX_VISIBILITY_M = 30000.0


def exact_lead_indices(lead_values: np.ndarray) -> Optional[np.ndarray]:
    lead_values = np.asarray(lead_values, dtype=float)
    indices = []
    for lead in CONDITION_LEADS:
        pos = int(np.argmin(np.abs(lead_values - lead)))
        if abs(float(lead_values[pos]) - float(lead)) > 0.1:
            return None
        indices.append(pos)
    if len(set(indices)) != len(indices):
        return None
    return np.asarray(indices, dtype=np.int64)


def shifted_lead_indices(
    lead_values: np.ndarray,
    requested_shift_hours: Union[str, float] = "auto",
    auto_candidates: Sequence[float] = (0.0, -8.0, 8.0),
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Resolve the 0--48 h trajectory after an explicit or audited time shift.

    Some Tianji station products store valid times in Beijing time while the
    run name is UTC.  In that case the raw lead coordinate is 8--56 h and the
    established normalization is -8 h.  Auto mode only tests the small,
    declared candidate set; it never invents a shift from the observations.
    """

    if isinstance(requested_shift_hours, str):
        value = requested_shift_hours.strip().lower()
        if value == "auto":
            candidates = tuple(float(v) for v in auto_candidates)
        else:
            candidates = (float(value),)
    else:
        candidates = (float(requested_shift_hours),)
    for shift in candidates:
        indices = exact_lead_indices(np.asarray(lead_values, dtype=float) + shift)
        if indices is not None:
            return indices, shift
    return None, None


def canonical_station_key(value: object) -> str:
    """Normalize numeric station-id representations without altering named ids."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer() and text.endswith(".0"):
        return str(int(numeric))
    return text


def visibility_grid(
    vis_da: Any,
    target_times: pd.DatetimeIndex,
    stations: np.ndarray,
    tolerance_minutes: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return visibility as [station, target_lead] with audited coordinate matching.

    ``vis_da`` is intentionally duck-typed so this data contract stays free of
    both Torch and an import-time xarray dependency.
    """

    required = {"time", "station_id"}
    if not required.issubset(vis_da.dims):
        raise ValueError(f"Visibility must have time/station_id dimensions; got {vis_da.dims}")
    extra = [dim for dim in vis_da.dims if dim not in required]
    if extra:
        non_singleton = {dim: int(vis_da.sizes[dim]) for dim in extra if int(vis_da.sizes[dim]) != 1}
        if non_singleton:
            raise ValueError(f"Visibility has unsupported non-singleton dimensions: {non_singleton}")
        vis_da = vis_da.isel({dim: 0 for dim in extra}, drop=True)
    ordered = vis_da.transpose("time", "station_id").sortby("time")
    source_times = pd.DatetimeIndex(pd.to_datetime(ordered.time.values))
    if source_times.has_duplicates:
        raise ValueError("Visibility time coordinate contains duplicates")
    time_pos = source_times.get_indexer(target_times, method="nearest")
    valid_t = time_pos >= 0
    if valid_t.any():
        valid_locations = np.flatnonzero(valid_t)
        delta = np.abs(source_times.asi8[time_pos[valid_t]] - target_times.asi8[valid_t])
        valid_t[valid_locations[delta > pd.Timedelta(minutes=tolerance_minutes).value]] = False

    source_keys = pd.Index([canonical_station_key(v) for v in ordered.station_id.values])
    if source_keys.has_duplicates:
        duplicates = source_keys[source_keys.duplicated()].unique().tolist()[:5]
        raise ValueError(f"Visibility station ids are duplicated after normalization: {duplicates}")
    station_pos = source_keys.get_indexer([canonical_station_key(v) for v in stations])
    valid_s = station_pos >= 0

    out = np.full((len(target_times), len(stations)), np.nan, dtype=np.float32)
    if valid_t.any() and valid_s.any():
        raw = ordered.isel(
            time=np.flatnonzero(valid_t).tolist(),
            station_id=station_pos[valid_s].tolist(),
        ).values
        out[np.ix_(np.flatnonzero(valid_t), np.flatnonzero(valid_s))] = np.asarray(raw, dtype=np.float32)
    diagnostics = {
        "matched_target_times": int(valid_t.sum()),
        "target_times": int(len(target_times)),
        "matched_stations": int(valid_s.sum()),
        "forecast_stations": int(len(stations)),
    }
    return out.T, diagnostics


def monthly_tail_masks(
    sample_times: Sequence[object],
    gap_hours: int = 24,
    val_last_days: int = 3,
    test_last_days: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = pd.DatetimeIndex(pd.to_datetime(sample_times))
    if gap_hours % 24:
        raise ValueError("gap_hours must be a multiple of 24")
    train = np.zeros(len(times), dtype=bool)
    val = np.zeros(len(times), dtype=bool)
    test = np.zeros(len(times), dtype=bool)
    gap_days = gap_hours // 24
    for period in times.to_period("M").unique():
        dim = period.days_in_month
        d_test0 = dim - test_last_days + 1
        d_val1 = d_test0 - gap_days - 1
        d_val0 = d_val1 - val_last_days + 1
        d_train_end = d_val0 - gap_days - 1
        if d_train_end < 1:
            continue
        start, end = period.start_time, period.end_time
        in_month = (times >= start) & (times <= end)
        train_end = pd.Timestamp(period.year, period.month, d_train_end) + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        val_start = pd.Timestamp(period.year, period.month, d_val0)
        val_end = pd.Timestamp(period.year, period.month, d_val1) + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        test_start = pd.Timestamp(period.year, period.month, d_test0)
        train |= in_month & (times <= train_end)
        val |= in_month & (times >= val_start) & (times <= val_end)
        test |= in_month & (times >= test_start)
    return train, val, test


def full_trajectory_split(
    target_times: Sequence[object],
    gap_hours: int = 24,
    val_last_days: int = 3,
    test_last_days: int = 3,
) -> Optional[str]:
    masks = monthly_tail_masks(target_times, gap_hours, val_last_days, test_last_days)
    for tag, mask in zip(("train", "val", "test"), masks):
        if bool(np.all(mask)):
            return tag
    return None


def time_features_from_init(values: Sequence[object]) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(values))
    hour = times.hour.to_numpy(dtype=np.float32)
    day = times.dayofyear.to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day / 366.0),
            np.cos(2.0 * np.pi * day / 366.0),
        ]
    ).astype(np.float32)
