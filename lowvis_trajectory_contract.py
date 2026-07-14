#!/usr/bin/env python3
"""Torch-free data contract for low-visibility trajectory datasets."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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
