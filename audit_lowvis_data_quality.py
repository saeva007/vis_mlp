#!/usr/bin/env python3
"""Read-only quality audit for low-visibility datasets and raw observations.

The audit never edits source arrays.  It understands both the historical flat
``X_*.npy/y_*.npy`` contract and the candidate trajectory contract, and uses
the repository's current canonical PM policy rather than treating historical
``kg m-3 * 1e12`` values as physical ``ug m-3`` values.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_PM_UNIT_POLICY = "pmst_canonical_units_v2_20260630"
EXPECTED_PM_QC_POLICY = "pm_explicit_legacy_scale_then_train_median_qc_v2_20260701"
VISIBILITY_MAX_M = 30000.0
COMPARISON_LEADS = tuple(range(12, 49))
SPLITS = ("train", "val", "test")

PM_NAMES = {"PM10", "PM10UGM3", "PM25", "PM25UGM3", "PM2P5"}

# Only ranges whose units are already established by the canonical PMST policy
# are enforced here.  PRECIP/SW_RAD/CAPE remain distribution-audit variables
# until each source's accumulation/flux convention is explicit.
PHYSICAL_BOUNDS: Mapping[str, Tuple[Optional[float], Optional[float]]] = {
    "T2M": (180.0, 340.0),
    "T925": (180.0, 340.0),
    "T1000": (180.0, 340.0),
    "D2M": (150.0, 340.0),
    "DP1000": (150.0, 340.0),
    "DP925": (150.0, 340.0),
    "MSLP": (50000.0, 120000.0),
    "RH2M": (0.0, 100.5),
    "RH925": (0.0, 100.5),
    "RH1000": (0.0, 100.5),
    "U10": (-150.0, 150.0),
    "V10": (-150.0, 150.0),
    "U925": (-150.0, 150.0),
    "V925": (-150.0, 150.0),
    "WSPD10": (0.0, 150.0),
    "WSPD925": (0.0, 150.0),
    "WDIR10": (0.0, 360.0),
    "Q1000": (0.0, 0.08),
    "Q925": (0.0, 0.08),
    "LCC": (0.0, 1.05),
    "ZENITH": (0.0, 180.0),
}

SENTINEL_VIS_VALUES = (999999.0, 999998.0, 90000.0, 81900.0, 75000.0, 70050.0, 40000.0, 35000.0)


def normalize_name(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def canonical_station_key(value: object) -> str:
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_specs(text: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid spec {item!r}; expected tag=/path")
        tag, raw_path = item.split("=", 1)
        tag = tag.strip()
        if not tag or tag in out:
            raise ValueError(f"Empty or duplicate tag in spec {item!r}")
        out[tag] = Path(raw_path.strip()).expanduser().resolve()
    return out


def mixed_datetime(values: pd.Series | Sequence[object]) -> pd.Series:
    series = pd.Series(values, copy=False).astype("string").str.strip()
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        return series.map(lambda value: pd.to_datetime(value, errors="coerce", utc=True))


def file_record(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def config_digest(config: Mapping[str, object]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class AuditState:
    rows: MutableMapping[str, List[Dict[str, object]]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )
    issues: List[Dict[str, object]] = field(default_factory=list)
    expected_rows: MutableMapping[str, Dict[str, int]] = field(
        default_factory=lambda: collections.defaultdict(dict)
    )
    metadata_hashes: MutableMapping[str, Dict[str, np.ndarray]] = field(
        default_factory=lambda: collections.defaultdict(dict)
    )

    def issue(self, severity: str, code: str, message: str, dataset: str = "", split: str = "") -> None:
        item = {
            "severity": str(severity).upper(),
            "code": code,
            "dataset": dataset,
            "split": split,
            "message": message,
        }
        self.issues.append(item)
        print(f"[{item['severity']}] {dataset}/{split} {code}: {message}", flush=True)


def load_pm_policy(common_dir: Path):
    common_dir = common_dir.resolve()
    if not (common_dir / "pmst_overlap_common.py").is_file():
        raise FileNotFoundError(f"Missing canonical PM policy: {common_dir / 'pmst_overlap_common.py'}")
    sys.path.insert(0, str(common_dir))
    policy = importlib.import_module("pmst_overlap_common")
    if str(policy.CANONICAL_UNIT_POLICY_VERSION) != EXPECTED_PM_UNIT_POLICY:
        raise RuntimeError(
            f"PM unit policy mismatch: {policy.CANONICAL_UNIT_POLICY_VERSION!r} != {EXPECTED_PM_UNIT_POLICY!r}"
        )
    if str(policy.PM_QC_POLICY_VERSION) != EXPECTED_PM_QC_POLICY:
        raise RuntimeError(
            f"PM QC policy mismatch: {policy.PM_QC_POLICY_VERSION!r} != {EXPECTED_PM_QC_POLICY!r}"
        )
    return policy


def load_dataset_config(path: Path) -> Tuple[Dict[str, object], Optional[Path]]:
    for name in ("dataset_build_config.json", "dataset_split_config.json", "dataset_metadata.json"):
        candidate = path / name
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise TypeError(f"Dataset config must be a JSON object: {candidate}")
            return value, candidate
    return {}, None


def available_splits(path: Path, trajectory: bool) -> List[str]:
    prefix = "dynamic" if trajectory else "X"
    return [split for split in SPLITS if (path / f"{prefix}_{split}.npy").is_file()]


def resolve_flat_layout(config: Mapping[str, object], x_path: Path) -> Tuple[int, List[str], int]:
    window = int(config.get("window_size", config.get("window", 12)))
    order_raw = config.get("dynamic_feature_order", config.get("dynamic_order"))
    order = [str(value) for value in order_raw] if isinstance(order_raw, list) else []
    dyn_vars = int(
        config.get(
            "dyn_vars",
            config.get("dyn_vars_count", config.get("dynamic_dim", len(order) if order else 0)),
        )
    )
    if order and dyn_vars != len(order):
        raise ValueError(f"dynamic_feature_order length={len(order)} != dyn_vars={dyn_vars}")
    if not order or dyn_vars <= 0:
        raise ValueError("dataset config must declare dynamic_feature_order and dyn_vars")
    width = int(np.load(x_path, mmap_mode="r").shape[1])
    if width < window * dyn_vars:
        raise ValueError(f"row width={width} is smaller than window*dyn_vars={window * dyn_vars}")
    return window, order, width


def pm_declared_units(config: Mapping[str, object], policy) -> Tuple[str, str]:
    unit_policy = str(config.get("canonical_unit_policy", ""))
    if unit_policy == EXPECTED_PM_UNIT_POLICY:
        return "ug m-3", "canonical_dataset_metadata"
    if unit_policy:
        return "", f"unknown_policy:{unit_policy}"
    if bool(config.get("include_pm")) or str(config.get("protocol", "")) in {
        "main_pm10_pm25",
        "s1_pmst_aligned_pm10_pm25",
    }:
        return str(policy.LEGACY_PM_1E12_UNITS), "explicit_historical_builder_lineage"
    return "", "magnitude_and_declared_unit_inference"


def quantiles(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    names = ("p001", "p01", "p05", "p50", "p95", "p99", "p999")
    if not finite.size:
        return {name: math.nan for name in names}
    result = np.percentile(finite, [0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9])
    return {name: float(value) for name, value in zip(names, result)}


def iter_row_slices(n: int, chunk_rows: int) -> Iterator[slice]:
    for start in range(0, n, chunk_rows):
        yield slice(start, min(start + chunk_rows, n))


def sampled_indices(n: int, max_rows: int) -> np.ndarray:
    take = min(int(n), max(int(max_rows), 1))
    return np.linspace(0, n - 1, num=take, dtype=np.int64) if n else np.empty(0, dtype=np.int64)


def empty_feature_state() -> Dict[str, object]:
    return {
        "values": 0,
        "finite": 0,
        "zero": 0,
        "outside": 0,
        "min": math.inf,
        "max": -math.inf,
        "sum": 0.0,
    }


def update_feature_state(state: MutableMapping[str, object], values: np.ndarray, bounds) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    state["values"] = int(state["values"]) + int(arr.size)
    state["finite"] = int(state["finite"]) + int(finite.sum())
    if not finite.any():
        return
    vals = arr[finite]
    state["zero"] = int(state["zero"]) + int((np.abs(vals) <= 1.0e-12).sum())
    state["sum"] = float(state["sum"]) + float(vals.sum(dtype=np.float64))
    state["min"] = min(float(state["min"]), float(vals.min()))
    state["max"] = max(float(state["max"]), float(vals.max()))
    if bounds is not None:
        lo, hi = bounds
        outside = np.zeros(vals.shape, dtype=bool)
        if lo is not None:
            outside |= vals < float(lo)
        if hi is not None:
            outside |= vals > float(hi)
        state["outside"] = int(state["outside"]) + int(outside.sum())


def feature_state_row(state: Mapping[str, object]) -> Dict[str, object]:
    total = int(state["values"])
    finite = int(state["finite"])
    return {
        "values_checked": total,
        "finite_values": finite,
        "finite_fraction": finite / max(total, 1),
        "zero_fraction_of_finite": int(state["zero"]) / max(finite, 1),
        "outside_plausible_values": int(state["outside"]),
        "outside_plausible_fraction": int(state["outside"]) / max(finite, 1),
        "mean": float(state["sum"]) / max(finite, 1),
        "min": float(state["min"]) if finite else math.nan,
        "max": float(state["max"]) if finite else math.nan,
    }


def scan_dynamic(
    state: AuditState,
    dataset: str,
    split: str,
    array: np.ndarray,
    window: int,
    order: Sequence[str],
    config: Mapping[str, object],
    policy,
    chunk_rows: int,
    quantile_rows: int,
    trajectory: bool,
) -> None:
    names = [normalize_name(name) for name in order]
    feature_states = {name: empty_feature_state() for name in order}
    pm_states = {name: empty_feature_state() for name, norm in zip(order, names) if norm in PM_NAMES}
    declared_units, pm_lineage = pm_declared_units(config, policy)

    for slc in iter_row_slices(len(array), chunk_rows):
        block = np.asarray(array[slc], dtype=np.float32)
        dyn = block if trajectory else block[:, : window * len(order)].reshape(-1, window, len(order))
        for idx, (name, norm) in enumerate(zip(order, names)):
            values = dyn[..., idx]
            update_feature_state(feature_states[name], values, PHYSICAL_BOUNDS.get(norm))
            if norm in PM_NAMES:
                canonical = policy.canonicalize_pm_concentration(values, declared_units)
                valid = np.isfinite(canonical) & (canonical >= 0.0) & (
                    canonical <= float(policy.PM_CONCENTRATION_MAX_UGM3)
                )
                canonical_for_state = np.where(valid, canonical, np.nan)
                update_feature_state(
                    pm_states[name], canonical_for_state, (0.0, float(policy.PM_CONCENTRATION_MAX_UGM3))
                )
                invalid = int((~valid).sum())
                pm_states[name]["canonical_invalid"] = int(pm_states[name].get("canonical_invalid", 0)) + invalid

    idx = sampled_indices(len(array), quantile_rows)
    sample = np.asarray(array[idx], dtype=np.float32) if len(idx) else np.empty((0,), dtype=np.float32)
    sample_dyn = sample if trajectory else sample[:, : window * len(order)].reshape(-1, window, len(order))
    for feature_idx, (name, norm) in enumerate(zip(order, names)):
        row = {
            "dataset": dataset,
            "split": split,
            "feature": name,
            "normalized_feature": norm,
            "window": int(window),
            "bounds_low": PHYSICAL_BOUNDS.get(norm, (None, None))[0],
            "bounds_high": PHYSICAL_BOUNDS.get(norm, (None, None))[1],
            **feature_state_row(feature_states[name]),
        }
        if len(idx):
            row.update(quantiles(sample_dyn[..., feature_idx]))
        state.rows["dynamic_feature_quality"].append(row)
        nonfinite_fraction = 1.0 - float(row["finite_fraction"])
        outside_fraction = float(row["outside_plausible_fraction"])
        if nonfinite_fraction > 0.10:
            state.issue("ERROR", "feature_nonfinite_gt_10pct", f"{name} non-finite={nonfinite_fraction:.3%}", dataset, split)
        elif nonfinite_fraction > 0.01:
            state.issue("WARN", "feature_nonfinite_gt_1pct", f"{name} non-finite={nonfinite_fraction:.3%}", dataset, split)
        if outside_fraction > 0.01:
            state.issue("ERROR", "feature_outside_gt_1pct", f"{name} outside={outside_fraction:.3%}", dataset, split)
        elif outside_fraction > 0.0:
            state.issue("WARN", "feature_outside_nonzero", f"{name} outside={outside_fraction:.3%}", dataset, split)

        if norm in PM_NAMES:
            pm_state = pm_states[name]
            total = int(feature_states[name]["values"])
            invalid = int(pm_state.get("canonical_invalid", 0))
            pm_row = {
                "dataset": dataset,
                "split": split,
                "feature": name,
                "stored_units_interpretation": declared_units or "magnitude_inference",
                "pm_lineage": pm_lineage,
                "unit_policy": str(config.get("canonical_unit_policy", "legacy_or_missing")),
                "qc_policy": str(config.get("pm_qc_policy", "legacy_or_missing")),
                "canonical_valid_range_ugm3": f"0..{float(policy.PM_CONCENTRATION_MAX_UGM3):g}",
                "values_checked": total,
                "canonical_invalid_values": invalid,
                "canonical_invalid_fraction": invalid / max(total, 1),
                **{f"canonical_{key}": value for key, value in feature_state_row(pm_state).items()},
            }
            if len(idx):
                canonical_sample = policy.canonicalize_pm_concentration(sample_dyn[..., feature_idx], declared_units)
                canonical_sample = np.where(
                    np.isfinite(canonical_sample)
                    & (canonical_sample >= 0.0)
                    & (canonical_sample <= float(policy.PM_CONCENTRATION_MAX_UGM3)),
                    canonical_sample,
                    np.nan,
                )
                pm_row.update({f"canonical_{key}": value for key, value in quantiles(canonical_sample).items()})
            state.rows["pm_quality"].append(pm_row)
            invalid_fraction = invalid / max(total, 1)
            if invalid_fraction > 0.01:
                state.issue("ERROR", "pm_invalid_gt_1pct", f"{name} canonical invalid={invalid_fraction:.3%}", dataset, split)
            elif invalid_fraction > 0.0:
                state.issue("WARN", "pm_invalid_nonzero", f"{name} canonical invalid={invalid_fraction:.3%}", dataset, split)


def visibility_row(values: np.ndarray, dataset: str, split: str, stored_mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    physical = finite & (arr >= 0.0) & (arr <= VISIBILITY_MAX_M)
    selected = arr[finite]
    row: Dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "values_checked": int(arr.size),
        "finite_values": int(finite.sum()),
        "physical_valid_values": int(physical.sum()),
        "physical_valid_fraction": float(physical.mean()) if arr.size else math.nan,
        "nonfinite_values": int((~finite).sum()),
        "negative_values": int((finite & (arr < 0.0)).sum()),
        "above_30km_values": int((finite & (arr > VISIBILITY_MAX_M)).sum()),
        "min_m": float(selected.min()) if selected.size else math.nan,
        "max_m": float(selected.max()) if selected.size else math.nan,
        "fog_lt500": int((physical & (arr < 500.0)).sum()),
        "mist_500_1000": int((physical & (arr >= 500.0) & (arr < 1000.0)).sum()),
        "clear_ge1000": int((physical & (arr >= 1000.0)).sum()),
    }
    for sentinel in SENTINEL_VIS_VALUES:
        row[f"count_{int(sentinel)}"] = int((finite & (arr == sentinel)).sum())
    if stored_mask is not None:
        mask = np.asarray(stored_mask, dtype=bool)
        row["stored_mask_valid"] = int(mask.sum())
        row["mask_physical_mismatch"] = int((mask != physical).sum())
    return row


def empty_visibility_state() -> Dict[str, object]:
    return {
        "values": 0,
        "finite": 0,
        "physical": 0,
        "nonfinite": 0,
        "negative": 0,
        "above": 0,
        "min": math.inf,
        "max": -math.inf,
        "fog": 0,
        "mist": 0,
        "clear": 0,
        "stored_valid": 0,
        "mask_mismatch": 0,
        "sentinels": collections.Counter(),
    }


def update_visibility_state(
    state: MutableMapping[str, object], values: np.ndarray, stored_mask: Optional[np.ndarray] = None
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    physical = finite & (arr >= 0.0) & (arr <= VISIBILITY_MAX_M)
    state["values"] = int(state["values"]) + int(arr.size)
    state["finite"] = int(state["finite"]) + int(finite.sum())
    state["physical"] = int(state["physical"]) + int(physical.sum())
    state["nonfinite"] = int(state["nonfinite"]) + int((~finite).sum())
    state["negative"] = int(state["negative"]) + int((finite & (arr < 0.0)).sum())
    state["above"] = int(state["above"]) + int((finite & (arr > VISIBILITY_MAX_M)).sum())
    if finite.any():
        finite_values = arr[finite]
        state["min"] = min(float(state["min"]), float(finite_values.min()))
        state["max"] = max(float(state["max"]), float(finite_values.max()))
    state["fog"] = int(state["fog"]) + int((physical & (arr < 500.0)).sum())
    state["mist"] = int(state["mist"]) + int((physical & (arr >= 500.0) & (arr < 1000.0)).sum())
    state["clear"] = int(state["clear"]) + int((physical & (arr >= 1000.0)).sum())
    sentinel_counts = state["sentinels"]
    for sentinel in SENTINEL_VIS_VALUES:
        sentinel_counts[sentinel] += int((finite & (arr == sentinel)).sum())
    if stored_mask is not None:
        mask = np.asarray(stored_mask, dtype=bool)
        state["stored_valid"] = int(state["stored_valid"]) + int(mask.sum())
        state["mask_mismatch"] = int(state["mask_mismatch"]) + int((mask != physical).sum())


def visibility_state_row(
    vis_state: Mapping[str, object], dataset: str, split: str, has_stored_mask: bool
) -> Dict[str, object]:
    total = int(vis_state["values"])
    finite = int(vis_state["finite"])
    row: Dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "values_checked": total,
        "finite_values": finite,
        "physical_valid_values": int(vis_state["physical"]),
        "physical_valid_fraction": int(vis_state["physical"]) / max(total, 1),
        "nonfinite_values": int(vis_state["nonfinite"]),
        "negative_values": int(vis_state["negative"]),
        "above_30km_values": int(vis_state["above"]),
        "min_m": float(vis_state["min"]) if finite else math.nan,
        "max_m": float(vis_state["max"]) if finite else math.nan,
        "fog_lt500": int(vis_state["fog"]),
        "mist_500_1000": int(vis_state["mist"]),
        "clear_ge1000": int(vis_state["clear"]),
    }
    sentinel_counts = vis_state["sentinels"]
    for sentinel in SENTINEL_VIS_VALUES:
        row[f"count_{int(sentinel)}"] = int(sentinel_counts[sentinel])
    if has_stored_mask:
        row["stored_mask_valid"] = int(vis_state["stored_valid"])
        row["mask_physical_mismatch"] = int(vis_state["mask_mismatch"])
    return row


def audit_metadata(
    state: AuditState,
    dataset: str,
    path: Path,
    splits: Sequence[str],
    chunksize: int,
) -> None:
    split_hashes: Dict[str, np.ndarray] = {}
    for split in splits:
        meta_path = path / f"meta_{split}.csv"
        if not meta_path.is_file():
            state.issue("WARN", "metadata_missing", str(meta_path), dataset, split)
            continue
        columns = list(pd.read_csv(meta_path, nrows=0).columns)
        time_col = next((name for name in ("time", "valid_time", "init_time") if name in columns), None)
        station_col = next((name for name in ("station_id", "station", "id") if name in columns), None)
        if time_col is None or station_col is None:
            state.issue("ERROR", "metadata_key_missing", f"columns={columns}", dataset, split)
            continue
        all_hashes: List[np.ndarray] = []
        invalid_times = 0
        rows = 0
        min_time = None
        max_time = None
        extra_key_columns = []
        if "lead_hour" in columns:
            extra_key_columns.append("lead_hour")
        if "init_time" in columns and time_col != "init_time":
            extra_key_columns.append("init_time")
        usecols = [time_col, station_col, *extra_key_columns]
        for chunk in pd.read_csv(meta_path, usecols=usecols, chunksize=chunksize):
            times = mixed_datetime(chunk[time_col])
            invalid_times += int(times.isna().sum())
            valid_times = times.dropna()
            if len(valid_times):
                current_min, current_max = valid_times.min(), valid_times.max()
                min_time = current_min if min_time is None else min(min_time, current_min)
                max_time = current_max if max_time is None else max(max_time, current_max)
            key_values: Dict[str, object] = {
                "time": times.astype("int64", copy=False),
                "station": chunk[station_col].map(canonical_station_key),
            }
            if "lead_hour" in extra_key_columns:
                key_values["lead_hour"] = pd.to_numeric(chunk["lead_hour"], errors="coerce")
            if "init_time" in extra_key_columns:
                key_values["init_time"] = mixed_datetime(chunk["init_time"]).astype("int64", copy=False)
            key_frame = pd.DataFrame(key_values)
            all_hashes.append(pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype=np.uint64))
            rows += len(chunk)
        hashes = np.concatenate(all_hashes) if all_hashes else np.empty(0, dtype=np.uint64)
        unique_hashes, counts = np.unique(hashes, return_counts=True)
        duplicates = int((counts[counts > 1] - 1).sum())
        split_hashes[split] = unique_hashes
        expected = state.expected_rows[dataset].get(split)
        state.rows["metadata_quality"].append(
            {
                "dataset": dataset,
                "split": split,
                "path": str(meta_path),
                "rows": int(rows),
                "expected_rows": expected,
                "row_count_matches": expected is None or int(expected) == int(rows),
                "time_column": time_col,
                "station_column": station_col,
                "invalid_times": int(invalid_times),
                "duplicate_keys": duplicates,
                "min_time_utc": str(min_time) if min_time is not None else "",
                "max_time_utc": str(max_time) if max_time is not None else "",
            }
        )
        if expected is not None and rows != expected:
            state.issue("ERROR", "metadata_row_mismatch", f"meta={rows}, arrays={expected}", dataset, split)
        if invalid_times:
            state.issue("ERROR", "metadata_invalid_time", f"invalid={invalid_times}", dataset, split)
        if duplicates:
            state.issue("ERROR", "metadata_duplicate_key", f"duplicates={duplicates}", dataset, split)

    state.metadata_hashes[dataset] = split_hashes
    for i, left in enumerate(splits):
        if left not in split_hashes:
            continue
        for right in splits[i + 1 :]:
            if right not in split_hashes:
                continue
            overlap = int(np.intersect1d(split_hashes[left], split_hashes[right], assume_unique=True).size)
            state.rows["split_overlap"].append(
                {"dataset": dataset, "left_split": left, "right_split": right, "overlap_keys": overlap}
            )
            if overlap:
                state.issue("ERROR", "split_key_overlap", f"{left}/{right} overlap={overlap}", dataset)


def audit_flat_dataset(
    state: AuditState,
    tag: str,
    path: Path,
    policy,
    chunk_rows: int,
    quantile_rows: int,
    metadata_chunksize: int,
) -> None:
    config, config_path = load_dataset_config(path)
    splits = available_splits(path, trajectory=False)
    if not splits:
        state.issue("ERROR", "flat_splits_missing", f"No X_*.npy under {path}", tag)
        return
    try:
        window, order, width = resolve_flat_layout(config, path / f"X_{splits[0]}.npy")
    except Exception as exc:
        state.issue("ERROR", "layout_invalid", str(exc), tag)
        return
    unit_policy = str(config.get("canonical_unit_policy", ""))
    if unit_policy and unit_policy != EXPECTED_PM_UNIT_POLICY:
        state.issue("ERROR", "pm_unit_policy_mismatch", unit_policy, tag)
    if not unit_policy and any(normalize_name(name) in PM_NAMES for name in order):
        state.issue(
            "WARN",
            "legacy_pm_policy_missing",
            "PM is audited through explicit historical/magnitude canonicalization; dataset itself predates the policy",
            tag,
        )
    state.rows["dataset_inventory"].append(
        {
            "dataset": tag,
            "path": str(path),
            "contract": "flat",
            "splits": ",".join(splits),
            "window": window,
            "dyn_vars": len(order),
            "row_width": width,
            "config_path": str(config_path) if config_path else "",
            "config_sha256": config_digest(config),
            "canonical_unit_policy": unit_policy or "legacy_or_missing",
            "pm_qc_policy": str(config.get("pm_qc_policy", "legacy_or_missing")),
        }
    )
    for split in splits:
        x_path = path / f"X_{split}.npy"
        y_path = path / f"y_{split}.npy"
        if not y_path.is_file():
            state.issue("ERROR", "label_file_missing", str(y_path), tag, split)
            continue
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        state.expected_rows[tag][split] = int(len(x))
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            state.issue("ERROR", "array_shape_invalid", f"X={x.shape}, y={y.shape}", tag, split)
            continue
        vis_row = visibility_row(y, tag, split)
        state.rows["visibility_quality"].append(vis_row)
        if int(vis_row["nonfinite_values"]) or int(vis_row["negative_values"]):
            state.issue("ERROR", "saved_visibility_invalid", json.dumps(vis_row, default=str), tag, split)
        if int(vis_row["above_30km_values"]):
            severity = "WARN" if "s1" in tag.lower() else "ERROR"
            state.issue(severity, "saved_visibility_above_30km", f"count={vis_row['above_30km_values']}", tag, split)
        scan_dynamic(state, tag, split, x, window, order, config, policy, chunk_rows, quantile_rows, False)
    audit_metadata(state, tag, path, splits, metadata_chunksize)


def audit_trajectory_dataset(
    state: AuditState,
    tag: str,
    path: Path,
    policy,
    chunk_rows: int,
    quantile_rows: int,
    metadata_chunksize: int,
) -> None:
    config, config_path = load_dataset_config(path)
    splits = available_splits(path, trajectory=True)
    if not splits:
        state.issue("ERROR", "trajectory_splits_missing", f"No dynamic_*.npy under {path}", tag)
        return
    order_raw = config.get("dynamic_feature_order")
    order = [str(value) for value in order_raw] if isinstance(order_raw, list) else []
    if not order:
        state.issue("ERROR", "trajectory_order_missing", "dynamic_feature_order is required", tag)
        return
    unit_policy = str(config.get("canonical_unit_policy", ""))
    qc_policy = str(config.get("pm_qc_policy", ""))
    if unit_policy != EXPECTED_PM_UNIT_POLICY:
        state.issue("ERROR", "pm_unit_policy_mismatch", unit_policy or "missing", tag)
    if qc_policy != EXPECTED_PM_QC_POLICY:
        state.issue("ERROR", "pm_qc_policy_mismatch", qc_policy or "missing", tag)
    target_leads = [int(value) for value in config.get("target_leads", list(range(1, 49)))]
    state.rows["dataset_inventory"].append(
        {
            "dataset": tag,
            "path": str(path),
            "contract": "trajectory",
            "splits": ",".join(splits),
            "window": int(config.get("condition_length", len(target_leads))),
            "dyn_vars": len(order),
            "row_width": len(order),
            "config_path": str(config_path) if config_path else "",
            "config_sha256": config_digest(config),
            "canonical_unit_policy": unit_policy or "missing",
            "pm_qc_policy": qc_policy or "missing",
        }
    )
    comparison_positions = [i for i, lead in enumerate(target_leads) if lead in COMPARISON_LEADS]
    for split in splits:
        dynamic = np.load(path / f"dynamic_{split}.npy", mmap_mode="r")
        visibility = np.load(path / f"visibility_{split}.npy", mmap_mode="r")
        mask = np.load(path / f"target_mask_{split}.npy", mmap_mode="r")
        state.expected_rows[tag][split] = int(len(dynamic))
        if dynamic.ndim != 3 or dynamic.shape[1:] != (len(target_leads), len(order)):
            state.issue("ERROR", "trajectory_dynamic_shape", f"shape={dynamic.shape}", tag, split)
            continue
        if visibility.shape != mask.shape or visibility.shape != (len(dynamic), len(target_leads)):
            state.issue("ERROR", "trajectory_target_shape", f"vis={visibility.shape}, mask={mask.shape}", tag, split)
            continue
        valid_by_lead = np.zeros(len(target_leads), dtype=np.int64)
        physical_by_lead = np.zeros(len(target_leads), dtype=np.int64)
        mismatch_by_lead = np.zeros(len(target_leads), dtype=np.int64)
        hist: collections.Counter[int] = collections.Counter()
        complete_all = 0
        complete_comparison = 0
        vis_state = empty_visibility_state()
        for slc in iter_row_slices(len(dynamic), chunk_rows):
            values = np.asarray(visibility[slc], dtype=np.float32)
            stored = np.asarray(mask[slc], dtype=bool)
            physical = np.isfinite(values) & (values >= 0.0) & (values <= VISIBILITY_MAX_M)
            valid_by_lead += stored.sum(axis=0)
            physical_by_lead += physical.sum(axis=0)
            mismatch_by_lead += (stored != physical).sum(axis=0)
            hist.update(int(value) for value in stored.sum(axis=1).tolist())
            complete_all += int(np.all(stored, axis=1).sum())
            if comparison_positions:
                complete_comparison += int(np.all(stored[:, comparison_positions], axis=1).sum())
            update_visibility_state(vis_state, values, stored)
        state.rows["visibility_quality"].append(visibility_state_row(vis_state, tag, split, True))
        for pos, lead in enumerate(target_leads):
            row = {
                "dataset": tag,
                "split": split,
                "lead_hour": int(lead),
                "rows": int(len(dynamic)),
                "stored_valid": int(valid_by_lead[pos]),
                "stored_valid_rate": float(valid_by_lead[pos]) / max(len(dynamic), 1),
                "physical_valid": int(physical_by_lead[pos]),
                "physical_valid_rate": float(physical_by_lead[pos]) / max(len(dynamic), 1),
                "mask_physical_mismatch": int(mismatch_by_lead[pos]),
                "complete_1_48": int(complete_all),
                "complete_12_48": int(complete_comparison),
                "valid_count_histogram": json.dumps(dict(sorted(hist.items()))),
            }
            state.rows["trajectory_coverage_by_lead"].append(row)
            if int(valid_by_lead[pos]) == 0:
                state.issue("ERROR", "trajectory_lead_zero_coverage", f"lead={lead}", tag, split)
            elif float(row["stored_valid_rate"]) < 0.80:
                state.issue("ERROR", "trajectory_lead_coverage_lt_80pct", f"lead={lead}, rate={row['stored_valid_rate']:.3%}", tag, split)
            elif float(row["stored_valid_rate"]) < 0.90:
                state.issue("WARN", "trajectory_lead_coverage_lt_90pct", f"lead={lead}, rate={row['stored_valid_rate']:.3%}", tag, split)
            if int(mismatch_by_lead[pos]):
                state.issue("ERROR", "trajectory_mask_value_mismatch", f"lead={lead}, count={mismatch_by_lead[pos]}", tag, split)
        if complete_comparison == 0:
            state.issue("ERROR", "trajectory_no_complete_12_48", "joint comparison metrics are undefined", tag, split)
        scan_dynamic(
            state,
            tag,
            split,
            dynamic,
            len(target_leads),
            order,
            config,
            policy,
            chunk_rows,
            quantile_rows,
            True,
        )
    audit_metadata(state, tag, path, splits, metadata_chunksize)


def open_xarray_dataset(path: Path):
    import xarray as xr

    errors = []
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
        except Exception as exc:
            errors.append(f"{engine or 'auto'}={exc}")
    raise RuntimeError(f"Cannot open {path}: {'; '.join(errors)}")


def visibility_dataarray(ds):
    for name in ("visibility", "vis", "VIS", "Visibility"):
        if name in ds.data_vars:
            da = ds[name]
            break
    else:
        raise KeyError(f"No visibility variable; data_vars={list(ds.data_vars)}")
    time_dim = next((name for name in ("time", "valid_time", "Time") if name in da.dims), None)
    station_dim = next((name for name in ("station_id", "station", "num_station", "id") if name in da.dims), None)
    if time_dim is None or station_dim is None:
        raise ValueError(f"Visibility dims must contain time/station; got {da.dims}")
    if time_dim != "time" or station_dim != "station_id":
        da = da.rename({time_dim: "time", station_dim: "station_id"})
    return da.transpose("time", "station_id")


def audit_raw_visibility(
    state: AuditState,
    tag: str,
    path: Path,
    time_chunk: int,
) -> None:
    ds = open_xarray_dataset(path)
    try:
        da = visibility_dataarray(ds)
        times = mixed_datetime(pd.Series(da["time"].values))
        hour_states = {hour: collections.Counter() for hour in range(24)}
        worst_rows: List[Dict[str, object]] = []
        over_counter: collections.Counter[float] = collections.Counter()
        summary_values = collections.Counter()
        for start in range(0, da.sizes["time"], time_chunk):
            stop = min(start + time_chunk, da.sizes["time"])
            block = np.asarray(da.isel(time=slice(start, stop)).values, dtype=np.float64)
            for local in range(block.shape[0]):
                values = block[local]
                finite = np.isfinite(values)
                valid = finite & (values >= 0.0) & (values <= VISIBILITY_MAX_M)
                negative = finite & (values < 0.0)
                over = finite & (values > VISIBILITY_MAX_M)
                timestamp = times.iloc[start + local]
                hour = int(timestamp.hour) if pd.notna(timestamp) else -1
                counts = {
                    "total": int(values.size),
                    "valid": int(valid.sum()),
                    "missing": int((~finite).sum()),
                    "negative": int(negative.sum()),
                    "over30km": int(over.sum()),
                }
                summary_values.update(counts)
                if hour in hour_states:
                    hour_states[hour].update(counts)
                if over.any():
                    over_counter.update(float(value) for value in values[over].tolist())
                worst_rows.append(
                    {
                        "source": tag,
                        "time": str(timestamp),
                        "valid_fraction": counts["valid"] / max(counts["total"], 1),
                        "missing_fraction": counts["missing"] / max(counts["total"], 1),
                        "over30km_fraction": counts["over30km"] / max(counts["total"], 1),
                    }
                )
        for hour, counts in hour_states.items():
            total = int(counts["total"])
            state.rows["raw_visibility_by_hour"].append(
                {
                    "source": tag,
                    "raw_hour": hour,
                    "values": total,
                    "valid_pct": 100.0 * counts["valid"] / max(total, 1),
                    "missing_pct": 100.0 * counts["missing"] / max(total, 1),
                    "negative_pct": 100.0 * counts["negative"] / max(total, 1),
                    "over30km_pct": 100.0 * counts["over30km"] / max(total, 1),
                }
            )
        state.rows["raw_visibility_worst_times"].extend(
            sorted(worst_rows, key=lambda row: (row["valid_fraction"], -row["missing_fraction"]))[:100]
        )
        for value, count in over_counter.most_common(100):
            state.rows["raw_visibility_over30_values"].append(
                {"source": tag, "value_m": value, "count": int(count)}
            )
        total = int(summary_values["total"])
        state.rows["raw_visibility_summary"].append(
            {
                "source": tag,
                "path": str(path),
                "time_steps": int(da.sizes["time"]),
                "stations": int(da.sizes["station_id"]),
                "time_min": str(times.min()),
                "time_max": str(times.max()),
                "values": total,
                "valid_fraction": int(summary_values["valid"]) / max(total, 1),
                "missing_fraction": int(summary_values["missing"]) / max(total, 1),
                "negative_fraction": int(summary_values["negative"]) / max(total, 1),
                "over30km_fraction": int(summary_values["over30km"]) / max(total, 1),
            }
        )
    finally:
        ds.close()


def pick_pm_dataarray(ds, tag: str):
    candidates = ("pm2p5", "pm25", "pm2_5", "PM2_5") if "25" in tag.lower() else ("pm10", "PM10")
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise KeyError(f"Cannot select PM variable for {tag}; data_vars={list(ds.data_vars)}")


def audit_raw_pm(state: AuditState, tag: str, path: Path, policy, time_chunk: int) -> None:
    ds = open_xarray_dataset(path)
    try:
        da = pick_pm_dataarray(ds, tag)
        units = str(da.attrs.get("units", ""))
        raw_state = empty_feature_state()
        canonical_state = empty_feature_state()
        invalid = 0
        first_dim = da.dims[0]
        for start in range(0, da.sizes[first_dim], time_chunk):
            values = np.asarray(da.isel({first_dim: slice(start, min(start + time_chunk, da.sizes[first_dim]))}).values)
            update_feature_state(raw_state, values, None)
            canonical = policy.canonicalize_pm_concentration(values, units)
            valid = np.isfinite(canonical) & (canonical >= 0.0) & (
                canonical <= float(policy.PM_CONCENTRATION_MAX_UGM3)
            )
            invalid += int((~valid).sum())
            update_feature_state(
                canonical_state,
                np.where(valid, canonical, np.nan),
                (0.0, float(policy.PM_CONCENTRATION_MAX_UGM3)),
            )
        total = int(raw_state["values"])
        state.rows["raw_pm_quality"].append(
            {
                "source": tag,
                "path": str(path),
                "variable": str(da.name),
                "declared_units": units,
                "unit_policy": EXPECTED_PM_UNIT_POLICY,
                "qc_policy": EXPECTED_PM_QC_POLICY,
                **{f"raw_{key}": value for key, value in feature_state_row(raw_state).items()},
                **{f"canonical_{key}": value for key, value in feature_state_row(canonical_state).items()},
                "canonical_invalid_values": invalid,
                "canonical_invalid_fraction": invalid / max(total, 1),
            }
        )
        if invalid / max(total, 1) > 0.01:
            state.issue("ERROR", "raw_pm_invalid_gt_1pct", f"invalid={invalid / max(total, 1):.3%}", tag)
        elif invalid:
            state.issue("WARN", "raw_pm_invalid_nonzero", f"invalid={invalid / max(total, 1):.3%}", tag)
    finally:
        ds.close()


def audit_trajectory_raw_consistency(
    state: AuditState,
    trajectory_tag: str,
    trajectory_path: Path,
    raw_tag: str,
    raw_path: Path,
    metadata_chunksize: int,
    max_rows: int,
) -> None:
    ds = open_xarray_dataset(raw_path)
    try:
        da = visibility_dataarray(ds)
        raw_times = pd.DatetimeIndex(mixed_datetime(pd.Series(da["time"].values)))
        raw_stations = pd.Index([canonical_station_key(value) for value in da["station_id"].values])
        raw_values = np.asarray(da.values, dtype=np.float32)
        leads = np.arange(1, 49, dtype=np.int64)
        lead_delta_ns = leads * int(pd.Timedelta(hours=1).value)
        for split in available_splits(trajectory_path, trajectory=True):
            meta_path = trajectory_path / f"meta_{split}.csv"
            if not meta_path.is_file():
                continue
            stored_mask = np.load(trajectory_path / f"target_mask_{split}.npy", mmap_mode="r")
            stored_vis = np.load(trajectory_path / f"visibility_{split}.npy", mmap_mode="r")
            counts = {
                "rows": 0,
                "raw_valid": np.zeros(48, dtype=np.int64),
                "stored_valid": np.zeros(48, dtype=np.int64),
                "mask_mismatch": np.zeros(48, dtype=np.int64),
                "value_mismatch": np.zeros(48, dtype=np.int64),
            }
            examples = 0
            offset = 0
            for meta in pd.read_csv(meta_path, chunksize=metadata_chunksize):
                if max_rows > 0 and offset >= max_rows:
                    break
                if max_rows > 0 and offset + len(meta) > max_rows:
                    meta = meta.iloc[: max_rows - offset].copy()
                init = mixed_datetime(meta["init_time"])
                init_ns = init.astype("int64", copy=False).to_numpy(dtype=np.int64)
                target_ns = init_ns[:, None] + lead_delta_ns[None, :]
                time_pos = raw_times.get_indexer(
                    pd.DatetimeIndex(pd.to_datetime(target_ns.reshape(-1), utc=True)),
                    method="nearest",
                    tolerance=pd.Timedelta(minutes=31),
                ).reshape(len(meta), 48)
                station_pos = raw_stations.get_indexer(meta["station_id"].map(canonical_station_key))
                raw = np.full((len(meta), 48), np.nan, dtype=np.float32)
                row_idx, lead_idx = np.where((time_pos >= 0) & (station_pos[:, None] >= 0))
                if len(row_idx):
                    raw[row_idx, lead_idx] = raw_values[time_pos[row_idx, lead_idx], station_pos[row_idx]]
                raw_valid = np.isfinite(raw) & (raw >= 0.0) & (raw <= VISIBILITY_MAX_M)
                stop = offset + len(meta)
                mask = np.asarray(stored_mask[offset:stop], dtype=bool)
                vis = np.asarray(stored_vis[offset:stop], dtype=np.float32)
                mask_mismatch = raw_valid != mask
                both = raw_valid & mask
                value_mismatch = both & (~np.isclose(raw, vis, rtol=0.0, atol=1.0e-3, equal_nan=True))
                counts["rows"] += len(meta)
                counts["raw_valid"] += raw_valid.sum(axis=0)
                counts["stored_valid"] += mask.sum(axis=0)
                counts["mask_mismatch"] += mask_mismatch.sum(axis=0)
                counts["value_mismatch"] += value_mismatch.sum(axis=0)
                if examples < 100:
                    erow, elead = np.where(mask_mismatch | value_mismatch)
                    for row, lead_pos in zip(erow.tolist(), elead.tolist()):
                        state.rows["trajectory_raw_mismatch_examples"].append(
                            {
                                "dataset": trajectory_tag,
                                "raw_source": raw_tag,
                                "split": split,
                                "row": offset + row,
                                "init_time": str(init.iloc[row]),
                                "station_id": str(meta.iloc[row]["station_id"]),
                                "lead_hour": int(lead_pos + 1),
                                "raw_visibility_m": float(raw[row, lead_pos]) if np.isfinite(raw[row, lead_pos]) else math.nan,
                                "stored_visibility_m": float(vis[row, lead_pos]) if np.isfinite(vis[row, lead_pos]) else math.nan,
                                "raw_valid": bool(raw_valid[row, lead_pos]),
                                "stored_valid": bool(mask[row, lead_pos]),
                            }
                        )
                        examples += 1
                        if examples >= 100:
                            break
                offset = stop
            rows = int(counts["rows"])
            for pos, lead in enumerate(leads):
                state.rows["trajectory_raw_consistency"].append(
                    {
                        "dataset": trajectory_tag,
                        "raw_source": raw_tag,
                        "split": split,
                        "lead_hour": int(lead),
                        "rows_compared": rows,
                        "raw_valid": int(counts["raw_valid"][pos]),
                        "raw_valid_rate": int(counts["raw_valid"][pos]) / max(rows, 1),
                        "stored_valid": int(counts["stored_valid"][pos]),
                        "stored_valid_rate": int(counts["stored_valid"][pos]) / max(rows, 1),
                        "mask_mismatch": int(counts["mask_mismatch"][pos]),
                        "value_mismatch": int(counts["value_mismatch"][pos]),
                    }
                )
            mismatch_total = int(np.asarray(counts["mask_mismatch"]).sum())
            value_total = int(np.asarray(counts["value_mismatch"]).sum())
            if mismatch_total or value_total:
                state.issue(
                    "ERROR",
                    "trajectory_raw_stored_mismatch",
                    f"mask={mismatch_total}, values={value_total}, rows={rows}",
                    trajectory_tag,
                    split,
                )
    finally:
        ds.close()


def write_outputs(state: AuditState, out_dir: Path, args: argparse.Namespace, repo_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Dict[str, object]] = []
    for name, rows in sorted(state.rows.items()):
        path = out_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        files.append(file_record(path))
    issues_path = out_dir / "issues.csv"
    pd.DataFrame(state.issues, columns=["severity", "code", "dataset", "split", "message"]).to_csv(
        issues_path, index=False
    )
    files.append(file_record(issues_path))
    counts = collections.Counter(item["severity"] for item in state.issues)
    status = "FAILED" if counts["ERROR"] else ("WARN" if counts["WARN"] else "PASS")
    summary = {
        "status": status,
        "issue_counts": dict(sorted(counts.items())),
        "pm_unit_policy": EXPECTED_PM_UNIT_POLICY,
        "pm_qc_policy": EXPECTED_PM_QC_POLICY,
        "visibility_valid_range_m": [0.0, VISIBILITY_MAX_M],
        "comparison_leads": list(COMPARISON_LEADS),
        "dataset_specs": str(args.dataset_specs),
        "raw_visibility_specs": str(args.raw_visibility_specs),
        "raw_pm_specs": str(args.raw_pm_specs),
        "trajectory_raw_tag": str(args.trajectory_raw_tag),
        "git_commit": git_commit(repo_dir),
        "command": " ".join(sys.argv),
    }
    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(file_record(summary_path))
    manifest = {
        "summary": summary,
        "files": files,
        "source_inventory": [
            file_record(path)
            for specs in (parse_specs(args.dataset_specs), parse_specs(args.raw_visibility_specs), parse_specs(args.raw_pm_specs))
            for path in specs.values()
            if path.is_file()
        ],
    }
    manifest_path = out_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[OUTPUT] {out_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-specs", required=True, help="Semicolon-separated tag=/dataset/path entries")
    parser.add_argument("--raw-visibility-specs", default="", help="Semicolon-separated tag=/file.nc entries")
    parser.add_argument("--raw-pm-specs", default="", help="Semicolon-separated tag=/file.nc entries")
    parser.add_argument("--trajectory-raw-tag", default="tianji_s2")
    parser.add_argument("--pmst-common-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--quantile-rows", type=int, default=5000)
    parser.add_argument("--metadata-chunksize", type=int, default=20000)
    parser.add_argument("--netcdf-time-chunk", type=int, default=168)
    parser.add_argument("--max-trajectory-raw-rows", type=int, default=0)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 2 after writing reports when ERROR issues exist")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    state = AuditState()
    policy = load_pm_policy(Path(args.pmst_common_dir))
    datasets = parse_specs(args.dataset_specs)
    raw_visibility = parse_specs(args.raw_visibility_specs)
    raw_pm = parse_specs(args.raw_pm_specs)

    for tag, path in datasets.items():
        print(f"[DATASET] {tag}={path}", flush=True)
        if not path.is_dir():
            severity = "ERROR" if args.require_all else "WARN"
            state.issue(severity, "dataset_missing", str(path), tag)
            continue
        try:
            if (path / "dynamic_train.npy").is_file() or (path / "dynamic_test.npy").is_file():
                audit_trajectory_dataset(
                    state, tag, path, policy, args.chunk_rows, args.quantile_rows, args.metadata_chunksize
                )
            else:
                audit_flat_dataset(
                    state, tag, path, policy, args.chunk_rows, args.quantile_rows, args.metadata_chunksize
                )
        except Exception as exc:
            state.issue("ERROR", "dataset_audit_exception", repr(exc), tag)

    for tag, path in raw_visibility.items():
        print(f"[RAW VISIBILITY] {tag}={path}", flush=True)
        if not path.is_file():
            severity = "ERROR" if args.require_all else "WARN"
            state.issue(severity, "raw_visibility_missing", str(path), tag)
            continue
        try:
            audit_raw_visibility(state, tag, path, args.netcdf_time_chunk)
        except Exception as exc:
            state.issue("ERROR", "raw_visibility_audit_exception", repr(exc), tag)

    for tag, path in raw_pm.items():
        print(f"[RAW PM] {tag}={path}", flush=True)
        if not path.is_file():
            severity = "ERROR" if args.require_all else "WARN"
            state.issue(severity, "raw_pm_missing", str(path), tag)
            continue
        try:
            audit_raw_pm(state, tag, path, policy, args.netcdf_time_chunk)
        except Exception as exc:
            state.issue("ERROR", "raw_pm_audit_exception", repr(exc), tag)

    raw_tag = str(args.trajectory_raw_tag)
    if raw_tag in raw_visibility and raw_visibility[raw_tag].is_file():
        for tag, path in datasets.items():
            if path.is_dir() and ((path / "dynamic_train.npy").is_file() or (path / "dynamic_test.npy").is_file()):
                try:
                    audit_trajectory_raw_consistency(
                        state,
                        tag,
                        path,
                        raw_tag,
                        raw_visibility[raw_tag],
                        args.metadata_chunksize,
                        args.max_trajectory_raw_rows,
                    )
                except Exception as exc:
                    state.issue("ERROR", "trajectory_raw_consistency_exception", repr(exc), tag)
    else:
        state.issue("WARN", "trajectory_raw_source_unavailable", raw_tag)

    summary = write_outputs(state, out_dir, args, repo_dir)
    return 2 if args.strict and summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
