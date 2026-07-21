#!/usr/bin/env python3
"""Build the candidate 1-48 h condition / 1-48 h visibility trajectory dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from lowvis_trajectory_contract import (
    CONDITION_LEADS,
    COMPARISON_LEADS,
    COMPARISON_TARGET_POSITIONS,
    DYNAMIC_FEATURE_ORDER,
    MAX_VISIBILITY_M,
    PM_QC_POLICY_VERSION,
    PM_UNIT_POLICY_VERSION,
    TARGET_LEADS,
    TARGET_CONDITION_POSITIONS,
    canonical_station_key,
    datetime_index_from_values,
    full_trajectory_split,
    shifted_lead_indices,
    time_features_from_init,
    visibility_grid,
)
from PMST_s2_data_48h_pm10 import (
    AppendableNpyWriter,
    BASE_PATH,
    CURRENT_48H_DIR,
    ORO_FILE,
    PM10_DIR,
    PM10_S2_FILE,
    PM25_DIR,
    PM25_S2_FILE,
    VEG_FILE,
    VIS_SOURCE_NC,
    get_run_list_from_current_48h,
    load_merged_run_ds,
    load_station_pm_dataarray,
)
from PMST_s2_data import FINAL_FEATURE_ORDER, UNIQUE_VEG_IDS, calculate_zenith_angle, extract_terrain, get_nearest_veg
from s2_data_aerosol import GAP_HOURS, TEST_LAST_DAYS, VAL_LAST_DAYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=os.environ.get("LOWVIS_TRAJ_BASE", BASE_PATH))
    p.add_argument("--current-48h-dir", default=os.environ.get("CURRENT_48H_DIR", CURRENT_48H_DIR))
    p.add_argument("--visibility-nc", default=os.environ.get("VIS_SOURCE_NC", VIS_SOURCE_NC))
    p.add_argument("--pm10-file", default=os.environ.get("PM10_S2_FILE", PM10_S2_FILE))
    p.add_argument("--pm25-file", default=os.environ.get("PM25_S2_FILE", PM25_S2_FILE))
    p.add_argument("--pm10-dir", default=os.environ.get("PM10_DIR", PM10_DIR))
    p.add_argument("--pm25-dir", default=os.environ.get("PM25_DIR", PM25_DIR))
    p.add_argument("--veg-file", default=os.environ.get("VEG_FILE", VEG_FILE))
    p.add_argument("--oro-file", default=os.environ.get("ORO_FILE", ORO_FILE))
    p.add_argument(
        "--out-dir",
        default=os.environ.get(
            "LOWVIS_TRAJ_DATA_DIR",
            os.path.join(BASE_PATH, "ml_dataset_s2_tianji_trajectory_1_48h_dyn27_initpm_v2"),
        ),
    )
    p.add_argument("--pmst-common-dir", default=os.environ.get("PMST_COMMON_DIR", ""))
    p.add_argument("--min-valid-targets", type=int, default=39)
    p.add_argument("--min-valid-comparison-targets", type=int, default=30)
    p.add_argument("--pm-min-finite-fraction", type=float, default=0.95)
    p.add_argument(
        "--pm-alignment-mode",
        choices=("init_snapshot_repeat",),
        default="init_snapshot_repeat",
        help=(
            "Leakage-safe PM contract. Current station files lost CAMS forecast_reference_time, "
            "so only the PM valid at initialization may be used and is repeated over 1-48 h."
        ),
    )
    p.add_argument("--limit-runs", type=int, default=0)
    p.add_argument(
        "--forecast-time-shift-hours",
        default=os.environ.get("TIANJI_INPUT_TIME_SHIFT_HOURS", "auto"),
        help="Shift raw forecast valid times before 1-48 h lead/target alignment; use auto, 0, -8, or 8.",
    )
    p.add_argument("--target-time-tolerance-minutes", type=float, default=31.0)
    p.add_argument(
        "--empty-run-fail-fast",
        type=int,
        default=24,
        help="Stop after this many attempted runs when a systemic alignment error still yields zero samples; 0 disables.",
    )
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def load_pm_policy(args: argparse.Namespace):
    candidates = []
    if args.pmst_common_dir:
        candidates.append(Path(args.pmst_common_dir))
    candidates.extend(
        [Path(args.base) / "ifs_baseline", Path(__file__).resolve().parents[1] / "ifs_baseline"]
    )
    module_dir = next((path for path in candidates if (path / "pmst_overlap_common.py").is_file()), None)
    if module_dir is None:
        raise FileNotFoundError(
            "pmst_overlap_common.py is required for canonical PM units; searched: "
            + ", ".join(str(v) for v in candidates)
        )
    sys.path.insert(0, str(module_dir))
    import pmst_overlap_common as policy  # type: ignore

    if policy.CANONICAL_UNIT_POLICY_VERSION != PM_UNIT_POLICY_VERSION:
        raise RuntimeError(
            f"Canonical unit policy mismatch: {policy.CANONICAL_UNIT_POLICY_VERSION!r} != {PM_UNIT_POLICY_VERSION!r}"
        )
    if policy.PM_QC_POLICY_VERSION != PM_QC_POLICY_VERSION:
        raise RuntimeError(f"PM QC policy mismatch: {policy.PM_QC_POLICY_VERSION!r} != {PM_QC_POLICY_VERSION!r}")
    return policy, module_dir


def station_pm_raw_grid(
    pm_da: xr.DataArray,
    times: pd.DatetimeIndex,
    stations: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if pm_da is None:
        raise RuntimeError("PM source is required for the 27-variable trajectory contract")
    time_index = datetime_index_from_values(pm_da.time.values)
    if time_index.has_duplicates:
        raise ValueError("PM valid_time contains duplicates; forecast-reference provenance is ambiguous")
    sid_index = pd.Index([canonical_station_key(v) for v in pm_da.station_id.values])
    if sid_index.has_duplicates:
        duplicates = sid_index[sid_index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"PM station ids are duplicated after normalization: {duplicates}")
    time_pos = time_index.get_indexer(times, method="nearest")
    sid_pos = sid_index.get_indexer([canonical_station_key(v) for v in stations])
    nt, ns = len(times), len(stations)
    out = np.full((nt, ns), np.nan, dtype=np.float32)
    if np.any(time_pos < 0) or np.any(sid_pos < 0):
        valid_t = time_pos >= 0
        valid_s = sid_pos >= 0
    else:
        valid_t = np.ones(nt, dtype=bool)
        valid_s = np.ones(ns, dtype=bool)
    if valid_t.any() and valid_s.any():
        delta = np.abs(time_index[time_pos[valid_t]].asi8 - times[valid_t].asi8)
        valid_t_positions = np.where(valid_t)[0]
        valid_t[valid_t_positions[delta > pd.Timedelta("31min").value]] = False
        raw = np.asarray(pm_da.values, dtype=np.float32)
        target_rows = np.flatnonzero(valid_t)
        target_cols = np.flatnonzero(valid_s)
        out[np.ix_(target_rows, target_cols)] = raw[np.ix_(time_pos[valid_t], sid_pos[valid_s])]
    diagnostics = {
        "matched_times": int(valid_t.sum()),
        "requested_times": int(nt),
        "matched_stations": int(valid_s.sum()),
        "requested_stations": int(ns),
    }
    return out, diagnostics


def canonical_pm_grid(
    policy,
    pm_da: xr.DataArray,
    times: pd.DatetimeIndex,
    stations: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw, diagnostics = station_pm_raw_grid(pm_da, times, stations)
    units = str(pm_da.attrs.get("units", ""))
    canonical = policy.canonicalize_pm_concentration(raw, units)
    valid = np.isfinite(canonical) & (canonical >= 0.0) & (canonical <= policy.PM_CONCENTRATION_MAX_UGM3)
    result = np.where(valid, canonical, np.nan).astype(np.float32)
    diagnostics = {
        **diagnostics,
        "canonical_finite_values": int(valid.sum()),
        "canonical_total_values": int(valid.size),
        "canonical_finite_fraction": float(valid.mean()) if valid.size else 0.0,
    }
    return result, diagnostics


def canonicalize_meteorology(policy, values: np.ndarray) -> np.ndarray:
    """Apply the shared canonical-unit policy to every meteorological channel."""

    out = np.asarray(values, dtype=np.float32).copy()
    for feature_idx, name in enumerate(FINAL_FEATURE_ORDER):
        out[..., feature_idx] = policy.canonicalize_pmst_field(name, out[..., feature_idx])
    return out


def repeat_pm_snapshot(snapshot: np.ndarray) -> np.ndarray:
    values = np.asarray(snapshot, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"PM initialization snapshot must have shape [1, station], got {values.shape}")
    return np.repeat(values.T[:, :, None], len(CONDITION_LEADS), axis=1)


def expected_valid_times(init_time: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [pd.Timestamp(init_time) + pd.Timedelta(hours=int(lead)) for lead in CONDITION_LEADS]
    )


def validate_forecast_valid_times(
    selected_times: pd.DatetimeIndex,
    expected_times: pd.DatetimeIndex,
    tolerance_minutes: float,
) -> float:
    if len(selected_times) != len(expected_times):
        raise ValueError(f"selected/expected valid-time length mismatch: {len(selected_times)} != {len(expected_times)}")
    delta_minutes = np.abs(selected_times.asi8 - expected_times.asi8) / float(pd.Timedelta(minutes=1).value)
    max_error = float(delta_minutes.max()) if len(delta_minutes) else 0.0
    if max_error > float(tolerance_minutes):
        raise ValueError(
            f"forecast valid times disagree with init+lead: max_error={max_error:.3f} min, "
            f"tolerance={float(tolerance_minutes):.3f} min"
        )
    return max_error


def trajectory_split(target_times: pd.DatetimeIndex) -> Optional[str]:
    return full_trajectory_split(target_times, GAP_HOURS, VAL_LAST_DAYS, TEST_LAST_DAYS)


class TrajectoryWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        shapes = {
            "dynamic": (len(CONDITION_LEADS), len(DYNAMIC_FEATURE_ORDER)),
            "static_cont": (5,),
            "veg_id": (),
            "time_features": (4,),
            "visibility": (len(TARGET_LEADS),),
            "target_mask": (len(TARGET_LEADS),),
        }
        dtypes = {
            "dynamic": "float32",
            "static_cont": "float32",
            "veg_id": "int16",
            "time_features": "float32",
            "visibility": "float32",
            "target_mask": "bool",
        }
        self.writers: Dict[str, Dict[str, AppendableNpyWriter]] = {}
        self.counts = {tag: 0 for tag in ("train", "val", "test")}
        for tag in self.counts:
            self.writers[tag] = {
                name: AppendableNpyWriter(out_dir / f"{name}_{tag}.npy", dtypes[name], shape)
                for name, shape in shapes.items()
            }
            pd.DataFrame(columns=["init_time", "station_id", "lat", "lon", "target_start", "target_end"]).to_csv(
                out_dir / f"meta_{tag}.csv", index=False
            )

    def write(self, tag: str, arrays: Mapping[str, np.ndarray], meta: pd.DataFrame) -> None:
        n = int(len(meta))
        if n == 0:
            return
        for name, writer in self.writers[tag].items():
            writer.write(arrays[name])
        meta.to_csv(self.out_dir / f"meta_{tag}.csv", mode="a", header=False, index=False)
        self.counts[tag] += n

    def close(self) -> None:
        for group in self.writers.values():
            for writer in group.values():
                writer.close()


def ensure_fresh_output(out_dir: Path, allow_overwrite: bool) -> None:
    tracked = list(out_dir.glob("*.npy")) + list(out_dir.glob("meta_*.csv"))
    tracked += [path for path in (out_dir / "pm_provenance.csv",) if path.exists()]
    if tracked and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing trajectory dataset files in {out_dir}")
    if allow_overwrite:
        for path in tracked + [out_dir / "dataset_build_config.json"]:
            if path.exists() and path.is_file():
                path.unlink()


def sha256_file_manifest(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in paths):
        path = Path(value)
        stat = path.stat() if path.exists() else None
        digest.update(f"{path}|{getattr(stat, 'st_size', -1)}|{getattr(stat, 'st_mtime_ns', -1)}\n".encode())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_valid_targets <= len(TARGET_LEADS)):
        raise ValueError(f"--min-valid-targets must be in [1, {len(TARGET_LEADS)}]")
    if not (0 < args.min_valid_comparison_targets <= len(COMPARISON_LEADS)):
        raise ValueError(
            f"--min-valid-comparison-targets must be in [1, {len(COMPARISON_LEADS)}]"
        )
    if not (0.0 < args.pm_min_finite_fraction <= 1.0):
        raise ValueError("--pm-min-finite-fraction must be in (0, 1]")
    if args.target_time_tolerance_minutes <= 0:
        raise ValueError("--target-time-tolerance-minutes must be positive")
    if args.empty_run_fail_fast < 0:
        raise ValueError("--empty-run-fail-fast must be non-negative")
    if str(args.forecast_time_shift_hours).strip().lower() != "auto":
        float(args.forecast_time_shift_hours)
    out_dir = Path(args.out_dir)
    ensure_fresh_output(out_dir, args.allow_overwrite)
    policy, policy_dir = load_pm_policy(args)

    # Override imported module globals used by load_merged_run_ds/get_run_list.
    import PMST_s2_data_48h_pm10 as legacy

    legacy.BASE_PATH = args.base
    legacy.CURRENT_48H_DIR = args.current_48h_dir
    legacy.VIS_SOURCE_NC = args.visibility_nc

    if not Path(args.visibility_nc).is_file():
        raise FileNotFoundError(f"Visibility source is required: {args.visibility_nc}")
    for path, label in ((args.pm10_file, "PM10"), (args.pm25_file, "PM2.5"), (args.veg_file, "vegetation"), (args.oro_file, "orography")):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} source is required: {path}")

    data_veg = xr.open_dataset(args.veg_file, engine="h5netcdf")
    data_oro = xr.open_dataset(args.oro_file, engine="h5netcdf")
    ds_vis = xr.open_dataset(args.visibility_nc, engine="h5netcdf")
    if "vis" in ds_vis and "visibility" not in ds_vis:
        ds_vis = ds_vis.rename({"vis": "visibility"})
    if "station_id" not in ds_vis.dims and "station_id" not in ds_vis.coords:
        station_alias = next(
            (name for name in ("num_station", "station", "id") if name in ds_vis.dims or name in ds_vis.coords),
            None,
        )
        if station_alias is not None:
            ds_vis = ds_vis.rename({station_alias: "station_id"})
    vis_da = ds_vis["visibility"]
    print(
        "[SOURCE] visibility "
        f"dims={dict(vis_da.sizes)} "
        f"time=[{pd.Timestamp(vis_da.time.values[0])}, {pd.Timestamp(vis_da.time.values[-1])}]",
        flush=True,
    )
    pm10_da = load_station_pm_dataarray(args.pm10_file, args.pm10_dir, ("pm10", "PM10"), "PM10")
    pm25_da = load_station_pm_dataarray(args.pm25_file, args.pm25_dir, ("pm2p5", "pm25", "pm2_5", "PM2_5"), "PM2.5")
    if pm10_da is None or pm25_da is None:
        raise RuntimeError("Both PM10 and PM2.5 sources are required; zero-filled fallback is forbidden")

    runs = get_run_list_from_current_48h()
    if args.limit_runs > 0:
        runs = runs[: args.limit_runs]
    writer = TrajectoryWriter(out_dir)
    audit = {
        "runs_found": len(runs),
        "runs_attempted": 0,
        "runs_processed": 0,
        "runs_missing_leads": 0,
        "runs_crossing_split": 0,
        "runs_failed": 0,
        "runs_no_visibility_time_match": 0,
        "runs_no_visibility_station_match": 0,
        "runs_pm_alignment_failed": 0,
        "runs_no_trajectory_kept": 0,
        "trajectories_seen": 0,
        "trajectories_kept": 0,
        "trajectories_low_coverage": 0,
        "trajectories_low_comparison_coverage": 0,
        "visibility_valid_targets_min": None,
        "visibility_valid_targets_max": None,
        "forecast_valid_time_max_error_minutes": 0.0,
        "pm10_canonical_finite_values": 0,
        "pm10_canonical_total_values": 0,
        "pm25_canonical_finite_values": 0,
        "pm25_canonical_total_values": 0,
        "forecast_time_shift_counts": {},
        "early_stop_reason": None,
    }
    pm_provenance_rows = []
    try:
        for run_str in tqdm(runs, desc="trajectory runs"):
            audit["runs_attempted"] += 1
            ds_run, init_time = load_merged_run_ds(run_str, data_veg, data_oro)
            if ds_run is None:
                audit["runs_failed"] += 1
                continue
            pm_alignment_started = False
            pm_alignment_complete = False
            try:
                indices, time_shift_hours = shifted_lead_indices(
                    ds_run["lead_time"].values,
                    args.forecast_time_shift_hours,
                )
                if indices is None:
                    audit["runs_missing_leads"] += 1
                    lead_values = np.asarray(ds_run["lead_time"].values, dtype=float)
                    if audit["runs_missing_leads"] == 1:
                        print(
                            f"[ALIGN] run={run_str} cannot resolve 1-48 h leads: "
                            f"raw_range=[{np.nanmin(lead_values):.3f}, {np.nanmax(lead_values):.3f}], "
                            f"requested_shift={args.forecast_time_shift_hours!r}",
                            flush=True,
                        )
                    continue
                shift_key = f"{float(time_shift_hours):g}"
                shift_counts = audit["forecast_time_shift_counts"]
                shift_counts[shift_key] = int(shift_counts.get(shift_key, 0)) + 1
                ds = ds_run.isel(time=indices)
                selected_times = datetime_index_from_values(ds.time.values) + pd.Timedelta(
                    hours=float(time_shift_hours)
                )
                times = expected_valid_times(pd.Timestamp(init_time))
                time_error = validate_forecast_valid_times(
                    selected_times,
                    times,
                    args.target_time_tolerance_minutes,
                )
                audit["forecast_valid_time_max_error_minutes"] = max(
                    float(audit["forecast_valid_time_max_error_minutes"]),
                    float(time_error),
                )
                target_times = times[np.asarray(TARGET_CONDITION_POSITIONS, dtype=int)]
                split = trajectory_split(target_times)
                if split is None:
                    audit["runs_crossing_split"] += 1
                    continue
                stations = np.asarray(ds.station_id.values)
                lats = np.asarray(ds.lat.values, dtype=np.float32)
                lons = np.asarray(ds.lon.values, dtype=np.float32)
                missing = [name for name in FINAL_FEATURE_ORDER if name not in ds]
                if missing:
                    raise RuntimeError(f"Missing dynamic fields: {missing}")
                met = ds[FINAL_FEATURE_ORDER].to_array(dim="feature").transpose("station_id", "time", "feature").values.astype(np.float32)
                met = canonicalize_meteorology(policy, met)
                zenith = calculate_zenith_angle(lats, lons, times).transpose(1, 0, 2).astype(np.float32)
                pm_alignment_started = True
                pm_query_times = pd.DatetimeIndex([pd.Timestamp(init_time)])
                pm10_snapshot, pm10_diagnostics = canonical_pm_grid(policy, pm10_da, pm_query_times, stations)
                pm25_snapshot, pm25_diagnostics = canonical_pm_grid(policy, pm25_da, pm_query_times, stations)
                pm_min_fraction = float(args.pm_min_finite_fraction)
                for label, diagnostics in (("PM10", pm10_diagnostics), ("PM2.5", pm25_diagnostics)):
                    if int(diagnostics["matched_times"]) != 1:
                        raise RuntimeError(
                            f"{label} initialization-time match failed: {diagnostics}"
                        )
                    if float(diagnostics["canonical_finite_fraction"]) < pm_min_fraction:
                        raise RuntimeError(
                            f"{label} canonical coverage {float(diagnostics['canonical_finite_fraction']):.3%} "
                            f"is below required {pm_min_fraction:.3%}: {diagnostics}"
                        )
                audit["pm10_canonical_finite_values"] += int(pm10_diagnostics["canonical_finite_values"])
                audit["pm10_canonical_total_values"] += int(pm10_diagnostics["canonical_total_values"])
                audit["pm25_canonical_finite_values"] += int(pm25_diagnostics["canonical_finite_values"])
                audit["pm25_canonical_total_values"] += int(pm25_diagnostics["canonical_total_values"])
                pm_provenance_rows.append(
                    {
                        "init_time": pd.Timestamp(init_time),
                        "pm_query_valid_time": pm_query_times[0],
                        "pm_alignment_mode": args.pm_alignment_mode,
                        "pm_future_valid_time_used": False,
                        "pm10_matched_stations": int(pm10_diagnostics["matched_stations"]),
                        "pm10_requested_stations": int(pm10_diagnostics["requested_stations"]),
                        "pm10_finite_fraction": float(pm10_diagnostics["canonical_finite_fraction"]),
                        "pm25_matched_stations": int(pm25_diagnostics["matched_stations"]),
                        "pm25_requested_stations": int(pm25_diagnostics["requested_stations"]),
                        "pm25_finite_fraction": float(pm25_diagnostics["canonical_finite_fraction"]),
                    }
                )
                pm10 = repeat_pm_snapshot(pm10_snapshot)
                pm25 = repeat_pm_snapshot(pm25_snapshot)
                pm_alignment_complete = True
                dynamic = np.concatenate([met, zenith, pm10, pm25], axis=-1).astype(np.float32)
                if dynamic.shape[1:] != (len(CONDITION_LEADS), len(DYNAMIC_FEATURE_ORDER)):
                    raise RuntimeError(f"Unexpected dynamic shape {dynamic.shape}")

                veg_raw = get_nearest_veg(lats, lons, data_veg)
                veg_map = {value: i for i, value in enumerate(UNIQUE_VEG_IDS)}
                veg = np.asarray([veg_map.get(value, 0) for value in veg_raw], dtype=np.int16)
                terrain = extract_terrain(lats, lons, data_oro).astype(np.float32)
                static = np.concatenate([lats[:, None] / 90.0, lons[:, None] / 180.0, terrain], axis=1).astype(np.float32)
                init_features = np.repeat(time_features_from_init([init_time]), len(stations), axis=0)

                visibility, vis_diagnostics = visibility_grid(
                    vis_da,
                    target_times,
                    stations,
                    args.target_time_tolerance_minutes,
                )
                if vis_diagnostics["matched_target_times"] == 0:
                    audit["runs_no_visibility_time_match"] += 1
                if vis_diagnostics["matched_stations"] == 0:
                    audit["runs_no_visibility_station_match"] += 1
                audit["trajectories_seen"] += int(len(stations))
                valid = np.isfinite(visibility) & (visibility >= 0.0) & (visibility <= MAX_VISIBILITY_M)
                visibility = np.where(valid, visibility, np.nan).astype(np.float32)
                valid_counts = valid.sum(axis=1)
                comparison_valid_counts = valid[:, np.asarray(COMPARISON_TARGET_POSITIONS, dtype=int)].sum(axis=1)
                current_min = int(valid_counts.min()) if len(valid_counts) else 0
                current_max = int(valid_counts.max()) if len(valid_counts) else 0
                if audit["trajectories_seen"] == int(len(stations)):
                    print(
                        "[ALIGN] first split-eligible run "
                        f"run={run_str} split={split} shift_hours={float(time_shift_hours):g} "
                        f"matched_times={vis_diagnostics['matched_target_times']}/{vis_diagnostics['target_times']} "
                        f"matched_stations={vis_diagnostics['matched_stations']}/{vis_diagnostics['forecast_stations']} "
                        f"valid_targets_per_station=[{current_min}, {current_max}]",
                        flush=True,
                    )
                audit["visibility_valid_targets_min"] = (
                    current_min
                    if audit["visibility_valid_targets_min"] is None
                    else min(int(audit["visibility_valid_targets_min"]), current_min)
                )
                audit["visibility_valid_targets_max"] = (
                    current_max
                    if audit["visibility_valid_targets_max"] is None
                    else max(int(audit["visibility_valid_targets_max"]), current_max)
                )
                full_coverage_ok = valid_counts >= int(args.min_valid_targets)
                comparison_coverage_ok = comparison_valid_counts >= int(args.min_valid_comparison_targets)
                audit["trajectories_low_coverage"] += int((~full_coverage_ok).sum())
                audit["trajectories_low_comparison_coverage"] += int((~comparison_coverage_ok).sum())
                keep = full_coverage_ok & comparison_coverage_ok
                if not keep.any():
                    audit["runs_no_trajectory_kept"] += 1
                    continue
                meta = pd.DataFrame(
                    {
                        "init_time": np.repeat(pd.Timestamp(init_time), int(keep.sum())),
                        "station_id": stations[keep],
                        "lat": lats[keep],
                        "lon": lons[keep],
                        "target_start": np.repeat(target_times[0], int(keep.sum())),
                        "target_end": np.repeat(target_times[-1], int(keep.sum())),
                    }
                )
                first_retained_run = sum(writer.counts.values()) == 0
                writer.write(
                    split,
                    {
                        "dynamic": dynamic[keep],
                        "static_cont": static[keep],
                        "veg_id": veg[keep],
                        "time_features": init_features[keep],
                        "visibility": visibility[keep],
                        "target_mask": valid[keep],
                    },
                    meta,
                )
                audit["runs_processed"] += 1
                audit["trajectories_kept"] += int(keep.sum())
                if first_retained_run:
                    print(
                        f"[WRITE] first retained run={run_str} split={split} "
                        f"trajectories={int(keep.sum())} valid_target_range=[{current_min}, {current_max}]",
                        flush=True,
                    )
            except Exception as exc:
                audit["runs_failed"] += 1
                if pm_alignment_started and not pm_alignment_complete:
                    audit["runs_pm_alignment_failed"] += 1
                print(f"[WARN] run {run_str} failed: {exc}", flush=True)
            finally:
                ds_run.close()
                gc.collect()
                fail_fast = int(args.empty_run_fail_fast)
                if fail_fast > 0 and audit["runs_attempted"] >= fail_fast and sum(writer.counts.values()) == 0:
                    systemic = None
                    attempted = int(audit["runs_attempted"])
                    if int(audit["runs_missing_leads"]) == attempted:
                        systemic = "all attempted runs are missing an aligned 1-48 h lead trajectory"
                    elif int(audit["runs_no_visibility_station_match"]) >= fail_fast:
                        systemic = "no forecast station matches the visibility source"
                    elif int(audit["runs_no_visibility_time_match"]) >= fail_fast:
                        systemic = "no target time matches the visibility source within tolerance"
                    elif int(audit["runs_pm_alignment_failed"]) >= fail_fast:
                        systemic = "PM initialization snapshots cannot be aligned at the required coverage"
                    elif int(audit["runs_no_trajectory_kept"]) >= fail_fast:
                        systemic = "all aligned stations fail the minimum target-coverage rule"
                    if systemic is not None:
                        audit["early_stop_reason"] = systemic
                        print(f"[ALIGN][FAIL-FAST] {systemic}; audit={json.dumps(audit, sort_keys=True)}", flush=True)
                        break
    finally:
        writer.close()
        data_veg.close()
        data_oro.close()
        ds_vis.close()

    pm_provenance_columns = [
        "init_time",
        "pm_query_valid_time",
        "pm_alignment_mode",
        "pm_future_valid_time_used",
        "pm10_matched_stations",
        "pm10_requested_stations",
        "pm10_finite_fraction",
        "pm25_matched_stations",
        "pm25_requested_stations",
        "pm25_finite_fraction",
    ]
    pd.DataFrame(pm_provenance_rows, columns=pm_provenance_columns).to_csv(
        out_dir / "pm_provenance.csv", index=False
    )

    complete = all(writer.counts[tag] > 0 for tag in ("train", "val", "test"))
    forecast_files = [
        str(Path(args.current_48h_dir) / f"{variable}_{run}_0-48h_IDW.nc")
        for run in runs
        for variable in legacy.VARIABLES_48H
    ]
    config = {
        "builder": Path(__file__).name,
        "data_contract_version": "lowvis_trajectory_1_48h_v2_20260721",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "failed_empty_split",
        "candidate_only": True,
        "condition_length": len(CONDITION_LEADS),
        "condition_leads": list(CONDITION_LEADS),
        "target_leads": list(TARGET_LEADS),
        "comparison_leads": list(COMPARISON_LEADS),
        "dynamic_feature_order": list(DYNAMIC_FEATURE_ORDER),
        "static_continuous_order": ["lat_norm", "lon_norm", "orography", "orography_anom", "orography_std"],
        "time_feature_order": ["init_hour_sin", "init_hour_cos", "init_doy_sin", "init_doy_cos"],
        "visibility_valid_range_m": [0.0, MAX_VISIBILITY_M],
        "min_valid_targets": int(args.min_valid_targets),
        "min_valid_comparison_targets": int(args.min_valid_comparison_targets),
        "forecast_valid_time_definition": "init_time_plus_lead_hour",
        "visibility_alignment_contract": "source_time_pos_nearest_v2",
        "forecast_time_shift_hours_requested": str(args.forecast_time_shift_hours),
        "forecast_time_shift_counts": audit["forecast_time_shift_counts"],
        "target_time_tolerance_minutes": float(args.target_time_tolerance_minutes),
        "split": {
            "type": "monthly_tail_full_trajectory_containment",
            "val_last_days": VAL_LAST_DAYS,
            "test_last_days": TEST_LAST_DAYS,
            "gap_hours": GAP_HOURS,
        },
        "canonical_unit_policy": policy.CANONICAL_UNIT_POLICY_VERSION,
        "dynamic_units": {
            name: str(policy.CANONICAL_DYNAMIC_UNITS.get(name, "source_native"))
            for name in DYNAMIC_FEATURE_ORDER
        },
        "pm_qc_policy": policy.PM_QC_POLICY_VERSION,
        "pm_max_valid_ugm3": float(policy.PM_CONCENTRATION_MAX_UGM3),
        "pm_policy_source": str(policy_dir / "pmst_overlap_common.py"),
        "pm_alignment_mode": str(args.pm_alignment_mode),
        "pm_availability_policy": "valid_time_not_after_init_snapshot_repeat_v1",
        "pm_forecast_reference_time_available": False,
        "pm_future_valid_time_used": False,
        "pm_min_finite_fraction": float(args.pm_min_finite_fraction),
        "pm_provenance_file": "pm_provenance.csv",
        "sample_counts": writer.counts,
        "audit": audit,
        "sources": {
            "current_48h_dir": args.current_48h_dir,
            "visibility_nc": args.visibility_nc,
            "pm10_file": args.pm10_file,
            "pm25_file": args.pm25_file,
            "veg_file": args.veg_file,
            "orography_file": args.oro_file,
        },
        "source_manifest_sha256": (
            sha256_file_manifest(
                forecast_files + [args.visibility_nc, args.pm10_file, args.pm25_file, args.veg_file, args.oro_file]
            )
            if complete
            else None
        ),
    }
    (out_dir / "dataset_build_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "sample_counts": writer.counts, "audit": audit}, indent=2), flush=True)
    if not complete:
        raise RuntimeError(
            "Every split must be non-empty; "
            f"counts={writer.counts}; audit saved to {out_dir / 'dataset_build_config.json'}; audit={audit}"
        )


if __name__ == "__main__":
    main()
