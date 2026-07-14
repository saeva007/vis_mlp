#!/usr/bin/env python3
"""Build the candidate 0-48 h condition / 12-48 h visibility trajectory dataset."""

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
    DYNAMIC_FEATURE_ORDER,
    MAX_VISIBILITY_M,
    PM_QC_POLICY_VERSION,
    PM_UNIT_POLICY_VERSION,
    TARGET_LEADS,
    exact_lead_indices,
    full_trajectory_split,
    time_features_from_init,
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
            os.path.join(BASE_PATH, "ml_dataset_s2_tianji_trajectory_0_48h_pm10_pm25_v1"),
        ),
    )
    p.add_argument("--pmst-common-dir", default=os.environ.get("PMST_COMMON_DIR", ""))
    p.add_argument("--min-valid-targets", type=int, default=30)
    p.add_argument("--limit-runs", type=int, default=0)
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


def station_pm_raw_grid(pm_da: xr.DataArray, times: pd.DatetimeIndex, stations: np.ndarray) -> np.ndarray:
    if pm_da is None:
        raise RuntimeError("PM source is required for the 27-variable trajectory contract")
    time_index = pd.DatetimeIndex(pd.to_datetime(pm_da.time.values))
    sid_index = pd.Index(pm_da.station_id.values)
    time_pos = time_index.get_indexer(times, method="nearest")
    sid_pos = sid_index.get_indexer(stations)
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
        out[np.ix_(valid_t, valid_s)] = raw[np.ix_(time_pos[valid_t], sid_pos[valid_s])]
    return out


def canonical_pm_grid(policy, pm_da: xr.DataArray, times: pd.DatetimeIndex, stations: np.ndarray) -> np.ndarray:
    raw = station_pm_raw_grid(pm_da, times, stations)
    units = str(pm_da.attrs.get("units", ""))
    canonical = policy.canonicalize_pm_concentration(raw, units)
    valid = np.isfinite(canonical) & (canonical >= 0.0) & (canonical <= policy.PM_CONCENTRATION_MAX_UGM3)
    return np.where(valid, canonical, np.nan).astype(np.float32)


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
        raise ValueError("--min-valid-targets must be in [1, 37]")
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
    vis_da = ds_vis["visibility"]
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
        "runs_processed": 0,
        "runs_missing_leads": 0,
        "runs_crossing_split": 0,
        "runs_failed": 0,
        "trajectories_low_coverage": 0,
    }
    try:
        for run_str in tqdm(runs, desc="trajectory runs"):
            ds_run, init_time = load_merged_run_ds(run_str, data_veg, data_oro)
            if ds_run is None:
                audit["runs_failed"] += 1
                continue
            try:
                indices = exact_lead_indices(ds_run["lead_time"].values)
                if indices is None:
                    audit["runs_missing_leads"] += 1
                    continue
                ds = ds_run.isel(time=indices)
                times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
                target_times = times[np.asarray(TARGET_LEADS, dtype=int)]
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
                zenith = calculate_zenith_angle(lats, lons, times).transpose(1, 0, 2).astype(np.float32)
                pm10 = canonical_pm_grid(policy, pm10_da, times, stations).T[..., None]
                pm25 = canonical_pm_grid(policy, pm25_da, times, stations).T[..., None]
                dynamic = np.concatenate([met, zenith, pm10, pm25], axis=-1).astype(np.float32)
                if dynamic.shape[1:] != (len(CONDITION_LEADS), len(DYNAMIC_FEATURE_ORDER)):
                    raise RuntimeError(f"Unexpected dynamic shape {dynamic.shape}")

                veg_raw = get_nearest_veg(lats, lons, data_veg)
                veg_map = {value: i for i, value in enumerate(UNIQUE_VEG_IDS)}
                veg = np.asarray([veg_map.get(value, 0) for value in veg_raw], dtype=np.int16)
                terrain = extract_terrain(lats, lons, data_oro).astype(np.float32)
                static = np.concatenate([lats[:, None] / 90.0, lons[:, None] / 180.0, terrain], axis=1).astype(np.float32)
                init_features = np.repeat(time_features_from_init([init_time]), len(stations), axis=0)

                vis = vis_da.sel(time=target_times, method="nearest").reindex(station_id=stations)
                visibility = np.asarray(vis.values, dtype=np.float32).T
                valid = np.isfinite(visibility) & (visibility >= 0.0) & (visibility <= MAX_VISIBILITY_M)
                visibility = np.where(valid, visibility, np.nan).astype(np.float32)
                keep = valid.sum(axis=1) >= int(args.min_valid_targets)
                audit["trajectories_low_coverage"] += int((~keep).sum())
                if not keep.any():
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
            except Exception as exc:
                audit["runs_failed"] += 1
                print(f"[WARN] run {run_str} failed: {exc}", flush=True)
            finally:
                ds_run.close()
                gc.collect()
    finally:
        writer.close()
        data_veg.close()
        data_oro.close()
        ds_vis.close()

    if writer.counts["train"] == 0 or writer.counts["val"] == 0 or writer.counts["test"] == 0:
        raise RuntimeError(f"Every split must be non-empty; counts={writer.counts}")
    forecast_files = [
        str(Path(args.current_48h_dir) / f"{variable}_{run}_0-48h_IDW.nc")
        for run in runs
        for variable in legacy.VARIABLES_48H
    ]
    config = {
        "builder": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "condition_leads": list(CONDITION_LEADS),
        "target_leads": list(TARGET_LEADS),
        "dynamic_feature_order": list(DYNAMIC_FEATURE_ORDER),
        "static_continuous_order": ["lat_norm", "lon_norm", "orography", "orography_anom", "orography_std"],
        "time_feature_order": ["init_hour_sin", "init_hour_cos", "init_doy_sin", "init_doy_cos"],
        "visibility_valid_range_m": [0.0, MAX_VISIBILITY_M],
        "min_valid_targets": int(args.min_valid_targets),
        "split": {
            "type": "monthly_tail_full_trajectory_containment",
            "val_last_days": VAL_LAST_DAYS,
            "test_last_days": TEST_LAST_DAYS,
            "gap_hours": GAP_HOURS,
        },
        "canonical_unit_policy": policy.CANONICAL_UNIT_POLICY_VERSION,
        "pm_qc_policy": policy.PM_QC_POLICY_VERSION,
        "pm_max_valid_ugm3": float(policy.PM_CONCENTRATION_MAX_UGM3),
        "pm_policy_source": str(policy_dir / "pmst_overlap_common.py"),
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
        "source_manifest_sha256": sha256_file_manifest(
            forecast_files + [args.visibility_nc, args.pm10_file, args.pm25_file, args.veg_file, args.oro_file]
        ),
    }
    (out_dir / "dataset_build_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "sample_counts": writer.counts, "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
