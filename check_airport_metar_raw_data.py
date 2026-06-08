#!/usr/bin/env python3
"""Quality-control raw airport METAR visibility training inputs.

The script checks the two primary raw files used by
build_static_rnn_airport_metar_dataset.py:

  1. station weather inputs
  2. METAR visibility targets

It writes a compact Markdown report, a machine-readable JSON summary, and per
station/per feature CSV tables.  It is intentionally read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from airport_visibility_common import (
    DYNAMIC_FEATURE_ORDER,
    LOCAL_TIME_OFFSET_HOURS,
    MAX_VISIBILITY_M,
    WEATHER_ALIASES,
    extract_dynamic_cube,
    maybe_convert_visibility_to_meters,
)
from s2_data_airport_metar import (
    DEFAULT_ORO_FILE,
    DEFAULT_STATION_FILE,
    DEFAULT_VEG_FILE,
    DEFAULT_VISIBILITY_FILE,
    DEFAULT_WEATHER_FILE,
    calculate_zenith_angle,
    load_station_table,
    open_dataset_auto,
)


DERIVED_FEATURES = {
    "WSPD10": ("U10", "V10"),
    "WDIR10": ("U10", "V10"),
    "WSPD925": ("U_925", "V_925"),
    "DPD": ("T2M", "RH2M"),
    "INVERSION": ("T_925", "T2M"),
    "ZENITH_PROXY": ("time", "station_table"),
}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def add_issue(issues: List[Dict[str, str]], level: str, area: str, message: str) -> None:
    issues.append({"level": level, "area": area, "message": message})


def shift_time_if_needed(ds: xr.Dataset, hours: float) -> xr.Dataset:
    if abs(float(hours)) < 1e-9:
        return ds
    if "time" not in ds.coords:
        raise ValueError("Cannot shift time because the dataset has no time coordinate")
    return ds.assign_coords(time=ds.time + pd.Timedelta(hours=float(hours)))


def resolve_visibility_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return str(list(ds.data_vars)[0])
    raise KeyError(
        f"Visibility variable {preferred!r} not found; available variables: {list(ds.data_vars)}"
    )


def coord_values(ds: xr.Dataset, name: str) -> Optional[np.ndarray]:
    if name in ds.coords:
        return ds[name].values
    if name in ds.dims:
        return np.arange(ds.sizes[name])
    return None


def time_summary(ds: xr.Dataset, label: str, issues: List[Dict[str, str]]) -> Dict[str, object]:
    values = coord_values(ds, "time")
    if values is None:
        add_issue(issues, "error", label, "missing time coordinate")
        return {"present": False}
    times = pd.DatetimeIndex(pd.to_datetime(values))
    summary: Dict[str, object] = {
        "present": True,
        "count": int(len(times)),
        "start": str(times.min()) if len(times) else None,
        "end": str(times.max()) if len(times) else None,
        "duplicates": int(times.duplicated().sum()),
        "monotonic_increasing": bool(times.is_monotonic_increasing),
    }
    if len(times) >= 2:
        deltas = np.diff(times.values).astype("timedelta64[s]").astype(np.float64)
        finite = deltas[np.isfinite(deltas)]
        positive = finite[finite > 0]
        if positive.size:
            summary["median_step_hours"] = float(np.median(positive) / 3600.0)
            summary["min_step_hours"] = float(np.min(positive) / 3600.0)
            summary["max_step_hours"] = float(np.max(positive) / 3600.0)
        if np.any(finite <= 0):
            add_issue(issues, "warning", label, "time coordinate has non-positive steps")
    if summary["duplicates"]:
        add_issue(issues, "warning", label, f"time coordinate has {summary['duplicates']} duplicates")
    if not summary["monotonic_increasing"]:
        add_issue(issues, "warning", label, "time coordinate is not monotonic increasing")
    return summary


def station_summary(ds: xr.Dataset, label: str, issues: List[Dict[str, str]]) -> Dict[str, object]:
    values = coord_values(ds, "station")
    if values is None:
        add_issue(issues, "error", label, "missing station coordinate")
        return {"present": False}
    as_str = pd.Index([str(v) for v in values])
    return {
        "present": True,
        "count": int(len(as_str)),
        "duplicates": int(as_str.duplicated().sum()),
        "first5": [str(v) for v in as_str[:5]],
        "last5": [str(v) for v in as_str[-5:]],
    }


def finite_stats(values: np.ndarray, quantiles: Sequence[float] = (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> Dict[str, object]:
    arr = np.asarray(values)
    total = int(arr.size)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask].astype(np.float64)
    out: Dict[str, object] = {
        "count": total,
        "finite": int(finite.size),
        "missing": int(total - finite.size),
        "missing_frac": float((total - finite.size) / total) if total else None,
    }
    if finite.size == 0:
        return out
    out.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    qs = np.quantile(finite, quantiles)
    for q, val in zip(quantiles, qs):
        out[f"q{int(round(q * 10000)):04d}"] = float(val)
    return out


def variable_exists(ds: xr.Dataset, aliases: Iterable[str]) -> Tuple[bool, Optional[str]]:
    data_vars = set(str(v) for v in ds.data_vars)
    for name in aliases:
        if name in data_vars:
            return True, name
    if "variable" in ds.coords:
        var_names = set(str(v) for v in ds["variable"].values)
        for name in aliases:
            if name in var_names:
                return True, f"variable={name}"
    return False, None


def common_time_station(weather_ds: xr.Dataset, vis_ds: xr.Dataset) -> Tuple[pd.DatetimeIndex, List[object], Dict[str, object]]:
    weather_times = pd.DatetimeIndex(pd.to_datetime(weather_ds["time"].values))
    vis_times = pd.DatetimeIndex(pd.to_datetime(vis_ds["time"].values))
    vis_time_set = set(vis_times.values)
    common_times = pd.DatetimeIndex([t for t in weather_times if t.to_datetime64() in vis_time_set])

    weather_stations = [str(s) for s in weather_ds["station"].values]
    vis_station_set = {str(s) for s in vis_ds["station"].values}
    common_stations = [s for s in weather_ds["station"].values if str(s) in vis_station_set]
    common_station_str = {str(s) for s in common_stations}

    audit = {
        "weather_times": int(len(weather_times)),
        "visibility_times": int(len(vis_times)),
        "common_times": int(len(common_times)),
        "weather_only_times": int(len(set(weather_times.values) - set(vis_times.values))),
        "visibility_only_times": int(len(set(vis_times.values) - set(weather_times.values))),
        "weather_stations": int(len(weather_stations)),
        "visibility_stations": int(len(vis_station_set)),
        "common_stations": int(len(common_stations)),
        "weather_only_station_sample": [s for s in weather_stations if s not in vis_station_set][:20],
        "visibility_only_station_sample": [s for s in sorted(vis_station_set) if s not in common_station_str][:20],
    }
    return common_times, common_stations, audit


def classify_visibility(values_m: np.ndarray, max_visibility_m: float) -> Dict[str, object]:
    arr = np.asarray(values_m, dtype=np.float64)
    finite = np.isfinite(arr)
    valid = finite & (arr >= 0.0) & (arr <= float(max_visibility_m))
    cls = {
        "valid": int(valid.sum()),
        "invalid_or_missing": int(arr.size - valid.sum()),
        "fog_lt_500m": int((valid & (arr < 500.0)).sum()),
        "mist_500_1000m": int((valid & (arr >= 500.0) & (arr < 1000.0)).sum()),
        "clear_ge_1000m": int((valid & (arr >= 1000.0)).sum()),
    }
    if cls["valid"]:
        cls["fog_frac"] = float(cls["fog_lt_500m"] / cls["valid"])
        cls["mist_frac"] = float(cls["mist_500_1000m"] / cls["valid"])
        cls["clear_frac"] = float(cls["clear_ge_1000m"] / cls["valid"])
    return cls


def visibility_caps(values_m: np.ndarray) -> Dict[str, int]:
    arr = np.asarray(values_m, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    caps = {}
    for cap in (0, 50, 100, 200, 500, 800, 999, 1000, 5000, 9999, 10000, 30000, 99999):
        caps[str(cap)] = int(np.isclose(finite, float(cap), rtol=0.0, atol=0.01).sum())
    return caps


def check_visibility(
    vis_da: xr.DataArray,
    max_visibility_m: float,
    issues: List[Dict[str, str]],
    label: str,
) -> Dict[str, object]:
    raw = vis_da.values
    meters, unit_rule = maybe_convert_visibility_to_meters(raw)
    raw_stats = finite_stats(raw)
    meter_stats = finite_stats(meters)
    finite_m = meters[np.isfinite(meters)]
    if finite_m.size:
        if np.nanmax(finite_m) <= 100.0:
            add_issue(issues, "warning", label, "visibility max <= 100 after conversion; unit may be unexpected")
        if np.nanmin(finite_m) < 0.0:
            add_issue(issues, "error", label, "visibility contains negative values")
        too_large = int((finite_m > max_visibility_m).sum())
        if too_large:
            add_issue(issues, "warning", label, f"{too_large} visibility values exceed max_visibility_m={max_visibility_m}")
    return {
        "unit_rule": unit_rule,
        "raw_stats": raw_stats,
        "meter_stats": meter_stats,
        "class_counts": classify_visibility(meters, max_visibility_m),
        "cap_value_counts_m": visibility_caps(meters),
    }


def visibility_station_table(
    vis_da: xr.DataArray,
    max_visibility_m: float,
) -> pd.DataFrame:
    meters, _ = maybe_convert_visibility_to_meters(vis_da.values)
    station_dim = "station"
    arr = np.asarray(vis_da.transpose("time", station_dim).values)
    meters = np.asarray(meters)
    if meters.shape != arr.shape:
        meters = np.asarray(vis_da.copy(data=meters).transpose("time", station_dim).values)
    stations = [str(s) for s in vis_da[station_dim].values]
    rows = []
    for i, station in enumerate(stations):
        vals = meters[:, i].astype(np.float64)
        stats = finite_stats(vals)
        cls = classify_visibility(vals, max_visibility_m)
        rows.append(
            {
                "station": station,
                "count": stats.get("count"),
                "finite": stats.get("finite"),
                "missing_frac": stats.get("missing_frac"),
                "min_m": stats.get("min"),
                "q01_m": stats.get("q0100"),
                "median_m": stats.get("q5000"),
                "q99_m": stats.get("q9900"),
                "max_m": stats.get("max"),
                "fog_lt_500m": cls.get("fog_lt_500m"),
                "mist_500_1000m": cls.get("mist_500_1000m"),
                "clear_ge_1000m": cls.get("clear_ge_1000m"),
            }
        )
    return pd.DataFrame(rows)


def feature_range_issue(name: str, stats: Mapping[str, object]) -> Optional[str]:
    if stats.get("finite", 0) == 0:
        return "all values are missing or non-finite"
    mn = float(stats.get("min"))
    mx = float(stats.get("max"))
    med = float(stats.get("q5000"))
    if name.startswith("RH") and (mn < -1.0 or mx > 105.0):
        return f"relative humidity outside expected range: min={mn:.3g}, max={mx:.3g}"
    if name in {"T2M", "T_925"}:
        if med < 100.0:
            return f"temperature median={med:.3g}; looks Celsius, while physics helpers expect Kelvin"
        if mn < 180.0 or mx > 340.0:
            return f"temperature outside broad Kelvin range: min={mn:.3g}, max={mx:.3g}"
    if name in {"PRECIP", "SW_RAD", "CAPE"} and mn < -1e-6:
        return f"{name} has negative values: min={mn:.3g}"
    if name in {"WSPD10", "WSPD925"} and mn < -1e-6:
        return f"{name} has negative wind speed: min={mn:.3g}"
    if name == "WDIR10" and (mn < -1e-6 or mx > 360.0 + 1e-6):
        return f"wind direction outside [0, 360]: min={mn:.3g}, max={mx:.3g}"
    if name.startswith("Q_") and (mn < -1e-8 or mx > 0.08):
        return f"specific humidity outside broad range: min={mn:.3g}, max={mx:.3g}"
    if name == "ZENITH_PROXY" and (mn < -1e-6 or mx > 180.0 + 1e-6):
        return f"zenith proxy outside [0, 180]: min={mn:.3g}, max={mx:.3g}"
    return None


def source_status_table(weather_ds: xr.Dataset) -> pd.DataFrame:
    rows = []
    for name in DYNAMIC_FEATURE_ORDER:
        exists, source = variable_exists(weather_ds, WEATHER_ALIASES.get(name, (name,)))
        if exists:
            status = "direct"
            requires = ""
        elif name in DERIVED_FEATURES:
            status = "derived"
            requires = ",".join(DERIVED_FEATURES[name])
        else:
            status = "missing"
            requires = ""
        rows.append({"feature": name, "status": status, "source": source or "", "requires": requires})
    return pd.DataFrame(rows)


def weather_feature_table(
    weather_ds: xr.Dataset,
    station_file: str,
    local_time_offset_hours: float,
    issues: List[Dict[str, str]],
) -> Tuple[pd.DataFrame, Optional[Tuple[int, int, int]]]:
    if os.path.exists(station_file):
        try:
            station_table = load_station_table(station_file, weather_ds["station"].values)
            latitudes = station_table["station_lat"].to_numpy(dtype=np.float32)
            longitudes = station_table["station_lon"].to_numpy(dtype=np.float32)
            zenith = calculate_zenith_angle(latitudes, longitudes, weather_ds["time"].values)
            weather_ds = weather_ds.assign(ZENITH_PROXY=(("time", "station"), zenith.astype(np.float32)))
        except Exception as exc:
            add_issue(issues, "warning", "station_table", f"could not calculate location-aware zenith: {exc}")
    else:
        add_issue(issues, "warning", "station_table", f"station file not found: {station_file}")

    try:
        cube, _, _ = extract_dynamic_cube(
            weather_ds,
            local_time_offset_hours=local_time_offset_hours,
            use_source_zenith=True,
        )
    except Exception as exc:
        add_issue(issues, "error", "weather", f"failed to extract dynamic cube: {exc}")
        return pd.DataFrame(), None

    rows = []
    for i, name in enumerate(DYNAMIC_FEATURE_ORDER):
        stats = finite_stats(cube[:, :, i])
        msg = feature_range_issue(name, stats)
        if msg:
            add_issue(issues, "warning", f"weather:{name}", msg)
        row = {"feature": name, **stats}
        rows.append(row)
    return pd.DataFrame(rows), tuple(int(v) for v in cube.shape)


def build_markdown(summary: Mapping[str, object], issues: Sequence[Mapping[str, str]]) -> str:
    align = summary.get("alignment", {})
    vis = summary.get("visibility_aligned", {})
    cls = vis.get("class_counts", {}) if isinstance(vis, dict) else {}
    lines = [
        "# Airport METAR Raw Data QC",
        "",
        f"- Created: {summary.get('created_at')}",
        f"- Weather file: `{summary.get('weather_file')}`",
        f"- Visibility file: `{summary.get('visibility_file')}`",
        f"- Visibility variable: `{summary.get('visibility_variable')}`",
        f"- Common times: {align.get('common_times')}",
        f"- Common stations: {align.get('common_stations')}",
        f"- Dynamic cube shape: {summary.get('dynamic_cube_shape')}",
        "",
        "## Visibility",
        "",
        f"- Unit rule: `{vis.get('unit_rule') if isinstance(vis, dict) else None}`",
        f"- Raw range: {vis.get('raw_stats', {}).get('min')} .. {vis.get('raw_stats', {}).get('max')}",
        f"- Meter range: {vis.get('meter_stats', {}).get('min')} .. {vis.get('meter_stats', {}).get('max')}",
        f"- Valid rows: {cls.get('valid')} / invalid or missing: {cls.get('invalid_or_missing')}",
        f"- Fog <500 m: {cls.get('fog_lt_500m')} ({cls.get('fog_frac')})",
        f"- Mist 500-1000 m: {cls.get('mist_500_1000m')} ({cls.get('mist_frac')})",
        f"- Clear >=1000 m: {cls.get('clear_ge_1000m')} ({cls.get('clear_frac')})",
        "",
        "## Issues",
        "",
    ]
    if issues:
        for item in issues:
            lines.append(f"- [{item.get('level')}] {item.get('area')}: {item.get('message')}")
    else:
        lines.append("- No obvious issues found by this script.")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `weather_feature_qc.csv`: per dynamic feature value ranges.",
            "- `weather_feature_source_status.csv`: direct/derived/missing source status.",
            "- `visibility_station_qc.csv`: per station target statistics.",
            "- `airport_raw_data_qc.json`: full machine-readable summary.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check raw airport METAR weather/visibility files.")
    p.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    p.add_argument("--visibility-file", default=DEFAULT_VISIBILITY_FILE)
    p.add_argument("--station-file", default=DEFAULT_STATION_FILE)
    p.add_argument("--vegetation-file", default=DEFAULT_VEG_FILE)
    p.add_argument("--orography-file", default=DEFAULT_ORO_FILE)
    p.add_argument("--visibility-var", default="visibility")
    p.add_argument("--output-dir", default="")
    p.add_argument("--local-time-offset-hours", type=float, default=LOCAL_TIME_OFFSET_HOURS)
    p.add_argument("--weather-time-shift-hours", type=float, default=0.0)
    p.add_argument("--visibility-time-shift-hours", type=float, default=0.0)
    p.add_argument("--max-visibility-m", type=float, default=MAX_VISIBILITY_M)
    p.add_argument("--fail-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or os.path.join(
        os.getcwd(), "airport_raw_qc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    issues: List[Dict[str, str]] = []
    summary: Dict[str, object] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "weather_file": args.weather_file,
        "visibility_file": args.visibility_file,
        "station_file": args.station_file,
        "visibility_var_requested": args.visibility_var,
    }

    print(f"[Load] weather    {args.weather_file}", flush=True)
    print(f"[Load] visibility {args.visibility_file}", flush=True)
    weather_ds = open_dataset_auto(args.weather_file)
    vis_ds = open_dataset_auto(args.visibility_file)
    weather_ds = shift_time_if_needed(weather_ds, args.weather_time_shift_hours)
    vis_ds = shift_time_if_needed(vis_ds, args.visibility_time_shift_hours)

    summary["weather_dims"] = dict(weather_ds.sizes)
    summary["visibility_dims"] = dict(vis_ds.sizes)
    summary["weather_data_vars"] = list(map(str, weather_ds.data_vars))
    summary["visibility_data_vars"] = list(map(str, vis_ds.data_vars))
    summary["weather_time"] = time_summary(weather_ds, "weather", issues)
    summary["visibility_time"] = time_summary(vis_ds, "visibility", issues)
    summary["weather_station"] = station_summary(weather_ds, "weather", issues)
    summary["visibility_station"] = station_summary(vis_ds, "visibility", issues)

    visibility_var = resolve_visibility_var(vis_ds, args.visibility_var)
    summary["visibility_variable"] = visibility_var
    summary["visibility_whole_file"] = check_visibility(
        vis_ds[visibility_var],
        args.max_visibility_m,
        issues,
        "visibility_whole_file",
    )

    common_times, common_stations, align = common_time_station(weather_ds, vis_ds)
    summary["alignment"] = align
    if len(common_times) < 12:
        add_issue(issues, "error", "alignment", f"only {len(common_times)} common times; need at least 12")
    if not common_stations:
        add_issue(issues, "error", "alignment", "no common stations")

    if common_times.size and common_stations:
        print(
            f"[Align] common times={len(common_times)}, stations={len(common_stations)}",
            flush=True,
        )
        weather_aligned = weather_ds.sel(time=common_times, station=common_stations)
        vis_aligned = vis_ds[visibility_var].sel(time=common_times, station=common_stations)

        summary["visibility_aligned"] = check_visibility(
            vis_aligned,
            args.max_visibility_m,
            issues,
            "visibility_aligned",
        )
        vis_station_df = visibility_station_table(vis_aligned, args.max_visibility_m)
        vis_station_df.to_csv(os.path.join(out_dir, "visibility_station_qc.csv"), index=False)

        source_df = source_status_table(weather_aligned)
        source_df.to_csv(os.path.join(out_dir, "weather_feature_source_status.csv"), index=False)
        for _, row in source_df.iterrows():
            if row["status"] == "missing":
                add_issue(issues, "error", f"weather:{row['feature']}", "required source variable is missing")

        feature_df, cube_shape = weather_feature_table(
            weather_aligned,
            args.station_file,
            args.local_time_offset_hours,
            issues,
        )
        summary["dynamic_cube_shape"] = cube_shape
        if not feature_df.empty:
            feature_df.to_csv(os.path.join(out_dir, "weather_feature_qc.csv"), index=False)

        n_wins = int((len(common_times) - 12) + 1) if len(common_times) >= 12 else 0
        summary["window_audit"] = {
            "window_size": 12,
            "n_windows_if_step_1": n_wins,
            "rows_before_visibility_filter": int(max(n_wins, 0) * len(common_stations)),
        }

    summary["issues"] = issues
    summary["issue_counts"] = {
        level: int(sum(1 for item in issues if item["level"] == level))
        for level in ("error", "warning", "info")
    }

    json_path = os.path.join(out_dir, "airport_raw_data_qc.json")
    md_path = os.path.join(out_dir, "airport_raw_data_qc_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(summary, issues))

    print(f"[Write] {md_path}", flush=True)
    print(f"[Write] {json_path}", flush=True)
    print(f"[Done] errors={summary['issue_counts']['error']} warnings={summary['issue_counts']['warning']}", flush=True)

    weather_ds.close()
    vis_ds.close()
    if args.fail_on_error and summary["issue_counts"]["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
