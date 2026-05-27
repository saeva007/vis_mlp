import argparse
import calendar
import json
import os
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


SOURCE_ZARR = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"

SOURCE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "wind_speed",
]

OUTPUT_ORDER = [
    "RH2M",
    "D2M",
    "T2M",
    "MSLP",
    "U10",
    "V10",
    "WSPD10",
    "WDIR10",
    "T_1000",
    "RH_1000",
    "Q_1000",
    "DP_1000",
    "U_1000",
    "V_1000",
    "WSPD1000",
    "WDIR1000",
    "T_925",
    "RH_925",
    "Q_925",
    "DP_925",
    "U_925",
    "V_925",
    "WSPD925",
    "WDIR925",
    "DPD",
    "INVERSION",
]


os.environ["GCSFS_NO_CREDENTIALS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Could not determine bucket type.*")


def log(message: str) -> None:
    print(message, flush=True)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_ranges(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_start = max(start, current)
        month_end = min(end, date(current.year, current.month, last_day))
        yield month_start, month_end
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def output_label(month_start: date, month_end: date) -> str:
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    if month_start.day == 1 and month_end.day == last_day:
        return month_start.strftime("%Y%m")
    return f"{month_start:%Y%m%d}_{month_end:%Y%m%d}"


def calc_rh_from_q(temperature_k, specific_humidity, pressure_hpa):
    temp_c = temperature_k - 273.15
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    q = specific_humidity.clip(min=1e-8, max=0.08)
    e = (q * pressure_hpa) / (0.622 + 0.378 * q)
    rh = (e / es.clip(min=1e-6)) * 100.0
    return rh.clip(min=0.0, max=100.0)


def calc_dewpoint_from_q(specific_humidity, pressure_hpa):
    q = specific_humidity.clip(min=1e-8, max=0.08)
    e_hpa = (q * pressure_hpa) / (0.622 + 0.378 * q).clip(min=1e-8)
    ln_ratio = np.log(e_hpa.clip(min=1e-6) / 6.112)
    td_c = (243.5 * ln_ratio) / (17.67 - ln_ratio).clip(min=1e-6)
    return td_c + 273.15


def calc_dewpoint_from_rh(temperature_k, rh_percent):
    temp_c = temperature_k - 273.15
    rh_frac = (rh_percent / 100.0).clip(min=1e-4, max=1.0)
    gamma = np.log(rh_frac) + (17.67 * temp_c) / (243.5 + temp_c)
    td_c = (243.5 * gamma) / (17.67 - gamma).clip(min=1e-6)
    return td_c + 273.15


def calc_wind_speed_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def pressure_level_dataset(ds: xr.Dataset, level_hpa: int) -> xr.Dataset | None:
    if "level" not in ds.coords:
        return None
    level_values = {int(v) for v in ds["level"].values.tolist()}
    if level_hpa not in level_values:
        return None
    ds_level = ds.sel(level=level_hpa)
    if "level" in ds_level.coords:
        ds_level = ds_level.drop_vars("level")
    return ds_level


def add_pressure_level_vars(out: dict, ds: xr.Dataset, level_hpa: int) -> None:
    ds_level = pressure_level_dataset(ds, level_hpa)
    if ds_level is None:
        log(f"  [WARN] Pangu 数据中没有 {level_hpa} hPa 层，跳过该层变量。")
        return

    temp_name = f"T_{level_hpa}"
    q_name = f"Q_{level_hpa}"
    rh_name = f"RH_{level_hpa}"
    dp_name = f"DP_{level_hpa}"

    if "temperature" in ds_level:
        out[temp_name] = ds_level["temperature"]
    if "specific_humidity" in ds_level:
        out[q_name] = ds_level["specific_humidity"]
    if temp_name in out and q_name in out:
        out[rh_name] = calc_rh_from_q(out[temp_name], out[q_name], pressure_hpa=level_hpa)
        out[dp_name] = calc_dewpoint_from_q(out[q_name], pressure_hpa=level_hpa)

    u_name = f"U_{level_hpa}"
    v_name = f"V_{level_hpa}"
    wspd_name = f"WSPD{level_hpa}"
    wdir_name = f"WDIR{level_hpa}"
    if "u_component_of_wind" in ds_level and "v_component_of_wind" in ds_level:
        out[u_name] = ds_level["u_component_of_wind"]
        out[v_name] = ds_level["v_component_of_wind"]
        out[wspd_name], out[wdir_name] = calc_wind_speed_dir(out[u_name], out[v_name])
    elif "wind_speed" in ds_level:
        out[wspd_name] = ds_level["wind_speed"]


def build_local_dataset(ds: xr.Dataset, lead_hours: int) -> xr.Dataset:
    out: dict[str, xr.DataArray] = {}

    if "2m_temperature" in ds:
        out["T2M"] = ds["2m_temperature"]
    if "mean_sea_level_pressure" in ds:
        out["MSLP"] = ds["mean_sea_level_pressure"]

    if "10m_u_component_of_wind" in ds and "10m_v_component_of_wind" in ds:
        out["U10"] = ds["10m_u_component_of_wind"]
        out["V10"] = ds["10m_v_component_of_wind"]
        out["WSPD10"], out["WDIR10"] = calc_wind_speed_dir(out["U10"], out["V10"])
    elif "10m_wind_speed" in ds:
        out["WSPD10"] = ds["10m_wind_speed"]

    add_pressure_level_vars(out, ds, 1000)
    add_pressure_level_vars(out, ds, 925)

    if "RH_1000" in out:
        log("  使用 1000 hPa 相对湿度近似生成 RH2M。")
        out["RH2M"] = out["RH_1000"].copy(deep=False)
        out["RH2M"].attrs["comment"] = "Approximated from 1000 hPa relative humidity."

    if "T2M" in out and "RH2M" in out:
        out["D2M"] = calc_dewpoint_from_rh(out["T2M"], out["RH2M"])
        out["DPD"] = out["T2M"] - out["D2M"]
        out["D2M"].attrs["comment"] = "Derived from T2M and RH2M."

    if "T_925" in out and "T2M" in out:
        out["INVERSION"] = out["T_925"] - out["T2M"]

    ordered = {name: out[name] for name in OUTPUT_ORDER if name in out}
    local_ds = xr.Dataset(ordered).astype("float32")
    local_ds.attrs.update(
        {
            "source": SOURCE_ZARR,
            "forecast_lead_hours": int(lead_hours),
            "time_coordinate": "valid_time",
            "note": (
                "Pangu source does not provide the Tianji/IFS overlap fields "
                "PRECIP, SW_RAD, CAPE, LCC, W_925, or W_1000 in this Zarr store."
            ),
        }
    )
    return local_ds


def netcdf_encoding(ds: xr.Dataset, compress_level: int) -> dict:
    encoding = {}
    for name in ds.data_vars:
        encoding[name] = {"dtype": "float32"}
        if compress_level > 0:
            encoding[name].update(
                {
                    "zlib": True,
                    "complevel": int(compress_level),
                    "shuffle": True,
                }
            )
    return encoding


def align_time_coordinate(ds: xr.Dataset, lead_hours: int, mode: str) -> xr.Dataset:
    lead_delta = np.timedelta64(int(lead_hours), "h")
    if mode == "valid":
        init_time = ds["time"].copy(deep=False)
        valid_time = ds["time"] + lead_delta
        ds = ds.assign_coords(init_time=init_time)
        ds = ds.assign_coords(time=valid_time)
    else:
        valid_time = ds["time"] + lead_delta
        ds = ds.assign_coords(valid_time=valid_time)
    return ds


def selection_window(month_start: date, month_end: date, lead_hours: int, time_mode: str):
    start_dt = datetime.combine(month_start, time.min)
    end_dt = datetime.combine(month_end, time.max.replace(microsecond=0))
    if time_mode == "valid":
        offset = timedelta(hours=int(lead_hours))
        start_dt = start_dt - offset
        end_dt = end_dt - offset
    return np.datetime64(start_dt), np.datetime64(end_dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download monthly China-region Pangu forecasts from WeatherBench2 "
            "and write canonical fields for low-visibility training experiments."
        )
    )
    parser.add_argument("--start-date", default="2021-01-01", help="Valid-date start, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2021-12-31", help="Valid-date end, YYYY-MM-DD.")
    parser.add_argument("--lead-hours", type=int, default=24, help="Forecast lead to extract.")
    parser.add_argument(
        "--time-coordinate",
        choices=["valid", "init"],
        default="valid",
        help="Use valid time or initialization time as the output time coordinate.",
    )
    parser.add_argument("--lat-min", type=float, default=18.0)
    parser.add_argument("--lat-max", type=float, default=54.0)
    parser.add_argument("--lon-min", type=float, default=73.0)
    parser.add_argument("--lon-max", type=float, default=135.0)
    parser.add_argument("--output-dir", default="pangu_2021_china_monthly")
    parser.add_argument("--compress-level", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--smoke-days", type=int, default=0, help="Limit to N valid days for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Build lazy monthly datasets but do not download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing monthly files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if args.smoke_days > 0:
        end = min(end, start + timedelta(days=args.smoke_days - 1))
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("正在连接谷歌云端 Pangu 数据集...")
    ds = xr.open_zarr(
        SOURCE_ZARR,
        storage_options={"token": "anon"},
        consolidated=True,
        decode_timedelta=True,
    )

    available_source_vars = [name for name in SOURCE_VARS if name in ds.data_vars]
    missing_source_vars = [name for name in SOURCE_VARS if name not in ds.data_vars]
    ds = ds[available_source_vars]
    if missing_source_vars:
        log(f"[WARN] Pangu 源数据缺少这些候选变量: {', '.join(missing_source_vars)}")

    lat_desc = float(ds.latitude[0]) > float(ds.latitude[-1])
    lat_slice = slice(args.lat_max, args.lat_min) if lat_desc else slice(args.lat_min, args.lat_max)
    lon_slice = slice(args.lon_min, args.lon_max)

    log(
        f"正在筛选中国区域，经度 {args.lon_min:g}-{args.lon_max:g}E，"
        f"纬度 {args.lat_min:g}-{args.lat_max:g}N..."
    )
    ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    target_lead_time = np.asarray(
        np.timedelta64(int(args.lead_hours), "h"),
        dtype="timedelta64[ns]",
    )
    if "prediction_timedelta" in ds.dims:
        available_leads = np.asarray(ds["prediction_timedelta"].values).astype("timedelta64[ns]")
        if not np.any(available_leads == target_lead_time):
            raise ValueError(f"Pangu 数据集中没有 {args.lead_hours} h 预报提前期。")
        log(f"检测到预报提前期维度，已指定仅筛选: {args.lead_hours} hours 的预报数据...")
        ds = ds.sel(prediction_timedelta=target_lead_time)
        if "prediction_timedelta" in ds.coords and "prediction_timedelta" not in ds.dims:
            ds = ds.drop_vars("prediction_timedelta")

    outputs: list[str] = []
    for month_start, month_end in month_ranges(start, end):
        label = output_label(month_start, month_end)
        output_file = output_dir / f"pangu_china_{label}_lead{args.lead_hours:02d}h.nc"
        tmp_file = output_dir / f"{output_file.stem}.tmp.nc"

        if output_file.exists() and not args.overwrite:
            log(f"[SKIP] {output_file} 已存在；如需重跑请加 --overwrite。")
            outputs.append(str(output_file))
            continue
        if tmp_file.exists():
            tmp_file.unlink()

        select_start, select_end = selection_window(
            month_start, month_end, args.lead_hours, args.time_coordinate
        )
        log(f"正在处理 {month_start} 至 {month_end}（输出按 {args.time_coordinate} time 对齐）...")
        ds_month = ds.sel(time=slice(select_start, select_end))
        ds_month = align_time_coordinate(ds_month, args.lead_hours, args.time_coordinate)

        log("  正在构建本地 Dataset...")
        local_ds = build_local_dataset(ds_month, lead_hours=args.lead_hours)
        log(
            "  输出变量: "
            + ", ".join(local_ds.data_vars)
            + f"; 维度: {dict(local_ds.sizes)}"
        )
        if args.dry_run:
            log(f"  [DRY RUN] 跳过保存: {output_file}")
            continue

        log(f"  已启动下载并计算数据，正在保存至 {output_file}...")
        with ProgressBar():
            local_ds.to_netcdf(
                tmp_file,
                encoding=netcdf_encoding(local_ds, args.compress_level),
            )
        tmp_file.replace(output_file)
        outputs.append(str(output_file))
        log(f"  完成: {output_file}")

    manifest = {
        "source": SOURCE_ZARR,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "lead_hours": args.lead_hours,
        "time_coordinate": args.time_coordinate,
        "region": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "available_source_vars": available_source_vars,
        "missing_source_vars": missing_source_vars,
        "outputs": outputs,
        "missing_overlap_fields_in_pangu_store": [
            "PRECIP",
            "SW_RAD",
            "CAPE",
            "LCC",
            "W_925",
            "W_1000",
        ],
    }
    manifest_path = output_dir / "pangu_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nPangu 数据处理完成。清单已写入: {manifest_path}")


if __name__ == "__main__":
    main()
