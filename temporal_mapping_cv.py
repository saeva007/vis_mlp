#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare leakage-safe temporal folds for the mapping-operator experiment.

The source dataset already has frozen train/validation/test files.  This module
keeps those roles unchanged and adds a calendar-time holdout:

* the observed months are split into deterministic contiguous blocks;
* a fold trains and validates only on months outside its held-out block;
* the existing frozen test rows inside the held-out months form that fold's
  test partition;
* a temporal embargo wider than the 12 h input window removes boundary rows
  from training and validation;
* fold construction reads timestamps only and never visibility labels.

The generated ``s2_{train,val,test}_indices.npy`` files are compatible with the
existing mapping-operator training and aggregation commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = "/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
SPLITS = ("train", "val", "test")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def _required_dataset_files(data_dir: Path) -> List[Path]:
    return [
        data_dir / f"{stem}_{split}.{suffix}"
        for split in SPLITS
        for stem, suffix in (("X", "npy"), ("y", "npy"), ("meta", "csv"))
    ]


def _dataset_shapes(data_dir: Path) -> Dict[str, Dict[str, object]]:
    missing = [str(path) for path in _required_dataset_files(data_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Temporal-CV dataset is incomplete: {missing}")
    result: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        x_shape = tuple(int(value) for value in np.load(data_dir / f"X_{split}.npy", mmap_mode="r").shape)
        y_shape = tuple(int(value) for value in np.load(data_dir / f"y_{split}.npy", mmap_mode="r").shape)
        if not x_shape or not y_shape or x_shape[0] != y_shape[0]:
            raise ValueError(f"X/y row mismatch for {split}: X={x_shape}, y={y_shape}")
        result[split] = {"x_shape": list(x_shape), "y_shape": list(y_shape)}
    return result


def _parse_times(values: Iterable[object], source: Path) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    times = pd.DatetimeIndex(parsed)
    if times.isna().any():
        raise ValueError(f"Unparseable time values in {source}")
    return times


def _month_strings(times: pd.DatetimeIndex) -> np.ndarray:
    naive = times.tz_convert(None)
    return naive.to_period("M").astype(str).to_numpy(dtype=str)


def _scan_months(meta_path: Path, chunksize: int) -> Tuple[List[str], int, str, str]:
    header = pd.read_csv(meta_path, nrows=0).columns.tolist()
    if "time" not in header:
        raise ValueError(f"{meta_path} lacks required time column; got {header}")
    months: set[str] = set()
    rows = 0
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    for chunk in pd.read_csv(meta_path, usecols=["time"], chunksize=chunksize):
        times = _parse_times(chunk["time"], meta_path)
        months.update(_month_strings(times).tolist())
        local_min = times.min()
        local_max = times.max()
        minimum = local_min if minimum is None or local_min < minimum else minimum
        maximum = local_max if maximum is None or local_max > maximum else maximum
        rows += len(times)
    if not rows or minimum is None or maximum is None:
        raise ValueError(f"No temporal metadata rows in {meta_path}")
    return sorted(months), rows, minimum.isoformat(), maximum.isoformat()


def _assign_contiguous_month_folds(months: Sequence[str], n_folds: int) -> Dict[int, List[str]]:
    unique = sorted(set(str(month) for month in months))
    if len(unique) < n_folds:
        raise ValueError(f"Only {len(unique)} observed months are available for {n_folds} temporal folds")
    periods = pd.PeriodIndex(unique, freq="M")
    if not periods.is_monotonic_increasing:
        raise AssertionError("Observed months are not chronological")
    assignments: Dict[int, List[str]] = {}
    for fold, positions in enumerate(np.array_split(np.arange(len(unique), dtype=np.int64), n_folds)):
        values = [unique[int(position)] for position in positions]
        if not values:
            raise AssertionError(f"Temporal fold {fold} is empty")
        assignments[int(fold)] = values
    counts = np.asarray([len(values) for values in assignments.values()], dtype=np.int64)
    if int(counts.max() - counts.min()) > 1:
        raise AssertionError(f"Temporal month counts are imbalanced: {counts.tolist()}")
    return assignments


def _fold_interval(months: Sequence[str], embargo_hours: float) -> Dict[str, object]:
    first = pd.Period(str(months[0]), freq="M")
    last = pd.Period(str(months[-1]), freq="M")
    block_start = first.start_time.tz_localize("UTC")
    block_end = (last + 1).start_time.tz_localize("UTC")
    embargo = pd.Timedelta(hours=float(embargo_hours))
    return {
        "block_start": block_start,
        "block_end_exclusive": block_end,
        "embargo_start": block_start - embargo,
        "embargo_end_exclusive": block_end + embargo,
    }


def _indices_by_temporal_fold(
    meta_path: Path,
    fold_months: Mapping[int, Sequence[str]],
    split: str,
    embargo_hours: float,
    chunksize: int,
) -> Tuple[Dict[int, np.ndarray], int]:
    pieces: Dict[int, List[np.ndarray]] = {int(fold): [] for fold in fold_months}
    intervals = {
        int(fold): _fold_interval(months, embargo_hours)
        for fold, months in fold_months.items()
    }
    offset = 0
    for chunk in pd.read_csv(meta_path, usecols=["time"], chunksize=chunksize):
        times = _parse_times(chunk["time"], meta_path)
        months = _month_strings(times)
        for fold, heldout in fold_months.items():
            if split == "test":
                mask = np.isin(months, np.asarray(heldout, dtype=str))
            else:
                interval = intervals[int(fold)]
                mask = np.asarray(
                    (times < interval["embargo_start"])
                    | (times >= interval["embargo_end_exclusive"]),
                    dtype=bool,
                )
            local = np.flatnonzero(mask).astype(np.int64) + offset
            if len(local):
                pieces[int(fold)].append(local)
        offset += len(times)
    result = {
        fold: (np.concatenate(values) if values else np.empty(0, dtype=np.int64))
        for fold, values in pieces.items()
    }
    return result, offset


def prepare_temporal_folds(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if float(args.embargo_hours) < float(args.window_hours):
        raise ValueError(
            f"Temporal embargo ({args.embargo_hours} h) must cover the input window "
            f"({args.window_hours} h)"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Fold directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = _dataset_shapes(data_dir)
    split_scans: Dict[str, Dict[str, object]] = {}
    reference_months: List[str] | None = None
    for split in SPLITS:
        months, meta_rows, minimum, maximum = _scan_months(
            data_dir / f"meta_{split}.csv", args.chunksize
        )
        expected_rows = int(shapes[split]["x_shape"][0])
        if meta_rows != expected_rows:
            raise ValueError(f"meta/X row mismatch for {split}: meta={meta_rows}, X={expected_rows}")
        if reference_months is None:
            reference_months = months
        elif months != reference_months:
            raise ValueError(
                "Formal temporal CV requires the same observed calendar months in train/val/test; "
                f"train={reference_months}, {split}={months}"
            )
        split_scans[split] = {
            "rows": meta_rows,
            "minimum_time_utc": minimum,
            "maximum_time_utc": maximum,
            "months": months,
        }
    assert reference_months is not None
    fold_months = _assign_contiguous_month_folds(reference_months, int(args.n_folds))

    summary_rows: List[Dict[str, object]] = []
    indices_by_split: Dict[str, Dict[int, np.ndarray]] = {}
    for split in SPLITS:
        by_fold, meta_rows = _indices_by_temporal_fold(
            data_dir / f"meta_{split}.csv",
            fold_months,
            split,
            float(args.embargo_hours),
            int(args.chunksize),
        )
        expected_rows = int(shapes[split]["x_shape"][0])
        if meta_rows != expected_rows:
            raise ValueError(f"meta/X row mismatch during selection for {split}: {meta_rows} != {expected_rows}")
        indices_by_split[split] = by_fold
        for fold, indices in by_fold.items():
            if not len(indices):
                raise ValueError(f"Temporal fold {fold} has no selected {split} rows")
            if np.any(indices[1:] <= indices[:-1]):
                raise AssertionError(f"Temporal fold {fold} {split} indices are not strictly increasing")
            _atomic_npy(output_dir / f"fold_{fold}" / f"s2_{split}_indices.npy", indices)
            summary_rows.append(
                {
                    "fold": int(fold),
                    "fold_label": f"T{int(fold) + 1}",
                    "split": split,
                    "selected_rows": int(len(indices)),
                    "source_rows": expected_rows,
                    "selected_fraction": float(len(indices) / expected_rows),
                    "selected_role": (
                        "heldout_temporal_block_from_frozen_test"
                        if split == "test"
                        else "outside_heldout_block_after_embargo"
                    ),
                }
            )

    combined_test = np.sort(
        np.concatenate([indices_by_split["test"][fold] for fold in range(int(args.n_folds))])
    )
    expected_test = np.arange(int(shapes["test"]["x_shape"][0]), dtype=np.int64)
    if not np.array_equal(combined_test, expected_test):
        raise AssertionError("Temporal folds do not partition frozen test rows exactly once")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fold_row_summary.csv", index=False)
    block_rows: List[Dict[str, object]] = []
    month_rows: List[Dict[str, object]] = []
    for fold, months in fold_months.items():
        interval = _fold_interval(months, float(args.embargo_hours))
        counts = summary.loc[summary["fold"] == fold].set_index("split")["selected_rows"]
        block_rows.append(
            {
                "fold": int(fold),
                "fold_label": f"T{int(fold) + 1}",
                "start_month": months[0],
                "end_month": months[-1],
                "heldout_month_count": int(len(months)),
                "heldout_months": ";".join(months),
                "block_start_utc": interval["block_start"].isoformat(),
                "block_end_exclusive_utc": interval["block_end_exclusive"].isoformat(),
                "embargo_start_utc": interval["embargo_start"].isoformat(),
                "embargo_end_exclusive_utc": interval["embargo_end_exclusive"].isoformat(),
                "train_rows": int(counts["train"]),
                "val_rows": int(counts["val"]),
                "test_rows": int(counts["test"]),
            }
        )
        for order, month in enumerate(months):
            month_rows.append(
                {
                    "fold": int(fold),
                    "fold_label": f"T{int(fold) + 1}",
                    "month": month,
                    "order_within_fold": int(order),
                }
            )
    pd.DataFrame(block_rows).to_csv(output_dir / "temporal_blocks.csv", index=False)
    pd.DataFrame(month_rows).to_csv(output_dir / "month_folds.csv", index=False)

    fold_manifest = {
        "schema_version": 1,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "cv_kind": "temporal",
        "data_dir": str(data_dir),
        "data_shapes": shapes,
        "n_folds": int(args.n_folds),
        "algorithm": "contiguous_observed_calendar_month_blocks_v1",
        "balance_policy": "held-out month counts differ by at most one",
        "observed_months": reference_months,
        "fold_months": {str(fold): months for fold, months in fold_months.items()},
        "embargo_hours": float(args.embargo_hours),
        "input_window_hours": float(args.window_hours),
        "label_access_during_fold_construction": False,
        "test_partition_exactly_once": True,
        "temporal_direction": (
            "complementary blocked CV; training may include dates before and after the held-out "
            "block; this is not rolling-origin evaluation"
        ),
        "source_split_contract": {
            "train": "existing X_train/y_train rows outside held-out months and embargo",
            "validation": "existing X_val/y_val rows outside held-out months and embargo",
            "test": "existing frozen X_test/y_test rows inside held-out months",
        },
        "split_time_coverage": split_scans,
        "files": {
            "blocks": str(output_dir / "temporal_blocks.csv"),
            "month_folds": str(output_dir / "month_folds.csv"),
            "row_summary": str(output_dir / "fold_row_summary.csv"),
        },
    }
    _atomic_json(output_dir / "fold_manifest.json", fold_manifest)
    print(json.dumps(fold_manifest, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create deterministic blocked temporal fold indices")
    prepare.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--n-folds", type=int, default=5)
    prepare.add_argument("--embargo-hours", type=float, default=24.0)
    prepare.add_argument("--window-hours", type=float, default=12.0)
    prepare.add_argument("--chunksize", type=int, default=500_000)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_temporal_folds)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
