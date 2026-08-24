#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal spatially blocked mapping-operator experiment.

The experiment preserves the existing temporal train/validation/test files and
adds a second, station-level separation:

* train/validation rows contain only stations outside the held-out spatial fold;
* test rows contain only stations in the held-out fold;
* fold construction uses station coordinates only (never visibility labels);
* preprocessing and any model-selection decision rule use training/validation
  rows only;
* test labels are loaded only after the fitted model and validation decision
  rule are frozen.

Subcommands prepare deterministic fold indices, fit an instantaneous
multinomial logistic baseline, evaluate MLP/GRU checkpoints, and aggregate the
five out-of-fold test partitions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

import train_static_rnn_lowvis as rnn


DEFAULT_DATA_DIR = "/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
SPLITS = ("train", "val", "test")
MODELS = ("logistic", "mlp", "gru")
DECISION_RULES = ("argmax", "val_search")
DECISION_EFFECT_METRICS = (
    "Fog_P",
    "Fog_R",
    "Fog_CSI",
    "Mist_P",
    "Mist_R",
    "Mist_CSI",
    "Clear_R",
    "low_vis_precision",
    "low_vis_recall",
    "low_vis_csi",
    "false_positive_rate",
    "accuracy",
)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _canonical_station_ids(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(values, dtype="string").str.strip().str.replace(r"\.0$", "", regex=True)
    if series.isna().any() or (series == "").any():
        raise ValueError("Station metadata contains missing or empty station_id values")
    return series.astype(str).to_numpy()


def _required_dataset_files(data_dir: Path) -> List[Path]:
    return [data_dir / f"{stem}_{split}.{suffix}" for split in SPLITS for stem, suffix in (("X", "npy"), ("y", "npy"), ("meta", "csv"))]


def _dataset_shapes(data_dir: Path) -> Dict[str, Dict[str, object]]:
    missing = [str(path) for path in _required_dataset_files(data_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Spatial CV dataset is incomplete: {missing}")
    result: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        x_shape = tuple(int(v) for v in np.load(data_dir / f"X_{split}.npy", mmap_mode="r").shape)
        y_shape = tuple(int(v) for v in np.load(data_dir / f"y_{split}.npy", mmap_mode="r").shape)
        if not x_shape or not y_shape or x_shape[0] != y_shape[0]:
            raise ValueError(f"X/y row mismatch for {split}: X={x_shape}, y={y_shape}")
        result[split] = {"x_shape": list(x_shape), "y_shape": list(y_shape)}
    return result


def _station_coordinate_table(data_dir: Path, chunksize: int) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for split in SPLITS:
        meta_path = data_dir / f"meta_{split}.csv"
        header = pd.read_csv(meta_path, nrows=0).columns.tolist()
        required = {"station_id", "lat", "lon"}
        if not required.issubset(header):
            raise ValueError(f"{meta_path} must contain {sorted(required)}, got {header}")
        for chunk in pd.read_csv(meta_path, usecols=["station_id", "lat", "lon"], chunksize=chunksize):
            chunk = chunk.copy()
            chunk["station_id"] = _canonical_station_ids(chunk["station_id"])
            chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
            chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
            if chunk[["lat", "lon"]].isna().any().any():
                raise ValueError(f"{meta_path} contains missing/non-numeric coordinates")
            pieces.append(chunk.drop_duplicates("station_id"))
    merged = pd.concat(pieces, ignore_index=True)
    grouped = merged.groupby("station_id", sort=True)
    spread = grouped[["lat", "lon"]].agg(lambda x: float(np.nanmax(x) - np.nanmin(x)))
    bad = spread[(spread["lat"] > 0.01) | (spread["lon"] > 0.01)]
    if len(bad):
        raise ValueError(f"Station coordinates are inconsistent across splits: {bad.head().to_dict('index')}")
    stations = grouped[["lat", "lon"]].median().reset_index()
    if not stations["lat"].between(-90.0, 90.0).all() or not stations["lon"].between(-180.0, 360.0).all():
        raise ValueError("Station coordinates fall outside valid latitude/longitude ranges")
    stations["lon"] = ((stations["lon"] + 180.0) % 360.0) - 180.0
    return stations.sort_values("station_id").reset_index(drop=True)


def _spherical_coordinates(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64))
    return np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )


def _assign_spatial_folds(stations: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    if len(stations) < n_folds * 2:
        raise ValueError(f"Too few stations ({len(stations)}) for {n_folds} spatial folds")

    xyz = _spherical_coordinates(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    station_count = len(stations)
    base, remainder = divmod(station_count, n_folds)
    capacities = np.full(n_folds, base, dtype=np.int64)
    capacities[:remainder] += 1
    labels = np.full(station_count, -1, dtype=np.int64)

    # Recursively bisect each region along its leading spherical-coordinate
    # principal axis.  The exact split positions come from fixed fold
    # capacities, so held-out station counts differ by at most one while every
    # split remains coordinate-only and spatially compact.  The seed affects
    # only exact projection ties; visibility labels are never accessed.
    tie_rank = np.empty(station_count, dtype=np.int64)
    tie_rank[np.random.default_rng(seed).permutation(station_count)] = np.arange(station_count)

    def assign_region(indices: np.ndarray, fold_ids: Sequence[int]) -> None:
        if len(fold_ids) == 1:
            fold = int(fold_ids[0])
            if len(indices) != int(capacities[fold]):
                raise AssertionError(
                    f"Balanced spatial split size mismatch for fold {fold}: "
                    f"got {len(indices)}, expected {capacities[fold]}"
                )
            labels[indices] = fold
            return

        left_fold_count = len(fold_ids) // 2
        left_folds = list(fold_ids[:left_fold_count])
        right_folds = list(fold_ids[left_fold_count:])
        left_size = int(capacities[left_folds].sum())

        local = xyz[indices]
        centered = local - local.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        orient = int(np.argmax(np.abs(axis)))
        if axis[orient] < 0:
            axis = -axis
        projection = centered @ axis
        order = np.lexsort((tie_rank[indices], projection))
        assign_region(indices[order[:left_size]], left_folds)
        assign_region(indices[order[left_size:]], right_folds)

    assign_region(np.arange(station_count, dtype=np.int64), list(range(n_folds)))
    if np.any(labels < 0):
        raise AssertionError("Balanced spatial partition left stations unassigned")

    assigned = stations.copy()
    assigned["raw_cluster"] = labels
    centroids = (
        assigned.groupby("raw_cluster", as_index=False)[["lat", "lon"]]
        .mean()
        .sort_values(["lon", "lat"], kind="stable")
        .reset_index(drop=True)
    )
    remap = {int(row.raw_cluster): int(idx) for idx, row in centroids.iterrows()}
    assigned["fold"] = assigned["raw_cluster"].map(remap).astype(int)
    return assigned.drop(columns=["raw_cluster"]).sort_values(["fold", "station_id"]).reset_index(drop=True)


def _haversine_min_distance_km(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    if not len(reference):
        return np.full(len(query), np.inf, dtype=np.float64)
    qlat = np.deg2rad(query[:, 0])
    qlon = np.deg2rad(query[:, 1])
    rlat = np.deg2rad(reference[:, 0])
    rlon = np.deg2rad(reference[:, 1])
    out = np.full(len(query), np.inf, dtype=np.float64)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        dlat = qlat[start:stop, None] - rlat[None, :]
        dlon = qlon[start:stop, None] - rlon[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(qlat[start:stop, None]) * np.cos(rlat[None, :]) * np.sin(dlon / 2.0) ** 2
        dist = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        out[start:stop] = np.min(dist, axis=1)
    return out


def _fold_station_sets(stations: pd.DataFrame, n_folds: int, buffer_km: float) -> Dict[int, Dict[str, set]]:
    result: Dict[int, Dict[str, set]] = {}
    coords = stations[["lat", "lon"]].to_numpy(dtype=np.float64)
    for fold in range(n_folds):
        held_mask = stations["fold"].to_numpy() == fold
        held = set(stations.loc[held_mask, "station_id"].astype(str))
        candidate = ~held_mask
        if buffer_km > 0:
            nearest = _haversine_min_distance_km(coords[candidate], coords[held_mask])
            candidate_indices = np.flatnonzero(candidate)
            buffer_indices = candidate_indices[nearest < float(buffer_km)]
        else:
            buffer_indices = np.empty(0, dtype=np.int64)
        buffered = set(stations.iloc[buffer_indices]["station_id"].astype(str))
        train = set(stations.loc[candidate, "station_id"].astype(str)) - buffered
        if train & held:
            raise AssertionError("Train and held-out station sets overlap")
        result[fold] = {"train": train, "heldout": held, "buffered": buffered}
    return result


def _split_indices_by_fold(
    meta_path: Path,
    fold_sets: Mapping[int, Mapping[str, set]],
    split: str,
    chunksize: int,
) -> Tuple[Dict[int, np.ndarray], int]:
    pieces: Dict[int, List[np.ndarray]] = {fold: [] for fold in fold_sets}
    offset = 0
    for chunk in pd.read_csv(meta_path, usecols=["station_id"], chunksize=chunksize):
        station_ids = _canonical_station_ids(chunk["station_id"])
        for fold, sets in fold_sets.items():
            allowed = sets["heldout"] if split == "test" else sets["train"]
            mask = np.fromiter((sid in allowed for sid in station_ids), dtype=bool, count=len(station_ids))
            local = np.flatnonzero(mask).astype(np.int64) + offset
            if len(local):
                pieces[fold].append(local)
        offset += len(chunk)
    result = {
        fold: (np.concatenate(values) if values else np.empty(0, dtype=np.int64))
        for fold, values in pieces.items()
    }
    return result, offset


def prepare_folds(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Fold directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = _dataset_shapes(data_dir)
    stations = _station_coordinate_table(data_dir, args.chunksize)
    stations = _assign_spatial_folds(stations, args.n_folds, args.seed)
    fold_sets = _fold_station_sets(stations, args.n_folds, args.buffer_km)

    counts = stations.groupby("fold").size().reindex(range(args.n_folds), fill_value=0)
    if int(counts.min()) < int(args.min_fold_stations):
        raise ValueError(f"Balanced spatial partition produced an undersized fold: {counts.to_dict()}")
    if int(counts.max() - counts.min()) > 1:
        raise AssertionError(f"Balanced spatial fold sizes diverged: {counts.to_dict()}")
    stations.to_csv(output_dir / "station_folds.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    for split in SPLITS:
        by_fold, meta_rows = _split_indices_by_fold(
            data_dir / f"meta_{split}.csv", fold_sets, split, args.chunksize
        )
        expected_rows = int(shapes[split]["x_shape"][0])
        if meta_rows != expected_rows:
            raise ValueError(f"meta/X row mismatch for {split}: meta={meta_rows}, X={expected_rows}")
        for fold, indices in by_fold.items():
            if not len(indices):
                raise ValueError(f"Fold {fold} has no selected {split} rows")
            _atomic_npy(output_dir / f"fold_{fold}" / f"s2_{split}_indices.npy", indices)
            summary_rows.append(
                {
                    "fold": int(fold),
                    "split": split,
                    "selected_rows": int(len(indices)),
                    "source_rows": expected_rows,
                    "selected_fraction": float(len(indices) / expected_rows),
                    "selected_role": "heldout_test" if split == "test" else "nonheldout_train_or_val",
                }
            )

    for fold, sets in fold_sets.items():
        fold_dir = output_dir / f"fold_{fold}"
        pd.Series(sorted(sets["train"]), name="station_id").to_csv(fold_dir / "train_stations.csv", index=False)
        pd.Series(sorted(sets["heldout"]), name="station_id").to_csv(fold_dir / "heldout_stations.csv", index=False)
        pd.Series(sorted(sets["buffered"]), name="station_id").to_csv(fold_dir / "buffer_excluded_stations.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fold_row_summary.csv", index=False)
    fold_manifest = {
        "schema_version": 2,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "cv_kind": "spatial",
        "data_dir": str(data_dir),
        "data_shapes": shapes,
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "algorithm": "balanced_recursive_spherical_pca_coordinates_only_v1",
        "balance_policy": "held-out station counts differ by at most one",
        "buffer_km": float(args.buffer_km),
        "station_count": int(len(stations)),
        "fold_station_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "fold_buffer_excluded_counts": {
            str(fold): int(len(sets["buffered"])) for fold, sets in fold_sets.items()
        },
        "label_access_during_fold_construction": False,
        "temporal_contract": {
            "train": "existing X_train/y_train rows, non-held-out stations only",
            "validation": "existing X_val/y_val rows, non-held-out stations only",
            "test": "existing frozen X_test/y_test rows, held-out stations only",
        },
        "files": {
            "station_folds": str(output_dir / "station_folds.csv"),
            "row_summary": str(output_dir / "fold_row_summary.csv"),
        },
    }
    _atomic_json(output_dir / "fold_manifest.json", fold_manifest)
    print(json.dumps(fold_manifest, indent=2, ensure_ascii=False), flush=True)


def _load_indices(fold_dir: Path, split: str) -> np.ndarray:
    path = fold_dir / f"s2_{split}_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path), dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(values[1:] <= values[:-1]):
        raise ValueError(f"Invalid row index file: {path}")
    return values


def _fold_provenance(fold_dir: Path, model_family: str) -> Tuple[str, str, str]:
    manifest_path = fold_dir.parent / "fold_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cv_kind = str(manifest.get("cv_kind", "spatial"))
    algorithm = str(manifest.get("algorithm", "unknown"))
    if cv_kind == "temporal":
        threshold_source = "validation rows outside the held-out temporal block only"
    elif model_family == "logistic":
        threshold_source = "held-in validation stations and validation times only"
    else:
        threshold_source = "checkpoint selected on non-held-out validation stations and validation times only"
    return cv_kind, algorithm, threshold_source


def _continuous_and_vegetation(
    x_source: np.ndarray,
    source_rows: np.ndarray,
    layout: rnn.Layout,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(x_source[source_rows], dtype=np.float32)
    last = (layout.window_size - 1) * layout.dyn_vars
    dynamic = raw[:, last : last + layout.dyn_vars].copy()
    for idx in rnn.log1p_dyn_indices(layout):
        dynamic[:, idx] = np.log1p(np.maximum(dynamic[:, idx], 0.0))
    static = raw[:, layout.split_dyn : layout.split_dyn + 5]
    continuous = np.concatenate([dynamic, static], axis=1).astype(np.float32, copy=False)
    vegetation = np.clip(np.rint(raw[:, layout.split_dyn + 5]), 0, 31).astype(np.int64)
    return continuous, vegetation


def _fit_logistic_preprocessor(
    x_train: np.ndarray,
    train_indices: np.ndarray,
    layout: rnn.Layout,
    seed: int,
    sample_rows: int,
) -> Tuple[np.ndarray, RobustScaler]:
    rng = np.random.default_rng(seed)
    if len(train_indices) > sample_rows:
        selected = np.sort(rng.choice(train_indices, size=sample_rows, replace=False))
    else:
        selected = train_indices
    continuous, _ = _continuous_and_vegetation(x_train, selected, layout)
    medians = np.nanmedian(continuous, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isfinite(continuous), continuous, medians[None, :])
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(filled)
    return medians, scaler


def _transform_logistic_batch(
    x_source: np.ndarray,
    source_rows: np.ndarray,
    layout: rnn.Layout,
    medians: np.ndarray,
    scaler: RobustScaler,
) -> np.ndarray:
    continuous, vegetation = _continuous_and_vegetation(x_source, source_rows, layout)
    continuous = np.where(np.isfinite(continuous), continuous, medians[None, :])
    continuous = scaler.transform(continuous)
    continuous = np.clip(np.nan_to_num(continuous, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    one_hot = np.zeros((len(vegetation), 32), dtype=np.float32)
    one_hot[np.arange(len(vegetation)), vegetation] = 1.0
    return np.concatenate([continuous.astype(np.float32, copy=False), one_hot], axis=1)


def _predict_logistic(
    model: SGDClassifier,
    x_source: np.ndarray,
    row_indices: np.ndarray,
    layout: rnn.Layout,
    medians: np.ndarray,
    scaler: RobustScaler,
    batch_rows: int,
) -> np.ndarray:
    probs = np.empty((len(row_indices), 3), dtype=np.float32)
    for start in range(0, len(row_indices), batch_rows):
        stop = min(start + batch_rows, len(row_indices))
        features = _transform_logistic_batch(
            x_source, row_indices[start:stop], layout, medians, scaler
        )
        block = model.predict_proba(features)
        aligned = np.zeros((len(block), 3), dtype=np.float32)
        aligned[:, np.asarray(model.classes_, dtype=np.int64)] = np.asarray(block, dtype=np.float32)
        probs[start:stop] = aligned
    return probs


def _threshold_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        threshold_grid_low=float(args.threshold_grid_low),
        threshold_grid_high=float(args.threshold_grid_high),
        threshold_grid_step=float(args.threshold_grid_step),
        min_fog_precision=float(args.min_fog_precision),
        min_mist_precision=float(args.min_mist_precision),
        min_clear_recall=float(args.min_clear_recall),
        selection_metric="recall_csi",
    )


def _metrics_with_probabilities(y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return rnn.add_probability_metrics(rnn.build_metrics(y_true, pred), y_true, probs)


def _argmax_predictions(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected [n, 3] class probabilities, got {values.shape}")
    return np.argmax(values, axis=1).astype(np.int64, copy=False)


def _saved_decision_rule(result: Mapping[str, object]) -> str:
    explicit = result.get("analysis_decision_rule") or result.get("decision_rule")
    if explicit in DECISION_RULES:
        return str(explicit)
    thresholds = result.get("thresholds")
    if isinstance(thresholds, Mapping) and thresholds.get("mode") == "argmax":
        return "argmax"
    return "val_search"


def _primary_predictions(payload: Mapping[str, np.ndarray], decision_rule: str) -> np.ndarray:
    if decision_rule == "argmax":
        return _argmax_predictions(payload["probs"])
    if decision_rule == "val_search":
        pred = np.asarray(payload["pred"], dtype=np.int64)
        if pred.ndim != 1 or len(pred) != len(payload["probs"]):
            raise ValueError("Stored predictions do not align with probability rows")
        return pred
    raise ValueError(f"Unsupported decision rule: {decision_rule}")


def _decision_effect_rows(
    cv_kind: str,
    scope: str,
    model: str,
    fold: Optional[int],
    saved_rule: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    saved_pred: np.ndarray,
) -> List[Dict[str, object]]:
    saved_metrics = rnn.build_metrics(y_true, saved_pred)
    argmax_metrics = rnn.build_metrics(y_true, _argmax_predictions(probs))
    rows: List[Dict[str, object]] = []
    for metric in DECISION_EFFECT_METRICS:
        saved_value = float(saved_metrics[metric])
        argmax_value = float(argmax_metrics[metric])
        rows.append(
            {
                "cv_kind": cv_kind,
                "scope": scope,
                "model": model,
                "fold": fold,
                "saved_decision_rule": saved_rule,
                "metric": metric,
                "saved_value": saved_value,
                "argmax_value": argmax_value,
                "argmax_minus_saved": argmax_value - saved_value,
            }
        )
    return rows


def train_logistic(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    fold_dir = Path(args.fold_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_kind, fold_algorithm, threshold_source = _fold_provenance(fold_dir, "logistic")

    train_indices = _load_indices(fold_dir, "train")
    val_indices = _load_indices(fold_dir, "val")
    x_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    x_val = np.load(data_dir / "X_val.npy", mmap_mode="r")
    y_train_source = np.load(data_dir / "y_train.npy", mmap_mode="r")
    y_val_source = np.load(data_dir / "y_val.npy", mmap_mode="r")
    layout = rnn.resolve_layout_from_file(str(data_dir / "X_train.npy"), args.window_size, str(data_dir))

    y_train = rnn.visibility_to_labels(np.asarray(y_train_source[train_indices]))[1]
    y_val = rnn.visibility_to_labels(np.asarray(y_val_source[val_indices]))[1]
    medians, scaler = _fit_logistic_preprocessor(
        x_train, train_indices, layout, args.seed, args.scaler_sample_rows
    )

    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training fold lacks a visibility class: {counts.tolist()}")
    natural = counts / counts.sum()
    target = np.asarray([args.fog_ratio, args.mist_ratio, 1.0 - args.fog_ratio - args.mist_ratio])
    if np.any(target <= 0):
        raise ValueError(f"Invalid target class proportions: {target.tolist()}")
    class_weight = target / natural
    class_weight /= float(np.sum(class_weight * natural))

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(args.alpha),
        fit_intercept=True,
        random_state=int(args.seed),
        learning_rate="optimal",
        average=True,
    )
    rng = np.random.default_rng(args.seed)
    best_model: Optional[SGDClassifier] = None
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    history: List[Dict[str, object]] = []
    fitted = False
    for epoch in range(1, args.max_epochs + 1):
        block_starts = np.arange(0, len(train_indices), args.batch_rows, dtype=np.int64)
        rng.shuffle(block_starts)
        for start in block_starts:
            local = np.arange(start, min(start + args.batch_rows, len(train_indices)), dtype=np.int64)
            features = _transform_logistic_batch(
                x_train, train_indices[local], layout, medians, scaler
            )
            labels = y_train[local]
            weights = class_weight[labels].astype(np.float64)
            if not fitted:
                model.partial_fit(features, labels, classes=np.asarray([0, 1, 2]), sample_weight=weights)
                fitted = True
            else:
                model.partial_fit(features, labels, sample_weight=weights)

        val_probs = _predict_logistic(
            model, x_val, val_indices, layout, medians, scaler, args.batch_rows
        )
        val_ap = float(average_precision_score((y_val <= 1).astype(np.int64), val_probs[:, :2].sum(axis=1)))
        if args.decision_rule == "argmax":
            val_pred = _argmax_predictions(val_probs)
            val_metrics = _metrics_with_probabilities(y_val, val_pred, val_probs)
            selection_score = float(rnn.score_metrics(_threshold_args(args), val_metrics))
            selection_metric = "recall_csi_argmax"
        else:
            selection_score = val_ap
            selection_metric = "low_vis_ap"
        improved = selection_score > best_score + float(args.min_delta)
        history.append(
            {
                "epoch": epoch,
                "decision_rule": args.decision_rule,
                "selection_metric": selection_metric,
                "val_selection_score": selection_score,
                "val_low_vis_ap": val_ap,
                "improved": bool(improved),
            }
        )
        print(
            f"[Logistic] epoch={epoch} decision={args.decision_rule} "
            f"selection_score={selection_score:.6f} val_low_vis_ap={val_ap:.6f} "
            f"improved={improved}",
            flush=True,
        )
        if improved:
            best_score = selection_score
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_model is None:
        raise RuntimeError("Logistic training did not produce a validation checkpoint")

    # Freeze the model and validation decision rule before any test target is loaded.
    val_probs = _predict_logistic(
        best_model, x_val, val_indices, layout, medians, scaler, args.batch_rows
    )
    if args.decision_rule == "argmax":
        thresholds: Dict[str, object] = {"mode": "argmax"}
        val_pred = _argmax_predictions(val_probs)
    else:
        _, thresholds, _ = rnn.threshold_search(_threshold_args(args), val_probs, y_val)
        val_pred = rnn.pred_from_thresholds(
            val_probs, float(thresholds["fog"]), float(thresholds["mist"])
        )
    val_metrics = _metrics_with_probabilities(y_val, val_pred, val_probs)

    test_indices = _load_indices(fold_dir, "test")
    x_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
    y_test_source = np.load(data_dir / "y_test.npy", mmap_mode="r")
    y_test = rnn.visibility_to_labels(np.asarray(y_test_source[test_indices]))[1]
    test_probs = _predict_logistic(
        best_model, x_test, test_indices, layout, medians, scaler, args.batch_rows
    )
    if args.decision_rule == "argmax":
        test_pred = _argmax_predictions(test_probs)
    else:
        test_pred = rnn.pred_from_thresholds(
            test_probs, float(thresholds["fog"]), float(thresholds["mist"])
        )
    test_metrics = _metrics_with_probabilities(y_test, test_pred, test_probs)

    joblib.dump(
        {
            "model": best_model,
            "medians": medians,
            "scaler": scaler,
            "layout": rnn.asdict(layout),
            "class_weight": class_weight,
            "target_class_proportions": target,
        },
        output_dir / "logistic_model.joblib",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    _atomic_npz(
        output_dir / "val_predictions.npz",
        row_index=val_indices,
        y_true=y_val.astype(np.int8),
        probs=val_probs.astype(np.float32),
        pred=val_pred.astype(np.int8),
    )
    _atomic_npz(
        output_dir / "test_predictions.npz",
        row_index=test_indices,
        y_true=y_test.astype(np.int8),
        probs=test_probs.astype(np.float32),
        pred=test_pred.astype(np.int8),
    )
    result = {
        "schema_version": 1,
        "cv_kind": cv_kind,
        "fold_algorithm": fold_algorithm,
        "model": "logistic",
        "fold": int(args.fold),
        "seed": int(args.seed),
        "input_contract": "final dynamic timestep + five static continuous fields + 32-level vegetation one-hot; no FE block",
        "objective": "multinomial_log_loss_with_training_only_class_prior_weights",
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "train_class_counts": counts.astype(int).tolist(),
        "target_class_proportions": target.tolist(),
        "best_validation_epoch": int(best_epoch),
        "checkpoint_selection_metric": (
            "recall_csi_argmax" if args.decision_rule == "argmax" else "low_vis_ap"
        ),
        "checkpoint_selection_rule": args.decision_rule,
        "analysis_decision_rule": args.decision_rule,
        "thresholds": thresholds,
        "threshold_source": (
            "not applicable; fixed argmax decision rule"
            if args.decision_rule == "argmax"
            else threshold_source
        ),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_access_policy": "loaded after model and validation decision rule were frozen",
    }
    _atomic_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def _load_neural_model(checkpoint: Path, device: torch.device) -> Tuple[rnn.StaticRNNLowVisNet, Dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    layout_raw = metadata.get("layout")
    architecture = metadata.get("architecture")
    if not isinstance(layout_raw, dict) or not isinstance(architecture, dict):
        raise ValueError("Checkpoint lacks formal spatial-CV layout/architecture metadata")
    layout = rnn.Layout(
        window_size=int(layout_raw["window_size"]),
        dyn_vars=int(layout_raw["dyn_vars"]),
        fe_dim=int(layout_raw["fe_dim"]),
        dynamic_feature_order=layout_raw.get("dynamic_feature_order"),
    )
    model = rnn.StaticRNNLowVisNet(
        layout=layout,
        encoder=str(metadata["encoder"]),
        hidden_dim=int(architecture["hidden_dim"]),
        static_hidden_dim=int(architecture["static_hidden_dim"]),
        fe_hidden_dim=int(architecture["fe_hidden_dim"]),
        fusion_hidden_dim=int(architecture["fusion_hidden_dim"]),
        veg_emb_dim=int(architecture["veg_emb_dim"]),
        rnn_layers=int(architecture["rnn_layers"]),
        dropout=float(architecture["dropout"]),
        bidirectional=bool(architecture["bidirectional"]),
        pooling=str(metadata["pooling"]),
        use_fe=bool(metadata["use_fe"]),
    )
    model.load_state_dict(rnn._normalise_state_dict_keys(state), strict=True)
    model.to(device).eval()
    return model, metadata


def _neural_split_predictions(
    args: SimpleNamespace,
    model: rnn.StaticRNNLowVisNet,
    scaler: RobustScaler,
    data_dir: Path,
    fold_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = _load_indices(fold_dir, split)
    x_path = str(data_dir / f"X_{split}.npy")
    y_source = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    y_raw, y_cls = rnn.visibility_to_labels(np.asarray(y_source[indices]))
    dataset = rnn.LowVisDataset(
        x_path,
        y_raw,
        y_cls,
        model.layout,
        scaler,
        bool(model.use_fe),
        not bool(getattr(args, "no_pm", False)),
        args,
        row_indices=indices,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for bx, _, _, _, _, _ in loader:
            logits, _ = model(bx.to(device, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    return indices, y_cls.astype(np.int64), np.concatenate(probs, axis=0)


def evaluate_neural(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    fold_dir = Path(args.fold_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_kind, fold_algorithm, threshold_source = _fold_provenance(fold_dir, "neural")
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.run_config).resolve()
    scaler_path = Path(args.scaler).resolve()
    if not checkpoint.is_file() or not config_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/config/scaler: {checkpoint}, {config_path}, {scaler_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        run_args = SimpleNamespace(**json.load(handle))
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, metadata = _load_neural_model(checkpoint, device)
    scaler = joblib.load(scaler_path)

    val_indices, y_val, val_probs = _neural_split_predictions(
        run_args, model, scaler, data_dir, fold_dir, "val", device, args.batch_size, args.num_workers
    )
    checkpoint_selection_rule = str(metadata.get("threshold_mode", "argmax"))
    thresholds = metadata.get("thresholds", {"mode": "argmax"})
    analysis_decision_rule = (
        checkpoint_selection_rule if args.decision_rule == "checkpoint" else args.decision_rule
    )
    if analysis_decision_rule == "argmax":
        val_pred = _argmax_predictions(val_probs)
    else:
        val_pred = rnn.pred_from_thresholds(val_probs, float(thresholds["fog"]), float(thresholds["mist"]))
    val_metrics = _metrics_with_probabilities(y_val, val_pred, val_probs)

    # Test data are opened only after the checkpoint and validation decision rule are fixed.
    test_indices, y_test, test_probs = _neural_split_predictions(
        run_args, model, scaler, data_dir, fold_dir, "test", device, args.batch_size, args.num_workers
    )
    if analysis_decision_rule == "argmax":
        test_pred = _argmax_predictions(test_probs)
    else:
        test_pred = rnn.pred_from_thresholds(test_probs, float(thresholds["fog"]), float(thresholds["mist"]))
    test_metrics = _metrics_with_probabilities(y_test, test_pred, test_probs)

    _atomic_npz(
        output_dir / "val_predictions.npz",
        row_index=val_indices,
        y_true=y_val.astype(np.int8),
        probs=val_probs.astype(np.float32),
        pred=val_pred.astype(np.int8),
    )
    _atomic_npz(
        output_dir / "test_predictions.npz",
        row_index=test_indices,
        y_true=y_test.astype(np.int8),
        probs=test_probs.astype(np.float32),
        pred=test_pred.astype(np.int8),
    )
    result = {
        "schema_version": 1,
        "cv_kind": cv_kind,
        "fold_algorithm": fold_algorithm,
        "model": str(args.model),
        "fold": int(args.fold),
        "seed": int(metadata.get("seed", -1)),
        "checkpoint": str(checkpoint),
        "run_config": str(config_path),
        "scaler": str(scaler_path),
        "encoder": str(metadata.get("encoder")),
        "input_contract": (
            "final dynamic timestep + static branch; no FE block"
            if str(metadata.get("encoder")) == "mlp"
            else "12-hour dynamic sequence + static branch; no FE block"
        ),
        "objective": str(metadata.get("loss_mode")),
        "checkpoint_selection_rule": checkpoint_selection_rule,
        "analysis_decision_rule": analysis_decision_rule,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_rows": int(len(test_indices)),
        "test_access_policy": "loaded after checkpoint and validation decision rule were frozen",
    }
    _atomic_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def _load_prediction(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _test_fold_assignments(folds_dir: Path, n_folds: int, n_test: int) -> np.ndarray:
    assignments = np.full(n_test, -1, dtype=np.int16)
    for fold in range(n_folds):
        indices = _load_indices(folds_dir / f"fold_{fold}", "test")
        if int(indices.max()) >= n_test:
            raise ValueError(f"Fold {fold} contains a frozen-test row outside [0, {n_test})")
        if np.any(assignments[indices] != -1):
            raise ValueError(f"Frozen-test rows occur in more than one fold; conflict at fold {fold}")
        assignments[indices] = fold
    missing = np.flatnonzero(assignments < 0)
    if len(missing):
        raise ValueError(
            f"Fold test indices do not partition the frozen test set; missing_rows={len(missing)}"
        )
    return assignments


def _sample_alignment_keys(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"station_id", "time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{source} lacks alignment columns: {sorted(required - set(frame.columns))}")
    times = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(f"{source} contains {int(times.isna().sum())} unparseable time values")
    keys = pd.DataFrame(
        {
            "station_key": _canonical_station_ids(frame["station_id"]),
            "time_ns_utc": times.astype("int64").to_numpy(dtype=np.int64),
        }
    )
    keys["duplicate_index"] = keys.groupby(
        ["time_ns_utc", "station_key"], sort=False
    ).cumcount()
    return keys


def _load_aligned_ifs_baseline(
    path: Path,
    meta_test: pd.DataFrame,
    reference_y: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    y_column = "y_true" if "y_true" in header else "y_cls" if "y_cls" in header else ""
    required = {"station_id", "time", "ifs_diagnostic_vis_m"}
    if not y_column:
        required.add("y_true")
    if not required.issubset(header):
        raise ValueError(f"IFS per-sample CSV lacks columns: {sorted(required - set(header))}")
    valid_column = "ifs_diagnostic_valid" if "ifs_diagnostic_valid" in header else ""
    usecols = ["station_id", "time", y_column, "ifs_diagnostic_vis_m"]
    prediction_column = "ifs_diagnostic_pred" if "ifs_diagnostic_pred" in header else ""
    if prediction_column:
        usecols.append(prediction_column)
    if valid_column:
        usecols.append(valid_column)
    baseline = pd.read_csv(path, usecols=usecols)
    if baseline.empty:
        raise ValueError(f"IFS per-sample CSV is empty: {path}")

    main_keys = _sample_alignment_keys(meta_test, "frozen meta_test.csv")
    main_keys["row_index"] = np.arange(len(main_keys), dtype=np.int64)
    baseline_keys = _sample_alignment_keys(baseline, str(path))
    baseline_keys["baseline_y"] = pd.to_numeric(baseline[y_column], errors="coerce")
    baseline_keys["ifs_diagnostic_vis_m"] = pd.to_numeric(
        baseline["ifs_diagnostic_vis_m"], errors="coerce"
    )
    if prediction_column:
        baseline_keys["source_ifs_diagnostic_pred"] = pd.to_numeric(
            baseline[prediction_column], errors="coerce"
        )
    if valid_column:
        baseline_keys["ifs_diagnostic_valid"] = (
            baseline[valid_column].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    else:
        baseline_keys["ifs_diagnostic_valid"] = True

    aligned = baseline_keys.merge(
        main_keys,
        on=["time_ns_utc", "station_key", "duplicate_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = aligned["row_index"].isna()
    if unmatched.any():
        raise ValueError(
            "IFS per-sample CSV is not a subset of the frozen test metadata; "
            f"unmatched_rows={int(unmatched.sum())}"
        )
    aligned["row_index"] = aligned["row_index"].astype(np.int64)
    row_indices = aligned["row_index"].to_numpy(dtype=np.int64)
    expected_rows = np.arange(len(meta_test), dtype=np.int64)
    if len(aligned) != len(meta_test) or not np.array_equal(
        np.sort(row_indices), expected_rows
    ):
        raise ValueError(
            "IFS per-sample CSV must cover every frozen-test row exactly once before "
            f"the diagnostic-valid mask is applied; source_rows={len(aligned)} "
            f"frozen_test_rows={len(meta_test)}"
        )
    baseline_y = aligned["baseline_y"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_y).all():
        raise ValueError("IFS per-sample CSV contains non-finite observed class labels")
    baseline_y_int = baseline_y.astype(np.int64)
    if not np.array_equal(baseline_y, baseline_y_int.astype(np.float64)):
        raise ValueError("IFS per-sample observed labels are not integer classes")
    expected = np.asarray(reference_y, dtype=np.int64)[row_indices]
    mismatch = baseline_y_int != expected
    if mismatch.any():
        raise ValueError(
            "IFS per-sample labels do not match frozen-test labels after exact alignment; "
            f"mismatched_rows={int(mismatch.sum())}"
        )

    valid = aligned["ifs_diagnostic_valid"].to_numpy(dtype=bool)
    diagnostic_vis = aligned["ifs_diagnostic_vis_m"].to_numpy(dtype=np.float64)
    valid &= np.isfinite(diagnostic_vis)
    if not np.any(valid):
        raise ValueError("IFS diagnostic baseline has no finite matched visibility values")
    recomputed_pred = np.full(len(aligned), 2, dtype=np.int64)
    recomputed_pred[diagnostic_vis < 1000.0] = 1
    recomputed_pred[diagnostic_vis < 500.0] = 0
    if prediction_column:
        source_pred = aligned["source_ifs_diagnostic_pred"].to_numpy(dtype=np.float64)
        source_valid = valid & np.isfinite(source_pred)
        if not np.all(source_valid[valid]):
            raise ValueError("IFS diagnostic class is missing for a visibility-valid row")
        if not np.array_equal(source_pred[valid], recomputed_pred[valid].astype(np.float64)):
            raise ValueError(
                "IFS diagnostic classes disagree with classes recomputed from native "
                "visibility at 500 m and 1000 m"
            )

    result = pd.DataFrame(
        {
            "row_index": row_indices[valid],
            "station_id": aligned.loc[valid, "station_key"].astype(str).to_numpy(),
            "y_true": baseline_y_int[valid],
            "ifs_diagnostic_vis_m": diagnostic_vis[valid],
            "ifs_diagnostic_pred": recomputed_pred[valid],
        }
    ).sort_values("row_index", kind="stable")
    if result["row_index"].duplicated().any():
        raise ValueError("IFS baseline maps more than once to a frozen-test row")
    provenance = {
        "path": str(path),
        "source_rows": int(len(baseline)),
        "aligned_rows": int(len(aligned)),
        "valid_matched_rows": int(len(result)),
        "frozen_test_rows": int(len(meta_test)),
        "valid_coverage": float(len(result) / max(len(meta_test), 1)),
        "alignment_keys": ["time_utc", "station_id", "duplicate_index_within_key"],
        "source_covers_frozen_test_exactly_once": True,
        "label_match_verified": True,
        "decision_rule": "recomputed from native IFS VIS: fog <500 m, mist <1000 m",
    }
    return result, provenance


def validate_ifs_baseline(args: argparse.Namespace) -> None:
    """Fail closed unless an IFS diagnostic file exactly matches frozen test rows."""
    data_dir = Path(args.data_dir).resolve()
    ifs_path = Path(args.ifs_csv).resolve()
    output_json = Path(args.output_json).resolve()
    meta_test = pd.read_csv(
        data_dir / "meta_test.csv", usecols=["station_id", "time"]
    )
    y_test_source = np.load(data_dir / "y_test.npy", mmap_mode="r")
    if len(meta_test) != len(y_test_source):
        raise ValueError(
            "Frozen test metadata and labels have different row counts; "
            f"meta={len(meta_test)} labels={len(y_test_source)}"
        )
    _, reference_y = rnn.visibility_to_labels(np.asarray(y_test_source))
    _, provenance = _load_aligned_ifs_baseline(
        ifs_path,
        meta_test,
        np.asarray(reference_y, dtype=np.int64),
    )
    payload = {
        "schema_version": 1,
        "status": "compatible",
        "data_dir": str(data_dir),
        "test_rows": int(len(meta_test)),
        "ifs_baseline": provenance,
        "verified_utc": pd.Timestamp.utcnow().isoformat(),
    }
    _atomic_json(output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def _parse_models(raw_models: str) -> List[str]:
    models = [value.strip() for value in raw_models.split(",") if value.strip()]
    if not models:
        raise ValueError("At least one learned mapping model is required")
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unsupported mapping models: {unknown}; allowed={list(MODELS)}")
    duplicates = sorted({model for model in models if models.count(model) > 1})
    if duplicates:
        raise ValueError(f"Duplicate mapping models are not allowed: {duplicates}")
    return models


def _parse_seed_list(raw_seeds: str) -> List[int]:
    """Parse an optional, ordered P13 seed list without accepting duplicates."""

    values = [
        value.strip()
        for value in str(raw_seeds or "").replace(":", ",").split(",")
        if value.strip()
    ]
    if not values:
        return []
    try:
        seeds = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"Invalid seed list: {raw_seeds!r}") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative: {seeds}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def _load_fold_seed_ensemble(
    fold_output: Path,
    model: str,
    fold: int,
    seeds: Sequence[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load one fold's P13 member outputs and average post-softmax probabilities.

    A CV test fold is valid only when all member outputs carry the same frozen
    row indices and observed labels.  Keeping this check here prevents a
    same-length but differently ordered seed output from being silently averaged.
    """

    payloads: List[Dict[str, np.ndarray]] = []
    checkpoints: List[str] = []
    run_configs: List[str] = []
    for seed in seeds:
        seed_dir = fold_output / f"seed_{int(seed)}"
        result_path = seed_dir / "result.json"
        prediction_path = seed_dir / "test_predictions.npz"
        if not result_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing completed P13 {model} fold {fold} seed {seed}: {seed_dir}"
            )
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if int(result.get("seed", -1)) != int(seed):
            raise ValueError(
                f"{model} fold {fold} seed directory {seed_dir} records "
                f"seed={result.get('seed')!r}"
            )
        if _saved_decision_rule(result) != "argmax":
            raise ValueError(
                f"P13 {model} fold {fold} seed {seed} was not selected with argmax"
            )
        payload = _load_prediction(prediction_path)
        required = {"row_index", "y_true", "probs", "pred"}
        missing = required - set(payload)
        if missing:
            raise KeyError(f"{prediction_path} is missing fields: {sorted(missing)}")
        probabilities = np.asarray(payload["probs"], dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(
                f"{prediction_path} has invalid class-probability shape {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all() or np.any(probabilities.sum(axis=1) <= 0):
            raise ValueError(f"{prediction_path} has invalid class probabilities")
        payloads.append(payload)
        checkpoints.append(str(result.get("checkpoint", "")))
        run_configs.append(str(result.get("run_config", "")))

    reference = payloads[0]
    row_index = np.asarray(reference["row_index"], dtype=np.int64)
    y_true = np.asarray(reference["y_true"], dtype=np.int64)
    for seed, payload in zip(seeds[1:], payloads[1:]):
        if not np.array_equal(np.asarray(payload["row_index"], dtype=np.int64), row_index):
            raise ValueError(f"P13 {model} fold {fold} seed {seed} row indices do not match")
        if not np.array_equal(np.asarray(payload["y_true"], dtype=np.int64), y_true):
            raise ValueError(f"P13 {model} fold {fold} seed {seed} labels do not match")

    mean_probs = np.mean(
        np.stack([np.asarray(payload["probs"], dtype=np.float64) for payload in payloads], axis=0),
        axis=0,
    )
    mean_probs /= mean_probs.sum(axis=1, keepdims=True)
    if not np.isfinite(mean_probs).all():
        raise ValueError(f"P13 {model} fold {fold} mean probabilities are non-finite")
    pred = _argmax_predictions(mean_probs)
    return (
        {
            "row_index": row_index,
            "y_true": y_true,
            "probs": mean_probs.astype(np.float32),
            # P13's stored decision rule is the argmax of the seed-mean softmax.
            "pred": pred,
        },
        {
            "member_seeds": [int(seed) for seed in seeds],
            "ensemble_size": int(len(seeds)),
            "combination": "equal_weight_mean_post_softmax_then_argmax",
            "checkpoints": checkpoints,
            "run_configs": run_configs,
        },
    )


def _station_metrics(
    model: str,
    row_index: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    station_ids: np.ndarray,
) -> List[Dict[str, object]]:
    frame = pd.DataFrame(
        {
            "row_index": row_index,
            "station_id": station_ids[row_index],
            "y_true": y_true,
            "pred": pred,
            "p_low": probs[:, 0] + probs[:, 1],
        }
    )
    rows: List[Dict[str, object]] = []
    for station_id, group in frame.groupby("station_id", sort=True):
        yt = group["y_true"].to_numpy(dtype=np.int64)
        yp = group["pred"].to_numpy(dtype=np.int64)
        p_low = group["p_low"].to_numpy(dtype=np.float64)
        metrics = rnn.build_metrics(yt, yp)
        metrics["low_vis_ap"] = (
            float(average_precision_score((yt <= 1).astype(np.int64), p_low))
            if np.unique(yt <= 1).size == 2
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "station_id": station_id,
                "n": int(len(group)),
                "n_low_vis": int(np.sum(yt <= 1)),
                **metrics,
            }
        )
    return rows


def aggregate_results(args: argparse.Namespace) -> None:
    folds_dir = Path(args.folds_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (folds_dir / "fold_manifest.json").open("r", encoding="utf-8") as handle:
        fold_manifest = json.load(handle)
    cv_kind = str(fold_manifest.get("cv_kind", "spatial"))
    fold_algorithm = str(fold_manifest.get("algorithm", "unknown"))
    n_folds = int(fold_manifest["n_folds"])
    data_dir = Path(fold_manifest["data_dir"])
    n_test = int(fold_manifest["data_shapes"]["test"]["x_shape"][0])
    meta_columns = ["station_id", "time"] if args.ifs_csv else ["station_id"]
    meta_test = pd.read_csv(data_dir / "meta_test.csv", usecols=meta_columns)
    station_ids = _canonical_station_ids(meta_test["station_id"])
    if len(station_ids) != n_test:
        raise ValueError("meta_test row count changed after fold construction")
    test_row_folds = _test_fold_assignments(folds_dir, n_folds, n_test)

    fold_rows: List[Dict[str, object]] = []
    pooled_rows: List[Dict[str, object]] = []
    station_rows: List[Dict[str, object]] = []
    decision_effect_rows: List[Dict[str, object]] = []
    checkpoint_rule_rows: List[Dict[str, object]] = []
    pooled_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    models = _parse_models(args.models)
    seeds = _parse_seed_list(args.seeds)
    ensemble_size = int(len(seeds) or 1)
    member_seed_text = ",".join(str(seed) for seed in seeds)
    if seeds and args.decision_rule != "argmax":
        raise ValueError(
            "Seed-mean P13 aggregation is defined as mean post-softmax probabilities "
            "followed by argmax; use --decision-rule argmax."
        )
    for model in models:
        if seeds and model == "logistic":
            raise ValueError("P13 seed aggregation is supported for neural MLP/GRU outputs only")
        pieces: List[Dict[str, np.ndarray]] = []
        for fold in range(n_folds):
            fold_output = results_dir / model / f"fold_{fold}"
            ensemble_info: Dict[str, object] = {}
            if seeds:
                payload, ensemble_info = _load_fold_seed_ensemble(
                    fold_output, model, fold, seeds
                )
                saved_rule = "argmax_mean_softmax"
                saved_pred = np.asarray(payload["pred"], dtype=np.int64)
                primary_pred = _argmax_predictions(np.asarray(payload["probs"], dtype=np.float64))
                result = {
                    "checkpoint_selection_rule": "argmax",
                    "analysis_decision_rule": "argmax",
                }
            else:
                result_path = fold_output / "result.json"
                prediction_path = fold_output / "test_predictions.npz"
                if not result_path.is_file() or not prediction_path.is_file():
                    raise FileNotFoundError(f"Missing completed {model} fold {fold}: {fold_output}")
                with result_path.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
                payload = _load_prediction(prediction_path)
                saved_rule = _saved_decision_rule(result)
                saved_pred = np.asarray(payload["pred"], dtype=np.int64)
                primary_pred = _primary_predictions(payload, args.decision_rule)
            y_true = np.asarray(payload["y_true"], dtype=np.int64)
            probs = np.asarray(payload["probs"], dtype=np.float64)
            fold_metrics = _metrics_with_probabilities(y_true, primary_pred, probs)
            fold_rows.append(
                {
                    "cv_kind": cv_kind,
                    "sample_scope": "full_frozen_test_for_learned_operators",
                    "model": model,
                    "fold": fold,
                    "decision_rule": args.decision_rule,
                    "ensemble_size": int(ensemble_info.get("ensemble_size", 1)),
                    "member_seeds": ",".join(
                        str(seed) for seed in ensemble_info.get("member_seeds", [])
                    ),
                    **fold_metrics,
                }
            )
            decision_effect_rows.extend(
                _decision_effect_rows(
                    cv_kind,
                    "fold",
                    model,
                    fold,
                    saved_rule,
                    y_true,
                    probs,
                    saved_pred,
                )
            )
            checkpoint_rule_rows.append(
                {
                    "cv_kind": cv_kind,
                    "model": model,
                    "fold": fold,
                    "checkpoint_selection_rule": str(
                        result.get("checkpoint_selection_rule", saved_rule)
                    ),
                    "saved_prediction_rule": saved_rule,
                    "analysis_decision_rule": args.decision_rule,
                    "ensemble_size": int(ensemble_info.get("ensemble_size", 1)),
                    "member_seeds": ",".join(
                        str(seed) for seed in ensemble_info.get("member_seeds", [])
                    ),
                }
            )
            payload["saved_pred"] = saved_pred
            payload["argmax_pred"] = _argmax_predictions(probs)
            payload["pred"] = primary_pred
            pieces.append(payload)
        combined = {name: np.concatenate([piece[name] for piece in pieces], axis=0) for name in pieces[0]}
        order = np.argsort(combined["row_index"], kind="stable")
        combined = {name: values[order] for name, values in combined.items()}
        expected = np.arange(n_test, dtype=np.int64)
        if not np.array_equal(combined["row_index"].astype(np.int64), expected):
            raise ValueError(f"{model} out-of-fold predictions do not partition the frozen test rows exactly once")
        metrics = _metrics_with_probabilities(
            combined["y_true"].astype(np.int64),
            combined["pred"].astype(np.int64),
            combined["probs"].astype(np.float64),
        )
        pooled_rows.append(
            {
                "cv_kind": cv_kind,
                "sample_scope": "full_frozen_test_for_learned_operators",
                "model": model,
                "n": n_test,
                "decision_rule": args.decision_rule,
                "ensemble_size": ensemble_size,
                "member_seeds": member_seed_text,
                **metrics,
            }
        )
        saved_rules = sorted(
            {
                str(row["saved_prediction_rule"])
                for row in checkpoint_rule_rows
                if row["model"] == model
            }
        )
        pooled_saved_rule = saved_rules[0] if len(saved_rules) == 1 else "mixed"
        decision_effect_rows.extend(
            _decision_effect_rows(
                cv_kind,
                "pooled",
                model,
                None,
                pooled_saved_rule,
                combined["y_true"].astype(np.int64),
                combined["probs"].astype(np.float64),
                combined["saved_pred"].astype(np.int64),
            )
        )
        station_metrics = _station_metrics(
            model,
            combined["row_index"].astype(np.int64),
            combined["y_true"].astype(np.int64),
            combined["pred"].astype(np.int64),
            combined["probs"].astype(np.float64),
            station_ids,
        )
        station_rows.extend(
            {
                "cv_kind": cv_kind,
                "sample_scope": "full_frozen_test_for_learned_operators",
                "decision_rule": args.decision_rule,
                "ensemble_size": ensemble_size,
                "member_seeds": member_seed_text,
                **row,
            }
            for row in station_metrics
        )
        pooled_by_model[model] = combined

    matched_fold_rows: List[Dict[str, object]] = []
    matched_pooled_rows: List[Dict[str, object]] = []
    ifs_provenance: Optional[Dict[str, object]] = None
    if args.ifs_csv:
        ifs_path = Path(args.ifs_csv).resolve()
        reference_model = models[0]
        reference_y = pooled_by_model[reference_model]["y_true"].astype(np.int64)
        ifs, ifs_provenance = _load_aligned_ifs_baseline(
            ifs_path,
            meta_test,
            reference_y,
        )
        ifs_row_index = ifs["row_index"].to_numpy(dtype=np.int64)
        ifs_y = ifs["y_true"].to_numpy(dtype=np.int64)
        ifs_pred = ifs["ifs_diagnostic_pred"].to_numpy(dtype=np.int64)
        ifs_fold = test_row_folds[ifs_row_index].astype(np.int64)
        per_fold_rows = {str(fold): int(np.sum(ifs_fold == fold)) for fold in range(n_folds)}
        empty_folds = [fold for fold, count in per_fold_rows.items() if count == 0]
        if empty_folds:
            raise ValueError(f"IFS diagnostic baseline has no matched rows in folds: {empty_folds}")
        ifs_provenance["matched_rows_by_fold"] = per_fold_rows
        ifs_provenance["common_sample_policy"] = (
            "All plotted operators are restricted to the same IFS-diagnostic-valid frozen-test rows"
        )

        for fold in range(n_folds):
            fold_mask = ifs_fold == fold
            fold_indices = ifs_row_index[fold_mask]
            fold_y = ifs_y[fold_mask]
            for model in models:
                combined = pooled_by_model[model]
                model_y = combined["y_true"][fold_indices].astype(np.int64)
                if not np.array_equal(model_y, fold_y):
                    raise ValueError(f"{model} labels differ from IFS-matched labels in fold {fold}")
                model_metrics = _metrics_with_probabilities(
                    model_y,
                    combined["pred"][fold_indices].astype(np.int64),
                    combined["probs"][fold_indices].astype(np.float64),
                )
                matched_fold_rows.append(
                    {
                        "cv_kind": cv_kind,
                        "sample_scope": "ifs_diagnostic_matched_test",
                        "model": model,
                        "fold": fold,
                        "n": int(len(fold_indices)),
                        "decision_rule": args.decision_rule,
                        "ensemble_size": ensemble_size,
                        "member_seeds": member_seed_text,
                        **model_metrics,
                    }
                )
            ifs_metrics = rnn.build_metrics(fold_y, ifs_pred[fold_mask])
            matched_fold_rows.append(
                {
                    "cv_kind": cv_kind,
                    "sample_scope": "ifs_diagnostic_matched_test",
                    "model": "ifs_native",
                    "fold": fold,
                    "n": int(len(fold_indices)),
                    "decision_rule": "native_visibility_500_1000m",
                    "low_vis_ap": float("nan"),
                    **ifs_metrics,
                }
            )

        for model in models:
            combined = pooled_by_model[model]
            model_y = combined["y_true"][ifs_row_index].astype(np.int64)
            if not np.array_equal(model_y, ifs_y):
                raise ValueError(f"{model} pooled labels differ from IFS-matched labels")
            model_metrics = _metrics_with_probabilities(
                model_y,
                combined["pred"][ifs_row_index].astype(np.int64),
                combined["probs"][ifs_row_index].astype(np.float64),
            )
            matched_pooled_rows.append(
                {
                    "cv_kind": cv_kind,
                    "sample_scope": "ifs_diagnostic_matched_test",
                    "model": model,
                    "n": int(len(ifs_row_index)),
                    "decision_rule": args.decision_rule,
                    "ensemble_size": ensemble_size,
                    "member_seeds": member_seed_text,
                    **model_metrics,
                }
            )
        matched_pooled_rows.append(
            {
                "cv_kind": cv_kind,
                "sample_scope": "ifs_diagnostic_matched_test",
                "model": "ifs_native",
                "n": int(len(ifs_row_index)),
                "decision_rule": "native_visibility_500_1000m",
                "low_vis_ap": float("nan"),
                **rnn.build_metrics(ifs_y, ifs_pred),
            }
        )

    full_fold_table = pd.DataFrame(fold_rows)
    full_pooled_table = pd.DataFrame(pooled_rows)
    if ifs_provenance is not None:
        full_fold_table.to_csv(output_dir / "fold_metrics_full_learned_test.csv", index=False)
        full_pooled_table.to_csv(output_dir / "pooled_metrics_full_learned_test.csv", index=False)
        fold_table = pd.DataFrame(matched_fold_rows)
        pooled_table = pd.DataFrame(matched_pooled_rows)
    else:
        fold_table = full_fold_table
        pooled_table = full_pooled_table
    station_table = pd.DataFrame(station_rows)
    fold_table.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_table.to_csv(output_dir / "pooled_metrics.csv", index=False)
    station_table.to_csv(output_dir / "station_metrics.csv", index=False)
    pd.DataFrame(decision_effect_rows).to_csv(
        output_dir / "decision_rule_effects.csv", index=False
    )
    checkpoint_rule_table = pd.DataFrame(checkpoint_rule_rows)
    checkpoint_rule_table.to_csv(output_dir / "checkpoint_decision_rules.csv", index=False)

    delta_rows: List[Dict[str, object]] = []
    metrics_to_compare = ["low_vis_ap", "low_vis_precision", "low_vis_recall", "low_vis_csi", "false_positive_rate"]
    pooled_indexed = pooled_table.set_index("model")
    comparison_scope = (
        str(pooled_table["sample_scope"].iloc[0])
        if "sample_scope" in pooled_table.columns and len(pooled_table)
        else "unknown"
    )
    comparisons = [("mlp", "logistic"), ("gru", "mlp"), ("gru", "logistic")]
    if "ifs_native" in pooled_indexed.index:
        comparisons.extend((model, "ifs_native") for model in models)
    for left, right in comparisons:
        if left not in pooled_indexed.index or right not in pooled_indexed.index:
            continue
        for metric in metrics_to_compare:
            if metric == "low_vis_ap" and right == "ifs_native":
                continue
            delta_rows.append(
                {
                    "cv_kind": cv_kind,
                    "sample_scope": comparison_scope,
                    "comparison": f"{left}_minus_{right}",
                    "decision_rule": args.decision_rule,
                    "metric": metric,
                    "delta": float(pooled_indexed.loc[left, metric] - pooled_indexed.loc[right, metric]),
                }
            )
    pd.DataFrame(delta_rows).to_csv(output_dir / "operator_deltas.csv", index=False)
    checkpoint_rules = sorted(
        checkpoint_rule_table["checkpoint_selection_rule"].astype(str).unique().tolist()
    )
    coverage = {
        "schema_version": 3,
        "cv_kind": cv_kind,
        "fold_algorithm": fold_algorithm,
        "n_folds": n_folds,
        "frozen_test_rows": n_test,
        "models": models + (["ifs_native"] if ifs_provenance is not None else []),
        "seed_ensemble": {
            "enabled": bool(seeds),
            "member_seeds": [int(seed) for seed in seeds],
            "combination": (
                "equal_weight_mean_post_softmax_then_argmax" if seeds else "single_checkpoint"
            ),
        },
        "each_learned_model_partitions_test_exactly_once": True,
        "analysis_decision_rule": args.decision_rule,
        "checkpoint_selection_rules": checkpoint_rules,
        "all_checkpoints_selected_with_argmax": checkpoint_rules == ["argmax"],
        "threshold_policy": (
            "probability thresholds are not used in primary predictions"
            if args.decision_rule == "argmax"
            else "use each fold's saved validation-frozen decision rule"
        ),
        "primary_sample_scope": (
            "ifs_diagnostic_matched_test"
            if ifs_provenance is not None
            else "full_frozen_test_for_learned_operators"
        ),
        "ifs_baseline": {
            "included": ifs_provenance is not None,
            **(ifs_provenance or {}),
        },
        "primary_metrics": ["pooled out-of-fold low_vis_csi", "pooled out-of-fold low_vis_recall"],
        "full_learned_sensitivity_tables": (
            ["fold_metrics_full_learned_test.csv", "pooled_metrics_full_learned_test.csv"]
            if ifs_provenance is not None
            else []
        ),
        "decision_rule_effects": "decision_rule_effects.csv compares argmax with each fold's saved predictions",
        "station_metric_policy": (
            "learned-operator station metrics use the full frozen test; rows with no "
            "positive/negative variation retain NaN AP"
        ),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    _atomic_json(output_dir / "coverage_manifest.json", coverage)
    print(pooled_table.to_csv(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Create deterministic spatial fold row indices")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--buffer-km", type=float, default=50.0)
    p.add_argument("--min-fold-stations", type=int, default=100)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=prepare_folds)

    p = sub.add_parser("train-logistic", help="Fit/evaluate one spatial-fold logistic baseline")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--fold-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-rows", type=int, default=65_536)
    p.add_argument("--scaler-sample-rows", type=int, default=200_000)
    p.add_argument("--fog-ratio", type=float, default=0.18)
    p.add_argument("--mist-ratio", type=float, default=0.22)
    p.add_argument(
        "--decision-rule",
        choices=DECISION_RULES,
        default="argmax",
        help="Validation/checkpoint and test classification rule for the logistic operator.",
    )
    p.add_argument("--min-fog-precision", type=float, default=0.10)
    p.add_argument("--min-mist-precision", type=float, default=0.10)
    p.add_argument("--min-clear-recall", type=float, default=0.88)
    p.add_argument("--threshold-grid-low", type=float, default=0.10)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--threshold-grid-step", type=float, default=0.03)
    p.set_defaults(func=train_logistic)

    p = sub.add_parser("evaluate-neural", help="Evaluate one validation-frozen MLP/GRU checkpoint")
    p.add_argument("--model", choices=["mlp", "gru"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--fold-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-config", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument(
        "--decision-rule",
        choices=["checkpoint", *DECISION_RULES],
        default="checkpoint",
        help="Use the checkpoint rule or explicitly override inference classification.",
    )
    p.set_defaults(func=evaluate_neural)

    p = sub.add_parser("aggregate", help="Aggregate five out-of-fold operator results")
    p.add_argument("--folds-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument(
        "--seeds",
        default="",
        help=(
            "Optional comma/colon-separated neural seed ensemble. Each fold is read from "
            "results/<model>/fold_<k>/seed_<seed>/ and combined by mean post-softmax then argmax."
        ),
    )
    p.add_argument("--ifs-csv", default="")
    p.add_argument(
        "--decision-rule",
        choices=DECISION_RULES,
        default="argmax",
        help="Primary OOF analysis rule; argmax recomputes labels from saved probabilities.",
    )
    p.set_defaults(func=aggregate_results)

    p = sub.add_parser(
        "validate-ifs-baseline",
        help="Verify exact frozen-test alignment of the native IFS diagnostic baseline",
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ifs-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.set_defaults(func=validate_ifs_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    args.func(args)
    print(f"[Done] command={args.command} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
