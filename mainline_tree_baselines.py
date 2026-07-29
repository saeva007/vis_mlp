#!/usr/bin/env python3
"""Leakage-safe RF/XGBoost/LightGBM replacements for the Static-MLP+GRU.

The module has three subcommands:

``prepare``
    Convert the established flat 12-hour mainline dataset into a shared,
    tree-native cache.  Log transforms, missing-value medians, and the
    vegetation vocabulary are learned from the training split only.

``train``
    Fit one fixed, literature-informed tree baseline.  Validation is used only
    for boosting early stopping; test labels are touched only after fitting.

``summarize``
    Collect completed model metrics into a compact comparison CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CONTRACT_VERSION = "mainline_tree_features_v1_20260729"
CLASS_NAMES = ("Fog", "Mist", "Clear")
MAINLINE_CLASS_WEIGHTS = np.asarray([2.0, 2.0, 0.8], dtype=np.float32)
LOG_DYNAMIC_NAMES = {"PRECIP", "SW_RAD", "CAPE", "PM10", "PM25"}
DEFAULT_BASE = Path("/public/home/putianshu/vis_mlp")
DEFAULT_DATA_DIR = DEFAULT_BASE / "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_CACHE_DIR = DEFAULT_BASE / "tree_baseline_cache" / "s2_monthtail_w12_v1"
DEFAULT_RUN_ROOT = DEFAULT_BASE / "tree_baseline_runs"
DEFAULT_SEED = 20260729


def normalize_feature_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .upper()
        .replace(".", "")
        .replace("_UGM3", "")
        .replace("PM2P5", "PM25")
        .replace("PM2_5", "PM25")
    )


@dataclass(frozen=True)
class FlatLayout:
    window_size: int
    dyn_vars: int
    fe_dim: int
    dynamic_feature_order: Tuple[str, ...]

    @property
    def dynamic_dim(self) -> int:
        return self.window_size * self.dyn_vars

    @property
    def static_dim(self) -> int:
        return 5

    @property
    def vegetation_index(self) -> int:
        return self.dynamic_dim + self.static_dim

    @property
    def expected_width(self) -> int:
        return self.dynamic_dim + self.static_dim + 1 + self.fe_dim


def _load_dataset_config(data_dir: Path) -> Tuple[Dict[str, object], Optional[Path]]:
    merged: Dict[str, object] = {}
    chosen: Optional[Path] = None
    for name in ("dataset_split_config.json", "dataset_build_config.json", "dataset_metadata.json"):
        path = data_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot parse {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        merged.update(payload)
        chosen = path
    return merged, chosen


def infer_layout(data_dir: Path, window_size: int) -> FlatLayout:
    x_path = data_dir / "X_train.npy"
    if not x_path.is_file():
        raise FileNotFoundError(x_path)
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{x_path} must be two-dimensional, got {shape}")
    width = int(shape[1])
    config, _ = _load_dataset_config(data_dir)
    order = config.get("dynamic_feature_order")
    dyn = config.get("dyn_vars", config.get("dyn_vars_count"))
    if isinstance(order, list):
        order_tuple = tuple(str(value) for value in order)
        if dyn is None:
            dyn = len(order_tuple)
    else:
        order_tuple = ()
    if dyn is None:
        candidates = []
        for value in (27, 26, 25, 24, 19, 18, 17):
            fe = width - int(window_size) * value - 6
            if 0 <= fe <= 128:
                candidates.append((value, fe))
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot uniquely infer flat layout from width={width}, window={window_size}; "
                f"candidates={candidates}. Add dyn_vars to the dataset config."
            )
        dyn, fe_dim = candidates[0]
    else:
        dyn = int(dyn)
        fe_dim = width - int(window_size) * dyn - 6
    if fe_dim < 0:
        raise ValueError(f"Negative engineered-feature dimension: width={width}, dyn={dyn}")
    if order_tuple and len(order_tuple) != dyn:
        raise ValueError(f"dynamic_feature_order has {len(order_tuple)} entries, expected {dyn}")
    if not order_tuple:
        order_tuple = tuple(f"dynamic_{idx:02d}" for idx in range(dyn))
    layout = FlatLayout(int(window_size), int(dyn), int(fe_dim), order_tuple)
    if layout.expected_width != width:
        raise ValueError(f"Resolved layout {layout} does not match width={width}")
    return layout


def visibility_to_labels(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("Visibility target contains no finite values")
    if float(np.nanmax(finite)) < 100.0:
        raw *= 1000.0
    if not np.all(np.isfinite(raw)):
        raise ValueError("Visibility target contains non-finite values")
    if np.any(raw < 0.0) or np.any(raw > 30000.0):
        raise ValueError(
            f"Visibility target violates 0-30000 m contract: min={float(raw.min())}, "
            f"max={float(raw.max())}"
        )
    labels = np.zeros(len(raw), dtype=np.int8)
    labels[raw >= 500.0] = 1
    labels[raw >= 1000.0] = 2
    return raw, labels


def dynamic_log_columns(layout: FlatLayout) -> np.ndarray:
    base = [
        idx
        for idx, name in enumerate(layout.dynamic_feature_order)
        if normalize_feature_name(name) in LOG_DYNAMIC_NAMES
    ]
    return np.asarray(
        [step * layout.dyn_vars + idx for step in range(layout.window_size) for idx in base],
        dtype=np.int64,
    )


def transform_numeric_in_place(values: np.ndarray, layout: FlatLayout, medians: Optional[np.ndarray]) -> None:
    log_cols = dynamic_log_columns(layout)
    if log_cols.size:
        block = values[:, log_cols]
        values[:, log_cols] = np.log1p(np.maximum(block, 0.0))
    veg_idx = layout.vegetation_index
    numeric_cols = np.r_[np.arange(veg_idx), np.arange(veg_idx + 1, values.shape[1])]
    numeric = values[:, numeric_cols]
    numeric[~np.isfinite(numeric)] = np.nan
    if medians is not None:
        missing = ~np.isfinite(numeric)
        if missing.any():
            replacement = np.asarray(medians, dtype=np.float32)[numeric_cols]
            rows, cols = np.nonzero(missing)
            numeric[rows, cols] = replacement[cols]
    values[:, numeric_cols] = numeric


def _training_vegetation_categories(x_train: np.ndarray, veg_idx: int, chunk_rows: int) -> List[int]:
    categories: set[int] = set()
    for start in range(0, len(x_train), chunk_rows):
        values = np.asarray(x_train[start : start + chunk_rows, veg_idx])
        finite = values[np.isfinite(values)]
        rounded = np.rint(finite)
        if not np.allclose(finite, rounded, atol=1e-5):
            raise ValueError("Vegetation column contains non-integral values")
        categories.update(int(value) for value in rounded.tolist())
    if not categories:
        raise ValueError("Training split has no finite vegetation categories")
    return sorted(categories)


def _training_medians(
    x_train: np.ndarray,
    layout: FlatLayout,
    sample_rows: int,
    seed: int,
) -> np.ndarray:
    n = len(x_train)
    sample_n = min(max(int(sample_rows), 1), n)
    rng = np.random.default_rng(seed)
    indices = np.arange(n) if sample_n == n else np.sort(rng.choice(n, size=sample_n, replace=False))
    sample = np.asarray(x_train[indices], dtype=np.float32).copy()
    transform_numeric_in_place(sample, layout, medians=None)
    sample[~np.isfinite(sample)] = np.nan
    medians = np.nanmedian(sample, axis=0).astype(np.float32)
    medians[layout.vegetation_index] = 0.0
    numeric_mask = np.ones(sample.shape[1], dtype=bool)
    numeric_mask[layout.vegetation_index] = False
    if not np.all(np.isfinite(medians[numeric_mask])):
        bad = np.flatnonzero(numeric_mask & ~np.isfinite(medians)).tolist()
        raise ValueError(f"Training-only median is undefined for columns {bad}")
    return medians


def build_feature_names(layout: FlatLayout, vegetation_categories: Sequence[int]) -> List[str]:
    names: List[str] = []
    for step in range(layout.window_size):
        for name in layout.dynamic_feature_order:
            names.append(f"t{step:02d}_{name}")
    names.extend(("lat_norm", "lon_norm", "orography", "orography_anom", "orography_std"))
    names.extend(f"veg_{int(value)}" for value in vegetation_categories)
    names.append("veg_unknown")
    names.extend(f"engineered_{idx:02d}" for idx in range(layout.fe_dim))
    return names


def prepare_feature_chunk(
    raw_values: np.ndarray,
    layout: FlatLayout,
    medians: np.ndarray,
    vegetation_categories: Sequence[int],
) -> np.ndarray:
    values = np.asarray(raw_values, dtype=np.float32).copy()
    transform_numeric_in_place(values, layout, medians=medians)
    veg_idx = layout.vegetation_index
    raw_veg = values[:, veg_idx]
    finite_veg = np.isfinite(raw_veg)
    rounded_veg = np.zeros(len(raw_veg), dtype=np.int64)
    rounded_veg[finite_veg] = np.rint(raw_veg[finite_veg]).astype(np.int64)
    one_hot = np.zeros((len(values), len(vegetation_categories) + 1), dtype=np.float32)
    known = np.zeros(len(values), dtype=bool)
    for column, category in enumerate(vegetation_categories):
        match = finite_veg & (rounded_veg == int(category))
        one_hot[match, column] = 1.0
        known |= match
    one_hot[~known, -1] = 1.0
    result = np.concatenate([values[:, :veg_idx], one_hot, values[:, veg_idx + 1 :]], axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("Prepared tree features still contain non-finite values")
    return result.astype(np.float32, copy=False)


def _file_record(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _config_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".partial.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def prepare_cache(args: argparse.Namespace) -> Dict[str, object]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    layout = infer_layout(data_dir, args.window_size)
    required = [data_dir / f"{stem}_{split}.npy" for split in ("train", "val", "test") for stem in ("X", "y")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Mainline dataset is incomplete: {missing}")
    shapes: Dict[str, Tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        x_shape = np.load(data_dir / f"X_{split}.npy", mmap_mode="r").shape
        y_shape = np.load(data_dir / f"y_{split}.npy", mmap_mode="r").shape
        if len(x_shape) != 2 or int(x_shape[1]) != layout.expected_width:
            raise ValueError(f"{split} X shape {x_shape} violates {layout}")
        if int(np.prod(y_shape)) != int(x_shape[0]):
            raise ValueError(f"{split} X/y length mismatch: {x_shape} vs {y_shape}")
        shapes[split] = (int(x_shape[0]), int(x_shape[1]))

    source_config, source_config_path = _load_dataset_config(data_dir)
    source_records = {path.name: _file_record(path) for path in required}
    source_signature = _config_sha256(
        {
            "source_records": source_records,
            "source_config": source_config,
            "layout": asdict(layout),
        }
    )
    complete_config_path = cache_dir / "preprocess_config.json"
    if complete_config_path.is_file() and not args.force:
        existing = json.loads(complete_config_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("source_signature") == source_signature
            and int(existing.get("row_limit", 0)) == int(args.row_limit)
        ):
            print(f"[prepare] cache hit: {cache_dir}", flush=True)
            return existing
        raise RuntimeError(
            f"Cache exists but does not match this source/row limit: {cache_dir}. "
            "Use a fresh cache directory; --force is intended only for controlled reconstruction."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.force:
        collisions = [
            cache_dir / f"{stem}_{split}.npy"
            for split in ("train", "val", "test")
            for stem in ("X", "y_cls", "y_raw")
            if (cache_dir / f"{stem}_{split}.npy").exists()
        ]
        if collisions:
            raise RuntimeError(f"Incomplete cache already contains outputs: {collisions}")

    x_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    categories = _training_vegetation_categories(x_train, layout.vegetation_index, args.chunk_rows)
    medians = _training_medians(x_train, layout, args.median_sample_rows, args.seed)
    feature_names = build_feature_names(layout, categories)
    row_counts: Dict[str, int] = {}
    class_counts: Dict[str, List[int]] = {}
    started = time.time()

    for split in ("train", "val", "test"):
        x_source = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
        y_source = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
        rows = len(x_source) if int(args.row_limit) <= 0 else min(len(x_source), int(args.row_limit))
        selected = (
            None
            if rows == len(x_source)
            else np.linspace(0, len(x_source) - 1, num=rows, dtype=np.int64)
        )
        row_counts[split] = int(rows)
        y_values = np.asarray(y_source[:rows] if selected is None else y_source[selected])
        y_raw, y_cls = visibility_to_labels(y_values)
        class_counts[split] = np.bincount(y_cls, minlength=3).astype(int).tolist()
        temporary = cache_dir / f"X_{split}.partial.npy"
        final = cache_dir / f"X_{split}.npy"
        target = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(rows, len(feature_names)),
        )
        for start in range(0, rows, args.chunk_rows):
            end = min(start + args.chunk_rows, rows)
            raw_chunk = (
                x_source[start:end]
                if selected is None
                else x_source[selected[start:end]]
            )
            target[start:end] = prepare_feature_chunk(
                raw_chunk,
                layout,
                medians,
                categories,
            )
            if start == 0 or end == rows or start % (args.chunk_rows * 20) == 0:
                print(f"[prepare] {split}: {end}/{rows}", flush=True)
        target.flush()
        del target
        os.replace(temporary, final)
        _atomic_save_array(cache_dir / f"y_raw_{split}.npy", y_raw)
        _atomic_save_array(cache_dir / f"y_cls_{split}.npy", y_cls)

    config: Dict[str, object] = {
        "status": "complete",
        "contract_version": CONTRACT_VERSION,
        "created_at_unix": time.time(),
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "source_signature": source_signature,
        "source_records": source_records,
        "source_config_path": str(source_config_path) if source_config_path else None,
        "source_config_sha256": _config_sha256(source_config),
        "layout": asdict(layout),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "vegetation_categories_train_only": categories,
        "vegetation_unknown_column": "veg_unknown",
        "log_dynamic_names": sorted(LOG_DYNAMIC_NAMES),
        "missing_value_policy": "training_sample_median",
        "median_sample_rows": min(int(args.median_sample_rows), len(x_train)),
        "medians": medians.astype(float).tolist(),
        "row_limit": int(args.row_limit),
        "row_limit_selection": "all_rows" if int(args.row_limit) <= 0 else "evenly_spaced_full_split_smoke_only",
        "row_counts": row_counts,
        "class_counts": class_counts,
        "split_use_contract": {
            "train": "fit preprocessing and models",
            "val": "boosting early stopping and reporting only",
            "test": "final reporting only; never fit or select",
        },
        "seed": int(args.seed),
        "elapsed_seconds": time.time() - started,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }
    temporary_config = complete_config_path.with_suffix(".partial.json")
    temporary_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_config, complete_config_path)
    print(json.dumps({"cache_dir": str(cache_dir), "row_counts": row_counts, "class_counts": class_counts}, indent=2))
    return config


def class_sample_weights(labels: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    if mode == "none":
        class_weights = np.ones(3, dtype=np.float32)
    elif mode == "mainline":
        class_weights = MAINLINE_CLASS_WEIGHTS.copy()
    else:
        raise ValueError(f"Unknown class-weight mode: {mode}")
    return class_weights[y].astype(np.float32), class_weights


def prior_correct_probabilities(probabilities: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(class_weights, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected [N,3] probabilities, got {probs.shape}")
    if weights.shape != (3,) or np.any(weights <= 0):
        raise ValueError(f"Expected three positive class weights, got {weights}")
    corrected = probs / weights[None, :]
    denom = corrected.sum(axis=1, keepdims=True)
    corrected = corrected / np.maximum(denom, 1e-15)
    return corrected.astype(np.float32)


def metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> Dict[str, object]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(y), 3):
        raise ValueError(f"Probability shape mismatch: y={y.shape}, probs={probs.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < -1e-7):
        raise ValueError("Probabilities contain invalid values")
    probs = np.maximum(probs, 0.0)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-15)
    prediction = np.argmax(probs, axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    eps = 1e-15
    metrics: Dict[str, object] = {
        "n": int(len(y)),
        "accuracy": float(np.mean(prediction == y)),
        "confusion_matrix": confusion.tolist(),
        "multi_logloss": float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], eps)))),
        "multiclass_brier": float(
            np.mean(np.sum((probs - np.eye(3, dtype=np.float64)[y]) ** 2, axis=1))
        ),
    }
    recalls: List[float] = []
    f1s: List[float] = []
    for cls, name in enumerate(CLASS_NAMES):
        tp = int(confusion[cls, cls])
        fn = int(confusion[cls, :].sum() - tp)
        fp = int(confusion[:, cls].sum() - tp)
        tn = int(confusion.sum() - tp - fn - fp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        csi = tp / max(tp + fp + fn, 1)
        far = fp / max(tp + fp, 1)
        metrics.update(
            {
                f"{name}_support": int(tp + fn),
                f"{name}_precision": float(precision),
                f"{name}_recall": float(recall),
                f"{name}_f1": float(f1),
                f"{name}_CSI": float(csi),
                f"{name}_FAR": float(far),
                f"Brier_{name}": float(np.mean((probs[:, cls] - (y == cls)) ** 2)),
                f"{name}_TP": tp,
                f"{name}_FP": fp,
                f"{name}_FN": fn,
                f"{name}_TN": tn,
            }
        )
        recalls.append(recall)
        f1s.append(f1)

    true_low = y <= 1
    pred_low = prediction <= 1
    low_tp = int(np.sum(true_low & pred_low))
    low_fp = int(np.sum(~true_low & pred_low))
    low_fn = int(np.sum(true_low & ~pred_low))
    low_tn = int(np.sum(~true_low & ~pred_low))
    metrics.update(
        {
            "macro_f1": float(np.mean(f1s)),
            "balanced_accuracy": float(np.mean(recalls)),
            "low_vis_precision": float(low_tp / max(low_tp + low_fp, 1)),
            "low_vis_recall": float(low_tp / max(low_tp + low_fn, 1)),
            "low_vis_CSI": float(low_tp / max(low_tp + low_fp + low_fn, 1)),
            "false_positive_rate": float(low_fp / max(low_fp + low_tn, 1)),
        }
    )
    confidence = probs.max(axis=1)
    correct = prediction == y
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    metrics["ECE"] = float(ece)
    return metrics


def literature_informed_parameters(model: str, threads: int, seed: int, args: argparse.Namespace) -> Dict[str, object]:
    if model == "rf":
        return {
            "n_estimators": int(args.rf_trees),
            "criterion": "gini",
            "max_depth": 22,
            "min_samples_split": 40,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "bootstrap": True,
            "max_samples": 0.65,
            "n_jobs": int(threads),
            "random_state": int(seed),
            "verbose": 1,
        }
    if model == "xgboost":
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "max_depth": 8,
            "min_child_weight": 20.0,
            "max_delta_step": 1.0,
            "eta": 0.03,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "gamma": 0.05,
            "alpha": 0.10,
            "lambda": 2.0,
            "max_bin": 256,
            "nthread": int(threads),
            "seed": int(seed),
        }
    if model == "lightgbm":
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 10,
            "min_data_in_leaf": 500,
            "min_sum_hessian_in_leaf": 10.0,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "lambda_l1": 0.10,
            "lambda_l2": 2.0,
            "min_gain_to_split": 0.01,
            "max_bin": 255,
            "num_threads": int(threads),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
            "verbosity": -1,
        }
    raise ValueError(model)


def _predict_in_chunks(
    output_path: Path,
    features: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    chunk_rows: int,
) -> np.ndarray:
    temporary = output_path.with_name(output_path.stem + ".partial.npy")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(features), 3))
    for start in range(0, len(features), chunk_rows):
        end = min(start + chunk_rows, len(features))
        result = np.asarray(predict_fn(np.asarray(features[start:end], dtype=np.float32)), dtype=np.float32)
        if result.shape != (end - start, 3):
            raise ValueError(f"Prediction shape {result.shape}, expected {(end - start, 3)}")
        target[start:end] = result
    target.flush()
    del target
    os.replace(temporary, output_path)
    return np.load(output_path, mmap_mode="r")


def _importance_rows(model_type: str, model: object, feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    scores = np.zeros(len(feature_names), dtype=np.float64)
    if model_type == "rf":
        scores[:] = np.asarray(getattr(model, "feature_importances_"), dtype=np.float64)
    elif model_type == "xgboost":
        raw = model.get_score(importance_type="gain")
        for key, value in raw.items():
            if str(key).startswith("f") and str(key)[1:].isdigit():
                idx = int(str(key)[1:])
                if 0 <= idx < len(scores):
                    scores[idx] = float(value)
    elif model_type == "lightgbm":
        scores[:] = np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64)
    total = scores.sum()
    normalized = scores / total if total > 0 else scores
    order = np.argsort(-normalized)
    return [(str(feature_names[idx]), float(normalized[idx])) for idx in order]


def train_model(args: argparse.Namespace) -> Dict[str, object]:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_config_path = cache_dir / "preprocess_config.json"
    if not cache_config_path.is_file():
        raise FileNotFoundError(f"Prepare the shared tree cache first: {cache_config_path}")
    cache_config = json.loads(cache_config_path.read_text(encoding="utf-8"))
    if cache_config.get("status") != "complete" or cache_config.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError(f"Unsafe or incomplete cache contract: {cache_config_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if (output_dir / "run_config.json").exists() and not args.force:
        existing = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[train] completed run already exists: {output_dir}", flush=True)
            return existing
        raise RuntimeError(f"Incomplete run exists: {output_dir}; use a fresh run id")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(cache_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(cache_dir / "y_cls_train.npy")
    x_val = np.load(cache_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(cache_dir / "y_cls_val.npy")
    x_test = np.load(cache_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(cache_dir / "y_cls_test.npy")
    feature_names = [str(value) for value in cache_config["feature_names"]]
    if x_train.shape[1] != len(feature_names):
        raise ValueError("Feature cache and feature-name count disagree")
    train_weight, class_weights = class_sample_weights(y_train, args.class_weight_mode)
    val_weight, _ = class_sample_weights(y_val, args.class_weight_mode)
    params = literature_informed_parameters(args.model, args.threads, args.seed, args)
    started = time.time()
    versions: Dict[str, str] = {"numpy": np.__version__}
    eval_history: Dict[str, object] = {}
    best_iteration: Optional[int] = None

    if args.model == "rf":
        from sklearn import __version__ as sklearn_version
        from sklearn.ensemble import RandomForestClassifier

        versions["scikit_learn"] = sklearn_version
        model = RandomForestClassifier(**params)
        print(f"[train:rf] rows={len(x_train)} features={x_train.shape[1]} params={params}", flush=True)
        model.fit(x_train, y_train, sample_weight=train_weight)
        if not np.array_equal(np.asarray(model.classes_), np.arange(3)):
            raise ValueError(f"RandomForest classes are {model.classes_}, expected [0,1,2]")
        raw_predict: Callable[[np.ndarray], np.ndarray] = model.predict_proba
        model_path = output_dir / "model.joblib"
        import joblib

        joblib.dump(model, model_path, compress=3)
    elif args.model == "xgboost":
        import xgboost as xgb

        versions["xgboost"] = xgb.__version__
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=train_weight, feature_names=feature_names)
        dval = xgb.DMatrix(x_val, label=y_val, weight=val_weight, feature_names=feature_names)
        print(f"[train:xgboost] rows={len(x_train)} features={x_train.shape[1]} params={params}", flush=True)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=int(args.num_boost_round),
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=eval_history,
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose_eval=50,
        )
        best_iteration = int(getattr(model, "best_iteration", int(args.num_boost_round) - 1))
        del dtrain, dval

        def raw_predict(block: np.ndarray) -> np.ndarray:
            matrix = xgb.DMatrix(block, feature_names=feature_names)
            try:
                return model.predict(matrix, iteration_range=(0, best_iteration + 1))
            except TypeError:
                return model.predict(matrix, ntree_limit=best_iteration + 1)

        model_path = output_dir / "model.json"
        model.save_model(model_path)
    elif args.model == "lightgbm":
        import lightgbm as lgb

        versions["lightgbm"] = lgb.__version__
        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            weight=train_weight,
            feature_name=feature_names,
            free_raw_data=True,
            params={"max_bin": int(params["max_bin"])},
        )
        val_set = lgb.Dataset(
            x_val,
            label=y_val,
            weight=val_weight,
            reference=train_set,
            feature_name=feature_names,
            free_raw_data=True,
        )
        print(f"[train:lightgbm] rows={len(x_train)} features={x_train.shape[1]} params={params}", flush=True)
        callbacks = [
            lgb.early_stopping(stopping_rounds=int(args.early_stopping_rounds), first_metric_only=True),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(eval_history),
        ]
        model = lgb.train(
            params,
            train_set,
            num_boost_round=int(args.num_boost_round),
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        best_iteration = int(model.best_iteration)

        def raw_predict(block: np.ndarray) -> np.ndarray:
            return model.predict(block, num_iteration=best_iteration)

        model_path = output_dir / "model.txt"
        model.save_model(str(model_path), num_iteration=best_iteration)
    else:
        raise ValueError(args.model)

    metrics: Dict[str, object] = {}
    for split, features, labels in (("val", x_val, y_val), ("test", x_test, y_test)):
        raw_path = output_dir / f"probs_{split}_weighted_raw.npy"
        raw_probs = _predict_in_chunks(raw_path, features, raw_predict, args.predict_chunk_rows)
        corrected_path = output_dir / f"probs_{split}.npy"
        corrected_target = np.lib.format.open_memmap(
            corrected_path.with_name(corrected_path.stem + ".partial.npy"),
            mode="w+",
            dtype=np.float32,
            shape=raw_probs.shape,
        )
        for start in range(0, len(raw_probs), args.predict_chunk_rows):
            end = min(start + args.predict_chunk_rows, len(raw_probs))
            corrected_target[start:end] = prior_correct_probabilities(raw_probs[start:end], class_weights)
        corrected_target.flush()
        del corrected_target
        os.replace(corrected_path.with_name(corrected_path.stem + ".partial.npy"), corrected_path)
        corrected = np.load(corrected_path, mmap_mode="r")
        metrics[f"{split}_weighted_raw"] = metrics_from_probabilities(labels, raw_probs)
        metrics[f"{split}_prior_corrected"] = metrics_from_probabilities(labels, corrected)

    shutil.copyfile(output_dir / "probs_test.npy", output_dir / "probs.npy")
    importance = _importance_rows(args.model, model, feature_names)
    with (output_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "feature", "normalized_gain"))
        for rank, (name, value) in enumerate(importance, start=1):
            writer.writerow((rank, name, f"{value:.12g}"))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if eval_history:
        (output_dir / "eval_history.json").write_text(
            json.dumps(eval_history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    flat_rows = []
    for split_variant, payload in metrics.items():
        row = {"split_variant": split_variant}
        row.update({key: value for key, value in payload.items() if np.isscalar(value)})
        flat_rows.append(row)
    keys = sorted({key for row in flat_rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)

    run_config: Dict[str, object] = {
        "status": "complete",
        "model": args.model,
        "model_path": str(model_path),
        "cache_dir": str(cache_dir),
        "cache_contract_version": cache_config.get("contract_version"),
        "cache_source_signature": cache_config.get("source_signature"),
        "data_dir": cache_config.get("data_dir"),
        "source_meta_test": str(Path(str(cache_config.get("data_dir"))) / "meta_test.csv"),
        "probability_file": "probs.npy",
        "probability_variant": "training_weight_prior_corrected",
        "weighted_raw_probability_files": {
            "val": "probs_val_weighted_raw.npy",
            "test": "probs_test_weighted_raw.npy",
        },
        "class_names": list(CLASS_NAMES),
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights.astype(float).tolist(),
        "decision_rule": "argmax; no validation or test threshold search",
        "prior_correction": "p_natural proportional to p_weighted divided by training class weight",
        "parameters": params,
        "num_boost_round": int(args.num_boost_round) if args.model != "rf" else None,
        "early_stopping_rounds": int(args.early_stopping_rounds) if args.model != "rf" else None,
        "best_iteration": best_iteration,
        "seed": int(args.seed),
        "threads": int(args.threads),
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
        "metrics_file": "metrics.json",
        "feature_importance_file": "feature_importance.csv",
    }
    temporary_config = output_dir / "run_config.partial.json"
    temporary_config.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_config, output_dir / "run_config.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model": args.model,
                "best_iteration": best_iteration,
                "test_prior_corrected": metrics["test_prior_corrected"],
            },
            indent=2,
        ),
        flush=True,
    )
    return run_config


def summarize_runs(args: argparse.Namespace) -> List[Dict[str, object]]:
    run_root = Path(args.run_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for config_path in sorted(run_root.glob("**/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status") != "complete":
            continue
        metrics_path = config_path.parent / str(config.get("metrics_file", "metrics.json"))
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for variant in ("val_prior_corrected", "test_prior_corrected", "test_weighted_raw"):
            payload = metrics.get(variant)
            if not isinstance(payload, dict):
                continue
            row: Dict[str, object] = {
                "run_dir": str(config_path.parent),
                "model": config.get("model"),
                "variant": variant,
                "best_iteration": config.get("best_iteration"),
            }
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "Fog_CSI",
                "Fog_precision",
                "Fog_recall",
                "Mist_CSI",
                "Mist_precision",
                "Mist_recall",
                "low_vis_precision",
                "low_vis_recall",
                "low_vis_CSI",
                "false_positive_rate",
                "multi_logloss",
                "ECE",
            ):
                row[key] = payload.get(key)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed tree runs found under {run_root}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] rows={len(rows)} csv={output_csv} json={output_json}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build shared, train-fitted tree feature cache")
    prepare.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    prepare.add_argument("--window-size", type=int, default=12)
    prepare.add_argument("--chunk-rows", type=int, default=32768)
    prepare.add_argument("--median-sample-rows", type=int, default=200000)
    prepare.add_argument("--row-limit", type=int, default=0, help="0 keeps every row; positive is smoke-only")
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--force", action="store_true")

    train = sub.add_parser("train", help="Train and evaluate one tree baseline")
    train.add_argument("--model", choices=("rf", "xgboost", "lightgbm"), required=True)
    train.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    train.add_argument("--output-dir", required=True)
    train.add_argument("--threads", type=int, default=32)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--class-weight-mode", choices=("mainline", "none"), default="mainline")
    train.add_argument("--rf-trees", type=int, default=400)
    train.add_argument("--num-boost-round", type=int, default=2500)
    train.add_argument("--early-stopping-rounds", type=int, default=100)
    train.add_argument("--predict-chunk-rows", type=int, default=100000)
    train.add_argument("--force", action="store_true")

    summary = sub.add_parser("summarize", help="Collect completed tree-run metrics")
    summary.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    summary.add_argument("--output-csv", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_cache(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "summarize":
        summarize_runs(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
