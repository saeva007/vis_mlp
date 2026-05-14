#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc diagnostics for the PM10+PM2.5 S2 low-visibility checkpoint.

This script does not train or modify a checkpoint.  It loads an existing
PMST checkpoint, runs inference on a split, and writes diagnostics for:
  - current three-class threshold decisions;
  - Mist -> Clear and Mist -> Fog error structure;
  - the unused binary low-visibility head (<1000 m);
  - binary-head gating as a possible post-processing path;
  - validation/test threshold sweeps and saved seasonal thresholds.

Default paths match the exp_1778563813_pm10_more_temp_search_utc run.
Run this from the remote checkout root:

  cd /public/home/putianshu/vis_mlp/train
  python diagnose_lowvis_checkpoint_pm10_pm25.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

torch = None
F = None


def ensure_torch():
    global torch, F
    if torch is None or F is None:
        import torch as _torch
        import torch.nn.functional as _F

        torch = _torch
        F = _F
    return torch, F


DEFAULT_RUN_ID = "exp_1778563813_pm10_more_temp_search_utc"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_MODEL_PY = "PMST_net_test_11_s2_pm10.py"
CLASS_NAMES = ("Fog", "Mist", "Clear")
SEASON_MAP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def abs_under_base(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def visibility_to_class(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_raw, dtype=np.float32).copy()
    finite = np.isfinite(y)
    if finite.any() and np.nanmax(y[finite]) < 100.0:
        y *= 1000.0
    cls = np.full(len(y), 2, dtype=np.int64)
    cls[y < 1000.0] = 1
    cls[y < 500.0] = 0
    cls[~np.isfinite(y)] = -1
    return cls, y


def infer_feature_layout(
    n_cols: int,
    window_size: int,
    dyn_vars_count: int = 0,
    extra_feat_dim: int = -1,
) -> Tuple[int, int]:
    if dyn_vars_count > 0:
        fe = n_cols - window_size * dyn_vars_count - 6
        if fe < 0:
            raise ValueError(
                f"Cannot infer layout: n_cols={n_cols}, dyn={dyn_vars_count}, "
                f"window={window_size} gives FE={fe}"
            )
        if extra_feat_dim >= 0 and fe != extra_feat_dim:
            raise ValueError(f"FE mismatch: inferred {fe}, expected {extra_feat_dim}")
        return int(dyn_vars_count), int(fe)

    candidates = [27, 26, 25, 24]
    plausible = []
    for dyn in candidates:
        fe = n_cols - window_size * dyn - 6
        if fe < 0:
            continue
        if extra_feat_dim >= 0 and fe != extra_feat_dim:
            continue
        if fe <= 96:
            plausible.append((dyn, fe))
    if not plausible:
        raise ValueError(
            f"Cannot infer dyn/FE layout from n_cols={n_cols}; pass "
            "--dyn-vars-count and --extra-feat-dim explicitly."
        )
    return plausible[0]


def dyn_log_indices(dyn_vars_count: int) -> List[int]:
    idxs = [2, 4, 9]
    if dyn_vars_count >= 27:
        idxs.extend([dyn_vars_count - 2, dyn_vars_count - 1])
    else:
        idxs.append(dyn_vars_count - 1)
    return [i for i in idxs if 0 <= i < dyn_vars_count]


def prepare_batch_rows(
    rows: np.ndarray,
    scaler,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
) -> np.ndarray:
    split_dyn = window_size * dyn_vars_count
    log_mask = np.zeros(split_dyn, dtype=bool)
    for t in range(window_size):
        for idx in dyn_log_indices(dyn_vars_count):
            log_mask[t * dyn_vars_count + idx] = True

    feats = rows[:, : split_dyn + 5].astype(np.float32, copy=True)
    feats[:, :split_dyn] = np.where(
        log_mask,
        np.log1p(np.maximum(feats[:, :split_dyn], 0.0)),
        feats[:, :split_dyn],
    )
    if scaler is not None:
        if len(scaler.center_) != feats.shape[1]:
            raise ValueError(
                f"Scaler dimension {len(scaler.center_)} does not match "
                f"feature block {feats.shape[1]}"
            )
        feats = (feats - scaler.center_) / (scaler.scale_ + 1e-6)

    veg = rows[:, split_dyn + 5 : split_dyn + 6].astype(np.float32, copy=False)
    extra = rows[:, split_dyn + 6 :].astype(np.float32, copy=True)
    if extra.shape[1] < extra_feat_dim:
        extra = np.pad(extra, ((0, 0), (0, extra_feat_dim - extra.shape[1])), mode="constant")
    elif extra.shape[1] > extra_feat_dim:
        extra = extra[:, :extra_feat_dim]

    final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)], axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def import_model_class(model_py: Path):
    spec = importlib.util.spec_from_file_location("pmst_lowvis_diag_model", str(model_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model script: {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ImprovedDualStreamPMSTNet


def load_checkpoint_into_model(model, ckpt_path: Path, device) -> None:
    torch, _ = ensure_torch()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object from {ckpt_path}: {type(state)}")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} first={missing[:5]}", flush=True)
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)


def load_torch_payload(path: Path):
    torch, _ = ensure_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup_runtime(args) -> Dict[str, object]:
    torch, _ = ensure_torch()
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = bool(args.distributed or env_world > 1)

    if distributed:
        if env_world <= 1:
            raise RuntimeError("--distributed was set but WORLD_SIZE<=1; launch with torchrun.")
        local_rank = args.local_rank if args.local_rank is not None else env_local_rank
        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            backend = "gloo"
        else:
            n_dev = max(torch.cuda.device_count(), 1)
            torch.cuda.set_device(local_rank % n_dev)
            device = torch.device("cuda", local_rank % n_dev)
            backend = "nccl"
        import torch.distributed as dist

        dist.init_process_group(backend=backend, init_method="env://")
        return {
            "distributed": True,
            "rank": env_rank,
            "world_size": env_world,
            "local_rank": local_rank,
            "device": device,
            "dist": dist,
        }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    return {
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "device": device,
        "dist": None,
    }


def shard_bounds(n_total: int, rank: int, world_size: int) -> Tuple[int, int]:
    start = (int(n_total) * int(rank)) // int(world_size)
    end = (int(n_total) * (int(rank) + 1)) // int(world_size)
    return start, end


def run_inference(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    start_index: int,
    end_index: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    torch, F = ensure_torch()
    X = np.load(x_path, mmap_mode="r")
    start_index = int(start_index)
    end_index = int(end_index)
    n = max(0, end_index - start_index)
    probs_out: List[np.ndarray] = []
    low_out: List[np.ndarray] = []
    model.eval()
    temp = max(float(temperature), 1e-6)

    for start in range(start_index, end_index, batch_size):
        end = min(start + batch_size, end_index)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_batch_rows(rows, scaler, window_size, dyn_vars_count, extra_feat_dim)
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            fine_logits, _, low_logit = model(x)
            probs = F.softmax(fine_logits / temp, dim=1)
            low_prob = torch.sigmoid(low_logit).reshape(-1)
        probs_out.append(probs.detach().cpu().numpy().astype(np.float32))
        low_out.append(low_prob.detach().cpu().numpy().astype(np.float32))
        done = end - start_index
        if start == start_index or end == end_index or (done // max(batch_size, 1)) % 20 == 0:
            print(f"[inference] local {done:,}/{n:,} global {end:,}", flush=True)

    if not probs_out:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(probs_out, axis=0), np.concatenate(low_out, axis=0)


def pred_from_thresholds_mutual(probs: np.ndarray, fog_th, mist_th) -> np.ndarray:
    p_fog = np.asarray(probs[:, 0], dtype=np.float64)
    p_mist = np.asarray(probs[:, 1], dtype=np.float64)
    fog_th = np.atleast_1d(np.asarray(fog_th, dtype=np.float64))
    mist_th = np.atleast_1d(np.asarray(mist_th, dtype=np.float64))
    if fog_th.size == 1:
        fog_th = np.full(probs.shape[0], fog_th.flat[0])
    if mist_th.size == 1:
        mist_th = np.full(probs.shape[0], mist_th.flat[0])
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    pred[(p_fog > fog_th) & (p_fog > p_mist)] = 0
    pred[(p_mist > mist_th) & (p_mist > p_fog)] = 1
    return pred


def pred_from_thresholds_default(probs: np.ndarray, fog_th, mist_th) -> np.ndarray:
    p_fog = probs[:, 0]
    p_mist = probs[:, 1]
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    pred[p_mist >= mist_th] = 1
    pred[p_fog >= fog_th] = 0
    return pred


def pred_from_joint_thresholds(probs: np.ndarray, fog_th, mist_th) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    fog_excess = (probs[:, 0] - fog_th) / max(1.0 - fog_th, 1e-6)
    mist_excess = (probs[:, 1] - mist_th) / max(1.0 - mist_th, 1e-6)
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    fog_on = fog_excess >= 0.0
    mist_on = mist_excess >= 0.0
    pred[fog_on & ~mist_on] = 0
    pred[mist_on & ~fog_on] = 1
    both = fog_on & mist_on
    pred[both] = np.where(fog_excess[both] >= mist_excess[both], 0, 1)
    return pred


def pred_from_rule(probs: np.ndarray, fog_th: float, mist_th: float, rule: str) -> np.ndarray:
    if rule == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64)
    if rule == "joint":
        return pred_from_joint_thresholds(probs, fog_th, mist_th)
    if rule == "default":
        return pred_from_thresholds_default(probs, fog_th, mist_th)
    return pred_from_thresholds_mutual(probs, fog_th, mist_th)


def binary_gate_pred(probs: np.ndarray, low_prob: np.ndarray, low_th: float) -> np.ndarray:
    pred = np.full(len(low_prob), 2, dtype=np.int64)
    low = low_prob >= low_th
    low_class = np.where(probs[:, 0] >= probs[:, 1], 0, 1)
    pred[low] = low_class[low]
    return pred


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    valid = (y_true >= 0) & (y_true <= 2) & (pred >= 0) & (pred <= 2)
    idx = y_true[valid].astype(np.int64) * 3 + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=9).reshape(3, 3)


def classification_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_counts(y_true, pred)
    n = int(cm.sum())
    out: Dict[str, float] = {"n": float(n), "accuracy": safe_div(float(np.trace(cm)), float(n))}
    for cid, name in enumerate(CLASS_NAMES):
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        support = float(cm[cid, :].sum())
        pred_count = float(cm[:, cid].sum())
        out[f"{name}_P"] = safe_div(tp, tp + fp)
        out[f"{name}_R"] = safe_div(tp, tp + fn)
        out[f"{name}_CSI"] = safe_div(tp, tp + fp + fn)
        out[f"{name}_FAR"] = safe_div(fp, tp + fp)
        out[f"{name}_support"] = support
        out[f"pred_{name.lower()}"] = pred_count

    true_low = y_true <= 1
    pred_low = pred <= 1
    true_clear = y_true == 2
    low_tp = float((true_low & pred_low).sum())
    low_fp = float((~true_low & pred_low).sum())
    low_fn = float((true_low & ~pred_low).sum())
    out["low_vis_precision"] = safe_div(low_tp, low_tp + low_fp)
    out["low_vis_recall"] = safe_div(low_tp, low_tp + low_fn)
    out["low_vis_csi"] = safe_div(low_tp, low_tp + low_fp + low_fn)
    out["false_positive_rate"] = safe_div(float((true_clear & pred_low).sum()), float(true_clear.sum()))
    out["balanced_acc"] = float(np.mean([out["Fog_R"], out["Mist_R"], out["Clear_R"]]))
    return out


def binary_metrics(y_true_low: np.ndarray, pred_low: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true_low, dtype=bool)
    p = np.asarray(pred_low, dtype=bool)
    tp = float((y & p).sum())
    fp = float((~y & p).sum())
    fn = float((y & ~p).sum())
    tn = float((~y & ~p).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "csi": safe_div(tp, tp + fp + fn),
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "f2": safe_div(5.0 * precision * recall, 4.0 * precision + recall),
        "fpr": safe_div(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def target_achievement(m: Dict[str, float]) -> float:
    return float(
        min(m["Fog_R"] / 0.65, 1.0) * 0.30
        + min(m["Mist_R"] / 0.75, 1.0) * 0.30
        + min(m["accuracy"] / 0.95, 1.0) * 0.25
        + min(m["low_vis_precision"] / 0.20, 1.0) * 0.10
        + min((1.0 - m["false_positive_rate"]) / 0.60, 1.0) * 0.05
    )


def add_metric_row(name: str, pred: np.ndarray, y_true: np.ndarray, extra: Optional[Dict[str, object]] = None):
    row = {"name": name}
    if extra:
        row.update({k: v for k, v in extra.items() if k != "name"})
    row.update(classification_metrics(y_true, pred))
    row["target_achievement"] = target_achievement(row)
    return row


def search_grid() -> np.ndarray:
    low_part = np.arange(0.10, 0.50, 0.04)
    high_part = np.arange(0.50, 0.96, 0.03)
    return np.unique(np.concatenate([low_part, high_part]))


def fine_threshold_sweep(
    probs: np.ndarray,
    y_true: np.ndarray,
    rule: str,
) -> pd.DataFrame:
    rows = []
    grid = search_grid()
    for f_th in grid:
        for m_th in grid:
            pred = pred_from_rule(probs, float(f_th), float(m_th), rule)
            row = add_metric_row(
                "fine_threshold",
                pred,
                y_true,
                {"fog_th": float(f_th), "mist_th": float(m_th), "threshold_rule": rule},
            )
            rows.append(row)
    return pd.DataFrame(rows)


def binary_sweeps(
    probs: np.ndarray,
    low_prob: np.ndarray,
    y_true: np.ndarray,
    fine_low_mass: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_low = y_true <= 1
    rows_bin = []
    rows_gate = []
    for th in np.round(np.linspace(0.01, 0.99, 99), 4):
        bm = binary_metrics(y_low, low_prob >= th)
        row = {"score_name": "binary_head", "threshold": float(th)}
        row.update(bm)
        rows_bin.append(row)

        fm = binary_metrics(y_low, fine_low_mass >= th)
        row_f = {"score_name": "fine_low_mass", "threshold": float(th)}
        row_f.update(fm)
        rows_bin.append(row_f)

        pred_gate = binary_gate_pred(probs, low_prob, float(th))
        rows_gate.append(add_metric_row("binary_gate_fine_argmax", pred_gate, y_true, {"binary_th": float(th)}))
    return pd.DataFrame(rows_bin), pd.DataFrame(rows_gate)


def quantile_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(list(group_cols), dropna=False, sort=True)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            for q in (0.1, 0.5, 0.9):
                row[f"{col}_q{int(q * 100):02d}"] = float(vals.quantile(q))
        rows.append(row)
    return pd.DataFrame(rows)


def load_months(data_dir: Path, split: str, X, window_size: int, dyn: int, fe: int, n: int) -> np.ndarray:
    meta_path = data_dir / f"meta_{split}.csv"
    if meta_path.exists():
        meta_cols = pd.read_csv(meta_path, nrows=0).columns.tolist()
        for col in ("month_analysis", "month"):
            if col in meta_cols:
                months = pd.read_csv(meta_path, usecols=[col])[col].to_numpy()
                return months[:n].astype(np.int64)
        for col in ("time_analysis", "time"):
            if col in meta_cols:
                times = pd.read_csv(meta_path, usecols=[col])[col]
                return pd.to_datetime(times, errors="coerce").dt.month.to_numpy()[:n].astype(np.int64)

    split_dyn = window_size * dyn
    extra_start = split_dyn + 6
    if fe < 4:
        raise FileNotFoundError(f"No usable meta_{split}.csv and FE dim < 4; cannot recover months.")
    month_sin = np.asarray(X[:n, extra_start + fe - 4], dtype=np.float64)
    month_cos = np.asarray(X[:n, extra_start + fe - 3], dtype=np.float64)
    angle = np.arctan2(month_sin, month_cos)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    month = np.round(angle * 6 / np.pi).astype(np.int64)
    month = np.where(month == 0, 12, month)
    return month


def load_months_from_features(X, start: int, end: int, window_size: int, dyn: int, fe: int) -> np.ndarray:
    split_dyn = window_size * dyn
    extra_start = split_dyn + 6
    if fe < 4:
        raise ValueError("FE dim < 4; cannot recover month_sin/month_cos from feature block.")
    month_sin = np.asarray(X[start:end, extra_start + fe - 4], dtype=np.float64)
    month_cos = np.asarray(X[start:end, extra_start + fe - 3], dtype=np.float64)
    angle = np.arctan2(month_sin, month_cos)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    month = np.round(angle * 6 / np.pi).astype(np.int64)
    return np.where(month == 0, 12, month)


def load_months_slice(data_dir: Path, split: str, X, window_size: int, dyn: int, fe: int, start: int, end: int) -> np.ndarray:
    for name in (f"month_{split}.npy", f"months_{split}.npy"):
        path = data_dir / name
        if path.exists():
            return np.asarray(np.load(path, mmap_mode="r")[start:end], dtype=np.int64)
    return load_months_from_features(X, start, end, window_size, dyn, fe)


def write_rank_part(
    part_dir: Path,
    rank: int,
    probs: np.ndarray,
    low_prob: np.ndarray,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    months: np.ndarray,
) -> Path:
    part_dir.mkdir(parents=True, exist_ok=True)
    path = part_dir / f"rank_{rank:05d}.npz"
    np.savez(
        path,
        probs=probs,
        low_prob=low_prob,
        y_cls=y_cls,
        y_raw=y_raw,
        months=months,
    )
    return path


def read_rank_parts(part_dir: Path, world_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs_l, low_l, cls_l, raw_l, months_l = [], [], [], [], []
    missing = []
    for rank in range(world_size):
        path = part_dir / f"rank_{rank:05d}.npz"
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(path) as z:
            probs_l.append(np.asarray(z["probs"], dtype=np.float32))
            low_l.append(np.asarray(z["low_prob"], dtype=np.float32))
            cls_l.append(np.asarray(z["y_cls"], dtype=np.int64))
            raw_l.append(np.asarray(z["y_raw"], dtype=np.float32))
            months_l.append(np.asarray(z["months"], dtype=np.int64))
    if missing:
        raise FileNotFoundError("Missing distributed rank part files: " + ", ".join(missing[:5]))
    return (
        np.concatenate(probs_l, axis=0),
        np.concatenate(low_l, axis=0),
        np.concatenate(cls_l, axis=0),
        np.concatenate(raw_l, axis=0),
        np.concatenate(months_l, axis=0),
    )


def load_season_thresholds(path: Optional[Path]) -> Tuple[Optional[dict], Optional[float]]:
    if path is None or not path.exists():
        return None, None
    payload = load_torch_payload(path)
    if not isinstance(payload, dict):
        return None, None
    temp = payload.get("temperature")
    if temp is not None:
        temp = float(temp)
    return payload.get("season_thresholds"), temp


def thresholds_from_seasons(months: np.ndarray, season_thresholds: dict, default_fog: float, default_mist: float):
    fog = np.full(len(months), float(default_fog), dtype=np.float64)
    mist = np.full(len(months), float(default_mist), dtype=np.float64)
    for i, month in enumerate(months):
        season = SEASON_MAP.get(int(month))
        if season and season in season_thresholds:
            fog[i] = float(season_thresholds[season]["fog_th"])
            mist[i] = float(season_thresholds[season]["mist_th"])
    return fog, mist


def season_comparison(
    probs: np.ndarray,
    y_true: np.ndarray,
    months: np.ndarray,
    global_pred: np.ndarray,
    season_pred: np.ndarray,
) -> pd.DataFrame:
    seasons = np.array([SEASON_MAP.get(int(m), "UNK") for m in months])
    rows = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        mask = seasons == season
        if not np.any(mask):
            continue
        global_m = classification_metrics(y_true[mask], global_pred[mask])
        season_m = classification_metrics(y_true[mask], season_pred[mask])
        row = {"season": season, "n": int(mask.sum())}
        for key in ("Fog_R", "Fog_P", "Fog_CSI", "Mist_R", "Mist_P", "Mist_CSI", "low_vis_precision", "low_vis_recall", "low_vis_csi", "false_positive_rate", "accuracy"):
            row[f"global_{key}"] = global_m[key]
            row[f"season_{key}"] = season_m[key]
            row[f"delta_{key}"] = season_m[key] - global_m[key]
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    args,
    summary: Dict[str, object],
    metric_df: pd.DataFrame,
    mist_breakdown: Dict[str, object],
    binary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    season_df: Optional[pd.DataFrame],
) -> None:
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    key_cols = [
        "name",
        "Fog_CSI",
        "Fog_R",
        "Fog_P",
        "Mist_CSI",
        "Mist_R",
        "Mist_P",
        "low_vis_csi",
        "low_vis_precision",
        "low_vis_recall",
        "false_positive_rate",
        "accuracy",
    ]
    display = metric_df[key_cols].copy()
    best_binary = binary_df[binary_df["score_name"] == "binary_head"].sort_values("csi", ascending=False).head(1)
    best_gate = gate_df.sort_values("low_vis_csi", ascending=False).head(1)

    lines = [
        "# Low-Visibility Checkpoint Diagnostics",
        "",
        "This is a post-hoc diagnostic run. It does not retrain the model.",
        "",
        "## Inputs",
        "",
        f"- split: `{args.split}`",
        f"- checkpoint: `{summary['ckpt_path']}`",
        f"- data_dir: `{summary['data_dir']}`",
        f"- scaler: `{summary['scaler_path']}`",
        f"- n_samples: {fmt(summary['n_samples'])}",
        f"- dyn_vars_count: {summary['dyn_vars_count']}",
        f"- extra_feat_dim: {summary['extra_feat_dim']}",
        f"- softmax_temperature_used: {fmt(summary['temperature_used'])}",
        "",
        "## Main Comparisons",
        "",
        "```text",
        display.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Mist Error Breakdown",
        "",
    ]
    for key, value in mist_breakdown.items():
        lines.append(f"- {key}: {fmt(value)}")

    if not best_binary.empty:
        b = best_binary.iloc[0].to_dict()
        lines.extend(
            [
                "",
                "## Binary Head",
                "",
                "Best binary low-visibility CSI from `sigmoid(low_vis_detector)`:",
                f"- threshold: {fmt(b['threshold'])}",
                f"- CSI: {fmt(b['csi'])}",
                f"- recall: {fmt(b['recall'])}",
                f"- precision: {fmt(b['precision'])}",
                f"- FPR: {fmt(b['fpr'])}",
            ]
        )
    if not best_gate.empty:
        g = best_gate.iloc[0].to_dict()
        lines.extend(
            [
                "",
                "Best binary-head gate plus Fog/Mist argmax:",
                f"- threshold: {fmt(g['binary_th'])}",
                f"- low_vis_csi: {fmt(g['low_vis_csi'])}",
                f"- Mist_CSI: {fmt(g['Mist_CSI'])}",
                f"- Mist_R: {fmt(g['Mist_R'])}",
                f"- FPR: {fmt(g['false_positive_rate'])}",
            ]
        )

    if season_df is not None and not season_df.empty:
        cols = [
            "season",
            "global_Mist_CSI",
            "season_Mist_CSI",
            "delta_Mist_CSI",
            "global_Mist_R",
            "season_Mist_R",
            "delta_Mist_R",
            "global_false_positive_rate",
            "season_false_positive_rate",
            "delta_false_positive_rate",
        ]
        lines.extend(
            [
                "",
                "## Seasonal Thresholds",
                "",
                "```text",
                season_df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
                "```",
            ]
        )

    (out_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Diagnose S2 PM10+PM2.5 checkpoint without retraining.")
    ap.add_argument("--base", default="/public/home/putianshu/vis_mlp")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--model-py", default=DEFAULT_MODEL_PY)
    ap.add_argument("--ckpt-path", default=f"checkpoints/{DEFAULT_RUN_ID}_S2_PhaseB_best_score.pt")
    ap.add_argument("--scaler-path", default=f"checkpoints/robust_scaler_{DEFAULT_RUN_ID}_w12_dyn27_s2_48h_pm10.pkl")
    ap.add_argument("--season-th-path", default=f"checkpoints/{DEFAULT_RUN_ID}_season_thresholds.pt")
    ap.add_argument("--out-dir", default=f"diagnostics/{DEFAULT_RUN_ID}_lowvis_checkpoint")
    ap.add_argument("--window-size", type=int, default=12)
    ap.add_argument("--dyn-vars-count", type=int, default=0)
    ap.add_argument("--extra-feat-dim", type=int, default=-1)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fog-th", type=float, default=0.34)
    ap.add_argument("--mist-th", type=float, default=0.56)
    ap.add_argument("--threshold-rule", choices=["mutual", "default", "joint", "argmax"], default="mutual")
    ap.add_argument("--use-calibration", action="store_true", help="Use temperature from --season-th-path when present.")
    ap.add_argument("--skip-fine-grid", action="store_true", help="Skip 676-combination fine-threshold sweep.")
    ap.add_argument("--distributed", action="store_true", help="Shard inference across torchrun ranks.")
    ap.add_argument("--local-rank", "--local_rank", type=int, default=None, help="Local rank supplied by torchrun.")
    return ap.parse_args()


def main():
    args = parse_args()
    torch, _ = ensure_torch()
    runtime = setup_runtime(args)
    rank = int(runtime["rank"])
    world_size = int(runtime["world_size"])
    device = runtime["device"]
    dist = runtime["dist"]
    base = Path(args.base)
    data_dir = abs_under_base(base, args.data_dir)
    ckpt_path = abs_under_base(base, args.ckpt_path)
    scaler_path = abs_under_base(base, args.scaler_path)
    season_path = abs_under_base(base, args.season_th_path) if args.season_th_path else None
    model_py = abs_under_base(Path.cwd(), args.model_py) if not Path(args.model_py).is_absolute() else Path(args.model_py)
    if not model_py.exists():
        model_py = abs_under_base(base, args.model_py)
    out_dir = abs_under_base(base, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir = out_dir / "_rank_parts"
    if world_size > 1:
        assert dist is not None
        if rank == 0 and part_dir.exists():
            shutil.rmtree(part_dir)
        dist.barrier()

    x_path = data_dir / f"X_{args.split}.npy"
    y_path = data_dir / f"y_{args.split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing split files: {x_path} / {y_path}")

    X = np.load(x_path, mmap_mode="r")
    n_total = len(X) if not args.limit_samples or args.limit_samples <= 0 else min(int(args.limit_samples), len(X))
    dyn, fe = infer_feature_layout(int(X.shape[1]), args.window_size, args.dyn_vars_count, args.extra_feat_dim)
    start_idx, end_idx = shard_bounds(n_total, rank, world_size)
    y_cls_part, y_raw_part = visibility_to_class(np.load(y_path, mmap_mode="r")[start_idx:end_idx])
    months_part = load_months_slice(data_dir, args.split, X, args.window_size, dyn, fe, start_idx, end_idx)
    season_thresholds, season_temp = load_season_thresholds(season_path)
    temperature = float(season_temp) if args.use_calibration and season_temp is not None else 1.0

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model_cls = import_model_class(model_py)
    model = model_cls(
        dyn_vars_count=dyn,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        extra_feat_dim=fe,
        dropout=args.dropout,
    ).to(device)
    load_checkpoint_into_model(model, ckpt_path, device)

    print(
        f"[setup rank={rank}/{world_size}] split={args.split} n_total={n_total:,} "
        f"shard=[{start_idx:,},{end_idx:,}) dyn={dyn} fe={fe} device={device} "
        f"temp={temperature:.4f}",
        flush=True,
    )
    probs_part, low_prob_part = run_inference(
        x_path,
        scaler,
        model,
        device,
        args.batch_size,
        args.window_size,
        dyn,
        fe,
        start_idx,
        end_idx,
        temperature,
    )

    valid_part = y_cls_part >= 0
    if not np.all(valid_part):
        print(f"[warn rank={rank}] dropping invalid labels: {int((~valid_part).sum())}", flush=True)
    probs_part = probs_part[valid_part]
    low_prob_part = low_prob_part[valid_part]
    y_cls_part = y_cls_part[valid_part]
    y_raw_part = y_raw_part[valid_part]
    months_part = months_part[valid_part]

    if world_size > 1:
        write_rank_part(part_dir, rank, probs_part, low_prob_part, y_cls_part, y_raw_part, months_part)
        assert dist is not None
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            print(f"[done rank={rank}] wrote shard and exited after rank0 barrier", flush=True)
            return
        probs, low_prob, y_cls, y_raw, months = read_rank_parts(part_dir, world_size)
    else:
        probs, low_prob, y_cls, y_raw, months = probs_part, low_prob_part, y_cls_part, y_raw_part, months_part

    fine_low_mass = np.clip(probs[:, 0] + probs[:, 1], 0.0, 1.0)

    current_pred = pred_from_rule(probs, args.fog_th, args.mist_th, args.threshold_rule)
    metric_rows = [
        add_metric_row(
            "current_fine_threshold",
            current_pred,
            y_cls,
            {"fog_th": args.fog_th, "mist_th": args.mist_th, "threshold_rule": args.threshold_rule},
        ),
        add_metric_row("fine_argmax", np.argmax(probs, axis=1), y_cls, {"threshold_rule": "argmax"}),
        add_metric_row("binary_gate_0.50", binary_gate_pred(probs, low_prob, 0.50), y_cls, {"binary_th": 0.50}),
    ]

    threshold_sweep_df = pd.DataFrame()
    if not args.skip_fine_grid:
        print("[sweep] fine threshold grid", flush=True)
        threshold_sweep_df = fine_threshold_sweep(probs, y_cls, args.threshold_rule)
        threshold_sweep_df.to_csv(out_dir / "fine_threshold_sweep.csv", index=False)
        for metric_name in ("target_achievement", "low_vis_csi", "Mist_CSI", "Mist_R"):
            row = threshold_sweep_df.sort_values(metric_name, ascending=False).iloc[0].to_dict()
            pred = pred_from_rule(probs, float(row["fog_th"]), float(row["mist_th"]), args.threshold_rule)
            metric_rows.append(add_metric_row(f"fine_grid_best_{metric_name}", pred, y_cls, row))

    print("[sweep] binary head and binary gate", flush=True)
    binary_df, gate_df = binary_sweeps(probs, low_prob, y_cls, fine_low_mass)
    binary_df.to_csv(out_dir / "binary_low_visibility_sweep.csv", index=False)
    gate_df.to_csv(out_dir / "binary_gate_multiclass_sweep.csv", index=False)
    if not gate_df.empty:
        best_gate = gate_df.sort_values("low_vis_csi", ascending=False).iloc[0].to_dict()
        metric_rows.append(
            add_metric_row(
                "binary_gate_best_low_vis_csi",
                binary_gate_pred(probs, low_prob, float(best_gate["binary_th"])),
                y_cls,
                best_gate,
            )
        )

    season_df = None
    if season_thresholds:
        fog_arr, mist_arr = thresholds_from_seasons(months, season_thresholds, args.fog_th, args.mist_th)
        season_pred = pred_from_thresholds_mutual(probs, fog_arr, mist_arr)
        metric_rows.append(add_metric_row("saved_season_thresholds", season_pred, y_cls, {"threshold_rule": "mutual"}))
        season_df = season_comparison(probs, y_cls, months, current_pred, season_pred)
        season_df.to_csv(out_dir / "season_threshold_comparison.csv", index=False)

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "metric_comparison.csv", index=False)

    cm = confusion_counts(y_cls, current_pred)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm_df.to_csv(out_dir / "confusion_current_threshold.csv")
    row_pct = cm_df.div(cm_df.sum(axis=1), axis=0)
    row_pct.to_csv(out_dir / "confusion_current_threshold_rowpct.csv")

    argmax = np.argmax(probs, axis=1)
    per_sample_diag = pd.DataFrame(
        {
            "y_true": y_cls,
            "pred_current": current_pred,
            "argmax_fine": argmax,
            "month": months,
            "pmst_p_fog": probs[:, 0],
            "pmst_p_mist": probs[:, 1],
            "pmst_p_clear": probs[:, 2],
            "fine_low_mass": fine_low_mass,
            "binary_low_prob": low_prob,
        }
    )
    quantile_summary(
        per_sample_diag,
        ["y_true"],
        ["pmst_p_fog", "pmst_p_mist", "pmst_p_clear", "fine_low_mass", "binary_low_prob"],
    ).to_csv(out_dir / "probability_quantiles_by_true_class.csv", index=False)
    quantile_summary(
        per_sample_diag[per_sample_diag["y_true"] == 1],
        ["pred_current"],
        ["pmst_p_fog", "pmst_p_mist", "pmst_p_clear", "fine_low_mass", "binary_low_prob"],
    ).to_csv(out_dir / "mist_probability_quantiles_by_prediction.csv", index=False)

    mist = per_sample_diag[per_sample_diag["y_true"] == 1]
    mist_to_clear = mist[mist["pred_current"] == 2]
    mist_breakdown = {
        "mist_support": int(len(mist)),
        "mist_to_fog": int((mist["pred_current"] == 0).sum()),
        "mist_to_mist": int((mist["pred_current"] == 1).sum()),
        "mist_to_clear": int((mist["pred_current"] == 2).sum()),
        "mist_to_clear_rate": safe_div(float((mist["pred_current"] == 2).sum()), float(len(mist))),
        "mist_to_clear_p_mist_below_or_eq_threshold": int((mist_to_clear["pmst_p_mist"] <= args.mist_th).sum()),
        "mist_to_clear_p_mist_argmax_below_threshold": int(
            ((mist_to_clear["argmax_fine"] == 1) & (mist_to_clear["pmst_p_mist"] <= args.mist_th)).sum()
        ),
        "mist_to_clear_p_clear_argmax": int((mist_to_clear["argmax_fine"] == 2).sum()),
        "mist_to_clear_p_fog_argmax_below_threshold": int(
            ((mist_to_clear["argmax_fine"] == 0) & (mist_to_clear["pmst_p_fog"] <= args.fog_th)).sum()
        ),
    }
    pd.DataFrame([mist_breakdown]).to_csv(out_dir / "mist_error_breakdown.csv", index=False)

    summary = {
        "split": args.split,
        "n_samples": int(len(y_cls)),
        "class_counts": {CLASS_NAMES[i]: int((y_cls == i).sum()) for i in range(3)},
        "data_dir": str(data_dir),
        "x_path": str(x_path),
        "ckpt_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "season_threshold_path": str(season_path) if season_path else None,
        "season_temperature_in_file": season_temp,
        "temperature_used": temperature,
        "dyn_vars_count": dyn,
        "extra_feat_dim": fe,
        "fog_threshold": args.fog_th,
        "mist_threshold": args.mist_th,
        "threshold_rule": args.threshold_rule,
        "output_dir": str(out_dir),
        "world_size": world_size,
        "rank_parts_dir": str(part_dir) if world_size > 1 else None,
    }
    with open(out_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_clean(summary), f, indent=2)

    write_report(out_dir, args, summary, metric_df, mist_breakdown, binary_df, gate_df, season_df)
    print(f"[done] diagnostics written to {out_dir}", flush=True)
    if world_size > 1 and dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
