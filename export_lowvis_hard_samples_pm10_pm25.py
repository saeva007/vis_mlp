#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export hard-sample pools from an existing PM10+PM2.5 S2 checkpoint.

The script is inference-only. It loads the same archived checkpoint/scaler used
by diagnose_lowvis_checkpoint_pm10_pm25.py, runs one split, and writes pool
indices plus weights for a later hard Mist/Clear fine-tune.

Default paths match exp_1778563813_pm10_more_temp_search_utc.
Run from the remote checkout root:

  cd /public/home/putianshu/vis_mlp/train
  python export_lowvis_hard_samples_pm10_pm25.py --split val
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
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

torch = None
F = None


DEFAULT_RUN_ID = "exp_1778563813_pm10_more_temp_search_utc"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_MODEL_PY = "PMST_net_test_11_s2_pm10.py"
CLASS_NAMES = ("Fog", "Mist", "Clear")


def ensure_torch():
    global torch, F
    if torch is None or F is None:
        import torch as _torch
        import torch.nn.functional as _F

        torch = _torch
        F = _F
    return torch, F


def abs_under_base(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


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
    if pd.isna(value):
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
        fe = int(n_cols) - int(window_size) * int(dyn_vars_count) - 6
        if fe < 0:
            raise ValueError(
                f"Cannot infer layout: n_cols={n_cols}, dyn={dyn_vars_count}, "
                f"window={window_size} gives FE={fe}."
            )
        if extra_feat_dim >= 0 and fe != extra_feat_dim:
            raise ValueError(f"FE mismatch: inferred {fe}, expected {extra_feat_dim}.")
        return int(dyn_vars_count), int(fe)

    plausible = []
    for dyn in (27, 26, 25, 24):
        fe = int(n_cols) - int(window_size) * dyn - 6
        if fe < 0:
            continue
        if extra_feat_dim >= 0 and fe != extra_feat_dim:
            continue
        if 0 <= fe <= 96:
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
                f"feature block {feats.shape[1]}."
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


def extract_aux_features(rows: np.ndarray, window_size: int, dyn_vars_count: int) -> Dict[str, np.ndarray]:
    last = (int(window_size) - 1) * int(dyn_vars_count)

    def col(idx: int) -> np.ndarray:
        if 0 <= idx < dyn_vars_count:
            return np.asarray(rows[:, last + idx], dtype=np.float32)
        return np.full(rows.shape[0], np.nan, dtype=np.float32)

    out = {
        "rh2m_last": col(0),
        "t2m_last": col(1),
        "wspd10_last": col(6),
        "lcc_last": col(10),
        "dpd_last": col(22),
    }
    if dyn_vars_count >= 27:
        out["pm10_last"] = col(dyn_vars_count - 2)
        out["pm25_last"] = col(dyn_vars_count - 1)
    elif dyn_vars_count >= 25:
        out["pm_last"] = col(dyn_vars_count - 1)
    return out


def import_model_class(model_py: Path):
    spec = importlib.util.spec_from_file_location("pmst_lowvis_hard_export_model", str(model_py))
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


def load_scaler(path: Path):
    joblib_error = None
    try:
        import joblib

        scaler = joblib.load(path)
        loader = "joblib"
    except Exception as exc:
        joblib_error = exc
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            loader = "pickle"
        except Exception as pickle_error:
            raise RuntimeError(
                f"Failed to load scaler {path}. "
                f"joblib.load error: {joblib_error!r}; pickle.load error: {pickle_error!r}"
            ) from pickle_error

    if not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
        raise TypeError(f"Loaded scaler from {path} with {loader}, but it lacks center_/scale_.")
    print(f"[scaler] loaded with {loader}: {path}", flush=True)
    return scaler


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
            backend = "nccl" if args.dist_backend == "nccl" else "gloo"
        if args.dist_backend == "nccl" and device.type != "cuda":
            raise RuntimeError("--dist-backend nccl requires a CUDA/DCU device.")
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


def dist_barrier(dist, device) -> None:
    if dist is None:
        return
    try:
        backend = dist.get_backend()
    except Exception:
        backend = ""
    if backend == "nccl" and getattr(device, "type", None) == "cuda":
        try:
            dist.barrier(device_ids=[int(device.index or 0)])
            return
        except TypeError:
            pass
    dist.barrier()


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    torch, F = ensure_torch()
    X = np.load(x_path, mmap_mode="r")
    start_index = int(start_index)
    end_index = int(end_index)
    n = max(0, end_index - start_index)
    probs_out: List[np.ndarray] = []
    low_out: List[np.ndarray] = []
    aux_out: Dict[str, List[np.ndarray]] = {}
    model.eval()

    for start in range(start_index, end_index, batch_size):
        end = min(start + batch_size, end_index)
        rows = np.asarray(X[start:end], dtype=np.float32)
        aux = extract_aux_features(rows, window_size, dyn_vars_count)
        for key, values in aux.items():
            aux_out.setdefault(key, []).append(values.astype(np.float32, copy=False))

        final = prepare_batch_rows(rows, scaler, window_size, dyn_vars_count, extra_feat_dim)
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            fine_logits, _, low_vis_logit = model(x)
            probs = F.softmax(fine_logits, dim=1)
            low_prob = torch.sigmoid(low_vis_logit).reshape(-1)
        probs_out.append(probs.detach().cpu().numpy().astype(np.float32))
        low_out.append(low_prob.detach().cpu().numpy().astype(np.float32))

        done = end - start_index
        if start == start_index or end == end_index or (done // max(batch_size, 1)) % 20 == 0:
            print(f"[inference] local {done:,}/{n:,} global {end:,}", flush=True)

    if not probs_out:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}
    aux_cat = {key: np.concatenate(chunks, axis=0) for key, chunks in aux_out.items()}
    return np.concatenate(probs_out, axis=0), np.concatenate(low_out, axis=0), aux_cat


def pred_from_thresholds_mutual(probs: np.ndarray, fog_th: float, mist_th: float) -> np.ndarray:
    p_fog = np.asarray(probs[:, 0], dtype=np.float64)
    p_mist = np.asarray(probs[:, 1], dtype=np.float64)
    pred = np.full(probs.shape[0], 2, dtype=np.int64)
    pred[(p_fog > fog_th) & (p_fog > p_mist)] = 0
    pred[(p_mist > mist_th) & (p_mist > p_fog)] = 1
    return pred


def binary_gate_pred(probs: np.ndarray, low_prob: np.ndarray, low_th: float) -> np.ndarray:
    pred = np.full(len(low_prob), 2, dtype=np.int64)
    low = low_prob >= low_th
    low_class = np.where(probs[:, 0] >= probs[:, 1], 0, 1)
    pred[low] = low_class[low]
    return pred


def write_rank_part(
    part_dir: Path,
    rank: int,
    probs: np.ndarray,
    low_prob: np.ndarray,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    orig_index: np.ndarray,
    aux: Dict[str, np.ndarray],
) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probs": probs,
        "low_prob": low_prob,
        "y_cls": y_cls,
        "y_raw": y_raw,
        "orig_index": orig_index,
    }
    payload.update({f"aux_{k}": v for k, v in aux.items()})
    np.savez_compressed(part_dir / f"rank_{rank:03d}.npz", **payload)


def read_rank_parts(
    part_dir: Path,
    world_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    probs_l, low_l, cls_l, raw_l, idx_l = [], [], [], [], []
    aux_l: Dict[str, List[np.ndarray]] = {}
    for rank in range(world_size):
        path = part_dir / f"rank_{rank:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing rank part: {path}")
        with np.load(path, allow_pickle=False) as z:
            probs_l.append(np.asarray(z["probs"], dtype=np.float32))
            low_l.append(np.asarray(z["low_prob"], dtype=np.float32))
            cls_l.append(np.asarray(z["y_cls"], dtype=np.int64))
            raw_l.append(np.asarray(z["y_raw"], dtype=np.float32))
            idx_l.append(np.asarray(z["orig_index"], dtype=np.int64))
            for key in z.files:
                if key.startswith("aux_"):
                    aux_l.setdefault(key[4:], []).append(np.asarray(z[key], dtype=np.float32))
    aux = {key: np.concatenate(chunks, axis=0) for key, chunks in aux_l.items()}
    return (
        np.concatenate(probs_l, axis=0),
        np.concatenate(low_l, axis=0),
        np.concatenate(cls_l, axis=0),
        np.concatenate(raw_l, axis=0),
        np.concatenate(idx_l, axis=0),
        aux,
    )


def stable_rank_values(pool_name: str, indices: np.ndarray, seed: int) -> np.ndarray:
    salt = np.uint64(zlib.crc32(pool_name.encode("utf-8")) ^ int(seed))
    x = indices.astype(np.uint64, copy=False) ^ salt
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xff51afd7ed558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xc4ceb9fe1a85ec53)
    x ^= x >> np.uint64(33)
    return x.astype(np.float64) / float(np.iinfo(np.uint64).max)


def weights_from_severity(severity: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    sev = np.nan_to_num(np.asarray(severity, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(sev) == 0:
        return np.zeros(0, dtype=np.float32)
    lo = float(np.min(sev))
    hi = float(np.max(sev))
    if hi <= lo:
        norm = np.ones_like(sev)
    else:
        norm = (sev - lo) / (hi - lo)
    return (float(min_weight) + (float(max_weight) - float(min_weight)) * norm).astype(np.float32)


def base_result_frame(
    probs: np.ndarray,
    low_prob: np.ndarray,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    orig_index: np.ndarray,
    aux: Dict[str, np.ndarray],
    fine_fog_th: float,
    fine_mist_th: float,
    lowvis_gate_th: float,
) -> pd.DataFrame:
    p_fog = probs[:, 0].astype(np.float32)
    p_mist = probs[:, 1].astype(np.float32)
    p_clear = probs[:, 2].astype(np.float32)
    fine_low_mass = np.clip(p_fog + p_mist, 0.0, 1.0).astype(np.float32)
    pred_fine = pred_from_thresholds_mutual(probs, fine_fog_th, fine_mist_th)
    pred_binary_gate = binary_gate_pred(probs, low_prob, lowvis_gate_th)

    data = {
        "orig_index": orig_index.astype(np.int64),
        "y_true": y_cls.astype(np.int64),
        "y_true_name": [CLASS_NAMES[int(c)] if 0 <= int(c) < 3 else "Invalid" for c in y_cls],
        "vis_raw_m": y_raw.astype(np.float32),
        "p_fog": p_fog,
        "p_mist": p_mist,
        "p_clear": p_clear,
        "p_lowvis_binary": low_prob.astype(np.float32),
        "fine_low_mass": fine_low_mass,
        "pred_fine": pred_fine.astype(np.int64),
        "pred_binary_gate": pred_binary_gate.astype(np.int64),
    }
    for key, values in aux.items():
        data[key] = values.astype(np.float32, copy=False)
    return pd.DataFrame(data)


def build_pool_frame(
    base: pd.DataFrame,
    pool_name: str,
    mask: np.ndarray,
    severity: np.ndarray,
    weight_range: Tuple[float, float],
    max_per_pool: int,
    seed: int,
) -> pd.DataFrame:
    idx = np.asarray(mask, dtype=bool)
    pool = base.loc[idx].copy()
    sev = np.asarray(severity, dtype=np.float64)[idx]
    pool.insert(0, "pool", pool_name)
    pool["severity"] = np.nan_to_num(sev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pool["hard_weight"] = weights_from_severity(pool["severity"].to_numpy(), *weight_range)
    pool["_stable_rank"] = stable_rank_values(pool_name, pool["orig_index"].to_numpy(), seed)
    pool = pool.sort_values(
        ["severity", "_stable_rank", "orig_index"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if max_per_pool and max_per_pool > 0:
        pool = pool.head(int(max_per_pool))
    pool = pool.drop(columns=["_stable_rank"]).reset_index(drop=True)
    return pool


def build_pools(
    base: pd.DataFrame,
    lowvis_gate_th: float,
    fine_fog_th: float,
    fine_mist_th: float,
    max_per_pool: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    p_fog = base["p_fog"].to_numpy(dtype=np.float64)
    p_mist = base["p_mist"].to_numpy(dtype=np.float64)
    p_clear = base["p_clear"].to_numpy(dtype=np.float64)
    p_low = base["p_lowvis_binary"].to_numpy(dtype=np.float64)
    fine_low_mass = base["fine_low_mass"].to_numpy(dtype=np.float64)
    y = base["y_true"].to_numpy(dtype=np.int64)
    vis = base["vis_raw_m"].to_numpy(dtype=np.float64)
    pred_fine = base["pred_fine"].to_numpy(dtype=np.int64)

    rh = base["rh2m_last"].to_numpy(dtype=np.float64) if "rh2m_last" in base else np.full(len(base), np.nan)
    dpd = base["dpd_last"].to_numpy(dtype=np.float64) if "dpd_last" in base else np.full(len(base), np.nan)

    true_low = y <= 1
    true_clear = y == 2
    mist = y == 1
    fog_or_mist = (y == 0) | (y == 1)

    hard_mist_mask = mist & (
        (pred_fine != 1)
        | (p_mist < fine_mist_th)
        | (p_low < lowvis_gate_th)
    )
    hard_mist_sev = (
        np.maximum(fine_mist_th - p_mist, 0.0) * 1.6
        + np.maximum(lowvis_gate_th - p_low, 0.0)
        + np.maximum(p_clear - p_mist, 0.0)
        + np.maximum(p_fog - p_mist, 0.0) * 0.5
    )

    hard_clear_mask = true_clear & (
        (p_low >= lowvis_gate_th)
        | (p_fog >= fine_fog_th)
        | (p_mist >= fine_mist_th)
    )
    clear_vis_closeness = np.clip((3000.0 - np.maximum(vis, 1000.0)) / 2000.0, 0.0, 1.0)
    hard_clear_sev = (
        p_low * 1.2
        + fine_low_mass
        + np.maximum(p_fog - fine_fog_th, 0.0)
        + np.maximum(p_mist - fine_mist_th, 0.0)
        + clear_vis_closeness * 0.3
    )

    humid_signal = (
        (np.isfinite(rh) & (rh >= 90.0))
        | (np.isfinite(dpd) & (dpd <= 2.0))
        | ((vis >= 1000.0) & (vis < 3000.0))
    )
    near_low_gate = (p_low >= max(lowvis_gate_th - 0.15, 0.0)) & (p_low < lowvis_gate_th)
    near_fine_low = (fine_low_mass >= 0.35) & ~(hard_clear_mask)
    humid_clear_mask = true_clear & humid_signal & (near_low_gate | near_fine_low) & ~(hard_clear_mask)
    humid_clear_sev = (
        np.maximum(p_low - max(lowvis_gate_th - 0.15, 0.0), 0.0)
        + np.maximum(fine_low_mass - 0.35, 0.0)
        + np.nan_to_num((rh - 90.0) / 20.0, nan=0.0).clip(0.0, 1.0) * 0.3
        + np.nan_to_num((2.0 - dpd) / 5.0, nan=0.0).clip(0.0, 1.0) * 0.3
        + clear_vis_closeness * 0.2
    )

    physical_boundary = fog_or_mist & (vis >= 400.0) & (vis < 600.0)
    prob_boundary = fog_or_mist & (np.abs(p_fog - p_mist) <= 0.15) & (fine_low_mass >= 0.20)
    fog_mist_mask = physical_boundary | prob_boundary
    vis_boundary = np.clip(1.0 - np.abs(vis - 500.0) / 200.0, 0.0, 1.0)
    fog_mist_sev = (
        (1.0 - np.clip(np.abs(p_fog - p_mist), 0.0, 1.0))
        + vis_boundary
        + (pred_fine != y).astype(np.float64) * 0.5
        + np.maximum(lowvis_gate_th - p_low, 0.0) * 0.2
    )

    easy_clear_mask = true_clear & (
        (vis >= 5000.0)
        & (p_clear >= 0.90)
        & (p_low < min(lowvis_gate_th, 0.25))
        & (fine_low_mass <= 0.15)
        & (pred_fine == 2)
    )
    easy_clear_sev = (
        p_clear
        + (1.0 - p_low)
        + np.clip(vis, 0.0, 30000.0) / 30000.0 * 0.25
    )

    specs = [
        ("hard_mist_missed", hard_mist_mask, hard_mist_sev, (1.0, 3.0)),
        ("hard_clear_false_lowvis", hard_clear_mask, hard_clear_sev, (1.0, 3.0)),
        ("humid_clear_near_miss", humid_clear_mask, humid_clear_sev, (0.8, 2.3)),
        ("fog_mist_boundary", fog_mist_mask, fog_mist_sev, (1.0, 2.5)),
        ("easy_clear_anchor", easy_clear_mask, easy_clear_sev, (0.25, 0.60)),
    ]

    pool_frames = []
    stats: Dict[str, Dict[str, object]] = {}
    for name, mask, severity, weight_range in specs:
        before = int(np.asarray(mask, dtype=bool).sum())
        frame = build_pool_frame(base, name, mask, severity, weight_range, max_per_pool, seed)
        pool_frames.append(frame)
        stats[name] = {
            "candidate_count": before,
            "selected_count": int(len(frame)),
            "weight_min": float(frame["hard_weight"].min()) if len(frame) else None,
            "weight_max": float(frame["hard_weight"].max()) if len(frame) else None,
        }

    combined = pd.concat(pool_frames, axis=0, ignore_index=True) if pool_frames else pd.DataFrame()
    return combined, stats


def attach_meta(data_dir: Path, split: str, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Path]]:
    meta_path = data_dir / f"meta_{split}.csv"
    if frame.empty or not meta_path.exists():
        return frame, meta_path if meta_path.exists() else None

    meta = pd.read_csv(meta_path)
    if "orig_index" in meta.columns:
        meta = meta.drop(columns=["orig_index"])
    max_index = int(frame["orig_index"].max()) if len(frame) else -1
    if len(meta) <= max_index:
        print(
            f"[warn] {meta_path} has {len(meta):,} rows, fewer than max orig_index={max_index}; "
            "metadata will be skipped.",
            flush=True,
        )
        return frame, None

    selected_meta = meta.iloc[frame["orig_index"].to_numpy(dtype=np.int64)].reset_index(drop=True)
    duplicate_cols = [c for c in selected_meta.columns if c in frame.columns]
    selected_meta = selected_meta.drop(columns=duplicate_cols)
    out = pd.concat([frame.reset_index(drop=True), selected_meta], axis=1)
    return out, meta_path


def write_outputs(
    out_dir: Path,
    combined: pd.DataFrame,
    stats: Dict[str, Dict[str, object]],
    summary: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hard_sample_pools.csv"
    json_path = out_dir / "hard_sample_pools.json"
    npz_path = out_dir / "hard_sample_pools.npz"
    counts_path = out_dir / "hard_sample_pool_counts.csv"

    combined.to_csv(csv_path, index=False)
    pd.DataFrame(
        [
            {"pool": pool, **pool_stats}
            for pool, pool_stats in stats.items()
        ]
    ).to_csv(counts_path, index=False)

    npz_payload = {}
    for pool_name in stats:
        sub = combined[combined["pool"] == pool_name]
        npz_payload[f"{pool_name}_indices"] = sub["orig_index"].to_numpy(dtype=np.int64)
        npz_payload[f"{pool_name}_weights"] = sub["hard_weight"].to_numpy(dtype=np.float32)
        npz_payload[f"{pool_name}_y_true"] = sub["y_true"].to_numpy(dtype=np.int64)
        npz_payload[f"{pool_name}_vis_raw_m"] = sub["vis_raw_m"].to_numpy(dtype=np.float32)
    npz_payload["pool_names"] = np.asarray(list(stats.keys()))
    np.savez_compressed(npz_path, **npz_payload)

    payload = {
        "summary": summary,
        "pool_stats": stats,
        "pools": {
            pool_name: combined[combined["pool"] == pool_name].to_dict(orient="records")
            for pool_name in stats
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_clean(payload), f, indent=2)

    summary_path = out_dir / "hard_sample_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_clean({**summary, "pool_stats": stats}), f, indent=2)

    print(f"[write] {csv_path}", flush=True)
    print(f"[write] {json_path}", flush=True)
    print(f"[write] {npz_path}", flush=True)
    print(f"[write] {counts_path}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Export low-visibility hard-sample pools from an S2 checkpoint.")
    ap.add_argument("--base", default="/public/home/putianshu/vis_mlp")
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--model-py", default=DEFAULT_MODEL_PY)
    ap.add_argument("--ckpt-path", default=f"checkpoints/{DEFAULT_RUN_ID}_S2_PhaseB_best_score.pt")
    ap.add_argument("--scaler-path", default=f"checkpoints/robust_scaler_{DEFAULT_RUN_ID}_w12_dyn27_s2_48h_pm10.pkl")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window-size", type=int, default=12)
    ap.add_argument("--dyn-vars-count", type=int, default=0)
    ap.add_argument("--extra-feat-dim", type=int, default=-1)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lowvis-gate-th", type=float, default=0.81)
    ap.add_argument("--fine-fog-th", type=float, default=0.34)
    ap.add_argument("--fine-mist-th", type=float, default=0.56)
    ap.add_argument("--max-per-pool", type=int, default=0, help="0 means keep all sorted candidates.")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--distributed", action="store_true", help="Shard inference across torchrun ranks.")
    ap.add_argument("--dist-backend", choices=["gloo", "nccl"], default="gloo")
    ap.add_argument("--local-rank", "--local_rank", type=int, default=None)
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
    default_out = f"hard_samples/{args.run_id}_{args.split}_hard_samples"
    out_dir = abs_under_base(base, args.out_dir or default_out)
    model_py = Path(args.model_py)
    if not model_py.is_absolute():
        model_py = Path.cwd() / model_py
    if not model_py.exists():
        model_py = abs_under_base(base, args.model_py)
    assert data_dir is not None and ckpt_path is not None and scaler_path is not None and out_dir is not None

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    part_dir = out_dir / "_rank_parts"
    if world_size > 1:
        assert dist is not None
        if rank == 0 and part_dir.exists():
            shutil.rmtree(part_dir)
        dist_barrier(dist, device)

    x_path = data_dir / f"X_{args.split}.npy"
    y_path = data_dir / f"y_{args.split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing split files: {x_path} / {y_path}")

    X = np.load(x_path, mmap_mode="r")
    n_total = len(X) if not args.limit_samples or args.limit_samples <= 0 else min(int(args.limit_samples), len(X))
    dyn, fe = infer_feature_layout(int(X.shape[1]), args.window_size, args.dyn_vars_count, args.extra_feat_dim)
    start_idx, end_idx = shard_bounds(n_total, rank, world_size)
    orig_part = np.arange(start_idx, end_idx, dtype=np.int64)
    y_cls_part, y_raw_part = visibility_to_class(np.load(y_path, mmap_mode="r")[start_idx:end_idx])

    scaler = load_scaler(scaler_path)
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
        f"shard=[{start_idx:,},{end_idx:,}) dyn={dyn} fe={fe} device={device}",
        flush=True,
    )
    probs_part, low_prob_part, aux_part = run_inference(
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
    )

    valid_part = y_cls_part >= 0
    if not np.all(valid_part):
        print(f"[warn rank={rank}] dropping invalid labels: {int((~valid_part).sum())}", flush=True)
    probs_part = probs_part[valid_part]
    low_prob_part = low_prob_part[valid_part]
    y_cls_part = y_cls_part[valid_part]
    y_raw_part = y_raw_part[valid_part]
    orig_part = orig_part[valid_part]
    aux_part = {key: values[valid_part] for key, values in aux_part.items()}

    if world_size > 1:
        write_rank_part(part_dir, rank, probs_part, low_prob_part, y_cls_part, y_raw_part, orig_part, aux_part)
        assert dist is not None
        dist_barrier(dist, device)
        if rank != 0:
            dist.destroy_process_group()
            print(f"[done rank={rank}] wrote shard and exited after rank0 barrier", flush=True)
            return
        probs, low_prob, y_cls, y_raw, orig_index, aux = read_rank_parts(part_dir, world_size)
    else:
        probs, low_prob, y_cls, y_raw, orig_index, aux = (
            probs_part,
            low_prob_part,
            y_cls_part,
            y_raw_part,
            orig_part,
            aux_part,
        )

    base_frame = base_result_frame(
        probs,
        low_prob,
        y_cls,
        y_raw,
        orig_index,
        aux,
        args.fine_fog_th,
        args.fine_mist_th,
        args.lowvis_gate_th,
    )
    combined, stats = build_pools(
        base_frame,
        args.lowvis_gate_th,
        args.fine_fog_th,
        args.fine_mist_th,
        args.max_per_pool,
        args.seed,
    )
    combined, meta_path = attach_meta(data_dir, args.split, combined)

    class_counts = {
        CLASS_NAMES[i]: int((y_cls == i).sum())
        for i in range(3)
    }
    summary = {
        "run_id": args.run_id,
        "split": args.split,
        "n_samples_scored": int(len(y_cls)),
        "class_counts": class_counts,
        "data_dir": str(data_dir),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "meta_path": str(meta_path) if meta_path else None,
        "ckpt_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "model_py": str(model_py),
        "out_dir": str(out_dir),
        "dyn_vars_count": int(dyn),
        "extra_feat_dim": int(fe),
        "window_size": int(args.window_size),
        "lowvis_gate_th": float(args.lowvis_gate_th),
        "fine_fog_th": float(args.fine_fog_th),
        "fine_mist_th": float(args.fine_mist_th),
        "threshold_rule": "mutual",
        "max_per_pool": int(args.max_per_pool),
        "seed": int(args.seed),
        "limit_samples": int(args.limit_samples),
        "world_size": int(world_size),
        "dist_backend": dist.get_backend() if dist is not None else None,
    }
    write_outputs(out_dir, combined, stats, summary)
    print(f"[done] hard-sample pools written to {out_dir}", flush=True)
    if world_size > 1 and dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
