#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard Mist/Clear fine-tuning for the PM10+PM2.5 S2 model.

This file is intentionally a new entry point. It imports the archived
PMST_net_test_11_s2_pm10.py model/data/DDP utilities and does not modify that
archive script.

Default initialization checkpoint:
  /public/home/putianshu/vis_mlp/checkpoints/
  exp_1778563813_pm10_more_temp_search_utc_S2_PhaseB_best_score.pt

Default output run id:
  exp_1778563813_pm10_more_temp_search_utc_hard_mist_clear_ft
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import PMST_net_test_11_s2_pm10 as base


DEFAULT_INIT_RUN_ID = "exp_1778563813_pm10_more_temp_search_utc"
DEFAULT_TAG = "S2_HardMistClearFT"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune S2 from hard Mist misses and hard Clear false-lowvis samples."
    )
    parser.add_argument("--init-run-id", default=env_str("INIT_RUN_ID", DEFAULT_INIT_RUN_ID))
    parser.add_argument("--hard-ft-run-id", default=env_str("HARD_FT_RUN_ID", ""))
    parser.add_argument("--init-ckpt-path", default=env_str("INIT_CKPT_PATH", ""))
    parser.add_argument("--scaler-path", default=env_str("SCALER_PATH", ""))
    parser.add_argument("--hard-pool-dir", default=env_str("HARD_POOL_DIR", ""))
    parser.add_argument("--data-dir", default=env_str("S2_DATA_DIR", base.CONFIG["S2_DATA_DIR"]))

    parser.add_argument("--lowvis-gate-th", type=float, default=env_float("LOWVIS_GATE_TH", 0.81))
    parser.add_argument("--target-max-fpr", type=float, default=env_float("TARGET_MAX_FPR", 0.045))
    parser.add_argument("--target-soft-fpr", type=float, default=env_float("TARGET_SOFT_FPR", 0.040))

    parser.add_argument("--hard-ft-steps", type=int, default=env_int("HARD_FT_STEPS", 12000))
    parser.add_argument("--val-interval", type=int, default=env_int("HARD_FT_VAL_INTERVAL", 500))
    parser.add_argument("--patience", type=int, default=env_int("HARD_FT_PATIENCE", 8))
    parser.add_argument("--num-workers", type=int, default=env_int("HARD_FT_NUM_WORKERS", 0))
    parser.add_argument("--batch-size", type=int, default=env_int("HARD_FT_BATCH_SIZE", 512))
    parser.add_argument("--grad-accum", type=int, default=env_int("HARD_FT_GRAD_ACCUM", 2))
    parser.add_argument("--epoch-length", type=int, default=env_int("HARD_FT_EPOCH_LENGTH", 2000))

    parser.add_argument("--lr-backbone", type=float, default=env_float("HARD_FT_LR_BACKBONE", 8.0e-7))
    parser.add_argument("--lr-fusion", type=float, default=env_float("HARD_FT_LR_FUSION", 6.0e-6))
    parser.add_argument("--lr-head", type=float, default=env_float("HARD_FT_LR_HEAD", 2.5e-5))
    parser.add_argument("--l2sp", type=float, default=env_float("HARD_FT_L2SP", 2.0e-5))

    parser.add_argument("--fog-ratio", type=float, default=env_float("HARD_FT_FOG_RATIO", 0.16))
    parser.add_argument("--mist-bg-ratio", type=float, default=env_float("HARD_FT_MIST_BG_RATIO", 0.18))
    parser.add_argument("--hard-mist-ratio", type=float, default=env_float("HARD_FT_HARD_MIST_RATIO", 0.20))
    parser.add_argument("--hard-clear-ratio", type=float, default=env_float("HARD_FT_HARD_CLEAR_RATIO", 0.22))

    args = parser.parse_args()
    if not args.hard_ft_run_id:
        args.hard_ft_run_id = f"{args.init_run_id}_hard_mist_clear_ft"
    return args


def update_hard_ft_config(args: argparse.Namespace) -> str:
    cfg = base.CONFIG
    cfg["EXPERIMENT_ID"] = args.init_run_id
    cfg["S2_RUN_SUFFIX"] = "hard_mist_clear_ft"
    cfg["S2_DATA_DIR"] = args.data_dir

    cfg["NUM_WORKERS"] = int(args.num_workers)
    cfg["S2_BATCH_SIZE"] = int(args.batch_size)
    cfg["S2_GRAD_ACCUM"] = int(args.grad_accum)
    cfg["S2_VAL_INTERVAL"] = int(args.val_interval)
    cfg["S2_WARMUP_STEPS"] = env_int("HARD_FT_WARMUP_STEPS", 300)
    cfg["S2_ES_PATIENCE"] = int(args.patience)

    cfg["HARD_FT_RUN_ID"] = args.hard_ft_run_id
    cfg["HARD_POOL_DIR"] = args.hard_pool_dir
    cfg["HARD_FT_STEPS"] = int(args.hard_ft_steps)
    cfg["HARD_FT_EPOCH_LENGTH"] = int(args.epoch_length)
    cfg["HARD_FT_LR_BACKBONE"] = float(args.lr_backbone)
    cfg["HARD_FT_LR_FUSION"] = float(args.lr_fusion)
    cfg["HARD_FT_LR_HEAD"] = float(args.lr_head)
    cfg["HARD_FT_L2SP"] = float(args.l2sp)

    cfg["HARD_FT_FOG_RATIO"] = float(args.fog_ratio)
    cfg["HARD_FT_MIST_BG_RATIO"] = float(args.mist_bg_ratio)
    cfg["HARD_FT_HARD_MIST_RATIO"] = float(args.hard_mist_ratio)
    cfg["HARD_FT_HARD_CLEAR_RATIO"] = float(args.hard_clear_ratio)

    cfg["LOWVIS_GATE_TH"] = float(args.lowvis_gate_th)
    cfg["TARGET_MAX_FPR"] = float(args.target_max_fpr)
    cfg["TARGET_SOFT_FPR"] = float(args.target_soft_fpr)

    cfg["S2_BINARY_POS_WEIGHT"] = env_float("HARD_FT_BINARY_POS_WEIGHT", 2.2)
    cfg["S2_FINE_CLASS_WEIGHT_FOG"] = env_float("HARD_FT_WEIGHT_FOG", 2.0)
    cfg["S2_FINE_CLASS_WEIGHT_MIST"] = env_float("HARD_FT_WEIGHT_MIST", 3.1)
    cfg["S2_FINE_CLASS_WEIGHT_CLEAR"] = env_float("HARD_FT_WEIGHT_CLEAR", 0.9)
    cfg["S2_LOSS_ALPHA_BINARY"] = env_float("HARD_FT_ALPHA_BINARY", 1.05)
    cfg["S2_LOSS_ALPHA_FINE"] = env_float("HARD_FT_ALPHA_FINE", 1.0)
    cfg["S2_LOSS_ALPHA_FP"] = env_float("HARD_FT_ALPHA_FP", 3.4)
    cfg["S2_LOSS_ALPHA_FOG_BOOST"] = env_float("HARD_FT_ALPHA_FOG_BOOST", 0.75)
    cfg["S2_LOSS_ALPHA_MIST_BOOST"] = env_float("HARD_FT_ALPHA_MIST_BOOST", 1.25)
    cfg["S2_CLEAR_MARGIN"] = env_float("HARD_FT_CLEAR_MARGIN", 0.25)
    cfg["S2_LOSS_ALPHA_CLEAR_MARGIN"] = env_float("HARD_FT_ALPHA_CLEAR_MARGIN", 2.2)
    cfg["S2_PAIR_MARGIN"] = env_float("HARD_FT_PAIR_MARGIN", 0.55)
    cfg["S2_LOSS_ALPHA_PAIR_MARGIN"] = env_float("HARD_FT_ALPHA_PAIR_MARGIN", 0.35)

    cfg["HARD_FT_ALPHA_LOW_MISS"] = env_float("HARD_FT_ALPHA_LOW_MISS", 0.55)
    cfg["HARD_FT_ALPHA_FINE_LOW_MASS"] = env_float("HARD_FT_ALPHA_FINE_LOW_MASS", 0.35)
    cfg["HARD_FT_MIST_CLEAR_MARGIN"] = env_float("HARD_FT_MIST_CLEAR_MARGIN", 0.45)
    cfg["HARD_FT_ALPHA_MIST_CLEAR_MARGIN"] = env_float("HARD_FT_ALPHA_MIST_CLEAR_MARGIN", 0.95)
    cfg["HARD_FT_HARD_MIST_EXTRA_WEIGHT"] = env_float("HARD_FT_HARD_MIST_EXTRA_WEIGHT", 1.0)
    cfg["HARD_FT_ALPHA_HARD_CLEAR_MARGIN"] = env_float("HARD_FT_ALPHA_HARD_CLEAR_MARGIN", 1.35)
    cfg["HARD_FT_ALPHA_HARD_CLEAR_BINARY"] = env_float("HARD_FT_ALPHA_HARD_CLEAR_BINARY", 1.65)
    cfg["HARD_FT_ALPHA_HARD_CLEAR_LOW_MASS"] = env_float("HARD_FT_ALPHA_HARD_CLEAR_LOW_MASS", 0.75)

    cfg["HARD_FT_GOAL_LOWVIS_RECALL"] = env_float("HARD_FT_GOAL_LOWVIS_RECALL", 0.75)
    cfg["HARD_FT_GOAL_MIST_RECALL"] = env_float("HARD_FT_GOAL_MIST_RECALL", 0.45)
    cfg["HARD_FT_GOAL_FOG_RECALL"] = env_float("HARD_FT_GOAL_FOG_RECALL", 0.55)
    cfg["HARD_FT_GOAL_LOWVIS_CSI"] = env_float("HARD_FT_GOAL_LOWVIS_CSI", 0.22)
    cfg["HARD_FT_GOAL_LOWVIS_PRECISION"] = env_float("HARD_FT_GOAL_LOWVIS_PRECISION", 0.16)
    cfg["HARD_FT_MIN_LOWVIS_PRECISION"] = env_float("HARD_FT_MIN_LOWVIS_PRECISION", 0.12)
    cfg["HARD_FT_MIN_MIST_PRECISION"] = env_float("HARD_FT_MIN_MIST_PRECISION", 0.06)

    cfg["HARD_FT_W_LOWVIS_RECALL"] = env_float("HARD_FT_W_LOWVIS_RECALL", 0.30)
    cfg["HARD_FT_W_MIST_RECALL"] = env_float("HARD_FT_W_MIST_RECALL", 0.25)
    cfg["HARD_FT_W_FOG_RECALL"] = env_float("HARD_FT_W_FOG_RECALL", 0.20)
    cfg["HARD_FT_W_LOWVIS_CSI"] = env_float("HARD_FT_W_LOWVIS_CSI", 0.10)
    cfg["HARD_FT_W_PRECISION"] = env_float("HARD_FT_W_PRECISION", 0.10)
    cfg["HARD_FT_W_BALANCED_ACC"] = env_float("HARD_FT_W_BALANCED_ACC", 0.05)
    cfg["HARD_FT_W_FPR_PENALTY"] = env_float("HARD_FT_W_FPR_PENALTY", 0.75)
    cfg["HARD_FT_W_PRECISION_PENALTY"] = env_float("HARD_FT_W_PRECISION_PENALTY", 0.40)

    if args.init_ckpt_path:
        return args.init_ckpt_path
    return os.path.join(
        cfg["SAVE_CKPT_DIR"],
        f"{args.init_run_id}_S2_PhaseB_best_score.pt",
    )


def resolve_scaler_path(args: argparse.Namespace, dyn_vars_count: int) -> str:
    if args.scaler_path:
        return args.scaler_path
    return os.path.join(
        base.CONFIG["SAVE_CKPT_DIR"],
        f"robust_scaler_{args.init_run_id}_w{base.CONFIG['WINDOW_SIZE']}_dyn{dyn_vars_count}_s2_48h_pm10.pkl",
    )


def compute_soft_targets_hard(vis_raw, hard_labels, num_classes: int = 3):
    vis = vis_raw.clone()
    if vis.max() < 100:
        vis = vis * 1000.0

    soft = F.one_hot(hard_labels, num_classes).float()

    fm_mask = (vis >= 400) & (vis < 600)
    if fm_mask.any():
        alpha = (vis[fm_mask] - 400) / 200.0
        soft[fm_mask, 0] = 1 - alpha
        soft[fm_mask, 1] = alpha
        soft[fm_mask, 2] = 0

    mc_mask = (vis >= 900) & (vis < 1050)
    if mc_mask.any():
        alpha = (vis[mc_mask] - 900) / 150.0
        soft[mc_mask, 0] = 0
        soft[mc_mask, 1] = 1 - alpha
        soft[mc_mask, 2] = alpha

    return soft


def _role_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    if "hard_mist_missed" in lower or "mist_missed" in lower or "mist_to_clear" in lower:
        return "hard_mist_missed"
    if "hard_clear_false_lowvis" in lower or "clear_false_lowvis" in lower:
        return "hard_clear_false_lowvis"
    if "humid_clear_near_miss" in lower:
        return "hard_clear_false_lowvis"
    if "clear_false_low_vis" in lower or "clear_to_mist" in lower or "clear_to_fog" in lower:
        return "hard_clear_false_lowvis"
    if "mist" in lower and "miss" in lower:
        return "hard_mist_missed"
    if "clear" in lower and "false" in lower and ("lowvis" in lower or "low_vis" in lower):
        return "hard_clear_false_lowvis"
    return None


def _as_index_array(value, expected_len: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    arr = np.squeeze(arr)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr.reshape(-1)).astype(np.int64)
    flat = arr.reshape(-1)
    if flat.size == expected_len:
        finite_flat = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.number) else flat
        uniq = np.unique(finite_flat[: min(finite_flat.size, 10000)])
        if uniq.size <= 2 and set(uniq.astype(int).tolist()).issubset({0, 1}):
            return np.flatnonzero(flat.astype(bool)).astype(np.int64)
    if not np.issubdtype(flat.dtype, np.number):
        return np.empty(0, dtype=np.int64)
    flat = flat[np.isfinite(flat)]
    return flat.astype(np.int64, copy=False)


def _parse_int_cell(row: Dict[str, str], names: Sequence[str]) -> Optional[int]:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return None


def _parse_float_cell(row: Dict[str, str], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _read_csv_hard_indices(path: Path, gate_th: float) -> Tuple[List[int], List[int]]:
    role = _role_from_name(path.stem)
    mist_rows: List[int] = []
    clear_rows: List[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return mist_rows, clear_rows
        lower_names = {name.lower(): name for name in reader.fieldnames}
        for row_num, raw_row in enumerate(reader):
            row = {k.lower(): v for k, v in raw_row.items()}
            row_role = _role_from_name(str(row.get("pool", "")))
            idx = _parse_int_cell(
                row,
                ("row_index", "sample_index", "train_index", "orig_index", "index", "idx"),
            )
            if idx is None:
                idx = row_num
            y_true = _parse_int_cell(row, ("y_true", "target", "label", "y_cls", "class"))
            pred = _parse_int_cell(
                row,
                (
                    "pred_current",
                    "pred",
                    "prediction",
                    "argmax_fine",
                    "pred_fine",
                    "pred_binary_gate",
                    "y_pred",
                ),
            )
            low_prob = _parse_float_cell(
                row,
                ("p_lowvis_binary", "binary_low_prob", "low_prob", "p_low", "low_vis_prob", "lowvis_prob"),
            )
            err = str(row.get("error_type", row.get("error", ""))).strip().lower()
            effective_role = row_role or role

            if effective_role == "hard_mist_missed":
                mist_rows.append(idx)
                continue
            if effective_role == "hard_clear_false_lowvis":
                clear_rows.append(idx)
                continue

            if err in {"hard_mist_missed", "mist_missed", "mist_to_clear"}:
                mist_rows.append(idx)
            elif y_true == 1 and pred is not None and pred != 1:
                mist_rows.append(idx)
            elif y_true == 1 and low_prob is not None and low_prob < gate_th:
                mist_rows.append(idx)

            if err in {
                "hard_clear_false_lowvis",
                "clear_false_lowvis",
                "clear_false_low_vis",
                "clear_to_mist",
                "clear_to_fog",
            }:
                clear_rows.append(idx)
            elif y_true == 2 and pred is not None and pred <= 1:
                clear_rows.append(idx)
            elif y_true == 2 and low_prob is not None and low_prob >= gate_th:
                clear_rows.append(idx)

    _ = lower_names
    return mist_rows, clear_rows


def _indices_to_dataset_positions(
    dataset,
    row_indices: Sequence[int],
    class_id: int,
) -> np.ndarray:
    if len(row_indices) == 0:
        return np.empty(0, dtype=np.int64)
    raw = np.unique(np.asarray(row_indices, dtype=np.int64))
    raw = raw[raw >= 0]
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    orig = np.asarray(dataset.orig_indices, dtype=np.int64)
    _, pos, _ = np.intersect1d(orig, raw, return_indices=True)
    if pos.size == 0:
        return np.empty(0, dtype=np.int64)
    y = dataset.y_cls.numpy()
    pos = pos[y[pos] == class_id]
    return np.unique(pos.astype(np.int64, copy=False))


def load_hard_pool_positions(dataset, hard_pool_dir: str, gate_th: float, rank: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    expected_len = len(dataset)
    raw_mist: List[int] = []
    raw_clear: List[int] = []
    summary = {
        "hard_pool_dir": hard_pool_dir,
        "files_read": [],
        "raw_hard_mist_missed": 0,
        "raw_hard_clear_false_lowvis": 0,
        "train_hard_mist_missed": 0,
        "train_hard_clear_false_lowvis": 0,
    }

    if not hard_pool_dir:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), summary

    root = Path(hard_pool_dir).expanduser()
    if not root.exists():
        if rank == 0:
            print(f"[HardPool] WARN: HARD_POOL_DIR does not exist: {root}", flush=True)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), summary

    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".npz", ".csv"}])
    for path in files:
        try:
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as z:
                    for key in z.files:
                        role = _role_from_name(key)
                        if role is None:
                            continue
                        arr = np.asarray(z[key])
                        key_lower = key.lower()
                        is_index_like = any(tok in key_lower for tok in ("index", "indices", "idx", "mask"))
                        is_bool_mask = arr.dtype == np.bool_ and arr.size == expected_len
                        if not is_index_like and not is_bool_mask:
                            continue
                        idx = _as_index_array(arr, expected_len)
                        if role == "hard_mist_missed":
                            raw_mist.extend(idx.tolist())
                        elif role == "hard_clear_false_lowvis":
                            raw_clear.extend(idx.tolist())
                summary["files_read"].append(str(path))
            elif path.suffix.lower() == ".csv":
                mist_rows, clear_rows = _read_csv_hard_indices(path, gate_th)
                raw_mist.extend(mist_rows)
                raw_clear.extend(clear_rows)
                if mist_rows or clear_rows:
                    summary["files_read"].append(str(path))
        except Exception as exc:
            if rank == 0:
                print(f"[HardPool] WARN: failed to read {path}: {exc}", flush=True)

    summary["raw_hard_mist_missed"] = len(raw_mist)
    summary["raw_hard_clear_false_lowvis"] = len(raw_clear)
    hard_mist = _indices_to_dataset_positions(dataset, raw_mist, class_id=1)
    hard_clear = _indices_to_dataset_positions(dataset, raw_clear, class_id=2)
    summary["train_hard_mist_missed"] = int(hard_mist.size)
    summary["train_hard_clear_false_lowvis"] = int(hard_clear.size)
    return hard_mist, hard_clear, summary


class HardFlagDataset:
    def __init__(self, base_dataset, hard_mist_pos: np.ndarray, hard_clear_pos: np.ndarray):
        self.base_dataset = base_dataset
        self.y_cls = base_dataset.y_cls
        self.y_raw = base_dataset.y_raw
        self.y_reg = base_dataset.y_reg
        self.orig_indices = base_dataset.orig_indices
        self.hard_mist_mask = np.zeros(len(base_dataset), dtype=np.bool_)
        self.hard_clear_mask = np.zeros(len(base_dataset), dtype=np.bool_)
        self.hard_mist_mask[np.asarray(hard_mist_pos, dtype=np.int64)] = True
        self.hard_clear_mask[np.asarray(hard_clear_pos, dtype=np.int64)] = True

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx):
        bx, by, blog, braw = self.base_dataset[idx]
        flags = torch.tensor(
            [float(self.hard_mist_mask[idx]), float(self.hard_clear_mask[idx])],
            dtype=torch.float32,
        )
        return bx, by, blog, braw, flags


class HardMistClearBatchSampler(Sampler):
    def __init__(
        self,
        dataset: HardFlagDataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        epoch_length: int = 2000,
        fog_ratio: float = 0.16,
        mist_bg_ratio: float = 0.18,
        hard_mist_ratio: float = 0.20,
        hard_clear_ratio: float = 0.22,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch_length = int(epoch_length)
        self._epoch = 0

        y = dataset.y_cls.numpy()
        all_pos = np.arange(len(y), dtype=np.int64)
        hard_mist = np.flatnonzero(dataset.hard_mist_mask & (y == 1)).astype(np.int64)
        hard_clear = np.flatnonzero(dataset.hard_clear_mask & (y == 2)).astype(np.int64)
        self.pools = {
            "fog_bg": all_pos[y == 0],
            "mist_bg": all_pos[(y == 1) & ~dataset.hard_mist_mask],
            "clear_bg": all_pos[(y == 2) & ~dataset.hard_clear_mask],
            "hard_mist": hard_mist,
            "hard_clear": hard_clear,
            "mist_all": all_pos[y == 1],
            "clear_all": all_pos[y == 2],
        }
        for key, pool in list(self.pools.items()):
            self.pools[key] = self._rank_shard(pool)

        self.counts = self._build_counts(
            fog_ratio=fog_ratio,
            mist_bg_ratio=mist_bg_ratio,
            hard_mist_ratio=hard_mist_ratio,
            hard_clear_ratio=hard_clear_ratio,
        )

    def _rank_shard(self, pool: np.ndarray) -> np.ndarray:
        pool = np.asarray(pool, dtype=np.int64)
        if pool.size == 0 or self.world_size <= 1:
            return pool
        splits = np.array_split(pool, self.world_size)
        shard = splits[self.rank % self.world_size]
        return shard if shard.size > 0 else pool

    def _build_counts(self, **ratios: float) -> Dict[str, int]:
        safe_ratios = {k: max(0.0, float(v)) for k, v in ratios.items()}
        counts = {
            "fog_bg": int(round(self.batch_size * safe_ratios["fog_ratio"])),
            "mist_bg": int(round(self.batch_size * safe_ratios["mist_bg_ratio"])),
            "hard_mist": int(round(self.batch_size * safe_ratios["hard_mist_ratio"])),
            "hard_clear": int(round(self.batch_size * safe_ratios["hard_clear_ratio"])),
        }
        while sum(counts.values()) > max(0, self.batch_size - 1):
            largest = max(counts, key=lambda k: counts[k])
            if counts[largest] <= 0:
                break
            counts[largest] -= 1
        counts["clear_bg"] = self.batch_size - sum(counts.values())
        return counts

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    @staticmethod
    def _sample(rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if pool.size == 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(pool, size=n, replace=True)

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed + self.rank + self._epoch * 997)
        for _ in range(self.epoch_length):
            hard_mist_pool = self.pools["hard_mist"]
            if hard_mist_pool.size == 0:
                hard_mist_pool = self.pools["mist_all"]
            hard_clear_pool = self.pools["hard_clear"]
            if hard_clear_pool.size == 0:
                hard_clear_pool = self.pools["clear_all"]
            mist_bg_pool = self.pools["mist_bg"]
            if mist_bg_pool.size == 0:
                mist_bg_pool = self.pools["mist_all"]
            clear_bg_pool = self.pools["clear_bg"]
            if clear_bg_pool.size == 0:
                clear_bg_pool = self.pools["clear_all"]

            parts = [
                self._sample(rng, self.pools["fog_bg"], self.counts["fog_bg"]),
                self._sample(rng, mist_bg_pool, self.counts["mist_bg"]),
                self._sample(rng, hard_mist_pool, self.counts["hard_mist"]),
                self._sample(rng, hard_clear_pool, self.counts["hard_clear"]),
                self._sample(rng, clear_bg_pool, self.counts["clear_bg"]),
            ]
            batch = np.concatenate([p for p in parts if p.size > 0])
            if batch.size < self.batch_size:
                fallback = np.arange(len(self.dataset), dtype=np.int64)
                fill = rng.choice(fallback, size=self.batch_size - batch.size, replace=True)
                batch = np.concatenate([batch, fill])
            rng.shuffle(batch)
            yield batch[: self.batch_size].tolist()

    def __len__(self):
        return self.epoch_length


class HardMistClearFineTuneLoss(nn.Module):
    def __init__(self, cfg: Dict[str, float]):
        super().__init__()
        self.cfg = cfg
        self.base_loss = base.DualBranchLoss(
            binary_pos_weight=cfg["S2_BINARY_POS_WEIGHT"],
            fine_class_weight=[
                cfg["S2_FINE_CLASS_WEIGHT_FOG"],
                cfg["S2_FINE_CLASS_WEIGHT_MIST"],
                cfg["S2_FINE_CLASS_WEIGHT_CLEAR"],
            ],
            loss_type="ordinal_focal",
            gamma_per_class=[2.5, 3.0, 0.5],
            ordinal_cost=[[0, 1, 3], [1, 0, 2], [3, 2, 0]],
            alpha_binary=cfg["S2_LOSS_ALPHA_BINARY"],
            alpha_fine=cfg["S2_LOSS_ALPHA_FINE"],
            alpha_fp=cfg["S2_LOSS_ALPHA_FP"],
            alpha_fog_boost=cfg["S2_LOSS_ALPHA_FOG_BOOST"],
            alpha_mist_boost=cfg["S2_LOSS_ALPHA_MIST_BOOST"],
            alpha_clear_margin=cfg["S2_LOSS_ALPHA_CLEAR_MARGIN"],
            clear_margin=cfg["S2_CLEAR_MARGIN"],
            alpha_pair_margin=cfg["S2_LOSS_ALPHA_PAIR_MARGIN"],
            pair_margin=cfg["S2_PAIR_MARGIN"],
        )

    @staticmethod
    def class_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(mask.sum(), min=1.0)
        return (values * mask).sum() / denom

    def forward(self, fine_logits, low_vis_logit, targets, soft_targets=None, sample_flags=None):
        total, logs = self.base_loss(fine_logits, low_vis_logit, targets, soft_targets=soft_targets)
        probs = F.softmax(fine_logits, dim=1)
        low_logits = torch.clamp(low_vis_logit.reshape(-1), -20, 20)
        low_prob = torch.sigmoid(low_logits)

        if sample_flags is None:
            hard_mist = torch.zeros_like(low_prob)
            hard_clear = torch.zeros_like(low_prob)
        else:
            hard_mist = sample_flags[:, 0].float()
            hard_clear = sample_flags[:, 1].float()

        is_fog = (targets == 0).float()
        is_mist = (targets == 1).float()
        is_clear = (targets == 2).float()
        is_low = (targets <= 1).float()
        hard_clear_mask = hard_clear * is_clear

        fine_low_mass = torch.clamp(probs[:, 0] + probs[:, 1], 0.0, 1.0)
        low_miss = self.class_mean((1.0 - low_prob) ** 2, is_low)
        fine_low_miss = self.class_mean((1.0 - fine_low_mass) ** 2, is_low)

        margin = float(self.cfg["HARD_FT_MIST_CLEAR_MARGIN"])
        logit_mist = fine_logits[:, 1]
        logit_clear = fine_logits[:, 2]
        mist_weight = 1.0 + hard_mist * float(self.cfg["HARD_FT_HARD_MIST_EXTRA_WEIGHT"])
        mist_clear = torch.relu(margin - (logit_mist - logit_clear))
        hard_clear_margin = torch.relu(margin - (logit_clear - logit_mist))

        mist_clear_loss = self.class_mean(mist_clear * mist_weight, is_mist)
        hard_clear_margin_loss = self.class_mean(hard_clear_margin, hard_clear_mask)
        hard_clear_binary_loss = self.class_mean(F.softplus(low_logits), hard_clear_mask)
        hard_clear_low_mass = self.class_mean(fine_low_mass ** 2, hard_clear_mask)

        total = (
            total
            + self.cfg["HARD_FT_ALPHA_LOW_MISS"] * low_miss
            + self.cfg["HARD_FT_ALPHA_FINE_LOW_MASS"] * fine_low_miss
            + self.cfg["HARD_FT_ALPHA_MIST_CLEAR_MARGIN"] * mist_clear_loss
            + self.cfg["HARD_FT_ALPHA_HARD_CLEAR_MARGIN"] * hard_clear_margin_loss
            + self.cfg["HARD_FT_ALPHA_HARD_CLEAR_BINARY"] * hard_clear_binary_loss
            + self.cfg["HARD_FT_ALPHA_HARD_CLEAR_LOW_MASS"] * hard_clear_low_mass
        )
        logs["lowmiss"] = float(low_miss.detach().cpu())
        logs["finelow"] = float(fine_low_miss.detach().cpu())
        logs["mc"] = float(mist_clear_loss.detach().cpu())
        logs["hcm"] = float(hard_clear_margin_loss.detach().cpu())
        logs["hcb"] = float(hard_clear_binary_loss.detach().cpu())
        logs["hclow"] = float(hard_clear_low_mass.detach().cpu())
        return total, logs


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def binary_gate_pred(probs: np.ndarray, low_prob: np.ndarray, low_th: float) -> np.ndarray:
    pred = np.full(len(low_prob), 2, dtype=np.int64)
    low = low_prob >= low_th
    low_class = np.where(probs[:, 0] >= probs[:, 1], 0, 1)
    pred[low] = low_class[low]
    return pred


def classification_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    valid = (y_true >= 0) & (y_true <= 2) & (pred >= 0) & (pred <= 2)
    y = y_true[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    idx = y * 3 + p
    cm = np.bincount(idx, minlength=9).reshape(3, 3)
    n = float(cm.sum())
    out: Dict[str, float] = {"n": n, "accuracy": _safe_div(float(np.trace(cm)), n)}
    for cid, name in enumerate(("Fog", "Mist", "Clear")):
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        out[f"{name}_P"] = _safe_div(tp, tp + fp)
        out[f"{name}_R"] = _safe_div(tp, tp + fn)
        out[f"{name}_CSI"] = _safe_div(tp, tp + fp + fn)
        out[f"{name}_support"] = float(cm[cid, :].sum())
        out[f"pred_{name.lower()}"] = float(cm[:, cid].sum())

    true_low = y <= 1
    pred_low = p <= 1
    true_clear = y == 2
    low_tp = float((true_low & pred_low).sum())
    low_fp = float((~true_low & pred_low).sum())
    low_fn = float((true_low & ~pred_low).sum())
    out["low_vis_precision"] = _safe_div(low_tp, low_tp + low_fp)
    out["low_vis_recall"] = _safe_div(low_tp, low_tp + low_fn)
    out["low_vis_csi"] = _safe_div(low_tp, low_tp + low_fp + low_fn)
    out["false_positive_rate"] = _safe_div(float((true_clear & pred_low).sum()), float(true_clear.sum()))
    out["balanced_acc"] = float(np.mean([out["Fog_R"], out["Mist_R"], out["Clear_R"]]))
    out["recall_500"] = out["Fog_R"]
    out["recall_1000"] = out["Mist_R"]
    return out


def _bounded_ratio(value: float, goal: float) -> float:
    return float(min(max(value, 0.0) / max(goal, 1e-6), 1.0))


def gate_weighted_score(stats: Dict[str, float], cfg: Dict[str, float]) -> float:
    score = (
        cfg["HARD_FT_W_LOWVIS_RECALL"]
        * _bounded_ratio(stats["low_vis_recall"], cfg["HARD_FT_GOAL_LOWVIS_RECALL"])
        + cfg["HARD_FT_W_MIST_RECALL"]
        * _bounded_ratio(stats["Mist_R"], cfg["HARD_FT_GOAL_MIST_RECALL"])
        + cfg["HARD_FT_W_FOG_RECALL"]
        * _bounded_ratio(stats["Fog_R"], cfg["HARD_FT_GOAL_FOG_RECALL"])
        + cfg["HARD_FT_W_LOWVIS_CSI"]
        * _bounded_ratio(stats["low_vis_csi"], cfg["HARD_FT_GOAL_LOWVIS_CSI"])
        + cfg["HARD_FT_W_PRECISION"]
        * _bounded_ratio(stats["low_vis_precision"], cfg["HARD_FT_GOAL_LOWVIS_PRECISION"])
        + cfg["HARD_FT_W_BALANCED_ACC"] * float(stats["balanced_acc"])
    )
    max_fpr = float(cfg["TARGET_MAX_FPR"])
    fpr_excess = max(0.0, float(stats["false_positive_rate"]) - max_fpr)
    score -= cfg["HARD_FT_W_FPR_PENALTY"] * fpr_excess / max(max_fpr, 1e-6)

    lv_short = max(0.0, cfg["HARD_FT_MIN_LOWVIS_PRECISION"] - float(stats["low_vis_precision"]))
    mist_short = max(0.0, cfg["HARD_FT_MIN_MIST_PRECISION"] - float(stats["Mist_P"]))
    score -= cfg["HARD_FT_W_PRECISION_PENALTY"] * (
        lv_short / max(cfg["HARD_FT_MIN_LOWVIS_PRECISION"], 1e-6)
        + mist_short / max(cfg["HARD_FT_MIN_MIST_PRECISION"], 1e-6)
    )
    soft_fpr = float(cfg.get("TARGET_SOFT_FPR", max_fpr))
    if stats["false_positive_rate"] > soft_fpr:
        score -= 0.10 * (stats["false_positive_rate"] - soft_fpr) / max(max_fpr, 1e-6)
    return float(score)


class HardMistClearGateMetrics:
    def __init__(self, config: Dict[str, float]):
        self.cfg = config
        self.gate_th = float(config["LOWVIS_GATE_TH"])

    def evaluate(self, model, loader, device, rank=0, world_size=1, actual_val_size=None):
        model.eval()
        probs_l, low_l, targets_l = [], [], []

        if world_size > 1:
            torch.cuda.synchronize(device)
            n_batches = torch.tensor([len(loader)], dtype=torch.long, device=device)
            min_b = n_batches.clone()
            max_b = n_batches.clone()
            dist.all_reduce(min_b, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_b, op=dist.ReduceOp.MAX)
            if min_b.item() != max_b.item():
                raise RuntimeError(
                    f"[GateEval] Per-rank val DataLoader length mismatch: min={min_b.item()} max={max_b.item()}."
                )

        with torch.no_grad():
            for batch in loader:
                bx, by = batch[0], batch[1]
                bx = bx.to(device, non_blocking=True)
                fine, _, low_logit = model(bx)
                probs_l.append(F.softmax(fine, dim=1))
                low_l.append(torch.sigmoid(torch.clamp(low_logit.reshape(-1), -20, 20)))
                targets_l.append(by.to(device))

        if not probs_l:
            raise RuntimeError("[GateEval] Empty validation loader on at least one rank.")

        local_probs = torch.cat(probs_l, dim=0)
        local_low = torch.cat(low_l, dim=0)
        local_targets = torch.cat(targets_l, dim=0)

        if world_size > 1:
            torch.cuda.synchronize(device)
            local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=device)
            max_size = local_size.clone()
            dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

            if local_size < max_size:
                pad_size = max_size.item() - local_size.item()
                local_probs = torch.cat(
                    [
                        local_probs,
                        torch.zeros((pad_size, local_probs.size(1)), dtype=local_probs.dtype, device=device),
                    ],
                    dim=0,
                )
                local_low = torch.cat(
                    [local_low, torch.zeros((pad_size,), dtype=local_low.dtype, device=device)],
                    dim=0,
                )
                local_targets = torch.cat(
                    [
                        local_targets,
                        torch.full((pad_size,), -1, dtype=local_targets.dtype, device=device),
                    ],
                    dim=0,
                )

            gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
            gathered_low = [torch.zeros_like(local_low) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
            dist.all_gather(gathered_probs, local_probs)
            dist.all_gather(gathered_low, local_low)
            dist.all_gather(gathered_targets, local_targets)
            all_probs = torch.cat(gathered_probs, dim=0).cpu().numpy()
            all_low = torch.cat(gathered_low, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
        else:
            all_probs = local_probs.cpu().numpy()
            all_low = local_low.cpu().numpy()
            all_targets = local_targets.cpu().numpy()

        best_score = -1.0
        best_stats = None
        if rank == 0:
            n = actual_val_size if actual_val_size is not None else len(loader.dataset)
            probs = all_probs[:n]
            low_prob = all_low[:n]
            targets = all_targets[:n]
            valid = targets >= 0
            probs = probs[valid]
            low_prob = low_prob[valid]
            targets = targets[valid]

            gate_pred = binary_gate_pred(probs, low_prob, self.gate_th)
            best_stats = classification_metrics(targets, gate_pred)
            best_stats["binary_th"] = self.gate_th
            best_stats["score_name"] = "fixed_binary_gate"
            best_score = gate_weighted_score(best_stats, self.cfg)

            argmax_stats = classification_metrics(targets, np.argmax(probs, axis=1).astype(np.int64))
            print(
                f"  [GateEval] binary_gate(th={self.gate_th:.3f}) "
                f"LowR={best_stats['low_vis_recall']:.3f} "
                f"FogR={best_stats['Fog_R']:.3f} MistR={best_stats['Mist_R']:.3f} "
                f"LVPrec={best_stats['low_vis_precision']:.3f} "
                f"MistP={best_stats['Mist_P']:.3f} "
                f"FPR={best_stats['false_positive_rate']:.4f} "
                f"CSI={best_stats['low_vis_csi']:.3f} "
                f"Score={best_score:.4f}",
                flush=True,
            )
            print(
                f"  [GateEval] fine_argmax reference: "
                f"LowR={argmax_stats['low_vis_recall']:.3f} "
                f"MistR={argmax_stats['Mist_R']:.3f} "
                f"LVPrec={argmax_stats['low_vis_precision']:.3f} "
                f"FPR={argmax_stats['false_positive_rate']:.4f}",
                flush=True,
            )

        if world_size > 1:
            score_tensor = torch.tensor([best_score], dtype=torch.float32, device=device)
            dist.broadcast(score_tensor, src=0)
            best_score = float(score_tensor.item())
            base.safe_barrier(world_size, device)

        return {"score": best_score, "stats": best_stats, "thresholds": {"binary": self.gate_th}}


def build_model_raw(device: torch.device):
    return base.ImprovedDualStreamPMSTNet(
        window_size=base.CONFIG["WINDOW_SIZE"],
        hidden_dim=base.CONFIG["MODEL_HIDDEN_DIM"],
        num_classes=3,
        extra_feat_dim=base.CONFIG["FE_EXTRA_DIMS"],
        dyn_vars_count=base.CONFIG["DYN_VARS_COUNT"],
        dropout=base.CONFIG["MODEL_DROPOUT"],
    ).to(device)


def build_hard_optimizer(raw_model, local_rank: int, rank: int, world_size: int):
    head_names = {"fine_classifier", "low_vis_detector", "reg_head"}
    fusion_names = {"fusion_kan", "temporal_norm", "extra_encoder"}

    head_params = []
    fusion_params = []
    backbone_params = []
    for name, param in raw_model.named_parameters():
        param.requires_grad = True
        top = name.split(".")[0]
        if top in head_names:
            head_params.append(param)
        elif top in fusion_names:
            fusion_params.append(param)
        else:
            backbone_params.append(param)

    if rank == 0:
        print(
            "[HardFT-Optim] "
            f"backbone={sum(p.numel() for p in backbone_params)/1e6:.3f}M "
            f"fusion={sum(p.numel() for p in fusion_params)/1e6:.3f}M "
            f"head={sum(p.numel() for p in head_params)/1e6:.3f}M",
            flush=True,
        )
        print(
            "[HardFT-Optim] LR "
            f"backbone={base.CONFIG['HARD_FT_LR_BACKBONE']:.2e}, "
            f"fusion={base.CONFIG['HARD_FT_LR_FUSION']:.2e}, "
            f"head={base.CONFIG['HARD_FT_LR_HEAD']:.2e}",
            flush=True,
        )

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": base.CONFIG["HARD_FT_LR_BACKBONE"]},
            {"params": fusion_params, "lr": base.CONFIG["HARD_FT_LR_FUSION"]},
            {"params": head_params, "lr": base.CONFIG["HARD_FT_LR_HEAD"]},
        ],
        weight_decay=base.CONFIG["S2_WEIGHT_DECAY"],
    )
    return base.wrap_ddp(raw_model, local_rank, world_size), optimizer


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    while hasattr(dataset, "base_dataset"):
        dataset = dataset.base_dataset
    if hasattr(dataset, "X"):
        dataset.X = None


def build_loader_kwargs(num_workers: int) -> Dict[str, object]:
    if num_workers > 0:
        return {"num_workers": num_workers, "prefetch_factor": 2, "persistent_workers": True}
    return {"num_workers": 0}


def train_hard_stage(
    tag,
    model,
    tr_ds,
    val_ds,
    optimizer,
    loss_fn,
    device,
    rank,
    world_size,
    total_steps,
    val_int,
    batch_size,
    grad_accum,
    exp_id,
    patience=10,
    pretrained_state=None,
    l2sp_alpha=0.0,
):
    sampler = HardMistClearBatchSampler(
        tr_ds,
        batch_size,
        rank=rank,
        world_size=world_size,
        epoch_length=base.CONFIG["HARD_FT_EPOCH_LENGTH"],
        fog_ratio=base.CONFIG["HARD_FT_FOG_RATIO"],
        mist_bg_ratio=base.CONFIG["HARD_FT_MIST_BG_RATIO"],
        hard_mist_ratio=base.CONFIG["HARD_FT_HARD_MIST_RATIO"],
        hard_clear_ratio=base.CONFIG["HARD_FT_HARD_CLEAR_RATIO"],
    )
    num_workers = int(base.CONFIG.get("NUM_WORKERS", 0))
    loader = DataLoader(
        tr_ds,
        batch_sampler=sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        **build_loader_kwargs(num_workers),
    )

    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        **build_loader_kwargs(num_workers),
    )
    actual_val_size = len(val_ds)

    metrics_evaluator = HardMistClearGateMetrics(base.CONFIG)
    warmup_steps = base.CONFIG.get("S2_WARMUP_STEPS", 300)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps)),
        ],
        milestones=[warmup_steps],
    )

    ckpt_dir = base.CONFIG["SAVE_CKPT_DIR"]
    history_path = os.path.join(ckpt_dir, f"{exp_id}_{tag}_history.json")
    path_best_score = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_score.pt")
    path_best_fog_recall = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_fog_recall.pt")
    path_best_mist_recall = os.path.join(ckpt_dir, f"{exp_id}_{tag}_best_mist_recall.pt")
    path_latest = os.path.join(ckpt_dir, f"{exp_id}_{tag}_latest.pt")

    history = {
        "steps": [],
        "train_loss": [],
        "val_gate_score": [],
        "val_low_vis_recall": [],
        "val_fog_recall": [],
        "val_mist_recall": [],
        "val_fog_precision": [],
        "val_mist_precision": [],
        "val_clear_recall": [],
        "val_lv_precision": [],
        "val_low_vis_csi": [],
        "val_fpr": [],
        "val_accuracy": [],
    }
    best_score = -1.0
    best_fog_recall = -1.0
    best_mist_recall = -1.0
    no_improve_count = 0
    train_loss_accum, train_loss_count = 0.0, 0

    if rank == 0:
        print(
            f"\n[{tag}] Training started. total_steps={total_steps}, "
            f"grad_accum={grad_accum}, batch_size={batch_size}, "
            f"patience={patience}, num_workers={num_workers}",
            flush=True,
        )
        print(
            f"[{tag}] Batch counts per rank: {sampler.counts} "
            f"(clear_bg is the FPR guardrail background).",
            flush=True,
        )

    if world_size > 1:
        warmup_t = torch.zeros(1, device=device)
        dist.all_reduce(warmup_t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    if rank == 0:
        print(f"[{tag}] RCCL warmup all-reduce passed.", flush=True)

    step = 0
    batch_count = 0
    pseudo_epoch = 0
    iterator = iter(loader)
    model.train()
    first_batch_logged = False
    first_step_logged = False

    while step < total_steps:
        try:
            bx, by, blog, braw, flags = next(iterator)
        except StopIteration:
            pseudo_epoch += 1
            sampler.set_epoch(pseudo_epoch)
            iterator = iter(loader)
            bx, by, blog, braw, flags = next(iterator)

        soft_targets = None
        if base.CONFIG.get("USE_LABEL_SMOOTHING", False):
            soft_targets = compute_soft_targets_hard(braw, by).to(device)

        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)
        blog = blog.to(device, non_blocking=True)
        flags = flags.to(device, non_blocking=True)
        batch_count += 1
        if rank == 0 and not first_batch_logged:
            print(f"[{tag}] First batch received.", flush=True)
            first_batch_logged = True

        fine, reg, bin_out = model(bx)
        l_dual, loss_dict = loss_fn(fine, bin_out, by, soft_targets=soft_targets, sample_flags=flags)
        l_reg = F.mse_loss(reg.view(-1), blog)
        loss = l_dual + base.CONFIG["REG_LOSS_ALPHA"] * l_reg

        if pretrained_state is not None and l2sp_alpha > 0:
            raw_m = model.module if hasattr(model, "module") else model
            l2_sp = sum(
                ((p - pretrained_state[n]) ** 2).sum()
                for n, p in raw_m.named_parameters()
                if n in pretrained_state and p.requires_grad and p.shape == pretrained_state[n].shape
            )
            loss = loss + l2sp_alpha * l2_sp

        loss = loss / grad_accum
        is_last_accum_step = batch_count % grad_accum == 0
        if world_size > 1 and not is_last_accum_step:
            ctx = model.no_sync()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            loss.backward()

        if is_last_accum_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), base.CONFIG["GRAD_CLIP_NORM"]
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                if rank == 0:
                    print(
                        f"\n[WARNING] Step {step}: NaN/Inf grad norm ({grad_norm:.4f}), "
                        "skipping optimizer step.",
                        flush=True,
                    )

            optimizer.zero_grad()
            step += 1
            if rank == 0 and not first_step_logged:
                print(f"[{tag}] First step done (grad sync passed).", flush=True)
                first_step_logged = True

            if rank == 0:
                train_loss_accum += loss.item() * grad_accum
                train_loss_count += 1

            if rank == 0 and step % 50 == 0:
                print(
                    f"\r[{tag}] Step {step:>6}/{total_steps} | "
                    f"Loss={loss.item() * grad_accum:.4f} | "
                    f"bin={loss_dict['bin']:.4f} "
                    f"fine={loss_dict['fine']:.4f} "
                    f"mc={loss_dict.get('mc', 0.0):.4f} "
                    f"hcm={loss_dict.get('hcm', 0.0):.4f} "
                    f"hcb={loss_dict.get('hcb', 0.0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"NoImprove={no_improve_count}/{patience}",
                    end="",
                    flush=True,
                )

            if step % val_int == 0:
                if rank == 0:
                    print(f"\n[{tag}] === Validation at step {step} ===", flush=True)

                res = metrics_evaluator.evaluate(
                    model, val_loader, device, rank, world_size, actual_val_size=actual_val_size
                )
                model.train()
                ta = res["score"]
                stats = res["stats"]

                base.save_checkpoint(model, path_latest, rank, world_size)

                if rank == 0:
                    avg_loss = train_loss_accum / train_loss_count if train_loss_count > 0 else 0.0
                    train_loss_accum, train_loss_count = 0.0, 0
                    history["steps"].append(step)
                    history["train_loss"].append(round(avg_loss, 6))
                    history["val_gate_score"].append(round(ta, 6))
                    if stats is not None:
                        history["val_low_vis_recall"].append(round(stats.get("low_vis_recall", -1.0), 6))
                        history["val_fog_recall"].append(round(stats.get("Fog_R", -1.0), 6))
                        history["val_mist_recall"].append(round(stats.get("Mist_R", -1.0), 6))
                        history["val_fog_precision"].append(round(stats.get("Fog_P", -1.0), 6))
                        history["val_mist_precision"].append(round(stats.get("Mist_P", -1.0), 6))
                        history["val_clear_recall"].append(round(stats.get("Clear_R", -1.0), 6))
                        history["val_lv_precision"].append(round(stats.get("low_vis_precision", -1.0), 6))
                        history["val_low_vis_csi"].append(round(stats.get("low_vis_csi", -1.0), 6))
                        history["val_fpr"].append(round(stats.get("false_positive_rate", -1.0), 6))
                        history["val_accuracy"].append(round(stats.get("accuracy", -1.0), 6))
                    else:
                        for key in history:
                            if key not in {"steps", "train_loss", "val_gate_score"}:
                                history[key].append(-1.0)
                    try:
                        with open(history_path, "w", encoding="utf-8") as f:
                            json.dump(history, f, indent=2, ensure_ascii=False)
                    except Exception as exc:
                        print(f"  [History] WARN: failed to save {history_path}: {exc}", flush=True)

                if rank == 0 and stats is not None:
                    fog_r = stats.get("Fog_R", -1.0)
                    mist_r = stats.get("Mist_R", -1.0)

                    if ta > best_score:
                        best_score = ta
                        no_improve_count = 0
                        base.save_checkpoint(model, path_best_score, rank, world_size)
                        print(f"  [Ckpt] New best fixed-gate score = {best_score:.4f}", flush=True)
                    else:
                        no_improve_count += 1

                    if fog_r > best_fog_recall:
                        best_fog_recall = fog_r
                        base.save_checkpoint(model, path_best_fog_recall, rank, world_size)
                        print(f"  [Ckpt] New best gate Fog Recall = {best_fog_recall:.4f}", flush=True)

                    if mist_r > best_mist_recall:
                        best_mist_recall = mist_r
                        base.save_checkpoint(model, path_best_mist_recall, rank, world_size)
                        print(f"  [Ckpt] New best gate Mist Recall = {best_mist_recall:.4f}", flush=True)

                    print(
                        f"  [Best so far] GateScore={best_score:.4f} | "
                        f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f} | "
                        f"NoImprove={no_improve_count}/{patience}",
                        flush=True,
                    )

                if world_size > 1:
                    stop_tensor = torch.tensor([no_improve_count], dtype=torch.long, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    no_improve_count = int(stop_tensor.item())

                if patience > 0 and no_improve_count >= patience:
                    if rank == 0:
                        print(
                            f"\n[{tag}] Early stopping at step {step}. "
                            f"Best fixed-gate score={best_score:.4f}",
                            flush=True,
                        )
                    base.safe_barrier(world_size, device)
                    break

    if rank == 0:
        print(f"\n[{tag}] Training complete.", flush=True)
        print(
            f"  Final Best -> GateScore={best_score:.4f} | "
            f"FogR={best_fog_recall:.4f} | MistR={best_mist_recall:.4f}",
            flush=True,
        )


def main():
    args = parse_args()
    init_ckpt = update_hard_ft_config(args)

    local_rank, global_rank, world_size = base.init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        os.makedirs(base.CONFIG["SAVE_CKPT_DIR"], exist_ok=True)
        print("[HardFT] Hard Mist/Clear fine-tuning", flush=True)
        print(f"[HardFT] Run ID: {args.hard_ft_run_id}", flush=True)
        print(f"[HardFT] Init run: {args.init_run_id}", flush=True)
        print(f"[HardFT] Init checkpoint: {init_ckpt}", flush=True)
        print(f"[HardFT] HARD_POOL_DIR: {args.hard_pool_dir or '(empty; fallback sampler only)'}", flush=True)
        print(
            f"[HardFT] Fixed binary gate: th={base.CONFIG['LOWVIS_GATE_TH']:.3f}, "
            f"target_max_fpr={base.CONFIG['TARGET_MAX_FPR']:.4f}",
            flush=True,
        )
        print(f"[HardFT] World size: {world_size}", flush=True)

    base.safe_barrier(world_size, device)

    dyn_res, fe_res = base.resolve_feature_layout_from_x_train(
        base.CONFIG["S2_DATA_DIR"], base.CONFIG["WINDOW_SIZE"]
    )
    base.CONFIG["DYN_VARS_COUNT"] = int(dyn_res)
    base.CONFIG["FE_EXTRA_DIMS"] = int(fe_res)
    scaler_path = resolve_scaler_path(args, int(dyn_res))

    raw_model = build_model_raw(device)
    base.load_checkpoint(raw_model, init_ckpt, global_rank, world_size, device)
    pretrained_state = {
        k: v.clone().detach().to(device)
        for k, v in raw_model.state_dict().items()
    }

    if global_rank == 0:
        print(f"[HardFT] Scaler: {scaler_path}", flush=True)
    scaler = base.joblib.load(scaler_path)
    tr_ds_raw, val_ds, _ = base.load_data(
        base.CONFIG["S2_DATA_DIR"],
        scaler,
        global_rank,
        local_rank,
        device,
        True,
        base.CONFIG["WINDOW_SIZE"],
        world_size,
        args.hard_ft_run_id,
    )

    hard_mist_pos, hard_clear_pos, hard_summary = load_hard_pool_positions(
        tr_ds_raw,
        args.hard_pool_dir,
        base.CONFIG["LOWVIS_GATE_TH"],
        global_rank,
    )
    tr_ds = HardFlagDataset(tr_ds_raw, hard_mist_pos, hard_clear_pos)

    if global_rank == 0:
        print(f"[HardFT] Train={len(tr_ds)} Val={len(val_ds)}", flush=True)
        print(
            "[HardFT] Hard pools: "
            f"mist_missed={hard_summary['train_hard_mist_missed']} "
            f"clear_false_lowvis={hard_summary['train_hard_clear_false_lowvis']} "
            f"files={len(hard_summary['files_read'])}",
            flush=True,
        )
        if hard_summary["train_hard_mist_missed"] == 0 or hard_summary["train_hard_clear_false_lowvis"] == 0:
            print(
                "[HardFT] WARN: one or both hard pools are empty after train split/class filtering; "
                "sampler will fall back to class background pools for missing buckets.",
                flush=True,
            )

    model, optimizer = build_hard_optimizer(raw_model, local_rank, global_rank, world_size)
    loss_fn = HardMistClearFineTuneLoss(base.CONFIG).to(device)

    train_hard_stage(
        tag=DEFAULT_TAG,
        model=model,
        tr_ds=tr_ds,
        val_ds=val_ds,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        rank=global_rank,
        world_size=world_size,
        total_steps=base.CONFIG["HARD_FT_STEPS"],
        val_int=base.CONFIG["S2_VAL_INTERVAL"],
        batch_size=base.CONFIG["S2_BATCH_SIZE"],
        grad_accum=base.CONFIG["S2_GRAD_ACCUM"],
        exp_id=args.hard_ft_run_id,
        patience=base.CONFIG["S2_ES_PATIENCE"],
        pretrained_state=pretrained_state,
        l2sp_alpha=base.CONFIG["HARD_FT_L2SP"],
    )

    raw_final = base.rewrap_ddp(model, world_size)
    best_path = os.path.join(
        base.CONFIG["SAVE_CKPT_DIR"],
        f"{args.hard_ft_run_id}_{DEFAULT_TAG}_best_score.pt",
    )
    if os.path.exists(best_path):
        base.load_checkpoint(raw_final, best_path, global_rank, world_size, device)

    if global_rank == 0:
        print("[HardFT] Calibrating temperature on validation set.", flush=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=base.CONFIG["S2_BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    temp = base.calibrate_temperature(raw_final, val_loader, device, base.CONFIG, rank=global_rank)

    if global_rank == 0:
        meta_path = os.path.join(base.CONFIG["SAVE_CKPT_DIR"], f"{args.hard_ft_run_id}_meta.json")
        meta = {
            "run_exp_id": args.hard_ft_run_id,
            "init_run_id": args.init_run_id,
            "init_checkpoint": init_ckpt,
            "scaler_path": scaler_path,
            "hard_pool_summary": hard_summary,
            "temperature": temp,
            "config_subset": {
                k: base.CONFIG[k]
                for k in sorted(base.CONFIG)
                if k.startswith("HARD_FT")
                or k.startswith("TARGET_")
                or k in (
                    "LOWVIS_GATE_TH",
                    "S2_BINARY_POS_WEIGHT",
                    "S2_FINE_CLASS_WEIGHT_FOG",
                    "S2_FINE_CLASS_WEIGHT_MIST",
                    "S2_FINE_CLASS_WEIGHT_CLEAR",
                    "S2_LOSS_ALPHA_FP",
                    "S2_CLEAR_MARGIN",
                )
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[HardFT] Wrote metadata: {meta_path}", flush=True)

    del pretrained_state
    base.cleanup_temp_files(args.hard_ft_run_id)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

    if global_rank == 0:
        print("[HardFT] Job finished.", flush=True)


if __name__ == "__main__":
    main()
