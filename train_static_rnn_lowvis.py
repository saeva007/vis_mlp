#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the paper-facing Static-MLP + RNN low-visibility model.

This script is intentionally simpler than ``PMST_net_test_11_s2_pm10.py``:

* dynamic forecast fields, zenith, and optional PM channels go through one GRU
  or LSTM encoder;
* station static variables and vegetation type go through one MLP branch;
* optional feature-engineering (FE) variables are encoded by one small MLP;
* a compact fusion MLP produces three visibility classes, with an optional
  log-visibility auxiliary regression head.

It keeps the training tricks that have been stable for this repository:
runtime feature-layout checks, PM ablation by masking channels in the same data
files, log1p transforms for skewed dynamic variables, RobustScaler caching,
stratified low-visibility batch sampling, focal loss with class weights,
validation-time threshold search, gradient clipping, warmup+cosine LR, optional
L2-SP, and compatible Stage-1-to-Stage-2 checkpoint loading.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

# Reuse the repository's already-debugged DCU/DDP and local-copy utilities.
from PMST_net_test_11_s2_pm10 import copy_to_local, init_distributed, safe_barrier


DEFAULT_BASE = "/public/home/putianshu/vis_mlp"
DEFAULT_S1_DIR = f"{DEFAULT_BASE}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25"
DEFAULT_S2_DIR = f"{DEFAULT_BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_CKPT_DIR = f"{DEFAULT_BASE}/checkpoints"


@dataclass
class Layout:
    window_size: int
    dyn_vars: int
    fe_dim: int

    @property
    def split_dyn(self) -> int:
        return self.window_size * self.dyn_vars

    @property
    def core_dim(self) -> int:
        return self.split_dyn + 5

    @property
    def total_expected_dim(self) -> int:
        return self.split_dyn + 5 + 1 + self.fe_dim


class StratifiedBalancedBatchSampler(Sampler[List[int]]):
    """DDP-safe balanced sampler over dataset-local indices."""

    def __init__(
        self,
        dataset: "LowVisDataset",
        batch_size: int,
        fog_ratio: float,
        mist_ratio: float,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        epoch_length: int = 2000,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch_length = int(epoch_length)
        self.epoch = 0

        y = dataset.y_cls.numpy()
        self.n_fog = max(1, int(self.batch_size * float(fog_ratio)))
        self.n_mist = max(1, int(self.batch_size * float(mist_ratio)))
        self.n_clear = self.batch_size - self.n_fog - self.n_mist
        if self.n_clear < 1:
            raise ValueError("fog_ratio + mist_ratio leaves no clear samples in a batch")

        all_pos = np.arange(len(y))
        self.pos = {
            0: all_pos[y == 0],
            1: all_pos[y == 1],
            2: all_pos[y == 2],
        }
        for k in self.pos:
            if len(self.pos[k]) == 0:
                self.pos[k] = all_pos[:1]
            chunks = np.array_split(self.pos[k], max(1, self.world_size))
            shard = chunks[self.rank % len(chunks)]
            self.pos[k] = shard if len(shard) else self.pos[k]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed + self.rank + 997 * self.epoch)
        for _ in range(self.epoch_length):
            f = rng.choice(self.pos[0], size=self.n_fog, replace=True)
            m = rng.choice(self.pos[1], size=self.n_mist, replace=True)
            c = rng.choice(self.pos[2], size=self.n_clear, replace=True)
            batch = np.concatenate([f, m, c])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.epoch_length


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static MLP + GRU/LSTM low-vis training")

    p.add_argument("--mode", choices=["s1", "s2", "both"], default="both")
    p.add_argument("--encoder", choices=["gru", "lstm"], default="gru")
    p.add_argument("--run-id", default=os.environ.get("LOWVIS_RNN_RUN_ID", f"exp_{int(time.time())}_static_rnn"))
    p.add_argument("--base-path", default=DEFAULT_BASE)
    p.add_argument("--s1-data-dir", default=DEFAULT_S1_DIR)
    p.add_argument("--s2-data-dir", default=DEFAULT_S2_DIR)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--pretrained-ckpt", default="")

    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--static-hidden-dim", type=int, default=96)
    p.add_argument("--fe-hidden-dim", type=int, default=128)
    p.add_argument("--fusion-hidden-dim", type=int, default=256)
    p.add_argument("--veg-emb-dim", type=int, default=16)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--pooling", choices=["mean", "last", "attention"], default="mean")
    p.add_argument("--no-fe", action="store_true", help="Drop the FE block at training time.")
    p.add_argument("--no-pm", action="store_true", help="Zero PM channels while keeping data layout fixed.")
    p.add_argument("--aux-reg-weight", type=float, default=0.0)

    p.add_argument("--s1-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S1_STEPS", "15000")))
    p.add_argument("--s2-phase-a-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_A_STEPS", "8000")))
    p.add_argument("--s2-phase-b-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_B_STEPS", "22000")))
    p.add_argument("--val-interval", type=int, default=int(os.environ.get("LOWVIS_RNN_VAL_INTERVAL", "500")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("LOWVIS_RNN_BATCH_SIZE", "512")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("LOWVIS_RNN_GRAD_ACCUM", "2")))
    p.add_argument("--epoch-length", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("LOWVIS_RNN_NUM_WORKERS", "0")))
    p.add_argument("--patience", type=int, default=int(os.environ.get("LOWVIS_RNN_PATIENCE", "10")))

    p.add_argument("--s1-lr", type=float, default=2e-4)
    p.add_argument("--s2-lr-head-a", type=float, default=8e-5)
    p.add_argument("--s2-lr-backbone-b", type=float, default=3e-6)
    p.add_argument("--s2-lr-head-b", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--l2sp-alpha-a", type=float, default=1e-4)
    p.add_argument("--l2sp-alpha-b", type=float, default=5e-5)

    p.add_argument("--fog-ratio-s1", type=float, default=0.20)
    p.add_argument("--mist-ratio-s1", type=float, default=0.20)
    p.add_argument("--fog-ratio-s2", type=float, default=0.18)
    p.add_argument("--mist-ratio-s2", type=float, default=0.22)

    p.add_argument("--class-weight-fog", type=float, default=2.0)
    p.add_argument("--class-weight-mist", type=float, default=2.0)
    p.add_argument("--class-weight-clear", type=float, default=0.8)
    p.add_argument("--focal-gamma-fog", type=float, default=2.5)
    p.add_argument("--focal-gamma-mist", type=float, default=3.0)
    p.add_argument("--focal-gamma-clear", type=float, default=0.5)
    p.add_argument("--alpha-clear-fp", type=float, default=2.0)
    p.add_argument("--alpha-recall-boost", type=float, default=0.2)
    p.add_argument("--label-smoothing", action="store_true", default=True)
    p.add_argument("--no-label-smoothing", dest="label_smoothing", action="store_false")

    p.add_argument("--selection-metric", choices=["recall_csi", "csi", "recall"], default="recall_csi")
    p.add_argument("--min-fog-precision", type=float, default=0.10)
    p.add_argument("--min-mist-precision", type=float, default=0.10)
    p.add_argument("--min-clear-recall", type=float, default=0.88)
    p.add_argument("--threshold-grid-low", type=float, default=0.10)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--threshold-grid-step", type=float, default=0.03)
    return p.parse_args()


def rank0(rank: int, text: str) -> None:
    if rank == 0:
        print(text, flush=True)


def resolve_dyn_and_fe_dims(total_dim: int, win_size: int) -> Tuple[int, int]:
    rest = int(total_dim) - 6
    if rest <= 0:
        raise ValueError(f"total_dim={total_dim} too small for dyn+static+veg+FE layout")
    for dyn in (27, 26, 25, 24):
        fe = rest - dyn * int(win_size)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot resolve feature layout: total_dim={total_dim}, window={win_size}")


def resolve_layout_from_file(path_x: str, win_size: int) -> Layout:
    shape = np.load(path_x, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{path_x} must be 2D, got shape={shape}")
    dyn, fe = resolve_dyn_and_fe_dims(int(shape[1]), win_size)
    return Layout(window_size=win_size, dyn_vars=dyn, fe_dim=fe)


def pm_indices(dyn_vars: int) -> List[int]:
    if dyn_vars >= 27:
        return [dyn_vars - 2, dyn_vars - 1]
    if dyn_vars >= 25:
        return [dyn_vars - 1]
    return []


def log1p_dyn_indices(dyn_vars: int) -> List[int]:
    idxs = [2, 4, 9]
    idxs.extend(pm_indices(dyn_vars))
    return sorted(set(i for i in idxs if 0 <= i < dyn_vars))


def visibility_to_labels(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_raw = np.asarray(y, dtype=np.float32).copy()
    if len(y_raw) and np.nanmax(y_raw) < 100:
        y_raw *= 1000.0
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    y_cls[y_raw >= 500.0] = 1
    y_cls[y_raw >= 1000.0] = 2
    return y_raw, y_cls


def build_dyn_log_mask(layout: Layout) -> np.ndarray:
    mask = np.zeros(layout.split_dyn, dtype=bool)
    for t in range(layout.window_size):
        for idx in log1p_dyn_indices(layout.dyn_vars):
            mask[t * layout.dyn_vars + idx] = True
    return mask


def apply_core_transform(core: np.ndarray, layout: Layout, use_pm: bool, log_mask: np.ndarray) -> np.ndarray:
    out = core.astype(np.float32, copy=True)
    if not use_pm:
        for t in range(layout.window_size):
            for idx in pm_indices(layout.dyn_vars):
                out[:, t * layout.dyn_vars + idx] = 0.0
    dyn = out[:, : layout.split_dyn]
    dyn[:] = np.where(log_mask, np.log1p(np.maximum(dyn, 0.0)), dyn)
    return out


def scaler_cache_path(args: argparse.Namespace, stage: str, layout: Layout, use_pm: bool) -> str:
    pm_tag = "pm" if use_pm else "nopm"
    name = f"robust_scaler_{args.run_id}_{stage}_w{layout.window_size}_dyn{layout.dyn_vars}_{pm_tag}.pkl"
    return os.path.join(args.ckpt_dir, name)


class LowVisDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_raw: np.ndarray,
        y_cls: np.ndarray,
        layout: Layout,
        scaler: Optional[RobustScaler],
        use_fe: bool,
        use_pm: bool,
    ) -> None:
        self.x_path = x_path
        self.layout = layout
        self.scaler = scaler
        self.use_fe = bool(use_fe)
        self.use_pm = bool(use_pm)
        self.y_raw = torch.as_tensor(np.maximum(y_raw, 0.0), dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.log1p(self.y_raw)
        self.log_mask = build_dyn_log_mask(layout)
        self.X = None

    def __len__(self) -> int:
        return len(self.y_cls)

    def __getitem__(self, idx: int):
        if self.X is None:
            self.X = np.load(self.x_path, mmap_mode="r")
        row = self.X[idx]
        core = row[: self.layout.core_dim][None, :]
        core = apply_core_transform(core, self.layout, self.use_pm, self.log_mask)[0]
        if self.scaler is not None:
            core = (core - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        core = np.clip(core, -10.0, 10.0).astype(np.float32)
        veg = np.asarray([row[self.layout.split_dyn + 5]], dtype=np.float32)
        parts = [core, veg]
        if self.use_fe:
            fe = row[self.layout.split_dyn + 6 : self.layout.split_dyn + 6 + self.layout.fe_dim]
            parts.append(np.clip(fe.astype(np.float32), -10.0, 10.0))
        final = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.from_numpy(final).float(), self.y_cls[idx], self.y_reg[idx], self.y_raw[idx]


def load_split_paths(
    data_dir: str,
    stage: str,
    rank: int,
    local_rank: int,
    world_size: int,
    exp_id: str,
) -> Tuple[str, str, str, str]:
    req = [f"X_train.npy", f"y_train.npy", f"X_val.npy", f"y_val.npy"]
    for name in req:
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} for {stage}: {path}")
    x_tr = copy_to_local(os.path.join(data_dir, "X_train.npy"), rank, local_rank, world_size, exp_id)
    y_tr = copy_to_local(os.path.join(data_dir, "y_train.npy"), rank, local_rank, world_size, exp_id)
    x_va = copy_to_local(os.path.join(data_dir, "X_val.npy"), rank, local_rank, world_size, exp_id)
    y_va = copy_to_local(os.path.join(data_dir, "y_val.npy"), rank, local_rank, world_size, exp_id)
    return x_tr, y_tr, x_va, y_va


def fit_or_load_scaler(
    args: argparse.Namespace,
    stage: str,
    x_train_path: str,
    layout: Layout,
    use_pm: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> RobustScaler:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = scaler_cache_path(args, stage, layout, use_pm)
    log_mask = build_dyn_log_mask(layout)
    safe_barrier(world_size, device)
    if rank == 0:
        if not os.path.exists(path):
            print(f"[Scaler] fitting {path}", flush=True)
            x_m = np.load(x_train_path, mmap_mode="r")
            n = len(x_m)
            max_samples = min(200000, n)
            rng = np.random.default_rng(42)
            idx = np.arange(n) if n <= max_samples else np.sort(rng.choice(n, size=max_samples, replace=False))
            core = x_m[idx, : layout.core_dim].astype(np.float32)
            core = apply_core_transform(core, layout, use_pm, log_mask)
            scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(core)
            joblib.dump(scaler, path)
            print(f"[Scaler] saved {path}", flush=True)
        else:
            print(f"[Scaler] cache hit {path}", flush=True)
    safe_barrier(world_size, device)
    return joblib.load(path)


def load_data(
    args: argparse.Namespace,
    data_dir: str,
    stage: str,
    use_fe: bool,
    use_pm: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> Tuple[LowVisDataset, LowVisDataset, Layout, RobustScaler]:
    x_tr, y_tr, x_va, y_va = load_split_paths(data_dir, stage, rank, local_rank, world_size, args.run_id)
    layout = resolve_layout_from_file(x_tr, args.window_size)
    va_layout = resolve_layout_from_file(x_va, args.window_size)
    if asdict(layout) != asdict(va_layout):
        raise ValueError(f"train/val layout mismatch: {layout} vs {va_layout}")
    y_raw_tr, y_cls_tr = visibility_to_labels(np.load(y_tr))
    y_raw_va, y_cls_va = visibility_to_labels(np.load(y_va))
    if len(y_raw_tr) != np.load(x_tr, mmap_mode="r").shape[0]:
        raise ValueError("train X/y length mismatch")
    if len(y_raw_va) != np.load(x_va, mmap_mode="r").shape[0]:
        raise ValueError("val X/y length mismatch")
    scaler = fit_or_load_scaler(args, stage, x_tr, layout, use_pm, rank, world_size, device)
    tr_ds = LowVisDataset(x_tr, y_raw_tr, y_cls_tr, layout, scaler, use_fe, use_pm)
    va_ds = LowVisDataset(x_va, y_raw_va, y_cls_va, layout, scaler, use_fe, use_pm)
    rank0(rank, f"[Data:{stage}] train={len(tr_ds)} val={len(va_ds)} layout={layout} use_fe={use_fe} use_pm={use_pm}")
    return tr_ds, va_ds, layout, scaler


class StaticRNNLowVisNet(nn.Module):
    def __init__(
        self,
        layout: Layout,
        encoder: str,
        hidden_dim: int,
        static_hidden_dim: int,
        fe_hidden_dim: int,
        fusion_hidden_dim: int,
        veg_emb_dim: int,
        rnn_layers: int,
        dropout: float,
        bidirectional: bool,
        pooling: str,
        use_fe: bool,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.encoder = encoder
        self.use_fe = bool(use_fe)
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.bidirectional = bool(bidirectional)
        rnn_dropout = dropout if rnn_layers > 1 else 0.0

        self.dynamic_proj = nn.Sequential(
            nn.Linear(layout.dyn_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        rnn_cls = nn.GRU if encoder == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        dyn_out = hidden_dim * (2 if bidirectional else 1)
        self.dynamic_norm = nn.LayerNorm(dyn_out)
        self.attn_pool = nn.Linear(dyn_out, 1) if pooling == "attention" else None

        self.veg_embedding = nn.Embedding(32, veg_emb_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(5 + veg_emb_dim, static_hidden_dim),
            nn.LayerNorm(static_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, static_hidden_dim),
            nn.GELU(),
        )

        if self.use_fe:
            self.fe_encoder = nn.Sequential(
                nn.Linear(layout.fe_dim, fe_hidden_dim),
                nn.LayerNorm(fe_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fe_encoder = None
            fe_hidden_dim = 0

        fusion_in = dyn_out + static_hidden_dim + fe_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
        )
        out_dim = fusion_hidden_dim // 2
        self.class_head = nn.Linear(out_dim, 3)
        self.reg_head = nn.Linear(out_dim, 1)

    def _pool_dynamic(self, seq: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return seq[:, -1, :]
        if self.pooling == "attention":
            w = torch.softmax(self.attn_pool(seq).squeeze(-1), dim=1)
            return torch.sum(seq * w.unsqueeze(-1), dim=1)
        return seq.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split_dyn = self.layout.split_dyn
        split_static = split_dyn + 5
        dyn = x[:, :split_dyn].reshape(-1, self.layout.window_size, self.layout.dyn_vars)
        stat = x[:, split_dyn:split_static]
        veg = torch.clamp(x[:, split_static].long(), 0, 31)
        extra = x[:, split_static + 1 :] if self.use_fe else None

        dyn_in = self.dynamic_proj(dyn)
        dyn_seq, _ = self.rnn(dyn_in)
        dyn_feat = self.dynamic_norm(self._pool_dynamic(dyn_seq))

        stat_feat = self.static_encoder(torch.cat([stat, self.veg_embedding(veg)], dim=1))
        parts = [dyn_feat, stat_feat]
        if self.fe_encoder is not None and extra is not None:
            parts.append(self.fe_encoder(extra))
        emb = self.fusion(torch.cat(parts, dim=1))
        return self.class_head(emb), self.reg_head(emb).squeeze(1)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights: List[float], gamma: List[float]) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, soft_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1).clamp(1e-7, 1.0 - 1e-7)
        if soft_targets is None:
            soft_targets = F.one_hot(targets, 3).float()
        focal = (1.0 - probs) ** self.gamma.unsqueeze(0)
        weight = self.class_weights.unsqueeze(0)
        loss = -(soft_targets * weight * focal * torch.log(probs)).sum(dim=1)
        return loss.mean()


def soft_targets_from_visibility(raw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    soft = F.one_hot(labels, 3).float()
    fm = (raw >= 400.0) & (raw < 600.0)
    if fm.any():
        alpha = (raw[fm] - 400.0) / 200.0
        soft[fm, 0] = 1.0 - alpha
        soft[fm, 1] = alpha
        soft[fm, 2] = 0.0
    mc = (raw >= 800.0) & (raw < 1200.0)
    if mc.any():
        alpha = (raw[mc] - 800.0) / 400.0
        soft[mc, 0] = 0.0
        soft[mc, 1] = 1.0 - alpha
        soft[mc, 2] = alpha
    return soft


def combined_loss(
    args: argparse.Namespace,
    focal: WeightedFocalLoss,
    logits: torch.Tensor,
    reg: torch.Tensor,
    y: torch.Tensor,
    y_reg: torch.Tensor,
    y_raw: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    soft = soft_targets_from_visibility(y_raw, y) if args.label_smoothing else None
    l_cls = focal(logits, y, soft)
    probs = torch.softmax(logits, dim=1)
    clear = (y == 2).float()
    fog = (y == 0).float()
    mist = (y == 1).float()
    p_low = torch.clamp(probs[:, 0] + probs[:, 1], 0.0, 1.0)
    l_fp = torch.mean((p_low ** 2) * clear)
    l_boost = torch.mean(((1.0 - probs[:, 0]) ** 2) * fog) + torch.mean(((1.0 - probs[:, 1]) ** 2) * mist)
    # Keep the auxiliary head in the autograd graph even when the auxiliary
    # objective is disabled; otherwise DDP reports reg_head as an unused branch.
    l_reg = F.mse_loss(reg, y_reg) if args.aux_reg_weight > 0 else reg.sum() * 0.0
    total = l_cls + args.alpha_clear_fp * l_fp + args.alpha_recall_boost * l_boost + args.aux_reg_weight * l_reg
    return total, {"cls": float(l_cls.detach()), "fp": float(l_fp.detach()), "boost": float(l_boost.detach()), "reg": float(l_reg.detach())}


def class_stats(y_true: np.ndarray, pred: np.ndarray, cls: int) -> Tuple[float, float, float]:
    tp = np.sum((pred == cls) & (y_true == cls))
    fp = np.sum((pred == cls) & (y_true != cls))
    fn = np.sum((pred != cls) & (y_true == cls))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    csi = tp / (tp + fp + fn + 1e-6)
    return float(precision), float(recall), float(csi)


def pred_from_thresholds(probs: np.ndarray, fog_th: float, mist_th: float) -> np.ndarray:
    pred = np.full(len(probs), 2, dtype=np.int64)
    fog = (probs[:, 0] > fog_th) & (probs[:, 0] >= probs[:, 1])
    mist = (probs[:, 1] > mist_th) & (probs[:, 1] > probs[:, 0])
    pred[fog] = 0
    pred[mist] = 1
    return pred


def build_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    fp, fr, fc = class_stats(y_true, pred, 0)
    mp, mr, mc = class_stats(y_true, pred, 1)
    cp, cr, cc = class_stats(y_true, pred, 2)
    low_pred = pred <= 1
    low_true = y_true <= 1
    clear = y_true == 2
    lv_prec = np.sum(low_pred & low_true) / (np.sum(low_pred) + 1e-6)
    lv_recall = np.sum(low_pred & low_true) / (np.sum(low_true) + 1e-6)
    lv_csi = np.sum(low_pred & low_true) / (np.sum(low_pred & ~low_true) + np.sum(~low_pred & low_true) + np.sum(low_pred & low_true) + 1e-6)
    fpr = np.sum(low_pred & clear) / (np.sum(clear) + 1e-6)
    return {
        "Fog_P": fp, "Fog_R": fr, "Fog_CSI": fc,
        "Mist_P": mp, "Mist_R": mr, "Mist_CSI": mc,
        "Clear_P": cp, "Clear_R": cr, "Clear_CSI": cc,
        "low_vis_precision": float(lv_prec),
        "low_vis_recall": float(lv_recall),
        "low_vis_csi": float(lv_csi),
        "false_positive_rate": float(fpr),
        "accuracy": float(np.mean(pred == y_true)),
    }


def score_metrics(args: argparse.Namespace, metrics: Dict[str, float]) -> float:
    if args.selection_metric == "csi":
        return 0.45 * metrics["Fog_CSI"] + 0.45 * metrics["Mist_CSI"] + 0.10 * metrics["low_vis_precision"] - 0.05 * metrics["false_positive_rate"]
    if args.selection_metric == "recall":
        return 0.45 * metrics["Fog_R"] + 0.45 * metrics["Mist_R"] + 0.10 * metrics["low_vis_precision"] - 0.10 * metrics["false_positive_rate"]
    return (
        0.25 * metrics["Fog_CSI"]
        + 0.25 * metrics["Mist_CSI"]
        + 0.20 * metrics["Fog_R"]
        + 0.20 * metrics["Mist_R"]
        + 0.10 * metrics["low_vis_precision"]
        - 0.05 * metrics["false_positive_rate"]
    )


def threshold_search(args: argparse.Namespace, probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    grid = np.arange(args.threshold_grid_low, args.threshold_grid_high + 1e-9, args.threshold_grid_step)
    best = (-1e9, {"fog": 0.5, "mist": 0.5}, build_metrics(y_true, np.argmax(probs, axis=1)))
    tiers = [
        (args.min_fog_precision, args.min_mist_precision, args.min_clear_recall),
        (max(0.05, args.min_fog_precision - 0.05), max(0.05, args.min_mist_precision - 0.05), max(0.84, args.min_clear_recall - 0.04)),
    ]
    for tier_id, (min_fp, min_mp, min_cr) in enumerate(tiers, start=1):
        found = False
        for fth in grid:
            for mth in grid:
                metrics = build_metrics(y_true, pred_from_thresholds(probs, float(fth), float(mth)))
                if metrics["Fog_P"] >= min_fp and metrics["Mist_P"] >= min_mp and metrics["Clear_R"] >= min_cr:
                    score = score_metrics(args, metrics) - 0.02 * (tier_id - 1)
                    if score > best[0]:
                        best = (score, {"fog": float(fth), "mist": float(mth)}, metrics)
                    found = True
        if found:
            return best
    fallback = build_metrics(y_true, np.argmax(probs, axis=1))
    return score_metrics(args, fallback) - 0.2, {"fog": 0.5, "mist": 0.5}, fallback


def gather_eval_arrays(
    probs: torch.Tensor,
    targets: torch.Tensor,
    world_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if world_size <= 1:
        return probs.cpu().numpy(), targets.cpu().numpy()
    local_n = torch.tensor([probs.shape[0]], dtype=torch.long, device=probs.device)
    max_n = local_n.clone()
    dist.all_reduce(max_n, op=dist.ReduceOp.MAX)
    pad_n = int(max_n.item() - local_n.item())
    if pad_n:
        probs = torch.cat([probs, torch.zeros((pad_n, probs.shape[1]), dtype=probs.dtype, device=probs.device)], dim=0)
        targets = torch.cat([targets, torch.full((pad_n,), -1, dtype=targets.dtype, device=targets.device)], dim=0)
    gp = [torch.zeros_like(probs) for _ in range(world_size)]
    gt = [torch.zeros_like(targets) for _ in range(world_size)]
    dist.all_gather(gp, probs)
    dist.all_gather(gt, targets)
    all_probs = torch.cat(gp, dim=0).cpu().numpy()
    all_targets = torch.cat(gt, dim=0).cpu().numpy()
    m = all_targets >= 0
    return all_probs[m], all_targets[m]


def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    n_actual: int,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    probs_l, targets_l = [], []
    with torch.no_grad():
        for bx, by, _, _ in loader:
            bx = bx.to(device, non_blocking=True)
            logits, _ = model(bx)
            probs_l.append(torch.softmax(logits, dim=1))
            targets_l.append(by.to(device))
    probs = torch.cat(probs_l, dim=0)
    targets = torch.cat(targets_l, dim=0)
    all_probs, all_targets = gather_eval_arrays(probs, targets, world_size)
    all_probs = all_probs[:n_actual]
    all_targets = all_targets[:n_actual]
    if rank == 0:
        score, th, metrics = threshold_search(args, all_probs, all_targets.astype(np.int64))
        return score, th, metrics
    return -1.0, {"fog": 0.5, "mist": 0.5}, {}


def wrap_ddp(model: nn.Module, local_rank: int, world_size: int, find_unused: bool = False) -> nn.Module:
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        safe_barrier(world_size, device)
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model: nn.Module, path: str, rank: int, metadata: Optional[Dict] = None) -> None:
    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": unwrap(model).state_dict(), "metadata": metadata or {}}, path)
    print(f"[Ckpt] saved {path}", flush=True)


def load_compatible_checkpoint(model: nn.Module, path: str, rank: int, device: torch.device) -> None:
    if not path:
        return
    if not os.path.exists(path):
        rank0(rank, f"[Ckpt] pretrained path not found, skip: {path}")
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    target = unwrap(model)
    own = target.state_dict()
    compatible = {}
    for k, v in state.items():
        kk = k.replace("module.", "")
        if kk in own and tuple(v.shape) == tuple(own[kk].shape):
            compatible[kk] = v
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    rank0(rank, f"[Ckpt] loaded {len(compatible)} tensors from {path}; missing={len(missing)} unexpected={len(unexpected)}")


def clone_state(model: nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(device) for k, v in unwrap(model).state_dict().items()}


def l2sp_penalty(model: nn.Module, ref: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
    if not ref:
        return next(unwrap(model).parameters()).new_tensor(0.0)
    total = None
    for name, param in unwrap(model).named_parameters():
        if param.requires_grad and name in ref and tuple(param.shape) == tuple(ref[name].shape):
            val = torch.sum((param - ref[name]) ** 2)
            total = val if total is None else total + val
    if total is None:
        return next(unwrap(model).parameters()).new_tensor(0.0)
    return total


def set_trainable(model: nn.Module, mode: str) -> None:
    raw = unwrap(model)
    for p in raw.parameters():
        p.requires_grad = mode == "all"
    if mode == "head":
        train_prefixes = ("fusion", "class_head", "reg_head", "fe_encoder", "dynamic_norm", "attn_pool")
        for name, p in raw.named_parameters():
            if name.split(".")[0] in train_prefixes:
                p.requires_grad = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, head_only: bool = False):
    raw = unwrap(model)
    if head_only:
        return [p for p in raw.parameters() if p.requires_grad]
    head_names = {"fusion", "class_head", "reg_head", "fe_encoder", "dynamic_norm", "attn_pool"}
    head, back = [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if name.split(".")[0] in head_names:
            head.append(p)
        else:
            back.append(p)
    groups = []
    if back:
        groups.append({"params": back, "lr": lr_backbone})
    if head:
        groups.append({"params": head, "lr": lr_head})
    return groups


def make_loaders(
    args: argparse.Namespace,
    train_ds: LowVisDataset,
    val_ds: LowVisDataset,
    fog_ratio: float,
    mist_ratio: float,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, StratifiedBalancedBatchSampler]:
    def worker_init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.X = None

    sampler = StratifiedBalancedBatchSampler(
        train_ds,
        args.batch_size,
        fog_ratio=fog_ratio,
        mist_ratio=mist_ratio,
        rank=rank,
        world_size=world_size,
        epoch_length=args.epoch_length,
    )
    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["timeout"] = 900
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, sampler


def train_stage(
    args: argparse.Namespace,
    tag: str,
    model: nn.Module,
    train_ds: LowVisDataset,
    val_ds: LowVisDataset,
    device: torch.device,
    rank: int,
    world_size: int,
    total_steps: int,
    fog_ratio: float,
    mist_ratio: float,
    lr_backbone: float,
    lr_head: Optional[float],
    trainable: str,
    l2sp_ref: Optional[Dict[str, torch.Tensor]],
    l2sp_alpha: float,
) -> str:
    if total_steps <= 0:
        return ""
    set_trainable(model, trainable)
    if lr_head is None or trainable == "head":
        groups = param_groups(model, lr_backbone, lr_backbone, head_only=(trainable == "head"))
    else:
        groups = param_groups(model, lr_backbone, lr_head, head_only=False)
    optimizer = optim.AdamW(groups, lr=lr_backbone, weight_decay=args.weight_decay)
    warm = min(args.warmup_steps, max(1, total_steps - 1))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warm)),
        ],
        milestones=[warm],
    )
    focal = WeightedFocalLoss(
        [args.class_weight_fog, args.class_weight_mist, args.class_weight_clear],
        [args.focal_gamma_fog, args.focal_gamma_mist, args.focal_gamma_clear],
    ).to(device)
    train_loader, val_loader, batch_sampler = make_loaders(args, train_ds, val_ds, fog_ratio, mist_ratio, rank, world_size)
    if world_size > 1:
        dist.all_reduce(torch.zeros(1, device=device), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    rank0(rank, f"[{tag}] start steps={total_steps} trainable={trainable} fog_ratio={fog_ratio} mist_ratio={mist_ratio}")

    ckpt_best = os.path.join(args.ckpt_dir, f"{args.run_id}_{tag}_best_score.pt")
    ckpt_latest = os.path.join(args.ckpt_dir, f"{args.run_id}_{tag}_latest.pt")
    history_path = os.path.join(args.ckpt_dir, f"{args.run_id}_{tag}_history.json")
    history = []
    best_score = -1e9
    no_improve = 0
    step = 0
    batch_count = 0
    epoch = 0
    iterator = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        try:
            bx, by, breg, braw = next(iterator)
        except StopIteration:
            epoch += 1
            batch_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            bx, by, breg, braw = next(iterator)
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)
        breg = breg.to(device, non_blocking=True)
        braw = braw.to(device, non_blocking=True)
        batch_count += 1
        is_sync = batch_count % args.grad_accum == 0
        ctx = model.no_sync() if world_size > 1 and not is_sync else contextlib.nullcontext()
        with ctx:
            logits, reg = model(bx)
            loss, loss_parts = combined_loss(args, focal, logits, reg, by, breg, braw)
            if l2sp_ref and l2sp_alpha > 0:
                loss = loss + l2sp_alpha * l2sp_penalty(model, l2sp_ref)
            (loss / args.grad_accum).backward()
        if not is_sync:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()
            scheduler.step()
        else:
            rank0(rank, f"[{tag}] nonfinite grad norm at step={step}: {grad_norm}")
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if rank == 0 and step % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[{tag}] step={step}/{total_steps} loss={float(loss):.4f} "
                f"cls={loss_parts['cls']:.4f} fp={loss_parts['fp']:.4f} "
                f"boost={loss_parts['boost']:.4f} lr={lr_now:.2e} no_improve={no_improve}/{args.patience}",
                flush=True,
            )

        if step % args.val_interval == 0 or step == total_steps:
            score, th, metrics = evaluate(args, model, val_loader, device, rank, world_size, len(val_ds))
            model.train()
            ckpt_meta = {
                "run_id": args.run_id,
                "tag": tag,
                "step": step,
                "score": score,
                "thresholds": th,
                "metrics": metrics,
                "selection_metric": args.selection_metric,
            }
            save_checkpoint(model, ckpt_latest, rank, ckpt_meta)
            if rank == 0:
                row = {"step": step, "score": score, "thresholds": th, **metrics}
                history.append(row)
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                print(
                    f"[{tag}] val score={score:.4f} th={th} "
                    f"Fog CSI/R/P={metrics.get('Fog_CSI', -1):.3f}/{metrics.get('Fog_R', -1):.3f}/{metrics.get('Fog_P', -1):.3f} "
                    f"Mist CSI/R/P={metrics.get('Mist_CSI', -1):.3f}/{metrics.get('Mist_R', -1):.3f}/{metrics.get('Mist_P', -1):.3f} "
                    f"LVPrec={metrics.get('low_vis_precision', -1):.3f} FPR={metrics.get('false_positive_rate', -1):.3f}",
                    flush=True,
                )
                if score > best_score:
                    best_score = score
                    no_improve = 0
                    save_checkpoint(model, ckpt_best, rank, ckpt_meta)
                else:
                    no_improve += 1
            if world_size > 1:
                t = torch.tensor([no_improve], dtype=torch.long, device=device)
                dist.broadcast(t, src=0)
                no_improve = int(t.item())
            if args.patience > 0 and no_improve >= args.patience:
                rank0(rank, f"[{tag}] early stop at step={step}, best_score={best_score:.4f}")
                break
    return ckpt_best


def build_model(args: argparse.Namespace, layout: Layout, use_fe: bool, device: torch.device) -> StaticRNNLowVisNet:
    return StaticRNNLowVisNet(
        layout=layout,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        static_hidden_dim=args.static_hidden_dim,
        fe_hidden_dim=args.fe_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        veg_emb_dim=args.veg_emb_dim,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        pooling=args.pooling,
        use_fe=use_fe,
    ).to(device)


def write_run_config(args: argparse.Namespace, rank: int) -> None:
    if rank != 0:
        return
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"{args.run_id}_static_rnn_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[Config] saved {path}", flush=True)


def main() -> None:
    args = parse_args()
    use_fe = not args.no_fe
    use_pm = not args.no_pm
    os.makedirs(args.ckpt_dir, exist_ok=True)

    local_rank, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    write_run_config(args, rank)

    model = None
    ddp_model = None
    s1_best = ""
    if args.mode in ("s1", "both"):
        tr, va, layout, _ = load_data(args, args.s1_data_dir, "s1", use_fe, use_pm, rank, local_rank, world_size, device)
        model = build_model(args, layout, use_fe, device)
        set_trainable(model, "all")
        ddp_model = wrap_ddp(model, local_rank, world_size, find_unused=False)
        rank0(rank, f"[Model:S1] params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
        s1_best = train_stage(
            args, "S1", ddp_model, tr, va, device, rank, world_size,
            args.s1_steps, args.fog_ratio_s1, args.mist_ratio_s1,
            args.s1_lr, None, "all", None, 0.0,
        )
        safe_barrier(world_size, device)

    if args.mode in ("s2", "both"):
        tr, va, layout, _ = load_data(args, args.s2_data_dir, "s2", use_fe, use_pm, rank, local_rank, world_size, device)
        model = build_model(args, layout, use_fe, device)
        pretrained = args.pretrained_ckpt or s1_best
        if pretrained:
            load_compatible_checkpoint(model, pretrained, rank, device)
        l2_ref = clone_state(model, device) if pretrained else None
        set_trainable(model, "head")
        ddp_model = wrap_ddp(model, local_rank, world_size, find_unused=True)
        rank0(rank, f"[Model:S2] params={sum(p.numel() for p in unwrap(ddp_model).parameters()) / 1e6:.3f}M pretrained={pretrained or 'none'}")

        phase_a_best = train_stage(
            args, "S2_PhaseA", ddp_model, tr, va, device, rank, world_size,
            args.s2_phase_a_steps, args.fog_ratio_s2, args.mist_ratio_s2,
            args.s2_lr_head_a, None, "head", l2_ref, args.l2sp_alpha_a,
        )
        safe_barrier(world_size, device)
        raw_model = unwrap(ddp_model)
        if world_size > 1:
            del ddp_model
            torch.cuda.empty_cache()
            safe_barrier(world_size, device)
        if phase_a_best:
            load_compatible_checkpoint(raw_model, phase_a_best, rank, device)
        set_trainable(raw_model, "all")
        ddp_model = wrap_ddp(raw_model, local_rank, world_size, find_unused=False)
        train_stage(
            args, "S2_PhaseB", ddp_model, tr, va, device, rank, world_size,
            args.s2_phase_b_steps, args.fog_ratio_s2, args.mist_ratio_s2,
            args.s2_lr_backbone_b, args.s2_lr_head_b, "all", l2_ref, args.l2sp_alpha_b,
        )

    safe_barrier(world_size, device)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    rank0(rank, "[Done] train_static_rnn_lowvis.py finished.")


if __name__ == "__main__":
    main()
