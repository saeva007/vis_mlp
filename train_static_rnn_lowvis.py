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
runtime feature-layout checks, optional PM ablation by masking PM channels when
they are present, log1p transforms for skewed dynamic variables, RobustScaler caching,
stratified low-visibility batch sampling, focal loss with class weights,
validation-time threshold search, gradient clipping, warmup+cosine LR, optional
L2-SP, and compatible Stage-1-to-Stage-2 checkpoint loading.

For manuscript loss-function ablations, the architecture and training protocol
can be held fixed while switching only the objective with ``--loss-mode``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from static_rnn_threshold_search import metrics_from_threshold_counts, threshold_prediction_counts

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
    dynamic_feature_order: Optional[List[str]] = None

    @property
    def split_dyn(self) -> int:
        return self.window_size * self.dyn_vars

    @property
    def core_dim(self) -> int:
        return self.split_dyn + 5

    @property
    def total_expected_dim(self) -> int:
        return self.split_dyn + 5 + 1 + self.fe_dim


@dataclass
class TimeGroupIndex:
    order: np.ndarray
    group_values: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    fog_counts: np.ndarray
    low_vis_counts: np.ndarray


@dataclass
class FootprintDualState:
    area: float
    recall: float


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


class EventTimeBatchSampler(Sampler[List[int]]):
    """Sample natural station sets within one valid time per batch."""

    def __init__(
        self,
        group_index: TimeGroupIndex,
        batch_size: int,
        min_fog_count: int,
        event_batch_ratio: float,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        epoch_length: int = 2000,
        full_group: bool = False,
    ) -> None:
        self.group_index = group_index
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch_length = int(epoch_length)
        self.epoch = 0
        self.event_batch_ratio = float(event_batch_ratio)
        self.full_group = bool(full_group)

        event = np.flatnonzero(group_index.fog_counts >= int(min_fog_count))
        background = np.flatnonzero(group_index.fog_counts < int(min_fog_count))
        self.event_groups = self._rank_shard(event)
        self.background_groups = self._rank_shard(background)
        self.all_groups = self._rank_shard(np.arange(len(group_index.starts), dtype=np.int64))
        if len(self.all_groups) == 0:
            raise ValueError("EventTimeBatchSampler received no valid-time groups")
        self.event_group_values = set(
            np.asarray(group_index.group_values[event], dtype=np.int64).tolist()
        )

    def _rank_shard(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if len(values) == 0:
            return values
        chunks = np.array_split(values, max(1, self.world_size))
        shard = chunks[self.rank % len(chunks)]
        return shard if len(shard) else values

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed + self.rank + 1543 * self.epoch)
        for _ in range(self.epoch_length):
            use_event = len(self.event_groups) > 0 and rng.random() < self.event_batch_ratio
            pool = self.event_groups if use_event else self.background_groups
            if len(pool) == 0:
                pool = self.all_groups
            group_pos = int(rng.choice(pool))
            start = int(self.group_index.starts[group_pos])
            count = int(self.group_index.counts[group_pos])
            rows = self.group_index.order[start : start + count]
            if self.full_group and count <= self.batch_size:
                chosen = rows
            else:
                chosen = rng.choice(rows, size=self.batch_size, replace=count < self.batch_size)
            yield np.asarray(chosen, dtype=np.int64).tolist()

    def __len__(self) -> int:
        return self.epoch_length

    def is_event_group(self, group_id: int) -> bool:
        return int(group_id) in self.event_group_values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static MLP + GRU/LSTM low-vis training")

    p.add_argument("--mode", choices=["s1", "s2", "both"], default="both")
    p.add_argument("--encoder", choices=["gru", "lstm"], default="gru")
    p.add_argument("--run-id", default=os.environ.get("LOWVIS_RNN_RUN_ID", f"exp_{int(time.time())}_static_rnn"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("LOWVIS_RNN_SEED", "42")))
    p.add_argument(
        "--local-cache-id",
        default=os.environ.get("LOWVIS_RNN_LOCAL_CACHE_ID", ""),
        help=(
            "Optional stable id for /tmp data copies. Use this to let matrix "
            "experiments share local cached npy files while keeping run-id "
            "separate for checkpoints."
        ),
    )
    p.add_argument("--base-path", default=DEFAULT_BASE)
    p.add_argument("--s1-data-dir", default=DEFAULT_S1_DIR)
    p.add_argument("--s2-data-dir", default=DEFAULT_S2_DIR)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--pretrained-ckpt", default="")
    p.add_argument(
        "--pretrained-layout-policy",
        choices=["strict", "compatible"],
        default=os.environ.get("LOWVIS_RNN_PRETRAINED_LAYOUT_POLICY", "strict"),
        help=(
            "strict refuses S1/S2 transfers with different dynamic or FE input "
            "layouts; compatible keeps the historical shape-matching partial load."
        ),
    )

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
    p.add_argument("--no-pm", action="store_true", help="Zero PM channels if the current data layout contains them.")
    p.add_argument("--aux-reg-weight", type=float, default=0.0)
    p.add_argument(
        "--loss-mode",
        choices=["designed_focal", "ce", "regression"],
        default=os.environ.get("LOWVIS_RNN_LOSS_MODE", "designed_focal"),
        help=(
            "Objective for loss-function ablations: designed_focal keeps the "
            "paper loss, ce is plain hard-label cross entropy, and regression "
            "uses only MSE on log1p(visibility)."
        ),
    )

    p.add_argument("--s1-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S1_STEPS", "15000")))
    p.add_argument("--s2-phase-a-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_A_STEPS", "8000")))
    p.add_argument("--s2-phase-b-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_B_STEPS", "22000")))
    p.add_argument("--s2-phase-c-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_C_STEPS", "0")))
    p.add_argument("--s2-phase-d-steps", type=int, default=int(os.environ.get("LOWVIS_RNN_S2_D_STEPS", "0")))
    p.add_argument("--val-interval", type=int, default=int(os.environ.get("LOWVIS_RNN_VAL_INTERVAL", "500")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("LOWVIS_RNN_BATCH_SIZE", "512")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("LOWVIS_RNN_GRAD_ACCUM", "2")))
    p.add_argument("--epoch-length", type=int, default=2000)
    p.add_argument(
        "--sampler-mode",
        choices=["stratified_balanced", "natural_shuffle"],
        default=os.environ.get("LOWVIS_RNN_SAMPLER_MODE", "stratified_balanced"),
        help=(
            "Training batch sampler. 'stratified_balanced' is the current "
            "low-vis oversampling protocol; 'natural_shuffle' preserves the "
            "dataset class distribution and only shuffles rows."
        ),
    )
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("LOWVIS_RNN_NUM_WORKERS", "0")))
    p.add_argument("--patience", type=int, default=int(os.environ.get("LOWVIS_RNN_PATIENCE", "10")))

    p.add_argument("--s1-lr", type=float, default=2e-4)
    p.add_argument("--s2-lr-head-a", type=float, default=8e-5)
    p.add_argument("--s2-lr-backbone-b", type=float, default=3e-6)
    p.add_argument("--s2-lr-head-b", type=float, default=1e-5)
    p.add_argument("--s2-lr-head-c", type=float, default=2e-5)
    p.add_argument("--s2-lr-head-d", type=float, default=2e-5)
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
    p.add_argument(
        "--event-fp-weight",
        type=float,
        default=None,
        help=(
            "Weight for the Low-vis event false-positive rate surrogate. Used "
            "when --event-loss-normalization=conditional; defaults to "
            "--alpha-clear-fp for backward-compatible configuration loading."
        ),
    )
    p.add_argument(
        "--event-fn-weight",
        type=float,
        default=None,
        help=(
            "Weight for the Low-vis event false-negative rate surrogate. Used "
            "when --event-loss-normalization=conditional; defaults to "
            "--alpha-recall-boost."
        ),
    )
    p.add_argument(
        "--event-loss-normalization",
        choices=["legacy_batch", "conditional"],
        default=os.environ.get("LOWVIS_RNN_EVENT_LOSS_NORMALIZATION", "legacy_batch"),
        help=(
            "legacy_batch preserves the historical batch-average clear-FP and "
            "class recall terms. conditional averages FP only over Clear and FN "
            "only over Low-vis samples, making their scale invariant to sampler ratios."
        ),
    )
    p.add_argument(
        "--clear-to-fog-weight",
        type=float,
        default=0.0,
        help="Conditional penalty on Ultra-low probability for high-visibility Clear samples.",
    )
    p.add_argument(
        "--clear-to-mist-weight",
        type=float,
        default=0.0,
        help="Conditional penalty on Moderate-low probability for high-visibility Clear samples.",
    )
    p.add_argument(
        "--clear-pair-vis-min",
        type=float,
        default=3000.0,
        help="Minimum observed visibility for pair-specific Clear false-alarm penalties.",
    )
    p.add_argument(
        "--moderate-fn-weight",
        type=float,
        default=0.0,
        help="Conditional Moderate-low false-negative guard used with pair-specific Clear penalties.",
    )
    p.add_argument("--label-smoothing", action="store_true", default=True)
    p.add_argument("--no-label-smoothing", dest="label_smoothing", action="store_false")
    p.add_argument("--soft-fog-mist-low", type=float, default=400.0)
    p.add_argument("--soft-fog-mist-high", type=float, default=600.0)
    p.add_argument("--soft-mist-clear-low", type=float, default=800.0)
    p.add_argument("--soft-mist-clear-high", type=float, default=1200.0)
    p.add_argument("--boundary-weight", type=float, default=0.0, help="Extra sample weight near 500 m and 1000 m boundaries.")
    p.add_argument("--boundary-fog-sigma", type=float, default=100.0, help="Width for the 500 m Fog/Mist boundary kernel.")
    p.add_argument("--boundary-mist-sigma", type=float, default=200.0, help="Width for the 1000 m Mist/Clear boundary kernel.")
    p.add_argument("--physical-hard-weight", type=float, default=0.0, help="Extra weight for high-humidity Clear-like hard negatives.")
    p.add_argument("--humid-rh-th", type=float, default=90.0)
    p.add_argument("--humid-dpd-th", type=float, default=2.0)
    p.add_argument("--humid-clear-vis-max", type=float, default=3000.0)
    p.add_argument("--aerosol-hard-weight", type=float, default=0.0, help="Extra weight for humid aerosol transition samples.")
    p.add_argument("--aerosol-rh-th", type=float, default=85.0)
    p.add_argument("--pm25-hard-th", type=float, default=75.0)
    p.add_argument("--pm10-hard-th", type=float, default=150.0)
    p.add_argument("--ordinal-cost-weight", type=float, default=0.0, help="Penalty on probability mass assigned far from the ordered target class.")
    p.add_argument("--sample-weight-cap", type=float, default=4.0)

    p.add_argument(
        "--phase-c-prior-beta",
        type=float,
        default=0.0,
        help=(
            "Tempered natural/sampler prior correction applied only to the Phase-C focal term. "
            "Zero preserves historical behavior."
        ),
    )
    p.add_argument("--event-footprint-csi-weight", type=float, default=0.0)
    p.add_argument("--event-footprint-area-ratio-cap", type=float, default=1.5)
    p.add_argument("--event-footprint-area-slack", type=float, default=0.005)
    p.add_argument("--event-footprint-min-recall", type=float, default=0.50)
    p.add_argument("--event-footprint-min-fog-count", type=int, default=40)
    p.add_argument("--event-footprint-event-batch-ratio", type=float, default=0.50)
    p.add_argument("--event-footprint-smoothmax-temperature", type=float, default=0.10)
    p.add_argument("--event-footprint-decision-temperature", type=float, default=0.15)
    p.add_argument("--event-footprint-dual-init", type=float, default=1.0)
    p.add_argument("--event-footprint-dual-rho", type=float, default=5.0)
    p.add_argument("--event-footprint-dual-lr", type=float, default=0.05)
    p.add_argument("--event-footprint-dual-max", type=float, default=20.0)

    p.add_argument(
        "--phase-c-selection-metric",
        choices=["recall_csi", "csi", "recall", "footprint_csi"],
        default="footprint_csi",
    )
    p.add_argument("--phase-c-min-low-vis-recall", type=float, default=0.55)
    p.add_argument("--phase-c-max-fpr", type=float, default=0.03)
    p.add_argument("--phase-c-min-event-mean-recall", type=float, default=0.55)
    p.add_argument("--phase-c-min-event-recall", type=float, default=0.40)
    p.add_argument("--phase-c-max-event-area-ratio-mean", type=float, default=1.80)
    p.add_argument("--phase-c-max-event-area-ratio", type=float, default=2.20)

    p.add_argument(
        "--phase-d-natural-mix",
        type=float,
        default=0.0,
        help=(
            "Final natural-snapshot loss fraction for Phase D. The complementary stream "
            "keeps the configured stratified low-visibility sampler."
        ),
    )
    p.add_argument(
        "--phase-d-ramp-start",
        type=float,
        default=0.50,
        help="Fraction of Phase D completed before cosine ramping the natural stream from zero.",
    )
    p.add_argument("--phase-d-event-batch-ratio", type=float, default=0.50)
    p.add_argument("--phase-d-min-fog-count", type=int, default=40)
    p.add_argument(
        "--phase-d-selection-metric",
        choices=["recall_csi", "csi", "recall", "sampling_csi"],
        default="sampling_csi",
    )
    p.add_argument("--phase-d-min-low-vis-csi", type=float, default=0.195)
    p.add_argument("--phase-d-max-fpr", type=float, default=0.025)
    p.add_argument("--phase-d-min-event-mean-csi", type=float, default=0.235)
    p.add_argument("--phase-d-min-event-mean-recall", type=float, default=0.45)
    p.add_argument("--phase-d-min-event-recall", type=float, default=0.20)
    p.add_argument("--phase-d-min-event-area-ratio-mean", type=float, default=0.80)
    p.add_argument("--phase-d-max-event-area-ratio-mean", type=float, default=1.80)
    p.add_argument("--phase-d-max-event-area-ratio", type=float, default=2.20)

    p.add_argument("--selection-metric", choices=["recall_csi", "csi", "recall", "footprint_csi", "sampling_csi"], default="recall_csi")
    p.add_argument(
        "--threshold-mode",
        choices=["val_search", "argmax"],
        default="val_search",
        help="Decision rule used during validation and best-checkpoint selection.",
    )
    p.add_argument("--min-fog-precision", type=float, default=0.10)
    p.add_argument("--min-mist-precision", type=float, default=0.10)
    p.add_argument("--min-clear-recall", type=float, default=0.88)
    p.add_argument("--threshold-grid-low", type=float, default=0.10)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--threshold-grid-step", type=float, default=0.03)
    args = p.parse_args()
    if args.soft_fog_mist_high <= args.soft_fog_mist_low:
        p.error("--soft-fog-mist-high must be greater than --soft-fog-mist-low")
    if args.soft_mist_clear_high <= args.soft_mist_clear_low:
        p.error("--soft-mist-clear-high must be greater than --soft-mist-clear-low")
    if args.event_fp_weight is not None and args.event_fp_weight < 0:
        p.error("--event-fp-weight must be non-negative")
    if args.event_fn_weight is not None and args.event_fn_weight < 0:
        p.error("--event-fn-weight must be non-negative")
    if args.clear_to_fog_weight < 0:
        p.error("--clear-to-fog-weight must be non-negative")
    if args.clear_to_mist_weight < 0:
        p.error("--clear-to-mist-weight must be non-negative")
    if args.clear_pair_vis_min < 1000:
        p.error("--clear-pair-vis-min must be at least 1000 m")
    if args.moderate_fn_weight < 0:
        p.error("--moderate-fn-weight must be non-negative")
    if not 0.0 <= args.phase_c_prior_beta <= 1.0:
        p.error("--phase-c-prior-beta must be within [0, 1]")
    if args.s2_phase_c_steps < 0:
        p.error("--s2-phase-c-steps must be non-negative")
    if args.s2_phase_d_steps < 0:
        p.error("--s2-phase-d-steps must be non-negative")
    if args.event_footprint_csi_weight < 0:
        p.error("--event-footprint-csi-weight must be non-negative")
    if args.event_footprint_area_ratio_cap <= 0:
        p.error("--event-footprint-area-ratio-cap must be positive")
    if args.event_footprint_area_slack < 0:
        p.error("--event-footprint-area-slack must be non-negative")
    if not 0.0 <= args.event_footprint_min_recall <= 1.0:
        p.error("--event-footprint-min-recall must be within [0, 1]")
    if not 0.0 <= args.event_footprint_event_batch_ratio <= 1.0:
        p.error("--event-footprint-event-batch-ratio must be within [0, 1]")
    if args.event_footprint_smoothmax_temperature <= 0 or args.event_footprint_decision_temperature <= 0:
        p.error("event-footprint temperatures must be positive")
    if args.event_footprint_dual_init < 0 or args.event_footprint_dual_rho < 0:
        p.error("event-footprint dual init/rho must be non-negative")
    if args.event_footprint_dual_lr < 0 or args.event_footprint_dual_max <= 0:
        p.error("event-footprint dual lr/max are invalid")
    if not 0.0 <= args.phase_d_natural_mix <= 1.0:
        p.error("--phase-d-natural-mix must be within [0, 1]")
    if args.s2_phase_d_steps > 0 and args.phase_d_natural_mix <= 0:
        p.error("Phase D requires --phase-d-natural-mix > 0")
    if args.s2_phase_d_steps > 0 and args.sampler_mode != "stratified_balanced":
        p.error("Phase D requires --sampler-mode stratified_balanced for its recall-preserving stream")
    if not 0.0 <= args.phase_d_ramp_start < 1.0:
        p.error("--phase-d-ramp-start must be within [0, 1)")
    if not 0.0 <= args.phase_d_event_batch_ratio <= 1.0:
        p.error("--phase-d-event-batch-ratio must be within [0, 1]")
    if args.phase_d_min_fog_count < 1:
        p.error("--phase-d-min-fog-count must be positive")
    if args.phase_d_min_event_area_ratio_mean < 0:
        p.error("--phase-d-min-event-area-ratio-mean must be non-negative")
    if args.phase_d_max_event_area_ratio_mean <= args.phase_d_min_event_area_ratio_mean:
        p.error("Phase-D maximum mean event area ratio must exceed its minimum")
    return args


def rank0(rank: int, text: str) -> None:
    if rank == 0:
        print(text, flush=True)


COMPACT_COMMON_CORE_INDEX = {
    "T2M": 0,
    "MSLP": 1,
    "U10": 2,
    "WSPD10": 3,
    "V10": 4,
    "WDIR10": 5,
    "RH_925": 6,
    "U_925": 7,
    "WSPD925": 8,
    "V_925": 9,
    "DP_1000": 10,
    "DP_925": 11,
    "Q_1000": 12,
    "Q_925": 13,
    "DPD": 14,
    "ZENITH": 15,
    "PM10": 16,
    "PM25": 17,
}
COMPACT_COMMON_CORE_WITH_RH_INDEX = {
    "RH2M": 0,
    "T2M": 1,
    "MSLP": 2,
    "U10": 3,
    "WSPD10": 4,
    "V10": 5,
    "WDIR10": 6,
    "RH_925": 7,
    "U_925": 8,
    "WSPD925": 9,
    "V_925": 10,
    "DP_1000": 11,
    "DP_925": 12,
    "Q_1000": 13,
    "Q_925": 14,
    "DPD": 15,
    "ZENITH": 16,
    "PM10": 17,
    "PM25": 18,
}
PMST27_INDEX = {
    "RH2M": 0,
    "T2M": 1,
    "WSPD10": 6,
    "DPD": 22,
    "PM10": 25,
    "PM25": 26,
}


def normalize_feature_name(name: str) -> str:
    raw = str(name).strip().upper().replace("-", "_").replace(" ", "_")
    compact = raw.replace("_", "")
    aliases = {
        "PM10UGM3": "PM10",
        "PM10UG_M3": "PM10",
        "PM25UGM3": "PM25",
        "PM25UG_M3": "PM25",
        "PM2P5": "PM25",
    }
    return aliases.get(compact, raw)


def dyn_index(layout_or_dyn, name: str) -> Optional[int]:
    if isinstance(layout_or_dyn, Layout) and layout_or_dyn.dynamic_feature_order:
        target = normalize_feature_name(name)
        for i, feat in enumerate(layout_or_dyn.dynamic_feature_order):
            if normalize_feature_name(feat) == target:
                return i
        return None
    dyn_vars = layout_or_dyn.dyn_vars if isinstance(layout_or_dyn, Layout) else int(layout_or_dyn)
    if dyn_vars == 18:
        return COMPACT_COMMON_CORE_INDEX.get(name)
    if dyn_vars == 19:
        return COMPACT_COMMON_CORE_WITH_RH_INDEX.get(name)
    if name == "PM10":
        return 25 if dyn_vars >= 26 else None
    if name == "PM25":
        return 26 if dyn_vars >= 27 else None
    idx = PMST27_INDEX.get(name)
    return idx if idx is not None and idx < dyn_vars else None


def resolve_dyn_and_fe_dims(total_dim: int, win_size: int) -> Tuple[int, int]:
    rest = int(total_dim) - 6
    if rest <= 0:
        raise ValueError(f"total_dim={total_dim} too small for dyn+static+veg+FE layout")
    for dyn in (27, 26, 25, 24, 19, 18):
        fe = rest - dyn * int(win_size)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot resolve feature layout: total_dim={total_dim}, window={win_size}")


def load_dynamic_feature_order(data_dir: str) -> Tuple[Optional[int], Optional[List[str]]]:
    cfg: Dict[str, object] = {}
    for name in ("dataset_build_config.json", "dataset_metadata.json"):
        cfg_path = os.path.join(data_dir, name)
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            break
        except Exception:
            continue
    if not cfg:
        return None, None
    dyn_vars = cfg.get("dyn_vars", cfg.get("dyn_vars_count"))
    order = cfg.get("dynamic_feature_order")
    if isinstance(order, list):
        order = [str(v) for v in order]
    else:
        order = None
    return (int(dyn_vars) if dyn_vars is not None else None), order


def resolve_layout_from_file(path_x: str, win_size: int, data_dir: str = "") -> Layout:
    shape = np.load(path_x, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{path_x} must be 2D, got shape={shape}")
    cfg_dyn, order = load_dynamic_feature_order(data_dir) if data_dir else (None, None)
    if cfg_dyn is not None:
        rest = int(shape[1]) - 6
        fe = rest - cfg_dyn * int(win_size)
        if fe < 0:
            raise ValueError(f"{path_x}: config dyn_vars={cfg_dyn} incompatible with row_dim={shape[1]}")
        if order is not None and len(order) != cfg_dyn:
            raise ValueError(f"{path_x}: dynamic_feature_order length {len(order)} != dyn_vars {cfg_dyn}")
        return Layout(window_size=win_size, dyn_vars=cfg_dyn, fe_dim=fe, dynamic_feature_order=order)
    dyn, fe = resolve_dyn_and_fe_dims(int(shape[1]), win_size)
    return Layout(window_size=win_size, dyn_vars=dyn, fe_dim=fe)


def pm_indices(layout_or_dyn) -> List[int]:
    if isinstance(layout_or_dyn, Layout) and layout_or_dyn.dynamic_feature_order:
        return [i for i, name in enumerate(layout_or_dyn.dynamic_feature_order) if normalize_feature_name(name) in {"PM10", "PM25"}]
    dyn_vars = layout_or_dyn.dyn_vars if isinstance(layout_or_dyn, Layout) else int(layout_or_dyn)
    if dyn_vars == 18:
        return [16, 17]
    if dyn_vars == 19:
        return [17, 18]
    if dyn_vars >= 27:
        return [dyn_vars - 2, dyn_vars - 1]
    if dyn_vars >= 25:
        return [dyn_vars - 1]
    return []


def log1p_dyn_indices(layout_or_dyn) -> List[int]:
    if isinstance(layout_or_dyn, Layout) and layout_or_dyn.dynamic_feature_order:
        log_names = {"PRECIP", "SW_RAD", "CAPE", "PM10", "PM25"}
        return [i for i, name in enumerate(layout_or_dyn.dynamic_feature_order) if normalize_feature_name(name) in log_names]
    dyn_vars = layout_or_dyn.dyn_vars if isinstance(layout_or_dyn, Layout) else int(layout_or_dyn)
    if dyn_vars == 18:
        return pm_indices(dyn_vars)
    if dyn_vars == 19:
        return pm_indices(dyn_vars)
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
        for idx in log1p_dyn_indices(layout):
            mask[t * layout.dyn_vars + idx] = True
    return mask


def apply_core_transform(core: np.ndarray, layout: Layout, use_pm: bool, log_mask: np.ndarray) -> np.ndarray:
    out = core.astype(np.float32, copy=True)
    if not use_pm:
        for t in range(layout.window_size):
            for idx in pm_indices(layout):
                out[:, t * layout.dyn_vars + idx] = 0.0
    dyn = out[:, : layout.split_dyn]
    dyn[:] = np.where(log_mask, np.log1p(np.maximum(dyn, 0.0)), dyn)
    return out


def boundary_weight_from_visibility(y_raw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    weights = np.ones(len(y_raw), dtype=np.float32)
    if args.boundary_weight <= 0:
        return weights
    vis = np.asarray(y_raw, dtype=np.float32)
    fog_sigma = max(float(args.boundary_fog_sigma), 1.0)
    mist_sigma = max(float(args.boundary_mist_sigma), 1.0)
    fog_mist = np.exp(-np.abs(vis - 500.0) / fog_sigma)
    mist_clear = np.exp(-np.abs(vis - 1000.0) / mist_sigma)
    boundary = np.maximum(fog_mist, mist_clear)
    weights += float(args.boundary_weight) * boundary.astype(np.float32)
    cap = max(float(args.sample_weight_cap), 1.0)
    return np.clip(weights, 0.0, cap).astype(np.float32)


def scaler_cache_path(args: argparse.Namespace, stage: str, layout: Layout, use_pm: bool) -> str:
    pm_tag = "pm" if use_pm else "nopm"
    name = f"robust_scaler_{args.run_id}_{stage}_w{layout.window_size}_dyn{layout.dyn_vars}_{pm_tag}.pkl"
    return os.path.join(args.ckpt_dir, name)


def event_footprint_enabled(args: argparse.Namespace) -> bool:
    return bool(args.s2_phase_c_steps > 0 and args.event_footprint_csi_weight > 0)


def sampling_calibration_enabled(args: argparse.Namespace) -> bool:
    return bool(args.s2_phase_d_steps > 0 and args.phase_d_natural_mix > 0)


def sampling_calibration_mix(
    step: int,
    total_steps: int,
    ramp_start: float,
    final_mix: float,
) -> float:
    if total_steps <= 0 or final_mix <= 0:
        return 0.0
    progress = min(max(float(step) / float(max(total_steps - 1, 1)), 0.0), 1.0)
    if progress <= float(ramp_start):
        return 0.0
    ramp = (progress - float(ramp_start)) / max(1.0 - float(ramp_start), 1e-6)
    return float(final_mix) * 0.5 * (1.0 - math.cos(math.pi * ramp))


def build_time_group_index(group_ids: np.ndarray, y_cls: np.ndarray) -> TimeGroupIndex:
    groups = np.asarray(group_ids, dtype=np.int64)
    labels = np.asarray(y_cls, dtype=np.int64)
    if groups.ndim != 1 or labels.ndim != 1 or len(groups) != len(labels):
        raise ValueError("group_ids and y_cls must be aligned one-dimensional arrays")
    if len(groups) == 0:
        raise ValueError("cannot build an event-time index for an empty dataset")
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1].astype(np.int64)
    counts = np.diff(np.r_[starts, len(order)]).astype(np.int64)
    sorted_labels = labels[order]
    fog_counts = np.add.reduceat((sorted_labels == 0).astype(np.int64), starts)
    low_vis_counts = np.add.reduceat((sorted_labels <= 1).astype(np.int64), starts)
    return TimeGroupIndex(
        order=order.astype(np.int64, copy=False),
        group_values=sorted_groups[starts].astype(np.int64, copy=False),
        starts=starts,
        counts=counts,
        fog_counts=fog_counts.astype(np.int64, copy=False),
        low_vis_counts=low_vis_counts.astype(np.int64, copy=False),
    )


def ensure_time_group_ids(
    args: argparse.Namespace,
    data_dir: str,
    stage: str,
    split: str,
    n_rows: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> np.ndarray:
    meta_path = os.path.join(data_dir, f"meta_{split}.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Phase-C event footprint requires {meta_path}")
    cache_path = os.path.join(args.ckpt_dir, f"{args.run_id}_{stage}_{split}_time_group_ids.npy")
    safe_barrier(world_size, device)
    if rank == 0 and not os.path.isfile(cache_path):
        tmp_path = cache_path + ".tmp.npy"
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.int64, shape=(int(n_rows),))
        offset = 0
        for chunk in pd.read_csv(meta_path, usecols=["time"], chunksize=1_000_000):
            times = pd.to_datetime(chunk["time"], errors="coerce", utc=True)
            values = times.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
            if np.any(values == np.iinfo(np.int64).min):
                raise ValueError(f"Unparseable time values in {meta_path}")
            end = offset + len(values)
            if end > n_rows:
                raise ValueError(f"{meta_path} has more than the expected {n_rows} rows")
            out[offset:end] = values
            offset = end
        out.flush()
        del out
        if offset != n_rows:
            raise ValueError(f"{meta_path} has {offset} rows, expected {n_rows}")
        os.replace(tmp_path, cache_path)
        print(f"[EventFootprint] cached time groups: {cache_path}", flush=True)
    safe_barrier(world_size, device)
    groups = np.load(cache_path, mmap_mode="r")
    if len(groups) != n_rows:
        raise ValueError(f"time-group cache length mismatch: {len(groups)} != {n_rows}")
    return groups


def ensure_train_group_index(
    args: argparse.Namespace,
    stage: str,
    group_ids: np.ndarray,
    y_cls: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> TimeGroupIndex:
    prefix = os.path.join(args.ckpt_dir, f"{args.run_id}_{stage}_train_time_index")
    paths = {
        "order": prefix + "_order.npy",
        "group_values": prefix + "_group_values.npy",
        "starts": prefix + "_starts.npy",
        "counts": prefix + "_counts.npy",
        "fog_counts": prefix + "_fog_counts.npy",
        "low_vis_counts": prefix + "_low_vis_counts.npy",
    }
    safe_barrier(world_size, device)
    if rank == 0 and not all(os.path.isfile(path) for path in paths.values()):
        index = build_time_group_index(group_ids, y_cls)
        for name, path in paths.items():
            tmp_path = path + ".tmp.npy"
            np.save(tmp_path, getattr(index, name))
            os.replace(tmp_path, path)
        print(f"[EventFootprint] cached train time index: {prefix}", flush=True)
    safe_barrier(world_size, device)
    return TimeGroupIndex(**{name: np.load(path, mmap_mode="r") for name, path in paths.items()})


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
        args: argparse.Namespace,
        time_group_ids: Optional[np.ndarray] = None,
        time_group_index: Optional[TimeGroupIndex] = None,
    ) -> None:
        self.x_path = x_path
        self.layout = layout
        self.scaler = scaler
        self.use_fe = bool(use_fe)
        self.use_pm = bool(use_pm)
        self.args = args
        self.time_group_ids = time_group_ids
        self.time_group_index = time_group_index
        self.y_raw = torch.as_tensor(np.maximum(y_raw, 0.0), dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.log1p(self.y_raw)
        self.base_weights = boundary_weight_from_visibility(np.maximum(y_raw, 0.0), args)
        self.log_mask = build_dyn_log_mask(layout)
        self.X = None

    def __len__(self) -> int:
        return len(self.y_cls)

    def _physical_weight(self, row: np.ndarray, idx: int) -> float:
        weight = float(self.base_weights[idx])
        args = self.args
        cls = int(self.y_cls[idx])
        vis = float(self.y_raw[idx])
        last = (self.layout.window_size - 1) * self.layout.dyn_vars
        rh_i = dyn_index(self.layout, "RH2M")
        wspd_i = dyn_index(self.layout, "WSPD10")
        dpd_i = dyn_index(self.layout, "DPD")
        pm10_i = dyn_index(self.layout, "PM10")
        pm25_i = dyn_index(self.layout, "PM25")
        rh = float(row[last + rh_i]) if rh_i is not None else math.nan
        wspd = float(row[last + wspd_i]) if wspd_i is not None else math.nan
        dpd = float(row[last + dpd_i]) if dpd_i is not None else math.nan
        pm10 = float(row[last + pm10_i]) if pm10_i is not None else math.nan
        pm25 = float(row[last + pm25_i]) if pm25_i is not None else math.nan

        if args.physical_hard_weight > 0 and cls == 2:
            humid_clear = (
                (math.isfinite(rh) and rh >= args.humid_rh_th)
                or (math.isfinite(dpd) and dpd <= args.humid_dpd_th)
                or (1000.0 <= vis < args.humid_clear_vis_max)
            )
            ventilated_humid = (
                math.isfinite(rh)
                and math.isfinite(wspd)
                and rh >= max(args.humid_rh_th - 5.0, 0.0)
                and wspd >= 4.0
            )
            if humid_clear or ventilated_humid:
                weight += float(args.physical_hard_weight)

        if args.aerosol_hard_weight > 0:
            humid = math.isfinite(rh) and rh >= args.aerosol_rh_th
            aerosol = (
                (math.isfinite(pm25) and pm25 >= args.pm25_hard_th)
                or (math.isfinite(pm10) and pm10 >= args.pm10_hard_th)
            )
            if humid and aerosol and vis < args.humid_clear_vis_max:
                weight += float(args.aerosol_hard_weight)

        cap = max(float(args.sample_weight_cap), 1.0)
        return float(np.clip(weight, 0.0, cap))

    def __getitem__(self, idx: int):
        idx = int(idx)
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
        sample_weight = np.asarray(self._physical_weight(row, idx), dtype=np.float32)
        group_id = -1 if self.time_group_ids is None else int(self.time_group_ids[idx])
        return (
            torch.from_numpy(final).float(),
            self.y_cls[idx],
            self.y_reg[idx],
            self.y_raw[idx],
            torch.from_numpy(sample_weight),
            torch.tensor(group_id, dtype=torch.long),
        )


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
    local_cache_id = args.local_cache_id.strip() or args.run_id
    x_tr, y_tr, x_va, y_va = load_split_paths(data_dir, stage, rank, local_rank, world_size, local_cache_id)
    layout = resolve_layout_from_file(x_tr, args.window_size, data_dir)
    va_layout = resolve_layout_from_file(x_va, args.window_size, data_dir)
    if asdict(layout) != asdict(va_layout):
        raise ValueError(f"train/val layout mismatch: {layout} vs {va_layout}")
    y_raw_tr, y_cls_tr = visibility_to_labels(np.load(y_tr))
    y_raw_va, y_cls_va = visibility_to_labels(np.load(y_va))
    if len(y_raw_tr) != np.load(x_tr, mmap_mode="r").shape[0]:
        raise ValueError("train X/y length mismatch")
    if len(y_raw_va) != np.load(x_va, mmap_mode="r").shape[0]:
        raise ValueError("val X/y length mismatch")
    scaler = fit_or_load_scaler(args, stage, x_tr, layout, use_pm, rank, world_size, device)
    train_groups = None
    val_groups = None
    train_group_index = None
    if stage == "s2" and (event_footprint_enabled(args) or sampling_calibration_enabled(args)):
        train_groups = ensure_time_group_ids(
            args, data_dir, stage, "train", len(y_cls_tr), rank, world_size, device
        )
        val_groups = ensure_time_group_ids(
            args, data_dir, stage, "val", len(y_cls_va), rank, world_size, device
        )
        train_group_index = ensure_train_group_index(
            args, stage, train_groups, y_cls_tr, rank, world_size, device
        )
    tr_ds = LowVisDataset(
        x_tr,
        y_raw_tr,
        y_cls_tr,
        layout,
        scaler,
        use_fe,
        use_pm,
        args,
        time_group_ids=train_groups,
        time_group_index=train_group_index,
    )
    va_ds = LowVisDataset(
        x_va,
        y_raw_va,
        y_cls_va,
        layout,
        scaler,
        use_fe,
        use_pm,
        args,
        time_group_ids=val_groups,
    )
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        ordinal_cost_weight: float = 0.0,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1).clamp(1e-7, 1.0 - 1e-7)
        if soft_targets is None:
            soft_targets = F.one_hot(targets, 3).float()
        focal = (1.0 - probs) ** self.gamma.unsqueeze(0)
        weight = self.class_weights.unsqueeze(0)
        if ordinal_cost_weight > 0:
            cls_ids = torch.arange(logits.size(1), device=logits.device, dtype=logits.dtype).unsqueeze(0)
            dist = torch.abs(cls_ids - targets.float().unsqueeze(1))
            weight = weight * (1.0 + float(ordinal_cost_weight) * dist)
        loss = -(soft_targets * weight * focal * torch.log(probs)).sum(dim=1)
        if sample_weight is not None:
            sw = sample_weight.to(loss.device, dtype=loss.dtype).clamp_min(0.0)
            sw = sw / sw.mean().clamp_min(1e-6)
            loss = loss * sw
        return loss.mean()


def soft_targets_from_visibility(
    raw: torch.Tensor,
    labels: torch.Tensor,
    fog_mist_low: float = 400.0,
    fog_mist_high: float = 600.0,
    mist_clear_low: float = 800.0,
    mist_clear_high: float = 1200.0,
) -> torch.Tensor:
    soft = F.one_hot(labels, 3).float()
    fm_width = max(float(fog_mist_high) - float(fog_mist_low), 1e-6)
    mc_width = max(float(mist_clear_high) - float(mist_clear_low), 1e-6)
    fm = (raw >= float(fog_mist_low)) & (raw < float(fog_mist_high))
    if fm.any():
        alpha = (raw[fm] - float(fog_mist_low)) / fm_width
        soft[fm, 0] = 1.0 - alpha
        soft[fm, 1] = alpha
        soft[fm, 2] = 0.0
    mc = (raw >= float(mist_clear_low)) & (raw < float(mist_clear_high))
    if mc.any():
        alpha = (raw[mc] - float(mist_clear_low)) / mc_width
        soft[mc, 0] = 0.0
        soft[mc, 1] = 1.0 - alpha
        soft[mc, 2] = alpha
    return soft


def weighted_mean(value: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_weight is None:
        return torch.mean(value)
    sw = sample_weight.to(value.device, dtype=value.dtype).clamp_min(0.0)
    sw = sw / sw.mean().clamp_min(1e-6)
    return torch.mean(value * sw)


def conditional_weighted_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Average ``value`` within ``mask`` without sampler-prevalence scaling."""
    m = mask.to(value.device, dtype=value.dtype).clamp_min(0.0)
    if sample_weight is None:
        weights = m
    else:
        weights = sample_weight.to(value.device, dtype=value.dtype).clamp_min(0.0) * m
    denom = weights.sum()
    if float(denom.detach()) <= 0.0:
        return value.sum() * 0.0
    return torch.sum(value * weights) / denom.clamp_min(1e-6)


def class_prior_correction_weights(
    y_cls: np.ndarray,
    sampler_prior: np.ndarray,
    beta: float,
) -> np.ndarray:
    beta = float(beta)
    if beta <= 0:
        return np.ones(3, dtype=np.float32)
    counts = np.bincount(np.asarray(y_cls, dtype=np.int64), minlength=3).astype(np.float64)
    natural_prior = counts / max(float(counts.sum()), 1.0)
    sampler = np.asarray(sampler_prior, dtype=np.float64)
    if sampler.shape != (3,) or np.any(sampler <= 0):
        raise ValueError(f"sampler_prior must contain three positive values, got {sampler}")
    weights = np.power(natural_prior / sampler, beta)
    return weights.astype(np.float32)


def event_footprint_loss(
    args: argparse.Namespace,
    logits: torch.Tensor,
    labels: torch.Tensor,
    dual_state: FootprintDualState,
    is_widespread_event: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    smooth_t = float(args.event_footprint_smoothmax_temperature)
    decision_t = float(args.event_footprint_decision_temperature)
    low_logit = smooth_t * torch.logsumexp(logits[:, :2] / smooth_t, dim=1)
    soft_low = torch.sigmoid((low_logit - logits[:, 2]) / decision_t)
    low_true = (labels <= 1).to(dtype=logits.dtype)
    obs_area = low_true.mean()
    pred_area = soft_low.mean()
    tp = torch.mean(soft_low * low_true)
    fp = torch.mean(soft_low * (1.0 - low_true))
    fn = torch.mean((1.0 - soft_low) * low_true)
    soft_csi = tp / (tp + fp + fn).clamp_min(1e-6)
    soft_recall = tp / obs_area.clamp_min(1e-6)
    has_low_vis = (torch.sum(low_true) > 0).to(dtype=logits.dtype)
    if is_widespread_event is None:
        is_widespread = (torch.sum(labels == 0) >= int(args.event_footprint_min_fog_count)).to(dtype=logits.dtype)
    else:
        is_widespread = logits.new_tensor(float(bool(is_widespread_event)))

    area_violation = F.relu(
        pred_area
        - float(args.event_footprint_area_ratio_cap) * obs_area
        - float(args.event_footprint_area_slack)
    )
    recall_violation = F.relu(float(args.event_footprint_min_recall) - soft_recall) * is_widespread
    rho = float(args.event_footprint_dual_rho)
    loss = (
        float(args.event_footprint_csi_weight) * (1.0 - soft_csi) * has_low_vis
        + float(dual_state.area) * area_violation
        + 0.5 * rho * area_violation.square()
        + float(dual_state.recall) * recall_violation
        + 0.5 * rho * recall_violation.square()
    )
    return loss, {
        "footprint": loss,
        "soft_csi": soft_csi,
        "soft_recall": soft_recall,
        "pred_area": pred_area,
        "obs_area": obs_area,
        "area_violation": area_violation,
        "recall_violation": recall_violation,
        "is_widespread": is_widespread,
    }


def update_footprint_duals(
    args: argparse.Namespace,
    state: FootprintDualState,
    area_violation: torch.Tensor,
    recall_violation: torch.Tensor,
    world_size: int,
) -> None:
    violations = torch.stack([area_violation.detach(), recall_violation.detach()]).to(dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(violations, op=dist.ReduceOp.SUM)
        violations /= float(world_size)
    lr = float(args.event_footprint_dual_lr)
    upper = float(args.event_footprint_dual_max)
    state.area = float(np.clip(state.area + lr * float(violations[0].item()), 0.0, upper))
    state.recall = float(np.clip(state.recall + lr * float(violations[1].item()), 0.0, upper))


def combined_loss(
    args: argparse.Namespace,
    focal: WeightedFocalLoss,
    logits: torch.Tensor,
    reg: torch.Tensor,
    y: torch.Tensor,
    y_reg: torch.Tensor,
    y_raw: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
    focal_prior_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_mode = getattr(args, "loss_mode", "designed_focal")

    if loss_mode == "ce":
        l_cls = F.cross_entropy(logits, y)
        l_reg = reg.sum() * 0.0
        total = l_cls + l_reg
        zero = float(l_reg.detach())
        return total, {
            "cls": float(l_cls.detach()),
            "fp": zero,
            "boost": zero,
            "clear_fog": zero,
            "clear_mist": zero,
            "mist_guard": zero,
            "ord": zero,
            "reg": zero,
        }

    if loss_mode == "regression":
        l_reg = torch.mean((reg - y_reg) ** 2)
        l_cls = logits.sum() * 0.0
        total = l_reg + l_cls
        zero = float(l_cls.detach())
        return total, {
            "cls": zero,
            "fp": zero,
            "boost": zero,
            "clear_fog": zero,
            "clear_mist": zero,
            "mist_guard": zero,
            "ord": zero,
            "reg": float(l_reg.detach()),
        }

    soft = (
        soft_targets_from_visibility(
            y_raw,
            y,
            args.soft_fog_mist_low,
            args.soft_fog_mist_high,
            args.soft_mist_clear_low,
            args.soft_mist_clear_high,
        )
        if args.label_smoothing
        else None
    )
    focal_sample_weight = sample_weight
    if focal_prior_weights is not None:
        prior = focal_prior_weights.to(logits.device, dtype=logits.dtype)[y]
        focal_sample_weight = prior if sample_weight is None else sample_weight * prior
    l_cls = focal(logits, y, soft, focal_sample_weight, 0.0)
    probs = torch.softmax(logits, dim=1)
    clear = (y == 2).float()
    fog = (y == 0).float()
    mist = (y == 1).float()
    p_low = torch.clamp(probs[:, 0] + probs[:, 1], 0.0, 1.0)
    event_fp_weight = float(args.alpha_clear_fp if args.event_fp_weight is None else args.event_fp_weight)
    event_fn_weight = float(args.alpha_recall_boost if args.event_fn_weight is None else args.event_fn_weight)
    if args.event_loss_normalization == "conditional":
        low_vis = fog + mist
        l_fp = conditional_weighted_mean(p_low ** 2, clear, sample_weight)
        l_boost = conditional_weighted_mean((1.0 - p_low) ** 2, low_vis, sample_weight)
        fp_weight = event_fp_weight
        boost_weight = event_fn_weight
    else:
        l_fp = weighted_mean((p_low ** 2) * clear, sample_weight)
        l_boost = weighted_mean(((1.0 - probs[:, 0]) ** 2) * fog, sample_weight) + weighted_mean(
            ((1.0 - probs[:, 1]) ** 2) * mist,
            sample_weight,
        )
        fp_weight = float(args.alpha_clear_fp)
        boost_weight = float(args.alpha_recall_boost)
    high_vis_clear = clear * (y_raw >= float(args.clear_pair_vis_min)).to(clear.dtype)
    l_clear_fog = conditional_weighted_mean(probs[:, 0] ** 2, high_vis_clear, sample_weight)
    l_clear_mist = conditional_weighted_mean(probs[:, 1] ** 2, high_vis_clear, sample_weight)
    l_mist_guard = conditional_weighted_mean((1.0 - probs[:, 1]) ** 2, mist, sample_weight)
    if args.ordinal_cost_weight > 0:
        cls_ids = torch.arange(probs.size(1), device=probs.device, dtype=probs.dtype).unsqueeze(0)
        ordinal_distance = torch.abs(cls_ids - y.float().unsqueeze(1))
        l_ord = weighted_mean(torch.sum(probs * ordinal_distance, dim=1), sample_weight)
    else:
        l_ord = probs.sum() * 0.0
    # Keep the auxiliary head in the autograd graph even when the auxiliary
    # objective is disabled; otherwise DDP reports reg_head as an unused branch.
    l_reg = weighted_mean((reg - y_reg) ** 2, sample_weight) if args.aux_reg_weight > 0 else reg.sum() * 0.0
    total = (
        l_cls
        + fp_weight * l_fp
        + boost_weight * l_boost
        + float(args.clear_to_fog_weight) * l_clear_fog
        + float(args.clear_to_mist_weight) * l_clear_mist
        + float(args.moderate_fn_weight) * l_mist_guard
        + args.ordinal_cost_weight * l_ord
        + args.aux_reg_weight * l_reg
    )
    return total, {
        "cls": float(l_cls.detach()),
        "fp": float(l_fp.detach()),
        "boost": float(l_boost.detach()),
        "clear_fog": float(l_clear_fog.detach()),
        "clear_mist": float(l_clear_mist.detach()),
        "mist_guard": float(l_mist_guard.detach()),
        "ord": float(l_ord.detach()),
        "reg": float(l_reg.detach()),
    }


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


def pred_from_regression_logvis(logvis_pred: np.ndarray) -> np.ndarray:
    logvis = np.asarray(logvis_pred, dtype=np.float64)
    vis = np.expm1(np.clip(logvis, 0.0, np.log1p(80000.0)))
    pred = np.full(len(vis), 2, dtype=np.int64)
    pred[vis < 1000.0] = 1
    pred[vis < 500.0] = 0
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


def event_group_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    group_ids: np.ndarray,
    min_fog_count: int,
) -> Dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(pred, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.int64)
    valid = groups >= 0
    if not np.any(valid):
        return {}
    labels = labels[valid]
    predictions = predictions[valid]
    groups = groups[valid]
    _, inverse = np.unique(groups, return_inverse=True)
    low_true = labels <= 1
    low_pred = predictions <= 1
    fog_true = labels == 0
    n_groups = int(inverse.max()) + 1
    obs_low = np.bincount(inverse, weights=low_true.astype(np.float64), minlength=n_groups)
    obs_fog = np.bincount(inverse, weights=fog_true.astype(np.float64), minlength=n_groups)
    pred_low = np.bincount(inverse, weights=low_pred.astype(np.float64), minlength=n_groups)
    tp = np.bincount(inverse, weights=(low_true & low_pred).astype(np.float64), minlength=n_groups)
    fp = np.bincount(inverse, weights=(~low_true & low_pred).astype(np.float64), minlength=n_groups)
    fn = np.bincount(inverse, weights=(low_true & ~low_pred).astype(np.float64), minlength=n_groups)
    event = (obs_fog >= int(min_fog_count)) & (obs_low > 0)
    if not np.any(event):
        return {"event_group_count": 0.0}
    recall = tp[event] / np.maximum(obs_low[event], 1.0)
    csi = tp[event] / np.maximum(tp[event] + fp[event] + fn[event], 1.0)
    area_ratio = pred_low[event] / np.maximum(obs_low[event], 1.0)
    return {
        "event_group_count": float(np.sum(event)),
        "event_low_vis_recall_mean": float(np.mean(recall)),
        "event_low_vis_recall_min": float(np.min(recall)),
        "event_low_vis_csi_mean": float(np.mean(csi)),
        "event_low_vis_area_ratio_mean": float(np.mean(area_ratio)),
        "event_low_vis_area_ratio_max": float(np.max(area_ratio)),
    }


def footprint_constraint_shortfall(args: argparse.Namespace, metrics: Dict[str, float]) -> float:
    terms = [
        max(0.0, float(args.phase_c_min_low_vis_recall) - metrics.get("low_vis_recall", 0.0))
        / max(float(args.phase_c_min_low_vis_recall), 1e-6),
        max(0.0, metrics.get("false_positive_rate", 1.0) - float(args.phase_c_max_fpr))
        / max(float(args.phase_c_max_fpr), 1e-6),
    ]
    if metrics.get("event_group_count", 0.0) > 0:
        terms.extend(
            [
                max(
                    0.0,
                    float(args.phase_c_min_event_mean_recall)
                    - metrics.get("event_low_vis_recall_mean", 0.0),
                )
                / max(float(args.phase_c_min_event_mean_recall), 1e-6),
                max(
                    0.0,
                    float(args.phase_c_min_event_recall)
                    - metrics.get("event_low_vis_recall_min", 0.0),
                )
                / max(float(args.phase_c_min_event_recall), 1e-6),
                max(
                    0.0,
                    metrics.get("event_low_vis_area_ratio_mean", float("inf"))
                    - float(args.phase_c_max_event_area_ratio_mean),
                )
                / max(float(args.phase_c_max_event_area_ratio_mean), 1e-6),
                max(
                    0.0,
                    metrics.get("event_low_vis_area_ratio_max", float("inf"))
                    - float(args.phase_c_max_event_area_ratio),
                )
                / max(float(args.phase_c_max_event_area_ratio), 1e-6),
            ]
        )
    else:
        # A footprint-selected checkpoint must have auditable event groups.
        terms.append(4.0)
    return float(np.sum(terms))


def sampling_constraint_shortfall(args: argparse.Namespace, metrics: Dict[str, float]) -> float:
    terms = [
        max(0.0, float(args.phase_d_min_low_vis_csi) - metrics.get("low_vis_csi", 0.0))
        / max(float(args.phase_d_min_low_vis_csi), 1e-6),
        max(0.0, metrics.get("false_positive_rate", 1.0) - float(args.phase_d_max_fpr))
        / max(float(args.phase_d_max_fpr), 1e-6),
    ]
    if metrics.get("event_group_count", 0.0) > 0:
        mean_area = metrics.get("event_low_vis_area_ratio_mean", 0.0)
        terms.extend(
            [
                max(
                    0.0,
                    float(args.phase_d_min_event_mean_csi)
                    - metrics.get("event_low_vis_csi_mean", 0.0),
                )
                / max(float(args.phase_d_min_event_mean_csi), 1e-6),
                max(
                    0.0,
                    float(args.phase_d_min_event_mean_recall)
                    - metrics.get("event_low_vis_recall_mean", 0.0),
                )
                / max(float(args.phase_d_min_event_mean_recall), 1e-6),
                max(
                    0.0,
                    float(args.phase_d_min_event_recall)
                    - metrics.get("event_low_vis_recall_min", 0.0),
                )
                / max(float(args.phase_d_min_event_recall), 1e-6),
                max(0.0, float(args.phase_d_min_event_area_ratio_mean) - mean_area)
                / max(float(args.phase_d_min_event_area_ratio_mean), 1e-6),
                max(0.0, mean_area - float(args.phase_d_max_event_area_ratio_mean))
                / max(float(args.phase_d_max_event_area_ratio_mean), 1e-6),
                max(
                    0.0,
                    metrics.get("event_low_vis_area_ratio_max", float("inf"))
                    - float(args.phase_d_max_event_area_ratio),
                )
                / max(float(args.phase_d_max_event_area_ratio), 1e-6),
            ]
        )
    else:
        terms.append(4.0)
    return float(np.sum(terms))


def score_metrics(
    args: argparse.Namespace,
    metrics: Dict[str, float],
    selection_metric: Optional[str] = None,
) -> float:
    metric_name = selection_metric or args.selection_metric
    if metric_name == "footprint_csi":
        base = (
            0.55 * metrics["low_vis_csi"]
            + 0.15 * metrics["Fog_CSI"]
            + 0.15 * metrics["Mist_CSI"]
            + 0.15 * metrics["low_vis_precision"]
            - 0.25 * metrics["false_positive_rate"]
        )
        return float(base - 2.0 * footprint_constraint_shortfall(args, metrics))
    if metric_name == "sampling_csi":
        base = (
            0.45 * metrics["low_vis_csi"]
            + 0.20 * metrics.get("event_low_vis_csi_mean", 0.0)
            + 0.20 * metrics["low_vis_precision"]
            + 0.075 * metrics["Fog_CSI"]
            + 0.075 * metrics["Mist_CSI"]
            - 0.25 * metrics["false_positive_rate"]
        )
        return float(base - 2.0 * sampling_constraint_shortfall(args, metrics))
    if metric_name == "csi":
        return 0.45 * metrics["Fog_CSI"] + 0.45 * metrics["Mist_CSI"] + 0.10 * metrics["low_vis_precision"] - 0.05 * metrics["false_positive_rate"]
    if metric_name == "recall":
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
    fog_counts, mist_counts, class_counts = threshold_prediction_counts(probs, y_true, grid)
    tiers = [
        (args.min_fog_precision, args.min_mist_precision, args.min_clear_recall),
        (max(0.05, args.min_fog_precision - 0.05), max(0.05, args.min_mist_precision - 0.05), max(0.84, args.min_clear_recall - 0.04)),
    ]
    for tier_id, (min_fp, min_mp, min_cr) in enumerate(tiers, start=1):
        found = False
        for fog_idx, fth in enumerate(grid):
            for mist_idx, mth in enumerate(grid):
                metrics = metrics_from_threshold_counts(
                    fog_counts[fog_idx], mist_counts[mist_idx], class_counts
                )
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
    group_ids: Optional[torch.Tensor],
    world_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if world_size <= 1:
        groups_np = group_ids.cpu().numpy() if group_ids is not None else None
        return probs.cpu().numpy(), targets.cpu().numpy(), groups_np
    local_n = torch.tensor([probs.shape[0]], dtype=torch.long, device=probs.device)
    max_n = local_n.clone()
    dist.all_reduce(max_n, op=dist.ReduceOp.MAX)
    pad_n = int(max_n.item() - local_n.item())
    if pad_n:
        probs = torch.cat([probs, torch.zeros((pad_n, probs.shape[1]), dtype=probs.dtype, device=probs.device)], dim=0)
        targets = torch.cat([targets, torch.full((pad_n,), -1, dtype=targets.dtype, device=targets.device)], dim=0)
        if group_ids is not None:
            group_ids = torch.cat(
                [group_ids, torch.full((pad_n,), -1, dtype=group_ids.dtype, device=group_ids.device)],
                dim=0,
            )
    gp = [torch.zeros_like(probs) for _ in range(world_size)]
    gt = [torch.zeros_like(targets) for _ in range(world_size)]
    dist.all_gather(gp, probs)
    dist.all_gather(gt, targets)
    all_probs = torch.cat(gp, dim=0).cpu().numpy()
    all_targets = torch.cat(gt, dim=0).cpu().numpy()
    m = all_targets >= 0
    all_groups = None
    if group_ids is not None:
        gg = [torch.zeros_like(group_ids) for _ in range(world_size)]
        dist.all_gather(gg, group_ids)
        all_groups = torch.cat(gg, dim=0).cpu().numpy()[m]
    return all_probs[m], all_targets[m], all_groups


def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    n_actual: int,
    selection_metric: Optional[str] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    probs_l, targets_l, reg_l, groups_l = [], [], [], []
    with torch.no_grad():
        for bx, by, _, braw, _, bgroups in loader:
            bx = bx.to(device, non_blocking=True)
            logits, reg = model(bx)
            if getattr(args, "loss_mode", "designed_focal") == "regression":
                reg_l.append(torch.stack([reg, braw.to(device)], dim=1))
            else:
                probs_l.append(torch.softmax(logits, dim=1))
            targets_l.append(by.to(device))
            groups_l.append(bgroups.to(device))
    targets = torch.cat(targets_l, dim=0)
    group_ids = torch.cat(groups_l, dim=0)
    if getattr(args, "loss_mode", "designed_focal") == "regression":
        reg_pack = torch.cat(reg_l, dim=0)
        all_reg, all_targets, _ = gather_eval_arrays(reg_pack, targets, None, world_size)
        all_reg = all_reg[:n_actual]
        all_targets = all_targets[:n_actual]
        if rank == 0:
            pred = pred_from_regression_logvis(all_reg[:, 0])
            metrics = build_metrics(all_targets.astype(np.int64), pred)
            vis_pred = np.expm1(np.clip(all_reg[:, 0], 0.0, np.log1p(80000.0)))
            vis_true = np.maximum(all_reg[:, 1], 0.0)
            err = vis_pred - vis_true
            metrics["regression_mae_m"] = float(np.mean(np.abs(err)))
            metrics["regression_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
            return score_metrics(args, metrics, selection_metric), {"fog_vis_m": 500.0, "mist_vis_m": 1000.0}, metrics
        return -1.0, {"fog_vis_m": 500.0, "mist_vis_m": 1000.0}, {}

    probs = torch.cat(probs_l, dim=0)
    all_probs, all_targets, all_groups = gather_eval_arrays(probs, targets, group_ids, world_size)
    all_probs = all_probs[:n_actual]
    all_targets = all_targets[:n_actual]
    if all_groups is not None:
        all_groups = all_groups[:n_actual]
    if rank == 0:
        if args.threshold_mode == "argmax":
            pred = np.argmax(all_probs, axis=1)
            metrics = build_metrics(all_targets.astype(np.int64), pred)
            metric_name = selection_metric or args.selection_metric
            if all_groups is not None:
                min_fog_count = (
                    args.phase_d_min_fog_count
                    if metric_name == "sampling_csi"
                    else args.event_footprint_min_fog_count
                )
                metrics.update(
                    event_group_metrics(
                        all_targets.astype(np.int64),
                        pred,
                        all_groups,
                        min_fog_count,
                    )
                )
            score = score_metrics(args, metrics, selection_metric)
            if metric_name == "footprint_csi":
                shortfall = footprint_constraint_shortfall(args, metrics)
                metrics["selection_constraint_shortfall"] = shortfall
                metrics["selection_feasible"] = float(shortfall <= 1e-12)
            elif metric_name == "sampling_csi":
                shortfall = sampling_constraint_shortfall(args, metrics)
                metrics["selection_constraint_shortfall"] = shortfall
                metrics["selection_feasible"] = float(shortfall <= 1e-12)
            th = {"mode": "argmax"}
        else:
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


def _normalise_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = str(k)
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        out[kk] = v
    return out


def infer_checkpoint_layout(
    state: Dict[str, torch.Tensor],
    metadata: Optional[Dict],
) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
    meta_layout = (metadata or {}).get("layout") if isinstance(metadata, dict) else None
    dyn_vars = None
    fe_dim = None
    use_fe = None
    if isinstance(meta_layout, dict):
        if meta_layout.get("dyn_vars") is not None:
            dyn_vars = int(meta_layout["dyn_vars"])
        if meta_layout.get("fe_dim") is not None:
            fe_dim = int(meta_layout["fe_dim"])
    if isinstance(metadata, dict) and metadata.get("use_fe") is not None:
        use_fe = bool(metadata["use_fe"])

    state_n = _normalise_state_dict_keys(state)
    w = state_n.get("dynamic_proj.0.weight")
    if dyn_vars is None and torch.is_tensor(w) and w.ndim == 2:
        dyn_vars = int(w.shape[1])
    fe_w = state_n.get("fe_encoder.0.weight")
    if fe_dim is None and torch.is_tensor(fe_w) and fe_w.ndim == 2:
        fe_dim = int(fe_w.shape[1])
    if use_fe is None:
        use_fe = "fe_encoder.0.weight" in state_n
    return dyn_vars, fe_dim, use_fe


def validate_pretrained_layout(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    metadata: Optional[Dict],
    path: str,
    policy: str,
) -> None:
    if policy == "compatible":
        return
    target = unwrap(model)
    ckpt_dyn, ckpt_fe, ckpt_use_fe = infer_checkpoint_layout(state, metadata)
    problems: List[str] = []
    if ckpt_dyn is not None and ckpt_dyn != target.layout.dyn_vars:
        problems.append(f"dyn_vars checkpoint={ckpt_dyn} target={target.layout.dyn_vars}")
    meta_layout = (metadata or {}).get("layout") if isinstance(metadata, dict) else None
    ckpt_order = meta_layout.get("dynamic_feature_order") if isinstance(meta_layout, dict) else None
    if ckpt_order and target.layout.dynamic_feature_order:
        ckpt_norm = [normalize_feature_name(v) for v in ckpt_order]
        target_norm = [normalize_feature_name(v) for v in target.layout.dynamic_feature_order]
        if ckpt_norm != target_norm:
            problems.append("dynamic_feature_order differs between checkpoint and target")
    if target.use_fe:
        if ckpt_use_fe is False:
            problems.append("checkpoint has no FE encoder but target uses FE")
        if ckpt_fe is not None and ckpt_fe != target.layout.fe_dim:
            problems.append(f"fe_dim checkpoint={ckpt_fe} target={target.layout.fe_dim}")
    if problems:
        raise ValueError(
            "Pretrained checkpoint layout mismatch for "
            f"{path}: {'; '.join(problems)}. "
            "Use a matching S1 checkpoint for this feature layout, or pass "
            "--pretrained-layout-policy compatible only for an intentional "
            "partial tensor load."
        )


def load_compatible_checkpoint(
    model: nn.Module,
    path: str,
    rank: int,
    device: torch.device,
    layout_policy: str = "strict",
) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state_dict-like mapping")
    target = unwrap(model)
    own = target.state_dict()
    validate_pretrained_layout(target, state, metadata, path, layout_policy)
    state = _normalise_state_dict_keys(state)
    compatible = {}
    for k, v in state.items():
        if k in own and tuple(v.shape) == tuple(own[k].shape):
            compatible[k] = v
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    ckpt_dyn, ckpt_fe, ckpt_use_fe = infer_checkpoint_layout(state, metadata)
    rank0(
        rank,
        f"[Ckpt] loaded {len(compatible)} tensors from {path}; "
        f"policy={layout_policy} ckpt_dyn={ckpt_dyn} ckpt_fe={ckpt_fe} "
        f"ckpt_use_fe={ckpt_use_fe} target_dyn={target.layout.dyn_vars} "
        f"target_fe={target.layout.fe_dim} missing={len(missing)} unexpected={len(unexpected)}",
    )


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
) -> Tuple[DataLoader, DataLoader, object]:
    def worker_init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.X = None

    if args.sampler_mode == "stratified_balanced":
        sampler = StratifiedBalancedBatchSampler(
            train_ds,
            args.batch_size,
            fog_ratio=fog_ratio,
            mist_ratio=mist_ratio,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            epoch_length=args.epoch_length,
        )
        loader_kwargs = {
            "batch_sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
        }
    elif args.sampler_mode == "natural_shuffle":
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "drop_last": True,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
        }
    else:
        raise ValueError(f"Unknown sampler_mode: {args.sampler_mode}")
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


def make_event_footprint_loader(
    args: argparse.Namespace,
    train_ds: LowVisDataset,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, EventTimeBatchSampler]:
    if train_ds.time_group_index is None:
        raise ValueError("Phase-C event footprint requires a train time-group index")

    def worker_init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.X = None

    sampler = EventTimeBatchSampler(
        train_ds.time_group_index,
        args.batch_size,
        min_fog_count=args.event_footprint_min_fog_count,
        event_batch_ratio=args.event_footprint_event_batch_ratio,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 7001,
        epoch_length=args.epoch_length,
    )
    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 1, "timeout": 900})
    return DataLoader(train_ds, **loader_kwargs), sampler


def make_sampling_calibration_loader(
    args: argparse.Namespace,
    train_ds: LowVisDataset,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, EventTimeBatchSampler]:
    if train_ds.time_group_index is None:
        raise ValueError("Phase-D sampling calibration requires a train time-group index")

    def worker_init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.X = None

    sampler = EventTimeBatchSampler(
        train_ds.time_group_index,
        args.batch_size,
        min_fog_count=args.phase_d_min_fog_count,
        event_batch_ratio=args.phase_d_event_batch_ratio,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 9001,
        epoch_length=args.epoch_length,
        full_group=True,
    )
    if args.phase_d_event_batch_ratio > 0 and len(sampler.event_groups) == 0:
        raise ValueError("Phase-D sampling calibration found no widespread-event time groups")
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
    return DataLoader(train_ds, **loader_kwargs), sampler


def sampling_metadata(
    args: argparse.Namespace,
    train_ds: LowVisDataset,
    fog_ratio: float,
    mist_ratio: float,
) -> Dict[str, object]:
    y = train_ds.y_cls.numpy()
    counts = np.bincount(y.astype(np.int64), minlength=3)
    meta: Dict[str, object] = {
        "sampler_mode": args.sampler_mode,
        "batch_size": int(args.batch_size),
        "train_class_counts": {
            "fog": int(counts[0]),
            "mist": int(counts[1]),
            "clear": int(counts[2]),
        },
    }
    if args.sampler_mode == "stratified_balanced":
        n_fog = max(1, int(args.batch_size * float(fog_ratio)))
        n_mist = max(1, int(args.batch_size * float(mist_ratio)))
        n_clear = int(args.batch_size) - n_fog - n_mist
        meta.update(
            {
                "fog_ratio": float(fog_ratio),
                "mist_ratio": float(mist_ratio),
                "epoch_length": int(args.epoch_length),
                "batch_class_counts": {
                    "fog": int(n_fog),
                    "mist": int(n_mist),
                    "clear": int(n_clear),
                },
            }
        )
    return meta


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
    sampling_meta = sampling_metadata(args, train_ds, fog_ratio, mist_ratio)
    is_phase_c = tag == "S2_PhaseC"
    is_phase_d = tag == "S2_PhaseD"
    if is_phase_c:
        selection_metric = args.phase_c_selection_metric
    elif is_phase_d:
        selection_metric = args.phase_d_selection_metric
    else:
        selection_metric = args.selection_metric
    prior_beta = float(args.phase_c_prior_beta) if is_phase_c else 0.0
    if args.sampler_mode == "stratified_balanced":
        sampler_prior = np.asarray([fog_ratio, mist_ratio, 1.0 - fog_ratio - mist_ratio], dtype=np.float64)
    else:
        counts = np.bincount(train_ds.y_cls.numpy().astype(np.int64), minlength=3).astype(np.float64)
        sampler_prior = counts / max(float(counts.sum()), 1.0)
    prior_weights_np = class_prior_correction_weights(train_ds.y_cls.numpy(), sampler_prior, prior_beta)
    focal_prior_weights = torch.as_tensor(prior_weights_np, dtype=torch.float32, device=device)
    sampling_meta.update(
        {
            "phase_c_prior_beta": prior_beta,
            "phase_c_sampler_prior": sampler_prior.tolist(),
            "phase_c_focal_prior_weights": prior_weights_np.tolist(),
        }
    )
    use_footprint = bool(is_phase_c and event_footprint_enabled(args))
    footprint_loader = None
    footprint_sampler = None
    footprint_iterator = None
    footprint_epoch = 0
    dual_state = FootprintDualState(
        area=float(args.event_footprint_dual_init),
        recall=float(args.event_footprint_dual_init),
    )
    if use_footprint:
        footprint_loader, footprint_sampler = make_event_footprint_loader(args, train_ds, rank, world_size)
        footprint_iterator = iter(footprint_loader)
        sampling_meta["event_footprint"] = {
            "enabled": True,
            "min_fog_count": int(args.event_footprint_min_fog_count),
            "event_batch_ratio": float(args.event_footprint_event_batch_ratio),
            "n_event_groups_rank": int(len(footprint_sampler.event_groups)),
            "n_background_groups_rank": int(len(footprint_sampler.background_groups)),
        }
    use_sampling_calibration = bool(is_phase_d and sampling_calibration_enabled(args))
    calibration_loader = None
    calibration_sampler = None
    calibration_iterator = None
    calibration_epoch = 0
    if use_sampling_calibration:
        calibration_loader, calibration_sampler = make_sampling_calibration_loader(
            args, train_ds, rank, world_size
        )
        calibration_iterator = iter(calibration_loader)
        sampling_meta["sampling_calibration"] = {
            "enabled": True,
            "balanced_loss": "designed_focal",
            "natural_snapshot_loss": "unweighted_cross_entropy",
            "final_natural_mix": float(args.phase_d_natural_mix),
            "ramp_start": float(args.phase_d_ramp_start),
            "min_fog_count": int(args.phase_d_min_fog_count),
            "event_batch_ratio": float(args.phase_d_event_batch_ratio),
            "full_group": True,
            "n_event_groups_rank": int(len(calibration_sampler.event_groups)),
            "n_background_groups_rank": int(len(calibration_sampler.background_groups)),
        }
    if world_size > 1:
        dist.all_reduce(torch.zeros(1, device=device), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    rank0(
        rank,
        f"[{tag}] start steps={total_steps} trainable={trainable} "
        f"loss_mode={getattr(args, 'loss_mode', 'designed_focal')} "
        f"sampler_mode={args.sampler_mode} fog_ratio={fog_ratio} mist_ratio={mist_ratio} "
        f"prior_beta={prior_beta} footprint={use_footprint} "
        f"sampling_calibration={use_sampling_calibration} selection={selection_metric}",
    )

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
            bx, by, breg, braw, bw, _ = next(iterator)
        except StopIteration:
            epoch += 1
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            bx, by, breg, braw, bw, _ = next(iterator)
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)
        breg = breg.to(device, non_blocking=True)
        braw = braw.to(device, non_blocking=True)
        bw = bw.to(device, non_blocking=True)
        batch_count += 1
        is_sync = batch_count % args.grad_accum == 0
        fx = None
        fy = None
        footprint_is_event = False
        if use_footprint:
            assert footprint_loader is not None and footprint_sampler is not None
            assert footprint_iterator is not None
            try:
                fx, fy, _, _, _, fgroups = next(footprint_iterator)
            except StopIteration:
                footprint_epoch += 1
                footprint_sampler.set_epoch(footprint_epoch)
                footprint_iterator = iter(footprint_loader)
                fx, fy, _, _, _, fgroups = next(footprint_iterator)
            fx = fx.to(device, non_blocking=True)
            fy = fy.to(device, non_blocking=True)
            footprint_is_event = footprint_sampler.is_event_group(int(fgroups[0]))
        calibration_mix = (
            sampling_calibration_mix(
                step,
                total_steps,
                args.phase_d_ramp_start,
                args.phase_d_natural_mix,
            )
            if use_sampling_calibration
            else 0.0
        )
        cx = None
        cy = None
        if use_sampling_calibration and calibration_mix > 0:
            assert calibration_loader is not None and calibration_sampler is not None
            assert calibration_iterator is not None
            try:
                cx, cy, _, _, _, _ = next(calibration_iterator)
            except StopIteration:
                calibration_epoch += 1
                calibration_sampler.set_epoch(calibration_epoch)
                calibration_iterator = iter(calibration_loader)
                cx, cy, _, _, _, _ = next(calibration_iterator)
            cx = cx.to(device, non_blocking=True)
            cy = cy.to(device, non_blocking=True)
        ctx = model.no_sync() if world_size > 1 and not is_sync else contextlib.nullcontext()
        with ctx:
            if use_footprint:
                assert fx is not None and fy is not None
                joined_logits, joined_reg = model(torch.cat([bx, fx], dim=0))
                logits = joined_logits[: len(bx)]
                reg = joined_reg[: len(bx)]
                footprint_logits = joined_logits[len(bx) :]
                calibration_logits = None
            elif use_sampling_calibration and cx is not None:
                joined_logits, joined_reg = model(torch.cat([bx, cx], dim=0))
                logits = joined_logits[: len(bx)]
                reg = joined_reg[: len(bx)]
                footprint_logits = None
                calibration_logits = joined_logits[len(bx) :]
            else:
                logits, reg = model(bx)
                footprint_logits = None
                calibration_logits = None
            loss, loss_parts = combined_loss(
                args,
                focal,
                logits,
                reg,
                by,
                breg,
                braw,
                bw,
                focal_prior_weights=focal_prior_weights,
            )
            footprint_parts = None
            if use_footprint:
                assert footprint_logits is not None and fy is not None
                footprint_term, footprint_parts = event_footprint_loss(
                    args,
                    footprint_logits,
                    fy,
                    dual_state,
                    is_widespread_event=footprint_is_event,
                )
                loss = loss + footprint_term
                loss_parts.update(
                    {
                        "footprint": float(footprint_parts["footprint"].detach()),
                        "soft_csi": float(footprint_parts["soft_csi"].detach()),
                        "soft_recall": float(footprint_parts["soft_recall"].detach()),
                        "pred_area": float(footprint_parts["pred_area"].detach()),
                        "obs_area": float(footprint_parts["obs_area"].detach()),
                        "area_violation": float(footprint_parts["area_violation"].detach()),
                        "recall_violation": float(footprint_parts["recall_violation"].detach()),
                    }
                )
            if use_sampling_calibration:
                natural_ce = logits.sum() * 0.0
                if calibration_logits is not None and cy is not None:
                    natural_ce = F.cross_entropy(calibration_logits, cy)
                    loss = (1.0 - calibration_mix) * loss + calibration_mix * natural_ce
                loss_parts.update(
                    {
                        "calibration_mix": float(calibration_mix),
                        "natural_ce": float(natural_ce.detach()),
                    }
                )
            if l2sp_ref and l2sp_alpha > 0:
                loss = loss + l2sp_alpha * l2sp_penalty(model, l2sp_ref)
            (loss / args.grad_accum).backward()
        if not is_sync:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()
            scheduler.step()
            if use_footprint and footprint_parts is not None:
                update_footprint_duals(
                    args,
                    dual_state,
                    footprint_parts["area_violation"],
                    footprint_parts["recall_violation"],
                    world_size,
                )
        else:
            rank0(rank, f"[{tag}] nonfinite grad norm at step={step}: {grad_norm}")
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if rank == 0 and step % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            footprint_log = (
                f"ef={loss_parts.get('footprint', 0.0):.4f} "
                f"ecsi={loss_parts.get('soft_csi', 0.0):.3f} "
                f"area={loss_parts.get('pred_area', 0.0):.3f}/{loss_parts.get('obs_area', 0.0):.3f} "
                f"dual={dual_state.area:.3f}/{dual_state.recall:.3f} "
                if use_footprint
                else ""
            )
            calibration_log = (
                f"mix={loss_parts.get('calibration_mix', 0.0):.3f} "
                f"natural_ce={loss_parts.get('natural_ce', 0.0):.4f} "
                if use_sampling_calibration
                else ""
            )
            print(
                f"[{tag}] step={step}/{total_steps} loss={float(loss):.4f} "
                f"cls={loss_parts['cls']:.4f} fp={loss_parts['fp']:.4f} "
                f"boost={loss_parts['boost']:.4f} ord={loss_parts['ord']:.4f} "
                f"cf={loss_parts['clear_fog']:.4f} cm={loss_parts['clear_mist']:.4f} "
                f"mg={loss_parts['mist_guard']:.4f} "
                f"reg={loss_parts['reg']:.4f} "
                f"{footprint_log}"
                f"{calibration_log}"
                f"lr={lr_now:.2e} no_improve={no_improve}/{args.patience}",
                flush=True,
            )

        if step % args.val_interval == 0 or step == total_steps:
            score, th, metrics = evaluate(
                args,
                model,
                val_loader,
                device,
                rank,
                world_size,
                len(val_ds),
                selection_metric=selection_metric,
            )
            model.train()
            raw_for_meta = unwrap(model)
            ckpt_meta = {
                "run_id": args.run_id,
                "seed": int(args.seed),
                "tag": tag,
                "step": step,
                "layout": asdict(raw_for_meta.layout),
                "use_fe": bool(raw_for_meta.use_fe),
                "encoder": raw_for_meta.encoder,
                "pooling": raw_for_meta.pooling,
                "score": score,
                "thresholds": th,
                "metrics": metrics,
                "selection_metric": selection_metric,
                "loss_mode": getattr(args, "loss_mode", "designed_focal"),
                "decision_type": (
                    "regression_threshold"
                    if getattr(args, "loss_mode", "designed_focal") == "regression"
                    else ("argmax" if args.threshold_mode == "argmax" else "probability_threshold")
                ),
                "threshold_mode": args.threshold_mode,
                "sampling": sampling_meta,
                "loss_terms": {
                    "class_weight_fog": float(args.class_weight_fog),
                    "class_weight_mist": float(args.class_weight_mist),
                    "class_weight_clear": float(args.class_weight_clear),
                    "focal_gamma_fog": float(args.focal_gamma_fog),
                    "focal_gamma_mist": float(args.focal_gamma_mist),
                    "focal_gamma_clear": float(args.focal_gamma_clear),
                    "alpha_clear_fp": float(args.alpha_clear_fp),
                    "alpha_recall_boost": float(args.alpha_recall_boost),
                    "event_fp_weight": float(args.alpha_clear_fp if args.event_fp_weight is None else args.event_fp_weight),
                    "event_fn_weight": float(args.alpha_recall_boost if args.event_fn_weight is None else args.event_fn_weight),
                    "event_loss_normalization": str(args.event_loss_normalization),
                    "clear_to_fog_weight": float(args.clear_to_fog_weight),
                    "clear_to_mist_weight": float(args.clear_to_mist_weight),
                    "clear_pair_vis_min": float(args.clear_pair_vis_min),
                    "moderate_fn_weight": float(args.moderate_fn_weight),
                    "label_smoothing": bool(args.label_smoothing),
                    "soft_fog_mist_low": float(args.soft_fog_mist_low),
                    "soft_fog_mist_high": float(args.soft_fog_mist_high),
                    "soft_mist_clear_low": float(args.soft_mist_clear_low),
                    "soft_mist_clear_high": float(args.soft_mist_clear_high),
                    "boundary_weight": float(args.boundary_weight),
                    "physical_hard_weight": float(args.physical_hard_weight),
                    "aerosol_hard_weight": float(args.aerosol_hard_weight),
                    "ordinal_cost_weight": float(args.ordinal_cost_weight),
                    "aux_reg_weight": float(args.aux_reg_weight),
                    "phase_c_prior_beta": prior_beta,
                    "phase_c_focal_prior_weights": prior_weights_np.tolist(),
                    "event_footprint_csi_weight": float(args.event_footprint_csi_weight),
                    "event_footprint_area_ratio_cap": float(args.event_footprint_area_ratio_cap),
                    "event_footprint_area_slack": float(args.event_footprint_area_slack),
                    "event_footprint_min_recall": float(args.event_footprint_min_recall),
                    "event_footprint_dual_area": float(dual_state.area),
                    "event_footprint_dual_recall": float(dual_state.recall),
                    "phase_d_natural_mix": float(args.phase_d_natural_mix),
                    "phase_d_current_natural_mix": float(calibration_mix),
                    "phase_d_ramp_start": float(args.phase_d_ramp_start),
                    "phase_d_event_batch_ratio": float(args.phase_d_event_batch_ratio),
                },
            }
            save_checkpoint(model, ckpt_latest, rank, ckpt_meta)
            if rank == 0:
                row = {
                    "step": step,
                    "score": score,
                    "thresholds": th,
                    "phase_d_natural_mix": float(calibration_mix),
                    **metrics,
                }
                history.append(row)
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                print(
                    f"[{tag}] val score={score:.4f} th={th} "
                    f"Fog CSI/R/P={metrics.get('Fog_CSI', -1):.3f}/{metrics.get('Fog_R', -1):.3f}/{metrics.get('Fog_P', -1):.3f} "
                    f"Mist CSI/R/P={metrics.get('Mist_CSI', -1):.3f}/{metrics.get('Mist_R', -1):.3f}/{metrics.get('Mist_P', -1):.3f} "
                    f"LVPrec={metrics.get('low_vis_precision', -1):.3f} FPR={metrics.get('false_positive_rate', -1):.3f} "
                    f"EventCSI={metrics.get('event_low_vis_csi_mean', -1):.3f} "
                    f"EventArea={metrics.get('event_low_vis_area_ratio_mean', -1):.3f} "
                    f"mix={calibration_mix:.3f} "
                    f"feasible={bool(metrics.get('selection_feasible', 1.0))}",
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
    seed = int(args.seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
            load_compatible_checkpoint(model, pretrained, rank, device, args.pretrained_layout_policy)
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
            load_compatible_checkpoint(raw_model, phase_a_best, rank, device, args.pretrained_layout_policy)
        set_trainable(raw_model, "all")
        ddp_model = wrap_ddp(raw_model, local_rank, world_size, find_unused=False)
        phase_b_best = train_stage(
            args, "S2_PhaseB", ddp_model, tr, va, device, rank, world_size,
            args.s2_phase_b_steps, args.fog_ratio_s2, args.mist_ratio_s2,
            args.s2_lr_backbone_b, args.s2_lr_head_b, "all", l2_ref, args.l2sp_alpha_b,
        )
        safe_barrier(world_size, device)

        phase_c_best = ""
        if args.s2_phase_c_steps > 0:
            raw_model = unwrap(ddp_model)
            if world_size > 1:
                del ddp_model
                torch.cuda.empty_cache()
                safe_barrier(world_size, device)
            if phase_b_best:
                load_compatible_checkpoint(raw_model, phase_b_best, rank, device, args.pretrained_layout_policy)
            set_trainable(raw_model, "head")
            ddp_model = wrap_ddp(raw_model, local_rank, world_size, find_unused=True)
            phase_c_best = train_stage(
                args, "S2_PhaseC", ddp_model, tr, va, device, rank, world_size,
                args.s2_phase_c_steps, args.fog_ratio_s2, args.mist_ratio_s2,
                args.s2_lr_head_c, None, "head", l2_ref, args.l2sp_alpha_a,
            )
            safe_barrier(world_size, device)

        if args.s2_phase_d_steps > 0:
            raw_model = unwrap(ddp_model)
            if world_size > 1:
                del ddp_model
                torch.cuda.empty_cache()
                safe_barrier(world_size, device)
            if phase_c_best:
                load_compatible_checkpoint(raw_model, phase_c_best, rank, device, args.pretrained_layout_policy)
            elif phase_b_best:
                load_compatible_checkpoint(raw_model, phase_b_best, rank, device, args.pretrained_layout_policy)
            set_trainable(raw_model, "head")
            ddp_model = wrap_ddp(raw_model, local_rank, world_size, find_unused=True)
            train_stage(
                args, "S2_PhaseD", ddp_model, tr, va, device, rank, world_size,
                args.s2_phase_d_steps, args.fog_ratio_s2, args.mist_ratio_s2,
                args.s2_lr_head_d, None, "head", l2_ref, args.l2sp_alpha_a,
            )

    safe_barrier(world_size, device)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    rank0(rank, "[Done] train_static_rnn_lowvis.py finished.")


if __name__ == "__main__":
    main()
