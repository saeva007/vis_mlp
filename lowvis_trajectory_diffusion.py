#!/usr/bin/env python3
"""Shared data/model utilities for the candidate low-visibility trajectory models.

This module is intentionally independent from the paper-facing Static-RNN.  It
contains only the trajectory dataset contract, train-only preprocessing,
conditional Transformer models, and DDPM/DDIM helpers used by training and
evaluation.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


CONDITION_LEADS = tuple(range(0, 49))
TARGET_LEADS = tuple(range(12, 49))
TARGET_LENGTH = len(TARGET_LEADS)
DYNAMIC_FEATURE_ORDER = (
    "RH2M", "T2M", "PRECIP", "MSLP", "SW_RAD", "U10", "WSPD10",
    "V10", "WDIR10", "CAPE", "LCC", "T_925", "RH_925", "U_925",
    "WSPD925", "V_925", "DP_1000", "DP_925", "Q_1000", "Q_925",
    "W_925", "W_1000", "DPD", "INVERSION", "ZENITH", "PM10_ugm3",
    "PM25_ugm3",
)
LOG1P_DYNAMIC_NAMES = frozenset({"PRECIP", "SW_RAD", "CAPE", "PM10_ugm3", "PM25_ugm3"})
LOG1P_DYNAMIC_INDICES = tuple(i for i, name in enumerate(DYNAMIC_FEATURE_ORDER) if name in LOG1P_DYNAMIC_NAMES)
PM_UNIT_POLICY_VERSION = "pmst_canonical_units_v2_20260630"
PM_QC_POLICY_VERSION = "pm_explicit_legacy_scale_then_train_median_qc_v2_20260701"
MAX_VISIBILITY_M = 30000.0


def exact_lead_indices(lead_values: np.ndarray) -> Optional[np.ndarray]:
    lead_values = np.asarray(lead_values, dtype=float)
    indices = []
    for lead in CONDITION_LEADS:
        pos = int(np.argmin(np.abs(lead_values - lead)))
        if abs(float(lead_values[pos]) - float(lead)) > 0.1:
            return None
        indices.append(pos)
    if len(set(indices)) != len(indices):
        return None
    return np.asarray(indices, dtype=np.int64)


def monthly_tail_masks(
    sample_times: Sequence[object], gap_hours: int = 24, val_last_days: int = 3, test_last_days: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project monthly-tail valid-time partitions without importing legacy builders."""
    times = pd.DatetimeIndex(pd.to_datetime(sample_times))
    if gap_hours % 24:
        raise ValueError("gap_hours must be a multiple of 24")
    train = np.zeros(len(times), dtype=bool)
    val = np.zeros(len(times), dtype=bool)
    test = np.zeros(len(times), dtype=bool)
    gap_days = gap_hours // 24
    for period in times.to_period("M").unique():
        dim = period.days_in_month
        d_test0 = dim - test_last_days + 1
        d_val1 = d_test0 - gap_days - 1
        d_val0 = d_val1 - val_last_days + 1
        d_train_end = d_val0 - gap_days - 1
        if d_train_end < 1:
            continue
        start, end = period.start_time, period.end_time
        in_month = (times >= start) & (times <= end)
        train_end = pd.Timestamp(period.year, period.month, d_train_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        val_start = pd.Timestamp(period.year, period.month, d_val0)
        val_end = pd.Timestamp(period.year, period.month, d_val1) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        test_start = pd.Timestamp(period.year, period.month, d_test0)
        train |= in_month & (times <= train_end)
        val |= in_month & (times >= val_start) & (times <= val_end)
        test |= in_month & (times >= test_start)
    return train, val, test


def full_trajectory_split(
    target_times: Sequence[object], gap_hours: int = 24, val_last_days: int = 3, test_last_days: int = 3
) -> Optional[str]:
    masks = monthly_tail_masks(target_times, gap_hours, val_last_days, test_last_days)
    for tag, mask in zip(("train", "val", "test"), masks):
        if bool(np.all(mask)):
            return tag
    return None


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_features_from_init(values: Sequence[object]) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(values))
    hour = times.hour.to_numpy(dtype=np.float32)
    day = times.dayofyear.to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day / 366.0),
            np.cos(2.0 * np.pi * day / 366.0),
        ]
    ).astype(np.float32)


def transform_dynamic_numpy(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    for idx in LOG1P_DYNAMIC_INDICES:
        col = out[..., idx]
        finite = np.isfinite(col)
        col[finite] = np.log1p(np.maximum(col[finite], 0.0))
        out[..., idx] = col
    return out


@dataclass
class TrajectoryScaler:
    dynamic_median: List[float]
    dynamic_scale: List[float]
    static_median: List[float]
    static_scale: List[float]
    target_mean: List[float]
    target_scale: List[float]
    fitted_rows: int

    @classmethod
    def fit(cls, data_dir: str | Path, max_rows: int = 200_000) -> "TrajectoryScaler":
        base = Path(data_dir)
        dynamic = np.load(base / "dynamic_train.npy", mmap_mode="r")
        static = np.load(base / "static_cont_train.npy", mmap_mode="r")
        visibility = np.load(base / "visibility_train.npy", mmap_mode="r")
        target_mask = np.load(base / "target_mask_train.npy", mmap_mode="r")
        return cls.fit_arrays(dynamic, static, visibility, target_mask, max_rows=max_rows)

    @classmethod
    def fit_arrays(
        cls,
        dynamic: np.ndarray,
        static: np.ndarray,
        visibility: np.ndarray,
        target_mask: np.ndarray,
        max_rows: int = 200_000,
    ) -> "TrajectoryScaler":
        n = int(dynamic.shape[0])
        if n == 0:
            raise ValueError("Cannot fit trajectory scaler on an empty training split")
        take = min(n, int(max_rows))
        idx = np.linspace(0, n - 1, num=take, dtype=np.int64)
        dyn_sample = transform_dynamic_numpy(np.asarray(dynamic[idx], dtype=np.float32))
        flat = dyn_sample.reshape(-1, dyn_sample.shape[-1])
        dyn_med = np.nanmedian(flat, axis=0)
        q25, q75 = np.nanpercentile(flat, [25.0, 75.0], axis=0)
        dyn_scale = q75 - q25
        dyn_med = np.where(np.isfinite(dyn_med), dyn_med, 0.0)
        dyn_scale = np.where(np.isfinite(dyn_scale) & (dyn_scale > 1.0e-6), dyn_scale, 1.0)

        stat_sample = np.asarray(static[idx], dtype=np.float32)
        stat_med = np.nanmedian(stat_sample, axis=0)
        sq25, sq75 = np.nanpercentile(stat_sample, [25.0, 75.0], axis=0)
        stat_scale = sq75 - sq25
        stat_med = np.where(np.isfinite(stat_med), stat_med, 0.0)
        stat_scale = np.where(np.isfinite(stat_scale) & (stat_scale > 1.0e-6), stat_scale, 1.0)

        vis = np.asarray(visibility[idx], dtype=np.float32)
        mask = np.asarray(target_mask[idx], dtype=bool) & np.isfinite(vis)
        log_vis = np.where(mask, np.log1p(np.maximum(vis, 0.0)), np.nan)
        target_mean = np.nanmean(log_vis, axis=0)
        target_scale = np.nanstd(log_vis, axis=0)
        target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
        target_scale = np.where(np.isfinite(target_scale) & (target_scale > 1.0e-6), target_scale, 1.0)
        return cls(
            dynamic_median=dyn_med.astype(float).tolist(),
            dynamic_scale=dyn_scale.astype(float).tolist(),
            static_median=stat_med.astype(float).tolist(),
            static_scale=stat_scale.astype(float).tolist(),
            target_mean=target_mean.astype(float).tolist(),
            target_scale=target_scale.astype(float).tolist(),
            fitted_rows=take,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrajectoryScaler":
        return cls(**dict(value))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryScaler":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform_dynamic(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = transform_dynamic_numpy(values)
        valid = np.isfinite(out)
        med = np.asarray(self.dynamic_median, dtype=np.float32)
        scale = np.asarray(self.dynamic_scale, dtype=np.float32)
        out = np.where(valid, out, med)
        out = (out - med) / scale
        return out.astype(np.float32), valid.astype(np.float32)

    def transform_static(self, values: np.ndarray) -> np.ndarray:
        med = np.asarray(self.static_median, dtype=np.float32)
        scale = np.asarray(self.static_scale, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        return ((np.where(np.isfinite(values), values, med) - med) / scale).astype(np.float32)

    def transform_target(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.target_mean, dtype=np.float32)
        scale = np.asarray(self.target_scale, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
        log_vis = np.where(valid, np.log1p(np.maximum(values, 0.0)), mean)
        return ((log_vis - mean) / scale).astype(np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.target_mean, dtype=np.float32)
        scale = np.asarray(self.target_scale, dtype=np.float32)
        log_vis = np.asarray(values, dtype=np.float32) * scale + mean
        log_vis = np.clip(log_vis, 0.0, np.log1p(MAX_VISIBILITY_M))
        return np.expm1(log_vis).astype(np.float32)


class LowVisTrajectoryDataset(Dataset):
    """Memory-mapped trajectory dataset with train-only preprocessing."""

    def __init__(self, data_dir: str | Path, split: str, scaler: TrajectoryScaler):
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.scaler = scaler
        self.dynamic = np.load(self.data_dir / f"dynamic_{split}.npy", mmap_mode="r")
        self.static = np.load(self.data_dir / f"static_cont_{split}.npy", mmap_mode="r")
        self.veg = np.load(self.data_dir / f"veg_id_{split}.npy", mmap_mode="r")
        self.time_features = np.load(self.data_dir / f"time_features_{split}.npy", mmap_mode="r")
        self.visibility = np.load(self.data_dir / f"visibility_{split}.npy", mmap_mode="r")
        self.target_mask = np.load(self.data_dir / f"target_mask_{split}.npy", mmap_mode="r")
        n = int(self.dynamic.shape[0])
        expected = [self.static, self.veg, self.time_features, self.visibility, self.target_mask]
        if any(int(v.shape[0]) != n for v in expected):
            raise ValueError(f"Split {split} arrays have inconsistent row counts")
        if tuple(self.dynamic.shape[1:]) != (len(CONDITION_LEADS), len(DYNAMIC_FEATURE_ORDER)):
            raise ValueError(f"Unexpected dynamic shape: {self.dynamic.shape}")
        if tuple(self.visibility.shape[1:]) != (TARGET_LENGTH,):
            raise ValueError(f"Unexpected target shape: {self.visibility.shape}")

    def __len__(self) -> int:
        return int(self.dynamic.shape[0])

    def raw_visibility(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.visibility[idx], dtype=np.float32),
            np.asarray(self.target_mask[idx], dtype=bool),
        )

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        dynamic, dynamic_mask = self.scaler.transform_dynamic(np.asarray(self.dynamic[idx], dtype=np.float32))
        condition = np.concatenate([dynamic, dynamic_mask], axis=-1)
        static = self.scaler.transform_static(np.asarray(self.static[idx], dtype=np.float32))
        visibility = np.asarray(self.visibility[idx], dtype=np.float32)
        target_mask = np.asarray(self.target_mask[idx], dtype=bool) & np.isfinite(visibility)
        target = self.scaler.transform_target(visibility, target_mask)
        return {
            "condition": torch.from_numpy(condition),
            "static": torch.from_numpy(static),
            "veg": torch.tensor(int(self.veg[idx]), dtype=torch.long),
            "time_features": torch.from_numpy(np.asarray(self.time_features[idx], dtype=np.float32).copy()),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
            "visibility": torch.from_numpy(np.where(target_mask, visibility, 0.0).astype(np.float32)),
            "index": torch.tensor(int(idx), dtype=torch.long),
        }


def sinusoidal_embedding(steps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    freq = torch.exp(torch.arange(half, device=steps.device, dtype=torch.float32) * -scale)
    args = steps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TrajectoryConditionEncoder(nn.Module):
    def __init__(
        self,
        condition_dim: int = len(DYNAMIC_FEATURE_ORDER) * 2,
        static_dim: int = 5,
        time_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.dynamic_projection = nn.Linear(condition_dim, d_model)
        self.condition_lead_embedding = nn.Parameter(torch.randn(len(CONDITION_LEADS), d_model) * 0.02)
        self.veg_embedding = nn.Embedding(32, 16)
        self.global_encoder = nn.Sequential(
            nn.Linear(static_dim + time_dim + 16, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))

    def forward(self, condition: Tensor, static: Tensor, veg: Tensor, time_features: Tensor) -> Tuple[Tensor, Tensor]:
        veg = torch.clamp(veg, 0, self.veg_embedding.num_embeddings - 1)
        global_context = self.global_encoder(
            torch.cat([static, time_features, self.veg_embedding(veg)], dim=-1)
        )
        memory = self.dynamic_projection(condition)
        memory = memory + self.condition_lead_embedding.unsqueeze(0) + global_context.unsqueeze(1)
        return self.encoder(memory), global_context


class ConditionalTrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        condition_layers: int = 4,
        denoiser_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.condition_encoder = TrajectoryConditionEncoder(
            d_model=d_model, nhead=nhead, layers=condition_layers, dropout=dropout
        )
        self.noisy_projection = nn.Linear(1, d_model)
        self.target_lead_embedding = nn.Parameter(torch.randn(TARGET_LENGTH, d_model) * 0.02)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=denoiser_layers, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, 1)

    def forward(
        self,
        noisy_target: Tensor,
        diffusion_step: Tensor,
        condition: Tensor,
        static: Tensor,
        veg: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        memory, global_context = self.condition_encoder(condition, static, veg, time_features)
        step = self.diffusion_time_mlp(sinusoidal_embedding(diffusion_step, self.d_model))
        target = self.noisy_projection(noisy_target.unsqueeze(-1))
        target = target + self.target_lead_embedding.unsqueeze(0)
        target = target + global_context.unsqueeze(1) + step.unsqueeze(1)
        return self.output(self.decoder(target, memory)).squeeze(-1)


class GaussianTrajectoryModel(nn.Module):
    """Independent heteroscedastic Gaussian trajectory baseline."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        condition_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.condition_encoder = TrajectoryConditionEncoder(
            d_model=d_model, nhead=nhead, layers=condition_layers, dropout=dropout
        )
        self.target_queries = nn.Parameter(torch.randn(TARGET_LENGTH, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, 2)

    def forward(self, condition: Tensor, static: Tensor, veg: Tensor, time_features: Tensor) -> Tuple[Tensor, Tensor]:
        memory, global_context = self.condition_encoder(condition, static, veg, time_features)
        queries = self.target_queries.unsqueeze(0).expand(condition.shape[0], -1, -1)
        decoded = self.decoder(queries + global_context.unsqueeze(1), memory)
        out = self.output(decoded)
        mean = out[..., 0]
        log_scale = torch.clamp(out[..., 1], -5.0, 3.0)
        return mean, log_scale


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 1.0e-5, 0.999).float()


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, clean: Tensor, step: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        abar = self.alpha_bar[step].unsqueeze(1)
        noisy = torch.sqrt(abar) * clean + torch.sqrt(1.0 - abar) * noise
        return noisy, noise


def masked_diffusion_loss(predicted_noise: Tensor, noise: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (((predicted_noise - noise) ** 2) * mask).sum() / denom


def masked_gaussian_nll(mean: Tensor, log_scale: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    inv_var = torch.exp(-2.0 * log_scale)
    nll = 0.5 * (target - mean) ** 2 * inv_var + log_scale
    return (nll * mask).sum() / torch.clamp(mask.sum(), min=1.0)


@torch.no_grad()
def ddim_sample(
    model: ConditionalTrajectoryDenoiser,
    schedule: DiffusionSchedule,
    batch: Mapping[str, Tensor],
    members: int = 50,
    steps: int = 50,
    member_chunk_size: int = 10,
) -> Tensor:
    device = batch["condition"].device
    bsz = int(batch["condition"].shape[0])
    outputs: List[Tensor] = []
    schedule_steps = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    for member_start in range(0, int(members), max(1, int(member_chunk_size))):
        chunk = min(max(1, int(member_chunk_size)), int(members) - member_start)
        repeated = {
            key: batch[key].repeat_interleave(chunk, dim=0)
            for key in ("condition", "static", "veg", "time_features")
        }
        current = torch.randn(bsz * chunk, TARGET_LENGTH, device=device)
        for pos, step_value in enumerate(schedule_steps):
            step = torch.full((bsz * chunk,), int(step_value.item()), device=device, dtype=torch.long)
            eps = model(
                current,
                step,
                repeated["condition"],
                repeated["static"],
                repeated["veg"],
                repeated["time_features"],
            )
            abar = schedule.alpha_bar[step_value]
            clean = (current - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
            if pos + 1 == len(schedule_steps):
                current = clean
            else:
                next_abar = schedule.alpha_bar[schedule_steps[pos + 1]]
                current = torch.sqrt(next_abar) * clean + torch.sqrt(1.0 - next_abar) * eps
        outputs.append(current.reshape(bsz, chunk, TARGET_LENGTH))
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def gaussian_sample(model: GaussianTrajectoryModel, batch: Mapping[str, Tensor], members: int = 50) -> Tensor:
    mean, log_scale = model(batch["condition"], batch["static"], batch["veg"], batch["time_features"])
    noise = torch.randn(mean.shape[0], int(members), mean.shape[1], device=mean.device)
    return mean.unsqueeze(1) + torch.exp(log_scale).unsqueeze(1) * noise


def ensemble_crps_numpy(samples: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    first = np.mean(np.abs(samples - target[:, None, :]), axis=1)
    pair = np.mean(np.abs(samples[:, :, None, :] - samples[:, None, :, :]), axis=(1, 2))
    point = first - 0.5 * pair
    point = np.where(mask, point, np.nan)
    by_lead = np.nanmean(point, axis=0)
    return float(np.nanmean(point)), by_lead.astype(np.float64)


def visibility_class_probabilities(samples_m: np.ndarray) -> np.ndarray:
    samples_m = np.asarray(samples_m)
    fog = np.mean(samples_m < 500.0, axis=1)
    mist = np.mean((samples_m >= 500.0) & (samples_m < 1000.0), axis=1)
    clear = np.mean(samples_m >= 1000.0, axis=1)
    probs = np.stack([fog, mist, clear], axis=-1).astype(np.float32)
    total = probs.sum(axis=-1, keepdims=True)
    return probs / np.where(total > 0, total, 1.0)


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def read_dataset_config(data_dir: str | Path) -> Dict[str, object]:
    path = Path(data_dir) / "dataset_build_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config.get("dynamic_feature_order", [])) != DYNAMIC_FEATURE_ORDER:
        raise ValueError("Dataset dynamic_feature_order does not match the trajectory model contract")
    if tuple(config.get("condition_leads", [])) != CONDITION_LEADS:
        raise ValueError("Dataset condition leads do not match 0-48 h")
    if tuple(config.get("target_leads", [])) != TARGET_LEADS:
        raise ValueError("Dataset target leads do not match 12-48 h")
    if config.get("canonical_unit_policy") != PM_UNIT_POLICY_VERSION:
        raise ValueError("Dataset canonical unit policy is missing or stale")
    if config.get("pm_qc_policy") != PM_QC_POLICY_VERSION:
        raise ValueError("Dataset PM QC policy is missing or stale")
    return config


def model_from_config(config: Mapping[str, object]) -> nn.Module:
    model_type = str(config.get("model_type", "diffusion"))
    kwargs = dict(
        d_model=int(config.get("d_model", 128)),
        nhead=int(config.get("nhead", 8)),
        condition_layers=int(config.get("condition_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    if model_type == "diffusion":
        return ConditionalTrajectoryDenoiser(
            **kwargs, denoiser_layers=int(config.get("decoder_layers", 4))
        )
    if model_type == "gaussian":
        return GaussianTrajectoryModel(
            **kwargs, decoder_layers=int(config.get("decoder_layers", 4))
        )
    raise ValueError(f"Unknown model_type={model_type!r}")
