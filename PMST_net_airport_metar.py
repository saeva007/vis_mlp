#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Mapping, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from airport_visibility_common import (
    DYNAMIC_FEATURE_ORDER,
    EXTRA_FEATURE_DIM,
    LOCAL_TIME_OFFSET_HOURS,
    WINDOW_SIZE,
    airport_model_config,
    build_airport_model,
    get_dynamic_log_indices,
    predict_classes_from_probs,
    raw_rows_to_continuous_matrix,
    visibility_to_classes,
)


BASE_PATH = "/public/home/putianshu/vis_mlp"
DEFAULT_DATA_DIR = os.path.join(BASE_PATH, "ml_dataset_airport_metar_2025_12h")
DEFAULT_CKPT_DIR = os.path.join(BASE_PATH, "checkpoints")


class AirportVisibilityDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_raw: np.ndarray,
        indices: np.ndarray,
        scaler,
        window_size: int,
        dyn_vars_count: int,
    ):
        self.x_path = x_path
        self.y_raw_all = np.asarray(y_raw, dtype=np.float32)
        self.y_cls_all = visibility_to_classes(self.y_raw_all)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scaler = scaler
        self.window_size = int(window_size)
        self.dyn_vars_count = int(dyn_vars_count)
        self.split_dyn = self.window_size * self.dyn_vars_count
        self.log_indices = get_dynamic_log_indices(DYNAMIC_FEATURE_ORDER)
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for t in range(self.window_size):
            for idx in self.log_indices:
                self.log_mask[t * self.dyn_vars_count + int(idx)] = True
        self.center = getattr(scaler, "center_", None)
        self.scale = getattr(scaler, "scale_", None)
        self.x_mmap = None

    def __len__(self):
        return int(len(self.indices))

    def _transform_row(self, row: np.ndarray) -> np.ndarray:
        dyn = row[: self.split_dyn].astype(np.float32, copy=True)
        dyn[self.log_mask] = np.log1p(np.maximum(dyn[self.log_mask], 0.0))
        station_idx = row[self.split_dyn : self.split_dyn + 1].astype(np.float32, copy=True)
        extra = row[self.split_dyn + 1 :].astype(np.float32, copy=True)
        cont = np.concatenate([dyn, extra]).astype(np.float32)
        if self.center is not None and self.scale is not None:
            cont = (cont - self.center) / (self.scale + 1e-6)
        cont = np.clip(cont, -10.0, 10.0)
        final = np.concatenate([cont[: self.split_dyn], station_idx, cont[self.split_dyn :]])
        return np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def __getitem__(self, item):
        if self.x_mmap is None:
            self.x_mmap = np.load(self.x_path, mmap_mode="r")
        real_idx = int(self.indices[item])
        row = self.x_mmap[real_idx]
        x = self._transform_row(row)
        y_raw = float(max(self.y_raw_all[real_idx], 0.0))
        y_cls = int(self.y_cls_all[real_idx])
        y_reg = np.float32(np.log1p(y_raw))
        return (
            torch.from_numpy(x).float(),
            torch.tensor(y_cls, dtype=torch.long),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_raw, dtype=torch.float32),
        )


class AirportVisibilityLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        low_vis_pos_weight: torch.Tensor,
        alpha_binary: float = 0.5,
        alpha_reg: float = 0.05,
        alpha_clear_margin: float = 2.0,
        clear_margin: float = 0.25,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.register_buffer("low_vis_pos_weight", low_vis_pos_weight.float())
        self.alpha_binary = float(alpha_binary)
        self.alpha_reg = float(alpha_reg)
        self.alpha_clear_margin = float(alpha_clear_margin)
        self.clear_margin = float(clear_margin)

    def forward(
        self,
        logits: torch.Tensor,
        reg_pred: torch.Tensor,
        low_vis_logit: torch.Tensor,
        y_cls: torch.Tensor,
        y_reg: torch.Tensor,
        y_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = F.cross_entropy(logits, y_cls, weight=self.class_weights)
        low_target = (y_cls < 2).float()
        bce = F.binary_cross_entropy_with_logits(
            low_vis_logit.squeeze(-1),
            low_target,
            pos_weight=self.low_vis_pos_weight,
        )
        reg = F.smooth_l1_loss(reg_pred.squeeze(-1), y_reg)
        probs = F.softmax(logits, dim=1)
        clear_mask = y_cls == 2
        if torch.any(clear_mask):
            low_prob_clear = probs[clear_mask, 0] + probs[clear_mask, 1]
            clear_margin = torch.relu(low_prob_clear - self.clear_margin).mean()
        else:
            clear_margin = torch.zeros((), device=logits.device)
        loss = (
            ce
            + self.alpha_binary * bce
            + self.alpha_reg * reg
            + self.alpha_clear_margin * clear_margin
        )
        return loss, {
            "ce": float(ce.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "reg": float(reg.detach().cpu()),
            "clear_margin": float(clear_margin.detach().cpu()),
        }


def load_metadata(data_dir: str) -> Dict:
    path = os.path.join(data_dir, "dataset_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split(y_cls: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in (0, 1, 2):
        idx = np.where(y_cls == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * float(val_ratio)))
        if len(idx) > 0 and float(val_ratio) > 0.0:
            n_val = max(1, n_val)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def fit_or_load_scaler(
    x_path: str,
    scaler_path: str,
    window_size: int,
    dyn_vars_count: int,
    max_samples: int,
    seed: int,
    reuse_scaler: bool,
):
    if reuse_scaler and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[Scaler] loaded: {scaler_path}", flush=True)
        return scaler

    x_m = np.load(x_path, mmap_mode="r")
    n_total = len(x_m)
    rng = np.random.default_rng(seed)
    if n_total > max_samples:
        sample_idx = rng.choice(n_total, size=max_samples, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n_total)
    print(f"[Scaler] fitting on {len(sample_idx)} rows...", flush=True)
    sample_rows = x_m[sample_idx].astype(np.float32)
    cont = raw_rows_to_continuous_matrix(sample_rows, window_size, dyn_vars_count)
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(cont)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Scaler] saved: {scaler_path}", flush=True)
    return scaler


def class_weights_from_labels(y_cls: np.ndarray, indices: np.ndarray, device: torch.device):
    labels = y_cls[indices]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (3.0 * counts)
    weights = np.clip(weights, 0.5, 6.0)
    low_pos = np.maximum((labels < 2).sum(), 1)
    low_neg = np.maximum((labels == 2).sum(), 1)
    pos_weight = np.array([low_neg / low_pos], dtype=np.float32)
    return (
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(pos_weight, dtype=torch.float32, device=device),
        counts,
    )


def make_weighted_sampler(y_cls: np.ndarray, train_idx: np.ndarray) -> WeightedRandomSampler:
    labels = y_cls[train_idx]
    counts = np.maximum(np.bincount(labels, minlength=3).astype(np.float32), 1.0)
    class_weight = counts.sum() / counts
    class_weight[0] *= 1.5
    class_weight[1] *= 1.2
    sample_weight = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )


def compute_metrics_from_predictions(
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = max(len(y_cls), 1)
    out["accuracy"] = float((pred == y_cls).sum() / total)
    for cls, name in ((0, "fog"), (1, "mist"), (2, "clear")):
        tp = int(((pred == cls) & (y_cls == cls)).sum())
        fp = int(((pred == cls) & (y_cls != cls)).sum())
        fn = int(((pred != cls) & (y_cls == cls)).sum())
        out[f"{name}_precision"] = float(tp / max(tp + fp, 1))
        out[f"{name}_recall"] = float(tp / max(tp + fn, 1))

    true_low = y_cls < 2
    pred_low = pred < 2
    out["low_vis_precision"] = float(((true_low & pred_low).sum()) / max(pred_low.sum(), 1))
    out["low_vis_recall"] = float(((true_low & pred_low).sum()) / max(true_low.sum(), 1))
    clear = y_cls == 2
    out["low_vis_fpr"] = float(((clear & pred_low).sum()) / max(clear.sum(), 1))
    true_500 = y_raw < 500.0
    true_1000 = y_raw < 1000.0
    out["recall_500"] = float(((true_500) & (pred == 0)).sum() / max(true_500.sum(), 1))
    out["recall_1000"] = float(((true_1000) & (pred < 2)).sum() / max(true_1000.sum(), 1))
    return out


def target_score(metrics: Mapping[str, float]) -> float:
    score = (
        min(metrics["recall_500"] / 0.65, 1.0) * 0.30
        + min(metrics["recall_1000"] / 0.75, 1.0) * 0.25
        + min(metrics["accuracy"] / 0.95, 1.0) * 0.20
        + min(metrics["low_vis_precision"] / 0.20, 1.0) * 0.15
        + min(metrics["mist_recall"] / 0.20, 1.0) * 0.10
    )
    return float(score - 0.05 * metrics["low_vis_fpr"])


def search_thresholds(
    probabilities: np.ndarray,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    fog_min: float = 0.20,
    fog_max: float = 0.85,
    mist_min: float = 0.20,
    mist_max: float = 0.85,
    step: float = 0.03,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_score = -1e9
    best_thresholds = {"fog": 0.50, "mist": 0.50}
    best_metrics = compute_metrics_from_predictions(
        y_cls, y_raw, predict_classes_from_probs(probabilities, best_thresholds)
    )
    fog_grid = np.arange(fog_min, fog_max + 1e-6, step)
    mist_grid = np.arange(mist_min, mist_max + 1e-6, step)
    for fog_th in fog_grid:
        for mist_th in mist_grid:
            thresholds = {"fog": float(fog_th), "mist": float(mist_th)}
            pred = predict_classes_from_probs(probabilities, thresholds)
            metrics = compute_metrics_from_predictions(y_cls, y_raw, pred)
            score = target_score(metrics)
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                best_metrics = metrics
    best_metrics["target_score"] = float(best_score)
    return best_thresholds, best_metrics


@torch.no_grad()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    y_cls_all = []
    y_raw_all = []
    temp = max(float(temperature), 1e-6)
    for x, y_cls, _y_reg, y_raw in loader:
        x = x.to(device, non_blocking=True)
        logits, _reg, _low = model(x)
        probs = F.softmax(logits / temp, dim=1)
        probs_all.append(probs.detach().cpu().numpy())
        y_cls_all.append(y_cls.numpy())
        y_raw_all.append(y_raw.numpy())
    return (
        np.concatenate(probs_all, axis=0),
        np.concatenate(y_cls_all, axis=0),
        np.concatenate(y_raw_all, axis=0),
    )


def save_training_artifacts(
    model: nn.Module,
    checkpoint_path: str,
    preprocessor_path: str,
    scaler,
    metadata: Mapping,
    model_config: Mapping,
    train_config: Mapping,
    thresholds: Mapping[str, float],
    metrics: Optional[Mapping[str, float]],
) -> None:
    station_order = [str(s) for s in metadata["station_order"]]
    station_to_idx = {name: i for i, name in enumerate(station_order)}
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "model_state_dict": state,
        "model_config": dict(model_config),
        "config": dict(model_config),
        "train_config": dict(train_config),
        "thresholds": dict(thresholds),
        "temperature": 1.0,
        "metrics": dict(metrics or {}),
        "station_order": station_order,
        "station_to_idx": station_to_idx,
        "dynamic_feature_order": DYNAMIC_FEATURE_ORDER,
        "fill_values": dict(metadata.get("fill_values", {})),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    torch.save(payload, checkpoint_path)

    preprocessor = {
        "scaler": scaler,
        "model_type": "airport_pmst",
        "model_config": dict(model_config),
        "thresholds": dict(thresholds),
        "temperature": 1.0,
        "station_order": station_order,
        "station_to_idx": station_to_idx,
        "dynamic_feature_order": DYNAMIC_FEATURE_ORDER,
        "fill_values": dict(metadata.get("fill_values", {})),
        "window_size": int(model_config["window_size"]),
        "dyn_vars_count": int(model_config["dyn_vars_count"]),
        "extra_feature_dim": int(model_config["extra_feat_dim"]),
        "local_time_offset_hours": float(
            train_config.get("local_time_offset_hours", LOCAL_TIME_OFFSET_HOURS)
        ),
        "use_source_zenith": bool(train_config.get("use_source_zenith", False)),
    }
    joblib.dump(preprocessor, preprocessor_path)


def train(args) -> None:
    os.makedirs(args.save_ckpt_dir, exist_ok=True)
    metadata = load_metadata(args.data_dir)
    x_path = os.path.join(args.data_dir, "X_train.npy")
    y_path = os.path.join(args.data_dir, "y_train.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Missing X_train.npy/y_train.npy in {args.data_dir}; run s2_data_airport_metar.py first."
        )

    y_raw = np.load(y_path).astype(np.float32)
    y_cls = visibility_to_classes(y_raw)
    dyn_vars_count = int(metadata.get("dyn_vars_count", len(DYNAMIC_FEATURE_ORDER)))
    window_size = int(metadata.get("window_size", args.window_size))
    x_shape = np.load(x_path, mmap_mode="r").shape
    expected_dim = window_size * dyn_vars_count + 1 + int(metadata.get("extra_feature_dim", EXTRA_FEATURE_DIM))
    if x_shape[1] != expected_dim:
        raise ValueError(f"X shape mismatch: got {x_shape}, expected feature dim {expected_dim}")

    scaler_path = os.path.join(
        args.save_ckpt_dir,
        f"{args.exp_id}_airport_scaler_w{window_size}_dyn{dyn_vars_count}.pkl",
    )
    preprocessor_path = os.path.join(args.save_ckpt_dir, f"{args.exp_id}_airport_preprocessor.pkl")
    scaler = fit_or_load_scaler(
        x_path=x_path,
        scaler_path=scaler_path,
        window_size=window_size,
        dyn_vars_count=dyn_vars_count,
        max_samples=args.scaler_max_samples,
        seed=args.seed,
        reuse_scaler=args.reuse_scaler,
    )

    train_idx, val_idx = stratified_split(y_cls, args.val_ratio, args.seed)
    print(
        f"[Data] total={len(y_raw)}, train={len(train_idx)}, val={len(val_idx)}, "
        f"class_counts={np.bincount(y_cls, minlength=3).tolist()}",
        flush=True,
    )
    if len(train_idx) == 0:
        raise ValueError("No training samples after split")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_config = airport_model_config(
        station_count=len(metadata["station_order"]),
        window_size=window_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        station_emb_dim=args.station_emb_dim,
    )
    model = build_airport_model(model_config).to(device)

    class_weights, low_vis_pos_weight, train_counts = class_weights_from_labels(
        y_cls, train_idx, device
    )
    print(
        f"[Loss] train class counts={train_counts.tolist()}, "
        f"class_weights={class_weights.detach().cpu().numpy().round(3).tolist()}, "
        f"low_vis_pos_weight={float(low_vis_pos_weight.item()):.3f}",
        flush=True,
    )
    loss_fn = AirportVisibilityLoss(
        class_weights=class_weights,
        low_vis_pos_weight=low_vis_pos_weight,
        alpha_binary=args.alpha_binary,
        alpha_reg=args.alpha_reg,
        alpha_clear_margin=args.alpha_clear_margin,
        clear_margin=args.clear_margin,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )

    train_ds = AirportVisibilityDataset(
        x_path, y_raw, train_idx, scaler, window_size, dyn_vars_count
    )
    val_ds = AirportVisibilityDataset(
        x_path, y_raw, val_idx, scaler, window_size, dyn_vars_count
    ) if len(val_idx) else None
    sampler = make_weighted_sampler(y_cls, train_idx) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    ) if val_ds is not None else None

    train_config = {
        "exp_id": args.exp_id,
        "data_dir": args.data_dir,
        "save_ckpt_dir": args.save_ckpt_dir,
        "scaler_path": scaler_path,
        "preprocessor_path": preprocessor_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "local_time_offset_hours": float(metadata.get("local_time_offset_hours", LOCAL_TIME_OFFSET_HOURS)),
        "use_source_zenith": False,
    }

    best_score = -1e9
    best_thresholds = {"fog": 0.50, "mist": 0.50}
    best_metrics: Dict[str, float] = {}
    latest_path = os.path.join(args.save_ckpt_dir, f"{args.exp_id}_latest.pt")
    best_path = os.path.join(args.save_ckpt_dir, f"{args.exp_id}_best_score.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_parts = {"ce": 0.0, "bce": 0.0, "reg": 0.0, "clear_margin": 0.0}
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, yb, yreg, yraw) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yreg = yreg.to(device, non_blocking=True)
            yraw = yraw.to(device, non_blocking=True)
            logits, reg_pred, low_logit = model(x)
            loss, parts = loss_fn(logits, reg_pred, low_logit, yb, yreg, yraw)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            bs = int(x.size(0))
            running_loss += float(loss.detach().cpu()) * bs
            for key in running_parts:
                running_parts[key] += parts[key] * bs
            n_seen += bs

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)
        part_msg = ", ".join(
            f"{k}={running_parts[k] / max(n_seen, 1):.4f}" for k in running_parts
        )

        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            probs, val_y_cls, val_y_raw = collect_probabilities(model, val_loader, device)
            thresholds, metrics = search_thresholds(
                probs,
                val_y_cls,
                val_y_raw,
                fog_min=args.fog_threshold_min,
                fog_max=args.fog_threshold_max,
                mist_min=args.mist_threshold_min,
                mist_max=args.mist_threshold_max,
                step=args.threshold_step,
            )
            score = float(metrics["target_score"])
            print(
                f"[Epoch {epoch:03d}] loss={train_loss:.4f} ({part_msg}) "
                f"score={score:.4f} acc={metrics['accuracy']:.3f} "
                f"R500={metrics['recall_500']:.3f} R1000={metrics['recall_1000']:.3f} "
                f"lowP={metrics['low_vis_precision']:.3f} lowFPR={metrics['low_vis_fpr']:.3f} "
                f"th=({thresholds['fog']:.2f},{thresholds['mist']:.2f}) "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                best_metrics = metrics
                save_training_artifacts(
                    model,
                    best_path,
                    preprocessor_path,
                    scaler,
                    metadata,
                    model_config,
                    train_config,
                    best_thresholds,
                    best_metrics,
                )
                print(f"[Save] best -> {best_path}", flush=True)
        else:
            print(
                f"[Epoch {epoch:03d}] loss={train_loss:.4f} ({part_msg}) "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )

        save_training_artifacts(
            model,
            latest_path,
            preprocessor_path,
            scaler,
            metadata,
            model_config,
            train_config,
            best_thresholds,
            best_metrics,
        )

    if best_score <= -1e8:
        save_training_artifacts(
            model,
            best_path,
            preprocessor_path,
            scaler,
            metadata,
            model_config,
            train_config,
            best_thresholds,
            best_metrics,
        )
    print("[Done] training finished", flush=True)
    print(f"  best checkpoint: {best_path}", flush=True)
    print(f"  preprocessor:    {preprocessor_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AirportPMSTNet on METAR visibility data.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--exp-id", default="airport_metar_2025")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--station-emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler-max-samples", type=int, default=200000)
    parser.add_argument("--reuse-scaler", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true", default=True)
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-binary", type=float, default=0.5)
    parser.add_argument("--alpha-reg", type=float, default=0.05)
    parser.add_argument("--alpha-clear-margin", type=float, default=2.0)
    parser.add_argument("--clear-margin", type=float, default=0.25)
    parser.add_argument("--fog-threshold-min", type=float, default=0.20)
    parser.add_argument("--fog-threshold-max", type=float, default=0.85)
    parser.add_argument("--mist-threshold-min", type=float, default=0.20)
    parser.add_argument("--mist-threshold-max", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.03)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
