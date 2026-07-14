#!/usr/bin/env python3
"""Train the standalone conditional trajectory diffusion or Gaussian baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from lowvis_trajectory_diffusion import (
    DYNAMIC_FEATURE_ORDER,
    DiffusionSchedule,
    LowVisTrajectoryDataset,
    TrajectoryScaler,
    ddim_sample,
    gaussian_sample,
    masked_diffusion_loss,
    masked_gaussian_nll,
    min_snr_weights,
    model_from_config,
    move_batch,
    read_dataset_config,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=("diffusion", "gaussian"), default="diffusion")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-id", default=f"exp_{int(time.time())}_lowvis_trajectory_diffusion")
    p.add_argument("--checkpoint-dir", default="/public/home/putianshu/vis_mlp/checkpoints")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--condition-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--condition-token-version", type=int, choices=(1, 2), default=2)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--ddim-clip-x0", type=float, default=6.0)
    p.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument("--stratified-timesteps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-interval", type=int, default=2_000)
    p.add_argument("--val-monitor-size", type=int, default=512)
    p.add_argument("--val-members", type=int, default=20)
    p.add_argument("--val-ddim-steps", type=int, default=20)
    p.add_argument("--validation-seed", type=int, default=20260703)
    p.add_argument("--lowvis-selection-weight", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scaler-max-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def setup_distributed(device_arg: str) -> Tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg == "cpu" or (device_arg == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        backend = "gloo"
    else:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    if world > 1 and not dist.is_initialized():
        timeout_seconds = int(os.environ.get("LOWVIS_TRAJ_DIST_TIMEOUT", "3600"))
        dist.init_process_group(
            backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_seconds)
        )
    return rank, world, local_rank, device


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


class ExponentialMovingAverage:
    """Small device-local EMA used for validation and exported inference weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # Warm up the averaging horizon so early checkpoints are not biased
        # toward the random initialization.
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, value in model.state_dict().items():
            source = value.detach()
            if torch.is_floating_point(source):
                self.shadow[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(source)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        if self.backup:
            raise RuntimeError("EMA weights are already applied")
        self.backup = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}


def cosine_learning_rate(
    update_index: int,
    max_steps: int,
    base_learning_rate: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    update = int(update_index) + 1
    warmup = max(0, min(int(warmup_steps), int(max_steps)))
    if warmup > 0 and update <= warmup:
        return float(base_learning_rate) * update / warmup
    progress = (update - warmup) / max(1, int(max_steps) - warmup)
    progress = min(max(progress, 0.0), 1.0)
    floor = min(max(float(min_lr_ratio), 0.0), 1.0)
    multiplier = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_learning_rate) * multiplier


def sample_diffusion_steps(
    batch_size: int, timesteps: int, device: torch.device, stratified: bool
) -> torch.Tensor:
    if not stratified:
        return torch.randint(0, timesteps, (batch_size,), device=device)
    # One draw from every equal-width timestep stratum lowers gradient variance
    # without changing the uniform timestep objective.
    positions = (torch.arange(batch_size, device=device, dtype=torch.float32) + torch.rand(1, device=device))
    steps = torch.floor(positions * float(timesteps) / float(batch_size)).long()
    steps = torch.clamp(steps, 0, timesteps - 1)
    return steps[torch.randperm(batch_size, device=device)]


def make_model_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model_type": args.model_type,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "condition_layers": args.condition_layers,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "condition_token_version": args.condition_token_version,
        "diffusion_steps": args.diffusion_steps,
        "ddim_clip_x0": args.ddim_clip_x0,
    }


def monitor_indices(data_dir: Path, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    vis = np.load(data_dir / "visibility_val.npy", mmap_mode="r")
    mask = np.load(data_dir / "target_mask_val.npy", mmap_mode="r")
    n = int(vis.shape[0])
    overall = np.linspace(0, n - 1, num=min(n, limit), dtype=np.int64)
    low = []
    chunk = 20_000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        values = np.asarray(vis[start:stop], dtype=np.float32)
        valid = np.asarray(mask[start:stop], dtype=bool) & np.isfinite(values)
        minima = np.min(np.where(valid, values, np.inf), axis=1)
        low.extend((np.where(minima < 1000.0)[0] + start).tolist())
    if len(low) > limit:
        positions = np.linspace(0, len(low) - 1, num=limit, dtype=np.int64)
        low_arr = np.asarray(low, dtype=np.int64)[positions]
    else:
        low_arr = np.asarray(low, dtype=np.int64)
    if len(low_arr) == 0:
        low_arr = overall.copy()
    return overall, low_arr


def normalized_to_log(samples: np.ndarray, scaler: TrajectoryScaler) -> np.ndarray:
    mean = np.asarray(scaler.target_mean, dtype=np.float32)
    scale = np.asarray(scaler.target_scale, dtype=np.float32)
    return samples * scale[None, None, :] + mean[None, None, :]


@torch.no_grad()
def validation_crps(
    model: torch.nn.Module,
    model_type: str,
    schedule: DiffusionSchedule,
    dataset: LowVisTrajectoryDataset,
    indices: Sequence[int],
    scaler: TrajectoryScaler,
    device: torch.device,
    batch_size: int,
    members: int,
    ddim_steps: int,
    seed: int,
    rank: int,
    world: int,
) -> float:
    model.eval()
    local_indices = np.asarray(indices, dtype=np.int64)[rank::world]
    loader = DataLoader(
        Subset(dataset, list(map(int, local_indices))),
        batch_size=min(batch_size, 64),
        shuffle=False,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(rank))
    local_sum = 0.0
    local_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if model_type == "diffusion":
            generated = ddim_sample(
                model, schedule, batch, members=members, steps=ddim_steps, generator=generator
            )
        else:
            generated = gaussian_sample(model, batch, members=members, generator=generator)
        samples = normalized_to_log(generated.cpu().numpy(), scaler).astype(np.float64)
        target = batch["target"].cpu().numpy().astype(np.float64)
        target = target * np.asarray(scaler.target_scale) + np.asarray(scaler.target_mean)
        mask = batch["target_mask"].cpu().numpy().astype(bool)
        first = np.mean(np.abs(samples - target[:, None, :]), axis=1)
        pair = np.mean(
            np.abs(samples[:, :, None, :] - samples[:, None, :, :]), axis=(1, 2)
        )
        point = first - 0.5 * pair
        local_sum += float(point[mask].sum())
        local_count += int(mask.sum())
    stats = torch.tensor([local_sum, float(local_count)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / torch.clamp(stats[1], min=1.0)).item())


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: TrajectoryScaler,
    args: argparse.Namespace,
    model_config: Mapping[str, object],
    dataset_config: Mapping[str, object],
    step: int,
    metrics: Mapping[str, float],
    ema_updates: int,
) -> None:
    state_model = unwrap_model(model)
    payload = {
        "model_state": state_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler": scaler.to_dict(),
        "model_config": dict(model_config),
        "training_config": vars(args),
        "dataset_config": dict(dataset_config),
        "feature_order": list(DYNAMIC_FEATURE_ORDER),
        "run_id": args.run_id,
        "step": int(step),
        "validation_metrics": dict(metrics),
        "git_revision": git_revision(Path(__file__).resolve().parent),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "weights_source": "ema" if args.ema_decay > 0 else "online",
        "ema_updates": int(ema_updates),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup_distributed(args.device)
    seed_everything(args.seed + rank)
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = read_dataset_config(data_dir)
    scaler_path = checkpoint_dir / f"trajectory_scaler_{args.run_id}.json"
    if rank == 0 and not scaler_path.exists():
        TrajectoryScaler.fit(data_dir, max_rows=args.scaler_max_rows).save(scaler_path)
    barrier(world)
    scaler = TrajectoryScaler.load(scaler_path)
    train_dataset = LowVisTrajectoryDataset(data_dir, "train", scaler)
    val_dataset = LowVisTrajectoryDataset(data_dir, "val", scaler)
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    model_config = make_model_config(args)
    model = model_from_config(model_config).to(device)
    schedule = DiffusionSchedule(args.diffusion_steps, ddim_clip_x0=args.ddim_clip_x0).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema = ExponentialMovingAverage(unwrap_model(model), args.ema_decay) if args.ema_decay > 0 else None

    overall_idx, low_idx = monitor_indices(data_dir, args.val_monitor_size)
    if rank == 0:
        run_config = {
            "run_id": args.run_id,
            "candidate_only": True,
            "model_config": model_config,
            "training_config": vars(args),
            "dataset_config_path": str(data_dir / "dataset_build_config.json"),
            "scaler_path": str(scaler_path),
            "checkpoint_path": str(checkpoint_dir / f"{args.run_id}_best.pt"),
            "world_size": int(world),
            "global_batch_size": int(world * args.batch_size),
        }
        (checkpoint_dir / f"{args.run_id}_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(json.dumps(run_config, indent=2), flush=True)
    step = 0
    epoch = 0
    best_score = float("inf")
    bad_validations = 0
    best_path = checkpoint_dir / f"{args.run_id}_best.pt"
    train_iter: Iterable[Mapping[str, torch.Tensor]]
    stop_training = False
    while step < args.max_steps and not stop_training:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for raw_batch in loader:
            if step >= args.max_steps:
                break
            model.train()
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            current_lr = cosine_learning_rate(
                step,
                args.max_steps,
                args.learning_rate,
                args.warmup_steps,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            if args.model_type == "diffusion":
                diffusion_step = sample_diffusion_steps(
                    int(batch["target"].shape[0]),
                    args.diffusion_steps,
                    device,
                    args.stratified_timesteps,
                )
                noisy, noise = schedule.q_sample(batch["target"], diffusion_step)
                predicted = model(
                    noisy,
                    diffusion_step,
                    batch["condition"],
                    batch["static"],
                    batch["veg"],
                    batch["time_features"],
                )
                sample_weight = min_snr_weights(schedule, diffusion_step, args.min_snr_gamma)
                loss = masked_diffusion_loss(
                    predicted, noise, batch["target_mask"], sample_weight=sample_weight
                )
            else:
                mean, log_scale = model(batch["condition"], batch["static"], batch["veg"], batch["time_features"])
                loss = masked_gaussian_nll(mean, log_scale, batch["target"], batch["target_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(unwrap_model(model))
            step += 1
            if rank == 0 and (step == 1 or step % 100 == 0):
                print(
                    f"[train] step={step}/{args.max_steps} loss={float(loss.item()):.6f} "
                    f"lr={current_lr:.3e}",
                    flush=True,
                )

            do_val = step % args.val_interval == 0 or step == args.max_steps
            if do_val:
                barrier(world)
                eval_model = unwrap_model(model)
                if ema is not None:
                    ema.apply(eval_model)
                try:
                    overall_crps = validation_crps(
                        eval_model,
                        args.model_type,
                        schedule,
                        val_dataset,
                        overall_idx,
                        scaler,
                        device,
                        args.batch_size,
                        args.val_members,
                        args.val_ddim_steps,
                        args.validation_seed,
                        rank,
                        world,
                    )
                    lowvis_crps = validation_crps(
                        eval_model,
                        args.model_type,
                        schedule,
                        val_dataset,
                        low_idx,
                        scaler,
                        device,
                        args.batch_size,
                        args.val_members,
                        args.val_ddim_steps,
                        args.validation_seed + 1,
                        rank,
                        world,
                    )
                    if rank == 0:
                        low_weight = min(max(float(args.lowvis_selection_weight), 0.0), 1.0)
                        score = (1.0 - low_weight) * overall_crps + low_weight * lowvis_crps
                        metrics = {
                            "overall_log_crps": float(overall_crps),
                            "lowvis_log_crps": float(lowvis_crps),
                            "lowvis_selection_weight": float(low_weight),
                            "selection_score": float(score),
                        }
                        print(f"[val] step={step} {json.dumps(metrics)}", flush=True)
                        if score < best_score:
                            best_score = score
                            bad_validations = 0
                            save_checkpoint(
                                best_path,
                                model,
                                optimizer,
                                scaler,
                                args,
                                model_config,
                                dataset_config,
                                step,
                                metrics,
                                ema.updates if ema is not None else 0,
                            )
                        else:
                            bad_validations += 1
                        stop_training = bad_validations >= args.patience
                finally:
                    if ema is not None:
                        ema.restore(eval_model)
                if world > 1:
                    flag = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int32)
                    dist.broadcast(flag, src=0)
                    stop_training = bool(flag.item())
                barrier(world)
            if stop_training:
                break
        epoch += 1

    if rank == 0 and not best_path.exists():
        eval_model = unwrap_model(model)
        if ema is not None:
            ema.apply(eval_model)
        try:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                args,
                model_config,
                dataset_config,
                step,
                {"selection_score": float("nan")},
                ema.updates if ema is not None else 0,
            )
        finally:
            if ema is not None:
                ema.restore(eval_model)
    barrier(world)
    if rank == 0:
        print(f"[done] best_checkpoint={best_path} best_score={best_score:g}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
