#!/usr/bin/env python3
"""Train the standalone conditional trajectory diffusion or Gaussian baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
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
    ensemble_crps_numpy,
    gaussian_sample,
    masked_diffusion_loss,
    masked_gaussian_nll,
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
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-interval", type=int, default=2_000)
    p.add_argument("--val-monitor-size", type=int, default=512)
    p.add_argument("--val-members", type=int, default=20)
    p.add_argument("--val-ddim-steps", type=int, default=20)
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
        dist.init_process_group(backend=backend, init_method="env://")
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


def make_model_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model_type": args.model_type,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "condition_layers": args.condition_layers,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "diffusion_steps": args.diffusion_steps,
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
) -> float:
    model.eval()
    loader = DataLoader(Subset(dataset, list(map(int, indices))), batch_size=min(batch_size, 64), shuffle=False)
    all_samples: List[np.ndarray] = []
    all_target: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if model_type == "diffusion":
            generated = ddim_sample(model, schedule, batch, members=members, steps=ddim_steps)
        else:
            generated = gaussian_sample(model, batch, members=members)
        all_samples.append(normalized_to_log(generated.cpu().numpy(), scaler))
        target = batch["target"].cpu().numpy()
        all_target.append(target * np.asarray(scaler.target_scale) + np.asarray(scaler.target_mean))
        all_mask.append(batch["target_mask"].cpu().numpy().astype(bool))
    samples = np.concatenate(all_samples, axis=0)
    target = np.concatenate(all_target, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    return ensemble_crps_numpy(samples, target, mask)[0]


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
) -> None:
    state_model = model.module if isinstance(model, DDP) else model
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
        drop_last=False,
    )
    model_config = make_model_config(args)
    model = model_from_config(model_config).to(device)
    schedule = DiffusionSchedule(args.diffusion_steps).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if rank == 0:
        overall_idx, low_idx = monitor_indices(data_dir, args.val_monitor_size)
        run_config = {
            "run_id": args.run_id,
            "candidate_only": True,
            "model_config": model_config,
            "training_config": vars(args),
            "dataset_config_path": str(data_dir / "dataset_build_config.json"),
            "scaler_path": str(scaler_path),
            "checkpoint_path": str(checkpoint_dir / f"{args.run_id}_best.pt"),
        }
        (checkpoint_dir / f"{args.run_id}_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(json.dumps(run_config, indent=2), flush=True)
    else:
        overall_idx = low_idx = np.empty(0, dtype=np.int64)

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
            if args.model_type == "diffusion":
                diffusion_step = torch.randint(0, args.diffusion_steps, (batch["target"].shape[0],), device=device)
                noisy, noise = schedule.q_sample(batch["target"], diffusion_step)
                predicted = model(
                    noisy,
                    diffusion_step,
                    batch["condition"],
                    batch["static"],
                    batch["veg"],
                    batch["time_features"],
                )
                loss = masked_diffusion_loss(predicted, noise, batch["target_mask"])
            else:
                mean, log_scale = model(batch["condition"], batch["static"], batch["veg"], batch["time_features"])
                loss = masked_gaussian_nll(mean, log_scale, batch["target"], batch["target_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if rank == 0 and (step == 1 or step % 100 == 0):
                print(f"[train] step={step}/{args.max_steps} loss={float(loss.item()):.6f}", flush=True)

            do_val = step % args.val_interval == 0 or step == args.max_steps
            if do_val:
                barrier(world)
                if rank == 0:
                    eval_model = model.module if isinstance(model, DDP) else model
                    overall_crps = validation_crps(
                        eval_model, args.model_type, schedule, val_dataset, overall_idx, scaler, device,
                        args.batch_size, args.val_members, args.val_ddim_steps,
                    )
                    lowvis_crps = validation_crps(
                        eval_model, args.model_type, schedule, val_dataset, low_idx, scaler, device,
                        args.batch_size, args.val_members, args.val_ddim_steps,
                    )
                    score = 0.5 * (overall_crps + lowvis_crps)
                    metrics = {
                        "overall_log_crps": float(overall_crps),
                        "lowvis_log_crps": float(lowvis_crps),
                        "selection_score": float(score),
                    }
                    print(f"[val] step={step} {json.dumps(metrics)}", flush=True)
                    if score < best_score:
                        best_score = score
                        bad_validations = 0
                        save_checkpoint(best_path, model, optimizer, scaler, args, model_config, dataset_config, step, metrics)
                    else:
                        bad_validations += 1
                    stop_training = bad_validations >= args.patience
                if world > 1:
                    flag = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int32)
                    dist.broadcast(flag, src=0)
                    stop_training = bool(flag.item())
                barrier(world)
            if stop_training:
                break
        epoch += 1

    if rank == 0 and not best_path.exists():
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
        )
    barrier(world)
    if rank == 0:
        print(f"[done] best_checkpoint={best_path} best_score={best_score:g}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
