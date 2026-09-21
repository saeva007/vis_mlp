#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "A": {"name": "A_linear_frozen", "head": "linear", "frozen": True},
    "B": {"name": "B_mlp_frozen", "head": "mlp", "frozen": True},
    "C": {"name": "C_mlp_e2e", "head": "mlp", "frozen": False},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def visibility(path):
    values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if len(values) and float(np.nanmax(values)) >= 100.0:
        values = values / 1000.0
    return values


class GateDataset(Dataset):
    def __init__(self, cfg, source_cfg, split):
        self.data = Path(cfg["data_dir"])
        x_root = Path(os.environ.get("GATE_X_DIR", self.data))
        self.x_path = x_root / f"X_{split}.npy"
        self.v = visibility(self.data / f"y_{split}.npy")
        self.y = (self.v >= float(cfg["gate_vc_km"])).astype(np.float32)
        self.scaler = joblib.load(cfg["scaler"])
        self.x = None
        self.window = int(source_cfg["window_size"])
        self.dyn_vars = int(source_cfg["dyn_vars"])
        self.split_dyn = self.window * self.dyn_vars
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        log_indices = np.asarray(source_cfg["log1p_dyn_indices"], dtype=np.int64)
        for step in range(self.window):
            self.log_mask[step * self.dyn_vars + log_indices] = True

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode="r")
        idx = int(idx)
        row = self.x[idx]
        core = row[: self.split_dyn + 5].astype(np.float32)
        dynamic = core[: self.split_dyn]
        dynamic[self.log_mask] = np.log1p(np.maximum(dynamic[self.log_mask], 0.0))
        core = (core - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        vegetation = np.asarray([row[self.split_dyn + 5]], dtype=np.float32)
        engineered = row[self.split_dyn + 6 :].astype(np.float32)
        features = np.concatenate([
            np.clip(core, -10.0, 10.0),
            vegetation,
            np.clip(engineered, -10.0, 10.0),
        ])
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return (
            torch.from_numpy(features).float(),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.v[idx], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.int64),
        )


class PretrainedGateEncoder(nn.Module):
    def __init__(self, source_cfg, source_checkpoint, source_root):
        super().__init__()
        sys.path.insert(0, str(Path(source_root) / "scripts"))
        from r5_model import R5Model

        source = R5Model(source_cfg, "R5-Diffusion")
        checkpoint = torch.load(source_checkpoint, map_location="cpu")
        weights = checkpoint.get("ema_model") or checkpoint["model"]
        source.load_state_dict(weights, strict=True)
        self.dynamic_tokenizer = copy.deepcopy(source.dynamic_tokenizer)
        self.dynamic_encoder = copy.deepcopy(source.dynamic_encoder)
        self.context_encoder = copy.deepcopy(source.context_encoder)
        self.gate_query = copy.deepcopy(source.gate_query)
        self.window = int(source_cfg["window_size"])
        self.dyn_vars = int(source_cfg["dyn_vars"])
        self.weather_indices = source_cfg["weather_indices"]
        self.pm_indices = source_cfg["pm_indices"]
        del source, checkpoint, weights

    def forward(self, x):
        dynamic_end = self.window * self.dyn_vars
        dynamic = x[:, :dynamic_end].reshape(-1, self.window, self.dyn_vars)
        static = x[:, dynamic_end : dynamic_end + 5]
        vegetation = x[:, dynamic_end + 5].long()
        engineered = x[:, dynamic_end + 6 :]
        weather_tokens = self.dynamic_encoder(
            self.dynamic_tokenizer(dynamic[:, :, self.weather_indices])
        )
        memory = self.context_encoder(
            weather_tokens,
            static,
            vegetation,
            dynamic[:, :, self.pm_indices],
            engineered,
        )
        return self.gate_query(memory)


class GateProbe(nn.Module):
    def __init__(self, source_cfg, source_checkpoint, source_root, head_type, frozen):
        super().__init__()
        self.encoder = PretrainedGateEncoder(source_cfg, source_checkpoint, source_root)
        d_model = int(source_cfg["d_model"])
        if head_type == "linear":
            self.head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
        self.frozen = bool(frozen)
        if self.frozen:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                encoded = self.encoder(x)
        else:
            encoded = self.encoder(x)
        return self.head(encoded).squeeze(-1)


def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 4:
        raise RuntimeError(f"Gate study requires exactly 4 DCUs; WORLD_SIZE={world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, local_rank, world, torch.device(f"cuda:{local_rank}")


def loader(dataset, batch_size, workers, sampler, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
        ap = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probabilities))
        ap = float(average_precision_score(labels, probabilities))
    return {
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
        "AUROC": auroc,
        "AP": ap,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "Brier": float(brier_score_loss(labels, probabilities)),
    }


@torch.inference_mode()
def distributed_predict(model, dataset, cfg, run_dir, tag, rank, world, device):
    indices = np.arange(rank, len(dataset), world, dtype=np.int64)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    data_loader = loader(
        subset,
        int(cfg["eval_batch_size_per_dcu"]),
        int(cfg["num_workers_per_rank"]),
        sampler=None,
        shuffle=False,
    )
    model.eval()
    all_prob, all_label, all_v, all_index = [], [], [], []
    loss_sum = 0.0
    count = 0
    for features, labels, vis, sample_index in data_loader:
        features = features.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(features)
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, labels_device, reduction="sum").item())
        count += len(labels)
        all_prob.append(torch.sigmoid(logits).cpu().numpy())
        all_label.append(labels.numpy())
        all_v.append(vis.numpy())
        all_index.append(sample_index.numpy())
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{tag}_rank{rank}.npz"
    np.savez_compressed(
        shard_path,
        probability=np.concatenate(all_prob),
        label=np.concatenate(all_label),
        visibility=np.concatenate(all_v),
        index=np.concatenate(all_index),
        loss_sum=np.asarray([loss_sum]),
        count=np.asarray([count]),
    )
    dist.barrier()
    result = None
    if rank == 0:
        shards = [np.load(shard_dir / f"{tag}_rank{r}.npz") for r in range(world)]
        order = np.argsort(np.concatenate([s["index"] for s in shards]))
        result = {
            "probability": np.concatenate([s["probability"] for s in shards])[order],
            "label": np.concatenate([s["label"] for s in shards])[order],
            "visibility": np.concatenate([s["visibility"] for s in shards])[order],
            "loss": float(sum(float(s["loss_sum"][0]) for s in shards) / sum(int(s["count"][0]) for s in shards)),
        }
        for s in shards:
            s.close()
        for r in range(world):
            (shard_dir / f"{tag}_rank{r}.npz").unlink()
    dist.barrier()
    return result


def visibility_bins(visibility_values, probabilities):
    v = np.asarray(visibility_values)
    p = np.asarray(probabilities)
    definitions = [
        ("<25", v < 25.0),
        ("25-27", (v >= 25.0) & (v < 27.0)),
        ("27-28.4", (v >= 27.0) & (v < 28.4)),
        ("28.4-29.4", (v >= 28.4) & (v < 29.4)),
        ("29.4-30", (v >= 29.4) & (v < 30.0)),
        ("30 km", np.isclose(v, 30.0, atol=1e-6)),
    ]
    output = {}
    for name, mask in definitions:
        values = p[mask]
        output[name] = {
            "n": int(mask.sum()),
            "mean_p_gate": float(np.mean(values)) if len(values) else None,
            "median_p_gate": float(np.median(values)) if len(values) else None,
        }
    return output


def test_report(result, cfg):
    probability = result["probability"]
    labels = result["label"]
    vis = result["visibility"]
    margins = {}
    for margin in cfg["test_margins_km"]:
        margin = float(margin)
        keep = np.ones(len(vis), dtype=bool) if margin == 0.0 else np.abs(vis - cfg["gate_vc_km"]) >= margin
        key = "all" if margin == 0.0 else f"exclude_pm_{margin:g}km"
        margins[key] = binary_metrics(labels[keep], probability[keep], cfg["decision_threshold"])
    return {
        "bce": result["loss"],
        "decision_threshold": cfg["decision_threshold"],
        "margin_metrics": margins,
        "visibility_bins": visibility_bins(vis, probability),
    }


def learning_rate_scale(step, max_steps, warmup):
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    args = parser.parse_args()
    cfg = read_json(ROOT / "configs" / "gate_study.json")
    definition = EXPERIMENTS[args.experiment]
    run_dir = Path(cfg["project_root"]) / "runs" / definition["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    rank, local_rank, world, device = init_distributed()

    seed = int(cfg["seed"]) + ord(args.experiment)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    source_cfg = read_json(cfg["source_config"])
    train_set = GateDataset(cfg, source_cfg, "train")
    val_set = GateDataset(cfg, source_cfg, "val")
    test_set = GateDataset(cfg, source_cfg, "test")
    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    train_loader = loader(
        train_set,
        int(cfg["batch_size_per_dcu"]),
        int(cfg["num_workers_per_rank"]),
        train_sampler,
    )

    model = GateProbe(
        source_cfg,
        cfg["source_checkpoint"],
        cfg["source_root"],
        definition["head"],
        definition["frozen"],
    ).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    raw = model.module
    if definition["frozen"]:
        groups = [{"params": raw.head.parameters(), "lr": cfg["frozen_head_lr"]}]
    else:
        groups = [
            {"params": raw.encoder.parameters(), "lr": cfg["finetune_encoder_lr"]},
            {"params": raw.head.parameters(), "lr": cfg["finetune_head_lr"]},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))
    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    if rank == 0:
        atomic_json(run_dir / "run_contract.json", {
            "experiment": args.experiment,
            **definition,
            "source_checkpoint": cfg["source_checkpoint"],
            "source_weights": "ema_model",
            "gate_vc_km": cfg["gate_vc_km"],
            "world_size": world,
            "batch_size_per_dcu": cfg["batch_size_per_dcu"],
            "effective_batch_size": cfg["batch_size_per_dcu"] * world,
            "num_workers_total": cfg["num_workers_per_rank"] * world,
            "selection": "mean(val_AUROC, val_AP); fixed p>=0.5 for class metrics",
            "train_n": len(train_set),
            "val_n": len(val_set),
            "test_n": len(test_set),
        })

    best_score = -float("inf")
    best_step = 0
    bad_validations = 0
    max_steps = int(cfg["max_steps"])
    step = 0
    epoch = 0
    stop = False
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps and not stop:
        train_sampler.set_epoch(epoch)
        model.train()
        for features, labels, _, _ in train_loader:
            if step >= max_steps:
                break
            step += 1
            scale = learning_rate_scale(step, max_steps, int(cfg["warmup_steps"]))
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = initial_lr * scale
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                reduced = loss.detach().clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced /= world
                if rank == 0:
                    record = {
                        "step": step,
                        "epoch": epoch + step / max(len(train_loader), 1),
                        "train_bce": float(reduced.item()),
                        "lr": [g["lr"] for g in optimizer.param_groups],
                        "elapsed_seconds": time.time() - start_time,
                    }
                    with (run_dir / "training_history.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

            if step % int(cfg["validation_interval"]) == 0:
                val_result = distributed_predict(model, val_set, cfg, run_dir, f"val_{step:06d}", rank, world, device)
                decision = [False, best_score, best_step, bad_validations]
                if rank == 0:
                    metrics = binary_metrics(val_result["label"], val_result["probability"], cfg["decision_threshold"])
                    score = float(np.mean([metrics["AUROC"], metrics["AP"]]))
                    improved = score > best_score + 1e-8
                    if improved:
                        best_score = score
                        best_step = step
                        bad_validations = 0
                        atomic_torch(run_dir / "checkpoints" / "best.pt", {
                            "model": raw.state_dict(),
                            "experiment": args.experiment,
                            "definition": definition,
                            "step": step,
                            "epoch": epoch,
                            "config": cfg,
                            "source_config": source_cfg,
                            "source_checkpoint": cfg["source_checkpoint"],
                            "validation": {"bce": val_result["loss"], **metrics, "selection_score": score},
                        })
                    else:
                        bad_validations += 1
                    with (run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "step": step,
                            "bce": val_result["loss"],
                            **metrics,
                            "selection_score": score,
                            "improved": improved,
                        }) + "\n")
                    should_stop = step >= int(cfg["minimum_steps"]) and bad_validations >= int(cfg["early_stop_patience"])
                    decision = [should_stop, best_score, best_step, bad_validations]
                dist.broadcast_object_list(decision, src=0)
                stop, best_score, best_step, bad_validations = decision
                model.train()
                if stop:
                    break
        epoch += 1

    dist.barrier()
    best = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    raw.load_state_dict(best["model"], strict=True)
    test_result = distributed_predict(model, test_set, cfg, run_dir, "test", rank, world, device)
    if rank == 0:
        report = {
            "experiment": args.experiment,
            **definition,
            "best_step": int(best["step"]),
            "best_validation": best["validation"],
            "actual_stop_step": step,
            "stop_reason": "early_stopping" if stop else "max_steps",
            "wall_seconds": time.time() - start_time,
            "resource": {
                "nodes": 1,
                "DCUs": world,
                "CPUs": int(os.environ.get("SLURM_CPUS_PER_TASK", "32")),
                "workers_total": int(cfg["num_workers_per_rank"]) * world,
                "effective_batch_size": int(cfg["batch_size_per_dcu"]) * world,
            },
            "test": test_report(test_result, cfg),
        }
        atomic_json(run_dir / "test_results.json", report)
        atomic_json(run_dir / "complete.json", {"complete": True, "best_step": int(best["step"])})
        print(json.dumps({"status": "complete", "experiment": args.experiment, "best_step": best["step"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
