#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from r5_model import R5CensoredFlowModel

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("R5-CensoredFlow",)


def config():
    return json.loads((ROOT / "configs" / "r5_censored_flow.json").read_text(encoding="utf-8"))


def contract(stage):
    return json.loads((ROOT / "contracts" / f"{stage}.json").read_text(encoding="utf-8"))


def visibility(path):
    values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if len(values) and float(np.nanmax(values)) >= 100.0:
        values = values / 1000.0
    return values


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class VisibilityDataset(Dataset):
    def __init__(self, cfg, stage, split, indices=None, x_path_override=None, local_rows=False):
        data = Path(cfg["data"][stage])
        x_root = Path(os.environ.get("R5_X_DIR", data))
        self.x_path = Path(x_path_override) if x_path_override else x_root / f"X_{split}.npy"
        self.v = visibility(data / f"y_{split}.npy")
        self.indices = np.arange(len(self.v), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.local_rows = bool(local_rows)
        self.scaler = joblib.load(cfg["scaler"][stage])
        self.x = None
        self.split_dyn = cfg["window_size"] * cfg["dyn_vars"]
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        for step in range(cfg["window_size"]):
            self.log_mask[step * cfg["dyn_vars"] + np.asarray(cfg["log1p_dyn_indices"])] = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode="r")
        index = int(self.indices[position])
        row = self.x[position if self.local_rows else index]
        core = row[: self.split_dyn + 5].astype(np.float32)
        core[: self.split_dyn] = np.where(
            self.log_mask, np.log1p(np.maximum(core[: self.split_dyn], 0)), core[: self.split_dyn]
        )
        core = (core - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        vegetation = np.asarray([row[self.split_dyn + 5]], dtype=np.float32)
        engineered = row[self.split_dyn + 6 :].astype(np.float32)
        x = np.concatenate([np.clip(core, -10, 10), vegetation, np.clip(engineered, -10, 10)])
        return torch.from_numpy(np.nan_to_num(x, nan=0.0)).float(), torch.tensor(self.v[index]), torch.tensor(index)


class BlockShuffleSampler(Sampler):
    def __init__(self, size, seed, block_size=65536):
        self.size, self.seed, self.block_size, self.epoch = int(size), int(seed), int(block_size), 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        blocks = np.arange((self.size + self.block_size - 1) // self.block_size)
        rng.shuffle(blocks)
        for block in blocks:
            start = int(block) * self.block_size
            values = np.arange(start, min(start + self.block_size, self.size), dtype=np.int64)
            rng.shuffle(values)
            yield from values.tolist()


def loader_kwargs(workers):
    result = {"num_workers": workers, "pin_memory": True}
    if workers:
        result.update(persistent_workers=True, prefetch_factor=2)
    return result


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, local, world, torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")


def barrier(world):
    if world > 1:
        dist.barrier()


def subset_params(params, mask):
    return tuple(value[mask] for value in params)


def likelihood_from_output(raw_model, output, visibility_km, cfg):
    limit, constant = cfg["censoring_limit_km"], cfg["extinction_constant"]
    uncensored = (visibility_km > 0) & (visibility_km < limit)
    censored = visibility_km == limit
    eligible = uncensored | censored
    if not eligible.any():
        raise RuntimeError("batch contains no likelihood-eligible samples")

    z = torch.zeros_like(visibility_km)
    z[uncensored] = torch.log(constant / visibility_km[uncensored])
    params = output["flow_params"]
    log_prob = raw_model.flow.log_prob(z, params)
    z30 = torch.full_like(visibility_km, math.log(constant / limit))
    log_cdf30 = raw_model.flow.log_cdf(z30, params)
    terms = torch.zeros_like(visibility_km)
    terms[uncensored] = -log_prob[uncensored]
    terms[censored] = -log_cdf30[censored]
    total = terms[eligible].mean()
    zero = total * 0
    parts = {
        "censored_NLL": total,
        "uncensored_NLL": terms[uncensored].mean() if uncensored.any() else zero,
        "censored_30km_NLL": terms[censored].mean() if censored.any() else zero,
    }
    return total, parts, {"uncensored": uncensored, "censored": censored, "eligible": eligible,
                          "z": z, "log_prob": log_prob, "log_cdf30": log_cdf30}


def event_metrics(v, probs, keep=None):
    if keep is None:
        keep = np.ones(len(v), dtype=bool)
    v, probs = v[keep], probs[keep]
    truth_class = np.zeros(len(v), np.int8)
    truth_class[v >= 0.5] = 1
    truth_class[v >= 1.0] = 2
    prediction = probs.argmax(1)
    specs = {
        "lt500m": (truth_class == 0, prediction == 0, probs[:, 0]),
        "500_1000m": (truth_class == 1, prediction == 1, probs[:, 1]),
        "lt1000m": (truth_class < 2, prediction < 2, probs[:, :2].sum(1)),
    }
    result = {}
    for name, (truth, guess, score) in specs.items():
        tp = int((truth & guess).sum())
        fp = int((~truth & guess).sum())
        fn = int((truth & ~guess).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[name] = {
            "Recall": recall,
            "Precision": precision,
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "AP": float(average_precision_score(truth, score)),
            "support": int(truth.sum()),
        }
    return result


def metrics_from_arrays(arrays, totals):
    v, probs, v_point = arrays["v"], arrays["probs"], arrays["v_point"]
    positive = v > 0
    events = event_metrics(v, probs)
    middle = (v >= 0.5) & (v < 1.0)
    middle_argmax = probs[middle].argmax(1)
    pit = arrays["pit"][np.isfinite(arrays["pit"])]
    pit_histogram, _ = np.histogram(pit, bins=np.linspace(0, 1, 11))
    error = v_point - v
    return {
        "events_including_v0": events,
        "events_excluding_v0": event_metrics(v, probs, positive),
        "mean_event_AP": float(np.mean([events[k]["AP"] for k in ("lt500m", "500_1000m", "lt1000m")])),
        "mean_event_CSI": float(np.mean([events[k]["CSI"] for k in ("lt500m", "500_1000m", "lt1000m")])),
        "censored_NLL": totals["nll_sum"] / max(totals["eligible_n"], 1),
        "uncensored_NLL": totals["uncensored_nll_sum"] / max(totals["uncensored_n"], 1),
        "censored_30km_NLL": totals["censored_nll_sum"] / max(totals["censored_n"], 1),
        "continuous": {
            "MAE_km": float(np.abs(error[positive]).mean()),
            "RMSE_km": float(np.sqrt(np.mean(error[positive] ** 2))),
            "MAE_v_lt_1km": float(np.abs(error[(v > 0) & (v < 1)]).mean()),
            "MAE_v_lt_0p5km": float(np.abs(error[(v > 0) & (v < 0.5)]).mean()),
        },
        "PIT_uncensored": {
            "samples": int(len(pit)),
            "mean": float(pit.mean()),
            "std": float(pit.std()),
            "fraction_lt_0p05": float((pit < 0.05).mean()),
            "fraction_gt_0p95": float((pit > 0.95).mean()),
            "decile_counts": pit_histogram.tolist(),
        },
        "censoring_calibration": {
            "samples_30km": int(totals["censored_n"]),
            "mean_P_latent_V_ge_30_on_30km": totals["censor_cdf_sum"] / max(totals["censored_n"], 1),
        },
        "true_500_1000m": {
            "samples": int(middle.sum()),
            "mean_P_lt500m": float(probs[middle, 0].mean()),
            "mean_P_500_1000m": float(probs[middle, 1].mean()),
            "mean_P_ge1000m": float(probs[middle, 2].mean()),
            "argmax_fraction_lt500m": float((middle_argmax == 0).mean()),
            "argmax_fraction_500_1000m": float((middle_argmax == 1).mean()),
            "argmax_fraction_ge1000m": float((middle_argmax == 2).mean()),
        },
        "probability_checks": {
            "minimum": float(probs.min()),
            "maximum": float(probs.max()),
            "max_sum_error": float(np.abs(probs.sum(1) - 1).max()),
            "all_finite": bool(np.isfinite(probs).all()),
        },
        "CRPS_V_km": None,
        "CRPS_note": "Not computed in this round because stable analytic CDF metrics are non-sampling and CRPS is optional.",
    }


@torch.inference_mode()
def inference_batch(raw_model, output, v, cfg):
    params = output["flow_params"]
    probs = raw_model.event_probabilities(params)
    v_point = raw_model.observed_visibility_median(params)
    pit = torch.full((len(v),), float("nan"), device=v.device, dtype=torch.float64)
    uncensored = (v > 0) & (v < cfg["censoring_limit_km"])
    if uncensored.any():
        z = torch.log(cfg["extinction_constant"] / v[uncensored])
        pit[uncensored] = raw_model.flow.cdf(z, subset_params(params, uncensored))
    z30 = torch.full_like(v, math.log(cfg["extinction_constant"] / cfg["censoring_limit_km"]))
    censor_cdf = raw_model.flow.cdf(z30, params)
    return probs, v_point, pit, censor_cdf


def empty_totals():
    return {"eligible_n": 0.0, "nll_sum": 0.0, "uncensored_n": 0.0, "uncensored_nll_sum": 0.0,
            "censored_n": 0.0, "censored_nll_sum": 0.0, "censor_cdf_sum": 0.0}


def add_likelihood_totals(totals, terms):
    uncensored, censored, eligible = terms["uncensored"], terms["censored"], terms["eligible"]
    values = torch.zeros_like(terms["z"])
    values[uncensored] = -terms["log_prob"][uncensored]
    values[censored] = -terms["log_cdf30"][censored]
    totals["eligible_n"] += int(eligible.sum())
    totals["nll_sum"] += float(values[eligible].sum())
    totals["uncensored_n"] += int(uncensored.sum())
    totals["uncensored_nll_sum"] += float(values[uncensored].sum())
    totals["censored_n"] += int(censored.sum())
    totals["censored_nll_sum"] += float(values[censored].sum())


@torch.inference_mode()
def evaluate(route, stage, split, checkpoint_path, output_path, save_predictions=False):
    cfg = config()
    rank, local, world, device = init_dist()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model = R5CensoredFlowModel(cfg).to(device)
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    raw_model.eval()
    n = len(visibility(Path(cfg["data"][stage]) / f"y_{split}.npy"))
    indices = np.arange(rank, n, world, dtype=np.int64)
    dataset = VisibilityDataset(cfg, stage, split, indices)
    workers = max(cfg["num_workers"] // world, 0)
    loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False, **loader_kwargs(workers))
    payload = {key: [] for key in ("index", "v", "probs", "v_point", "pit")}
    totals = empty_totals()
    started = time.time()
    for x, v, index in loader:
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        output = raw_model(x)
        _, _, terms = likelihood_from_output(raw_model, output, v, cfg)
        add_likelihood_totals(totals, terms)
        probs, v_point, pit, censor_cdf = inference_batch(raw_model, output, v, cfg)
        censored = terms["censored"]
        totals["censor_cdf_sum"] += float(censor_cdf[censored].sum())
        for key, value in (("index", index), ("v", v), ("probs", probs), ("v_point", v_point), ("pit", pit)):
            payload[key].append(value.cpu().numpy())
    payload = {key: np.concatenate(value) for key, value in payload.items()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.parent / f".{output_path.stem}.rank{rank}.npz"
    np.savez(part_path, **payload)
    total_keys = list(totals)
    total_tensor = torch.tensor([totals[key] for key in total_keys], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(total_tensor)
    totals = {key: float(total_tensor[i].cpu()) for i, key in enumerate(total_keys)}
    barrier(world)
    if rank == 0:
        pieces = [np.load(output_path.parent / f".{output_path.stem}.rank{r}.npz") for r in range(world)]
        arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in payload}
        order = np.argsort(arrays["index"])
        arrays = {key: value[order] for key, value in arrays.items()}
        result = metrics_from_arrays(arrays, totals)
        result.update({
            "route": route, "stage": stage, "split": split, "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]), "checkpoint_epoch": float(checkpoint["epoch"]),
            "samples": int(len(arrays["v"])), "wall_seconds": time.time() - started,
            "dcu_count": world, "inference_only": True, "sampling_used": False,
        })
        atomic_json(output_path, result)
        if save_predictions:
            np.savez_compressed(output_path.with_name(output_path.stem + "_predictions.npz"), **arrays)
        for piece in pieces:
            piece.close()
        for r in range(world):
            (output_path.parent / f".{output_path.stem}.rank{r}.npz").unlink()
        print(json.dumps(result), flush=True)
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


@torch.inference_mode()
def quick_validate(raw_model, loader, cfg):
    raw_model.eval()
    arrays = {key: [] for key in ("v", "probs", "v_point", "pit")}
    totals = empty_totals()
    for x, v, _ in loader:
        x, v = x.cuda(non_blocking=True), v.cuda(non_blocking=True)
        output = raw_model(x)
        _, _, terms = likelihood_from_output(raw_model, output, v, cfg)
        add_likelihood_totals(totals, terms)
        probs, v_point, pit, censor_cdf = inference_batch(raw_model, output, v, cfg)
        censored = terms["censored"]
        totals["censor_cdf_sum"] += float(censor_cdf[censored].sum())
        for key, value in (("v", v), ("probs", probs), ("v_point", v_point), ("pit", pit)):
            arrays[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    return metrics_from_arrays(arrays, totals)


def lr_factor(step, max_steps, warmup):
    if step <= warmup:
        return step / max(warmup, 1)
    progress = min(max((step - warmup) / max(max_steps - warmup, 1), 0), 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(run_dir, step, epoch, model, optimizer, cfg, route, stage):
    path = run_dir / "checkpoints" / f"ckpt_step_{step:05d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
        "step": step, "epoch": epoch, "route": route, "stage": stage,
        "censoring_limit_km": cfg["censoring_limit_km"],
        "z30": math.log(cfg["extinction_constant"] / cfg["censoring_limit_km"]),
    }
    torch.save(state, tmp)
    os.replace(tmp, path)
    ready_tmp = path.with_suffix(".ready.tmp")
    ready_tmp.write_text("ready\n", encoding="utf-8")
    os.replace(ready_tmp, path.with_suffix(".ready"))
    return path


def completed_full_results(run_dir):
    rows = []
    for path in sorted((run_dir / "validation").glob("full_step_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(rows, key=lambda row: row["checkpoint_step"])


def validator_already_submitted(run_dir, step):
    path = run_dir / "validator_jobs.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if int(json.loads(line)["step"]) == int(step):
                return True
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return False


def train(route, stage, seed, init_checkpoint=None):
    cfg = config()
    rank, local, world, device = init_dist()
    if world != cfg["training_dcu_count"]:
        raise RuntimeError("formal R5-CensoredFlow training must use four-DCU DDP")
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_dir = ROOT / "runs" / route / f"seed{seed}" / stage
    if rank == 0:
        if (run_dir / "training_complete.json").exists():
            raise RuntimeError(f"completed run exists: {run_dir}")
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    barrier(world)

    train_dataset = VisibilityDataset(cfg, stage, "train")
    sampler = DistributedSampler(train_dataset, world, rank, shuffle=True, seed=seed)
    workers = max(cfg["num_workers"] // world, 0)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True,
        **loader_kwargs(workers),
    )
    quick_indices = np.load(ROOT / "contracts" / f"quick_indices_{stage}.npy")
    quick_x = os.environ.get("R5_QUICK_X")
    quick_dataset = VisibilityDataset(
        cfg, stage, "val", quick_indices, x_path_override=quick_x, local_rows=bool(quick_x)
    )
    quick_loader = DataLoader(
        quick_dataset, batch_size=cfg["quick_eval_batch_size"], shuffle=False,
        **loader_kwargs(workers),
    )

    raw_model = R5CensoredFlowModel(cfg).to(device)
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location=device)
        raw_model.load_state_dict(initial["model"], strict=True)
    model = DDP(raw_model, device_ids=[local], find_unused_parameters=False)
    head_parameters, backbone_parameters = [], []
    for name, parameter in raw_model.named_parameters():
        (head_parameters if name.startswith(("flow_query", "flow.")) else backbone_parameters).append(parameter)
    backbone_lr = cfg["learning_rate"][f"{stage}_backbone"]
    head_lr = cfg["learning_rate"][f"{stage}_head"]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_parameters, "lr": backbone_lr}, {"params": head_parameters, "lr": head_lr}],
        weight_decay=cfg["weight_decay"],
    )
    max_steps, warmup = cfg["max_steps"][stage][route], cfg["warmup_steps"][stage]
    train_log = open(run_dir / "train_history.jsonl", "a", buffering=1) if rank == 0 else None
    quick_log = open(run_dir / "quick_validation_history.jsonl", "a", buffering=1) if rank == 0 else None
    validator_ids, step, processed, epoch, started = [], 0, 0, 0, time.time()

    while step < max_steps:
        sampler.set_epoch(epoch)
        for x, v, _ in train_loader:
            step += 1
            processed += cfg["batch_size"] * world
            factor = lr_factor(step, max_steps, warmup)
            optimizer.param_groups[0]["lr"] = backbone_lr * factor
            optimizer.param_groups[1]["lr"] = head_lr * factor
            x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            output = model(x)
            loss, parts, _ = likelihood_from_output(raw_model, output, v, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - t0
            if rank == 0:
                train_log.write(json.dumps({
                    "step": step, "epoch": processed / len(train_dataset), "processed_samples": processed,
                    "train_censored_NLL": float(loss),
                    "loss_components": {key: float(value) for key, value in parts.items()},
                    "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                    "step_seconds": step_seconds,
                    "samples_per_second": cfg["batch_size"] * world / step_seconds,
                }) + "\n")

            if step % cfg["quick_val_interval"] == 0:
                quick_started = time.perf_counter()
                quick = quick_validate(raw_model, quick_loader, cfg)
                if rank == 0:
                    quick_log.write(json.dumps({
                        "step": step, "epoch": processed / len(train_dataset), "processed_samples": processed,
                        "validation_kind": "quick_analytic", "subset_samples": len(quick_dataset),
                        "wall_seconds": time.perf_counter() - quick_started, "sampling": False, **quick,
                    }) + "\n")
                model.train()

            if step % cfg["checkpoint_interval"] == 0:
                if rank == 0:
                    checkpoint_path = save_checkpoint(
                        run_dir, step, processed / len(train_dataset), raw_model, optimizer, cfg, route, stage
                    )
                    if stage == "S2" and step % cfg["async_full_interval"] == 0 and not validator_already_submitted(run_dir, step):
                        output_path = run_dir / "validation" / f"full_step_{step:05d}.json"
                        command = [
                            "sbatch", "--parsable", f"--job-name=r5flowv_{stage}_s{seed}_{step}",
                            str(ROOT / "scripts" / "sub_validate.slurm"), route, stage, str(seed),
                            str(checkpoint_path), str(output_path),
                        ]
                        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
                        validator_ids.append(job_id)
                        with open(run_dir / "validator_jobs.jsonl", "a") as handle:
                            handle.write(json.dumps({
                                "step": step, "job_id": job_id, "checkpoint": str(checkpoint_path),
                                "output": str(output_path),
                            }) + "\n")
            if step >= max_steps:
                break
        epoch += 1

    if rank == 0:
        if step % cfg["checkpoint_interval"] != 0:
            save_checkpoint(run_dir, step, processed / len(train_dataset), raw_model, optimizer, cfg, route, stage)
        summary = {
            "route": route, "stage": stage, "seed": seed, "steps": step,
            "processed_samples": processed, "epochs": processed / len(train_dataset),
            "wall_seconds": time.time() - started, "stop_reason": "max_steps",
            "allocated_dcu_count": int(os.environ.get("SLURM_GPUS_ON_NODE", cfg["allocated_dcu_count"])),
            "training_dcu_count": world, "batch_size_per_dcu": cfg["batch_size"],
            "effective_batch_size": cfg["batch_size"] * world, "num_workers_total": cfg["num_workers"],
            "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
            "validator_job_ids": validator_ids,
        }
        atomic_json(run_dir / "training_complete.json", summary)
        train_log.close()
        quick_log.close()
        if stage == "S1":
            select_s1_handoff(route, seed, submit_s2=True)
        else:
            dependency = "afterok:" + ":".join(validator_ids) if validator_ids else None
            command = ["sbatch", "--parsable"]
            if dependency:
                command.append(f"--dependency={dependency}")
            command += ["--job-name=r5flow_finalize_S2", str(ROOT / "scripts" / "sub_finalize.slurm"), route, stage, str(seed)]
            finalizer = subprocess.check_output(command, text=True).strip().split(";")[0]
            atomic_json(run_dir / "finalizer_job.json", {"job_id": finalizer, "dependency": dependency})
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


def select_best(route, stage, seed):
    run_dir = ROOT / "runs" / route / f"seed{seed}" / stage
    rows = completed_full_results(run_dir)
    if not rows:
        raise RuntimeError(f"no completed full validation results under {run_dir}")
    best = max(rows, key=lambda row: (row["mean_event_AP"], row["mean_event_CSI"]))
    history = run_dir / "validation_history.jsonl"
    tmp = history.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, history)
    selection = {
        "route": route, "stage": stage, "seed": seed,
        "selection_primary": "mean_event_AP", "selection_secondary": "mean_event_CSI",
        "best_step": best["checkpoint_step"], "best_checkpoint": best["checkpoint"],
        "best_mean_event_AP": best["mean_event_AP"], "best_mean_event_CSI": best["mean_event_CSI"],
        "validated_checkpoints": len(rows),
    }
    atomic_json(run_dir / "selection.json", selection)
    print(json.dumps(selection))


def select_s1_handoff(route, seed, submit_s2=False):
    cfg = config()
    run_dir = ROOT / "runs" / route / f"seed{seed}" / "S1"
    rows = [json.loads(line) for line in (run_dir / "quick_validation_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = []
    for row in rows:
        step = int(row["step"])
        checkpoint_path = run_dir / "checkpoints" / f"ckpt_step_{step:05d}.pt"
        score = float(row["censored_NLL"])
        if math.isfinite(score) and step % cfg["checkpoint_interval"] == 0 and checkpoint_path.is_file() and checkpoint_path.with_suffix(".ready").is_file():
            candidates.append((score, step, checkpoint_path, row))
    if not candidates:
        raise RuntimeError("no finite analytic S1 validation result has a ready checkpoint")
    score, step, checkpoint_path, row = min(candidates, key=lambda item: (item[0], item[1]))
    state = torch.load(checkpoint_path, map_location="cpu")
    model = R5CensoredFlowModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if torch.is_floating_point(value)):
        raise RuntimeError(f"non-finite weights in {checkpoint_path}")
    selection = {
        "route": route, "stage": "S1", "seed": seed,
        "selection_metric": "quick_analytic_censored_NLL", "best_step": step,
        "best_checkpoint": str(checkpoint_path), "censored_NLL": score,
        "mean_event_AP": row["mean_event_AP"], "mean_event_CSI": row["mean_event_CSI"],
        "checkpoint_load_strict": True, "weights_all_finite": True, "sampling_used": False,
    }
    atomic_json(run_dir / "handoff_selection.json", selection)
    if submit_s2:
        handoff_path = run_dir / "s2_handoff_job.json"
        if handoff_path.exists():
            raise RuntimeError(f"S2 handoff was already submitted: {handoff_path}")
        command = [
            "sbatch", "--parsable", f"--job-name=r5flow_S2_s{seed}",
            str(ROOT / "scripts" / "sub_train.slurm"), route, "S2", str(seed), str(checkpoint_path),
        ]
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        atomic_json(handoff_path, {"job_id": job_id, "checkpoint": str(checkpoint_path), "dependency": None})
        selection["s2_job_id"] = job_id
    print(json.dumps(selection))
    return selection


def best_path(route, stage, seed):
    value = json.loads((ROOT / "runs" / route / f"seed{seed}" / stage / "selection.json").read_text(encoding="utf-8"))
    print(value["best_checkpoint"])


def describe():
    cfg = config()
    model = R5CensoredFlowModel(cfg)
    modules = {
        "DynamicTokenizer": model.dynamic_tokenizer,
        "DynamicEncoder": model.dynamic_encoder,
        "ContextEncoder": model.context_encoder,
        "FlowQuery": model.flow_query,
        "RQSConditioner": model.flow.conditioner,
    }
    result = {
        "R5-CensoredFlow": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "module_parameters": {
                name: sum(parameter.numel() for parameter in module.parameters()) for name, module in modules.items()
            },
            "spline_bins": cfg["spline_bins"], "spline_blocks": 1,
            "gate_head": False, "cap_head": False, "diffusion": False, "classification_head": False,
        }
    }
    atomic_json(ROOT / "model_description.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--route", choices=ROUTES, required=True)
    train_parser.add_argument("--stage", choices=("S1", "S2"), required=True)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--init-checkpoint")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--route", choices=ROUTES, required=True)
    validate_parser.add_argument("--stage", choices=("S1", "S2"), required=True)
    validate_parser.add_argument("--split", choices=("val", "test"), required=True)
    validate_parser.add_argument("--checkpoint", required=True)
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--save-predictions", action="store_true")
    handoff_parser = commands.add_parser("handoff-s1")
    handoff_parser.add_argument("--route", choices=ROUTES, required=True)
    handoff_parser.add_argument("--seed", type=int, required=True)
    handoff_parser.add_argument("--submit-s2", action="store_true")
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--route", choices=ROUTES, required=True)
    select_parser.add_argument("--stage", choices=("S1", "S2"), required=True)
    select_parser.add_argument("--seed", type=int, default=1)
    best_parser = commands.add_parser("best-path")
    best_parser.add_argument("--route", choices=ROUTES, required=True)
    best_parser.add_argument("--stage", choices=("S1", "S2"), required=True)
    best_parser.add_argument("--seed", type=int, default=1)
    commands.add_parser("describe")
    args = parser.parse_args()
    if args.command == "train":
        train(args.route, args.stage, args.seed, args.init_checkpoint)
    elif args.command == "validate":
        evaluate(args.route, args.stage, args.split, args.checkpoint, args.output, args.save_predictions)
    elif args.command == "handoff-s1":
        select_s1_handoff(args.route, args.seed, args.submit_s2)
    elif args.command == "select":
        select_best(args.route, args.stage, args.seed)
    elif args.command == "best-path":
        best_path(args.route, args.stage, args.seed)
    else:
        describe()


if __name__ == "__main__":
    main()
