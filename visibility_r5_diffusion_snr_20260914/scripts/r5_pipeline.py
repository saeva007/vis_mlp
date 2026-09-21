#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from r5_model import R5Model

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("R5-Gaussian", "R5-Diffusion")


def config():
    return json.loads((ROOT / "configs" / "r5.json").read_text(encoding="utf-8"))


def contract(stage):
    return json.loads((ROOT / "contracts" / f"{stage}.json").read_text(encoding="utf-8"))


def visibility(path):
    v = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if len(v) and float(np.nanmax(v)) >= 100.0:
        v = v / 1000.0
    return v


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class VisibilityDataset(Dataset):
    def __init__(self, cfg, stage, split, indices=None, x_path_override=None, local_rows=False):
        self.cfg, self.stage, self.split = cfg, stage, split
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
        for t in range(cfg["window_size"]):
            self.log_mask[t * cfg["dyn_vars"] + np.asarray(cfg["log1p_dyn_indices"])] = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode="r")
        idx = int(self.indices[position])
        row = self.x[position if self.local_rows else idx]
        core = row[: self.split_dyn + 5].astype(np.float32)
        core[: self.split_dyn] = np.where(self.log_mask, np.log1p(np.maximum(core[: self.split_dyn], 0)), core[: self.split_dyn])
        core = (core - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        vegetation = np.asarray([row[self.split_dyn + 5]], dtype=np.float32)
        engineered = row[self.split_dyn + 6 :].astype(np.float32)
        x = np.concatenate([np.clip(core, -10, 10), vegetation, np.clip(engineered, -10, 10)])
        return torch.from_numpy(np.nan_to_num(x, nan=0.0)).float(), torch.tensor(self.v[idx]), torch.tensor(idx)


class BlockShuffleSampler(Sampler):
    """Uniform full-dataset shuffle with bounded, locality-friendly index blocks."""
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
    out = {"num_workers": workers, "pin_memory": True}
    if workers:
        out.update(persistent_workers=True, prefetch_factor=2)
    return out


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, local, world, device


def barrier(world):
    if world > 1:
        dist.barrier()


def cosine_schedule(steps, device):
    s = 0.008
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64, device=device)
    abar = torch.cos(((x / steps + s) / (1 + s)) * math.pi * 0.5).square()
    abar = abar / abar[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999).float()
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, 0)


def z_target(v, stats, cfg):
    emin = cfg["extinction_constant"] / cfg["gate_vc_km"]
    z = torch.log(cfg["extinction_constant"] / v.double() - emin)
    return ((z - stats["z_mean"]) / stats["z_std"]).float()


def z_threshold(v_km, stats, cfg):
    emin = cfg["extinction_constant"] / cfg["gate_vc_km"]
    raw = math.log(cfg["extinction_constant"] / v_km - emin)
    return (raw - stats["z_mean"]) / stats["z_std"]


def z_to_visibility(z_scaled, stats, cfg):
    raw = z_scaled.double() * stats["z_std"] + stats["z_mean"]
    e = cfg["extinction_constant"] / cfg["gate_vc_km"] + torch.exp(raw)
    return cfg["extinction_constant"] / e, e


def interval_bce_from_probs(probs, v, cfg):
    losses = []
    for i, threshold in enumerate(cfg["interval_thresholds_km"]):
        losses.append(F.binary_cross_entropy(probs[:, i].clamp(1e-6, 1 - 1e-6), (v < threshold).float()))
    return torch.stack(losses).mean()


def balanced_boundary_bce(logits, targets, sample_weight=None):
    """Balance positive/negative classes only inside the boundary regularizer."""
    losses = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    if sample_weight is not None:
        losses = losses * sample_weight
    groups = []
    positive = targets.bool()
    if positive.any():
        groups.append(losses[positive].mean())
    if (~positive).any():
        groups.append(losses[~positive].mean())
    return torch.stack(groups).mean() if groups else losses.sum() * 0


def gaussian_conditional_probs(mu, sigma, stats, cfg):
    values = []
    for threshold in cfg["interval_thresholds_km"]:
        zt = z_threshold(threshold, stats, cfg)
        cdf = 0.5 * (1 + torch.erf((zt - mu) / (sigma * math.sqrt(2))))
        values.append(1 - cdf)
    p_lt_05, p_lt_1 = values
    return torch.stack([p_lt_05, (p_lt_1 - p_lt_05).clamp_min(0), (1 - p_lt_1).clamp_min(0)], 1)


def train_loss(raw_model, model, x, v, stats, cfg, schedule):
    vc = cfg["gate_vc_km"]
    gate_truth = v >= vc
    cont = (v > 0) & (v < vc)
    if raw_model.route == "R5-Gaussian":
        out = model(x)
        gate = F.binary_cross_entropy_with_logits(out["gate_logit"], gate_truth.float())
        if cont.any():
            target = z_target(v[cont], stats, cfg)
            mu, log_sigma = out["mu_z"][cont], out["log_sigma_z"][cont]
            sigma = log_sigma.exp()
            nll = (0.5 * ((target - mu) / sigma).square() + log_sigma + 0.5 * math.log(2 * math.pi)).mean()
            # The second boundary probability is P(V<1), i.e. p(class0)+p(class1).
            cond_probs = gaussian_conditional_probs(mu, sigma, stats, cfg)
            interval = interval_bce_from_probs(torch.stack([cond_probs[:, 0], cond_probs[:, :2].sum(1)], 1), v[cont], cfg)
        else:
            nll = interval = out["gate_logit"].sum() * 0
        parts = {"gate": gate, "nll": nll, "interval": interval}
    else:
        b = len(v)
        zt_all = torch.zeros(b, device=v.device)
        time_all = torch.zeros(b, dtype=torch.long, device=v.device)
        noise_all = torch.zeros(b, device=v.device)
        target_v_all = torch.zeros(b, device=v.device)
        if cont.any():
            x0 = z_target(v[cont], stats, cfg)
            time = torch.randint(0, cfg["diffusion_steps"], (int(cont.sum()),), device=v.device)
            noise = torch.randn_like(x0)
            abar = schedule[2][time]
            zt = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
            target_v = abar.sqrt() * noise - (1 - abar).sqrt() * x0
            zt_all[cont], time_all[cont], noise_all[cont], target_v_all[cont] = zt, time, noise, target_v
        out = model(x, zt_all, time_all)
        gate = F.binary_cross_entropy_with_logits(out["gate_logit"], gate_truth.float())
        if cont.any():
            pred = out["v_pred"][cont]
            diff_loss = F.mse_loss(pred, target_v_all[cont])
            abar = schedule[2][time_all[cont]]
            x0_pred = abar.sqrt() * zt_all[cont] - (1 - abar).sqrt() * pred
            boundary_losses = []
            snr_boundary_losses = []
            for threshold in cfg["interval_thresholds_km"]:
                logits = (x0_pred - z_threshold(threshold, stats, cfg)) / cfg["boundary_temperature_z"]
                truth = v[cont] < threshold
                boundary_losses.append(balanced_boundary_bce(logits, truth))
                snr_boundary_losses.append(balanced_boundary_bce(logits, truth, abar))
            boundary = torch.stack(boundary_losses).mean()
            boundary_snr = torch.stack(snr_boundary_losses).mean()
            boundary_weight = abar.mean()
        else:
            diff_loss = boundary = boundary_snr = boundary_weight = out["gate_logit"].sum() * 0
        parts = {"gate": gate, "diffusion": diff_loss, "boundary": boundary,
                 "boundary_snr": boundary_snr, "boundary_weight": boundary_weight}
    if raw_model.route == "R5-Gaussian":
        total = cfg["lambda_gate"] * parts["gate"] + parts["nll"] + cfg["lambda_boundary"] * parts["interval"]
    else:
        total = cfg["lambda_gate"] * parts["gate"] + parts["diffusion"] + cfg["lambda_boundary"] * parts["boundary_snr"]
    return total, parts


@torch.inference_mode()
def diffusion_samples(model, encoded, draws, stats, cfg, schedule, reverse_steps=None):
    memory, h_cont = encoded
    b, device = len(h_cont), h_cont.device
    total_steps = cfg["diffusion_steps"]
    reverse_steps = total_steps if reverse_steps is None else int(reverse_steps)
    if reverse_steps < total_steps:
        timesteps = np.unique(np.linspace(0, total_steps - 1, reverse_steps, dtype=np.int64))[::-1].tolist()
    else:
        timesteps = list(reversed(range(total_steps)))
    chunks = []
    for start in range(0, draws, cfg["sample_draw_chunk"]):
        n = min(cfg["sample_draw_chunk"], draws - start)
        x = torch.randn(b * n, device=device)
        mem = memory[:, None].expand(b, n, *memory.shape[1:]).reshape(b * n, *memory.shape[1:])
        hc = h_cont[:, None].expand(b, n, h_cont.shape[1]).reshape(b * n, h_cont.shape[1])
        for position, step in enumerate(timesteps):
            t = torch.full((b * n,), step, dtype=torch.long, device=device)
            beta, alpha, abar = schedule[0][step], schedule[1][step], schedule[2][step]
            v_pred = model.predict_v(x, t, hc, mem)
            eps = abar.sqrt() * v_pred + (1 - abar).sqrt() * x
            if reverse_steps < total_steps:
                x0 = abar.sqrt() * x - (1 - abar).sqrt() * v_pred
                previous = timesteps[position + 1] if position + 1 < len(timesteps) else -1
                if previous < 0:
                    x = x0
                else:
                    previous_abar = schedule[2][previous]
                    x = previous_abar.sqrt() * x0 + (1 - previous_abar).sqrt() * eps
            else:
                mean = (x - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
                x = mean if step == 0 else mean + beta.sqrt() * torch.randn_like(x)
        chunks.append(x.view(b, n))
    return torch.cat(chunks, 1)


@torch.inference_mode()
def prediction_batch(model, x, draws, stats, cfg, schedule, reverse_steps=None):
    out = model(x)
    p_gate = out["gate_logit"].sigmoid().double()
    if model.route == "R5-Gaussian":
        mu, sigma = out["mu_z"], out["log_sigma_z"].exp()
        cond_probs = gaussian_conditional_probs(mu, sigma, stats, cfg).double()
        z_samples = mu[:, None] + sigma[:, None] * torch.randn(len(mu), draws, device=x.device)
    else:
        z_samples = diffusion_samples(model, (out["memory"], out["h_cont"]), draws, stats, cfg, schedule, reverse_steps)
        v_samples, _ = z_to_visibility(z_samples, stats, cfg)
        cond_probs = torch.stack([
            (v_samples < 0.5).double().mean(1),
            ((v_samples >= 0.5) & (v_samples < 1.0)).double().mean(1),
            (v_samples >= 1.0).double().mean(1),
        ], 1)
    v_samples, e_samples = z_to_visibility(z_samples, stats, cfg)
    probs = torch.stack([
        (1 - p_gate) * cond_probs[:, 0],
        (1 - p_gate) * cond_probs[:, 1],
        p_gate + (1 - p_gate) * cond_probs[:, 2],
    ], 1)
    v_point = p_gate * stats["gate_region_mean_visibility_km"] + (1 - p_gate) * v_samples.mean(1)
    return probs, p_gate, v_point, v_samples, e_samples, out


def crps_samples(samples, truth):
    s = samples.sort(1).values
    n = s.shape[1]
    weights = (2 * torch.arange(1, n + 1, device=s.device) - n - 1).double()
    return (samples - truth[:, None]).abs().mean(1) - (s * weights).sum(1) / (n * n)


def event_metrics(v, probs, keep=None):
    if keep is None:
        keep = np.ones(len(v), dtype=bool)
    v, probs = v[keep], probs[keep]
    truth_cls = np.zeros(len(v), np.int8)
    truth_cls[v >= 0.5] = 1
    truth_cls[v >= 1.0] = 2
    pred = probs.argmax(1)
    specs = {
        "lt500m": (truth_cls == 0, pred == 0, probs[:, 0]),
        "500_1000m": (truth_cls == 1, pred == 1, probs[:, 1]),
        "lt1000m": (truth_cls < 2, pred < 2, probs[:, :2].sum(1)),
    }
    result = {}
    for name, (truth, guess, score) in specs.items():
        tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[name] = {
            "Recall": recall, "Precision": precision, "CSI": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "AP": float(average_precision_score(truth, score)), "support": int(truth.sum()),
        }
    return result


def metrics_from_arrays(v, probs, p_gate, v_point, totals, cfg, route, draws):
    gate_truth = v >= cfg["gate_vc_km"]
    gate_pred = p_gate >= 0.5
    tp, fp, fn, tn = (int((gate_truth & gate_pred).sum()), int((~gate_truth & gate_pred).sum()),
                      int((gate_truth & ~gate_pred).sum()), int((~gate_truth & ~gate_pred).sum()))
    positive = v > 0
    err = v_point - v
    events = event_metrics(v, probs)
    middle = (v >= 0.5) & (v < 1.0)
    middle_pred = probs[middle].argmax(1)
    loss_excluded = {"n", "crps_sum", "crps_n", "nll_sum", "nll_n",
                     "generated_draws", "generated_e_le_emin", "generated_v_ge_vc"}
    return {
        "events_including_v0": events,
        "events_excluding_v0": event_metrics(v, probs, positive),
        "mean_event_AP": float(np.mean([events[k]["AP"] for k in ("lt500m", "500_1000m", "lt1000m")])),
        "mean_event_CSI": float(np.mean([events[k]["CSI"] for k in ("lt500m", "500_1000m", "lt1000m")])),
        "gate": {
            "accuracy": (tp + tn) / len(v), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "AUROC": float(roc_auc_score(gate_truth, p_gate)),
            "AP": float(average_precision_score(gate_truth, p_gate)), "Brier": float(np.mean((p_gate - gate_truth) ** 2)),
        },
        "continuous": {
            "MAE_km": float(np.abs(err[positive]).mean()), "RMSE_km": float(np.sqrt(np.mean(err[positive] ** 2))),
            "MAE_v_lt_1km": float(np.abs(err[(v > 0) & (v < 1)]).mean()),
            "MAE_v_lt_0p5km": float(np.abs(err[(v > 0) & (v < 0.5)]).mean()),
        },
        "true_500_1000m": {
            "samples": int(middle.sum()),
            "mean_P_lt500m": float(probs[middle, 0].mean()),
            "mean_P_500_1000m": float(probs[middle, 1].mean()),
            "mean_P_ge1000m": float(probs[middle, 2].mean()),
            "argmax_fraction_lt500m": float((middle_pred == 0).mean()),
            "argmax_fraction_500_1000m": float((middle_pred == 1).mean()),
            "argmax_fraction_ge1000m": float((middle_pred == 2).mean()),
        },
        "physical_support": {
            "generated_draws": int(totals.get("generated_draws", 0)),
            "E_le_Emin_count": int(totals.get("generated_e_le_emin", 0)),
            "continuous_V_ge_Vc_count": int(totals.get("generated_v_ge_vc", 0)),
        },
        "loss": {k: totals[k] / max(totals["n"], 1) for k in totals if k not in loss_excluded},
        "CRPS_V_km": totals["crps_sum"] / max(totals["crps_n"], 1),
        "NLL_z": totals["nll_sum"] / max(totals["nll_n"], 1) if route == "R5-Gaussian" else None,
        "draws": draws,
    }


@torch.inference_mode()
def evaluate(route, stage, split, checkpoint_path, draws, output_path, save_predictions=False, reverse_steps=None):
    cfg, stats = config(), contract(stage)
    rank, local, world, device = init_dist()
    torch.manual_seed(910000 + int(Path(checkpoint_path).stem.split("_")[-1]) + rank)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = R5Model(cfg, route).to(device)
    weights = ckpt.get("ema_model") if route == "R5-Diffusion" and ckpt.get("ema_model") is not None else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    schedule = cosine_schedule(cfg["diffusion_steps"], device)
    data_root = Path(cfg["data"][stage])
    n = len(visibility(data_root / f"y_{split}.npy"))
    indices = np.arange(rank, n, world, dtype=np.int64)
    ds = VisibilityDataset(cfg, stage, split, indices)
    workers = max(cfg["num_workers"] // world, 0)
    loader = DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False, **loader_kwargs(workers))
    payload = {"index": [], "v": [], "probs": [], "p_gate": [], "v_point": []}
    totals = {"n": 0.0, "total": 0.0, "gate": 0.0, "nll": 0.0, "diffusion": 0.0, "interval": 0.0,
              "boundary": 0.0, "boundary_snr": 0.0, "boundary_weight": 0.0,
              "crps_sum": 0.0, "crps_n": 0.0, "nll_sum": 0.0, "nll_n": 0.0,
              "generated_draws": 0.0, "generated_e_le_emin": 0.0, "generated_v_ge_vc": 0.0}
    started = time.time()
    effective_reverse_steps = (cfg["diffusion_steps"] if reverse_steps is None else int(reverse_steps)) \
        if route == "R5-Diffusion" else None
    total_batches = len(loader)
    for batch_number, (x, v, idx) in enumerate(loader, 1):
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        loss, parts = train_loss(model, model, x, v, stats, cfg, schedule)
        probs, p_gate, v_point, v_samples, e_samples, out = prediction_batch(
            model, x, draws, stats, cfg, schedule,
            effective_reverse_steps)
        cont = (v > 0) & (v < cfg["gate_vc_km"])
        totals["n"] += len(v); totals["total"] += float(loss) * len(v)
        for k, value in parts.items(): totals[k] += float(value) * len(v)
        totals["generated_draws"] += e_samples.numel()
        totals["generated_e_le_emin"] += int((e_samples <= cfg["extinction_constant"] / cfg["gate_vc_km"]).sum())
        totals["generated_v_ge_vc"] += int((v_samples >= cfg["gate_vc_km"]).sum())
        if cont.any():
            totals["crps_sum"] += float(crps_samples(v_samples[cont], v[cont].double()).sum())
            totals["crps_n"] += int(cont.sum())
            if route == "R5-Gaussian":
                target = z_target(v[cont], stats, cfg).double()
                mu, log_sigma = out["mu_z"][cont].double(), out["log_sigma_z"][cont].double()
                totals["nll_sum"] += float((0.5 * ((target - mu) / log_sigma.exp()).square() + log_sigma + 0.5 * math.log(2 * math.pi) + math.log(stats["z_std"])).sum())
                totals["nll_n"] += int(cont.sum())
        payload["index"].append(idx.numpy()); payload["v"].append(v.cpu().numpy())
        payload["probs"].append(probs.float().cpu().numpy()); payload["p_gate"].append(p_gate.float().cpu().numpy())
        payload["v_point"].append(v_point.float().cpu().numpy())
        if rank == 0 and (batch_number % 250 == 0 or batch_number == total_batches):
            elapsed = time.time() - started
            eta = elapsed / batch_number * (total_batches - batch_number)
            print(json.dumps({"validation_progress": True, "checkpoint_step": int(ckpt["step"]),
                              "rank": rank, "batches_done": batch_number, "batches_total": total_batches,
                              "elapsed_seconds": elapsed, "eta_seconds": eta}), flush=True)
    payload = {k: np.concatenate(v) for k, v in payload.items()}
    out_path = Path(output_path)
    part = out_path.parent / f".{out_path.stem}.rank{rank}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(part, **payload)
    total_keys = list(totals)
    total_tensor = torch.tensor([totals[k] for k in total_keys], dtype=torch.float64, device=device)
    if world > 1: dist.all_reduce(total_tensor)
    totals = {k: float(total_tensor[i].cpu()) for i, k in enumerate(total_keys)}
    barrier(world)
    if rank == 0:
        pieces = [np.load(out_path.parent / f".{out_path.stem}.rank{r}.npz") for r in range(world)]
        arrays = {k: np.concatenate([p[k] for p in pieces]) for k in payload}
        order = np.argsort(arrays["index"])
        arrays = {k: value[order] for k, value in arrays.items()}
        result = metrics_from_arrays(arrays["v"], arrays["probs"], arrays["p_gate"], arrays["v_point"], totals, cfg, route, draws)
        result.update({"route": route, "stage": stage, "split": split, "checkpoint": str(checkpoint_path),
                       "checkpoint_step": int(ckpt["step"]), "checkpoint_epoch": float(ckpt["epoch"]),
                       "samples": int(len(arrays["v"])), "wall_seconds": time.time() - started,
                       "validator_dcu_count": world, "inference_only": True,
                       "reverse_steps": effective_reverse_steps})
        atomic_json(out_path, result)
        if save_predictions:
            np.savez_compressed(out_path.with_name(out_path.stem + "_predictions.npz"), **arrays)
        for p in pieces: p.close()
        for r in range(world): (out_path.parent / f".{out_path.stem}.rank{r}.npz").unlink()
        print(json.dumps(result), flush=True)
    barrier(world)
    if world > 1: dist.destroy_process_group()


@torch.inference_mode()
def quick_validate(model, loader, route, stats, cfg, schedule, draws, seed):
    model.eval()
    torch.manual_seed(810000 + seed)
    arrays = {"v": [], "probs": [], "p_gate": [], "v_point": []}
    sums = {"n": 0.0, "total": 0.0, "gate": 0.0, "nll": 0.0, "diffusion": 0.0, "interval": 0.0,
            "boundary": 0.0, "boundary_snr": 0.0, "boundary_weight": 0.0,
            "crps_sum": 0.0, "crps_n": 0.0, "nll_sum": 0.0, "nll_n": 0.0}
    for x, v, _ in loader:
        x, v = x.cuda(non_blocking=True), v.cuda(non_blocking=True)
        loss, parts = train_loss(model, model, x, v, stats, cfg, schedule)
        probs, p_gate, v_point, v_samples, _, out = prediction_batch(
            model, x, draws, stats, cfg, schedule,
            cfg["quick_reverse_steps"] if route == "R5-Diffusion" else None)
        cont = (v > 0) & (v < cfg["gate_vc_km"])
        sums["n"] += len(v); sums["total"] += float(loss) * len(v)
        for k, value in parts.items(): sums[k] += float(value) * len(v)
        if cont.any():
            sums["crps_sum"] += float(crps_samples(v_samples[cont], v[cont].double()).sum()); sums["crps_n"] += int(cont.sum())
            if route == "R5-Gaussian":
                target, mu, ls = z_target(v[cont], stats, cfg).double(), out["mu_z"][cont].double(), out["log_sigma_z"][cont].double()
                sums["nll_sum"] += float((0.5 * ((target - mu) / ls.exp()).square() + ls + 0.5 * math.log(2 * math.pi) + math.log(stats["z_std"])).sum())
                sums["nll_n"] += int(cont.sum())
        for key, value in (("v", v), ("probs", probs), ("p_gate", p_gate), ("v_point", v_point)):
            arrays[key].append(value.float().cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    return metrics_from_arrays(arrays["v"], arrays["probs"], arrays["p_gate"], arrays["v_point"], sums, cfg, route, draws)


@torch.inference_mode()
def quick_validate_objective(model, loader, stats, cfg, schedule, seed):
    """Fixed-subset forward-only validation; never invokes reverse diffusion."""
    model.eval()
    torch.manual_seed(810000 + seed)
    sums = {"samples": 0, "total": 0.0, "gate": 0.0, "nll": 0.0, "diffusion": 0.0, "interval": 0.0,
            "boundary": 0.0, "boundary_snr": 0.0, "boundary_weight": 0.0}
    for x, v, _ in loader:
        x, v = x.cuda(non_blocking=True), v.cuda(non_blocking=True)
        loss, parts = train_loss(model, model, x, v, stats, cfg, schedule)
        count = len(v)
        sums["samples"] += count
        sums["total"] += float(loss) * count
        for key, value in parts.items():
            sums[key] += float(value) * count
    count = max(sums.pop("samples"), 1)
    objective = {key: value / count for key, value in sums.items()}
    return {"sampling": False, "subset_samples": count, "validation_objective": objective}


def update_ema(ema, model, decay):
    source = dict(model.named_parameters())
    with torch.no_grad():
        for name, target in ema.named_parameters(): target.mul_(decay).add_(source[name], alpha=1 - decay)
        buffers = dict(model.named_buffers())
        for name, target in ema.named_buffers(): target.copy_(buffers[name])


def lr_factor(step, max_steps, warmup):
    if step <= warmup: return step / max(warmup, 1)
    p = min(max((step - warmup) / max(max_steps - warmup, 1), 0), 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def save_checkpoint(run_dir, step, epoch, model, ema, optimizer, cfg, stats, route, stage):
    path = run_dir / "checkpoints" / f"ckpt_step_{step:05d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    state = {"model": model.state_dict(), "ema_model": ema.state_dict() if ema is not None else None,
             "optimizer": optimizer.state_dict(), "config": cfg, "step": step, "epoch": epoch,
             "normalization": stats, "gate_vc_km": cfg["gate_vc_km"],
             "extinction_constant": cfg["extinction_constant"], "e_min": cfg["extinction_constant"] / cfg["gate_vc_km"],
             "route": route, "stage": stage}
    torch.save(state, tmp)
    os.replace(tmp, path)
    ready_tmp = path.with_suffix(".ready.tmp")
    ready_tmp.write_text("ready\n", encoding="utf-8")
    os.replace(ready_tmp, path.with_suffix(".ready"))
    return path


def completed_full_results(run_dir):
    rows = []
    for path in sorted((run_dir / "validation").glob("full_step_*.json")):
        try: rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception: pass
    return sorted(rows, key=lambda x: x["checkpoint_step"])


def asynchronous_stop(run_dir, cfg, stage):
    rows = completed_full_results(run_dir)
    patience = cfg["async_early_stop_patience_full"]
    if len(rows) <= patience: return False
    scores = [(r["mean_event_AP"], r["mean_event_CSI"]) for r in rows]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return len(rows) - 1 - best >= patience and rows[-1]["checkpoint_step"] >= cfg["minimum_steps_before_async_stop"][stage]


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
    cfg, stats = config(), contract(stage)
    rank, local, world, device = init_dist()
    if world != cfg["training_dcu_count"]: raise RuntimeError("formal R5 training must use the configured training DCU count")
    torch.manual_seed(seed); np.random.seed(seed)
    run_dir = ROOT / "runs" / route / f"seed{seed}" / stage
    if rank == 0:
        if (run_dir / "training_complete.json").exists(): raise RuntimeError(f"completed run exists: {run_dir}")
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    barrier(world)
    train_ds = VisibilityDataset(cfg, stage, "train")
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=seed) if world > 1 else BlockShuffleSampler(len(train_ds), seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False, sampler=sampler,
                              drop_last=True, **loader_kwargs(max(cfg["num_workers"] // world, 0)))
    quick_idx = np.load(ROOT / "contracts" / f"quick_indices_{stage}.npy")
    quick_x = os.environ.get("R5_QUICK_X")
    quick_ds = VisibilityDataset(cfg, stage, "val", quick_idx, x_path_override=quick_x, local_rows=bool(quick_x))
    quick_loader = DataLoader(quick_ds, batch_size=cfg["quick_eval_batch_size"], shuffle=False,
                              **loader_kwargs(cfg["num_workers"]))
    raw_model = R5Model(cfg, route).to(device)
    if init_checkpoint:
        init = torch.load(init_checkpoint, map_location=device)
        raw_model.load_state_dict(init.get("ema_model") if route == "R5-Diffusion" and init.get("ema_model") is not None else init["model"])
    model = DDP(raw_model, device_ids=[local], find_unused_parameters=False) if world > 1 else raw_model
    ema = copy.deepcopy(raw_model).eval() if route == "R5-Diffusion" else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    head_params, backbone_params = [], []
    for name, p in raw_model.named_parameters():
        (head_params if name.startswith(("gate_", "continuous_")) else backbone_params).append(p)
    b_lr = cfg["learning_rate"][f"{stage}_backbone"]
    h_lr = cfg["learning_rate"][f"{stage}_head"]
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": b_lr}, {"params": head_params, "lr": h_lr}], weight_decay=cfg["weight_decay"])
    schedule = cosine_schedule(cfg["diffusion_steps"], device)
    max_steps, warmup = cfg["max_steps"][stage][route], cfg["warmup_steps"][stage]
    train_log = open(run_dir / "train_history.jsonl", "a", buffering=1) if rank == 0 else None
    quick_log = open(run_dir / "quick_validation_history.jsonl", "a", buffering=1) if rank == 0 else None
    validator_ids, step, processed, epoch, started, stop_reason = [], 0, 0, 0, time.time(), "max_steps"
    while step < max_steps:
        sampler.set_epoch(epoch)
        for x, v, _ in train_loader:
            step += 1; processed += cfg["batch_size"] * world
            factor = lr_factor(step, max_steps, warmup)
            optimizer.param_groups[0]["lr"] = b_lr * factor; optimizer.param_groups[1]["lr"] = h_lr * factor
            x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
            model.train(); optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter(); loss, parts = train_loss(raw_model, model, x, v, stats, cfg, schedule)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]); optimizer.step()
            if ema is not None: update_ema(ema, raw_model, cfg["ema_decay"])
            torch.cuda.synchronize(device); step_seconds = time.perf_counter() - t0
            if rank == 0:
                train_log.write(json.dumps({"step": step, "epoch": processed / len(train_ds), "processed_samples": processed,
                    "train_total_loss": float(loss), "loss_components": {k: float(vv) for k, vv in parts.items()},
                    "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                    "step_seconds": step_seconds, "samples_per_second": cfg["batch_size"] * world / step_seconds}) + "\n")
            if step % cfg["quick_val_interval"] == 0:
                eval_model = ema if ema is not None else raw_model
                quick_started = time.perf_counter()
                quick = quick_validate_objective(eval_model, quick_loader, stats, cfg, schedule, seed)
                quick_seconds = time.perf_counter() - quick_started
                if rank == 0:
                    quick_log.write(json.dumps({"step": step, "epoch": processed / len(train_ds), "processed_samples": processed,
                                                "validation_kind": "quick", "wall_seconds": quick_seconds,
                                                "subset_samples": len(quick_ds), **quick}) + "\n")
                model.train()
            if step % cfg["checkpoint_interval"] == 0:
                if rank == 0:
                    ckpt = save_checkpoint(run_dir, step, processed / len(train_ds), raw_model, ema, optimizer, cfg, stats, route, stage)
                    if stage == "S2" and step % cfg["async_full_interval"] == 0 and not validator_already_submitted(run_dir, step):
                        output = run_dir / "validation" / f"full_step_{step:05d}.json"
                        command = ["sbatch", "--parsable", f"--job-name=r5snrv_D_{stage}_s{seed}_{step}",
                                   str(ROOT / "scripts" / "sub_validate.slurm"), route, stage, str(seed), str(ckpt), str(output)]
                        jobid = subprocess.check_output(command, text=True).strip().split(";")[0]
                        validator_ids.append(jobid)
                        with open(run_dir / "validator_jobs.jsonl", "a") as f:
                            f.write(json.dumps({"step": step, "job_id": jobid, "checkpoint": str(ckpt), "output": str(output)}) + "\n")
                    if asynchronous_stop(run_dir, cfg, stage): stop_reason = "async_full_validation_patience"
                holder = [stop_reason]
                if world > 1: dist.broadcast_object_list(holder, src=0)
                stop_reason = holder[0]
                if stop_reason != "max_steps": break
            if step >= max_steps: break
        epoch += 1
        if stop_reason != "max_steps": break
    if rank == 0:
        if step % cfg["checkpoint_interval"] != 0:
            save_checkpoint(run_dir, step, processed / len(train_ds), raw_model, ema, optimizer, cfg, stats, route, stage)
        summary = {"route": route, "stage": stage, "seed": seed, "steps": step, "processed_samples": processed,
                   "epochs": processed / len(train_ds), "wall_seconds": time.time() - started, "stop_reason": stop_reason,
                   "allocated_dcu_count": int(os.environ.get("SLURM_GPUS_ON_NODE", cfg["allocated_dcu_count"])),
                   "training_dcu_count": world, "batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"],
                   "parameter_count": sum(p.numel() for p in raw_model.parameters()), "validator_job_ids": validator_ids}
        atomic_json(run_dir / "training_complete.json", summary)
        train_log.close(); quick_log.close()
        if stage == "S1":
            select_s1_handoff(route, seed, submit_s2=True)
        else:
            dependency = "afterok:" + ":".join(validator_ids) if validator_ids else None
            command = ["sbatch", "--parsable"]
            if dependency: command.append(f"--dependency={dependency}")
            command += ["--job-name=r5snr_finalize_D_S2", str(ROOT / "scripts" / "sub_finalize.slurm"), route, stage, str(seed)]
            finalizer = subprocess.check_output(command, text=True).strip().split(";")[0]
            atomic_json(run_dir / "finalizer_job.json", {"job_id": finalizer, "dependency": dependency})
    barrier(world)
    if world > 1: dist.destroy_process_group()


def select_best(route, stage, seed):
    run_dir = ROOT / "runs" / route / f"seed{seed}" / stage
    rows = completed_full_results(run_dir)
    if not rows: raise RuntimeError(f"no completed full validation results under {run_dir}")
    best = max(rows, key=lambda r: (r["mean_event_AP"], r["mean_event_CSI"]))
    history = run_dir / "validation_history.jsonl"
    tmp = history.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, history)
    result = {"route": route, "stage": stage, "seed": seed, "selection_primary": "mean_event_AP",
              "selection_secondary": "mean_event_CSI", "best_step": best["checkpoint_step"],
              "best_checkpoint": best["checkpoint"], "best_mean_event_AP": best["mean_event_AP"],
              "best_mean_event_CSI": best["mean_event_CSI"], "validated_checkpoints": len(rows)}
    atomic_json(run_dir / "selection.json", result)
    print(json.dumps(result))


def select_s1_handoff(route, seed, submit_s2=False):
    cfg = config()
    run_dir = ROOT / "runs" / route / f"seed{seed}" / "S1"
    rows = [json.loads(line) for line in (run_dir / "quick_validation_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = []
    nonfinite = 0
    for row in rows:
        objective = row.get("validation_objective", row.get("loss"))
        if not objective or not all(math.isfinite(float(value)) for value in objective.values() if value is not None):
            nonfinite += 1
            continue
        step = int(row["step"])
        checkpoint = run_dir / "checkpoints" / f"ckpt_step_{step:05d}.pt"
        if step % cfg["checkpoint_interval"] == 0 and checkpoint.is_file() and checkpoint.with_suffix(".ready").is_file():
            candidates.append((float(objective["total"]), step, objective, checkpoint))
    if not candidates:
        raise RuntimeError("no finite forward-only S1 validation objective has a ready checkpoint")
    total, step, objective, checkpoint = min(candidates, key=lambda item: (item[0], item[1]))
    state = torch.load(checkpoint, map_location="cpu")
    weight_key = "ema_model" if route == "R5-Diffusion" and state.get("ema_model") is not None else "model"
    model = R5Model(cfg, route)
    model.load_state_dict(state[weight_key], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if torch.is_floating_point(value)):
        raise RuntimeError(f"non-finite weights in {checkpoint}")
    selection = {"route": route, "stage": "S1", "seed": seed, "selection_metric": "forward_validation_total_objective",
                 "best_step": step, "best_checkpoint": str(checkpoint), "weights": weight_key,
                 "objective": objective, "quick_rows": len(rows), "nonfinite_objective_rows": nonfinite,
                 "checkpoint_load_strict": True, "weights_all_finite": True, "sampling_used": False}
    atomic_json(run_dir / "handoff_selection.json", selection)
    if submit_s2:
        handoff_path = run_dir / "s2_handoff_job.json"
        lock_path = run_dir / "s2_handoff_submit.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"S2 handoff submission is already claimed: {lock_path}") from error
        try:
            os.write(lock_fd, f"pid={os.getpid()}\n".encode())
            os.close(lock_fd)
            if handoff_path.exists():
                raise RuntimeError(f"S2 handoff was already submitted: {handoff_path}")
            command = ["sbatch", "--parsable", f"--job-name=r5snr_D_S2_s{seed}",
                       str(ROOT / "scripts" / "sub_train.slurm"), route, "S2", str(seed), str(checkpoint)]
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            handoff = {"job_id": job_id, "dependency": None, "checkpoint": str(checkpoint), "weights": weight_key,
                       "selection_metric": selection["selection_metric"]}
            atomic_json(handoff_path, handoff)
            selection["s2_job_id"] = job_id
        except Exception:
            lock_path.unlink(missing_ok=True)
            raise
    print(json.dumps(selection))
    return selection


def best_path(route, stage, seed):
    value = json.loads((ROOT / "runs" / route / f"seed{seed}" / stage / "selection.json").read_text(encoding="utf-8"))
    print(value["best_checkpoint"])


def describe():
    cfg = config()
    result = {}
    for route in ROUTES:
        model = R5Model(cfg, route)
        modules = {
            "DynamicTokenizer": model.dynamic_tokenizer,
            "DynamicEncoder": model.dynamic_encoder,
            "ContextEncoder": model.context_encoder,
            "GateQuery": model.gate_query,
            "GateHead": model.gate_head,
            "ContinuousQuery": model.continuous_query,
            "GaussianHead" if route == "R5-Gaussian" else "DiffusionHead": model.continuous_head,
        }
        result[route] = {"parameters": sum(p.numel() for p in model.parameters()),
                         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         "module_parameters": {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}}
    atomic_json(ROOT / "model_description.json", result)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("--route", choices=ROUTES, required=True); t.add_argument("--stage", choices=("S1", "S2"), required=True); t.add_argument("--seed", type=int, default=1); t.add_argument("--init-checkpoint")
    v = sub.add_parser("validate"); v.add_argument("--route", choices=ROUTES, required=True); v.add_argument("--stage", choices=("S1", "S2"), required=True); v.add_argument("--split", choices=("val", "test"), required=True); v.add_argument("--checkpoint", required=True); v.add_argument("--draws", type=int, required=True); v.add_argument("--output", required=True); v.add_argument("--save-predictions", action="store_true"); v.add_argument("--reverse-steps", type=int)
    h = sub.add_parser("handoff-s1"); h.add_argument("--route", choices=ROUTES, required=True); h.add_argument("--seed", type=int, required=True); h.add_argument("--submit-s2", action="store_true")
    s = sub.add_parser("select"); s.add_argument("--route", choices=ROUTES, required=True); s.add_argument("--stage", choices=("S1", "S2"), required=True); s.add_argument("--seed", type=int, default=1)
    b = sub.add_parser("best-path"); b.add_argument("--route", choices=ROUTES, required=True); b.add_argument("--stage", choices=("S1", "S2"), required=True); b.add_argument("--seed", type=int, default=1)
    sub.add_parser("describe")
    args = ap.parse_args()
    if args.command == "train": train(args.route, args.stage, args.seed, args.init_checkpoint)
    elif args.command == "validate": evaluate(args.route, args.stage, args.split, args.checkpoint, args.draws, args.output, args.save_predictions, args.reverse_steps)
    elif args.command == "handoff-s1": select_s1_handoff(args.route, args.seed, args.submit_s2)
    elif args.command == "select": select_best(args.route, args.stage, args.seed)
    elif args.command == "best-path": best_path(args.route, args.stage, args.seed)
    else: describe()


if __name__ == "__main__":
    main()
