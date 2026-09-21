#!/usr/bin/env python3
"""R2b/R3b/R4b with a physically supported continuous visibility branch."""
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("R2b", "R3b", "R4b")
COMPONENTS = ("cap", "e", "nll", "diff")


def load_cfg():
    with open(ROOT / "configs" / "experiment_perf.json", encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_visibility_km(path):
    y = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if len(y) and float(np.nanmax(y)) >= 100.0:
        y = y / 1000.0
    return y


def masks(v, cap_km=30.0):
    cap = np.isclose(v, cap_km, atol=1e-6)
    continuous = (v > 0.0) & (v < cap_km)
    zero = v == 0.0
    if np.any(cap & continuous) or np.any(zero & continuous) or np.any(cap & zero):
        raise AssertionError("visibility masks overlap")
    return cap, continuous, zero


def prepare_contract(cfg):
    data = Path(cfg["data_dir"])
    out = ROOT / "configs"
    out.mkdir(parents=True, exist_ok=True)
    cached = out / "data_contract.json"
    if cached.is_file() and all((out / f"masks_{s}.npz").is_file() for s in ("train", "val", "test")):
        with open(cached, encoding="utf-8") as f:
            contract = json.load(f)
        if "z_scaler_train_only" not in contract:
            raise RuntimeError("cached contract predates z-space target")
        return contract
    emin = cfg["extinction_constant"] / cfg["cap_km"]
    manifest = {"data_dir": str(data), "e_min": emin, "splits": {}}
    train_z = None
    for split in ("train", "val", "test"):
        y_path, meta_path = data / f"y_{split}.npy", data / f"meta_{split}.csv"
        v = raw_visibility_km(y_path)
        cap, cont, zero = masks(v, cfg["cap_km"])
        np.savez_compressed(out / f"masks_{split}.npz", cap=cap, continuous=cont, zero=zero)
        manifest["splits"][split] = {
            "n": int(len(v)), "y_sha256": sha256(y_path), "meta_sha256": sha256(meta_path),
            "n_cap": int(cap.sum()), "n_continuous": int(cont.sum()), "n_zero": int(zero.sum())
        }
        if split == "train":
            e = cfg["extinction_constant"] / v[cont].astype(np.float64)
            delta = e - emin
            if not np.all(delta > 0):
                raise AssertionError("continuous target violates E > E_min")
            train_z = np.log(delta)
    manifest["z_scaler_train_only"] = {
        "mean": float(train_z.mean(dtype=np.float64)), "std": float(train_z.std(dtype=np.float64))
    }
    manifest["sample_id_columns"] = ["time", "station_id", "lat", "lon"]
    with open(cached, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class FeatureDataset(Dataset):
    def __init__(self, cfg, split, indices=None):
        self.cfg, self.split = cfg, split
        x_dir = Path(os.environ.get("VISCAST_X_DIR", cfg["data_dir"]))
        self.x_path = x_dir / f"X_{split}.npy"
        self.v = raw_visibility_km(Path(cfg["data_dir"]) / f"y_{split}.npy")
        self.indices = np.arange(len(self.v), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.scaler = joblib.load(cfg["p13_scaler"])
        self.x = None
        self.split_dyn = cfg["window_size"] * cfg["dyn_vars"]
        self.log_mask = np.zeros(self.split_dyn, dtype=bool)
        log_vars = [2, 4, 9, cfg["dyn_vars"] - 2, cfg["dyn_vars"] - 1]
        for t in range(cfg["window_size"]):
            self.log_mask[t * cfg["dyn_vars"] + np.asarray(log_vars)] = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode="r")
        idx = int(self.indices[position])
        row = self.x[idx]
        feats = row[: self.split_dyn + 5].astype(np.float32)
        feats[: self.split_dyn] = np.where(
            self.log_mask, np.log1p(np.maximum(feats[: self.split_dyn], 0.0)), feats[: self.split_dyn]
        )
        feats = (feats - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        veg = np.asarray([row[self.split_dyn + 5]], dtype=np.float32)
        extra = row[self.split_dyn + 6:].astype(np.float32)
        final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)])
        return torch.from_numpy(np.nan_to_num(final, nan=0.0)).float(), torch.tensor(self.v[idx]), torch.tensor(idx)


class BalancedBatchSampler(Sampler):
    """Preserve the R2-R4 sampling protocol; do not introduce a new sampler."""
    def __init__(self, visibility, batch_size, batches, seed, rank):
        labels = np.zeros(len(visibility), dtype=np.int8)
        labels[visibility >= 0.5] = 1
        labels[visibility >= 1.0] = 2
        self.pools = [np.flatnonzero(labels == k) for k in range(3)]
        self.bs, self.batches, self.seed, self.rank, self.epoch = batch_size, batches, seed, rank, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 100003 * self.epoch + self.rank)
        n0, n1 = int(round(self.bs * 0.18)), int(round(self.bs * 0.22))
        counts = (n0, n1, self.bs - n0 - n1)
        for _ in range(self.batches):
            batch = np.concatenate([rng.choice(pool, n, replace=len(pool) < n) for pool, n in zip(self.pools, counts)])
            rng.shuffle(batch)
            yield batch.tolist()


def import_p13(path):
    source_dir = str(Path(path).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location("p13_formal_b", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RouteModel(nn.Module):
    def __init__(self, cfg, route):
        super().__init__()
        p13 = import_p13(cfg["p13_source"])
        layout = p13.Layout(window_size=cfg["window_size"], dyn_vars=cfg["dyn_vars"], fe_dim=cfg["fe_dim"])
        self.encoder = p13.StaticRNNLowVisNet(
            layout=layout, encoder="gru", hidden_dim=cfg["hidden_dim"],
            static_hidden_dim=cfg["static_hidden_dim"], fe_hidden_dim=cfg["fe_hidden_dim"],
            fusion_hidden_dim=cfg["fusion_hidden_dim"], veg_emb_dim=cfg["veg_emb_dim"],
            rnn_layers=cfg["rnn_layers"], dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"], pooling=cfg["pooling"], use_fe=True
        )
        raw = torch.load(cfg["p13_checkpoint"], map_location="cpu")
        state = raw.get("model_state_dict", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
        state = {k.removeprefix("module."): v for k, v in state.items()}
        loaded = self.encoder.load_state_dict(state, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"p13 checkpoint mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}")
        for old_head in (self.encoder.class_head, self.encoder.reg_head):
            old_head.requires_grad_(False)
        self.route = route
        h = cfg["fusion_hidden_dim"] // 2
        self.cap = nn.Linear(h, 1)
        if route == "R2b":
            self.ext = nn.Linear(h, 1)
        elif route == "R3b":
            self.dist = nn.Linear(h, 2)
        elif route == "R4b":
            self.time_dim = 64
            self.denoiser = nn.Sequential(
                nn.Linear(h + self.time_dim + 1, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
            )
            betas = cosine_betas(cfg["diffusion_steps"])
            alphas = 1.0 - betas
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bars", torch.cumprod(alphas, 0))
        else:
            raise ValueError(route)

    def encode(self, x):
        e = self.encoder
        split_dyn = e.layout.split_dyn
        split_static = split_dyn + 5
        dyn = x[:, :split_dyn].reshape(-1, e.layout.window_size, e.layout.dyn_vars)
        stat, veg = x[:, split_dyn:split_static], torch.clamp(x[:, split_static].long(), 0, 31)
        extra = x[:, split_static + 1:] if e.fe_encoder is not None else None
        if e.encoder == "mlp":
            dyn_feat = e.dynamic_norm(e.dynamic_proj(dyn[:, -1, :]))
        else:
            dyn_seq, _ = e.rnn(e.dynamic_proj(dyn))
            dyn_feat = e.dynamic_norm(e._pool_dynamic(dyn_seq))
        parts = [dyn_feat, e.static_encoder(torch.cat([stat, e.veg_embedding(veg)], 1))]
        if extra is not None:
            parts.append(e.fe_encoder(extra))
        return e.fusion(torch.cat(parts, 1))

    def forward(self, x, diffusion_noisy=None, diffusion_t=None, diffusion_mask=None):
        z = self.encode(x)
        result = {"cap": self.cap(z).squeeze(1), "z": z}
        if self.route == "R2b":
            result["e_raw"] = self.ext(z).squeeze(1)
        elif self.route == "R3b":
            pars = self.dist(z)
            result.update(mu_s=pars[:, 0], log_sigma_s=pars[:, 1].clamp(-7, 5))
        elif diffusion_noisy is not None:
            result["eps"] = self.predict_noise(diffusion_noisy, diffusion_t, z[diffusion_mask])
        return result

    def predict_noise(self, noisy, t, z):
        return self.denoiser(torch.cat([noisy[:, None], timestep_embedding(t, self.time_dim), z], 1)).squeeze(1)


def cosine_betas(steps, s=0.008):
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    ac = torch.cos(((x / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999).float()


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    emb = t.float()[:, None] * freqs[None]
    return torch.cat([emb.sin(), emb.cos()], 1)


def target_z_scaled(v, zscale, cfg):
    emin = cfg["extinction_constant"] / cfg["cap_km"]
    z = torch.log(cfg["extinction_constant"] / v - emin)
    return (z - zscale["mean"]) / zscale["std"]


def z_scaled_to_e(zs, zscale, cfg):
    """Exact transform: no epsilon, clipping, rejection, or repair."""
    z = zs.to(torch.float64) * zscale["std"] + zscale["mean"]
    return cfg["extinction_constant"] / cfg["cap_km"] + torch.exp(z)


def deterministic_e(raw, cfg):
    return cfg["extinction_constant"] / cfg["cap_km"] + F.softplus(raw.to(torch.float64))


def diffusion_inputs(model, v, zscale, cfg):
    cont = (v > 0) & (v < cfg["cap_km"])
    target = target_z_scaled(v[cont].to(torch.float64), zscale, cfg).float()
    t = torch.randint(0, cfg["diffusion_steps"], (int(cont.sum()),), device=v.device)
    noise = torch.randn(len(t), device=v.device)
    ab = model.alpha_bars[t]
    return torch.sqrt(ab) * target + torch.sqrt(1 - ab) * noise, t, cont, noise


def route_loss(model, output, v, zscale, cfg, diffusion_noise=None):
    cap = torch.isclose(v, torch.tensor(cfg["cap_km"], device=v.device), atol=1e-6)
    cont = (v > 0) & (v < cfg["cap_km"])
    zero = v == 0
    losses = {"cap": F.binary_cross_entropy_with_logits(output["cap"], cap.float())}
    if model.route == "R2b":
        truth_e = cfg["extinction_constant"] / v[cont].to(torch.float64)
        losses["e"] = F.huber_loss(deterministic_e(output["e_raw"][cont], cfg), truth_e)
    elif model.route == "R3b":
        target = target_z_scaled(v[cont].to(torch.float64), zscale, cfg).float()
        sigma = output["log_sigma_s"][cont].exp()
        losses["nll"] = (0.5 * ((target - output["mu_s"][cont]) / sigma) ** 2 + output["log_sigma_s"][cont] + 0.5 * math.log(2 * math.pi)).mean()
    else:
        losses["diff"] = F.mse_loss(output["eps"], diffusion_noise)
    return sum(losses.values()), losses, (cap, cont, zero)


def ddpm_samples(model, z, draws):
    b, device = len(z), z.device
    x = torch.randn(b * draws, device=device)
    cond = z[:, None, :].expand(b, draws, z.shape[1]).reshape(b * draws, z.shape[1])
    for step in reversed(range(len(model.betas))):
        t = torch.full((b * draws,), step, device=device, dtype=torch.long)
        beta, alpha, abar = model.betas[step], model.alphas[step], model.alpha_bars[step]
        mean = (x - beta / torch.sqrt(1 - abar) * model.predict_noise(x, t, cond)) / torch.sqrt(alpha)
        x = mean if step == 0 else mean + beta.sqrt() * torch.randn_like(x)
    return x.view(b, draws)


def class_probs(route, output, model, zscale, cfg, draws):
    if route == "R2b":
        e_samples = deterministic_e(output["e_raw"], cfg)[:, None]
        q0 = (e_samples[:, 0] > 7.824).double()
        q1 = ((e_samples[:, 0] > 3.912) & (e_samples[:, 0] <= 7.824)).double()
        q2 = (e_samples[:, 0] <= 3.912).double()
    elif route == "R3b":
        sigma_s = output["log_sigma_s"].exp()
        zs = output["mu_s"][:, None] + sigma_s[:, None] * torch.randn(len(sigma_s), draws, device=sigma_s.device)
        e_samples = z_scaled_to_e(zs, zscale, cfg)
        emin = cfg["extinction_constant"] / cfg["cap_km"]
        t500 = (math.log(7.824 - emin) - zscale["mean"]) / zscale["std"]
        t1000 = (math.log(3.912 - emin) - zscale["mean"]) / zscale["std"]
        c500 = 0.5 * (1 + torch.erf((t500 - output["mu_s"]) / (sigma_s * math.sqrt(2))))
        c1000 = 0.5 * (1 + torch.erf((t1000 - output["mu_s"]) / (sigma_s * math.sqrt(2))))
        q0, q1, q2 = (1 - c500).double(), (c500 - c1000).double(), c1000.double()
    else:
        e_samples = z_scaled_to_e(ddpm_samples(model, output["z"], draws), zscale, cfg)
        q0 = (e_samples > 7.824).double().mean(1)
        q1 = ((e_samples > 3.912) & (e_samples <= 7.824)).double().mean(1)
        q2 = (e_samples <= 3.912).double().mean(1)
    pcap = output["cap"].sigmoid().double()
    probs = torch.stack([(1 - pcap) * q0, (1 - pcap) * q1, pcap + (1 - pcap) * q2], 1)
    if not torch.allclose(probs.sum(1), torch.ones_like(pcap), atol=1e-10):
        raise AssertionError("class probabilities do not sum to one")
    v_samples = cfg["extinction_constant"] / e_samples
    v_point = pcap * cfg["cap_km"] + (1 - pcap) * v_samples.mean(1)
    return probs, {
        "p_cap": pcap, "e_mean": e_samples.mean(1), "e_median": e_samples.median(1).values,
        "e_q05": e_samples.quantile(0.05, 1), "e_q25": e_samples.quantile(0.25, 1),
        "e_q75": e_samples.quantile(0.75, 1), "e_q95": e_samples.quantile(0.95, 1),
        "v_point": v_point, "samples_e": e_samples, "samples_v": v_samples,
    }


def crps_samples(samples, truth):
    s = samples.sort(1).values
    n = s.shape[1]
    first = (samples - truth[:, None]).abs().mean(1)
    weights = (2 * torch.arange(1, n + 1, device=s.device) - n - 1).double()
    return first - (s * weights).sum(1) / (n * n)


def init_dist():
    world, rank, local = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    return rank, local, world, torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")


def barrier(world):
    if world > 1:
        dist.barrier()


def allreduce_mean(values, device, world):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(t)
        t /= world
    return t.cpu().tolist()


def loader_kwargs(workers):
    kw = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return kw


def event_metrics(y, probs, keep=None):
    if keep is None:
        keep = np.ones(len(y), dtype=bool)
    y, p = y[keep], probs[keep]
    pred = p.argmax(1)
    specs = {"lt500m": (y == 0, pred == 0, p[:, 0]), "500_1000m": (y == 1, pred == 1, p[:, 1]), "lt1000m": (y < 2, pred < 2, p[:, 0] + p[:, 1])}
    out = {}
    for name, (truth, guess, score) in specs.items():
        tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"Recall": recall, "Precision": precision, "CSI": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                     "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                     "AP": float(average_precision_score(truth, score)), "support": int(truth.sum())}
    return out


def gate_metrics(v, pcap, cap_km):
    truth, pred = np.isclose(v, cap_km, atol=1e-6), pcap >= 0.5
    tp, fp, fn, tn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    return {"accuracy": (tp + tn) / len(v), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "AUROC": float(roc_auc_score(truth, pcap)),
            "AP": float(average_precision_score(truth, pcap)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def normal_nll_raw_z(output, v, cont, zscale, cfg):
    target_s = target_z_scaled(v[cont].double(), zscale, cfg)
    mu_s, sigma_s = output["mu_s"][cont].double(), output["log_sigma_s"][cont].double().exp()
    nll_s = 0.5 * ((target_s - mu_s) / sigma_s) ** 2 + torch.log(sigma_s) + 0.5 * math.log(2 * math.pi)
    return nll_s + math.log(zscale["std"])


@torch.no_grad()
def validate(model, loader, route, zscale, cfg, device, rank, world, seed, draws, amp_mode="none"):
    started = time.perf_counter()
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(900000 + seed + rank)
    total = count = crps_e_sum = crps_v_sum = crps_n = nll_sum = nll_n = 0.0
    part_sums = {k: 0.0 for k in COMPONENTS}
    local_idx, local_probs = [], []
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode)
    for x, v, idx in loader:
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
        with amp_context:
            if route == "R4b":
                noisy, t, cont, noise = diffusion_inputs(model, v, zscale, cfg)
                output = model(x, noisy, t, cont)
            else:
                noise, output = None, model(x)
            loss, parts, masks_here = route_loss(model, output, v, zscale, cfg, noise)
            probs, stats = class_probs(route, output, model, zscale, cfg, draws if route != "R2b" else 1)
        total += float(loss) * len(v)
        count += len(v)
        for key, value in parts.items():
            part_sums[key] += float(value) * len(v)
        cont = masks_here[1]
        if cont.any() and route in ("R3b", "R4b"):
            truth_e, truth_v = cfg["extinction_constant"] / v[cont].double(), v[cont].double()
            crps_e_sum += float(crps_samples(stats["samples_e"][cont], truth_e).sum())
            crps_v_sum += float(crps_samples(stats["samples_v"][cont], truth_v).sum())
            crps_n += int(cont.sum())
            if route == "R3b":
                nll_sum += float(normal_nll_raw_z(output, v, cont, zscale, cfg).sum())
                nll_n += int(cont.sum())
        local_idx.append(idx.numpy())
        local_probs.append(probs.cpu().numpy())
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    totals = torch.tensor([total, count, crps_e_sum, crps_v_sum, crps_n, nll_sum, nll_n] + [part_sums[k] for k in COMPONENTS], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(totals)
    payload = (np.concatenate(local_idx), np.concatenate(local_probs))
    gathered = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]
    result = None
    if rank == 0:
        idx = np.concatenate([g[0] for g in gathered])
        probs = np.concatenate([g[1] for g in gathered])
        order = np.argsort(idx)
        idx, probs = idx[order], probs[order]
        v_all = loader.dataset.v[idx]
        y = np.zeros(len(v_all), np.int8)
        y[v_all >= 0.5], y[v_all >= 1.0] = 1, 2
        metrics = event_metrics(y, probs)
        vals = totals.cpu().numpy()
        result = {"loss": {"total": vals[0] / max(vals[1], 1.0), **{k: vals[7 + i] / max(vals[1], 1.0) for i, k in enumerate(COMPONENTS)}},
                  "events": metrics,
                  "mean_event_AP": float(np.mean([metrics[k]["AP"] for k in ("lt500m", "500_1000m", "lt1000m")])),
                  "mean_event_CSI": float(np.mean([metrics[k]["CSI"] for k in ("lt500m", "500_1000m", "lt1000m")])),
                  "NLL_z": vals[5] / max(vals[6], 1.0) if route == "R3b" else None,
                  "CRPS_E": vals[2] / max(vals[4], 1.0) if route in ("R3b", "R4b") else None,
                  "CRPS_V_km": vals[3] / max(vals[4], 1.0) if route in ("R3b", "R4b") else None,
                  "draws": draws if route != "R2b" else 1,
                  "validation_samples": int(len(v_all)),
                  "reverse_steps": int(cfg["diffusion_steps"]) if route == "R4b" else 0,
                  "draw_parallelism": "vectorized_batch_dimension" if route == "R4b" else None,
                  "validation_seconds": time.perf_counter() - started}
    if world > 1:
        holder = [result]
        dist.broadcast_object_list(holder, src=0)
        result = holder[0]
    return result


def set_stage(model, frozen):
    for p in model.encoder.parameters():
        p.requires_grad_(not frozen)
    for old_head in (model.encoder.class_head, model.encoder.reg_head):
        old_head.requires_grad_(False)


def lr_factor(step, max_steps, warmup):
    if step <= warmup:
        return step / max(warmup, 1)
    progress = min(max((step - warmup) / max(max_steps - warmup, 1), 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def ema_update(ema_model, model, decay):
    source = dict(model.named_parameters())
    for name, target in ema_model.named_parameters():
        target.mul_(decay).add_(source[name], alpha=1 - decay)
    source_buffers = dict(model.named_buffers())
    for name, target in ema_model.named_buffers():
        target.copy_(source_buffers[name])


def train(route, seed, batch_size, total_workers, amp_mode="none"):
    cfg = load_cfg()
    rank, local, world, device = init_dist()
    contract = prepare_contract(cfg) if rank == 0 else None
    barrier(world)
    if rank != 0:
        with open(ROOT / "configs" / "data_contract.json") as f:
            contract = json.load(f)
    torch.manual_seed(seed + rank * 100003)
    np.random.seed(seed + rank * 100003)
    workers = max(total_workers // world, 0)
    train_ds = FeatureDataset(cfg, "train")
    val_size = len(raw_visibility_km(Path(cfg["data_dir"]) / "y_val.npy"))
    val_ds = FeatureDataset(cfg, "val", indices=np.arange(rank, val_size, world))
    fast_rng = np.random.default_rng(20260913)
    fast_global = np.sort(fast_rng.choice(val_size, size=min(cfg["fast_val_samples"], val_size), replace=False))
    fast_ds = FeatureDataset(cfg, "val", indices=fast_global[rank::world])
    max_steps = int(cfg["max_steps"][route])
    sampler = BalancedBatchSampler(train_ds.v, batch_size, max_steps, seed, rank)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs(workers))
    val_loader = DataLoader(val_ds, batch_size=cfg["eval_batch_size_per_rank"], shuffle=False, **loader_kwargs(workers))
    fast_val_loader = DataLoader(fast_ds, batch_size=cfg["eval_batch_size_per_rank"], shuffle=False, **loader_kwargs(workers))
    raw_model = RouteModel(cfg, route).to(device)
    set_stage(raw_model, frozen=True)
    head_names = ("ext", "cap", "dist", "denoiser")
    head, backbone = [], []
    for name, p in raw_model.named_parameters():
        (head if name.split(".")[0] in head_names else backbone).append(p)
    effective_batch = batch_size * world
    head_lr = cfg["head_lr"] * math.sqrt(effective_batch / 512.0)
    encoder_lr = head_lr * cfg["encoder_lr_ratio"]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": 0.0}, {"params": head, "lr": head_lr}], weight_decay=cfg["weight_decay"])
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_mode == "fp16")
    model = DDP(raw_model, device_ids=[local], find_unused_parameters=True) if world > 1 else raw_model
    ema_model = copy.deepcopy(raw_model).eval() if route == "R4b" else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)
    run_dir = ROOT / "runs" / route / f"seed{seed}"
    if rank == 0:
        if (run_dir / "checkpoints" / "best.pt").exists():
            raise RuntimeError(f"refusing to overwrite completed run: {run_dir}")
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        runtime_cfg = dict(cfg)
        runtime_cfg.update(route=route, seed=seed, allocated_dcu_count=int(os.environ.get("ALLOCATED_DCU_COUNT", world)),
                           used_dcu_count=world, cpu_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", total_workers)),
                           num_workers_total=total_workers, num_workers_per_rank=workers, batch_size_per_rank=batch_size,
                           effective_batch_size=effective_batch, actual_head_lr=head_lr, actual_encoder_lr=encoder_lr,
                           amp_mode=amp_mode,
                           node=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"))
        with open(run_dir / "config.json", "w") as f:
            json.dump(runtime_cfg, f, indent=2)
        train_log, val_log = open(run_dir / "train_steps.jsonl", "a", buffering=1), open(run_dir / "validation.jsonl", "a", buffering=1)
    else:
        train_log = val_log = None
    zscale = contract["z_scaler_train_only"]
    best_ap, best_csi, best_step, last_improve_step = -float("inf"), -float("inf"), None, cfg["stage_a_steps"]
    step, start, stage_b, stop_reason = 0, time.time(), False, "max_steps"
    previous_end = time.perf_counter()
    train_active_seconds = 0.0
    data_wait_seconds = 0.0
    for x, v, _ in train_loader:
        data_wait = time.perf_counter() - previous_end
        step += 1
        if step == cfg["stage_a_steps"] + 1:
            set_stage(raw_model, frozen=False)
            if world > 1:
                barrier(world)
                model = DDP(raw_model, device_ids=[local], find_unused_parameters=True)
            stage_b, last_improve_step = True, step - 1
        stage = "B" if stage_b else "A"
        model.train()
        if not stage_b:
            raw_model.encoder.eval()
        factor = lr_factor(step, max_steps, cfg["warmup_steps"])
        optimizer.param_groups[0]["lr"] = encoder_lr * factor if stage_b else 0.0
        optimizer.param_groups[1]["lr"] = head_lr * factor
        t0 = time.perf_counter()
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        raw_for_loss = model.module if isinstance(model, DDP) else model
        if route == "R4b":
            noisy, t, cont, noise = diffusion_inputs(raw_for_loss, v, zscale, cfg)
        else:
            noise = None
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
        with amp_context:
            output = model(x, noisy, t, cont) if route == "R4b" else model(x)
            loss, parts, _ = route_loss(raw_for_loss, output, v, zscale, cfg, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer); scaler.update()
        if ema_model is not None:
            ema_update(ema_model, raw_for_loss, cfg["ema_decay"])
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - t0
        train_active_seconds += step_seconds + data_wait
        data_wait_seconds += data_wait
        values = [float(loss.detach())] + [float(parts[k].detach()) if k in parts else float("nan") for k in COMPONENTS]
        reduced = allreduce_mean(values, device, world)
        if rank == 0:
            train_log.write(json.dumps({"step": step, "stage": stage, "train_total_loss": reduced[0],
                "loss_components": {k: reduced[i + 1] for i, k in enumerate(COMPONENTS) if not math.isnan(reduced[i + 1])},
                "lr_encoder": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                "dataloader_wait_seconds_rank0": data_wait, "step_seconds_rank0": step_seconds,
                "samples_per_second_effective": effective_batch / (step_seconds + data_wait)}) + "\n")
        if step % cfg["fast_val_interval"] == 0:
            eval_model = ema_model if ema_model is not None else raw_for_loss
            is_full = step % cfg["full_val_interval"] == 0
            active_loader = val_loader if is_full else fast_val_loader
            active_draws = cfg["validation_draws"] if is_full else cfg["fast_validation_draws"]
            val = validate(eval_model, active_loader, route, zscale, cfg, device, rank, world, seed, active_draws, amp_mode)
            val_row = {"step": step, "epoch": step / cfg["steps_per_epoch"],
                       "processed_samples": step * effective_batch, "stage": stage,
                       "validation_kind": "full" if is_full else "fast",
                       "minutes": (time.time() - start) / 60, **val}
            score_ap, score_csi = val["mean_event_AP"], val["mean_event_CSI"]
            improved = is_full and (score_ap > best_ap + 1e-12 or (abs(score_ap - best_ap) <= 1e-12 and score_csi > best_csi))
            if improved:
                best_ap, best_csi, best_step, last_improve_step = score_ap, score_csi, step, step
            if rank == 0:
                val_log.write(json.dumps(val_row) + "\n")
                print(json.dumps({"validation": val_row, "improved": improved}), flush=True)
                state = {"model": raw_for_loss.state_dict(), "ema_model": ema_model.state_dict() if ema_model is not None else None,
                         "route": route, "seed": seed, "best_step": best_step, "best_epoch": best_step / cfg["steps_per_epoch"],
                         "selection": {"mean_event_AP": best_ap, "mean_event_CSI": best_csi}, "z_scaler": zscale, "config": runtime_cfg}
                torch.save(state, run_dir / "checkpoints" / "last.pt")
                if improved:
                    torch.save(state, run_dir / "checkpoints" / "best.pt")
            if is_full and stage_b and step - last_improve_step >= cfg["early_stop_no_improve_steps"]:
                stop_reason = f"early_stop_no_mean_event_AP_improvement_{cfg['early_stop_no_improve_steps']}_steps"
                break
            model.train()
            if not stage_b:
                raw_model.encoder.eval()
            previous_end = time.perf_counter()
        if step >= max_steps:
            break
        previous_end = time.perf_counter()
    if rank == 0:
        with open(run_dir / "training_summary.json", "w") as f:
            json.dump({"route": route, "seed": seed, "allocated_dcu_count": int(os.environ.get("ALLOCATED_DCU_COUNT", world)),
                       "used_dcu_count": world, "cpu_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", total_workers)),
                       "num_workers_total": total_workers, "batch_size_per_rank": batch_size, "effective_batch_size": effective_batch,
                       "head_lr": head_lr, "encoder_lr": encoder_lr, "wall_seconds": time.time() - start,
                       "samples_per_second": effective_batch * step / max(train_active_seconds, 1e-9),
                       "mean_dataloader_wait_seconds": data_wait_seconds / max(step, 1),
                       "best_step": best_step, "best_mean_event_AP": best_ap, "best_mean_event_CSI": best_csi,
                       "stopped_step": step, "processed_samples": step * effective_batch,
                       "stop_reason": stop_reason}, f, indent=2)
        train_log.close(); val_log.close()
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


class ResourceMonitor:
    def __init__(self):
        self.stop_event, self.cpu, self.rocm = threading.Event(), [], []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self.thread.start()
    def stop(self): self.stop_event.set(); self.thread.join(timeout=10)

    def _run(self):
        try:
            import psutil
        except Exception:
            psutil = None
        while not self.stop_event.is_set():
            if psutil is not None:
                self.cpu.append(float(psutil.cpu_percent(interval=None)))
            try:
                p = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--json"], capture_output=True, text=True, timeout=10)
                if p.returncode == 0: self.rocm.append(p.stdout)
            except Exception:
                pass
            self.stop_event.wait(2.0)

    def summary(self):
        numeric = {}
        for raw in self.rocm:
            try: data = json.loads(raw)
            except Exception: continue
            stack = [("", data)]
            while stack:
                prefix, value = stack.pop()
                if isinstance(value, dict): stack.extend((f"{prefix}/{k}", v) for k, v in value.items())
                elif isinstance(value, (int, float)): numeric.setdefault(prefix, []).append(float(value))
                elif isinstance(value, str):
                    match = re.search(r"-?\d+(?:\.\d+)?", value)
                    if match:
                        numeric.setdefault(prefix, []).append(float(match.group(0)))
        return {"cpu_utilization_percent_mean": float(np.mean(self.cpu)) if self.cpu else None,
                "cpu_utilization_percent_max": float(np.max(self.cpu)) if self.cpu else None,
                "rocm_smi_numeric_mean": {k: float(np.mean(v)) for k, v in numeric.items()}, "rocm_smi_sample_count": len(self.rocm),
                "rocm_smi_first_raw": self.rocm[0][:4000] if self.rocm else None}


def benchmark(route, seed, batch_size, total_workers, tag, output_dir, amp_mode="none"):
    cfg = load_cfg()
    rank, local, world, device = init_dist()
    contract = prepare_contract(cfg) if rank == 0 else None
    barrier(world)
    if rank != 0:
        with open(ROOT / "configs" / "data_contract.json") as f: contract = json.load(f)
    torch.manual_seed(seed + rank * 100003); np.random.seed(seed + rank * 100003)
    workers = max(total_workers // world, 0)
    ds = FeatureDataset(cfg, "train")
    benchmark_pool = min(int(os.environ.get("VISCAST_BENCHMARK_POOL", len(ds.v))), len(ds.v))
    ds.indices = np.arange(benchmark_pool, dtype=np.int64)
    total_steps = cfg["benchmark_warmup_steps"] + cfg["benchmark_steps"]
    loader = DataLoader(ds, batch_sampler=BalancedBatchSampler(ds.v[:benchmark_pool], batch_size, total_steps, seed, rank), **loader_kwargs(workers))
    raw_model = RouteModel(cfg, route).to(device); set_stage(raw_model, frozen=False)
    model = DDP(raw_model, device_ids=[local], find_unused_parameters=True) if world > 1 else raw_model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["head_lr"], weight_decay=cfg["weight_decay"])
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_mode == "fp16")
    iterator, wait_times, step_times = iter(loader), [], []
    monitor = ResourceMonitor() if rank == 0 else None
    if monitor: monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    measured_start = None
    for i in range(total_steps):
        t0 = time.perf_counter(); x, v, _ = next(iterator); wait = time.perf_counter() - t0; t1 = time.perf_counter()
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); raw_for_loss = model.module if isinstance(model, DDP) else model
        noisy, t, cont, noise = diffusion_inputs(raw_for_loss, v, contract["z_scaler_train_only"], cfg)
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
        with amp_context:
            output = model(x, noisy, t, cont)
            loss, _, _ = route_loss(raw_for_loss, output, v, contract["z_scaler_train_only"], cfg, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite benchmark loss under amp={amp_mode}")
        elapsed = time.perf_counter() - t1
        if i == cfg["benchmark_warmup_steps"] - 1: measured_start = time.perf_counter()
        elif i >= cfg["benchmark_warmup_steps"]: wait_times.append(wait); step_times.append(elapsed + wait)
    measured_seconds = time.perf_counter() - measured_start
    if monitor: monitor.stop()
    values = [measured_seconds, np.mean(step_times), np.mean(wait_times), torch.cuda.max_memory_allocated(device)]
    gathered = [None] * world if rank == 0 else None
    if world > 1: dist.gather_object(values, gathered, dst=0)
    else: gathered = [values]
    if rank == 0:
        wall = max(v[0] for v in gathered)
        result = {"status": "ok", "tag": tag, "route": route, "seed": seed, "dcu_count": world,
                  "cpu_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", total_workers)), "num_workers_total": total_workers,
                  "benchmark_pool_samples": benchmark_pool,
                  "num_workers_per_rank": workers, "batch_size_per_rank": batch_size, "effective_batch_size": batch_size * world,
                  "measured_steps": cfg["benchmark_steps"], "samples_per_second": batch_size * world * cfg["benchmark_steps"] / wall,
                  "step_time_seconds_mean": float(np.mean([v[1] for v in gathered])),
                  "dataloader_wait_seconds_mean": float(np.mean([v[2] for v in gathered])),
                  "amp_mode": amp_mode, "final_loss": float(loss.detach()), "loss_finite": bool(torch.isfinite(loss)),
                  "device_memory_bytes_max_per_rank": int(max(v[3] for v in gathered)), "monitor": monitor.summary()}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{tag}.json", "w") as f: json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)
    barrier(world)
    if world > 1: dist.destroy_process_group()


@torch.no_grad()
def evaluate(route, seed, total_workers, amp_mode="none"):
    cfg = load_cfg(); rank, local, world, device = init_dist()
    with open(ROOT / "configs" / "data_contract.json") as f: contract = json.load(f)
    torch.manual_seed(1900000 + seed + rank)
    draws = cfg["test_draws"] if route in ("R3b", "R4b") else 1
    eval_batch = max(1, cfg["eval_batch_size_per_rank"] * min(64, draws) // draws)
    test_v = raw_visibility_km(Path(cfg["data_dir"]) / "y_test.npy")
    ds = FeatureDataset(cfg, "test", indices=np.arange(rank, len(test_v), world))
    loader = DataLoader(ds, batch_size=eval_batch, shuffle=False, **loader_kwargs(max(total_workers // world, 0)))
    run_dir = ROOT / "runs" / route / f"seed{seed}"
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model = RouteModel(cfg, route).to(device)
    model.load_state_dict(ckpt["ema_model"] if route == "R4b" else ckpt["model"]); model.eval()
    zscale = ckpt["z_scaler"]
    keys = ("index", "probs", "p_cap", "e_mean", "e_median", "e_q05", "e_q25", "e_q75", "e_q95", "v_point")
    collected = {k: [] for k in keys}
    crps_e_sum = crps_v_sum = nll_sum = 0.0; crps_n = nll_n = bad_e = bad_v = 0
    emin, min_e, max_v, start = cfg["extinction_constant"] / cfg["cap_km"], float("inf"), -float("inf"), time.time()
    for x, v, idx in loader:
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode)
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
        with amp_context:
            output = model(x)
            probs, stats = class_probs(route, output, model, zscale, cfg, draws)
        cont = (v > 0) & (v < cfg["cap_km"])
        if cont.any() and route in ("R3b", "R4b"):
            crps_e_sum += float(crps_samples(stats["samples_e"][cont], cfg["extinction_constant"] / v[cont].double()).sum())
            crps_v_sum += float(crps_samples(stats["samples_v"][cont], v[cont].double()).sum()); crps_n += int(cont.sum())
            if route == "R3b": nll_sum += float(normal_nll_raw_z(output, v, cont, zscale, cfg).sum()); nll_n += int(cont.sum())
        min_e, max_v = min(min_e, float(stats["samples_e"].min())), max(max_v, float(stats["samples_v"].max()))
        bad_e += int((stats["samples_e"] <= emin).sum()); bad_v += int((stats["samples_v"] >= cfg["cap_km"]).sum())
        collected["index"].append(idx.numpy()); collected["probs"].append(probs.cpu().numpy())
        for k in keys[2:]: collected[k].append(stats[k].cpu().numpy())
    payload = {k: np.concatenate(v) for k, v in collected.items()}
    scalar = {"crps_e_sum": crps_e_sum, "crps_v_sum": crps_v_sum, "crps_n": crps_n, "nll_sum": nll_sum, "nll_n": nll_n,
              "min_e": min_e, "max_v": max_v, "bad_e": bad_e, "bad_v": bad_v, "wall": time.time() - start}
    gp, gs = ([None] * world if rank == 0 else None), ([None] * world if rank == 0 else None)
    if world > 1: dist.gather_object(payload, gp, dst=0); dist.gather_object(scalar, gs, dst=0)
    else: gp, gs = [payload], [scalar]
    if rank == 0:
        arrays = {k: np.concatenate([p[k] for p in gp]) for k in keys}; order = np.argsort(arrays["index"]); arrays = {k: v[order] for k, v in arrays.items()}
        if not np.array_equal(arrays["index"], np.arange(len(test_v))): raise AssertionError("test sample index contract violated")
        probs, vpoint, pcap = arrays["probs"], arrays["v_point"], arrays["p_cap"]
        y = np.zeros(len(test_v), np.int8); y[test_v >= 0.5], y[test_v >= 1.0] = 1, 2
        positive, err = test_v > 0, vpoint - test_v
        cn, nn = sum(s["crps_n"] for s in gs), sum(s["nll_n"] for s in gs)
        result = {"route": route, "seed": seed, "best_step": ckpt["best_step"], "best_epoch": ckpt["best_epoch"], "n": len(test_v), "draws": draws,
                  "metrics_including_v0": event_metrics(y, probs), "metrics_excluding_v0": event_metrics(y, probs, positive),
                  "gate": gate_metrics(test_v, pcap, cfg["cap_km"]),
                  "continuous": {"MAE_km": float(np.abs(err[positive]).mean()), "RMSE_km": float(np.sqrt(np.mean(err[positive] ** 2))),
                                 "MAE_v_lt_1km": float(np.abs(err[(test_v > 0) & (test_v < 1)]).mean()),
                                 "MAE_v_lt_0p5km": float(np.abs(err[(test_v > 0) & (test_v < 0.5)]).mean())},
                  "NLL_z": sum(s["nll_sum"] for s in gs) / max(nn, 1) if route == "R3b" else None,
                  "CRPS_E": sum(s["crps_e_sum"] for s in gs) / max(cn, 1) if route in ("R3b", "R4b") else None,
                  "CRPS_V_km": sum(s["crps_v_sum"] for s in gs) / max(cn, 1) if route in ("R3b", "R4b") else None,
                  "support_checks": {"E_min_theoretical": emin, "generated_E_min": min(s["min_e"] for s in gs), "generated_V_max": max(s["max_v"] for s in gs),
                                     "count_E_le_E_min": sum(s["bad_e"] for s in gs), "count_V_ge_30": sum(s["bad_v"] for s in gs),
                                     "negative_E_rejection": False, "softplus_repair": False, "hard_V_clipping": False},
                  "probability_sum_max_abs_error": float(np.max(np.abs(probs.sum(1) - 1))), "test_wall_seconds": max(s["wall"] for s in gs),
                  "resource_config": ckpt["config"]}
        np.savez_compressed(run_dir / "eval" / "test_predictions.npz", **arrays)
        with open(run_dir / "eval" / "metrics.json", "w") as f: json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)
    barrier(world)
    if world > 1: dist.destroy_process_group()


@torch.no_grad()
def sanity():
    cfg = load_cfg(); contract = prepare_contract(cfg); _, _, _, device = init_dist()
    train_v = raw_visibility_km(Path(cfg["data_dir"]) / "y_train.npy"); cap, cont, zero = masks(train_v, cfg["cap_km"])
    selected = np.concatenate([np.flatnonzero(cap)[:6], np.flatnonzero(cont)[:6], np.flatnonzero(zero)[:6]])
    ds = FeatureDataset(cfg, "train", indices=selected); x, v, _ = next(iter(DataLoader(ds, batch_size=len(selected)))); x, v = x.to(device), v.to(device)
    zscale, emin = contract["z_scaler_train_only"], cfg["extinction_constant"] / cfg["cap_km"]
    checks = {"E_min": emin, "V1_to_E": cfg["extinction_constant"], "V0p5_to_E": cfg["extinction_constant"] / 0.5,
              "V0p1_to_E": cfg["extinction_constant"] / 0.1, "routes": {}}
    for route in ROUTES:
        model = RouteModel(cfg, route).to(device).eval()
        if route == "R4b": noisy, t, cmask, noise = diffusion_inputs(model, v, zscale, cfg); output = model(x, noisy, t, cmask)
        else: noise, output = None, model(x)
        loss, parts, mm = route_loss(model, output, v, zscale, cfg, noise); probs, stats = class_probs(route, output, model, zscale, cfg, 32 if route != "R2b" else 1)
        checks["routes"][route] = {"loss": float(loss), "parts": {k: float(z) for k, z in parts.items()}, "mask_counts": [int(z.sum()) for z in mm],
            "prob_sum_error": float((probs.sum(1) - 1).abs().max()), "generated_E_min": float(stats["samples_e"].min()),
            "generated_V_max": float(stats["samples_v"].max()), "count_E_le_E_min": int((stats["samples_e"] <= emin).sum()),
            "count_V_ge_30": int((stats["samples_v"] >= cfg["cap_km"]).sum())}
    checks["sanity_tolerance"] = 1e-12
    checks["passed"] = all(
        r["prob_sum_error"] < 1e-10
        and math.isfinite(r["generated_E_min"])
        and math.isfinite(r["generated_V_max"])
        and r["generated_E_min"] >= emin - checks["sanity_tolerance"]
        and r["generated_V_max"] <= cfg["cap_km"] + checks["sanity_tolerance"]
        for r in checks["routes"].values()
    )
    (ROOT / "eval").mkdir(exist_ok=True)
    with open(ROOT / "eval" / "sanity.json", "w") as f: json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))
    if not checks["passed"]: raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["sanity", "benchmark", "train", "eval"])
    ap.add_argument("--route", choices=ROUTES, default="R4b"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512); ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    ap.add_argument("--benchmark-tag", default="benchmark"); ap.add_argument("--output-dir", default=str(ROOT / "benchmark")); args = ap.parse_args()
    if args.command == "sanity": sanity()
    elif args.command == "benchmark": benchmark(args.route, args.seed, args.batch_size, args.num_workers, args.benchmark_tag, args.output_dir, args.amp)
    elif args.command == "train": train(args.route, args.seed, args.batch_size, args.num_workers, args.amp)
    else: evaluate(args.route, args.seed, args.num_workers, args.amp)


if __name__ == "__main__": main()
