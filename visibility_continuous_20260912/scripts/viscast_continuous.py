#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]


def load_cfg():
    with open(ROOT / "configs/experiment.json", encoding="utf-8") as f:
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


def masks(v):
    cap = np.isclose(v, 30.0, atol=1e-6)
    continuous = (v > 0.0) & (v < 30.0)
    zero = v == 0.0
    if np.any(cap & continuous) or np.any(zero & continuous) or np.any(cap & zero):
        raise AssertionError("visibility masks overlap")
    return cap, continuous, zero


def prepare_contract(cfg):
    data = Path(cfg["data_dir"])
    out = ROOT / "configs"
    out.mkdir(parents=True, exist_ok=True)
    cached = out / "data_contract.json"
    if cached.is_file() and all((out / f"masks_{split}.npz").is_file() for split in ("train", "val", "test")):
        with open(cached, encoding="utf-8") as f:
            return json.load(f)
    manifest = {"data_dir": str(data), "splits": {}}
    train_e = None
    for split in ("train", "val", "test"):
        y_path, meta_path = data / f"y_{split}.npy", data / f"meta_{split}.csv"
        v = raw_visibility_km(y_path)
        cap, cont, zero = masks(v)
        np.savez_compressed(out / f"masks_{split}.npz", cap=cap, continuous=cont, zero=zero)
        manifest["splits"][split] = {
            "n": int(len(v)), "y_sha256": sha256(y_path), "meta_sha256": sha256(meta_path),
            "n_cap": int(cap.sum()), "n_continuous": int(cont.sum()), "n_zero": int(zero.sum())
        }
        if split == "train":
            train_e = cfg["extinction_constant"] / v[cont]
    scaler = {"mean": float(train_e.mean(dtype=np.float64)), "std": float(train_e.std(dtype=np.float64))}
    if not scaler["std"] > 0:
        raise AssertionError("invalid extinction scaler")
    manifest["extinction_scaler_train_only"] = scaler
    manifest["sample_id_columns"] = ["time", "station_id", "lat", "lon"]
    with open(out / "data_contract.json", "w", encoding="utf-8") as f:
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
        extra = row[self.split_dyn + 6 :].astype(np.float32)
        final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)])
        final = np.nan_to_num(final, nan=0.0)
        return torch.from_numpy(final).float(), torch.tensor(self.v[idx]), torch.tensor(idx)


class BalancedBatchSampler(Sampler):
    def __init__(self, visibility, batch_size, batches, seed, rank, world):
        labels = np.zeros(len(visibility), dtype=np.int8)
        labels[visibility >= 0.5] = 1
        labels[visibility >= 1.0] = 2
        self.pools = [np.flatnonzero(labels == k) for k in range(3)]
        self.bs, self.batches, self.seed, self.rank, self.world, self.epoch = batch_size, batches, seed, rank, world, 0

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
    spec = importlib.util.spec_from_file_location("p13_formal", path)
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
        # These are the retired p13 output heads, not part of the shared encoder contract.
        for old_head in (self.encoder.class_head, self.encoder.reg_head):
            old_head.requires_grad_(False)
        self.route = route
        h = cfg["fusion_hidden_dim"] // 2
        if route == "R1":
            self.ext = nn.Linear(h, 1)
        elif route == "R2":
            self.cap, self.ext = nn.Linear(h, 1), nn.Linear(h, 1)
        elif route == "R3":
            self.cap, self.dist, self.low = nn.Linear(h, 1), nn.Linear(h, 2), nn.Linear(h, 1)
        elif route == "R4":
            self.cap = nn.Linear(h, 1)
            self.time_dim = 64
            self.denoiser = nn.Sequential(nn.Linear(h + self.time_dim + 1, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
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
        if self.route == "R1": return {"e_raw": self.ext(z).squeeze(1), "z": z}
        if self.route == "R2": return {"cap": self.cap(z).squeeze(1), "e_raw": self.ext(z).squeeze(1), "z": z}
        if self.route == "R3":
            pars = self.dist(z)
            return {"cap": self.cap(z).squeeze(1), "mu": pars[:, 0], "log_sigma": pars[:, 1].clamp(-7, 5), "low": self.low(z).squeeze(1), "z": z}
        result = {"cap": self.cap(z).squeeze(1), "z": z}
        if diffusion_noisy is not None:
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


def positive_e(raw, scale):
    physical = F.softplus(raw) + 1e-5
    return physical, (physical - scale["mean"]) / scale["std"]


def diffusion_inputs(model, v, scale, cfg):
    cont = (v > 0) & (v < cfg["cap_km"])
    target = (cfg["extinction_constant"] / v[cont] - scale["mean"]) / scale["std"]
    t = torch.randint(0, cfg["diffusion_steps"], (int(cont.sum()),), device=v.device)
    noise = torch.randn(len(t), device=v.device)
    ab = model.alpha_bars[t]
    return torch.sqrt(ab) * target + torch.sqrt(1 - ab) * noise, t, cont, noise


def route_loss(model, output, v, scale, cfg, diffusion_noise=None):
    cap = torch.isclose(v, torch.tensor(cfg["cap_km"], device=v.device), atol=1e-6)
    cont = (v > 0) & (v < cfg["cap_km"])
    zero = v == 0
    target_e = torch.zeros_like(v)
    target_e[cont] = cfg["extinction_constant"] / v[cont]
    target_s = (target_e - scale["mean"]) / scale["std"]
    losses = {}
    if model.route in ("R2", "R3", "R4"):
        losses["cap"] = F.binary_cross_entropy_with_logits(output["cap"], cap.float())
    if model.route in ("R1", "R2"):
        _, pred_s = positive_e(output["e_raw"], scale)
        losses["e"] = F.huber_loss(pred_s[cont], target_s[cont])
    elif model.route == "R3":
        sigma = output["log_sigma"].exp()
        losses["nll"] = (0.5 * ((target_s[cont] - output["mu"][cont]) / sigma[cont]) ** 2 + output["log_sigma"][cont] + 0.5 * math.log(2 * math.pi)).mean()
        losses["low"] = F.binary_cross_entropy_with_logits(output["low"], (v < 1.0).float())
    elif model.route == "R4":
        losses["diff"] = F.mse_loss(output["eps"], diffusion_noise)
    if model.route == "R1": total = losses["e"]
    elif model.route == "R2": total = losses["cap"] + cfg["lambda_e"] * losses["e"]
    elif model.route == "R3": total = losses["cap"] + cfg["lambda_dist"] * losses["nll"] + cfg["lambda_lowvis"] * losses["low"]
    else: total = losses["cap"] + cfg["lambda_dist"] * losses["diff"]
    return total, losses, (cap, cont, zero)


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


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    return rank, local, world, torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")


def reduce_mean(value, device, world):
    t = torch.tensor([float(value), 1.0], device=device)
    if world > 1: dist.all_reduce(t)
    return float(t[0] / t[1])


@torch.no_grad()
def validate(model, loader, scale, cfg, device, world):
    model.eval()
    total, count, parts = 0.0, 0, {}
    raw_model = model.module if isinstance(model, DDP) else model
    for x, v, _ in loader:
        x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
        if raw_model.route == "R4":
            noisy, t, cont, noise = diffusion_inputs(raw_model, v, scale, cfg)
            output = model(x, noisy, t, cont)
        else:
            noise, output = None, model(x)
        loss, items, _ = route_loss(raw_model, output, v, scale, cfg, noise)
        total += float(loss) * len(v); count += len(v)
        for k, val in items.items(): parts[k] = parts.get(k, 0.0) + float(val) * len(v)
    stats = torch.tensor([total, count] + [parts.get(k, 0.0) for k in ("cap", "e", "nll", "low", "diff")], device=device)
    if world > 1: dist.all_reduce(stats)
    den = max(float(stats[1]), 1.0)
    return {"total": float(stats[0]) / den, **{k: float(stats[i + 2]) / den for i, k in enumerate(("cap", "e", "nll", "low", "diff"))}}


def train(route):
    cfg = load_cfg(); rank, local, world, device = init_dist()
    contract_path = ROOT / "configs/data_contract.json"
    if rank == 0: contract = prepare_contract(cfg)
    if world > 1:
        sync = torch.ones(1, device=device)
        dist.all_reduce(sync)
        torch.cuda.synchronize(device)
    if rank != 0:
        with open(contract_path) as f: contract = json.load(f)
    seed = cfg["seed"] + rank
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds = FeatureDataset(cfg, "train"), FeatureDataset(cfg, "val")
    sampler = BalancedBatchSampler(train_ds.v, cfg["batch_size_per_rank"], cfg["steps_per_epoch"], cfg["seed"], rank, world)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg["num_workers"], pin_memory=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, world, rank, shuffle=False) if world > 1 else None
    val_loader = DataLoader(val_ds, batch_size=cfg["eval_batch_size_per_rank"], sampler=val_sampler, shuffle=False, num_workers=0, pin_memory=True)
    raw_model = RouteModel(cfg, route).to(device)
    head_names = ("ext", "cap", "dist", "low", "denoiser")
    head, backbone = [], []
    for name, p in raw_model.named_parameters():
        (head if name.split(".")[0] in head_names else backbone).append(p)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": cfg["lr_backbone"]}, {"params": head, "lr": cfg["lr_head"]}], weight_decay=cfg["weight_decay"])
    model = DDP(raw_model, device_ids=[local], find_unused_parameters=False) if world > 1 else raw_model
    scale = contract["extinction_scaler_train_only"]
    out = ROOT / "models" / route; out.mkdir(parents=True, exist_ok=True)
    history, initial_parts, best, bad, step = [], {}, float("inf"), 0, 0
    start = time.time()
    while step < cfg["max_steps"] and bad < cfg["patience"]:
        sampler.set_epoch(step // cfg["steps_per_epoch"])
        model.train()
        for x, v, _ in train_loader:
            x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            raw_for_loss = model.module if isinstance(model, DDP) else model
            if route == "R4":
                noisy, t, cont, noise = diffusion_inputs(raw_for_loss, v, scale, cfg)
                output = model(x, noisy, t, cont)
            else:
                noise, output = None, model(x)
            loss, parts, _ = route_loss(raw_for_loss, output, v, scale, cfg, noise)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); optimizer.step()
            step += 1
            if step <= 100:
                for k, val in parts.items(): initial_parts.setdefault(k, []).append(float(val.detach()))
            if step % cfg["val_interval"] == 0:
                val = validate(model, val_loader, scale, cfg, device, world)
                row = {"step": step, "epoch": step / cfg["steps_per_epoch"], "val": val, "minutes": (time.time() - start) / 60}
                history.append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    with open(out / "history.json", "w") as f: json.dump(history, f, indent=2)
                if val["total"] < best:
                    best, bad = val["total"], 0
                    if rank == 0:
                        torch.save({"model": raw_for_loss.state_dict(), "route": route, "best_step": step, "best_epoch": row["epoch"], "val": val, "extinction_scaler": scale, "config": cfg}, out / "best.pt")
                else: bad += 1
                model.train()
            if step >= cfg["max_steps"] or bad >= cfg["patience"]: break
    if rank == 0:
        summary = {"route": route, "best_val": best, "best_step": torch.load(out / "best.pt", map_location="cpu")["best_step"], "initial_100_step_loss_scales": {k: float(np.mean(v)) for k, v in initial_parts.items()}, "lambda_values": {k: cfg[k] for k in ("lambda_e", "lambda_dist", "lambda_lowvis")}}
        with open(out / "training_summary.json", "w") as f: json.dump(summary, f, indent=2)
    if world > 1: dist.destroy_process_group()


def event_metrics(y, probs, keep=None):
    if keep is None: keep = np.ones(len(y), dtype=bool)
    y, p = y[keep], probs[keep]
    pred = p.argmax(1)
    specs = {"lt500m": (y == 0, pred == 0, p[:, 0]), "500_1000m": (y == 1, pred == 1, p[:, 1]), "lt1000m": (y < 2, pred < 2, p[:, 0] + p[:, 1])}
    out = {}
    for name, (truth, guess, score) in specs.items():
        tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"Recall": recall, "Precision": precision, "CSI": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "AP": float(average_precision_score(truth, score)), "support": int(truth.sum())}
    return out


def run_r0():
    cfg = load_cfg(); contract = prepare_contract(cfg)
    probs = np.load(cfg["r0_probs"])
    frame = pd.read_csv(cfg["r0_per_sample"], usecols=["station_id", "time", "vis_raw_m"])
    meta = pd.read_csv(Path(cfg["data_dir"]) / "meta_test.csv", usecols=["station_id", "time"])
    if len(probs) != len(frame) or not np.array_equal(frame.station_id.astype(str).to_numpy(), meta.station_id.astype(str).to_numpy()) or not np.array_equal(frame.time.astype(str).to_numpy(), meta.time.astype(str).to_numpy()):
        raise AssertionError("R0 sample IDs do not exactly match p13 test split")
    v = frame.vis_raw_m.to_numpy(np.float32) / 1000.0
    y = np.zeros(len(v), np.int8); y[v >= 0.5] = 1; y[v >= 1.0] = 2
    result = {"route": "R0", "definition": "formal p13 3-seed probability mean, argmax; no threshold tuning", "sample_identity_sha256": "e3facb18c0a817b0d6d1b3d6c8adddceab510b69d940bca13a2c99ca828ee60b", "n": len(v), "metrics_including_v0": event_metrics(y, probs), "metrics_excluding_v0": event_metrics(y, probs, v > 0), "continuous": None}
    out = ROOT / "eval" / "R0"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def make_positive(draw_fn, attempts=8):
    values = draw_fn(); rejected = int((values <= 0).sum())
    for _ in range(attempts):
        mask = values <= 0
        if not mask.any(): break
        replacement = draw_fn()
        values = torch.where(mask, replacement, values)
        rejected += int((replacement[mask] <= 0).sum())
    fallback = int((values <= 0).sum())
    values = torch.where(values > 0, values, F.softplus(values) + 1e-5)
    return values, rejected, fallback


def class_probs(route, output, model, scale, cfg, draws):
    b = len(output["z"])
    if route in ("R1", "R2"):
        e, _ = positive_e(output["e_raw"], scale); samples = e[:, None]
    elif route == "R3":
        def draw(): return (output["mu"][:, None] + output["log_sigma"].exp()[:, None] * torch.randn(b, draws, device=output["mu"].device)) * scale["std"] + scale["mean"]
        samples, rejected, fallback = make_positive(draw)
    else:
        def draw(): return ddpm_samples(model, output["z"], draws) * scale["std"] + scale["mean"]
        samples, rejected, fallback = make_positive(draw, attempts=2)
    q0 = (samples > 7.824).float().mean(1); q1 = ((samples > 3.912) & (samples <= 7.824)).float().mean(1); q2 = (samples <= 3.912).float().mean(1)
    if route == "R1": pcap = torch.zeros_like(q0)
    else: pcap = output["cap"].sigmoid()
    probs = torch.stack([(1 - pcap) * q0, (1 - pcap) * q1, pcap + (1 - pcap) * q2], 1)
    if not torch.allclose(probs.sum(1), torch.ones_like(pcap), atol=1e-5): raise AssertionError("class probabilities do not sum to one")
    stats = {"p_cap": pcap, "e_mean": samples.mean(1), "e_median": samples.median(1).values, "e_q05": samples.quantile(0.05, 1), "e_q25": samples.quantile(0.25, 1), "e_q75": samples.quantile(0.75, 1), "e_q95": samples.quantile(0.95, 1)}
    v_samples = cfg["extinction_constant"] / samples
    conditional_v = v_samples.mean(1)
    stats["v_point"] = conditional_v if route == "R1" else pcap * cfg["cap_km"] + (1 - pcap) * conditional_v
    stats["samples_e"] = samples
    stats["rejected"] = 0 if route in ("R1", "R2") else rejected
    stats["fallback"] = 0 if route in ("R1", "R2") else fallback
    return probs, stats


def crps_samples(samples, truth):
    s = samples.sort(1).values; n = s.shape[1]
    first = (samples - truth[:, None]).abs().mean(1)
    weights = (2 * torch.arange(1, n + 1, device=s.device) - n - 1).float()
    second = (s * weights).sum(1) / (n * n)
    return first - second


@torch.no_grad()
def evaluate(route):
    cfg = load_cfg(); contract = prepare_contract(cfg); _, local, _, device = init_dist()
    ds = FeatureDataset(cfg, "test")
    loader = DataLoader(ds, batch_size=cfg["eval_batch_size_per_rank"], shuffle=False, num_workers=0, pin_memory=True)
    model = RouteModel(cfg, route).to(device)
    ckpt = torch.load(ROOT / "models" / route / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval(); scale = ckpt["extinction_scaler"]
    collected = {k: [] for k in ("index", "probs", "p_cap", "e_mean", "e_median", "e_q05", "e_q25", "e_q75", "e_q95", "v_point")}
    nll_sum = crps_sum = crps_n = rejected = fallback = 0
    start = time.time()
    for x, v, idx in loader:
        x, v = x.to(device), v.to(device); output = model(x)
        probs, stats = class_probs(route, output, model, scale, cfg, cfg["test_draws"])
        cont = (v > 0) & (v < cfg["cap_km"])
        if route == "R3":
            target = (cfg["extinction_constant"] / v[cont] - scale["mean"]) / scale["std"]
            sigma = output["log_sigma"][cont].exp()
            nll_sum += float((0.5 * ((target - output["mu"][cont]) / sigma) ** 2 + output["log_sigma"][cont] + 0.5 * math.log(2 * math.pi)).sum())
        if route in ("R3", "R4") and cont.any():
            truth_e = cfg["extinction_constant"] / v[cont]
            crps_sum += float(crps_samples(stats["samples_e"][cont], truth_e).sum()); crps_n += int(cont.sum())
        collected["index"].append(idx.numpy()); collected["probs"].append(probs.cpu().numpy())
        for k in collected:
            if k not in ("index", "probs"): collected[k].append(stats[k].cpu().numpy())
        rejected += stats["rejected"]; fallback += stats["fallback"]
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    v = ds.v; y = np.zeros(len(v), np.int8); y[v >= 0.5] = 1; y[v >= 1.0] = 2
    positive = v > 0
    errors = arrays["v_point"] - v
    continuous = {"MAE_km": float(np.abs(errors[positive]).mean()), "RMSE_km": float(np.sqrt(np.mean(errors[positive] ** 2))), "MAE_v_lt_1km": float(np.abs(errors[(v > 0) & (v < 1)]).mean()), "MAE_v_lt_0p5km": float(np.abs(errors[(v > 0) & (v < 0.5)]).mean())}
    result = {"route": route, "best_step": ckpt["best_step"], "best_epoch": ckpt["best_epoch"], "n": len(v), "draws": 1 if route in ("R1", "R2") else cfg["test_draws"], "wall_seconds": time.time() - start, "metrics_including_v0": event_metrics(y, arrays["probs"]), "metrics_excluding_v0": event_metrics(y, arrays["probs"], positive), "continuous": continuous, "NLL_scaled_E": nll_sum / max(int(((v > 0) & (v < 30)).sum()), 1) if route == "R3" else None, "CRPS_E": crps_sum / max(crps_n, 1) if route in ("R3", "R4") else None, "positive_resample_rejections": rejected, "positive_fallback_count": fallback, "probability_sum_max_abs_error": float(np.max(np.abs(arrays["probs"].sum(1) - 1)))}
    out = ROOT / "eval" / route; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_predictions.npz", **arrays)
    with open(out / "metrics.json", "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


@torch.no_grad()
def sanity():
    cfg = load_cfg(); contract = prepare_contract(cfg); _, _, _, device = init_dist()
    train_v = raw_visibility_km(Path(cfg["data_dir"]) / "y_train.npy")
    cap, cont, zero = masks(train_v)
    selected = np.concatenate([np.flatnonzero(cap)[:6], np.flatnonzero(cont)[:6], np.flatnonzero(zero)[:6]])
    ds = FeatureDataset(cfg, "train", indices=selected); x, v, idx = next(iter(DataLoader(ds, batch_size=len(selected))))
    x, v = x.to(device), v.to(device); scale = contract["extinction_scaler_train_only"]
    checks = {"V1_to_E": cfg["extinction_constant"] / 1.0, "V0p5_to_E": cfg["extinction_constant"] / 0.5, "V0p1_to_E": cfg["extinction_constant"] / 0.1}
    e = torch.tensor([3.912, 7.824, 39.12]); inv = (e - scale["mean"]) / scale["std"] * scale["std"] + scale["mean"]
    checks["scaling_max_abs_error"] = float((e - inv).abs().max())
    route_checks = {}
    for route in ("R1", "R2", "R3", "R4"):
        model = RouteModel(cfg, route).to(device).eval()
        if route == "R4":
            noisy, t, cont_mask, noise = diffusion_inputs(model, v, scale, cfg)
            output = model(x, noisy, t, cont_mask)
        else:
            noise, output = None, model(x)
        loss, parts, mm = route_loss(model, output, v, scale, cfg, noise)
        probs, stats = class_probs(route, output, model, scale, cfg, 32)
        route_checks[route] = {"loss": float(loss), "parts": {k: float(z) for k, z in parts.items()}, "mask_counts": [int(z.sum()) for z in mm], "prob_sum_error": float((probs.sum(1) - 1).abs().max()), "draw_shape": list(stats["samples_e"].shape)}
    checks["routes"] = route_checks
    checks["passed"] = checks["scaling_max_abs_error"] < 1e-6 and all(r["prob_sum_error"] < 1e-5 for r in route_checks.values()) and route_checks["R4"]["draw_shape"] == [len(selected), 32] and all(route_checks["R4"]["mask_counts"])
    out = ROOT / "eval"; out.mkdir(exist_ok=True)
    with open(out / "sanity.json", "w") as f: json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))
    if not checks["passed"]: raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["sanity", "r0", "train", "eval"]); ap.add_argument("--route", choices=["R1", "R2", "R3", "R4"])
    args = ap.parse_args()
    if args.command == "sanity": sanity()
    elif args.command == "r0": run_r0()
    elif args.command == "train": train(args.route)
    else: evaluate(args.route)


if __name__ == "__main__": main()
