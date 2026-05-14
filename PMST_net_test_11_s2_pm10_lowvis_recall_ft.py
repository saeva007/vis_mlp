#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recall-focused low-visibility fine-tuning for the PM10+PM2.5 S2 model.

This script intentionally leaves PMST_net_test_11_s2_pm10.py unchanged for
archive/reproducibility. It imports the archived model/data/DDP utilities,
loads the existing S2 Phase-B checkpoint, and runs a short Phase-C fine-tune
that prioritizes detecting low-visibility events.

Default initialization checkpoint:
  checkpoints/exp_1778563813_pm10_more_temp_search_utc_S2_PhaseB_best_score.pt

Default output run id:
  exp_1778563813_pm10_more_temp_search_utc_lowvis_recall_ft
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import PMST_net_test_11_s2_pm10 as base


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def update_recall_config() -> Tuple[str, str, str]:
    init_run_id = env_str("LOWVIS_FT_INIT_RUN_ID", "exp_1778563813_pm10_more_temp_search_utc")
    run_suffix = env_str("LOWVIS_FT_RUN_SUFFIX", "lowvis_recall_ft")
    run_exp_id = f"{init_run_id}_{run_suffix}"

    cfg = base.CONFIG
    cfg["EXPERIMENT_ID"] = init_run_id
    cfg["S2_RUN_SUFFIX"] = run_suffix

    cfg["NUM_WORKERS"] = env_int("LOWVIS_FT_NUM_WORKERS", 0)
    cfg["S2_BATCH_SIZE"] = env_int("LOWVIS_FT_BATCH_SIZE", 512)
    cfg["S2_GRAD_ACCUM"] = env_int("LOWVIS_FT_GRAD_ACCUM", 2)
    cfg["S2_VAL_INTERVAL"] = env_int("LOWVIS_FT_VAL_INTERVAL", 500)
    cfg["S2_WARMUP_STEPS"] = env_int("LOWVIS_FT_WARMUP_STEPS", 300)
    cfg["S2_ES_PATIENCE"] = env_int("LOWVIS_FT_PATIENCE", 8)

    cfg["LOWVIS_FT_STEPS"] = env_int("LOWVIS_FT_STEPS", 12000)
    cfg["LOWVIS_FT_LR_BACKBONE"] = env_float("LOWVIS_FT_LR_BACKBONE", 1.0e-6)
    cfg["LOWVIS_FT_LR_FUSION"] = env_float("LOWVIS_FT_LR_FUSION", 8.0e-6)
    cfg["LOWVIS_FT_LR_HEAD"] = env_float("LOWVIS_FT_LR_HEAD", 3.0e-5)
    cfg["LOWVIS_FT_L2SP"] = env_float("LOWVIS_FT_L2SP", 2.0e-5)

    # Recall-focused sampling: keep enough clear samples for FPR control, but
    # make low-vis events dominate the optimization signal.
    cfg["S2_FOG_RATIO"] = env_float("LOWVIS_FT_FOG_RATIO", 0.24)
    cfg["S2_MIST_RATIO"] = env_float("LOWVIS_FT_MIST_RATIO", 0.34)

    # Recall-focused loss. False-positive pressure is retained, but weaker than
    # in the archive run so the model can recover missed Fog/Mist events.
    cfg["S2_BINARY_POS_WEIGHT"] = env_float("LOWVIS_FT_BINARY_POS_WEIGHT", 2.6)
    cfg["S2_FINE_CLASS_WEIGHT_FOG"] = env_float("LOWVIS_FT_WEIGHT_FOG", 2.2)
    cfg["S2_FINE_CLASS_WEIGHT_MIST"] = env_float("LOWVIS_FT_WEIGHT_MIST", 3.2)
    cfg["S2_FINE_CLASS_WEIGHT_CLEAR"] = env_float("LOWVIS_FT_WEIGHT_CLEAR", 0.7)
    cfg["S2_LOSS_ALPHA_BINARY"] = env_float("LOWVIS_FT_ALPHA_BINARY", 1.25)
    cfg["S2_LOSS_ALPHA_FINE"] = env_float("LOWVIS_FT_ALPHA_FINE", 1.0)
    cfg["S2_LOSS_ALPHA_FP"] = env_float("LOWVIS_FT_ALPHA_FP", 2.5)
    cfg["S2_LOSS_ALPHA_FOG_BOOST"] = env_float("LOWVIS_FT_ALPHA_FOG_BOOST", 0.85)
    cfg["S2_LOSS_ALPHA_MIST_BOOST"] = env_float("LOWVIS_FT_ALPHA_MIST_BOOST", 1.35)
    cfg["S2_CLEAR_MARGIN"] = env_float("LOWVIS_FT_CLEAR_MARGIN", 0.30)
    cfg["S2_LOSS_ALPHA_CLEAR_MARGIN"] = env_float("LOWVIS_FT_ALPHA_CLEAR_MARGIN", 1.5)
    cfg["S2_PAIR_MARGIN"] = env_float("LOWVIS_FT_PAIR_MARGIN", 0.65)
    cfg["S2_LOSS_ALPHA_PAIR_MARGIN"] = env_float("LOWVIS_FT_ALPHA_PAIR_MARGIN", 0.45)

    # Extra recall terms in LowVisRecallFineTuneLoss.
    cfg["LOWVIS_FT_ALPHA_LOW_MISS"] = env_float("LOWVIS_FT_ALPHA_LOW_MISS", 0.85)
    cfg["LOWVIS_FT_ALPHA_FINE_LOW_MASS"] = env_float("LOWVIS_FT_ALPHA_FINE_LOW_MASS", 0.55)
    cfg["LOWVIS_FT_ALPHA_MIST_CLEAR_MARGIN"] = env_float("LOWVIS_FT_ALPHA_MIST_CLEAR_MARGIN", 0.60)
    cfg["LOWVIS_FT_MIST_CLEAR_MARGIN"] = env_float("LOWVIS_FT_MIST_CLEAR_MARGIN", 0.18)

    # Validation checkpoint scoring: recall dominates, FPR/precision are guardrails.
    cfg["TARGET_RECALL_500_GOAL"] = env_float("LOWVIS_FT_GOAL_FOG_R", 0.55)
    cfg["TARGET_RECALL_1000_GOAL"] = env_float("LOWVIS_FT_GOAL_MIST_R", 0.45)
    cfg["TARGET_ACCURACY_GOAL"] = env_float("LOWVIS_FT_GOAL_ACC", 0.93)
    cfg["TARGET_LOW_VIS_PREC_GOAL"] = env_float("LOWVIS_FT_GOAL_LV_PREC", 0.16)
    cfg["TARGET_FPR_GOAL"] = env_float("LOWVIS_FT_GOAL_FPR", 0.08)
    cfg["TARGET_W_RECALL_500"] = env_float("LOWVIS_FT_W_FOG_R", 0.28)
    cfg["TARGET_W_RECALL_1000"] = env_float("LOWVIS_FT_W_MIST_R", 0.34)
    cfg["TARGET_W_ACCURACY"] = env_float("LOWVIS_FT_W_ACC", 0.05)
    cfg["TARGET_W_LOW_VIS_PREC"] = env_float("LOWVIS_FT_W_LV_PREC", 0.08)
    cfg["TARGET_W_FPR"] = env_float("LOWVIS_FT_W_FPR", 0.25)

    init_ckpt = env_str(
        "LOWVIS_FT_INIT_CKPT_PATH",
        os.path.join(cfg["SAVE_CKPT_DIR"], f"{init_run_id}_S2_PhaseB_best_score.pt"),
    )
    scaler_path = env_str(
        "LOWVIS_FT_SCALER_PATH",
        os.path.join(
            cfg["SAVE_CKPT_DIR"],
            f"robust_scaler_{init_run_id}_w{cfg['WINDOW_SIZE']}_dyn27_s2_48h_pm10.pkl",
        ),
    )
    return run_exp_id, init_ckpt, scaler_path


def compute_soft_targets_recall(vis_raw, hard_labels, num_classes: int = 3):
    """Use a narrower Mist/Clear transition so true Mist is less diluted."""
    vis = vis_raw.clone()
    if vis.max() < 100:
        vis = vis * 1000.0

    soft = F.one_hot(hard_labels, num_classes).float()

    fm_mask = (vis >= 400) & (vis < 600)
    if fm_mask.any():
        alpha = (vis[fm_mask] - 400) / 200.0
        soft[fm_mask, 0] = 1 - alpha
        soft[fm_mask, 1] = alpha
        soft[fm_mask, 2] = 0

    mc_mask = (vis >= 900) & (vis < 1050)
    if mc_mask.any():
        alpha = (vis[mc_mask] - 900) / 150.0
        soft[mc_mask, 0] = 0
        soft[mc_mask, 1] = 1 - alpha
        soft[mc_mask, 2] = alpha

    return soft


class LowVisRecallFineTuneLoss(nn.Module):
    def __init__(self, cfg: Dict[str, float]):
        super().__init__()
        self.cfg = cfg
        self.base_loss = base.DualBranchLoss(
            binary_pos_weight=cfg["S2_BINARY_POS_WEIGHT"],
            fine_class_weight=[
                cfg["S2_FINE_CLASS_WEIGHT_FOG"],
                cfg["S2_FINE_CLASS_WEIGHT_MIST"],
                cfg["S2_FINE_CLASS_WEIGHT_CLEAR"],
            ],
            loss_type="ordinal_focal",
            gamma_per_class=[2.5, 3.0, 0.5],
            ordinal_cost=[[0, 1, 3], [1, 0, 2], [3, 2, 0]],
            alpha_binary=cfg["S2_LOSS_ALPHA_BINARY"],
            alpha_fine=cfg["S2_LOSS_ALPHA_FINE"],
            alpha_fp=cfg["S2_LOSS_ALPHA_FP"],
            alpha_fog_boost=cfg["S2_LOSS_ALPHA_FOG_BOOST"],
            alpha_mist_boost=cfg["S2_LOSS_ALPHA_MIST_BOOST"],
            alpha_clear_margin=cfg["S2_LOSS_ALPHA_CLEAR_MARGIN"],
            clear_margin=cfg["S2_CLEAR_MARGIN"],
            alpha_pair_margin=cfg["S2_LOSS_ALPHA_PAIR_MARGIN"],
            pair_margin=cfg["S2_PAIR_MARGIN"],
        )

    @staticmethod
    def class_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(mask.sum(), min=1.0)
        return (values * mask).sum() / denom

    def forward(self, fine_logits, low_vis_logit, targets, soft_targets=None):
        total, logs = self.base_loss(fine_logits, low_vis_logit, targets, soft_targets=soft_targets)
        probs = F.softmax(fine_logits, dim=1)
        low_prob = torch.sigmoid(torch.clamp(low_vis_logit.reshape(-1), -20, 20))

        is_low = (targets <= 1).float()
        is_mist = (targets == 1).float()
        fine_low_mass = torch.clamp(probs[:, 0] + probs[:, 1], 0.0, 1.0)

        low_miss = self.class_mean((1.0 - low_prob) ** 2, is_low)
        fine_low_miss = self.class_mean((1.0 - fine_low_mass) ** 2, is_low)
        mist_clear_margin = torch.relu(
            self.cfg["LOWVIS_FT_MIST_CLEAR_MARGIN"] - (probs[:, 1] - probs[:, 2])
        )
        mist_clear_loss = self.class_mean(mist_clear_margin, is_mist)

        total = (
            total
            + self.cfg["LOWVIS_FT_ALPHA_LOW_MISS"] * low_miss
            + self.cfg["LOWVIS_FT_ALPHA_FINE_LOW_MASS"] * fine_low_miss
            + self.cfg["LOWVIS_FT_ALPHA_MIST_CLEAR_MARGIN"] * mist_clear_loss
        )
        logs["lowmiss"] = float(low_miss.detach().cpu())
        logs["finelow"] = float(fine_low_miss.detach().cpu())
        logs["mistclr"] = float(mist_clear_loss.detach().cpu())
        return total, logs


class LowVisRecallMetrics(base.ComprehensiveMetrics):
    def __init__(self, config):
        super().__init__(config)
        self.min_prec_threshold = float(config.get("LOWVIS_FT_MIN_PREC", 0.06))
        self.min_clear_recall = float(config.get("LOWVIS_FT_MIN_CLEAR_RECALL", 0.82))
        self.relaxed_prec_threshold = float(config.get("LOWVIS_FT_RELAXED_PREC", 0.035))
        self.relaxed_clear_recall = float(config.get("LOWVIS_FT_RELAXED_CLEAR_RECALL", 0.76))

    def _build_full_metrics(self, probs, targets, f_th, m_th):
        stats = super()._build_full_metrics(probs, targets, f_th, m_th)
        preds = stats["preds"]
        pred_low = preds <= 1
        true_low = targets <= 1
        low_tp = np_count(pred_low & true_low)
        low_fp = np_count(pred_low & ~true_low)
        low_fn = np_count(~pred_low & true_low)
        stats["low_vis_recall"] = low_tp / max(low_tp + low_fn, 1.0)
        stats["low_vis_csi"] = low_tp / max(low_tp + low_fp + low_fn, 1.0)
        return stats


def np_count(mask) -> float:
    return float(mask.sum())


def recall_weighted_score(metrics: dict, cfg: dict) -> float:
    def ratio(key: str, goal_key: str) -> float:
        goal = float(cfg.get(goal_key, 1.0))
        return min(float(metrics.get(key, 0.0)) / max(goal, 1e-6), 1.0)

    score = (
        ratio("recall_500", "TARGET_RECALL_500_GOAL") * cfg["TARGET_W_RECALL_500"]
        + ratio("recall_1000", "TARGET_RECALL_1000_GOAL") * cfg["TARGET_W_RECALL_1000"]
        + ratio("accuracy", "TARGET_ACCURACY_GOAL") * cfg["TARGET_W_ACCURACY"]
        + ratio("low_vis_precision", "TARGET_LOW_VIS_PREC_GOAL") * cfg["TARGET_W_LOW_VIS_PREC"]
        + min((1.0 - metrics["false_positive_rate"]) / max(1.0 - cfg["TARGET_FPR_GOAL"], 1e-6), 1.0)
        * cfg["TARGET_W_FPR"]
    )
    low_recall_bonus = 0.20 * min(float(metrics.get("low_vis_recall", 0.0)) / 0.75, 1.0)
    low_csi_bonus = 0.08 * min(float(metrics.get("low_vis_csi", 0.0)) / 0.22, 1.0)
    fpr_goal = float(cfg.get("TARGET_FPR_GOAL", 0.08))
    fpr_penalty = 0.35 * max(0.0, float(metrics["false_positive_rate"]) - fpr_goal) / max(fpr_goal, 1e-6)
    return float(score + low_recall_bonus + low_csi_bonus - fpr_penalty)


def build_model_raw(device: torch.device):
    return base.ImprovedDualStreamPMSTNet(
        window_size=base.CONFIG["WINDOW_SIZE"],
        hidden_dim=base.CONFIG["MODEL_HIDDEN_DIM"],
        num_classes=3,
        extra_feat_dim=base.CONFIG["FE_EXTRA_DIMS"],
        dyn_vars_count=base.CONFIG["DYN_VARS_COUNT"],
        dropout=base.CONFIG["MODEL_DROPOUT"],
    ).to(device)


def build_recall_optimizer(raw_model, local_rank: int, rank: int, world_size: int):
    head_names = {"fine_classifier", "low_vis_detector", "reg_head"}
    fusion_names = {"fusion_kan", "temporal_norm", "extra_encoder"}

    head_params = []
    fusion_params = []
    backbone_params = []
    for name, param in raw_model.named_parameters():
        param.requires_grad = True
        top = name.split(".")[0]
        if top in head_names:
            head_params.append(param)
        elif top in fusion_names:
            fusion_params.append(param)
        else:
            backbone_params.append(param)

    if rank == 0:
        print(
            "[LowVisFT-Optim] "
            f"backbone={sum(p.numel() for p in backbone_params)/1e6:.3f}M "
            f"fusion={sum(p.numel() for p in fusion_params)/1e6:.3f}M "
            f"head={sum(p.numel() for p in head_params)/1e6:.3f}M",
            flush=True,
        )
        print(
            "[LowVisFT-Optim] LR "
            f"backbone={base.CONFIG['LOWVIS_FT_LR_BACKBONE']:.2e}, "
            f"fusion={base.CONFIG['LOWVIS_FT_LR_FUSION']:.2e}, "
            f"head={base.CONFIG['LOWVIS_FT_LR_HEAD']:.2e}",
            flush=True,
        )

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": base.CONFIG["LOWVIS_FT_LR_BACKBONE"]},
            {"params": fusion_params, "lr": base.CONFIG["LOWVIS_FT_LR_FUSION"]},
            {"params": head_params, "lr": base.CONFIG["LOWVIS_FT_LR_HEAD"]},
        ],
        weight_decay=base.CONFIG["S2_WEIGHT_DECAY"],
    )
    return base.wrap_ddp(raw_model, local_rank, world_size), optimizer


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.X = None


def main():
    run_exp_id, init_ckpt, scaler_path = update_recall_config()

    base.compute_soft_targets = compute_soft_targets_recall
    base.compute_target_achievement = recall_weighted_score
    base.ComprehensiveMetrics = LowVisRecallMetrics

    local_rank, global_rank, world_size = base.init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        os.makedirs(base.CONFIG["SAVE_CKPT_DIR"], exist_ok=True)
        print("[LowVisFT] Recall-focused fine-tuning", flush=True)
        print(f"[LowVisFT] Run ID: {run_exp_id}", flush=True)
        print(f"[LowVisFT] Init checkpoint: {init_ckpt}", flush=True)
        print(f"[LowVisFT] Scaler: {scaler_path}", flush=True)
        print(f"[LowVisFT] World size: {world_size}", flush=True)

    base.safe_barrier(world_size, device)

    dyn_res, fe_res = base.resolve_feature_layout_from_x_train(
        base.CONFIG["S2_DATA_DIR"], base.CONFIG["WINDOW_SIZE"]
    )
    base.CONFIG["DYN_VARS_COUNT"] = int(dyn_res)
    base.CONFIG["FE_EXTRA_DIMS"] = int(fe_res)

    raw_model = build_model_raw(device)
    base.load_checkpoint(raw_model, init_ckpt, global_rank, world_size, device)

    pretrained_state = {
        k: v.clone().detach().to(device)
        for k, v in raw_model.state_dict().items()
    }

    scaler = base.joblib.load(scaler_path)
    tr_ds, val_ds, _ = base.load_data(
        base.CONFIG["S2_DATA_DIR"],
        scaler,
        global_rank,
        local_rank,
        device,
        True,
        base.CONFIG["WINDOW_SIZE"],
        world_size,
        run_exp_id,
    )

    if global_rank == 0:
        print(f"[LowVisFT] Train={len(tr_ds)} Val={len(val_ds)}", flush=True)

    model, optimizer = build_recall_optimizer(raw_model, local_rank, global_rank, world_size)
    loss_fn = LowVisRecallFineTuneLoss(base.CONFIG).to(device)

    base.train_stage(
        tag="S2_LowVisRecallFT",
        model=model,
        tr_ds=tr_ds,
        val_ds=val_ds,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        rank=global_rank,
        world_size=world_size,
        total_steps=base.CONFIG["LOWVIS_FT_STEPS"],
        val_int=base.CONFIG["S2_VAL_INTERVAL"],
        batch_size=base.CONFIG["S2_BATCH_SIZE"],
        grad_accum=base.CONFIG["S2_GRAD_ACCUM"],
        fog_ratio=base.CONFIG["S2_FOG_RATIO"],
        mist_ratio=base.CONFIG["S2_MIST_RATIO"],
        exp_id=run_exp_id,
        patience=base.CONFIG["S2_ES_PATIENCE"],
        pretrained_state=pretrained_state,
        l2sp_alpha=base.CONFIG["LOWVIS_FT_L2SP"],
    )

    raw_final = base.rewrap_ddp(model, world_size)
    best_path = os.path.join(
        base.CONFIG["SAVE_CKPT_DIR"],
        f"{run_exp_id}_S2_LowVisRecallFT_best_score.pt",
    )
    if os.path.exists(best_path):
        base.load_checkpoint(raw_final, best_path, global_rank, world_size, device)

    if global_rank == 0:
        print("[LowVisFT] Calibrating temperature on validation set.", flush=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=base.CONFIG["S2_BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    temp = base.calibrate_temperature(raw_final, val_loader, device, base.CONFIG, rank=global_rank)

    if global_rank == 0:
        meta_path = os.path.join(base.CONFIG["SAVE_CKPT_DIR"], f"{run_exp_id}_lowvis_recall_ft_meta.json")
        meta = {
            "run_exp_id": run_exp_id,
            "init_checkpoint": init_ckpt,
            "scaler_path": scaler_path,
            "temperature": temp,
            "config_subset": {
                k: base.CONFIG[k]
                for k in sorted(base.CONFIG)
                if k.startswith("LOWVIS_FT")
                or k.startswith("TARGET_")
                or k in ("S2_FOG_RATIO", "S2_MIST_RATIO", "S2_BINARY_POS_WEIGHT")
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[LowVisFT] Wrote metadata: {meta_path}", flush=True)

    del pretrained_state
    base.cleanup_temp_files(run_exp_id)
    if world_size > 1 and base.dist.is_initialized():
        base.dist.destroy_process_group()

    if global_rank == 0:
        print("[LowVisFT] Job finished.", flush=True)


if __name__ == "__main__":
    main()
