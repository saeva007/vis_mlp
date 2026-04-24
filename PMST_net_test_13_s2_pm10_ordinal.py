import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import PMST_net_test_11_s2_pm10 as base


class OrdinalDualStreamPMSTNet(base.ImprovedDualStreamPMSTNet):
    """
    Keep the S2 backbone unchanged and replace the 3-class head with
    two cumulative logits:
      - fine_logits[:, 0]: P(vis < 500 m)
      - fine_logits[:, 1]: P(vis < 1000 m)
    """

    def __init__(
        self,
        dyn_vars_count=25,
        window_size=12,
        static_cont_dim=5,
        veg_num_classes=21,
        hidden_dim=512,
        num_classes=3,
        extra_feat_dim=0,
        dropout=0.3,
    ):
        super().__init__(
            dyn_vars_count=dyn_vars_count,
            window_size=window_size,
            static_cont_dim=static_cont_dim,
            veg_num_classes=veg_num_classes,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            extra_feat_dim=extra_feat_dim,
            dropout=dropout,
        )
        self.fine_classifier = nn.Linear(hidden_dim, 2)


class OrdinalCumulativeLoss(nn.Module):
    """
    Two cumulative BCE tasks aligned with visibility thresholds:
      y500  = 1(vis < 500m)
      y1000 = 1(vis < 1000m)

    The auxiliary low_vis head is retained as a second view of y1000.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.register_buffer(
            "pos_weight",
            torch.tensor([kwargs.get("binary_pos_weight", 1.0)], dtype=torch.float32),
        )
        self.register_buffer(
            "fine_class_weight",
            torch.tensor(kwargs.get("fine_class_weight", [1.0, 1.0, 1.0]), dtype=torch.float32),
        )

    def forward(self, fine_logits, low_vis_logit, targets, soft_targets=None):
        if fine_logits.size(1) != 2:
            raise ValueError(f"Ordinal fine head expects 2 logits, got {fine_logits.size(1)}")

        hard_500 = (targets == 0).float()
        hard_1000 = (targets <= 1).float()

        if soft_targets is not None:
            target_500 = soft_targets[:, 0].float()
            target_1000 = soft_targets[:, 1].float()
        else:
            target_500 = hard_500
            target_1000 = hard_1000

        per_sample_weight = self.fine_class_weight[targets]
        l_500 = F.binary_cross_entropy_with_logits(
            fine_logits[:, 0], target_500, reduction="none"
        )
        l_1000 = F.binary_cross_entropy_with_logits(
            fine_logits[:, 1], target_1000, reduction="none"
        )
        l_fine = ((l_500 + l_1000) * 0.5 * per_sample_weight).mean()

        low_vis_logit = torch.clamp(low_vis_logit, -20, 20)
        l_bin = F.binary_cross_entropy_with_logits(
            low_vis_logit,
            hard_1000.unsqueeze(1),
            pos_weight=self.pos_weight,
        )

        p_500 = torch.sigmoid(fine_logits[:, 0])
        p_1000 = torch.sigmoid(fine_logits[:, 1])
        p_low_aux = torch.sigmoid(low_vis_logit.view(-1))
        p_low = torch.maximum(p_1000, p_low_aux)

        is_fog = (targets == 0).float()
        is_mist = (targets == 1).float()
        is_clear = (targets == 2).float()

        l_fb = torch.mean((((1.0 - p_500) ** 2 + (1.0 - p_1000) ** 2) * 0.5) * is_fog)
        l_mb = torch.mean(((p_500 ** 2 + (1.0 - p_1000) ** 2) * 0.5) * is_mist)
        l_fp = torch.mean((p_low ** 2) * is_clear)

        clear_margin = self.cfg.get("clear_margin", 0.20)
        margin_excess = torch.relu(p_low - clear_margin)
        l_clear_margin = torch.mean((margin_excess ** 2) * is_clear)

        # Enforce the ordinal constraint P(vis<500) <= P(vis<1000).
        pair_margin = self.cfg.get("pair_margin", 0.05)
        l_pair = torch.mean(torch.relu(fine_logits[:, 0] - fine_logits[:, 1] + pair_margin))

        total = (
            self.cfg.get("alpha_binary", 1.0) * l_bin
            + self.cfg.get("alpha_fine", 1.0) * l_fine
            + self.cfg.get("alpha_fp", 0.5) * l_fp
            + self.cfg.get("alpha_fog_boost", 0.2) * l_fb
            + self.cfg.get("alpha_mist_boost", 0.0) * l_mb
            + self.cfg.get("alpha_clear_margin", 0.0) * l_clear_margin
            + self.cfg.get("alpha_pair_margin", 0.0) * l_pair
        )

        return total, {
            "bin": l_bin.item(),
            "fine": l_fine.item(),
            "cm": l_clear_margin.item(),
            "pair": l_pair.item(),
        }


class OrdinalMetrics(base.ComprehensiveMetrics):
    def __init__(self, config):
        self.cfg = config
        self.best_th = {"fog": 0.50, "low_vis": 0.50}
        self.min_prec_threshold = 0.10
        self.min_clear_recall = 0.90
        self.relaxed_prec_threshold = 0.05
        self.relaxed_clear_recall = 0.88

    def _build_full_metrics(self, probs, targets, fog_th, low_vis_th):
        p_500 = np.asarray(probs[:, 0], dtype=np.float64)
        p_1000 = np.asarray(probs[:, 1], dtype=np.float64)
        p_1000 = np.maximum(p_1000, p_500)

        preds = np.full(len(targets), 2, dtype=np.int64)
        low_vis_on = p_1000 >= low_vis_th
        fog_on = p_500 >= fog_th
        preds[low_vis_on] = 1
        preds[fog_on] = 0

        p0, r0 = self._calc_metrics_per_class(targets, preds, 0)
        p1, r1 = self._calc_metrics_per_class(targets, preds, 1)
        p2, r2 = self._calc_metrics_per_class(targets, preds, 2)

        accuracy = float((preds == targets).mean())
        pred_low = preds <= 1
        true_low = targets <= 1
        is_clear = targets == 2
        lv_tp = (pred_low & true_low).sum()
        lv_fp = (pred_low & ~true_low).sum()
        low_vis_precision = lv_tp / (lv_tp + lv_fp + 1e-6)
        fpr = (pred_low & is_clear).sum() / (is_clear.sum() + 1e-6)

        return {
            "Fog_R": r0,
            "Fog_P": p0,
            "Mist_R": r1,
            "Mist_P": p1,
            "Clear_R": r2,
            "Clear_P": p2,
            "recall_500": r0,
            "recall_1000": r1,
            "accuracy": accuracy,
            "low_vis_precision": float(low_vis_precision),
            "false_positive_rate": float(fpr),
            "preds": preds,
        }

    def evaluate(self, model, loader, device, rank=0, world_size=1, actual_val_size=None):
        model.eval()
        probs_l, targets_l = [], []

        if world_size > 1:
            torch.cuda.synchronize(device)
            n_batches = torch.tensor([len(loader)], dtype=torch.long, device=device)
            min_b = n_batches.clone()
            max_b = n_batches.clone()
            base.dist.all_reduce(min_b, op=base.dist.ReduceOp.MIN)
            base.dist.all_reduce(max_b, op=base.dist.ReduceOp.MAX)
            if min_b.item() != max_b.item():
                raise RuntimeError(
                    f"[Eval] Per-rank val DataLoader length mismatch: min={min_b.item()} max={max_b.item()}."
                )

        with torch.no_grad():
            for bx, by, _, _ in loader:
                bx = bx.to(device, non_blocking=True)
                fine, _, _ = model(bx)
                probs_l.append(torch.sigmoid(fine))
                targets_l.append(by.to(device))

        if not probs_l:
            raise RuntimeError("[Eval] Empty validation loader on at least one rank.")

        local_probs = torch.cat(probs_l, dim=0)
        local_targets = torch.cat(targets_l, dim=0)

        if world_size > 1:
            torch.cuda.synchronize(device)
            local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=device)
            max_size = local_size.clone()
            base.dist.all_reduce(max_size, op=base.dist.ReduceOp.MAX)

            if local_size < max_size:
                pad_size = max_size.item() - local_size.item()
                pad_probs = torch.zeros((pad_size, local_probs.size(1)), dtype=local_probs.dtype, device=device)
                pad_targets = torch.full((pad_size,), -1, dtype=local_targets.dtype, device=device)
                local_probs = torch.cat([local_probs, pad_probs], dim=0)
                local_targets = torch.cat([local_targets, pad_targets], dim=0)

            gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
            base.dist.all_gather(gathered_probs, local_probs)
            base.dist.all_gather(gathered_targets, local_targets)

            all_probs = torch.cat(gathered_probs, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
        else:
            all_probs = local_probs.cpu().numpy()
            all_targets = local_targets.cpu().numpy()

        best_ta = -1.0
        best_stats = None
        tier_used = 0

        if rank == 0:
            n = actual_val_size if actual_val_size is not None else len(loader.dataset)
            probs = all_probs[:n]
            targets = all_targets[:n]

            valid_mask = targets >= 0
            probs = probs[valid_mask]
            targets = targets[valid_mask]

            search_space = self._build_search_grid()
            n_combos = len(search_space) ** 2
            print(
                f"  [Eval] Searching {n_combos} ordinal threshold combinations "
                f"(grid: {search_space[0]:.2f}-{search_space[-1]:.2f})...",
                flush=True,
            )

            for fog_th in search_space:
                for low_vis_th in search_space:
                    stats = self._build_full_metrics(probs, targets, fog_th, low_vis_th)
                    if (
                        stats["Fog_P"] >= self.min_prec_threshold
                        and stats["Mist_P"] >= self.min_prec_threshold
                        and stats["Clear_R"] >= self.min_clear_recall
                    ):
                        ta = base.compute_target_achievement(stats, self.cfg)
                        if ta > best_ta:
                            best_ta = ta
                            best_stats = stats
                            self.best_th = {"fog": float(fog_th), "low_vis": float(low_vis_th)}
                            tier_used = 1

            if best_stats is None:
                for fog_th in search_space:
                    for low_vis_th in search_space:
                        stats = self._build_full_metrics(probs, targets, fog_th, low_vis_th)
                        if (
                            stats["Fog_P"] >= self.relaxed_prec_threshold
                            and stats["Mist_P"] >= self.relaxed_prec_threshold
                            and stats["Clear_R"] >= self.relaxed_clear_recall
                        ):
                            fog_shortfall = max(0.0, self.min_prec_threshold - stats["Fog_P"])
                            mist_shortfall = max(0.0, self.min_prec_threshold - stats["Mist_P"])
                            ta = base.compute_target_achievement(stats, self.cfg) - (fog_shortfall + mist_shortfall)
                            if ta > best_ta:
                                best_ta = ta
                                best_stats = stats
                                self.best_th = {"fog": float(fog_th), "low_vis": float(low_vis_th)}
                                tier_used = 2

            if best_stats is None:
                tier_used = 3
                best_stats = self._build_full_metrics(probs, targets, 0.50, 0.50)
                best_ta = base.compute_target_achievement(best_stats, self.cfg)
                self.best_th = {"fog": 0.50, "low_vis": 0.50}
                print(
                    "  [Eval] WARN: Tier 1+2 all failed - using default ordinal thresholds 0.50/0.50.",
                    flush=True,
                )
            else:
                tier_label = "Strict" if tier_used == 1 else "Relaxed"
                print(
                    f"  [Eval] Tier {tier_used} ({tier_label}): "
                    f"Best Th -> Fog<500:{self.best_th['fog']:.2f}, LowVis<1000:{self.best_th['low_vis']:.2f}",
                    flush=True,
                )

            print(
                f"  [Eval] Fog  R={best_stats['Fog_R']:.3f} P={best_stats['Fog_P']:.3f} | "
                f"Mist R={best_stats['Mist_R']:.3f} P={best_stats['Mist_P']:.3f} | "
                f"Clear R={best_stats['Clear_R']:.3f} | "
                f"Acc={best_stats['accuracy']:.3f} | "
                f"LVPrec={best_stats['low_vis_precision']:.3f} | "
                f"FPR={best_stats['false_positive_rate']:.3f}",
                flush=True,
            )
            print(f"  [Eval] target_achievement = {best_ta:.4f}", flush=True)

        if world_size > 1:
            ta_tensor = torch.tensor([best_ta], dtype=torch.float32, device=device)
            base.dist.broadcast(ta_tensor, src=0)
            best_ta = ta_tensor.item()
            base.safe_barrier(world_size, device)

        return {"score": best_ta, "stats": best_stats, "thresholds": self.best_th}


def compute_soft_targets(vis_raw, hard_labels, num_classes=2):
    del hard_labels, num_classes
    vis = vis_raw.float()
    soft = torch.zeros((vis.shape[0], 2), dtype=torch.float32, device=vis.device)

    soft[:, 0] = (vis < 500).float()
    soft[:, 1] = (vis < 1000).float()

    fog_band = (vis >= 400) & (vis < 600)
    if fog_band.any():
        alpha = (vis[fog_band] - 400.0) / 200.0
        soft[fog_band, 0] = 1.0 - alpha

    low_vis_band = (vis >= 800) & (vis < 1200)
    if low_vis_band.any():
        alpha = (vis[low_vis_band] - 800.0) / 400.0
        soft[low_vis_band, 1] = 1.0 - alpha

    soft[:, 1] = torch.maximum(soft[:, 1], soft[:, 0])
    return soft


def calibrate_temperature(model, loader, device, config, rank=0):
    if rank != 0:
        return 1.0

    print(
        "  [Posthoc] Ordinal experiment: skip temperature scaling, search global thresholds directly.",
        flush=True,
    )
    evaluator = OrdinalMetrics(config)
    evaluator.evaluate(model, loader, device, rank=0, world_size=1, actual_val_size=len(loader.dataset))

    run_exp_id = base.build_s2_run_exp_id(config["EXPERIMENT_ID"], config.get("S2_RUN_SUFFIX", ""))
    out_path = os.path.join(config["SAVE_CKPT_DIR"], f"{run_exp_id}_ordinal_thresholds.pt")
    torch.save(
        {
            "scheme": "ordinal_cumulative",
            "fog_th": evaluator.best_th["fog"],
            "low_vis_th": evaluator.best_th["low_vis"],
        },
        out_path,
    )
    print(f"  [Save] Ordinal thresholds -> {out_path}", flush=True)
    return 1.0


def evaluate_per_season(model, loader, device, config, rank=0, temperature=1.0):
    del model, loader, device, config, temperature
    if rank == 0:
        print("  [Posthoc] Per-season thresholds disabled for ordinal cumulative experiment.", flush=True)
    return None


def main():
    base.ImprovedDualStreamPMSTNet = OrdinalDualStreamPMSTNet
    base.DualBranchLoss = OrdinalCumulativeLoss
    base.ComprehensiveMetrics = OrdinalMetrics
    base.compute_soft_targets = compute_soft_targets
    base.calibrate_temperature = calibrate_temperature
    base.evaluate_per_season = evaluate_per_season

    base.CONFIG["S2_RUN_SUFFIX"] = "pm10_ordinal_cumulative"
    base.CONFIG["MODEL_NUM_CLASSES"] = 2
    base.CONFIG["S2_LOSS_ALPHA_BINARY"] = 0.35
    base.CONFIG["S2_LOSS_ALPHA_MIST_BOOST"] = 0.8
    base.CONFIG["S2_PAIR_MARGIN"] = 0.10
    base.CONFIG["S2_LOSS_ALPHA_PAIR_MARGIN"] = 0.20

    base.main()


if __name__ == "__main__":
    main()
