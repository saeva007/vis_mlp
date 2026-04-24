import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import PMST_net_test_10_s1_pm10 as base


class OrdinalDualStreamPMSTNet(base.ImprovedDualStreamPMSTNet):
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

    def forward(self, fine_logits, low_vis_logit, targets):
        if fine_logits.size(1) != 2:
            raise ValueError("Ordinal fine head expects 2 logits, got {}".format(fine_logits.size(1)))

        hard_500 = (targets == 0).float()
        hard_1000 = (targets <= 1).float()
        per_sample_weight = self.fine_class_weight[targets]

        l_500 = F.binary_cross_entropy_with_logits(fine_logits[:, 0], hard_500, reduction="none")
        l_1000 = F.binary_cross_entropy_with_logits(fine_logits[:, 1], hard_1000, reduction="none")
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
        p_low = torch.maximum(torch.maximum(p_1000, p_500), p_low_aux)

        is_fog = (targets == 0).float()
        is_mist = (targets == 1).float()
        is_clear = (targets == 2).float()

        l_fb = torch.mean((((1.0 - p_500) ** 2 + (1.0 - p_1000) ** 2) * 0.5) * is_fog)
        l_mb = torch.mean(((p_500 ** 2 + (1.0 - p_1000) ** 2) * 0.5) * is_mist)
        l_fp = torch.mean((p_low ** 2) * is_clear)

        clear_margin = self.cfg.get("clear_margin", 0.30)
        margin_excess = torch.relu(p_low - clear_margin)
        l_clear_margin = torch.mean((margin_excess ** 2) * is_clear)

        total = (
            self.cfg.get("alpha_binary", 1.0) * l_bin
            + self.cfg.get("alpha_fine", 1.0) * l_fine
            + self.cfg.get("alpha_fp", 0.5) * l_fp
            + self.cfg.get("alpha_fog_boost", 0.2) * l_fb
            + self.cfg.get("alpha_mist_boost", 0.0) * l_mb
            + self.cfg.get("alpha_clear_margin", 0.0) * l_clear_margin
        )

        return total, {
            "bin": l_bin.item(),
            "fine": l_fine.item(),
            "cm": l_clear_margin.item(),
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
                    "[Eval] Per-rank val DataLoader length mismatch: min={} max={}.".format(
                        min_b.item(), max_b.item()
                    )
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
                "  [Eval] Searching {} ordinal threshold combinations (grid: {:.2f}-{:.2f})...".format(
                    n_combos, search_space[0], search_space[-1]
                ),
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
                print("  [Eval] WARN: Tier 1+2 all failed - using default ordinal thresholds 0.50/0.50.", flush=True)
            else:
                tier_label = "Strict" if tier_used == 1 else "Relaxed"
                print(
                    "  [Eval] Tier {} ({}): Best Th -> Fog<500:{:.2f}, LowVis<1000:{:.2f}".format(
                        tier_used, tier_label, self.best_th["fog"], self.best_th["low_vis"]
                    ),
                    flush=True,
                )

            print(
                "  [Eval] Fog  R={:.3f} P={:.3f} | Mist R={:.3f} P={:.3f} | Clear R={:.3f} | Acc={:.3f} | LVPrec={:.3f} | FPR={:.3f}".format(
                    best_stats["Fog_R"],
                    best_stats["Fog_P"],
                    best_stats["Mist_R"],
                    best_stats["Mist_P"],
                    best_stats["Clear_R"],
                    best_stats["accuracy"],
                    best_stats["low_vis_precision"],
                    best_stats["false_positive_rate"],
                ),
                flush=True,
            )
            print("  [Eval] target_achievement = {:.4f}".format(best_ta), flush=True)

        if world_size > 1:
            ta_tensor = torch.tensor([best_ta], dtype=torch.float32, device=device)
            base.dist.broadcast(ta_tensor, src=0)
            best_ta = ta_tensor.item()
            base.safe_barrier(world_size, device)

        return {"score": best_ta, "stats": best_stats, "thresholds": self.best_th}


def main():
    base.ImprovedDualStreamPMSTNet = OrdinalDualStreamPMSTNet
    base.DualBranchLoss = OrdinalCumulativeLoss
    base.ComprehensiveMetrics = OrdinalMetrics

    if os.environ.get("EXPERIMENT_JOB_ID") is None:
        base.CONFIG["EXPERIMENT_ID"] = "exp_{}_s1_ordinal".format(int(time.time()))
    base.CONFIG["MODEL_NUM_CLASSES"] = 2
    base.CONFIG["LOSS_ALPHA_BINARY"] = 0.35
    base.CONFIG["LOSS_ALPHA_MIST_BOOST"] = 0.40

    base.main()


if __name__ == "__main__":
    main()
