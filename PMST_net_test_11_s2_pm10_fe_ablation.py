#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-engineering ablation launcher for PMST_net_test_11_s2_pm10.py.

This wrapper keeps the Stage-2 model, loss, sampling, S1 loading, and
three-phase fine-tuning protocol from PMST_net_test_11_s2_pm10.py.  The only
experimental intervention is a deterministic mask on the precomputed FE block
after dynamic/static preprocessing and before the tensor is returned by the
dataset.

The current PM10+PM2.5 month-tail dataset stores PM10/PM2.5 in the dynamic
variables.  These ablations therefore target only the engineered FE stream
described by s2_data.py: 32 fog/meteorological proxies plus 4 cyclic time
features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import PMST_net_test_11_s2_pm10 as base


FE_FEATURE_NAMES: List[str] = [
    "saturation_dpd_proxy",
    "wind_favorability",
    "inversion_weak_wind_stability",
    "night_clear_sky_radiative_cooling",
    "rh2m_minus_rh925",
    "composite_fog_potential",
    "rh2m_delta_3h",
    "rh2m_delta_6h",
    "rh2m_std_12h",
    "rh2m_range_12h",
    "t2m_delta_3h",
    "t2m_delta_6h",
    "t2m_std_12h",
    "t2m_range_12h",
    "wspd10_delta_3h",
    "wspd10_delta_6h",
    "wspd10_std_12h",
    "wspd10_range_12h",
    "rh2m_acceleration",
    "moist_cold_proxy",
    "night_low_cloud_proxy",
    "cold_humid_weak_wind_indicator",
    "rh_low_cloud_ratio",
    "rh_squared_proxy",
    "low_level_shear_magnitude",
    "low_level_direction_turning",
    "convective_wet_proxy",
    "daytime_mixing_proxy",
    "ventilation_proxy",
    "moisture_stratification",
    "vertical_velocity_contrast",
    "warm_instability_proxy",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]


FE_GROUPS: Dict[str, Tuple[int, ...]] = {
    "core_physics": tuple(range(0, 6)),
    "temporal_stats": tuple(range(6, 19)),
    "empirical_flags": tuple(range(19, 24)),
    "boundary_layer": tuple(range(24, 32)),
    "time_cyc": tuple(range(32, 36)),
}


ABLATION_MODES: Dict[str, str] = {
    "full": "Keep all engineered features.",
    "no_fe_all": "Zero the entire engineered-feature block.",
    "no_core_physics": "Zero saturation, wind, stability, radiative cooling, humidity contrast, and composite fog potential proxies.",
    "no_temporal_stats": "Zero 12 h tendency, variability, range, and RH acceleration proxies.",
    "no_empirical_flags": "Zero compact empirical humid/cold/cloud flags.",
    "no_boundary_layer": "Zero shear, direction turning, convection, mixing, ventilation, moisture stratification, vertical-velocity contrast, and warm-instability proxies.",
    "no_time_cyc": "Zero cyclic month/hour features.",
    "only_core_physics": "Keep only the core physics group and zero all other FE dimensions.",
    "only_temporal_stats": "Keep only the temporal-statistics group and zero all other FE dimensions.",
    "only_time_cyc": "Keep only cyclic month/hour features and zero all other FE dimensions.",
    "custom_drop": "Zero indices passed with --custom_indices.",
    "custom_keep": "Keep only indices passed with --custom_indices.",
}


def _bounded(indices: Iterable[int], fe_dim: int) -> List[int]:
    return sorted({int(i) for i in indices if 0 <= int(i) < int(fe_dim)})


def parse_index_spec(spec: str, fe_dim: int) -> List[int]:
    """Parse comma-separated indices/ranges like '0,2,6:19'."""
    if not spec.strip():
        return []
    out: List[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
            start = int(left) if left.strip() else 0
            stop = int(right) if right.strip() else fe_dim
            out.extend(range(start, stop))
        else:
            out.append(int(token))
    return _bounded(out, fe_dim)


def mask_for_mode(mode: str, fe_dim: int, custom_indices: str = "") -> Tuple[np.ndarray, List[int], List[int]]:
    """Return a 0/1 mask plus kept and dropped FE indices."""
    if fe_dim <= 0:
        return np.ones(0, dtype=np.float32), [], []

    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown FE ablation mode: {mode}")

    named_dim = min(fe_dim, len(FE_FEATURE_NAMES))
    all_indices = set(range(fe_dim))

    if mode == "full":
        dropped: List[int] = []
    elif mode == "no_fe_all":
        dropped = list(range(fe_dim))
    elif mode == "custom_drop":
        dropped = parse_index_spec(custom_indices, fe_dim)
    elif mode == "custom_keep":
        keep = set(parse_index_spec(custom_indices, fe_dim))
        dropped = sorted(all_indices - keep)
    elif mode.startswith("no_"):
        group_name = mode[3:]
        dropped = _bounded(FE_GROUPS[group_name], fe_dim)
    elif mode.startswith("only_"):
        group_name = mode[5:]
        keep = set(_bounded(FE_GROUPS[group_name], fe_dim))
        dropped = sorted(all_indices - keep)
    else:
        dropped = []

    mask = np.ones(fe_dim, dtype=np.float32)
    if dropped:
        mask[np.asarray(dropped, dtype=np.int64)] = 0.0
    kept = [i for i in range(fe_dim) if mask[i] > 0.5]

    # Unknown legacy tail columns are intentionally reported, not hidden.
    if fe_dim > named_dim:
        print(
            f"[FE-Ablation] FE dim={fe_dim}; named current features cover 0:{named_dim}. "
            f"Tail columns {named_dim}:{fe_dim} are treated by the selected mode.",
            flush=True,
        )
    return mask, kept, dropped


class FeatureAblationPMSTDataset(base.PMSTDataset):
    """PMSTDataset variant that masks the FE block only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x_shape = np.load(self.X_path, mmap_mode="r").shape
        self.fe_dim = int(x_shape[1]) - int(self.split_dyn) - 6
        if self.fe_dim < 0:
            raise ValueError(
                f"Cannot infer FE dim from X shape={x_shape}, split_dyn={self.split_dyn}"
            )

        mode = str(base.CONFIG.get("FE_ABLATION_MODE", "full"))
        custom = str(base.CONFIG.get("FE_ABLATION_CUSTOM_INDICES", ""))
        mask, kept, dropped = mask_for_mode(mode, self.fe_dim, custom)
        self.fe_start = int(self.split_dyn) + 6
        self.fe_mask_t = torch.as_tensor(mask, dtype=torch.float32)
        self.fe_kept_indices = kept
        self.fe_dropped_indices = dropped

        if bool(base.CONFIG.get("FE_ABLATION_VERBOSE_DATASET", True)):
            print(
                f"[FE-Ablation] Dataset mode={mode} fe_dim={self.fe_dim} "
                f"kept={len(kept)} dropped={len(dropped)}",
                flush=True,
            )
            base.CONFIG["FE_ABLATION_VERBOSE_DATASET"] = False

    def __getitem__(self, idx):
        x, y_cls, y_reg, y_raw = super().__getitem__(idx)
        if self.fe_dim > 0 and torch.any(self.fe_mask_t == 0):
            end = min(x.numel(), self.fe_start + self.fe_dim)
            if end > self.fe_start:
                x[self.fe_start:end] *= self.fe_mask_t[: end - self.fe_start]
        return x, y_cls, y_reg, y_raw


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run Stage-2 PMST feature-engineering ablations by masking the FE block."
    )
    ap.add_argument("--ablation", choices=sorted(ABLATION_MODES), default="no_fe_all")
    ap.add_argument("--custom_indices", default="", help="Used by custom_drop/custom_keep, e.g. '0,2,6:19'.")
    ap.add_argument("--list_ablations", action="store_true")
    ap.add_argument("--dry_run_config", action="store_true")

    ap.add_argument("--run_suffix", default="", help="Override CONFIG['S2_RUN_SUFFIX']; default appends feabl_<mode>.")
    ap.add_argument("--run_suffix_prefix", default="", help="Optional prefix inserted before feabl_<mode>.")
    ap.add_argument("--experiment_id", default="", help="Override base experiment id.")
    ap.add_argument("--s2_data_dir", default="", help="Override S2 data directory.")
    ap.add_argument("--s1_best_ckpt", default="", help="Override S1 best checkpoint path.")
    ap.add_argument("--save_ckpt_dir", default="", help="Override checkpoint directory.")

    ap.add_argument("--phase_a1_steps", type=int, default=None)
    ap.add_argument("--phase_a2_steps", type=int, default=None)
    ap.add_argument("--phase_b_steps", type=int, default=None)
    ap.add_argument("--steps_scale", type=float, default=1.0)
    ap.add_argument("--val_interval", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    return ap


def apply_config_overrides(args: argparse.Namespace) -> str:
    cfg = base.CONFIG

    if args.experiment_id:
        cfg["EXPERIMENT_ID"] = args.experiment_id
    if args.s2_data_dir:
        cfg["S2_DATA_DIR"] = args.s2_data_dir
    if args.s1_best_ckpt:
        cfg["S1_BEST_CKPT_PATH"] = args.s1_best_ckpt
    if args.save_ckpt_dir:
        cfg["SAVE_CKPT_DIR"] = args.save_ckpt_dir

    if args.steps_scale <= 0:
        raise ValueError("--steps_scale must be positive")
    if args.steps_scale != 1.0:
        for key in ("S2_PHASE_A1_STEPS", "S2_PHASE_A2_STEPS", "S2_PHASE_B_STEPS"):
            cfg[key] = max(1, int(round(int(cfg[key]) * float(args.steps_scale))))

    if args.phase_a1_steps is not None:
        cfg["S2_PHASE_A1_STEPS"] = int(args.phase_a1_steps)
    if args.phase_a2_steps is not None:
        cfg["S2_PHASE_A2_STEPS"] = int(args.phase_a2_steps)
    if args.phase_b_steps is not None:
        cfg["S2_PHASE_B_STEPS"] = int(args.phase_b_steps)
    if args.val_interval is not None:
        cfg["S2_VAL_INTERVAL"] = int(args.val_interval)
    if args.batch_size is not None:
        cfg["S2_BATCH_SIZE"] = int(args.batch_size)
    if args.grad_accum is not None:
        cfg["S2_GRAD_ACCUM"] = int(args.grad_accum)
    if args.num_workers is not None:
        cfg["NUM_WORKERS"] = int(args.num_workers)
    if args.patience is not None:
        cfg["S2_ES_PATIENCE"] = int(args.patience)

    cfg["FE_ABLATION_MODE"] = args.ablation
    cfg["FE_ABLATION_CUSTOM_INDICES"] = args.custom_indices
    cfg["FE_ABLATION_FEATURE_NAMES"] = FE_FEATURE_NAMES
    cfg["FE_ABLATION_GROUPS"] = {k: list(v) for k, v in FE_GROUPS.items()}
    cfg["FE_ABLATION_VERBOSE_DATASET"] = True

    original_suffix = str(cfg.get("S2_RUN_SUFFIX", "")).strip()
    if args.run_suffix:
        suffix = args.run_suffix.strip()
    else:
        parts = [p for p in (original_suffix, args.run_suffix_prefix.strip(), f"feabl_{args.ablation}") if p]
        suffix = "_".join(parts)
    cfg["S2_RUN_SUFFIX"] = suffix

    return base.build_s2_run_exp_id(str(cfg["EXPERIMENT_ID"]), suffix)


def write_metadata(run_exp_id: str, args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    save_dir = Path(base.CONFIG["SAVE_CKPT_DIR"])
    save_dir.mkdir(parents=True, exist_ok=True)
    fe_dim = int(base.CONFIG.get("FE_EXTRA_DIMS", len(FE_FEATURE_NAMES)))
    mask, kept, dropped = mask_for_mode(args.ablation, fe_dim, args.custom_indices)

    metadata = {
        "run_exp_id": run_exp_id,
        "source_script": str(Path(__file__).name),
        "base_training_script": "PMST_net_test_11_s2_pm10.py",
        "ablation_mode": args.ablation,
        "description": ABLATION_MODES[args.ablation],
        "custom_indices": args.custom_indices,
        "fe_dim_config_at_launch": fe_dim,
        "kept_indices_config_at_launch": kept,
        "dropped_indices_config_at_launch": dropped,
        "feature_names_0_35": FE_FEATURE_NAMES,
        "feature_groups": {k: list(v) for k, v in FE_GROUPS.items()},
        "config_subset": {
            "S2_DATA_DIR": base.CONFIG.get("S2_DATA_DIR"),
            "S1_BEST_CKPT_PATH": base.CONFIG.get("S1_BEST_CKPT_PATH"),
            "S2_PHASE_A1_STEPS": base.CONFIG.get("S2_PHASE_A1_STEPS"),
            "S2_PHASE_A2_STEPS": base.CONFIG.get("S2_PHASE_A2_STEPS"),
            "S2_PHASE_B_STEPS": base.CONFIG.get("S2_PHASE_B_STEPS"),
            "S2_BATCH_SIZE": base.CONFIG.get("S2_BATCH_SIZE"),
            "S2_GRAD_ACCUM": base.CONFIG.get("S2_GRAD_ACCUM"),
            "S2_FOG_RATIO": base.CONFIG.get("S2_FOG_RATIO"),
            "S2_MIST_RATIO": base.CONFIG.get("S2_MIST_RATIO"),
        },
    }
    path = save_dir / f"{run_exp_id}_fe_ablation_config.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FE-Ablation] Metadata -> {path}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ablations:
        for name in sorted(ABLATION_MODES):
            print(f"{name}: {ABLATION_MODES[name]}")
        return

    run_exp_id = apply_config_overrides(args)
    base.PMSTDataset = FeatureAblationPMSTDataset

    print("[FE-Ablation] Training wrapper active.", flush=True)
    print(f"[FE-Ablation] mode={args.ablation} run_exp_id={run_exp_id}", flush=True)
    print(f"[FE-Ablation] intervention={ABLATION_MODES[args.ablation]}", flush=True)
    if args.custom_indices:
        print(f"[FE-Ablation] custom_indices={args.custom_indices}", flush=True)

    if args.dry_run_config:
        print(json.dumps(base.CONFIG, indent=2, ensure_ascii=False, default=str))
        return

    write_metadata(run_exp_id, args)
    base.main()


if __name__ == "__main__":
    main(sys.argv[1:])
