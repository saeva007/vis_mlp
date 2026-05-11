#!/usr/bin/env python3

import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

import PMST_net_test_11_s2_pm10 as base


BASE_PATH = "/public/home/putianshu/vis_mlp"
DEFAULT_DATA_DIR = os.path.join(BASE_PATH, "ml_dataset_airport_metar_2025_12h")
DEFAULT_CKPT_DIR = os.path.join(BASE_PATH, "checkpoints")


def airport_dyn_indices_log1p(dyn_vars_count: int):
    """Airport dataset has no aerosol channels; log-transform precip, SW radiation, CAPE only."""
    return [2, 4, 9]


base._dyn_indices_log1p = airport_dyn_indices_log1p
CONFIG = base.CONFIG


def load_airport_metadata(data_dir: str):
    path = os.path.join(data_dir, "dataset_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing airport dataset metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_airport_data(
    data_dir,
    scaler=None,
    rank=0,
    local_rank=0,
    device=None,
    reuse_scaler=False,
    win_size=12,
    world_size=1,
    exp_id=None,
):
    path_x = base.copy_to_local(
        os.path.join(data_dir, "X_train.npy"), rank, local_rank, world_size, exp_id
    )
    path_y = base.copy_to_local(
        os.path.join(data_dir, "y_train.npy"), rank, local_rank, world_size, exp_id
    )

    x_shape = np.load(path_x, mmap_mode="r").shape
    if len(x_shape) != 2:
        raise ValueError(f"X_train.npy must be 2D [N, D], got shape={x_shape}")
    dyn_vars_count, inferred_extra_dim = base._resolve_dyn_and_fe_dims(
        int(x_shape[1]), win_size
    )
    CONFIG["DYN_VARS_COUNT"] = int(dyn_vars_count)
    CONFIG["FE_EXTRA_DIMS"] = int(inferred_extra_dim)
    base_dim = int(win_size * dyn_vars_count + 5 + 1)
    if base_dim + inferred_extra_dim != int(x_shape[1]):
        raise ValueError(
            f"Layout check failed: dyn/static/veg base={base_dim}, "
            f"FE={inferred_extra_dim}, total={x_shape[1]}"
        )

    y_raw = np.load(path_y)
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    if np.nanmax(y_raw) < 100:
        y_raw = y_raw * 1000.0
    y_cls[y_raw >= 500] = 1
    y_cls[y_raw >= 1000] = 2

    scaler_path = os.path.join(
        CONFIG["SAVE_CKPT_DIR"],
        f"robust_scaler_w{win_size}_dyn{dyn_vars_count}_airport_metar.pkl",
    )

    if scaler is None and not reuse_scaler:
        base.safe_barrier(world_size, device)
        if rank == 0:
            if not os.path.exists(scaler_path):
                print("[Airport Scaler] Fitting airport METAR scaler...", flush=True)
                x_m = np.load(path_x, mmap_mode="r")
                n_total = len(x_m)
                max_samples = 200000
                rng = np.random.default_rng(seed=42)
                if n_total > max_samples:
                    sample_indices = rng.choice(n_total, size=max_samples, replace=False)
                    sample_indices.sort()
                else:
                    sample_indices = np.arange(n_total)
                sub = x_m[sample_indices, : win_size * dyn_vars_count + 5].astype(np.float32)

                log_mask = np.zeros(win_size * dyn_vars_count, dtype=bool)
                for t in range(win_size):
                    for i in airport_dyn_indices_log1p(dyn_vars_count):
                        log_mask[t * dyn_vars_count + i] = True
                sub[:, : win_size * dyn_vars_count] = np.where(
                    log_mask,
                    np.log1p(np.maximum(sub[:, : win_size * dyn_vars_count], 0)),
                    sub[:, : win_size * dyn_vars_count],
                )

                scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(sub)
                joblib.dump(scaler, scaler_path)
                print(f"[Airport Scaler] Saved -> {scaler_path}", flush=True)
            else:
                print(f"[Airport Scaler] Loading cached scaler: {scaler_path}", flush=True)
        base.safe_barrier(world_size, device)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler missing after barrier: {scaler_path}")
        scaler = joblib.load(scaler_path)

    n_total = len(y_cls)
    n_val = int(n_total * CONFIG["VAL_SPLIT_RATIO"])
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(n_total)

    train_ds = base.PMSTDataset(
        path_x,
        y_cls,
        y_raw,
        scaler,
        win_size,
        True,
        indices[n_val:],
        dyn_vars_count=dyn_vars_count,
    )
    val_ds = base.PMSTDataset(
        path_x,
        y_cls,
        y_raw,
        scaler,
        win_size,
        True,
        indices[:n_val],
        dyn_vars_count=dyn_vars_count,
    )
    return train_ds, val_ds, scaler


def build_airport_loss(device):
    return base.DualBranchLoss(
        binary_pos_weight=CONFIG["S2_BINARY_POS_WEIGHT"],
        fine_class_weight=[
            CONFIG["S2_FINE_CLASS_WEIGHT_FOG"],
            CONFIG["S2_FINE_CLASS_WEIGHT_MIST"],
            CONFIG["S2_FINE_CLASS_WEIGHT_CLEAR"],
        ],
        loss_type="ordinal_focal",
        gamma_per_class=[2.5, 3.0, 0.5],
        ordinal_cost=[[0, 1, 3], [1, 0, 2], [3, 2, 0]],
        alpha_binary=CONFIG["S2_LOSS_ALPHA_BINARY"],
        alpha_fine=CONFIG["S2_LOSS_ALPHA_FINE"],
        alpha_fp=CONFIG["S2_LOSS_ALPHA_FP"],
        alpha_fog_boost=CONFIG["S2_LOSS_ALPHA_FOG_BOOST"],
        alpha_mist_boost=CONFIG["S2_LOSS_ALPHA_MIST_BOOST"],
        alpha_clear_margin=CONFIG["S2_LOSS_ALPHA_CLEAR_MARGIN"],
        clear_margin=CONFIG["S2_CLEAR_MARGIN"],
        alpha_pair_margin=CONFIG["S2_LOSS_ALPHA_PAIR_MARGIN"],
        pair_margin=CONFIG["S2_PAIR_MARGIN"],
    ).to(device)


def build_full_optimizer(raw_model, local_rank, world_size, rank):
    for param in raw_model.parameters():
        param.requires_grad = True

    head_names = {"fine_classifier", "low_vis_detector", "reg_head"}
    head_params = []
    backbone_params = []
    for name, param in raw_model.named_parameters():
        if name.split(".")[0] in head_names:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if rank == 0:
        print(
            f"[Airport Optim] Full training from scratch. "
            f"backbone={sum(p.numel() for p in backbone_params)/1e6:.3f}M, "
            f"heads={sum(p.numel() for p in head_params)/1e6:.3f}M",
            flush=True,
        )
        print(
            f"[Airport Optim] LR backbone={CONFIG['AIRPORT_LR_BACKBONE']:.1e}, "
            f"LR head={CONFIG['AIRPORT_LR_HEAD']:.1e}",
            flush=True,
        )

    model = base.wrap_ddp(raw_model, local_rank, world_size, find_unused=False)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG["AIRPORT_LR_BACKBONE"]},
            {"params": head_params, "lr": CONFIG["AIRPORT_LR_HEAD"]},
        ],
        weight_decay=CONFIG["S2_WEIGHT_DECAY"],
    )
    return model, optimizer


def save_airport_inference_artifacts(
    raw_model,
    scaler,
    metadata,
    run_exp_id,
    temperature,
    season_thresholds,
    rank,
):
    if rank != 0:
        return

    model_config = {
        "model_type": "improved_dual_stream_pmst",
        "window_size": int(CONFIG["WINDOW_SIZE"]),
        "dyn_vars_count": int(CONFIG["DYN_VARS_COUNT"]),
        "static_cont_dim": 5,
        "veg_num_classes": 21,
        "hidden_dim": int(CONFIG["MODEL_HIDDEN_DIM"]),
        "num_classes": int(CONFIG["MODEL_NUM_CLASSES"]),
        "extra_feat_dim": int(CONFIG["FE_EXTRA_DIMS"]),
        "dropout": float(CONFIG["MODEL_DROPOUT"]),
    }
    payload = {
        "model_state_dict": raw_model.state_dict(),
        "model_config": model_config,
        "config": model_config,
        "temperature": float(temperature),
        "season_thresholds": season_thresholds,
        "dynamic_feature_order": metadata.get("dynamic_feature_order", []),
        "station_order": metadata.get("station_order", []),
        "station_static": metadata.get("station_static", {}),
        "fill_values": metadata.get("fill_values", {}),
        "dataset_metadata": metadata,
    }
    package_path = os.path.join(CONFIG["SAVE_CKPT_DIR"], f"{run_exp_id}_airport_inference_package.pt")
    torch.save(payload, package_path)

    preprocessor = {
        "scaler": scaler,
        "model_type": "improved_dual_stream_pmst",
        "model_config": model_config,
        "temperature": float(temperature),
        "season_thresholds": season_thresholds,
        "dynamic_feature_order": metadata.get("dynamic_feature_order", []),
        "station_order": metadata.get("station_order", []),
        "station_static": metadata.get("station_static", {}),
        "fill_values": metadata.get("fill_values", {}),
        "window_size": int(CONFIG["WINDOW_SIZE"]),
        "dyn_vars_count": int(CONFIG["DYN_VARS_COUNT"]),
        "extra_feature_dim": int(CONFIG["FE_EXTRA_DIMS"]),
        "local_time_offset_hours": float(metadata.get("local_time_offset_hours", 8)),
        "use_source_zenith": True,
    }
    preprocessor_path = os.path.join(CONFIG["SAVE_CKPT_DIR"], f"{run_exp_id}_airport_preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[Airport Save] Inference package -> {package_path}", flush=True)
    print(f"[Airport Save] Preprocessor      -> {preprocessor_path}", flush=True)


def configure(args):
    CONFIG.update(
        {
            "EXPERIMENT_ID": args.exp_id,
            "S2_RUN_SUFFIX": args.run_suffix,
            "BASE_PATH": BASE_PATH,
            "S2_DATA_DIR": args.data_dir,
            "SAVE_CKPT_DIR": args.save_ckpt_dir,
            "WINDOW_SIZE": args.window_size,
            "NUM_WORKERS": args.num_workers,
            "VAL_SPLIT_RATIO": args.val_ratio,
            "S1_BEST_CKPT_PATH": None,
            "DYN_VARS_COUNT": 25,
            "FE_EXTRA_DIMS": 36,
            "AIRPORT_TOTAL_STEPS": args.total_steps,
            "AIRPORT_LR_BACKBONE": args.lr_backbone,
            "AIRPORT_LR_HEAD": args.lr_head,
            "S2_BATCH_SIZE": args.batch_size,
            "S2_GRAD_ACCUM": args.grad_accum,
            "S2_VAL_INTERVAL": args.val_interval,
            "S2_ES_PATIENCE": args.patience,
            "S2_FOG_RATIO": args.fog_ratio,
            "S2_MIST_RATIO": args.mist_ratio,
            "S2_WARMUP_STEPS": args.warmup_steps,
        }
    )
    os.makedirs(CONFIG["SAVE_CKPT_DIR"], exist_ok=True)


def main():
    args = parse_args()
    configure(args)

    local_rank, global_rank, world_size = base.init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    base_exp_id = CONFIG["EXPERIMENT_ID"]
    run_exp_id = base.build_s2_run_exp_id(base_exp_id, CONFIG.get("S2_RUN_SUFFIX", ""))
    metadata = load_airport_metadata(CONFIG["S2_DATA_DIR"])

    if global_rank == 0:
        print("[Airport] Full training from scratch; no S1 pretraining is loaded.", flush=True)
        print(f"[Airport] Run ID: {run_exp_id}", flush=True)
        print(f"[Airport] World size: {world_size}", flush=True)

    base.safe_barrier(world_size, device)
    dyn_res, fe_res = base.resolve_feature_layout_from_x_train(
        CONFIG["S2_DATA_DIR"], CONFIG["WINDOW_SIZE"]
    )
    CONFIG["DYN_VARS_COUNT"] = int(dyn_res)
    CONFIG["FE_EXTRA_DIMS"] = int(fe_res)
    if global_rank == 0:
        print(
            f"[Airport] Feature layout: dyn={dyn_res}, static=5, veg=1, FE={fe_res}",
            flush=True,
        )

    raw_model = base.ImprovedDualStreamPMSTNet(
        window_size=CONFIG["WINDOW_SIZE"],
        hidden_dim=CONFIG["MODEL_HIDDEN_DIM"],
        num_classes=CONFIG["MODEL_NUM_CLASSES"],
        extra_feat_dim=CONFIG["FE_EXTRA_DIMS"],
        dyn_vars_count=CONFIG["DYN_VARS_COUNT"],
    ).to(device)
    if global_rank == 0:
        print(f"[Airport Model] Params={sum(p.numel() for p in raw_model.parameters())/1e6:.2f}M", flush=True)

    train_ds, val_ds, scaler = load_airport_data(
        CONFIG["S2_DATA_DIR"],
        None,
        global_rank,
        local_rank,
        device,
        False,
        CONFIG["WINDOW_SIZE"],
        world_size,
        run_exp_id,
    )
    if global_rank == 0:
        print(f"[Airport Data] Train={len(train_ds)}, Val={len(val_ds)}", flush=True)

    model, optimizer = build_full_optimizer(raw_model, local_rank, world_size, global_rank)
    loss_fn = build_airport_loss(device)

    base.train_stage(
        tag="Airport_Full",
        model=model,
        tr_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        rank=global_rank,
        world_size=world_size,
        total_steps=CONFIG["AIRPORT_TOTAL_STEPS"],
        val_int=CONFIG["S2_VAL_INTERVAL"],
        batch_size=CONFIG["S2_BATCH_SIZE"],
        grad_accum=CONFIG["S2_GRAD_ACCUM"],
        fog_ratio=CONFIG["S2_FOG_RATIO"],
        mist_ratio=CONFIG["S2_MIST_RATIO"],
        exp_id=run_exp_id,
        patience=CONFIG["S2_ES_PATIENCE"],
        pretrained_state=None,
        l2sp_alpha=0.0,
    )

    raw_final = base.rewrap_ddp(model, world_size)
    best_path = os.path.join(CONFIG["SAVE_CKPT_DIR"], f"{run_exp_id}_Airport_Full_best_score.pt")
    if os.path.exists(best_path):
        base.load_checkpoint(raw_final, best_path, global_rank, world_size, device)
    elif global_rank == 0:
        print(f"[Airport] WARNING: best checkpoint not found, using last in-memory weights: {best_path}", flush=True)

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info.dataset.X = None

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["S2_BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    optimal_temp = base.calibrate_temperature(raw_final, val_loader, device, CONFIG, rank=global_rank)
    season_thresholds = base.evaluate_per_season(
        raw_final, val_loader, device, CONFIG, rank=global_rank, temperature=optimal_temp
    )
    save_airport_inference_artifacts(
        raw_final, scaler, metadata, run_exp_id, optimal_temp, season_thresholds, global_rank
    )

    base.cleanup_temp_files(run_exp_id)
    if world_size > 1:
        base.dist.destroy_process_group()
    if global_rank == 0:
        print("[Airport] Job finished.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the original multi-GPU PMST model on airport METAR visibility data."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--exp-id", default="airport_metar_2025")
    parser.add_argument("--run-suffix", default="full_from_scratch")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--fog-ratio", type=float, default=0.18)
    parser.add_argument("--mist-ratio", type=float, default=0.22)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
