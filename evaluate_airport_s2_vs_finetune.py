#!/usr/bin/env python3
"""Compare airport25 S2 and airport-METAR fine-tuned Static-RNN checkpoints.

The evaluator intentionally reuses the same preprocessing as
train_static_rnn_lowvis.py: feature-layout metadata, log1p dynamic transforms,
RobustScaler, vegetation id, and optional FE block.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train_static_rnn_lowvis import (
    Layout,
    StaticRNNLowVisNet,
    _normalise_state_dict_keys,
    apply_core_transform,
    build_dyn_log_mask,
    build_metrics,
    resolve_layout_from_file,
    score_metrics,
    visibility_to_labels,
)


DEFAULT_BASE = "/public/home/putianshu/vis_mlp"
DEFAULT_DATA_DIR = f"{DEFAULT_BASE}/ml_dataset_static_rnn_airport_metar_2025_12h"
DEFAULT_CKPT_DIR = f"{DEFAULT_BASE}/checkpoints"


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def checkpoint_path_from_run(ckpt_dir: str, run_id: str) -> str:
    return os.path.join(ckpt_dir, f"{run_id}_S2_PhaseB_best_score.pt")


def scaler_path_from_run(ckpt_dir: str, run_id: str, window_size: int, dyn_vars: int) -> str:
    return os.path.join(ckpt_dir, f"robust_scaler_{run_id}_s2_w{window_size}_dyn{dyn_vars}_nopm.pkl")


def load_checkpoint_payload(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict]:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state_dict-like object")
    return _normalise_state_dict_keys(state), metadata if isinstance(metadata, dict) else {}


def layout_from_checkpoint_or_data(
    metadata: Mapping,
    x_path: str,
    data_dir: str,
    window_size: int,
) -> Layout:
    meta_layout = metadata.get("layout") if isinstance(metadata, Mapping) else None
    data_layout = resolve_layout_from_file(x_path, window_size, data_dir)
    if isinstance(meta_layout, Mapping):
        ckpt_layout = Layout(
            window_size=int(meta_layout.get("window_size", data_layout.window_size)),
            dyn_vars=int(meta_layout.get("dyn_vars", data_layout.dyn_vars)),
            fe_dim=int(meta_layout.get("fe_dim", data_layout.fe_dim)),
            dynamic_feature_order=(
                [str(v) for v in meta_layout.get("dynamic_feature_order")]
                if isinstance(meta_layout.get("dynamic_feature_order"), list)
                else data_layout.dynamic_feature_order
            ),
        )
        if asdict(ckpt_layout) != asdict(data_layout):
            raise ValueError(
                "Checkpoint/data layout mismatch: "
                f"checkpoint={ckpt_layout}, data={data_layout}. "
                "Use a checkpoint trained with the airport25/static_rnn_airport layout."
            )
        return ckpt_layout
    return data_layout


def infer_arch_from_state(state: Mapping[str, torch.Tensor], metadata: Mapping, defaults: argparse.Namespace) -> Dict[str, object]:
    dyn_w = state.get("dynamic_proj.0.weight")
    hidden_dim = int(dyn_w.shape[0]) if torch.is_tensor(dyn_w) and dyn_w.ndim == 2 else int(defaults.hidden_dim)
    stat_w = state.get("static_encoder.0.weight")
    static_hidden_dim = int(stat_w.shape[0]) if torch.is_tensor(stat_w) and stat_w.ndim == 2 else int(defaults.static_hidden_dim)
    fe_w = state.get("fe_encoder.0.weight")
    fe_hidden_dim = int(fe_w.shape[0]) if torch.is_tensor(fe_w) and fe_w.ndim == 2 else int(defaults.fe_hidden_dim)
    fusion_w = state.get("fusion.0.weight")
    fusion_hidden_dim = int(fusion_w.shape[0]) if torch.is_tensor(fusion_w) and fusion_w.ndim == 2 else int(defaults.fusion_hidden_dim)
    veg_w = state.get("veg_embedding.weight")
    veg_emb_dim = int(veg_w.shape[1]) if torch.is_tensor(veg_w) and veg_w.ndim == 2 else int(defaults.veg_emb_dim)

    # GRU/LSTM and number of layers can usually be inferred from recurrent keys.
    encoder = str(metadata.get("encoder", defaults.encoder)).lower()
    if any(str(k).startswith("rnn.weight_ih_l") for k in state):
        if any("weight_hr_l" in str(k) for k in state):
            encoder = "lstm"
    pooling = str(metadata.get("pooling", defaults.pooling)).lower()
    bidirectional = any("_reverse" in str(k) for k in state)
    layer_ids = []
    for key in state:
        text = str(key)
        if text.startswith("rnn.weight_ih_l"):
            suffix = text.split("rnn.weight_ih_l", 1)[1].split("_", 1)[0]
            if suffix.isdigit():
                layer_ids.append(int(suffix))
    rnn_layers = max(layer_ids) + 1 if layer_ids else int(defaults.rnn_layers)
    return {
        "encoder": "lstm" if encoder == "lstm" else "gru",
        "hidden_dim": hidden_dim,
        "static_hidden_dim": static_hidden_dim,
        "fe_hidden_dim": fe_hidden_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "veg_emb_dim": veg_emb_dim,
        "rnn_layers": rnn_layers,
        "dropout": float(defaults.dropout),
        "bidirectional": bool(bidirectional),
        "pooling": pooling if pooling in {"mean", "last", "attention"} else "mean",
    }


class StaticRNNEvalDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        layout: Layout,
        scaler,
        use_fe: bool,
        use_pm: bool,
    ) -> None:
        self.x_path = x_path
        self.y_path = y_path
        self.layout = layout
        self.scaler = scaler
        self.use_fe = bool(use_fe)
        self.use_pm = bool(use_pm)
        self.log_mask = build_dyn_log_mask(layout)
        self.X = np.load(x_path, mmap_mode="r")
        self.y_raw, self.y_cls = visibility_to_labels(np.load(y_path, mmap_mode="r"))
        if self.X.shape[0] != len(self.y_cls):
            raise ValueError(f"X/y length mismatch: X={self.X.shape[0]}, y={len(self.y_cls)}")

    def __len__(self) -> int:
        return len(self.y_cls)

    def __getitem__(self, idx: int):
        row = self.X[int(idx)]
        core = row[: self.layout.core_dim][None, :]
        core = apply_core_transform(core, self.layout, self.use_pm, self.log_mask)[0]
        core = (core - self.scaler.center_) / (self.scaler.scale_ + 1e-6)
        core = np.clip(core, -10.0, 10.0).astype(np.float32)
        veg = np.asarray([row[self.layout.split_dyn + 5]], dtype=np.float32)
        parts = [core, veg]
        if self.use_fe:
            fe = row[self.layout.split_dyn + 6 : self.layout.split_dyn + 6 + self.layout.fe_dim]
            parts.append(np.clip(fe.astype(np.float32), -10.0, 10.0))
        x = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.from_numpy(x).float(), int(self.y_cls[int(idx)]), float(self.y_raw[int(idx)]), int(idx)


def choose_split(data_dir: str, split: str) -> str:
    if split != "auto":
        return split
    if os.path.isfile(os.path.join(data_dir, "X_test.npy")) and os.path.isfile(os.path.join(data_dir, "y_test.npy")):
        return "test"
    return "val"


def split_paths(data_dir: str, split: str) -> Tuple[str, str, Optional[str]]:
    x_path = os.path.join(data_dir, f"X_{split}.npy")
    y_path = os.path.join(data_dir, f"y_{split}.npy")
    meta_path = os.path.join(data_dir, f"meta_{split}.csv")
    missing = [p for p in (x_path, y_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing split files for split={split}: {missing}. "
            "Airport builder defaults to train/val only; use --split val or rebuild with --test-ratio > 0."
        )
    return x_path, y_path, meta_path if os.path.isfile(meta_path) else None


def confusion_matrix(y_true: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def station_metrics(meta_path: Optional[str], y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    if not meta_path:
        return pd.DataFrame()
    meta = pd.read_csv(meta_path)
    station_col = "station" if "station" in meta.columns else ("station_name" if "station_name" in meta.columns else None)
    if station_col is None or len(meta) != len(y_true):
        return pd.DataFrame()
    rows = []
    for station, idx in meta.groupby(station_col).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        yt = y_true[idx_arr]
        pr = pred[idx_arr]
        low_true = yt <= 1
        low_pred = pr <= 1
        rows.append(
            {
                "station": station,
                "n": int(len(idx_arr)),
                "fog_true": int((yt == 0).sum()),
                "mist_true": int((yt == 1).sum()),
                "clear_true": int((yt == 2).sum()),
                "low_true": int(low_true.sum()),
                "low_pred": int(low_pred.sum()),
                "accuracy": float(np.mean(yt == pr)) if len(idx_arr) else np.nan,
                "low_vis_recall": float(np.sum(low_true & low_pred) / (np.sum(low_true) + 1e-6)),
                "low_vis_precision": float(np.sum(low_true & low_pred) / (np.sum(low_pred) + 1e-6)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_one(
    name: str,
    ckpt_path: str,
    scaler_path: str,
    x_path: str,
    y_path: str,
    meta_path: Optional[str],
    data_dir: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    print(f"[Eval:{name}] checkpoint={ckpt_path}", flush=True)
    print(f"[Eval:{name}] scaler={scaler_path}", flush=True)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found for {name}: {ckpt_path}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"scaler not found for {name}: {scaler_path}")
    state, metadata = load_checkpoint_payload(ckpt_path, device)
    layout = layout_from_checkpoint_or_data(metadata, x_path, data_dir, args.window_size)
    scaler = joblib.load(scaler_path)
    use_fe = bool(metadata.get("use_fe", not args.no_fe))
    use_pm = not args.no_pm
    arch = infer_arch_from_state(state, metadata, args)
    model = StaticRNNLowVisNet(layout=layout, use_fe=use_fe, **arch).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[Eval:{name}] load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()

    ds = StaticRNNEvalDataset(x_path, y_path, layout, scaler, use_fe=use_fe, use_pm=use_pm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    probs_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    raw_parts: List[np.ndarray] = []
    idx_parts: List[np.ndarray] = []
    with torch.no_grad():
        for bx, by, y_raw, idx in loader:
            bx = bx.to(device, non_blocking=True)
            logits, reg = model(bx)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            probs_parts.append(prob.detach().cpu().numpy().astype(np.float32))
            pred_parts.append(pred.detach().cpu().numpy().astype(np.int64))
            y_parts.append(by.numpy().astype(np.int64))
            raw_parts.append(y_raw.numpy().astype(np.float32))
            idx_parts.append(idx.numpy().astype(np.int64))

    probs = np.concatenate(probs_parts, axis=0)
    pred = np.concatenate(pred_parts, axis=0)
    y_true = np.concatenate(y_parts, axis=0)
    y_raw = np.concatenate(raw_parts, axis=0)
    row_idx = np.concatenate(idx_parts, axis=0)

    metrics = build_metrics(y_true, pred)
    score = score_metrics(args, metrics)
    cm = confusion_matrix(y_true, pred)
    station_df = station_metrics(meta_path, y_true, pred)
    if not station_df.empty:
        station_df.insert(0, "model", name)

    pred_df = pd.DataFrame(
        {
            "row": row_idx,
            "y_true": y_true,
            "visibility_m": y_raw,
            "pred": pred,
            "p_fog": probs[:, 0],
            "p_mist": probs[:, 1],
            "p_clear": probs[:, 2],
        }
    )
    if meta_path:
        meta = pd.read_csv(meta_path)
        if len(meta) == len(pred_df):
            keep = [c for c in ("time", "station", "station_name", "station_idx", "lat", "lon") if c in meta.columns]
            pred_df = pd.concat([meta[keep].reset_index(drop=True), pred_df], axis=1)
    pred_df.insert(0, "model", name)

    result = {
        "model": name,
        "checkpoint": ckpt_path,
        "scaler": scaler_path,
        "score": float(score),
        "n": int(len(y_true)),
        "true_fog": int((y_true == 0).sum()),
        "true_mist": int((y_true == 1).sum()),
        "true_clear": int((y_true == 2).sum()),
        "pred_fog": int((pred == 0).sum()),
        "pred_mist": int((pred == 1).sum()),
        "pred_clear": int((pred == 2).sum()),
        "layout": asdict(layout),
        "use_fe": bool(use_fe),
        "use_pm": bool(use_pm),
        "arch": arch,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "threshold_mode": "argmax",
    }
    return result, station_df, pred_df


def summary_rows(results: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for res in results:
        metrics = res["metrics"]
        row = {
            "model": res["model"],
            "score": res["score"],
            "n": res["n"],
            "true_fog": res["true_fog"],
            "true_mist": res["true_mist"],
            "true_clear": res["true_clear"],
            "pred_fog": res["pred_fog"],
            "pred_mist": res["pred_mist"],
            "pred_clear": res["pred_clear"],
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_report(results: List[Mapping[str, object]], split: str, winner_metric: str) -> str:
    df = summary_rows(results)
    if winner_metric == "score":
        best_idx = df["score"].astype(float).idxmax()
    else:
        best_idx = df[winner_metric].astype(float).idxmax()
    winner = str(df.loc[best_idx, "model"])
    def table_text(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return "```csv\n" + frame.to_csv(index=False) + "```"

    lines = [
        "# Airport S2 vs Fine-tune Evaluation",
        "",
        f"- Split: `{split}`",
        f"- Decision rule: `argmax`",
        f"- Winner metric: `{winner_metric}`",
        f"- Winner: `{winner}`",
        "",
        "## Summary",
        "",
        table_text(df),
        "",
        "## Confusion Matrices",
        "",
    ]
    labels = ["Fog(<500)", "Mist(500-1000)", "Clear(>=1000)"]
    for res in results:
        cm_df = pd.DataFrame(res["confusion_matrix"], index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        lines.extend([f"### {res['model']}", "", table_text(cm_df.reset_index().rename(columns={"index": "true/pred"})), ""])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate airport25 S2 vs airport METAR fine-tune on an airport split.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", choices=["auto", "test", "val", "train"], default="auto")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--s2-run-id", default=os.environ.get("AIRPORT25_MAIN_RUN_ID", ""))
    p.add_argument("--finetune-run-id", default=os.environ.get("AIRPORT25_FT_RUN_ID", ""))
    p.add_argument("--s2-ckpt", default="")
    p.add_argument("--s2-scaler", default="")
    p.add_argument("--finetune-ckpt", default="")
    p.add_argument("--finetune-scaler", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--no-pm", action="store_true", default=True)
    p.add_argument("--no-fe", action="store_true")
    p.add_argument("--winner-metric", default="score")
    p.add_argument("--selection-metric", choices=["recall_csi", "csi", "recall"], default="recall_csi")

    # Architecture defaults; actual checkpoint tensors override most of them.
    p.add_argument("--encoder", choices=["gru", "lstm"], default="gru")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--static-hidden-dim", type=int, default=96)
    p.add_argument("--fe-hidden-dim", type=int, default=128)
    p.add_argument("--fusion-hidden-dim", type=int, default=256)
    p.add_argument("--veg-emb-dim", type=int, default=16)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--pooling", choices=["mean", "last", "attention"], default="mean")
    return p.parse_args()


def resolve_model_paths(args: argparse.Namespace, layout: Layout) -> Dict[str, Dict[str, str]]:
    if not args.s2_ckpt:
        if not args.s2_run_id:
            raise ValueError("Provide --s2-ckpt/--s2-scaler or --s2-run-id/AIRPORT25_MAIN_RUN_ID")
        args.s2_ckpt = checkpoint_path_from_run(args.ckpt_dir, args.s2_run_id)
    if not args.finetune_ckpt:
        if not args.finetune_run_id:
            raise ValueError("Provide --finetune-ckpt/--finetune-scaler or --finetune-run-id/AIRPORT25_FT_RUN_ID")
        args.finetune_ckpt = checkpoint_path_from_run(args.ckpt_dir, args.finetune_run_id)
    if not args.s2_scaler:
        if not args.s2_run_id:
            raise ValueError("Provide --s2-scaler when --s2-run-id is not set")
        args.s2_scaler = scaler_path_from_run(args.ckpt_dir, args.s2_run_id, layout.window_size, layout.dyn_vars)
    if not args.finetune_scaler:
        if not args.finetune_run_id:
            raise ValueError("Provide --finetune-scaler when --finetune-run-id is not set")
        args.finetune_scaler = scaler_path_from_run(args.ckpt_dir, args.finetune_run_id, layout.window_size, layout.dyn_vars)
    return {
        "s2": {"ckpt": args.s2_ckpt, "scaler": args.s2_scaler},
        "finetune": {"ckpt": args.finetune_ckpt, "scaler": args.finetune_scaler},
    }


def main() -> None:
    args = parse_args()
    split = choose_split(args.data_dir, args.split)
    x_path, y_path, meta_path = split_paths(args.data_dir, split)
    layout = resolve_layout_from_file(x_path, args.window_size, args.data_dir)
    model_paths = resolve_model_paths(args, layout)
    out_dir = args.out_dir or os.path.join(args.data_dir, f"eval_s2_vs_finetune_{split}")
    os.makedirs(out_dir, exist_ok=True)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Data] split={split} X={x_path} y={y_path} layout={layout}", flush=True)
    print(f"[Device] {device}", flush=True)

    results = []
    station_tables = []
    prediction_tables = []
    for name in ("s2", "finetune"):
        res, station_df, pred_df = evaluate_one(
            name,
            model_paths[name]["ckpt"],
            model_paths[name]["scaler"],
            x_path,
            y_path,
            meta_path,
            args.data_dir,
            args,
            device,
        )
        results.append(res)
        if not station_df.empty:
            station_tables.append(station_df)
        prediction_tables.append(pred_df)

    summary_df = summary_rows(results)
    summary_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    with open(os.path.join(out_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe({"split": split, "results": results}), f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "comparison_summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(results, split, args.winner_metric))
    for res in results:
        pd.DataFrame(
            res["confusion_matrix"],
            index=["true_fog", "true_mist", "true_clear"],
            columns=["pred_fog", "pred_mist", "pred_clear"],
        ).to_csv(os.path.join(out_dir, f"{res['model']}_confusion_matrix.csv"))
    if station_tables:
        pd.concat(station_tables, ignore_index=True).to_csv(os.path.join(out_dir, "station_metrics.csv"), index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print(summary_df.to_string(index=False), flush=True)
    print(f"[Write] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
