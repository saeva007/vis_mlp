#!/usr/bin/env python3
"""Exact low-visibility threshold-search primitives with no ML dependencies."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def threshold_prediction_counts(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count fog/mist predictions by true class for every threshold.

    This exactly matches repeated strict-threshold predictions, but scans each
    score vector once rather than materializing one full prediction vector for
    every fog/mist threshold pair.
    """
    probabilities = np.asarray(probs)
    targets = np.asarray(y_true, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2 or len(probabilities) != len(targets):
        raise ValueError(f"Invalid threshold-search arrays: probs={probabilities.shape}, y={targets.shape}")
    if np.any((targets < 0) | (targets > 2)):
        raise ValueError("threshold search expects class labels in {0,1,2}")
    class_counts = np.bincount(targets, minlength=3).astype(np.int64)
    fog_eligible = probabilities[:, 0] >= probabilities[:, 1]
    mist_eligible = probabilities[:, 1] > probabilities[:, 0]
    compare_grid = np.asarray(grid, dtype=probabilities.dtype)
    fog_counts = np.zeros((len(grid), 3), dtype=np.int64)
    mist_counts = np.zeros((len(grid), 3), dtype=np.int64)
    for cls in range(3):
        fog_scores = np.sort(probabilities[fog_eligible & (targets == cls), 0])
        mist_scores = np.sort(probabilities[mist_eligible & (targets == cls), 1])
        fog_counts[:, cls] = len(fog_scores) - np.searchsorted(fog_scores, compare_grid, side="right")
        mist_counts[:, cls] = len(mist_scores) - np.searchsorted(mist_scores, compare_grid, side="right")
    return fog_counts, mist_counts, class_counts


def metrics_from_threshold_counts(
    fog_by_class: np.ndarray,
    mist_by_class: np.ndarray,
    class_counts: np.ndarray,
) -> Dict[str, float]:
    """Reconstruct the existing three-class and Low-vis metrics exactly."""
    fog = np.asarray(fog_by_class, dtype=np.float64)
    mist = np.asarray(mist_by_class, dtype=np.float64)
    total = np.asarray(class_counts, dtype=np.float64)
    eps = 1.0e-6

    fog_tp = fog[0]
    fog_fp = fog[1] + fog[2]
    fog_fn = total[0] - fog_tp
    mist_tp = mist[1]
    mist_fp = mist[0] + mist[2]
    mist_fn = total[1] - mist_tp
    clear_tp = total[2] - fog[2] - mist[2]
    clear_fp = (total[0] - fog[0] - mist[0]) + (total[1] - fog[1] - mist[1])
    clear_fn = fog[2] + mist[2]
    low_tp = fog[0] + fog[1] + mist[0] + mist[1]
    low_fp = fog[2] + mist[2]
    low_fn = total[0] + total[1] - low_tp
    n = float(total.sum())
    return {
        "Fog_P": float(fog_tp / (fog_tp + fog_fp + eps)),
        "Fog_R": float(fog_tp / (fog_tp + fog_fn + eps)),
        "Fog_CSI": float(fog_tp / (fog_tp + fog_fp + fog_fn + eps)),
        "Mist_P": float(mist_tp / (mist_tp + mist_fp + eps)),
        "Mist_R": float(mist_tp / (mist_tp + mist_fn + eps)),
        "Mist_CSI": float(mist_tp / (mist_tp + mist_fp + mist_fn + eps)),
        "Clear_P": float(clear_tp / (clear_tp + clear_fp + eps)),
        "Clear_R": float(clear_tp / (clear_tp + clear_fn + eps)),
        "Clear_CSI": float(clear_tp / (clear_tp + clear_fp + clear_fn + eps)),
        "low_vis_precision": float(low_tp / (low_tp + low_fp + eps)),
        "low_vis_recall": float(low_tp / (low_tp + low_fn + eps)),
        "low_vis_csi": float(low_tp / (low_tp + low_fp + low_fn + eps)),
        "false_positive_rate": float(low_fp / (total[2] + eps)),
        "accuracy": float((fog_tp + mist_tp + clear_tp) / n) if n > 0 else math.nan,
    }
