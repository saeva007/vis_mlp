#!/usr/bin/env python3
"""Regression tests for exact fast validation-threshold search."""

from __future__ import annotations

import argparse
import unittest

import numpy as np

from static_rnn_threshold_search import metrics_from_threshold_counts, threshold_prediction_counts


def class_stats(y_true, pred, cls):
    tp = np.sum((pred == cls) & (y_true == cls))
    fp = np.sum((pred == cls) & (y_true != cls))
    fn = np.sum((pred != cls) & (y_true == cls))
    return tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6), tp / (tp + fp + fn + 1e-6)


def build_metrics(y_true, pred):
    fp, fr, fc = class_stats(y_true, pred, 0)
    mp, mr, mc = class_stats(y_true, pred, 1)
    cp, cr, cc = class_stats(y_true, pred, 2)
    low_pred, low_true, clear = pred <= 1, y_true <= 1, y_true == 2
    return {
        "Fog_P": float(fp), "Fog_R": float(fr), "Fog_CSI": float(fc),
        "Mist_P": float(mp), "Mist_R": float(mr), "Mist_CSI": float(mc),
        "Clear_P": float(cp), "Clear_R": float(cr), "Clear_CSI": float(cc),
        "low_vis_precision": float(np.sum(low_pred & low_true) / (np.sum(low_pred) + 1e-6)),
        "low_vis_recall": float(np.sum(low_pred & low_true) / (np.sum(low_true) + 1e-6)),
        "low_vis_csi": float(np.sum(low_pred & low_true) / (np.sum(low_pred & ~low_true) + np.sum(~low_pred & low_true) + np.sum(low_pred & low_true) + 1e-6)),
        "false_positive_rate": float(np.sum(low_pred & clear) / (np.sum(clear) + 1e-6)),
        "accuracy": float(np.mean(pred == y_true)),
    }


def score_metrics(args, metrics):
    if args.selection_metric == "csi":
        return 0.45 * metrics["Fog_CSI"] + 0.45 * metrics["Mist_CSI"] + 0.10 * metrics["low_vis_precision"] - 0.05 * metrics["false_positive_rate"]
    if args.selection_metric == "recall":
        return 0.45 * metrics["Fog_R"] + 0.45 * metrics["Mist_R"] + 0.10 * metrics["low_vis_precision"] - 0.10 * metrics["false_positive_rate"]
    return 0.25 * metrics["Fog_CSI"] + 0.25 * metrics["Mist_CSI"] + 0.20 * metrics["Fog_R"] + 0.20 * metrics["Mist_R"] + 0.10 * metrics["low_vis_precision"] - 0.05 * metrics["false_positive_rate"]


def pred_from_thresholds(probs, fog_th, mist_th):
    pred = np.full(len(probs), 2, dtype=np.int64)
    fog = (probs[:, 0] > fog_th) & (probs[:, 0] >= probs[:, 1])
    mist = (probs[:, 1] > mist_th) & (probs[:, 1] > probs[:, 0])
    pred[fog], pred[mist] = 0, 1
    return pred


def fast_search(args, probs, y_true):
    grid = np.arange(args.threshold_grid_low, args.threshold_grid_high + 1e-9, args.threshold_grid_step)
    best = (-1e9, {"fog": 0.5, "mist": 0.5}, build_metrics(y_true, np.argmax(probs, axis=1)))
    fog_counts, mist_counts, class_counts = threshold_prediction_counts(probs, y_true, grid)
    tiers = [(args.min_fog_precision, args.min_mist_precision, args.min_clear_recall), (max(0.05, args.min_fog_precision - 0.05), max(0.05, args.min_mist_precision - 0.05), max(0.84, args.min_clear_recall - 0.04))]
    for tier_id, (min_fp, min_mp, min_cr) in enumerate(tiers, start=1):
        found = False
        for fog_idx, fth in enumerate(grid):
            for mist_idx, mth in enumerate(grid):
                metrics = metrics_from_threshold_counts(fog_counts[fog_idx], mist_counts[mist_idx], class_counts)
                if metrics["Fog_P"] >= min_fp and metrics["Mist_P"] >= min_mp and metrics["Clear_R"] >= min_cr:
                    score = score_metrics(args, metrics) - 0.02 * (tier_id - 1)
                    if score > best[0]:
                        best = (score, {"fog": float(fth), "mist": float(mth)}, metrics)
                    found = True
        if found:
            return best
    fallback = build_metrics(y_true, np.argmax(probs, axis=1))
    return score_metrics(args, fallback) - 0.2, {"fog": 0.5, "mist": 0.5}, fallback


def args_for_test(**overrides):
    values = {
        "threshold_grid_low": 0.10,
        "threshold_grid_high": 0.95,
        "threshold_grid_step": 0.03,
        "min_fog_precision": 0.10,
        "min_mist_precision": 0.10,
        "min_clear_recall": 0.88,
        "selection_metric": "recall_csi",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def brute_force_search(args, probs, y_true):
    grid = np.arange(args.threshold_grid_low, args.threshold_grid_high + 1e-9, args.threshold_grid_step)
    best = (-1e9, {"fog": 0.5, "mist": 0.5}, build_metrics(y_true, np.argmax(probs, axis=1)))
    tiers = [
        (args.min_fog_precision, args.min_mist_precision, args.min_clear_recall),
        (
            max(0.05, args.min_fog_precision - 0.05),
            max(0.05, args.min_mist_precision - 0.05),
            max(0.84, args.min_clear_recall - 0.04),
        ),
    ]
    for tier_id, (min_fp, min_mp, min_cr) in enumerate(tiers, start=1):
        found = False
        for fth in grid:
            for mth in grid:
                metrics = build_metrics(y_true, pred_from_thresholds(probs, float(fth), float(mth)))
                if metrics["Fog_P"] >= min_fp and metrics["Mist_P"] >= min_mp and metrics["Clear_R"] >= min_cr:
                    score = score_metrics(args, metrics) - 0.02 * (tier_id - 1)
                    if score > best[0]:
                        best = (score, {"fog": float(fth), "mist": float(mth)}, metrics)
                    found = True
        if found:
            return best
    fallback = build_metrics(y_true, np.argmax(probs, axis=1))
    return score_metrics(args, fallback) - 0.2, {"fog": 0.5, "mist": 0.5}, fallback


class FastThresholdSearchTest(unittest.TestCase):
    def assert_search_equal(self, args, probs, y_true):
        expected = brute_force_search(args, probs, y_true)
        actual = fast_search(args, probs, y_true)
        self.assertAlmostEqual(actual[0], expected[0], places=12)
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(set(actual[2]), set(expected[2]))
        for key in expected[2]:
            self.assertAlmostEqual(actual[2][key], expected[2][key], places=12, msg=key)

    def test_random_probabilities_match_brute_force(self):
        rng = np.random.default_rng(20260703)
        raw = rng.gamma(shape=1.5, scale=1.0, size=(5000, 3)).astype(np.float32)
        probs = raw / raw.sum(axis=1, keepdims=True)
        y_true = rng.integers(0, 3, size=len(probs), dtype=np.int64)
        self.assert_search_equal(args_for_test(), probs, y_true)

    def test_grid_boundary_scores_match_strict_greater_than(self):
        grid = np.arange(0.10, 0.95 + 1e-9, 0.03, dtype=np.float32)
        p0 = np.resize(grid, 600)
        p1 = np.resize(grid[::-1], 600)
        p2 = np.maximum(1.0 - p0 - p1, 0.01)
        probs = np.column_stack([p0, p1, p2]).astype(np.float32)
        probs /= probs.sum(axis=1, keepdims=True)
        y_true = np.arange(len(probs), dtype=np.int64) % 3
        self.assert_search_equal(args_for_test(), probs, y_true)

    def test_fallback_matches_brute_force(self):
        rng = np.random.default_rng(8)
        probs = rng.dirichlet([1.0, 1.0, 1.0], size=800).astype(np.float32)
        y_true = rng.integers(0, 3, size=len(probs), dtype=np.int64)
        self.assert_search_equal(
            args_for_test(min_fog_precision=1.1, min_mist_precision=1.1, min_clear_recall=1.1),
            probs,
            y_true,
        )


if __name__ == "__main__":
    unittest.main()
