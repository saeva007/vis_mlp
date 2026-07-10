#!/usr/bin/env python3
"""Unit tests for the optional P11/P12 precision objective."""

from __future__ import annotations

import argparse
import sys
import types
import unittest

import numpy as np
import torch

try:
    import joblib  # noqa: F401
except ModuleNotFoundError:
    joblib = types.ModuleType("joblib")
    joblib.dump = lambda *args, **kwargs: None
    joblib.load = lambda *args, **kwargs: None
    sys.modules["joblib"] = joblib

try:
    from sklearn.preprocessing import RobustScaler  # noqa: F401
except ModuleNotFoundError:
    sklearn = types.ModuleType("sklearn")
    preprocessing = types.ModuleType("sklearn.preprocessing")
    preprocessing.RobustScaler = type("RobustScaler", (), {})
    sklearn.preprocessing = preprocessing
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.preprocessing"] = preprocessing

from train_static_rnn_lowvis import (  # noqa: E402
    EventTimeBatchSampler,
    FootprintDualState,
    build_time_group_index,
    class_prior_correction_weights,
    event_footprint_loss,
    event_group_metrics,
)


def footprint_args(**overrides) -> argparse.Namespace:
    values = {
        "event_footprint_smoothmax_temperature": 0.10,
        "event_footprint_decision_temperature": 0.15,
        "event_footprint_min_fog_count": 1,
        "event_footprint_area_ratio_cap": 1.5,
        "event_footprint_area_slack": 0.005,
        "event_footprint_min_recall": 0.50,
        "event_footprint_csi_weight": 0.50,
        "event_footprint_dual_rho": 5.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PriorCorrectionTest(unittest.TestCase):
    def test_beta_zero_is_exact_identity(self):
        labels = np.array([0] * 2 + [1] * 3 + [2] * 95, dtype=np.int64)
        actual = class_prior_correction_weights(labels, np.array([0.18, 0.22, 0.60]), 0.0)
        np.testing.assert_array_equal(actual, np.ones(3, dtype=np.float32))

    def test_tempered_prior_matches_definition(self):
        labels = np.array([0] * 2 + [1] * 3 + [2] * 95, dtype=np.int64)
        sampler = np.array([0.18, 0.22, 0.60], dtype=np.float64)
        expected = np.sqrt(np.array([0.02, 0.03, 0.95]) / sampler)
        actual = class_prior_correction_weights(labels, sampler, 0.5)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


class EventGroupTest(unittest.TestCase):
    def test_group_index_counts_classes(self):
        groups = np.array([20, 10, 20, 10, 20], dtype=np.int64)
        labels = np.array([1, 0, 2, 2, 0], dtype=np.int64)
        index = build_time_group_index(groups, labels)
        np.testing.assert_array_equal(index.group_values, np.array([10, 20]))
        np.testing.assert_array_equal(index.counts, np.array([2, 3]))
        np.testing.assert_array_equal(index.fog_counts, np.array([1, 1]))
        np.testing.assert_array_equal(index.low_vis_counts, np.array([1, 2]))

    def test_event_sampler_keeps_one_time_group_per_batch(self):
        groups = np.array([10, 10, 20, 20, 20], dtype=np.int64)
        labels = np.array([0, 2, 1, 2, 0], dtype=np.int64)
        index = build_time_group_index(groups, labels)
        sampler = EventTimeBatchSampler(
            index,
            batch_size=6,
            min_fog_count=1,
            event_batch_ratio=1.0,
            seed=9,
            epoch_length=4,
        )
        for rows in sampler:
            self.assertEqual(len(np.unique(groups[np.asarray(rows)])), 1)
            self.assertTrue(sampler.is_event_group(int(groups[rows[0]])))

    def test_event_metrics_report_area_and_recall(self):
        labels = np.array([0, 1, 2, 2, 0, 1, 2, 2], dtype=np.int64)
        pred = np.array([0, 2, 1, 2, 0, 1, 1, 2], dtype=np.int64)
        groups = np.array([10] * 4 + [20] * 4, dtype=np.int64)
        metrics = event_group_metrics(labels, pred, groups, min_fog_count=1)
        self.assertEqual(metrics["event_group_count"], 2.0)
        self.assertAlmostEqual(metrics["event_low_vis_recall_mean"], 0.75)
        self.assertAlmostEqual(metrics["event_low_vis_area_ratio_mean"], 1.25)


class EventFootprintLossTest(unittest.TestCase):
    def test_overforecast_has_larger_area_violation(self):
        labels = torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.long)
        precise = torch.tensor(
            [[7.0, 0.0, -4.0], [0.0, 7.0, -4.0]] + [[-4.0, -4.0, 7.0]] * 4,
            requires_grad=True,
        )
        over = torch.tensor([[7.0, 0.0, -4.0]] * 6, requires_grad=True)
        state = FootprintDualState(area=1.0, recall=1.0)
        _, precise_parts = event_footprint_loss(
            footprint_args(), precise, labels, state, is_widespread_event=True
        )
        over_loss, over_parts = event_footprint_loss(
            footprint_args(), over, labels, state, is_widespread_event=True
        )
        self.assertGreater(float(over_parts["area_violation"]), float(precise_parts["area_violation"]))
        over_loss.backward()
        self.assertTrue(torch.isfinite(over.grad).all())

    def test_widespread_miss_activates_recall_constraint(self):
        labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        all_clear = torch.tensor([[-5.0, -5.0, 5.0]] * 4)
        _, parts = event_footprint_loss(
            footprint_args(),
            all_clear,
            labels,
            FootprintDualState(area=1.0, recall=1.0),
            is_widespread_event=True,
        )
        self.assertGreater(float(parts["recall_violation"]), 0.0)


if __name__ == "__main__":
    unittest.main()
