#!/usr/bin/env python3

import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

import lowvis_trajectory_diffusion as common
from lowvis_trajectory_contract import (
    COMPARISON_TARGET_POSITIONS,
    TARGET_CONDITION_POSITIONS,
    shifted_lead_indices,
    visibility_grid,
)


class TrajectoryContractTests(unittest.TestCase):
    def test_torch_free_builder_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import lowvis_trajectory_contract; "
                "assert 'torch' not in sys.modules; print('ok')",
            ],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_feature_and_lead_contract(self):
        self.assertEqual(len(common.DYNAMIC_FEATURE_ORDER), 27)
        self.assertEqual(common.CONDITION_LEADS, tuple(range(1, 49)))
        self.assertEqual(common.TARGET_LEADS, tuple(range(1, 49)))
        self.assertEqual(TARGET_CONDITION_POSITIONS, tuple(range(48)))
        self.assertEqual(COMPARISON_TARGET_POSITIONS, tuple(range(11, 48)))

    def test_exact_leads_reject_gap(self):
        leads = np.arange(1, 49, dtype=np.float32)
        self.assertTrue(np.array_equal(common.exact_lead_indices(leads), np.arange(48)))
        self.assertIsNone(common.exact_lead_indices(np.delete(leads, 17)))

    def test_lead_shift_auto_resolves_bjt_valid_times(self):
        indices, shift = shifted_lead_indices(np.arange(9, 57, dtype=np.float32), "auto")
        np.testing.assert_array_equal(indices, np.arange(48))
        self.assertEqual(shift, -8.0)
        missing, missing_shift = shifted_lead_indices(np.arange(2, 50, dtype=np.float32), 0)
        self.assertIsNone(missing)
        self.assertIsNone(missing_shift)

    def test_visibility_grid_normalizes_dimension_order_and_station_type(self):
        times = pd.date_range("2025-01-01T01:00", periods=common.TARGET_LENGTH, freq="h")
        values = np.vstack(
            [
                np.arange(common.TARGET_LENGTH, dtype=np.float32),
                np.arange(common.TARGET_LENGTH, dtype=np.float32) + 100.0,
            ]
        )
        source = xr.DataArray(
            values,
            dims=("station_id", "time"),
            coords={"station_id": ["54527.0", "A001"], "time": times},
        )
        aligned, diagnostics = visibility_grid(
            source,
            times,
            np.asarray([54527, "A001"], dtype=object),
            tolerance_minutes=31.0,
        )
        self.assertEqual(aligned.shape, (2, common.TARGET_LENGTH))
        np.testing.assert_array_equal(aligned, values)
        self.assertEqual(diagnostics["matched_target_times"], common.TARGET_LENGTH)
        self.assertEqual(diagnostics["matched_stations"], 2)

    def test_full_trajectory_split_containment(self):
        # January test is the final three days; this trajectory stays inside it.
        inside = np.array("2025-01-29T00", dtype="datetime64[h]") + np.arange(1, 49).astype("timedelta64[h]")
        self.assertEqual(common.full_trajectory_split(inside), "test")
        crossing = np.array("2025-01-27T00", dtype="datetime64[h]") + np.arange(1, 49).astype("timedelta64[h]")
        self.assertIsNone(common.full_trajectory_split(crossing))

    def test_scaler_round_trip_and_train_only_fill(self):
        rng = np.random.default_rng(4)
        dynamic = rng.normal(size=(8, len(common.CONDITION_LEADS), 27)).astype(np.float32)
        dynamic[0, 0, 0] = np.nan
        static = rng.normal(size=(8, 5)).astype(np.float32)
        visibility = rng.uniform(100.0, 5000.0, size=(8, common.TARGET_LENGTH)).astype(np.float32)
        mask = np.ones((8, common.TARGET_LENGTH), dtype=bool)
        scaler = common.TrajectoryScaler.fit_arrays(dynamic, static, visibility, mask, max_rows=8)
        encoded = scaler.transform_target(visibility, mask)
        decoded = scaler.inverse_target(encoded)
        np.testing.assert_allclose(decoded, visibility, rtol=2e-5, atol=2e-3)
        transformed, valid = scaler.transform_dynamic(dynamic[0])
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(valid[0, 0], 0.0)

    def test_diffusion_forward_sampling_and_probabilities(self):
        model = common.ConditionalTrajectoryDenoiser(
            d_model=32, nhead=4, condition_layers=1, denoiser_layers=1, dropout=0.0
        ).eval()
        schedule = common.DiffusionSchedule(10)
        batch = {
            "condition": torch.randn(2, len(common.CONDITION_LEADS), 54),
            "static": torch.randn(2, 5),
            "veg": torch.tensor([1, 2]),
            "time_features": torch.randn(2, 4),
            "target": torch.randn(2, common.TARGET_LENGTH),
            "target_mask": torch.ones(2, common.TARGET_LENGTH),
        }
        step = torch.tensor([2, 5])
        noisy, noise = schedule.q_sample(batch["target"], step)
        predicted = model(noisy, step, batch["condition"], batch["static"], batch["veg"], batch["time_features"])
        self.assertEqual(tuple(predicted.shape), (2, common.TARGET_LENGTH))
        self.assertTrue(torch.isfinite(common.masked_diffusion_loss(predicted, noise, batch["target_mask"])))
        samples = common.ddim_sample(model, schedule, batch, members=2, steps=2)
        self.assertEqual(tuple(samples.shape), (2, 2, common.TARGET_LENGTH))
        probs = common.visibility_class_probabilities(np.abs(samples.numpy()) * 1200.0)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    def test_condition_tokens_min_snr_and_fixed_sampling(self):
        model = common.ConditionalTrajectoryDenoiser(
            d_model=32,
            nhead=4,
            condition_layers=1,
            denoiser_layers=1,
            dropout=0.0,
            condition_token_version=2,
        ).eval()
        batch = {
            "condition": torch.randn(2, len(common.CONDITION_LEADS), 54),
            "static": torch.randn(2, 5),
            "veg": torch.tensor([1, 2]),
            "time_features": torch.randn(2, 4),
            "target": torch.randn(2, common.TARGET_LENGTH),
            "target_mask": torch.ones(2, common.TARGET_LENGTH),
        }
        memory, context = model.condition_encoder(
            batch["condition"], batch["static"], batch["veg"], batch["time_features"]
        )
        self.assertEqual(tuple(memory.shape), (2, len(common.CONDITION_LEADS) + 4, 32))
        self.assertEqual(tuple(context.shape), (2, 32))

        schedule = common.DiffusionSchedule(10, ddim_clip_x0=6.0)
        weights = common.min_snr_weights(schedule, torch.tensor([0, 5, 9]), gamma=5.0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(bool(torch.all((weights > 0) & (weights <= 1))))

        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)
        first = common.ddim_sample(
            model, schedule, batch, members=2, steps=2, generator=first_generator
        )
        second = common.ddim_sample(
            model, schedule, batch, members=2, steps=2, generator=second_generator
        )
        torch.testing.assert_close(first, second)

        legacy = common.model_from_config(
            {
                "model_type": "diffusion",
                "d_model": 32,
                "nhead": 4,
                "condition_layers": 1,
                "decoder_layers": 1,
                "dropout": 0.0,
            }
        )
        self.assertEqual(legacy.condition_encoder.condition_token_version, 1)

    def test_gaussian_baseline_shape_and_loss(self):
        model = common.GaussianTrajectoryModel(
            d_model=32, nhead=4, condition_layers=1, decoder_layers=1, dropout=0.0
        ).eval()
        batch = {
            "condition": torch.randn(2, len(common.CONDITION_LEADS), 54),
            "static": torch.randn(2, 5),
            "veg": torch.tensor([1, 2]),
            "time_features": torch.randn(2, 4),
            "target": torch.randn(2, common.TARGET_LENGTH),
            "target_mask": torch.ones(2, common.TARGET_LENGTH),
        }
        mean, log_scale = model(batch["condition"], batch["static"], batch["veg"], batch["time_features"])
        self.assertEqual(tuple(mean.shape), (2, common.TARGET_LENGTH))
        self.assertTrue(torch.isfinite(common.masked_gaussian_nll(mean, log_scale, batch["target"], batch["target_mask"])))
        self.assertEqual(
            tuple(common.gaussian_sample(model, batch, members=3).shape),
            (2, 3, common.TARGET_LENGTH),
        )

    def test_canonical_pm_policy(self):
        policy_dir = Path(__file__).resolve().parents[1] / "ifs_baseline"
        sys.path.insert(0, str(policy_dir))
        sys.modules.setdefault("pvlib", types.ModuleType("pvlib"))
        import pmst_overlap_common as policy

        raw_kgm3 = np.array([1.0e-8, 2.0e-8], dtype=np.float32)
        ugm3 = policy.canonicalize_pm_concentration(raw_kgm3, "kg m-3")
        np.testing.assert_allclose(ugm3, [10.0, 20.0], rtol=1e-6)
        self.assertEqual(policy.CANONICAL_UNIT_POLICY_VERSION, common.PM_UNIT_POLICY_VERSION)
        self.assertEqual(policy.PM_QC_POLICY_VERSION, common.PM_QC_POLICY_VERSION)


if __name__ == "__main__":
    unittest.main()
