#!/usr/bin/env python3

import json
import shutil
import sys
import types
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import audit_lowvis_data_quality as audit


@contextmanager
def workspace_tempdir():
    # tempfile.TemporaryDirectory(mode=0700) is not writable under the Codex
    # Windows sandbox ACL, while a normal workspace directory is.
    path = Path(__file__).resolve().parent / f"_audit_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class LowVisDataQualityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The audit uses only the Torch-free canonical PM helpers.  pvlib is a
        # cluster data-runtime dependency of pmst_overlap_common, but is not
        # needed by these unit tests.
        sys.modules.setdefault("pvlib", types.ModuleType("pvlib"))
        cls.policy = audit.load_pm_policy(Path(__file__).resolve().parents[1] / "ifs_baseline")

    def test_latest_pm_policy_repairs_legacy_and_raw_kgm3(self):
        legacy = self.policy.canonicalize_pm_concentration(
            np.asarray([50000.0], dtype=np.float32), self.policy.LEGACY_PM_1E12_UNITS
        )
        raw = self.policy.canonicalize_pm_concentration(
            np.asarray([1.0e-7], dtype=np.float32), "kg m-3"
        )
        self.assertAlmostEqual(float(legacy[0]), 50.0, places=5)
        self.assertAlmostEqual(float(raw[0]), 100.0, places=4)

    def test_flat_audit_reports_legacy_pm_and_above_30km_label(self):
        with workspace_tempdir() as root:
            order = ["T2M", "MSLP", "RH2M", "PM10", "PM2P5"]
            config = {
                "window_size": 2,
                "dyn_vars": len(order),
                "dynamic_feature_order": order,
                "include_pm": True,
                "protocol": "main_pm10_pm25",
            }
            (root / "dataset_build_config.json").write_text(json.dumps(config), encoding="utf-8")
            dyn = np.zeros((4, 2, len(order)), dtype=np.float32)
            dyn[..., 0] = 280.0
            dyn[..., 1] = 101000.0
            dyn[..., 2] = 80.0
            dyn[..., 3] = 50000.0
            dyn[..., 4] = 30000.0
            x = np.concatenate([dyn.reshape(4, -1), np.zeros((4, 6), dtype=np.float32)], axis=1)
            np.save(root / "X_train.npy", x)
            np.save(root / "y_train.npy", np.asarray([100.0, 700.0, 1200.0, 35000.0], dtype=np.float32))
            pd.DataFrame(
                {
                    "time": pd.date_range("2025-01-01", periods=4, freq="h"),
                    "station_id": ["A", "B", "C", "D"],
                }
            ).to_csv(root / "meta_train.csv", index=False)
            state = audit.AuditState()
            audit.audit_flat_dataset(state, "main_s1", root, self.policy, 2, 4, 2)
            self.assertEqual(state.rows["visibility_quality"][0]["above_30km_values"], 1)
            pm_rows = state.rows["pm_quality"]
            self.assertEqual(len(pm_rows), 2)
            self.assertEqual(pm_rows[0]["pm_lineage"], "explicit_historical_builder_lineage")
            self.assertAlmostEqual(pm_rows[0]["canonical_p50"], 50.0, places=5)

    def test_trajectory_audit_detects_zero_coverage_lead(self):
        with workspace_tempdir() as root:
            order = ["T2M", "PM10_ugm3", "PM25_ugm3"]
            config = {
                "dynamic_feature_order": order,
                "condition_length": 48,
                "target_leads": list(range(1, 49)),
                "canonical_unit_policy": audit.EXPECTED_PM_UNIT_POLICY,
                "pm_qc_policy": audit.EXPECTED_PM_QC_POLICY,
            }
            (root / "dataset_build_config.json").write_text(json.dumps(config), encoding="utf-8")
            dynamic = np.zeros((3, 48, len(order)), dtype=np.float32)
            dynamic[..., 0] = 280.0
            dynamic[..., 1] = 50.0
            dynamic[..., 2] = 25.0
            visibility = np.full((3, 48), 5000.0, dtype=np.float32)
            mask = np.ones((3, 48), dtype=bool)
            mask[:, 20] = False
            visibility[:, 20] = np.nan
            np.save(root / "dynamic_train.npy", dynamic)
            np.save(root / "visibility_train.npy", visibility)
            np.save(root / "target_mask_train.npy", mask)
            pd.DataFrame(
                {
                    "init_time": pd.date_range("2025-01-01", periods=3, freq="12h"),
                    "station_id": ["A", "B", "C"],
                }
            ).to_csv(root / "meta_train.csv", index=False)
            state = audit.AuditState()
            audit.audit_trajectory_dataset(state, "trajectory", root, self.policy, 2, 3, 2)
            lead21 = [row for row in state.rows["trajectory_coverage_by_lead"] if row["lead_hour"] == 21][0]
            self.assertEqual(lead21["stored_valid"], 0)
            self.assertTrue(any(item["code"] == "trajectory_lead_zero_coverage" for item in state.issues))

    def test_raw_stored_consistency_accepts_mixed_init_time_strings(self):
        with workspace_tempdir() as root:
            trajectory = root / "trajectory"
            trajectory.mkdir()
            times = pd.date_range("2025-01-01 01:00:00", periods=48, freq="h")
            raw_values = np.full((48, 2), 5000.0, dtype=np.float32)
            raw_path = root / "raw_visibility.nc"
            xr.Dataset(
                {"visibility": (("time", "station_id"), raw_values)},
                coords={"time": times, "station_id": ["A", "B"]},
            ).to_netcdf(raw_path, engine="scipy")
            stored = raw_values.T.copy()
            np.save(trajectory / "visibility_test.npy", stored)
            np.save(trajectory / "target_mask_test.npy", np.ones_like(stored, dtype=bool))
            np.save(trajectory / "dynamic_test.npy", np.zeros((2, 48, 1), dtype=np.float32))
            pd.DataFrame(
                {
                    "init_time": ["2025-01-01", "2025-01-01 00:00:00"],
                    "station_id": ["A", "B"],
                }
            ).to_csv(trajectory / "meta_test.csv", index=False)
            state = audit.AuditState()
            audit.audit_trajectory_raw_consistency(
                state, "trajectory", trajectory, "raw", raw_path, metadata_chunksize=1, max_rows=0
            )
            self.assertEqual(len(state.rows["trajectory_raw_consistency"]), 48)
            self.assertTrue(all(row["mask_mismatch"] == 0 for row in state.rows["trajectory_raw_consistency"]))
            self.assertTrue(all(row["value_mismatch"] == 0 for row in state.rows["trajectory_raw_consistency"]))

    def test_slurm_launcher_avoids_empty_array_expansion_under_nounset(self):
        launcher = (Path(__file__).resolve().parent / "sub_audit_lowvis_data_quality.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("command=(", launcher)
        self.assertIn('"${command[@]}"', launcher)
        self.assertNotIn("extra_args=()", launcher)
        self.assertNotIn('"${extra_args[@]}"', launcher)


if __name__ == "__main__":
    unittest.main()
