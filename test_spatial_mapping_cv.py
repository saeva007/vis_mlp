import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import spatial_mapping_cv as spcv
import train_static_rnn_lowvis as trainer


class SpatialFoldContractTests(unittest.TestCase):
    def test_spatial_assignment_is_balanced_and_deterministic(self):
        rng = np.random.default_rng(23)
        station_count = 53
        stations = pd.DataFrame(
            {
                "station_id": [f"S{i:03d}" for i in range(station_count)],
                "lat": np.concatenate(
                    [rng.normal(30.0, 0.3, 41), rng.normal(46.0, 0.3, 12)]
                ),
                "lon": np.concatenate(
                    [rng.normal(115.0, 0.3, 41), rng.normal(83.0, 0.3, 12)]
                ),
            }
        )
        first = spcv._assign_spatial_folds(stations, n_folds=5, seed=20260815)
        second = spcv._assign_spatial_folds(stations, n_folds=5, seed=20260815)
        counts = first.groupby("fold").size().sort_index().to_numpy()
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(int(counts.sum()), station_count)
        self.assertTrue(
            first[["station_id", "fold"]].reset_index(drop=True).equals(
                second[["station_id", "fold"]].reset_index(drop=True)
            )
        )

    def test_prepare_has_no_station_overlap_and_partitions_test_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            fold_dir = root / "folds"
            data_dir.mkdir()
            station_ids = np.asarray([f"S{i:03d}" for i in range(30)])
            lat = np.linspace(18.0, 52.0, len(station_ids))
            lon = 75.0 + (np.arange(len(station_ids)) % 10) * 5.0
            width = 12 * 3 + 6
            for split, repeats in (("train", 3), ("val", 2), ("test", 2)):
                ids = np.tile(station_ids, repeats)
                rows = len(ids)
                np.save(data_dir / f"X_{split}.npy", np.zeros((rows, width), dtype=np.float32))
                np.save(data_dir / f"y_{split}.npy", np.full(rows, 2000.0, dtype=np.float32))
                lookup = {sid: (lat[i], lon[i]) for i, sid in enumerate(station_ids)}
                pd.DataFrame(
                    {
                        "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
                        "station_id": ids,
                        "lat": [lookup[sid][0] for sid in ids],
                        "lon": [lookup[sid][1] for sid in ids],
                    }
                ).to_csv(data_dir / f"meta_{split}.csv", index=False)

            spcv.prepare_folds(
                Namespace(
                    data_dir=str(data_dir),
                    output_dir=str(fold_dir),
                    n_folds=5,
                    seed=17,
                    buffer_km=0.0,
                    min_fold_stations=2,
                    chunksize=17,
                    overwrite=False,
                )
            )
            station_fold_counts = pd.read_csv(fold_dir / "station_folds.csv").groupby("fold").size()
            self.assertLessEqual(
                int(station_fold_counts.max() - station_fold_counts.min()), 1
            )
            all_test = []
            meta_test = pd.read_csv(data_dir / "meta_test.csv", dtype={"station_id": "string"})
            for fold in range(5):
                held = set(pd.read_csv(fold_dir / f"fold_{fold}" / "heldout_stations.csv")["station_id"])
                train = set(pd.read_csv(fold_dir / f"fold_{fold}" / "train_stations.csv")["station_id"])
                self.assertFalse(held & train)
                train_idx = np.load(fold_dir / f"fold_{fold}" / "s2_train_indices.npy")
                test_idx = np.load(fold_dir / f"fold_{fold}" / "s2_test_indices.npy")
                self.assertTrue(set(pd.read_csv(data_dir / "meta_train.csv").iloc[train_idx]["station_id"]) <= train)
                self.assertEqual(set(meta_test.iloc[test_idx]["station_id"]), held)
                all_test.append(test_idx)
            self.assertTrue(np.array_equal(np.sort(np.concatenate(all_test)), np.arange(len(meta_test))))

    def test_logistic_smoke_uses_argmax_before_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            fold_root = root / "folds"
            output_dir = root / "result"
            data_dir.mkdir()
            station_ids = np.asarray([f"S{i:03d}" for i in range(25)])
            coords = {
                sid: (20.0 + i * 1.1, 78.0 + (i % 8) * 5.0)
                for i, sid in enumerate(station_ids)
            }
            rng = np.random.default_rng(5)
            width = 12 * 18 + 6 + 20
            for split, repeats in (("train", 9), ("val", 6), ("test", 6)):
                ids = np.tile(station_ids, repeats)
                rows = len(ids)
                x = rng.normal(size=(rows, width)).astype(np.float32)
                x[:, 12 * 18 + 5] = rng.integers(0, 8, size=rows)
                y = np.resize(np.asarray([300.0, 700.0, 2000.0], dtype=np.float32), rows)
                np.save(data_dir / f"X_{split}.npy", x)
                np.save(data_dir / f"y_{split}.npy", y)
                pd.DataFrame(
                    {
                        "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
                        "station_id": ids,
                        "lat": [coords[sid][0] for sid in ids],
                        "lon": [coords[sid][1] for sid in ids],
                    }
                ).to_csv(data_dir / f"meta_{split}.csv", index=False)
            spcv.prepare_folds(
                Namespace(
                    data_dir=str(data_dir), output_dir=str(fold_root), n_folds=5,
                    seed=19, buffer_km=0.0, min_fold_stations=2,
                    chunksize=31, overwrite=False,
                )
            )
            spcv.train_logistic(
                Namespace(
                    data_dir=str(data_dir), fold_dir=str(fold_root / "fold_0"),
                    output_dir=str(output_dir), fold=0, window_size=12, seed=19,
                    alpha=1e-4, max_epochs=2, patience=2, min_delta=0.0,
                    batch_rows=32, scaler_sample_rows=1000, fog_ratio=0.18,
                    mist_ratio=0.22, min_fog_precision=0.0,
                    min_mist_precision=0.0, min_clear_recall=0.0,
                    threshold_grid_low=0.1, threshold_grid_high=0.7,
                    threshold_grid_step=0.2, decision_rule="argmax",
                )
            )
            result = pd.read_json(output_dir / "result.json", typ="series")
            self.assertEqual(result["analysis_decision_rule"], "argmax")
            self.assertEqual(result["checkpoint_selection_rule"], "argmax")
            self.assertEqual(result["thresholds"], {"mode": "argmax"})
            self.assertEqual(
                result["test_access_policy"],
                "loaded after model and validation decision rule were frozen",
            )
            self.assertTrue((output_dir / "test_predictions.npz").is_file())
            with np.load(output_dir / "test_predictions.npz") as payload:
                self.assertTrue(
                    np.array_equal(payload["pred"], np.argmax(payload["probs"], axis=1))
                )


class InstantaneousMLPTests(unittest.TestCase):
    def test_mlp_is_invariant_to_preceding_timesteps(self):
        torch.manual_seed(3)
        layout = trainer.Layout(window_size=12, dyn_vars=4, fe_dim=0)
        model = trainer.StaticRNNLowVisNet(
            layout=layout,
            encoder="mlp",
            hidden_dim=8,
            static_hidden_dim=6,
            fe_hidden_dim=4,
            fusion_hidden_dim=10,
            veg_emb_dim=3,
            rnn_layers=1,
            dropout=0.0,
            bidirectional=False,
            pooling="mean",
            use_fe=False,
        ).eval()
        first = torch.zeros((2, layout.total_expected_dim), dtype=torch.float32)
        second = first.clone()
        second[:, : (layout.window_size - 1) * layout.dyn_vars] = 9.0
        with torch.no_grad():
            logits_a, reg_a = model(first)
            logits_b, reg_b = model(second)
        self.assertTrue(torch.allclose(logits_a, logits_b))
        self.assertTrue(torch.allclose(reg_a, reg_b))


class ArgmaxAggregationTests(unittest.TestCase):
    def test_model_parser_accepts_gru_only_and_rejects_duplicates(self):
        self.assertEqual(spcv._parse_models("gru"), ["gru"])
        with self.assertRaises(ValueError):
            spcv._parse_models("gru,gru")

    def test_ifs_baseline_preflight_verifies_frozen_test_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            times = pd.date_range("2025-01-01", periods=3, freq="h")
            pd.DataFrame(
                {"station_id": ["S0", "S1", "S2"], "time": times}
            ).to_csv(data_dir / "meta_test.csv", index=False)
            np.save(
                data_dir / "y_test.npy",
                np.asarray([300.0, 700.0, 2000.0], dtype=np.float32),
            )
            ifs_path = root / "ifs.csv"
            pd.DataFrame(
                {
                    "station_id": ["S2", "S0", "S1"],
                    "time": times[[2, 0, 1]],
                    "y_true": [2, 0, 1],
                    "ifs_diagnostic_vis_m": [1400.0, 350.0, 800.0],
                    "ifs_diagnostic_pred": [2, 0, 1],
                    "ifs_diagnostic_valid": [True, True, True],
                }
            ).to_csv(ifs_path, index=False)
            output_json = root / "compatibility.json"
            spcv.validate_ifs_baseline(
                Namespace(
                    data_dir=str(data_dir),
                    ifs_csv=str(ifs_path),
                    output_json=str(output_json),
                )
            )
            with output_json.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "compatible")
            self.assertEqual(payload["ifs_baseline"]["aligned_rows"], 3)
            self.assertTrue(payload["ifs_baseline"]["label_match_verified"])

    def test_aggregate_recomputes_argmax_and_reports_saved_rule_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            folds_dir = root / "folds"
            results_dir = root / "results"
            output_dir = root / "aggregate"
            data_dir.mkdir()
            folds_dir.mkdir()
            pd.DataFrame(
                {
                    "station_id": ["S0", "S1", "S2", "S3"],
                    "time": pd.date_range("2025-01-01", periods=4, freq="h"),
                }
            ).to_csv(data_dir / "meta_test.csv", index=False)
            with (folds_dir / "fold_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cv_kind": "spatial",
                        "algorithm": "synthetic",
                        "n_folds": 2,
                        "data_dir": str(data_dir),
                        "data_shapes": {"test": {"x_shape": [4, 3]}},
                    },
                    handle,
                )

            y_true = np.asarray([0, 2, 1, 2], dtype=np.int8)
            probs = np.asarray(
                [
                    [0.40, 0.35, 0.25],
                    [0.30, 0.20, 0.50],
                    [0.20, 0.45, 0.35],
                    [0.10, 0.20, 0.70],
                ],
                dtype=np.float32,
            )
            stored_pred = np.asarray([2, 0, 2, 2], dtype=np.int8)
            fold_rows = (np.asarray([0, 1]), np.asarray([2, 3]))
            for fold, rows in enumerate(fold_rows):
                fold_dir = folds_dir / f"fold_{fold}"
                fold_dir.mkdir()
                np.save(fold_dir / "s2_test_indices.npy", rows)
            for model in spcv.MODELS:
                for fold, rows in enumerate(fold_rows):
                    fold_output = results_dir / model / f"fold_{fold}"
                    fold_output.mkdir(parents=True)
                    with (fold_output / "result.json").open("w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "model": model,
                                "fold": fold,
                                "checkpoint_selection_rule": "val_search",
                                "thresholds": {"fog": 0.2, "mist": 0.2},
                            },
                            handle,
                        )
                    np.savez_compressed(
                        fold_output / "test_predictions.npz",
                        row_index=rows,
                        y_true=y_true[rows],
                        probs=probs[rows],
                        pred=stored_pred[rows],
                    )

            ifs_path = root / "per_sample_eval.csv"
            baseline_order = np.asarray([2, 0, 3, 1])
            pd.DataFrame(
                {
                    "station_id": np.asarray(["S0", "S1", "S2", "S3"])[baseline_order],
                    "time": pd.date_range("2025-01-01", periods=4, freq="h")[baseline_order],
                    "y_true": y_true[baseline_order],
                    "ifs_diagnostic_vis_m": np.asarray(
                        [300.0, 1500.0, 700.0, 1200.0], dtype=np.float64
                    )[baseline_order],
                    "ifs_diagnostic_pred": np.asarray([0, 2, 1, 2], dtype=np.int8)[baseline_order],
                    "ifs_diagnostic_valid": np.asarray([True, False, True, True])[baseline_order],
                }
            ).to_csv(ifs_path, index=False)

            spcv.aggregate_results(
                Namespace(
                    folds_dir=str(folds_dir),
                    results_dir=str(results_dir),
                    output_dir=str(output_dir),
                    models=",".join(spcv.MODELS),
                    ifs_csv=str(ifs_path),
                    decision_rule="argmax",
                )
            )

            pooled = pd.read_csv(output_dir / "pooled_metrics.csv")
            self.assertEqual(set(pooled["model"]), {*spcv.MODELS, "ifs_native"})
            self.assertEqual(set(pooled["sample_scope"]), {"ifs_diagnostic_matched_test"})
            self.assertEqual(
                set(pooled.loc[pooled["model"] != "ifs_native", "decision_rule"]),
                {"argmax"},
            )
            valid_rows = np.asarray([0, 2, 3])
            expected = trainer.build_metrics(
                y_true[valid_rows], np.argmax(probs[valid_rows], axis=1)
            )
            logistic = pooled.loc[pooled["model"] == "logistic"].iloc[0]
            self.assertAlmostEqual(float(logistic["low_vis_csi"]), expected["low_vis_csi"])
            self.assertTrue((output_dir / "fold_metrics_full_learned_test.csv").is_file())
            self.assertTrue((output_dir / "pooled_metrics_full_learned_test.csv").is_file())

            effects = pd.read_csv(output_dir / "decision_rule_effects.csv")
            pooled_csi = effects.loc[
                (effects["scope"] == "pooled")
                & (effects["model"] == "logistic")
                & (effects["metric"] == "low_vis_csi")
            ].iloc[0]
            self.assertNotEqual(float(pooled_csi["argmax_minus_saved"]), 0.0)

            with (output_dir / "coverage_manifest.json").open("r", encoding="utf-8") as handle:
                coverage = json.load(handle)
            self.assertEqual(coverage["analysis_decision_rule"], "argmax")
            self.assertFalse(coverage["all_checkpoints_selected_with_argmax"])
            self.assertEqual(coverage["primary_sample_scope"], "ifs_diagnostic_matched_test")
            self.assertTrue(coverage["ifs_baseline"]["included"])
            self.assertEqual(coverage["ifs_baseline"]["valid_matched_rows"], 3)
            self.assertTrue(
                coverage["ifs_baseline"]["source_covers_frozen_test_exactly_once"]
            )


if __name__ == "__main__":
    unittest.main()
