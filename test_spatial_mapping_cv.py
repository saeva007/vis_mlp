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

    def test_logistic_smoke_freezes_validation_thresholds_before_test(self):
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
                    threshold_grid_step=0.2,
                )
            )
            result = pd.read_json(output_dir / "result.json", typ="series")
            self.assertEqual(result["threshold_source"], "held-in validation stations and validation times only")
            self.assertEqual(result["test_access_policy"], "loaded after model and thresholds were frozen")
            self.assertTrue((output_dir / "test_predictions.npz").is_file())


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


if __name__ == "__main__":
    unittest.main()
