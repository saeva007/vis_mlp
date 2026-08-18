import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import temporal_mapping_cv as tpcv
from plot_mapping_cv_folds import MODEL_ORDER, plot_mapping_cv


class TemporalFoldContractTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(data_dir: Path) -> None:
        periods = pd.period_range("2025-01", "2025-10", freq="M")
        times = []
        for period in periods:
            times.extend(
                [
                    period.start_time + pd.Timedelta(hours=6),
                    period.start_time + pd.Timedelta(hours=30),
                    period.start_time + pd.Timedelta(days=14),
                    period.end_time.floor("h") - pd.Timedelta(hours=6),
                ]
            )
        frame = pd.DataFrame(
            {
                "time": pd.DatetimeIndex(times),
                "station_id": [f"S{index % 4:03d}" for index in range(len(times))],
                "lat": 30.0,
                "lon": 110.0,
            }
        )
        for split in ("train", "val", "test"):
            np.save(data_dir / f"X_{split}.npy", np.zeros((len(frame), 8), dtype=np.float32))
            np.save(data_dir / f"y_{split}.npy", np.full(len(frame), 2000.0, dtype=np.float32))
            frame.to_csv(data_dir / f"meta_{split}.csv", index=False)

    def test_temporal_prepare_partitions_test_once_and_enforces_embargo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            folds_dir = root / "folds"
            data_dir.mkdir()
            self._write_dataset(data_dir)
            tpcv.prepare_temporal_folds(
                Namespace(
                    data_dir=str(data_dir),
                    output_dir=str(folds_dir),
                    n_folds=5,
                    embargo_hours=24.0,
                    window_hours=12.0,
                    chunksize=7,
                    overwrite=False,
                )
            )

            with (folds_dir / "fold_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["cv_kind"], "temporal")
            self.assertFalse(manifest["label_access_during_fold_construction"])
            self.assertTrue(manifest["test_partition_exactly_once"])
            self.assertEqual([len(manifest["fold_months"][str(fold)]) for fold in range(5)], [2] * 5)

            meta_test = pd.read_csv(data_dir / "meta_test.csv")
            all_test = []
            for fold in range(5):
                months = manifest["fold_months"][str(fold)]
                interval = tpcv._fold_interval(months, 24.0)
                test_indices = np.load(folds_dir / f"fold_{fold}" / "s2_test_indices.npy")
                all_test.append(test_indices)
                selected_test_times = pd.to_datetime(
                    meta_test.iloc[test_indices]["time"], utc=True
                )
                selected_months = selected_test_times.dt.tz_convert(None).dt.to_period("M").astype(str)
                self.assertEqual(set(selected_months), set(months))

                for split in ("train", "val"):
                    meta = pd.read_csv(data_dir / f"meta_{split}.csv")
                    indices = np.load(folds_dir / f"fold_{fold}" / f"s2_{split}_indices.npy")
                    selected = pd.to_datetime(meta.iloc[indices]["time"], utc=True)
                    outside = (selected < interval["embargo_start"]) | (
                        selected >= interval["embargo_end_exclusive"]
                    )
                    self.assertTrue(bool(outside.all()))
            self.assertTrue(
                np.array_equal(
                    np.sort(np.concatenate(all_test)),
                    np.arange(len(meta_test), dtype=np.int64),
                )
            )

    def test_temporal_prepare_rejects_embargo_shorter_than_input_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            self._write_dataset(data_dir)
            with self.assertRaisesRegex(ValueError, "must cover the input window"):
                tpcv.prepare_temporal_folds(
                    Namespace(
                        data_dir=str(data_dir),
                        output_dir=str(root / "folds"),
                        n_folds=5,
                        embargo_hours=6.0,
                        window_hours=12.0,
                        chunksize=10,
                        overwrite=False,
                    )
                )


class MappingCVFigureTests(unittest.TestCase):
    @staticmethod
    def _write_result(root: Path, kind: str) -> None:
        aggregate = root / "aggregate"
        folds = root / "folds"
        aggregate.mkdir(parents=True)
        folds.mkdir(parents=True)
        rows = []
        pooled = []
        for model_index, model in enumerate(MODEL_ORDER):
            for fold in range(5):
                rows.append(
                    {
                        "cv_kind": kind,
                        "model": model,
                        "fold": fold,
                        "low_vis_csi": 0.20 + 0.05 * model_index + 0.01 * fold,
                        "low_vis_recall": 0.40 + 0.06 * model_index + 0.01 * fold,
                    }
                )
            pooled.append(
                {
                    "cv_kind": kind,
                    "model": model,
                    "low_vis_csi": 0.22 + 0.05 * model_index,
                    "low_vis_recall": 0.42 + 0.06 * model_index,
                }
            )
        pd.DataFrame(rows).to_csv(aggregate / "fold_metrics.csv", index=False)
        pd.DataFrame(pooled).to_csv(aggregate / "pooled_metrics.csv", index=False)
        manifest = {"cv_kind": kind, "n_folds": 5}
        if kind == "temporal":
            manifest["fold_months"] = {
                str(fold): [f"2025-{2 * fold + 1:02d}", f"2025-{2 * fold + 2:02d}"]
                for fold in range(5)
            }
        with (folds / "fold_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    def test_plot_exports_figure_and_source_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spatial = root / "spatial"
            temporal = root / "temporal"
            output = root / "figures"
            self._write_result(spatial, "spatial")
            self._write_result(temporal, "temporal")
            written = plot_mapping_cv(
                spatial,
                temporal,
                output,
                "mapping_cv_test",
                ["png"],
                120,
            )
            self.assertEqual(written, [output / "mapping_cv_test.png"])
            self.assertTrue(written[0].is_file())
            self.assertTrue((output / "mapping_cv_test_source_data.csv").is_file())
            self.assertTrue((output / "mapping_cv_test_summary.csv").is_file())
            self.assertTrue((output / "mapping_cv_test_manifest.json").is_file())
            source = pd.read_csv(output / "mapping_cv_test_source_data.csv")
            self.assertEqual(len(source), 2 * 3 * 5)


if __name__ == "__main__":
    unittest.main()
