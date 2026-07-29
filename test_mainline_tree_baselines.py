import argparse
import unittest

import numpy as np

import mainline_tree_baselines as tree


class MainlineTreeContractTests(unittest.TestCase):
    def test_visibility_labels(self):
        raw, labels = tree.visibility_to_labels(np.array([100.0, 499.9, 500.0, 999.9, 1000.0, 30000.0]))
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(raw[0], 100.0)

    def test_train_only_transform_and_one_hot(self):
        layout = tree.FlatLayout(
            window_size=2,
            dyn_vars=3,
            fe_dim=2,
            dynamic_feature_order=("RH2M", "PRECIP", "PM2P5"),
        )
        width = layout.expected_width
        values = np.zeros((3, width), dtype=np.float32)
        values[:, 1] = [0.0, 3.0, np.nan]
        values[:, 2] = [1.0, 4.0, 9.0]
        values[:, 4] = [8.0, 15.0, 24.0]
        values[:, 5] = [1.0, 2.0, 3.0]
        values[:, layout.vegetation_index] = [2, 5, 99]
        medians = np.zeros(width, dtype=np.float32)
        medians[1] = np.log1p(3.0)
        result = tree.prepare_feature_chunk(values, layout, medians, [2, 5])
        self.assertEqual(result.shape[1], width - 1 + 3)
        self.assertAlmostEqual(float(result[1, 1]), float(np.log1p(3.0)), places=6)
        self.assertAlmostEqual(float(result[2, 1]), float(np.log1p(3.0)), places=6)
        veg_start = layout.vegetation_index
        np.testing.assert_array_equal(result[:, veg_start : veg_start + 3], np.eye(3, dtype=np.float32))

    def test_prior_correction_is_normalized(self):
        raw = np.array([[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]], dtype=np.float32)
        corrected = tree.prior_correct_probabilities(raw, np.array([2.0, 2.0, 0.8]))
        np.testing.assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-7)
        self.assertGreater(corrected[0, 2], raw[0, 2])

    def test_metrics_toy_case(self):
        labels = np.array([0, 1, 2, 2])
        probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.1, 0.8],
                [0.6, 0.1, 0.3],
            ]
        )
        metrics = tree.metrics_from_probabilities(labels, probs)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["Fog_recall"], 1.0)
        self.assertAlmostEqual(metrics["Fog_precision"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)

    def test_layout_and_feature_names(self):
        layout = tree.FlatLayout(
            window_size=2,
            dyn_vars=3,
            fe_dim=2,
            dynamic_feature_order=("RH2M", "PRECIP", "PM2P5"),
        )
        self.assertEqual(layout.vegetation_index, 11)
        self.assertEqual(layout.expected_width, 14)
        names = tree.build_feature_names(layout, [2, 5, 9])
        self.assertEqual(len(names), layout.expected_width - 1 + 4)
        self.assertEqual(names[layout.vegetation_index : layout.vegetation_index + 4], [
            "veg_2",
            "veg_5",
            "veg_9",
            "veg_unknown",
        ])

    def test_parameter_presets_are_regularized(self):
        args = argparse.Namespace(rf_trees=400)
        rf = tree.literature_informed_parameters("rf", 8, 1, args)
        xgb = tree.literature_informed_parameters("xgboost", 8, 1, args)
        lgb = tree.literature_informed_parameters("lightgbm", 8, 1, args)
        self.assertGreaterEqual(rf["min_samples_leaf"], 20)
        self.assertLess(xgb["subsample"], 1.0)
        self.assertGreater(xgb["min_child_weight"], 1.0)
        self.assertGreaterEqual(lgb["min_data_in_leaf"], 100)
        self.assertLessEqual(lgb["num_leaves"], 2 ** lgb["max_depth"])


if __name__ == "__main__":
    unittest.main()
