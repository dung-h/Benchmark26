"""Group-balanced and pooled metrics, the R^2 policy, bootstrap contrasts and epoch selection."""

import unittest

import numpy as np
from analysis_core import aligned, bootstrap_draws, group_errors, metrics, paired_intervals
from validate_completed import earliest_minimum


class AnalysisTests(unittest.TestCase):
    def test_group_weighting_and_not_pooled(self):
        m = metrics([1, 1, 1, 1], [1, 1, 1, 5], ["a", "a", "a", "b"], ["x"] * 4)
        self.assertEqual(m["pooled_mae_seconds"], 1)
        self.assertEqual(m["group_balanced_mae_seconds"], 2)
        self.assertIsNone(m["raw_seconds_r2"])

    def test_raw_r2_and_log_policy(self):
        m = metrics([1, 2, 3], [1, 2, 4], ["a", "b", "c"], ["x"] * 3)
        self.assertAlmostEqual(m["raw_seconds_r2"], 0.5)
        m = metrics([1, 2, 3], [-0.5, 2, 4], ["a", "b", "c"], ["x"] * 3)
        self.assertIsNone(m["log1p_seconds_r2"])
        self.assertEqual(m["negative_predictions"], 1)
        self.assertEqual(m["rows"], 3)

    def test_alignment_rejects_missing_duplicate_foreign(self):
        with self.assertRaises(ValueError):
            aligned([{"canonical_observation_id": "a"}] * 2, ["a", "b"])

    def test_whole_group_strata_and_determinism(self):
        d = bootstrap_draws(["x", "x", "y"], True)
        self.assertTrue(np.array_equal(d, bootstrap_draws(["x", "x", "y"], True)))
        self.assertTrue((d[:, :2] < 2).all())
        self.assertTrue((d[:, 2] == 2).all())

    def test_contrast_sign_and_known_effect(self):
        methods = ["M0", "M1", "G0", "G1"]
        v = np.array([[2, 1, 1, 0.5], [4, 3, 3, 2.5]])
        r = paired_intervals(v, methods, bootstrap_draws(["x", "x"], False), 0.99375)
        self.assertEqual(r[0]["delta_group_balanced_mae_seconds"], -1)
        self.assertEqual(r[0]["ci95_upper_seconds"], -1)
        self.assertEqual(r[0]["empirical_bootstrap_fraction_delta_below_zero"], 1)
        self.assertFalse(r[0]["fraction_is_p_value"])

    def test_group_source_purity(self):
        with self.assertRaises(ValueError):
            group_errors(np.array([1, 2]), {"M0": np.array([1, 1])}, ["a", "a"], ["x", "y"])

    def test_earliest_tie(self):
        self.assertEqual(earliest_minimum([{"validation_mse": 3}, {"validation_mse": 1}, {"validation_mse": 1}]), 2)


if __name__ == "__main__":
    unittest.main()
