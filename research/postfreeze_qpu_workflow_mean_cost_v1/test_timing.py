"""Small correctness tests for the new timing boundary and replay checks."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from timing import check_prediction, neural_predict, predict_c0, predict_r0


class Linear(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))
    def forward(self, x):
        return x * self.weight


class Tests(unittest.TestCase):
    def test_ready_input_three_seed_median_and_inference_guard(self):
        models = {42: Linear(1), 1234: Linear(3), 31415: Linear(2)}
        events = []
        def prepare(x):
            events.append("prepare")
            return x
        with patch("timing.time.perf_counter_ns", side_effect=(100, 200)):
            value, seeds, elapsed = neural_predict(models, torch.tensor(2.), lambda x, **_: x,
                graph=False, device="cpu", prepare_fn=prepare)
        self.assertEqual(value, 4.)
        self.assertEqual(elapsed, 100)
        self.assertEqual(seeds, {"42": 2., "1234": 6., "31415": 4.})
        check_prediction(value, {"prediction": 4., "seeds": seeds}, "M0", seeds)

    def test_classical_and_constant_boundaries(self):
        class Estimator:
            def predict(self, x):
                self.seen = x.copy()
                return np.asarray([float(x.sum())])
        estimator = Estimator()
        x = np.arange(5, dtype=np.float64)
        with patch("timing.time.perf_counter_ns", side_effect=(10, 30)):
            value, elapsed = predict_r0(estimator, x)
        self.assertEqual(value, 10.)
        self.assertEqual(elapsed, 20)
        np.testing.assert_array_equal(estimator.seen, x.reshape(1, 5))
        with patch("timing.time.perf_counter_ns", side_effect=(11, 16)):
            value, elapsed = predict_c0(2.5)
        self.assertEqual((value, elapsed), (2.5, 5))

    def test_replay_tolerances_and_seed_median(self):
        seeds = {"42": 1.0, "1234": 2.0, "31415": 3.0}
        check_prediction(2.0, {"prediction": 2.0, "seeds": seeds}, "G0", seeds)
        with self.assertRaises(RuntimeError):
            check_prediction(2.0, {"prediction": 2.0, "seeds": {"42": 1, "1234": 2}}, "G0", {"42": 1, "1234": 2})
        with self.assertRaises(RuntimeError):
            check_prediction(2.00000001, {"prediction": 2.0}, "R0", {})


if __name__ == "__main__":
    unittest.main()
