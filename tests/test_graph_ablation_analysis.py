"""The graph-ablation estimand and the stratified paired bootstrap."""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/graph_ablation"))
from analyze import ALPHA, group_errors, paired_contrast, stratified_draws


def test_metric_uses_median_of_seeds_not_mean_of_seed_errors():
    # One observation; seeds predict 0, 10, 2 for a label of 1.
    predictions = {"a": {42: 0.0, 1234: 10.0, 31415: 2.0}}
    errors, per_seed = group_errors(predictions, {"a": 1.0}, {"g": ["a"]})
    assert errors["g"] == 1.0  # |median(0, 10, 2) - 1|
    assert np.mean(list(per_seed.values())) == (1 + 9 + 1) / 3


def test_group_error_is_mean_within_group():
    predictions = {c: {42: p, 1234: p, 31415: p} for c, p in (("a", 1.0), ("b", 3.0))}
    errors, _ = group_errors(predictions, {"a": 0.0, "b": 0.0}, {"g": ["a", "b"]})
    assert errors["g"] == 2.0


def test_draws_resample_within_source_and_keep_counts():
    names = [f"g{i}" for i in range(7)]
    source = {g: ("s1" if i < 5 else "s2") for i, g in enumerate(names)}
    draws = stratified_draws(names, source, draws=500, seed=1)
    assert draws.shape == (500, 7)
    assert np.isin(draws[:, :5], np.arange(5)).all() and np.isin(draws[:, 5:], [5, 6]).all()


def test_paired_contrast_is_constant_for_constant_difference():
    names = [f"g{i}" for i in range(6)]
    source = {g: "s" for g in names}
    treatment = {g: 1.0 + i for i, g in enumerate(names)}
    control = {g: 0.5 + i for i, g in enumerate(names)}
    result = paired_contrast(treatment, control, names, stratified_draws(names, source, draws=200, seed=3))
    assert np.isclose(result["delta"], 0.5) and np.isclose(result["lower"], 0.5) and np.isclose(result["upper"], 0.5)
    assert np.isclose(1 - ALPHA, 0.991666, atol=1e-6)


def test_shared_results_reproduce():
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/graph_ablation/verify_summary.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
