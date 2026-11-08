#!/usr/bin/env python3
"""Leakage-safe Qonductor model-selection reproduction on frozen outer folds.

The upstream implementation uses ordinary row-wise KFold.  This runner keeps
its estimator families, random seeds and candidate values, but replaces both
inner and outer splits with exact-QASM group-disjoint splits as required by
the benchmark contract.  Consequently its output is an *adaptation*, not a
paper-exact reproduction.  ``source_poly_grid`` is a cheap canary that uses
the upstream PolynomialFeatures candidate grid exactly; ``full_source_grid``
is intentionally explicit because its source grid is very large.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from benchmark26.paths import QONDUCTOR_TABLE, REGISTRY, record_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = QONDUCTOR_TABLE
DEFAULT_SPLIT = ROOT / "inputs/splits/qonductor_single_circuit_ibm.csv"
DEFAULT_VERSION = REGISTRY / "data_versions/archived_qpu_v1.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    """Return the recorded name of a path, or the path itself if it has none."""
    try:
        return record_name(path)
    except ValueError:
        return str(path)


def base_models() -> dict[str, Any]:
    # Same model classes and seeds as upstream RegressionEstimator.
    """Return the six model families with the same classes and seeds as the upstream estimator."""
    return {
        "Extra Trees": ExtraTreesRegressor(n_jobs=1, random_state=0),
        "Random Forest": RandomForestRegressor(n_jobs=1, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0),
        "AdaBoost": AdaBoostRegressor(random_state=0),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(random_state=0),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression()),
    }


def source_grids() -> dict[str, dict[str, list[Any]]]:
    # Transcribed from Qonductor-SC25/src/execution_time/regression_estimator.py.
    """Return the hyperparameter grid of each model family; the source modes search these grids."""
    return {
        "Extra Trees": {
            "n_estimators": [100, 300, 500, 700],
            "max_depth": [None, 5, 10, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10, 20, 30],
            "max_features": [None, "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 30, 50, 70],
            "bootstrap": [True, False],
        },
        "Random Forest": {
            "n_estimators": [100, 300, 500, 700],
            "max_depth": [None, 5, 10, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10, 20, 30],
            "max_features": [None, "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 30, 50, 70],
            "bootstrap": [True, False],
        },
        "Gradient Boosting": {
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "n_estimators": [100, 300, 500, 700],
            "subsample": [0.3, 0.5, 0.7, 1.0],
            "max_depth": [None, 5, 10, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10, 20, 30],
            "max_features": [None, "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 30, 50, 70],
        },
        "AdaBoost": {
            "n_estimators": [100, 300, 500, 700],
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "loss": ["linear", "square", "exponential"],
        },
        "Histogram Gradient Boosting": {
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "max_iter": [100, 300, 500, 700],
            "max_leaf_nodes": [10, 30, 50, 70],
            "max_depth": [None, 5, 10, 30, 50],
            "min_samples_leaf": [5, 10, 20, 30],
            "l2_regularization": [0, 0.01, 0.1, 1],
        },
        "Polynomial Regression": {"polynomialfeatures__degree": [2, 3, 4]},
    }


if __name__ == "__main__":
    main()
