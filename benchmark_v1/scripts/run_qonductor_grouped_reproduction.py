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

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/home/server/Documents/Capstone-Project/eda/phase2_dataset_audit/extracts/01_qonductor_single_circuit_ibm_job_time.parquet")
DEFAULT_SPLIT = ROOT / "benchmark_v1/splits/qonductor_single_circuit_ibm.csv"
DEFAULT_VERSION = ROOT / "benchmark_v1/registry/data_versions/archived_qpu_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_state() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": "unavailable"}
    return {"revision": revision, "dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def base_models() -> dict[str, Any]:
    # Same model classes and seeds as upstream RegressionEstimator.
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
    return {
        "Extra Trees": {"n_estimators": [100, 300, 500, 700], "max_depth": [None, 5, 10, 30, 50], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 5, 10, 20, 30], "max_features": [None, "sqrt", "log2"], "max_leaf_nodes": [None, 10, 30, 50, 70], "bootstrap": [True, False]},
        "Random Forest": {"n_estimators": [100, 300, 500, 700], "max_depth": [None, 5, 10, 30, 50], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 5, 10, 20, 30], "max_features": [None, "sqrt", "log2"], "max_leaf_nodes": [None, 10, 30, 50, 70], "bootstrap": [True, False]},
        "Gradient Boosting": {"learning_rate": [0.001, 0.01, 0.1, 1], "n_estimators": [100, 300, 500, 700], "subsample": [0.3, 0.5, 0.7, 1.0], "max_depth": [None, 5, 10, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 5, 10, 20, 30], "max_features": [None, "sqrt", "log2"], "max_leaf_nodes": [None, 10, 30, 50, 70]},
        "AdaBoost": {"n_estimators": [100, 300, 500, 700], "learning_rate": [0.001, 0.01, 0.1, 1], "loss": ["linear", "square", "exponential"]},
        "Histogram Gradient Boosting": {"learning_rate": [0.001, 0.01, 0.1, 1], "max_iter": [100, 300, 500, 700], "max_leaf_nodes": [10, 30, 50, 70], "max_depth": [None, 5, 10, 30, 50], "min_samples_leaf": [5, 10, 20, 30], "l2_regularization": [0, 0.01, 0.1, 1]},
        "Polynomial Regression": {"polynomialfeatures__degree": [2, 3, 4]},
    }


def canary_grids() -> dict[str, dict[str, list[Any]]]:
    # A predeclared, one-cell-per-family feasibility bank.  It is not used to
    # make an upstream-paper-exact claim.
    return {
        "Extra Trees": {"n_estimators": [100], "max_depth": [None], "min_samples_split": [2], "min_samples_leaf": [1], "max_features": [None], "max_leaf_nodes": [None], "bootstrap": [False]},
        "Random Forest": {"n_estimators": [100], "max_depth": [None], "min_samples_split": [2], "min_samples_leaf": [1], "max_features": [None], "max_leaf_nodes": [None], "bootstrap": [False]},
        "Gradient Boosting": {"learning_rate": [0.1], "n_estimators": [100], "subsample": [1.0], "max_depth": [5], "min_samples_split": [2], "min_samples_leaf": [1], "max_features": [None], "max_leaf_nodes": [None]},
        "AdaBoost": {"n_estimators": [100], "learning_rate": [0.1], "loss": ["linear"]},
        "Histogram Gradient Boosting": {"learning_rate": [0.1], "max_iter": [100], "max_leaf_nodes": [30], "max_depth": [None], "min_samples_leaf": [20], "l2_regularization": [0]},
        "Polynomial Regression": {"polynomialfeatures__degree": [2, 3, 4]},
    }


def select_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str) -> tuple[str, Any, dict[str, Any]]:
    models = base_models()
    grids = source_grids() if mode == "full_source_grid" else (source_grids() if mode == "source_poly_grid" else canary_grids())
    names = ["Polynomial Regression"] if mode == "source_poly_grid" else list(models)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("outer training subset has fewer than two exact-QASM groups")
    cv = GroupKFold(n_splits=n_splits)
    candidates: list[dict[str, Any]] = []
    for name in names:
        for params in ParameterGrid(grids[name]):
            scores = []
            for inner_train, inner_valid in cv.split(x, y, groups):
                model = clone(models[name]).set_params(**params)
                model.fit(x[inner_train], y[inner_train])
                scores.append(float(r2_score(y[inner_valid], model.predict(x[inner_valid]))))
            candidates.append({"name": name, "params": params, "inner_group_r2_mean": float(np.mean(scores)), "inner_group_r2_folds": scores})
    best = max(candidates, key=lambda value: value["inner_group_r2_mean"])
    fitted = clone(models[best["name"]]).set_params(**best["params"])
    fitted.fit(x, y)
    return str(best["name"]), fitted, {"candidate_count": len(candidates), "selected": best, "inner_cv": "GroupKFold(exact_qasm_hash, n_splits=%d)" % n_splits}


def frozen_outer_split(split: pd.DataFrame, groups: np.ndarray, kind: str, fold: int | None, held_out: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return predeclared train/test rows and reject cross-side exact QASM."""
    if kind == "primary_fold":
        if fold is None:
            raise ValueError("--fold is required for primary_fold")
        test = np.flatnonzero(split["primary_fold"].to_numpy(dtype=int) == fold)
        train = np.flatnonzero(split["primary_fold"].to_numpy(dtype=int) != fold)
        label = f"fold_{fold}"
    elif kind == "temporal":
        train = np.flatnonzero(split["temporal_role"].astype(str).to_numpy() == "train")
        test = np.flatnonzero(split["temporal_role"].astype(str).to_numpy() == "test")
        label = "strict_group_temporal_80_20"
    elif kind == "backend":
        if not held_out:
            raise ValueError("--held-out BACKEND is required for backend split")
        values = split["backend_holdout_id"].astype(str).to_numpy()
        test = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        label = held_out
    elif kind == "family_component":
        if not held_out:
            raise ValueError("--held-out COMPONENT is required for family_component split")
        values = split["family_component_holdout_id"].astype(str).to_numpy()
        test = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        label = held_out
    else:
        raise ValueError(kind)
    test_groups = set(groups[test])
    # Backend holdout can otherwise retain the identical QASM on another
    # backend, violating the benchmark's exact-circuit guard.
    train = np.asarray([index for index in train if groups[index] not in test_groups], dtype=int)
    if not len(train) or not len(test):
        raise ValueError(f"{kind}/{label} yields empty train or test")
    if set(groups[train]) & test_groups:
        raise ValueError(f"{kind}/{label} leaks exact QASM")
    return train, test, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--data-version", type=Path, default=DEFAULT_VERSION)
    parser.add_argument("--split-kind", choices=["primary_fold", "temporal", "backend", "family_component"], default="primary_fold")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--held-out", help="Frozen backend or family-component id for the named split kind")
    parser.add_argument("--mode", choices=["source_poly_grid", "family_canary", "full_source_grid"], default="source_poly_grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.data_version.exists():
        raise SystemExit("data version must be frozen before fitting: " + str(args.data_version))
    data = pd.read_parquet(args.data).reset_index(drop=True)
    split = pd.read_csv(args.split).reset_index(drop=True)
    if len(data) != len(split):
        raise ValueError(f"data/split row mismatch: {len(data)} != {len(split)}")
    if (data["circuit_count"].astype(int) != 1).any():
        raise ValueError("non-single-circuit row in Qonductor input")
    features = ["sum_depth", "sum_qubits", "sum_2q_gates", "shots", "circuit_count"]
    x = data[features].astype(float).to_numpy()
    y = data["taken_time_seconds"].astype(float).to_numpy()
    groups = split["circuit_group_hash"].astype(str).to_numpy()
    train, test, split_label = frozen_outer_split(split, groups, args.split_kind, args.fold, args.held_out)
    selected_name, model, selection = select_model(x[train], y[train], groups[train], args.mode)
    pred = np.maximum(np.asarray(model.predict(x[test]), dtype=float), 0.0)
    actual = y[test]
    prediction = pd.DataFrame({
        "row_index": test, "split_kind": args.split_kind, "held_out": split_label,
        "fold": args.fold if args.fold is not None else "", "circuit_group_hash": groups[test],
        "backend": data.iloc[test]["backend_name"].astype(str).to_numpy(),
        "actual_seconds": actual, "predicted_seconds": pred,
        "selected_model": selected_name,
    })
    summary = {
        "evaluation_kind": "group-safe Qonductor adaptation canary" if args.mode != "full_source_grid" else "group-safe full source-grid adaptation",
        "mode": args.mode, "split_kind": args.split_kind, "held_out": split_label, "fold": args.fold, "n_train": int(len(train)), "n_test": int(len(test)),
        "train_qasm_groups": int(len(set(groups[train]))), "test_qasm_groups": int(len(set(groups[test]))),
        "outer_qasm_overlap": 0,
        "selected_model": selected_name, "selection": selection,
        "mae_seconds": float(np.mean(np.abs(actual - pred))),
        "medae_seconds": float(np.median(np.abs(actual - pred))),
        "r2_seconds": float(r2_score(actual, pred)) if len(actual) > 1 else None,
        "mae_log1p_seconds": float(np.mean(np.abs(np.log1p(actual) - np.log1p(pred)))),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"
    prediction.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "protocol_id": "qre-benchmark-v1", "evaluation_id": args.output_dir.name,
        "domain": "archived_qpu_observation", "live_qpu_submission": False,
        "command": sys.argv, "code_state": git_state(), "seed_registry_id": "seed-registry-v1",
        "environment_fingerprint": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "input_manifests": [
            {"path": relative(args.data), "sha256": sha256(args.data)},
            {"path": relative(args.split), "sha256": sha256(args.split)},
            {"path": relative(args.data_version), "sha256": sha256(args.data_version)},
        ],
        "split_manifest": {"path": relative(args.split), "sha256": sha256(args.split)},
        "model_selection": {"upstream_source": "Qonductor-SC25/src/execution_time/regression_estimator.py", "outer_split": f"frozen {args.split_kind}: {split_label}", "inner_split": "GroupKFold exact-QASM", "mode": args.mode, "full_source_grid_warning": "Only full_source_grid enumerates all upstream candidates; source_poly_grid is exact only for polynomial degree candidates; family_canary is feasibility-only."},
        "outputs": [{"path": relative(predictions_path), "sha256": sha256(predictions_path)}, {"path": relative(summary_path), "sha256": sha256(summary_path)}],
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    manifest_path = args.output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(manifest_path)


if __name__ == "__main__":
    main()
