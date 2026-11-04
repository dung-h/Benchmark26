#!/usr/bin/env python3
"""Execute the frozen six-family Qonductor-style adaptation on the strict panel.

The input is the 340 Ma-Li observations only. Existing Qonductor feature rows
are reused only after exact logical/compiled-QASM, backend, shot and identity
joins against the frozen strict panel. No labels enter feature construction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from benchmark26.paths import ARTIFACTS, PROTOCOLS, QONDUCTOR_SOURCE

ROOT = Path(__file__).resolve().parents[1]
PANEL = ARTIFACTS / "real_qpu/strict_logical_panel"
CONTRACT = PROTOCOLS / "strict_method_completion.json"
PARENT = PROTOCOLS / "strict_logical_benchmark.json"
AUTH = PANEL / "execution_authorization.json"
FEATURES = ARTIFACTS / "real_qpu/qonductor_native_features/feature_rows.csv"
REPRESENTATIONS = ARTIFACTS / "benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
SOURCE_ROOT = QONDUCTOR_SOURCE
OUTPUT = PANEL / "qonductor"
SOURCE_REVISION = "5d1ac8a90cd574a23e7544e1044681641354ff67"
SOURCE_FILE = "src/execution_time/regression_estimator.py"
SOURCE_SHA256 = "d44ab196a56fba444f5cebfe9dee8737ad3b6ab6f39081336a41d5811fdb828a"
FEATURE_ORDER = ["swap", "depth", "num_qubits", "shots", "circuit_count"]
FAMILIES = [
    "Extra Trees",
    "Random Forest",
    "Gradient Boosting",
    "AdaBoost",
    "Histogram Gradient Boosting",
    "Polynomial Regression",
]
EXPECTED_COUNTS = [64, 64, 64, 48, 64, 3]


def sha_bytes(value: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into a list of dicts."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, obj: Any) -> None:
    """Write a value as indented JSON atomically."""
    payload = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, payload)


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a temporary `.partial` file and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows as a CSV file atomically."""
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def source_grids_and_models():
    """Use the audited transcription of the pinned upstream constructor/grid."""
    source_blob = (
        __import__("subprocess")
        .run(
            ["git", "-C", str(SOURCE_ROOT), "show", f"{SOURCE_REVISION}:{SOURCE_FILE}"],
            check=True,
            capture_output=True,
        )
        .stdout
    )
    if sha_bytes(source_blob) != SOURCE_SHA256:
        raise RuntimeError("pinned_qonductor_source_hash_mismatch")
    from benchmark26.qonductor_models import base_models, source_grids

    return base_models(), source_grids(), source_blob


def fit_one_candidate(family, candidate, x, y, splits, model_templates):
    """Fit one candidate on each inner split and return its inner R^2 scores, status and mean.

    A non-finite prediction marks the candidate as a numeric failure.
    """
    results = []
    for inner_fold, (fit_ids, valid_ids) in splits.items():
        fit_idx = np.asarray([x["id_to_index"][i] for i in fit_ids], dtype=int)
        valid_idx = np.asarray([x["id_to_index"][i] for i in valid_ids], dtype=int)
        estimator = clone(model_templates[family]).set_params(**candidate["raw_params"])
        with threadpool_limits(limits=1):
            estimator.fit(x["matrix"][fit_idx], y[fit_idx])
            prediction = np.asarray(estimator.predict(x["matrix"][valid_idx]), dtype=np.float64)
        if not np.isfinite(prediction).all():
            results.append({"inner_fold": inner_fold, "score": "", "status": "numeric_failure_nonfinite_prediction"})
        else:
            results.append(
                {"inner_fold": inner_fold, "score": float(r2_score(y[valid_idx], prediction)), "status": "scored"}
            )
    status = "scored" if all(r["status"] == "scored" for r in results) else "numeric_failure"
    return {
        "family": family,
        "candidate_index": int(candidate["candidate_index"]),
        "params_json": json.dumps(candidate["raw_params"], sort_keys=True, separators=(",", ":")),
        "status": status,
        "inner": results,
        "mean_inner_r2": (float(np.mean([r["score"] for r in results])) if status == "scored" else ""),
    }


def metric_rows(predictions):
    """Return the metrics of each method over its predicted rows, out of 340 assigned.

    MAE, median error, RMSE, R^2, tail errors and the number of negative predictions.
    """
    methods = sorted({row["method_id"] for row in predictions})
    out = []
    for method in methods:
        rows = [row for row in predictions if row["method_id"] == method and row["status"] == "predicted"]
        y = np.asarray([float(r["actual_seconds"]) for r in rows], dtype=np.float64)
        p = np.asarray([float(r["predicted_seconds"]) for r in rows], dtype=np.float64)
        if len(y) == 0:
            continue
        abs_error = np.abs(y - p)
        out.append(
            {
                "method_id": method,
                "assigned_rows": 340,
                "predicted_rows": len(y),
                "coverage": len(y) / 340,
                "mae_seconds": float(abs_error.mean()),
                "medae_seconds": float(np.median(abs_error)),
                "rmse_seconds": float(np.sqrt(np.mean((y - p) ** 2))),
                "r2_seconds": float(r2_score(y, p)),
                "p90_absolute_error_seconds": float(np.quantile(abs_error, 0.9)),
                "p99_absolute_error_seconds": float(np.quantile(abs_error, 0.99)),
                "max_absolute_error_seconds": float(abs_error.max()),
                "negative_prediction_count": sum(float(r["predicted_seconds"]) < 0 for r in rows),
            }
        )
    return out


if __name__ == "__main__":
    main()
