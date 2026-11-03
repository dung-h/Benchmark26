"""Time resident regression inference on five-feature inputs, without compilation."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-core", type=int, help="Linux CPU affinity; defaults to one available core")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        core = min(available) if args.cpu_core is None else args.cpu_core
        if core not in available:
            raise ValueError(f"CPU {core} is outside the allowed affinity")
        os.sched_setaffinity(0, {core})
    elif args.cpu_core is not None:
        raise RuntimeError("CPU affinity is unsupported on this platform")

    with args.features.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["task_key"] for row in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError("Feature inputs must have nonempty, unique task keys")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in keys):
        raise ValueError("Task keys must contain only letters, digits, underscores or hyphens")
    # One model file represents one selected fold. Loading is outside the timer.
    model = joblib.load(args.model)
    args.output.mkdir(parents=True)
    for row in rows:
        values = [float(row[f"feature_{i}"]) for i in range(5)]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite features for {row['task_key']}")

        def predict():
            features = np.asarray(values, dtype=np.float64).reshape(1, 5)
            return float(model.predict(features)[0])

        for session in (1, 2, 3):
            expected = predict()
            if not np.isfinite(expected):
                raise ValueError("Model returned a nonfinite prediction")
            if row.get("expected_prediction"):
                reference = float(row["expected_prediction"])
                if not np.isclose(expected, reference, atol=1e-9, rtol=0):
                    raise ValueError("Prediction differs from the supplied reference")
            for _ in range(3):
                predict()
            samples = []
            for _ in range(10):
                start = time.perf_counter_ns()
                value = predict()
                samples.append(time.perf_counter_ns() - start)
                if not np.isclose(value, expected, atol=1e-9, rtol=0):
                    raise ValueError("Prediction changed between repetitions")
            record = dict(task_key=row["task_key"], session=session,
                          prediction=expected, durations_ns=samples)
            path = args.output / f"{session}_{row['task_key']}.json"
            path.write_text(json.dumps(record, indent=2) + "\n")

    receipt = {
        "scope": "Feature conversion and resident model inference; loading excluded",
        "tasks": len(rows), "sessions": 3, "warmups": 3, "repetitions": 10,
        "input_sha256": {"features": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                         "model": hashlib.sha256(args.model.read_bytes()).hexdigest()},
        "environment": {"host": platform.node(), "python": sys.version,
                        "numpy": np.__version__,
                        "sklearn": importlib.metadata.version("scikit-learn"),
                        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
    }
    (args.output / "measurement.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
