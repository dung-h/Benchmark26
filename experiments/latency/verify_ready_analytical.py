#!/usr/bin/env python3
"""Recompute the shared ready-input analytical timing means from anonymous group means.

Needs only NumPy and ``results/ready_analytical_latency/``. The script recomputes
the group-balanced means in ``ready_analytical_summary.csv`` (pooled and per
source) and the observation and group counts. Observation medians and P90 need
the per-observation values, which stay with the external measurement records.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # repository root
from benchmark26.tables import read_csv as read

RESULTS = ROOT / "results/ready_analytical_latency"
COMPONENTS = ("qiskit", "qcre_conversion", "qcre_estimator", "qcre_total")


def manifest_problems() -> list[str]:
    """Files listed in the manifest must exist with the recorded SHA-256, and no other file may be present."""
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    root = RESULTS.parents[1]
    problems = [
        f"manifest hash differs: {name}"
        for name, expected in manifest["files"].items()
        if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
    ]
    listed = {Path(n).name for n in manifest["files"]}
    extra = {p.name for p in RESULTS.iterdir() if p.is_file()} - listed - {"manifest.json"}
    return problems + [f"file not in manifest: {name}" for name in sorted(extra)]


def main() -> int:
    """Recompute the group-balanced means from the group means and compare them with the summary and the manifest."""
    groups = read(RESULTS / "ready_analytical_group_means.csv")
    summary = read(RESULTS / "ready_analytical_summary.csv")
    problems = manifest_problems()
    for row in summary:
        chosen = [g for g in groups if row["scope"] == "all" or g["source"] == row["scope"]]
        mean_ms = float(np.mean([float(g[f"{row['component']}_ns"]) for g in chosen])) / 1e6
        if abs(mean_ms - float(row["group_balanced_mean_ms"])) > 1e-9 * max(1.0, mean_ms):
            problems.append(f"{row['scope']} {row['component']}: {mean_ms} vs {row['group_balanced_mean_ms']}")
        if len(chosen) != int(row["groups"]) or sum(int(g["observations"]) for g in chosen) != int(row["observations"]):
            problems.append(f"{row['scope']} {row['component']}: counts")
    print(json.dumps({"checked": len(summary), "problems": problems, "status": "PASS" if not problems else "FAIL"}))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
