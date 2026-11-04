#!/usr/bin/env python3
"""Recompute the shared graph-ablation results from the anonymous per-group errors.

Needs only NumPy and the files in ``results/graph_ablation/``. Group-balanced
MAE (pooled and per source) and the six adjusted paired contrasts are
recomputed from ``group_errors.csv`` and compared with ``summary.json``.
Groups are numbered in the order used for resampling, so the bootstrap draws
are the same as in the original analysis.

The per-group errors were derived from the private predictions and labels by
``analyze.py --export-groups``; this script cannot check that step.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repository root
from analyze import ARMS, CONTRASTS, EVALUATIONS, paired_contrast, stratified_draws

from benchmark26.codesig import code_signature

RESULTS = HERE.parents[1] / "results/graph_ablation"
TOLERANCE = 1e-12


def manifest_problems() -> list[str]:
    """Files listed in the manifest must exist with the recorded SHA-256."""
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    root = RESULTS.parents[1]
    problems = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"manifest hash differs: {name}")
    listed = {Path(n).name for n in manifest["files"]}
    extra = {p.name for p in RESULTS.iterdir() if p.is_file()} - listed - {"manifest.json"}
    return problems + [f"file not in manifest: {name}" for name in sorted(extra)]


def pinned_code_problems() -> list[str]:
    """The files pinned by the executed run must still have the same code.

    A Python file is compared by the signature of its syntax tree, so docstrings,
    comments and layout may change. Other files must be byte-identical.
    """
    manifest = json.loads((RESULTS / "manifest.json").read_text())["pinned_code"]
    root = RESULTS.parents[1]
    problems = []
    for name, expected in manifest["python_signature_sha256"].items():
        path = root / name
        if not path.is_file() or code_signature(path.read_text(encoding="utf-8")) != expected:
            problems.append(f"code differs from the executed run: {name}")
    for name, expected in manifest["byte_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"file differs from the executed run: {name}")
    return problems


def main() -> int:
    """Recompute the published group-balanced errors and the six contrasts from the per-group errors.

    Also checks the manifest hashes and the pinned code files. Prints a JSON status and
    returns 1 on any difference.
    """
    with (RESULTS / "group_errors.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((RESULTS / "summary.json").read_text())
    problems, checked = manifest_problems() + pinned_code_problems(), 0
    for evaluation in EVALUATIONS:
        part = [r for r in rows if r["evaluation"] == evaluation]
        names = [r["group"] for r in part]
        if names != sorted(names) or len(names) != 203:
            problems.append(f"{evaluation}: groups missing or out of order")
            continue
        source = {r["group"]: r["source"] for r in part}
        errors = {arm: {r["group"]: float(r[f"mae_{arm}"]) for r in part} for arm in ARMS}
        for metric in (m for m in summary["metrics"] if m["evaluation"] == evaluation):
            chosen = [g for g in names if metric["scope"] == "all" or source[g] == metric["scope"]]
            value = float(np.mean([errors[metric["arm"]][g] for g in chosen]))
            rows_in_scope = sum(int(r["observations"]) for r in part if r["group"] in set(chosen))
            if abs(value - metric["group_balanced_mae_seconds"]) > TOLERANCE or rows_in_scope != metric["rows"]:
                problems.append(
                    f"{evaluation} {metric['arm']} {metric['scope']}: {value} vs {metric['group_balanced_mae_seconds']}"
                )
            checked += 1
        draws = stratified_draws(names, source)
        for treatment, control in CONTRASTS:
            saved = next(
                c
                for c in summary["contrasts"]
                if c["evaluation"] == evaluation and c["contrast"] == f"{treatment}-{control}"
            )
            result = paired_contrast(errors[treatment], errors[control], names, draws)
            for key, field in (("delta", "delta_group_balanced_mae_seconds"), ("lower", "lower"), ("upper", "upper")):
                if abs(result[key] - saved[field]) > TOLERANCE:
                    problems.append(f"{evaluation} {treatment}-{control} {key}: {result[key]} vs {saved[field]}")
            checked += 1
    print(
        json.dumps({"checked": checked, "problems": problems, "status": "PASS" if not problems else "FAIL"}, indent=2)
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
