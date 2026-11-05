#!/usr/bin/env python3
"""Check the shared Scholten-style results without the study files.

The script checks that the files in ``results/scholten/`` match their manifest
hashes, that the CLOPS table is the one the run used, and that the run counts
agree: covered plus uncovered runs make 3,443, and the shot-range table and the
per-source scores each add up to the covered runs. Recomputing the scores needs
the study files; see ``docs/SCHOLTEN.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.integrity import sha256_file
from benchmark26.tables import read_csv as read

RESULTS = ROOT / "results/scholten"


def problems() -> list[str]:
    found = []
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != expected:
            found.append(f"manifest hash differs: {name}")
    extra = {p.name for p in RESULTS.iterdir()} - {Path(n).name for n in manifest["files"]} - {"manifest.json"}
    found += [f"file not in manifest: {name}" for name in sorted(extra)]
    if found:
        return found

    receipt = json.loads((RESULTS / "receipt.json").read_text())
    if sha256_file(ROOT / "experiments/scholten/clops.csv") != receipt["clops_table_sha256"]:
        found.append("experiments/scholten/clops.csv differs from the table the run used")
    covered = receipt["covered_runs"]
    if covered + sum(receipt["uncovered_runs_by_backend"].values()) != 3443:
        found.append("covered and uncovered runs do not add up to 3,443")
    scores = read(RESULTS / "scores.csv")
    if sum(int(r["runs"]) for r in read(RESULTS / "ratio_by_shots.csv")) != covered:
        found.append("shot-range table does not add up to the covered runs")
    by_source = [r for r in scores if r["method"] == "scholten_style" and r["scope"] == "source"]
    if sum(int(r["runs"]) for r in by_source) != covered:
        found.append("per-source scores do not add up to the covered runs")
    if any(int(r["runs"]) != covered for r in scores if r["scope"] == "all"):
        found.append("an estimator was scored on a different set of runs")
    return found


def main() -> None:
    found = problems()
    for p in found:
        print(f"FAIL {p}")
    if found:
        sys.exit(1)
    print("Scholten-style results: hashes and run counts agree")


if __name__ == "__main__":
    main()
