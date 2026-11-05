#!/usr/bin/env python3
"""Check the shared Qonductor model-choice tables without the study files.

The script checks the file hashes, that each of the 20 splits has six families
and exactly one chosen family with the highest score, and that the summary
table equals the means and counts of the per-split table.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.integrity import sha256_file
from benchmark26.tables import read_csv as read

RESULTS = ROOT / "results/qonductor_selection"


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

    splits = defaultdict(list)
    for r in read(RESULTS / "family_scores_by_split.csv"):
        splits[(r["setting"], r["fold"])].append(r)
    if len(splits) != 20:
        found.append(f"expected 20 splits, found {len(splits)}")
    for key, part in splits.items():
        chosen = [r for r in part if r["chosen"] == "1"]
        best = max(part, key=lambda r: float(r["best_mean_inner_r2"]))
        if len(part) != 6 or len(chosen) != 1 or chosen[0] is not best:
            found.append(f"split {key}: chosen family is not the single highest-scoring one")
    for r in read(RESULTS / "family_scores.csv"):
        part = [x for k, v in splits.items() if k[0] == r["setting"] for x in v if x["family"] == r["family"]]
        mean = st.mean(float(x["best_mean_inner_r2"]) for x in part)
        if (int(r["splits"]) != len(part) or not math.isclose(float(r["mean_best_inner_r2"]), mean, rel_tol=1e-12)
                or int(r["times_chosen"]) != sum(x["chosen"] == "1" for x in part)):
            found.append(f"summary row differs: {r['setting']} {r['family']}")
    return found


def main() -> None:
    found = problems()
    for p in found:
        print(f"FAIL {p}")
    if found:
        sys.exit(1)
    print("Qonductor model choice: hashes, choices and summary agree")


if __name__ == "__main__":
    main()
