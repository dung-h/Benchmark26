#!/usr/bin/env python3
"""Summarize which regression family the Qonductor procedure chose, and why.

In every outer split, the Qonductor procedure scores the candidates of six
model families by their mean R^2 in seconds over four inner folds, keeps the
best candidate and refits it. This script reads those saved candidate scores
from the study files (``B26_ARTIFACTS``) and writes, for each split and family,
the number of candidates, the best mean inner R^2 and whether the family was
chosen. It also checks that the chosen family is the one with the highest
score and the one named in the saved test predictions.

These are selection scores on the training part of each split, not test
scores. The script refuses to write into an existing folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.paths import ARTIFACTS
from benchmark26.tables import read_csv as read

CLASSICAL = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1/classical"
FAMILIES = ("Extra Trees", "Gradient Boosting", "Random Forest", "Histogram Gradient Boosting",
            "AdaBoost", "Polynomial Regression")
SPLITS = {"E1": 5, "E2": 5, "E3": 10}


def best_scores(folder: Path) -> dict[str, tuple[int, float]]:
    """Family -> (number of scored candidates, best mean inner R^2) for one split."""
    result = {}
    for path in sorted(folder.glob("*_candidate_scores.json")):
        data = json.loads(path.read_text())
        scored = [c["mean_inner_r2"] for c in data["scores"] if c.get("status") == "scored"]
        result[data["identity"]["family"]] = (len(scored), max(scored))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="new folder for the results")
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)

    rows, problems = [], []
    for setting, folds in SPLITS.items():
        for fold in range(folds):
            folder = CLASSICAL / setting / f"fold_{fold}"
            scores = best_scores(folder)
            if set(scores) != set(FAMILIES):
                problems.append(f"{setting} fold {fold}: families {sorted(scores)}")
                continue
            named = {r["selected_family"] for r in read(folder / "regression_predictions.csv")}
            highest = max(scores, key=lambda f: scores[f][1])
            if named != {highest}:
                problems.append(f"{setting} fold {fold}: predictions name {sorted(named)}, highest score is {highest}")
            manifest = json.loads((folder / "regression_manifest.json").read_text())
            held_out = manifest.get("heldout_backend") or manifest.get("identity", {}).get("heldout_backend", "")
            for family in FAMILIES:
                count, best = scores[family]
                rows.append({"setting": setting, "fold": fold, "held_out_backend": held_out.removeprefix("ibm_"),
                             "family": family, "candidates": count, "best_mean_inner_r2": best,
                             "chosen": int(family == highest)})
    if problems:
        sys.exit("\n".join(problems))

    summary = []
    for setting in SPLITS:
        for family in FAMILIES:
            part = [r for r in rows if r["setting"] == setting and r["family"] == family]
            summary.append({"setting": setting, "family": family, "splits": len(part),
                            "mean_best_inner_r2": st.mean(r["best_mean_inner_r2"] for r in part),
                            "times_chosen": sum(r["chosen"] for r in part)})

    for name, table in (("family_scores_by_split.csv", rows), ("family_scores.csv", summary)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(table)
    print(json.dumps({"splits": len(rows) // len(FAMILIES), "chosen": dict(sorted(
        defaultdict(int, {f: sum(r["chosen"] for r in rows if r["family"] == f) for f in FAMILIES}).items()))}))


if __name__ == "__main__":
    main()
