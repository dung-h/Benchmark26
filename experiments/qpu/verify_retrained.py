#!/usr/bin/env python3
"""Check results/retrained_neural/ and recompute its error metrics.

Without runtimes this checks the file hashes, the row counts and that each prediction
is the median of its three seed predictions. With the public dataset table it also
recomputes metrics.csv:

    python3 experiments/qpu/verify_retrained.py --runs /path/to/runs.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/retrained_neural"
SEEDS = (42, 1234, 31415)
EXPECTED = {"E1": ("M0", "M1", "G0", "G1"), "E2": ("M0", "M1", "G0", "G1"), "E3": ("M1", "G1")}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def metrics(predictions: list[dict[str, str]], runs: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Return GB-MAE, MAE and R2 for each evaluation and treatment, over all runs and by source and backend."""
    table = []
    for evaluation, treatments in EXPECTED.items():
        for treatment in treatments:
            cell = [r for r in predictions if r["evaluation"] == evaluation and r["treatment"] == treatment]
            scopes: dict[str, list] = {"all": []}
            for row in cell:
                run = runs[row["canonical_observation_id"]]
                item = (run["circuit_group"], float(run["runtime_seconds"]), float(row["predicted_seconds"]))
                for scope in ("all", f"source={run['source']}", f"backend={run['backend']}"):
                    scopes.setdefault(scope, []).append(item)
            for scope in sorted(scopes):
                items = scopes[scope]
                groups: dict[str, list[float]] = {}
                for group, truth, value in items:
                    groups.setdefault(group, []).append(abs(value - truth))
                center = mean(truth for _, truth, _ in items)
                residual = sum((value - truth) ** 2 for _, truth, value in items)
                total = sum((truth - center) ** 2 for _, truth, _ in items)
                table.append(
                    {
                        "evaluation": evaluation,
                        "treatment": treatment,
                        "scope": scope,
                        "runs": str(len(items)),
                        "groups": str(len(groups)),
                        "gb_mae_seconds": f"{mean(mean(v) for v in groups.values()):.6f}",
                        "mae_seconds": f"{mean(abs(value - truth) for _, truth, value in items):.6f}",
                        "r2": f"{1 - residual / total:.6f}",
                    }
                )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, help="runs.csv of the public dataset")
    parser.add_argument("--write", action="store_true", help="write metrics.csv instead of comparing with it")
    args = parser.parse_args()
    if not args.write:
        manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["files"].items():
            if hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() != digest:
                raise SystemExit(f"hash mismatch: {name}")
    predictions = read(RESULTS / "predictions.csv")
    cells: dict[tuple[str, str], set[str]] = {}
    for row in predictions:
        cells.setdefault((row["evaluation"], row["treatment"]), set()).add(row["canonical_observation_id"])
        median = sorted(float(row[f"seed_{seed}"]) for seed in SEEDS)[1]
        if float(row["predicted_seconds"]) != median:
            raise SystemExit("a prediction is not the median of its seed predictions")
    expected = {(e, t) for e, treatments in EXPECTED.items() for t in treatments}
    if set(cells) != expected or any(len(ids) != 3443 for ids in cells.values()) or len(predictions) != 34430:
        raise SystemExit("unexpected cells or row counts")
    if args.runs is None:
        if args.write:
            raise SystemExit("--write needs --runs")
        print("retrained_neural: hashes, row counts and seed medians agree (give --runs to recompute metrics)")
        return 0
    runs = {row["run_id"]: row for row in read(args.runs)}
    computed = metrics(predictions, runs)
    if args.write:
        with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(computed[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(computed)
        return 0
    saved = read(RESULTS / "metrics.csv")
    if saved != computed:
        raise SystemExit("recomputed metrics differ from metrics.csv")
    print(f"retrained_neural: {len(computed)} metric rows recomputed and equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
