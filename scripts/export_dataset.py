#!/usr/bin/env python3
"""Write the public runtime dataset: one table with one row per QPU run.

The table is built from two files of the study: the observation list
(`$B26_COHORT_PLAN/proposed_observations.csv`) and the prepared table with its
labels (`$B26_ARTIFACTS/postfreeze_qpu_workflow_mean_expansion_v1/cohort/`
`panel.csv` and `targets.csv`). Values are copied unchanged, except that source
names are shortened and shot counts are written as integers. Columns that only
record steps of the original study are left out.

    python3 scripts/export_dataset.py --output /path/to/benchmark26-data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from benchmark26.paths import ARTIFACTS, COHORT_PLAN

SOURCE_NAMES = {"mali_real_qpu": "ma_li", "qonductor_single_circuit_ibm": "qonductor", "qpack_mcp": "qpack"}
COLUMNS = [
    "run_id",
    "source",
    "circuit_group",
    "backend",
    "shots",
    "runtime_seconds",
    "timings_averaged",
    "logical_circuit_hash",
    "physical_circuit_hash",
]
PREPARED = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1/cohort"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build() -> list[dict[str, str]]:
    listed = read(COHORT_PLAN / "proposed_observations.csv")
    prepared = read(PREPARED / "panel.csv")
    labels = {row["canonical_observation_id"]: row["target_seconds"] for row in read(PREPARED / "targets.csv")}
    if len(listed) != len(prepared) or len(labels) != len(listed):
        raise SystemExit("the study tables have different row counts")
    rows = []
    for first, second in zip(listed, prepared):
        run_id = first["observation_id"]
        checks = {
            "run order": second["canonical_observation_id"] == run_id,
            "source": second["source_id"] == first["source"],
            "circuit group": second["group_id"] == first["proposed_conservative_group"],
            "logical circuit": second["logical_signature"] == first["logical_signature"],
            "label": labels[run_id] == first["target_seconds"],
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SystemExit(f"{run_id}: the study tables disagree on {', '.join(failed)}")
        shots = float(first["shots"])
        if not shots.is_integer():
            raise SystemExit(f"{run_id}: shot count {first['shots']} is not an integer")
        rows.append({
            "run_id": run_id,
            "source": SOURCE_NAMES[first["source"]],
            "circuit_group": first["proposed_conservative_group"],
            "backend": first["backend"],
            "shots": str(int(shots)),
            "runtime_seconds": first["target_seconds"],
            # Number of workflow timings averaged into a QPack label; other labels are one value.
            "timings_averaged": first["constituent_count"] if first["aggregation"] == "workflow_mean" else "",
            "logical_circuit_hash": first["logical_signature"],
            "physical_circuit_hash": first["physical_hash"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="folder to write runs.csv into")
    args = parser.parse_args()
    rows = build()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "runs.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{len(rows)} runs, {len({row['circuit_group'] for row in rows})} circuit groups -> {target}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
