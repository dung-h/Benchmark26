#!/usr/bin/env python3
"""Summarize the ready-physical-input timing of the Qiskit and QCRE estimators.

Inputs (outside the repository):

- ``--gate``: the correctness gate directory (``receipt.json``, ``tasks/``).
  Each task is one unique (saved circuit IR digest, nominal snapshot) pair. Its
  Qiskit and QCRE outputs were checked against the accuracy-route values of
  every observation that maps to it.
- ``--timing``: the timing directory (``protocol.json``, ``tasks/``,
  ``status_*.json``). It holds one record per gate task, with raw samples in
  nanoseconds for four components: Qiskit ``estimate_duration``, the
  Qiskit-to-BQSKit conversion, the QCRE estimator, and conversion + estimator
  timed as one interval.
- The study bundle (``B26_ARTIFACTS``): the cohort panel and the representation
  rows that map each observation to its task.

The timed boundary starts from a parsed physical circuit in memory and a
prepared target or duration map. It excludes QASM parsing and transpilation.
Samples were taken with eight concurrent single-core workers; summaries
describe that condition only.

For each task, the value of a component is the median of its samples. Every
observation takes its task's value. The script reports, per source and pooled:

- the group-balanced mean (mean over groups of the group's mean);
- the observation-level median and P90.

``qcre_total`` is summarized from its own samples, never as a sum of component
medians. The script writes a summary, anonymous group means for
``verify_ready_analytical.py``, and a receipt of the checks; it refuses to
overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.integrity import sha256_file as sha
from benchmark26.paths import ARTIFACTS
from benchmark26.tables import read_csv as read

PANEL = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv"
REPRESENTATIONS = ARTIFACTS / "benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
COMPONENTS = ("qiskit", "qcre_conversion", "qcre_estimator", "qcre_total")
EXPECTED = {"observations": 3443, "groups": 203, "tasks": 1452, "workers": 8}


def load(gate: Path, timing: Path) -> tuple[dict, list[str]]:
    """Load the timing records and check them against the correctness gate.

    Verifies the gate receipt and the protocol hash, that all workers finished, that every gate task
    was timed with the expected number of positive samples, that the total equals conversion plus
    estimator in every sample, and that every observation maps to a timed task. Returns the data and
    the list of problems."""
    problems = []
    receipt = json.loads((gate / "receipt.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("observations") != EXPECTED["observations"]:
        problems.append("gate_receipt_not_pass")
    protocol_path = timing / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha(protocol_path)
    if protocol.get("gate_receipt_sha256") != sha(gate / "receipt.json"):
        problems.append("protocol_names_a_different_gate_receipt")
    samples_per_task = int(protocol["samples_per_task"])
    workers = int(protocol["worker_count"])
    statuses = sorted(timing.glob("status_*.json"))
    if len(statuses) != workers or any(json.loads(p.read_text()).get("status") != "COMPLETE" for p in statuses):
        problems.append(f"worker_status_incomplete:{len(statuses)}_of_{workers}")

    gate_tasks = {p.stem: json.loads(p.read_text()) for p in (gate / "tasks").glob("*.json")}
    timed = {p.stem: json.loads(p.read_text()) for p in (timing / "tasks").glob("*.json")}
    if len(gate_tasks) != EXPECTED["tasks"]:
        problems.append(f"gate_task_count:{len(gate_tasks)}")
    if set(timed) != set(gate_tasks):
        problems.append(f"timed_tasks_differ_from_gate:missing={len(set(gate_tasks) - set(timed))}")
    values = {}
    for key, record in timed.items():
        ref = gate_tasks.get(key, {})
        if record.get("protocol_sha256") != protocol_sha:
            problems.append(f"protocol_hash:{key}")
        if (record.get("task"), record.get("representative_id"), record.get("snapshot"), record.get("source")) != (
            key,
            ref.get("representative_id"),
            ref.get("snapshot"),
            ref.get("source"),
        ):
            problems.append(f"task_identity:{key}")
        samples = record.get("samples_nanoseconds", {})
        if any(len(samples.get(c, [])) != samples_per_task or min(samples.get(c, [0])) <= 0 for c in COMPONENTS):
            problems.append(f"sample_count_or_value:{key}")
            continue
        joint = [a + b for a, b in zip(samples["qcre_conversion"], samples["qcre_estimator"])]
        if joint != samples["qcre_total"]:
            problems.append(f"total_is_not_conversion_plus_estimator_per_sample:{key}")
        values[key] = {c: float(statistics.median(samples[c])) for c in COMPONENTS}

    representation = {r["canonical_row_id"]: r for r in read(REPRESENTATIONS)}
    observations = []
    for row in read(PANEL):
        rep = representation[row["feature_source_observation_id"]]
        key = hashlib.sha256(
            (rep["compiled_or_submitted_digest"] + "|" + rep["target_snapshot_id"]).encode()
        ).hexdigest()
        if key not in values:
            problems.append(f"observation_without_task:{row['canonical_observation_id']}")
            continue
        if gate_tasks[key]["source"] != row["source_id"]:
            problems.append(f"source_mismatch:{row['canonical_observation_id']}")
        observations.append(
            {"id": row["canonical_observation_id"], "group": row["group_id"], "source": row["source_id"], "task": key}
        )
    if len(observations) != EXPECTED["observations"] or len({o["group"] for o in observations}) != EXPECTED["groups"]:
        problems.append("observation_or_group_count")
    data = {
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "gate_receipt_sha256": sha(gate / "receipt.json"),
        "values": values,
        "observations": observations,
        "full_ir_equal_tasks": sum(bool(t.get("same_full_ir")) for t in gate_tasks.values()),
        "tasks_by_source": dict(collections.Counter(t["source"] for t in gate_tasks.values())),
    }
    return data, problems


def summarize(data: dict) -> tuple[list[dict], list[dict]]:
    """Return the per-group means and the summary rows (group-balanced mean, observation median and P90)."""
    observations, values = data["observations"], data["values"]
    groups = collections.defaultdict(list)
    for o in observations:
        groups[o["group"]].append(o)
    names = sorted(groups)
    group_rows = []
    for index, name in enumerate(names):
        members = groups[name]
        group_rows.append(
            {
                "group": f"g{index:03d}",
                "source": members[0]["source"],
                "observations": len(members),
                **{f"{c}_ns": float(np.mean([values[o["task"]][c] for o in members])) for c in COMPONENTS},
            }
        )
    summary = []
    for scope in ["all"] + sorted({o["source"] for o in observations}):
        chosen = [o for o in observations if scope == "all" or o["source"] == scope]
        chosen_groups = [g for g in group_rows if scope == "all" or g["source"] == scope]
        for component in COMPONENTS:
            per_observation = np.array([values[o["task"]][component] for o in chosen])
            summary.append(
                {
                    "scope": scope,
                    "component": component,
                    "observations": len(chosen),
                    "groups": len(chosen_groups),
                    "tasks": len({o["task"] for o in chosen}),
                    "group_balanced_mean_ms": float(np.mean([g[f"{component}_ns"] for g in chosen_groups])) / 1e6,
                    "observation_median_ms": float(np.median(per_observation)) / 1e6,
                    "observation_p90_ms": float(np.quantile(per_observation, 0.9)) / 1e6,
                }
            )
    return group_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as a CSV file."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Summarize the ready-input timing and write the summary, the anonymous group means and a receipt.

    Exits with status 1 and prints the problems if any check fails. `--output` must not exist yet."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the summary")
    args = parser.parse_args()
    data, problems = load(args.gate, args.timing)
    receipt = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems[:50],
        "problem_count": len(problems),
        "protocol_sha256": data["protocol_sha256"],
        "gate_receipt_sha256": data["gate_receipt_sha256"],
        "tasks_timed": len(data["values"]),
        "tasks_by_source": data["tasks_by_source"],
        "observations_mapped": len(data["observations"]),
        "gate_tasks_with_identical_full_ir": data["full_ir_equal_tasks"],
        "boundary": data["protocol"]["scope"],
        "worker_count": data["protocol"]["worker_count"],
        "warmups": data["protocol"]["warmups"],
        "samples_per_task": data["protocol"]["samples_per_task"],
        "task_value": "median of the task's samples; each observation takes its task's value",
        "group_balanced_mean": "mean over groups of the group mean of observation values",
    }
    if problems:
        print(json.dumps(receipt, indent=2))
        return 1
    group_rows, summary = summarize(data)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    write_csv(args.output / "ready_analytical_summary.csv", summary)
    write_csv(args.output / "ready_analytical_group_means.csv", group_rows)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "tasks_timed", "observations_mapped")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
