#!/usr/bin/env python3
"""Verify new Target-path values against saved Qiskit estimates on exact shared inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import csv
import json
import math
from collections import Counter
from pathlib import Path

from benchmark26.integrity import sha256_file as sha
from benchmark26.paths import ARTIFACTS
from benchmark26.tables import read_csv as read

R = Path(__file__).resolve().parents[2]
N = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"


raw = {(x["graph_file"], x["snapshot_id"]): x for x in read(N / "raw_features.csv")}
old = {
    (x["graph_sha256"], x["snapshot_id"]): x
    for x in read(ARTIFACTS / "postfreeze_notads_hyperq_track_a_v1/features.csv")
}
lin = read(N / "lineage.csv")
runs_by_id = {
    x["canonical_observation_id"]: x
    for x in read(ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv")
}
out = []
for l in lin:
    k = (l["graph_file_sha256"], l["snapshot_id"])
    if k not in old:
        continue
    n = raw[(l["graph_file"], l["snapshot_id"])]
    o = old[k]
    a = float(n["hyperq_full_seconds"])
    b = float(o["qiskit_saved_duration_seconds"])
    d = abs(a - b)
    if not math.isclose(a, b, rel_tol=0, abs_tol=0):
        raise AssertionError((l["canonical_observation_id"], a, b))
    out.append(
        {
            "canonical_observation_id": l["canonical_observation_id"],
            "source_id": runs_by_id[l["canonical_observation_id"]]["source_id"],
            "representation_observation_id": l["representation_observation_id"],
            "graph_file_sha256": l["graph_file_sha256"],
            "snapshot_id": l["snapshot_id"],
            "saved_qiskit_estimate_seconds": repr(b),
            "current_weighted_path_seconds": repr(a),
            "absolute_difference_seconds": repr(d),
            "status": "EXACT",
        }
    )
if len(out) != 615:
    raise AssertionError(len(out))
with open(N / "qiskit_parity_rows.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out)
rec = {
    "status": "PASS_EXACT_QISKIT_ESTIMATE_PARITY_ON_SHARED_GRAPH_SNAPSHOT_INPUTS",
    "rows": len(out),
    "unique_graph_snapshot_pairs": len({(x["graph_file_sha256"], x["snapshot_id"]) for x in out}),
    "source_rows": dict(Counter(x["source_id"] for x in out)),
    "max_absolute_difference_seconds": 0.0,
    "comparison": "saved Qiskit estimate_duration-derived nominal path vs current duration-weighted DAG path",
    "unsupported_rows_outside_shared_population": 3443 - len(out),
    "extension_note": "Other rows were computed from the exact pinned native DAG and frozen Target via the same recurrence; they have no exact matching row in the older Qiskit estimate cache.",
    "rows_sha256": sha(N / "qiskit_parity_rows.csv"),
}
(N / "qiskit_parity_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
