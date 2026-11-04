#!/usr/bin/env python3
"""Check every reported route has explicit per-evaluation coverage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import json
from pathlib import Path

from benchmark26.integrity import sha256_file as sha
from benchmark26.paths import ARTIFACTS
from benchmark26.tables import read_csv as read

R = Path(__file__).resolve().parents[2]
N = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
S = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"


run_rows = read(S / "cohort/panel.csv")
p = {r["canonical_observation_id"]: r for r in run_rows}
ids = set(p)
rows = read(N / "coverage.csv")
keys = [(r["evaluation"], r["case"], r["method_id"]) for r in rows]
assert len(keys) == len(set(keys))
pred = read(N / "predictions.csv")
methods = {r["method_id"] for r in pred}
assert len(methods) == 22
cases = {("E1", "ALL"): sorted(ids), ("E2", "ALL"): sorted(ids)}
for b in sorted({r["backend"] for r in run_rows}):
    t = sorted(i for i in ids if p[i]["backend"] == b)
    if any(r["evaluation"] == "E3" and r["case"] == b for r in read(N / "metrics.csv")):
        cases[("E3", b)] = t
assert len(cases) == 12
unitless = {
    m
    for m in methods
    if m.startswith("NOTADS_")
    and (m.endswith(("SCORE", "SCORE_SHOT_SCALED", "FILTERED_DEPTH", "FILTERED_DEPTH_SHOT_SCALED")))
}
expected = methods | {"R0", "C0"}
for (e, c), ii in cases.items():
    rr = [r for r in rows if r["evaluation"] == e and r["case"] == c]
    assert {r["method_id"] for r in rr} == expected, (e, c, len(rr), len(expected))
    for r in rr:
        assert int(r["rows"]) == len(ii)
        assert int(r["groups"]) == len({p[i]["group_id"] for i in ii})
        assert int(r["sources"]) == len({p[i]["source_id"] for i in ii})
        want = "FULL_COMMON_COVERAGE_SCORE_ONLY_UNITLESS" if r["method_id"] in unitless else "FULL_COMMON_COVERAGE"
        assert r["status"] == want, (r["method_id"], r["status"], want)
out = {
    "status": "PASS_EXPLICIT_METHOD_COVERAGE",
    "cases": len(cases),
    "scored_methods_per_case": len(expected),
    "coverage_rows": len(rows),
    "method_ids": sorted(methods),
    "all_methods_full_3443_row_coverage_per_evaluation": "PASS",
    "unitless_scores_not_metric_scored": "PASS",
    "coverage_sha256": sha(N / "coverage.csv"),
    "script_sha256": sha(__file__),
}
(N / "coverage_validation_receipt.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
