#!/usr/bin/env python3
"""Independent source-table → cohort/mean/group replay; no circuit execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from core import OUT, atomic_csv, atomic_json, rows, sha

from benchmark26.paths import COHORT_PLAN, recorded


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "validation/dataset_lineage",
        help="new output directory (default: inside the study bundle)",
    )
    dest = parser.parse_args().output
    if dest.exists():
        raise RuntimeError("Refuse existing lineage output")
    panel = {r["canonical_observation_id"]: r for r in rows(OUT / "cohort/panel.csv")}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in rows(OUT / "cohort/targets.csv")}
    proposed = rows(COHORT_PLAN / "proposed_observations.csv")
    paths = [
        "artifacts/real_qpu/qualified_logical_panel/panel.csv",
        "artifacts/benchmark_v1/canonical_corpus_8767_20260926/qonductor_single_circuit_ibm.csv",
        "artifacts/benchmark_v1/canonical_corpus_8767_20260926/qpack_mcp.csv",
        "artifacts/postclosure_qpack_angle_runtime_audit/v1/workflow_final_parameters.csv",
        "artifacts/postfreeze_qonductor_generator_version_semantics_v1/semantics_v2_legacy_parser/observation_semantic_results.csv",
    ]
    old = rows(recorded(paths[0]))
    qc = {r["canonical_row_id"]: r for r in rows(recorded(paths[1]))}
    qp = rows(recorded(paths[2]))
    workflows = {r["workflow_hash"] for r in rows(recorded(paths[3]))}
    accepted = [
        r
        for r in rows(recorded(paths[4]))
        if r["version"] == "1.0.3"
        and r["scope"] == "ADDITIONAL_2840_CANDIDATE"
        and r["semantic_status"] == "PASS_PREPARED_READOUT_STATE"
    ]
    assert len(accepted) == len({r["canonical_observation_id"] for r in accepted}) == 2828
    expected = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in old if r["source_id"] != "qpack_mcp"}
    assert len(expected) == 570
    for r in accepted:
        c = r["canonical_observation_id"]
        assert c not in expected
        origin = qc[c]
        assert r["physical_qasm_sha256"] == origin["qasm_bytes_sha256"]
        assert origin["target_native_unit"] == "seconds"
        assert float(origin["target_seconds"]) == float(origin["target_native"])
        expected[c] = float(origin["target_seconds"])
        assert float(panel[c]["shots"]) == float(origin["shots"])
        assert panel[c]["backend"] == origin["backend"].replace("ibmq_", "ibm_")
    buckets = collections.defaultdict(list)
    for r in qp:
        raw = json.loads(r["raw_row_json"])
        w = hashlib.sha256(raw["workflow_id"].encode()).hexdigest()
        if w in workflows:
            buckets[w].append(r)
    assert len(buckets) == 45
    constituents = []
    for w, rr in sorted(buckets.items()):
        cid = "qpack_workflow_mean|" + w
        assert len({(r["backend"], r["shots"]) for r in rr}) == 1
        expected[cid] = statistics.mean(float(r["target_seconds"]) for r in rr)
        assert int(panel[cid]["constituent_count"]) == len(rr)
        for r in rr:
            assert r["target_native_unit"] == "milliseconds"
            assert math.isclose(
                float(r["target_seconds"]), float(r["target_native"]) / 1000, rel_tol=1e-14, abs_tol=1e-14
            )
            constituents.append(
                {
                    "workflow_mean_id": cid,
                    "constituent_id": r["canonical_row_id"],
                    "target_seconds": r["target_seconds"],
                    "target_native": r["target_native"],
                    "target_native_unit": r["target_native_unit"],
                    "backend": r["backend"],
                    "shots": r["shots"],
                }
            )
    assert len(constituents) == 3854 and len(expected) == 3443 and set(expected) == set(targets) == set(panel)
    assert all(expected[c] == targets[c] for c in expected)
    # Build connected components with key expansion, independent of the producer union-find.
    bykey = collections.defaultdict(set)
    keys = {}
    for r in proposed:
        c = r["observation_id"]
        assert float(r["target_seconds"]) == targets[c]
        keys[c] = [
            (kind, r[col])
            for kind, col in [
                ("logical", "logical_signature"),
                ("legacy", "legacy_group"),
                ("physical", "physical_hash"),
            ]
            if r[col]
        ]
        for k in keys[c]:
            bykey[k].add(c)
    assert set(keys) == set(panel)
    pending = set(panel)
    components = []
    while pending:
        todo = [min(pending)]
        members = set()
        used_keys = set()
        while todo:
            c = todo.pop()
            if c in members:
                continue
            members.add(c)
            for k in keys[c]:
                if k not in used_keys:
                    used_keys.add(k)
                    todo.extend(bykey[k] - members)
        pending -= members
        digest = hashlib.sha256(json.dumps(sorted(members), separators=(",", ":")).encode()).hexdigest()
        assert all(panel[c]["group_id"] == digest for c in members)
        components.append(members)
    assert len(components) == 203
    dest.mkdir(parents=True)
    atomic_csv(dest / "qpack_mean_constituents.csv", constituents)
    receipt = {
        "status": "PASS_SOURCE_TABLE_TO_COHORT_REPLAY",
        "rows": 3443,
        "groups": 203,
        "legacy_non_qpack_rows": 570,
        "additional_qonductor_qualified_rows": 2828,
        "qpack_workflow_means": 45,
        "qpack_constituents": 3854,
        "all_target_values_exactly_replayed": True,
        "group_components_independently_reconstructed": True,
        "qonductor_physical_hash_join_verified": True,
        "qpack_ms_to_seconds_verified": True,
        "no_training_simulation_or_label_changes": True,
        "script_sha256": sha(Path(__file__)),
        "source_table_hashes": {p: sha(recorded(p)) for p in paths},
        "scope_limit": "Starts at retained canonical source tables and qualification evidence, not raw external archives. Logical signatures taken from the locked preparation ledger; no circuit/generator/quantum equivalence recomputation.",
    }
    atomic_json(dest / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
