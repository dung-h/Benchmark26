#!/usr/bin/env python3
"""Read original CSV/SQLite/ZIP/JSON to check selected labels and source joins.

Does not regenerate logical candidates, prove equivalence, parse DAGs, train,
rebuild splits, execute quantum circuits, or modify any original input file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from benchmark26.paths import ARTIFACTS
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mali-root", type=Path)
    parser.add_argument("--qonductor-root", type=Path)
    parser.add_argument("--qpack-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt; never overwritten")
    args = parser.parse_args()
    if not any((args.mali_root, args.qonductor_root, args.qpack_root)):
        parser.error("Supply at least one original source root")
    require(not args.output.exists(), "Refusing existing receipt")
    require(STUDY not in args.output.resolve().parents, "Do not write into frozen scientific artifacts")
    manifest = json.loads((ROOT / "upstream/data_sources.json").read_text())
    panel = {r["canonical_observation_id"]: r for r in rows(STUDY / "cohort/panel.csv")}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in rows(STUDY / "cohort/targets.csv")}
    require(len(panel) == len(targets) == 3443 and panel.keys() == targets.keys(), "Cohort identity mismatch")
    details = {}
    checked_files = []
    maximum_difference = 0.0

    def compare(a, b, context):
        nonlocal maximum_difference
        a, b = float(a), float(b)
        require(math.isfinite(a) and math.isfinite(b), "Nonfinite: " + context)
        maximum_difference = max(maximum_difference, abs(a - b))
        require(math.isclose(a, b, rel_tol=0, abs_tol=1e-10), "Value differs: " + context)

    def asset_path(source, base, asset):
        path = base / asset["path"]
        require(path.is_file(), "Missing source file: " + str(path))
        require(
            path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"],
            "Source file hash/size differs; hydrate LFS if applicable: " + str(path),
        )
        checked_files.append({"source": source, "path": asset["path"], "sha256": asset["sha256"]})
        return path

    if args.mali_root:
        assets = {a["path"]: a for a in manifest["assets"] if a["source"] == "mali"}
        legacy = {
            r["canonical_observation_id"]: r for r in rows(ARTIFACTS / "real_qpu/qualified_logical_panel/panel.csv")
        }
        # Source ETL order is Osaka CSV then Kyoto CSV; IDs are positional in this locked order.
        # Runtime values are checked AFTER this join; they are never used as join keys.
        original = []
        for backend in ("osaka", "kyoto"):
            path = asset_path("mali", args.mali_root, assets[f"data/{backend}_time_taken.csv"])
            original.extend((backend, row) for row in rows(path))
        require(len(original) == 340, "Original Ma-Li CSV count differs")
        qasm_hash = {}
        for ordinal, (backend, row) in enumerate(original):
            cid = f"mali_real_qpu|row{ordinal}"
            require(
                cid in panel and panel[cid]["backend"] == "ibm_" + backend and row["device"] == backend,
                "Ma-Li source row/backend join differs: " + cid,
            )
            compare(row["time_taken"], targets[cid], cid)
            require(float(panel[cid]["shots"]) == 1024, "Ma-Li archived shots metadata differs")
            name = f"data/quantum_circuits/{row['quantum_circuit']}.qasm"
            if name not in qasm_hash:
                qasm_hash[name] = digest(asset_path("mali", args.mali_root, assets[name]))
            require(qasm_hash[name] == legacy[cid]["source_qasm_sha256"], "Ma-Li logical QASM differs: " + cid)
        details["mali"] = {
            "status": "PASS",
            "observations": 340,
            "unique_named_qasm_files": len(qasm_hash),
            "join": "Pinned Osaka-then-Kyoto source row order, checked against backend and exact logical QASM digest.",
            "limit": "CSV values are the published run-aggregated labels; underlying individual runs are not recovered. Shots=1024 is retained metadata, not a field measured by these CSVs.",
        }

    if args.qonductor_root:
        assets = {a["path"]: a for a in manifest["assets"] if a["source"] == "qonductor"}
        database = asset_path("qonductor", args.qonductor_root, assets["data/database/quantum_scheduler.db"])
        archive = asset_path("qonductor", args.qonductor_root, assets["data/database/circuits.zip"])
        canonical = {
            r["canonical_row_id"]: r
            for r in rows(ARTIFACTS / "benchmark_v1/canonical_corpus_8767_20260926/qonductor_single_circuit_ibm.csv")
        }
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            db = {
                int(r["job_id"]): dict(r)
                for r in connection.execute("""
                SELECT j.id AS job_id, j.backend_name, j.shots, j.taken_time,
                       c.id AS circuit_id, c.filename, counts.n
                FROM job j JOIN circuit c ON c.job_id=j.id
                JOIN (SELECT job_id, COUNT(*) AS n FROM circuit GROUP BY job_id) counts ON counts.job_id=j.id
                WHERE counts.n=1""")
            }
        selected = [cid for cid, r in panel.items() if r["source_id"] == "qonductor_single_circuit_ibm"]
        checked_members = {}
        with zipfile.ZipFile(archive) as zipped:
            for cid in selected:
                row = canonical[cid]
                raw = json.loads(row["raw_row_json"])
                original = db[int(raw["job_id"])]
                require(
                    original["n"] == 1 and int(raw["circuit_db_id"]) == original["circuit_id"],
                    "Job/circuit FK differs: " + cid,
                )
                require(
                    original["backend_name"].replace("ibmq_", "ibm_") == panel[cid]["backend"],
                    "Backend differs: " + cid,
                )
                require(
                    original["filename"] == raw["archived_qasm_filename"] == row["qasm_path_or_member"],
                    "Archive member join differs: " + cid,
                )
                compare(original["taken_time"], targets[cid], cid)
                compare(original["shots"], panel[cid]["shots"], cid + "/shots")
                member = original["filename"]
                if member not in checked_members:
                    checked_members[member] = hashlib.sha256(zipped.read(member)).hexdigest()
                require(checked_members[member] == row["qasm_bytes_sha256"], "Physical QASM differs: " + cid)
        require(len(selected) == 3058, "Selected Qonductor count differs")
        details["qonductor"] = {
            "status": "PASS",
            "observations": len(selected),
            "archive_members_checked": len(checked_members),
            "join": "Database job.id -> circuit.job_id -> circuit.filename -> ZIP member.",
            "limit": "Checks selected job labels and submitted physical bytes. Does not re-prove logical-recipe qualification, hardware provenance, or eligibility of every excluded job.",
        }

    if args.qpack_root:
        assets = {
            a.get("workflow_hash"): a
            for a in manifest["assets"]
            if a["source"] == "qpack" and a["selection"] == "INCLUDED_WORKFLOW"
        }
        canonical = rows(ARTIFACTS / "benchmark_v1/canonical_corpus_8767_20260926/qpack_mcp.csv")
        buckets = collections.defaultdict(list)
        for row in canonical:
            raw = json.loads(row["raw_row_json"])
            workflow = hashlib.sha256(raw["workflow_id"].encode()).hexdigest()
            if workflow in assets:
                buckets[workflow].append((int(raw["optimizer_iteration"]), row, raw))
        total = 0
        for workflow, a in sorted(assets.items()):
            data = json.loads(asset_path("qpack", args.qpack_root, a).read_text())
            require(
                data["Problem"] == "MCP" and int(data["P"]) == 3 and int(data["Shots"]) == 4096,
                "QPack configuration differs",
            )
            durations = data["Circuit execution durations [ms]"]
            rr = sorted(buckets[workflow], key=lambda x: x[0])
            require(len(rr) == len(durations) == int(data["Optimizer iterations"]), "QPack timing array length differs")
            require([x[0] for x in rr] == list(range(len(rr))), "QPack evaluation ordinal differs")
            for ordinal, row, raw in rr:
                require(
                    raw["qpu_id_hex"] == data["QPU"] and int(row["width_qubits"]) == int(data["Size"]),
                    "QPack workload identity differs",
                )
                compare(durations[ordinal], row["target_native"], workflow + "/native")
                compare(float(durations[ordinal]) / 1000, row["target_seconds"], workflow + "/seconds")
            cid = "qpack_workflow_mean|" + workflow
            compare(statistics.mean(float(x) / 1000 for x in durations), targets[cid], cid)
            require(int(panel[cid]["constituent_count"]) == len(rr), "QPack constituent count differs")
            total += len(rr)
        require(len(assets) == 45 and total == 3854, "QPack coverage differs")
        details["qpack"] = {
            "status": "PASS",
            "workflow_means": 45,
            "constituent_timings": total,
            "join": "Pinned source workflow path/hash and optimizer-evaluation ordinal; array parity before averaging.",
            "limit": "Does not recover per-evaluation angles, prove representative-circuit fidelity, or validate the excluded mismatch candidate.",
        }

    receipt = {
        "status": "PASS_REQUESTED_SOURCE_LABEL_CHECKS",
        "sources": details,
        "checked_input_files": checked_files,
        "max_compared_numeric_difference": maximum_difference,
        "absolute_comparison_tolerance": 1e-10,
        "script_sha256": digest(Path(__file__)),
        "acquisition_manifest_sha256": digest(ROOT / "upstream/data_sources.json"),
        "cohort_panel_sha256": digest(STUDY / "cohort/panel.csv"),
        "cohort_targets_sha256": digest(STUDY / "cohort/targets.csv"),
        "scope": "Selected original labels/source joins only; not a full source-to-features/training reproduction.",
        "training_quantum_execution_or_canonical_modification": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "checked_input_files"}, indent=2))


if __name__ == "__main__":
    main()
