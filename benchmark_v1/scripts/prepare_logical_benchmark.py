#!/usr/bin/env python3
"""Build C7-C10 logical-input sidecars and supplemental grouped splits; no fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import mali_full_features as mf
import run_recovery_c133_c135_v3 as c135

CANON = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
REP = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
QPACK = ROOT / "artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/qpack_mcp_structural_rows.csv"
RECOVERY = ROOT / "artifacts/real_qpu/logical_recovery/recovery_rows.csv"
CANDIDATES = ROOT / "artifacts/real_qpu/logical_recovery/candidate_recipe_manifest.json"
REGISTRY = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"
CONTRACT = ROOT / "benchmark_v1/protocol/logical_input_review.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_canonical_without_target(path: Path):
    required = {"canonical_row_id", "source_id", "source_row_index", "backend",
                "qasm_path_or_member", "qasm_bytes_sha256", "workflow_group_hash"}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        selected = [header.index(name) for name in required]
        names = [header[i] for i in selected]
        return [dict(zip(names, (row[i] for i in selected))) for row in reader]


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def canonical_instruction_digest(circuit, *, include_parameters=True) -> str:
    """Hash exact model-visible circuit structure, preserving wires and parameters."""
    body = {"qubits": circuit.num_qubits, "clbits": circuit.num_clbits,
            "instructions": []}
    for inst in circuit.data:
        op = inst.operation
        params = []
        for value in getattr(op, "params", ()):
            try:
                params.append(float(value))
            except (TypeError, ValueError):
                params.append(str(value))
        item_body = {
            "name": str(op.name), "qargs": [circuit.find_bit(q).index for q in inst.qubits],
            "cargs": [circuit.find_bit(b).index for b in inst.clbits],
            "condition": str(getattr(op, "condition", None)),
        }
        if include_parameters:
            item_body["params"] = params
        body["instructions"].append(item_body)
    return digest(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/real_qpu/logical_inputs")
    ap.add_argument("--stage", choices=["sidecar", "features", "groups", "splits", "validate", "all"], default="all")
    ap.add_argument("--max-rows", type=int, help="bounded canary only; cannot produce a release manifest")
    args = ap.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists() and any(out.iterdir()) and args.stage not in {"sidecar", "groups", "splits", "validate"}:
        raise SystemExit(f"output directory is nonempty; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    canonical, representation, recovery = read_canonical_without_target(CANON), read(REP), read(RECOVERY)
    rec_by_id = {r["canonical_observation_id"]: r for r in recovery}
    rep_by_id = {r["canonical_row_id"]: r for r in representation}
    if len(canonical) != 8767 or len(rep_by_id) != 8767 or len(rec_by_id) != 4482:
        raise ValueError("input_row_count_or_identity_mismatch")
    with REGISTRY.open(newline="", encoding="utf-8") as f:
        registry_rows = list(csv.DictReader(f))
    snapshots = {}
    for r in registry_rows:
        conf = Path(r["configuration_path"])
        if digest(conf.read_bytes()) != r["configuration_sha256"]:
            raise ValueError(f"snapshot_config_hash_mismatch:{r['snapshot_id']}")
        props_path = Path(r["properties_path"])
        if digest(props_path.read_bytes()) != r["properties_sha256"]:
            raise ValueError(f"snapshot_properties_hash_mismatch:{r['snapshot_id']}")
        config = json.loads(conf.read_text())
        if int(config["n_qubits"]) != int(r["num_qubits"]):
            raise ValueError(f"snapshot_capacity_mismatch:{r['snapshot_id']}")
        for alias in r["backend_observed_aliases"].split(";"):
            snapshots[alias] = {**r, "actual_capacity": str(config["n_qubits"])}
    recovery_side = []
    for c in canonical:
        rid = c["canonical_row_id"]
        r = rec_by_id.get(rid)
        rep = rep_by_id[rid]
        snap = snapshots.get(c["backend"])
        expected = int(snap["actual_capacity"]) if snap else ""
        if not snap or rep["target_snapshot_id"] != snap["snapshot_id"]:
            raise ValueError(f"backend_snapshot_identity_mismatch:{rid}")
        status = "source_has_logical_input" if c["source_id"] in {"mali_real_qpu", "qpack_mcp"} else "unavailable"
        if c["source_id"] == "qonductor_single_circuit_ibm":
            status = ("archive_supported_recipe_adaptation" if r and r["instance_status"] == "archive_resolved_recipe"
                      else "candidate_recipe_sensitivity_only" if r and r.get("generated_candidate_qasm_sha256")
                      else "unavailable")
        recovery_side.append({"canonical_observation_id": rid, "source_id": c["source_id"],
                              "backend": c["backend"], "snapshot_id": rep["target_snapshot_id"],
                              "configuration_sha256": rep["target_configuration_sha256"],
                              "properties_sha256": rep["target_properties_sha256"],
                              "capacity_from_snapshot": expected,
                              "legacy_capacity_field": (r or {}).get("backend_capacity_qubits", ""),
                              "legacy_capacity_matches_snapshot": str(bool(expected) and str(expected) == str((r or {}).get("backend_capacity_qubits", ""))).lower(),
                              "logical_input_tier": status,
                              "candidate_widths": (r or {}).get("candidate_logical_widths", ""),
            "runtime_label_accessed": "false"})
    if args.stage in {"sidecar", "features", "all"}:
        write_csv(out / "eligibility.csv", recovery_side, list(recovery_side[0]))
    if args.stage == "sidecar":
        return

    if args.stage == "splits":
        attempts = read(out / "feature_attempts.csv")
        group_rows = read(out / "model_input_groups.csv")
        group_by_id = {x["canonical_observation_id"]: x for x in group_rows}
        for r in attempts:
            r["model_input_group_digest"] = group_by_id[r["canonical_observation_id"]]["model_input_group_digest"]
        build_splits(out, canonical, recovery_side, attempts)
        return

    if args.stage == "groups":
        materialize_model_input_groups(out, canonical, representation, rec_by_id)
        return

    if args.stage == "validate":
        validate_outputs(out, canonical)
        return

    if args.stage not in {"features", "all"}:
        return

    import numpy as np
    from qiskit import qasm2, qasm3
    from zipfile import ZipFile
    extractor, source_pins = mf.load_pinned_global_extractor()
    cand_manifest = json.loads(CANDIDATES.read_text())
    cand_path = CANDIDATES.parent
    cand_by_digest = {r["qasm_sha256"]: r for r in cand_manifest["records"]}
    qpack_rows = {r["source_row_index"]: r for r in read(QPACK)}
    qond_zip = ZipFile(c135.QONDUCTOR_ZIP)
    # Cache source circuit/global features. Node calibration is context-specific.
    input_cache, global_cache, graph_cache, properties_cache = {}, {}, {}, {}
    attempts, global_rows = [], []
    mf.OUT_ROOT = out
    qasm3_mod = qasm3
    source_iter = canonical[:args.max_rows] if args.max_rows else canonical
    for i, c in enumerate(source_iter):
        rid, source = c["canonical_row_id"], c["source_id"]
        rep, side = rep_by_id[rid], next(x for x in recovery_side if x["canonical_observation_id"] == rid)
        tier, qasm_bytes, circuit, inp_digest = side["logical_input_tier"], None, None, ""
        reason = ""
        try:
            if tier == "unavailable":
                raise ValueError("logical_recipe_not_qualified")
            if source == "qonductor_single_circuit_ibm":
                rr = rec_by_id[rid]
                item = cand_by_digest.get(rr.get("generated_candidate_qasm_sha256", ""))
                if not item:
                    raise ValueError("candidate_qasm_missing")
                path = cand_path / item["path"]
                qasm_bytes = path.read_bytes()
                if digest(qasm_bytes) != item["qasm_sha256"]:
                    raise ValueError("candidate_qasm_hash_mismatch")
                circuit = qasm2.loads(qasm_bytes.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                inp_digest = canonical_instruction_digest(circuit)
            elif source == "mali_real_qpu":
                path = c135.MALI_QASM_ROOT / c["qasm_path_or_member"]
                qasm_bytes = path.read_bytes()
                if digest(qasm_bytes) != c["qasm_bytes_sha256"]:
                    raise ValueError("mali_logical_qasm_hash_mismatch")
                circuit, _ = mf.parse_qasm(qasm_bytes)
                inp_digest = canonical_instruction_digest(circuit)
            else:
                recipe = json.loads(rep["source_recipe_json"])
                circuit = c135.qpack_circuit(int(recipe["size"]), int(recipe["p"]))
                inp_digest = canonical_instruction_digest(circuit)
                qasm_bytes = qasm3.dumps(circuit).encode()
                tier = "qpack_reconstruction_qualified"
            cache_key = digest((source + ":" + inp_digest).encode())
            model_group_digest = canonical_instruction_digest(circuit, include_parameters=False)
            if cache_key not in global_cache:
                global_cache[cache_key] = mf.extract_global_features(circuit, extractor)
                input_cache[cache_key] = circuit
            features = global_cache[cache_key]
            prop_hash = rep["target_properties_sha256"]
            if prop_hash not in properties_cache:
                prop_path = Path(next(s["properties_path"] for s in registry_rows
                                      if s["snapshot_id"] == rep["target_snapshot_id"]))
                raw_props = prop_path.read_bytes()
                if digest(raw_props) != prop_hash:
                    raise ValueError("snapshot_properties_hash_mismatch")
                properties_cache[prop_hash] = json.loads(raw_props)
            props = properties_cache[prop_hash]
            calibration = {}
            for wi in range(circuit.num_qubits):
                vals = {x["name"]: x for x in props["qubits"][wi]}
                if not {"T1", "T2"} <= set(vals):
                    raise ValueError(f"missing_T1_T2_logical_wire_{wi}")
                unit_scale = {"s": 1e6, "ms": 1e3, "us": 1, "µs": 1, "ns": 1e-3}
                for key in ("T1", "T2"):
                    u = vals[key].get("unit", "").lower().replace("μ", "u")
                    if u not in unit_scale or float(vals[key]["value"]) <= 0:
                        raise ValueError(f"invalid_{key}_logical_wire_{wi}")
                calibration[wi] = (float(vals["T1"]["value"]) * unit_scale[vals["T1"]["unit"].lower().replace("μ", "u")],
                                   float(vals["T2"]["value"]) * unit_scale[vals["T2"]["unit"].lower().replace("μ", "u")])
            graph_key = (cache_key, rep["target_properties_sha256"])
            if graph_key not in graph_cache:
                graph, meta = mf.compact_graph_record(circuit, calibration)
                name, graph_hash = mf.save_sparse_graph(graph, digest(json.dumps(graph_key).encode())[:40])
                graph_cache[graph_key] = (name, graph_hash, meta)
            name, graph_hash, meta = graph_cache[graph_key]
            rec = {"canonical_observation_id": rid, "source_id": source, "status": "available",
                   "logical_input_tier": tier, "logical_input_digest": inp_digest,
                   "model_input_group_digest": model_group_digest,
                   "source_qasm_sha256": digest(qasm_bytes), "candidate_or_recipe_hash": digest(qasm_bytes),
                   "snapshot_id": rep["target_snapshot_id"], "properties_sha256": rep["target_properties_sha256"],
                   "backend_capacity": side["capacity_from_snapshot"], "allocated_width": circuit.num_qubits,
                   "active_width": meta["active_width"], "graph_file": name, "graph_sha256": graph_hash,
                   "node_count": meta["node_count"], "edge_count": meta["edge_count"], "reason": ""}
            rec.update({f"g{i:02d}": v for i, v in enumerate(features)})
        except Exception as exc:
            rec = {"canonical_observation_id": rid, "source_id": source, "status": "unavailable",
                   "logical_input_tier": tier, "logical_input_digest": inp_digest,
                   "reason": f"{type(exc).__name__}:{exc}"}
        attempts.append(rec)
        if i and i % 250 == 0:
            print(f"features {i}/{len(source_iter)} available={sum(x['status']=='available' for x in attempts)}", flush=True)
    fields = sorted({k for r in attempts for k in r})
    write_csv(out / "feature_attempts.csv", attempts, fields)
    meta = {"rows_attempted": len(attempts), "available": sum(r["status"] == "available" for r in attempts),
            "unavailable": sum(r["status"] == "unavailable" for r in attempts),
            "status_by_source": {s: dict(Counter(r["status"] for r in attempts if r["source_id"] == s))
                                 for s in {r["source_id"] for r in attempts}},
            "graph_files": len(graph_cache), "source_hashes": source_pins,
            "synthetic_T1_T2_used": False, "runtime_labels_read": False,
            "contract_sha256": digest(CONTRACT.read_bytes()), "feature_encoder_sha256": digest(Path(mf.__file__).read_bytes())}
    (out / "feature_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if args.stage != "all":
        return
    materialize_model_input_groups(out, canonical, representation, rec_by_id)
    group_by_id = {x["canonical_observation_id"]: x for x in read(out / "model_input_groups.csv")}
    for r in attempts:
        r["model_input_group_digest"] = group_by_id[r["canonical_observation_id"]]["model_input_group_digest"]
    build_splits(out, canonical, recovery_side, attempts)
    validate_outputs(out, canonical)


def build_splits(out, canonical, eligibility, attempts):
    parent = list(range(len(canonical)))
    index = {r["canonical_row_id"]: i for i, r in enumerate(canonical)}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union_group(rows):
        ids = [index[x] for x in rows]
        for x in ids[1:]:
            a, b = find(ids[0]), find(x)
            if a != b: parent[b] = a
    by_exact, by_workflow, by_candidate = {}, {}, {}
    for c in canonical:
        if c["qasm_bytes_sha256"]:
            by_exact.setdefault(c["qasm_bytes_sha256"], []).append(c["canonical_row_id"])
        if c["workflow_group_hash"]:
            by_workflow.setdefault(c["workflow_group_hash"], []).append(c["canonical_row_id"])
    for group in (by_exact, by_workflow):
        for ids in group.values(): union_group(ids)
    for r in attempts:
        if r["status"] == "available":
            by_candidate.setdefault(r["model_input_group_digest"], []).append(r["canonical_observation_id"])
    for ids in by_candidate.values(): union_group(ids)
    components = {}
    for c in canonical:
        components.setdefault(find(index[c["canonical_row_id"]]), []).append(c["canonical_row_id"])
    comp_id = {rid: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
               for ids in components.values() for rid in ids}
    component_rows = {comp_id[ids[0]]: ids for ids in components.values()}
    ids = [c["canonical_row_id"] for c in canonical]
    if len(components) < 5:
        raise ValueError("fewer_than_five_outer_components")
    row_source = {c["canonical_row_id"]: c["source_id"] for c in canonical}
    def assign(component_rows, seed, folds=5):
        """Greedy source-stratified group assignment; tie breaks use pinned hash seed."""
        labels = sorted(set(row_source.values()))
        totals = Counter(row_source[rid] for rows in component_rows.values() for rid in rows)
        loads = [{label: 0 for label in labels} for _ in range(folds)]
        assignment = {}
        ordered = sorted(component_rows, key=lambda g: (
            -max(Counter(row_source[rid] for rid in component_rows[g])[s] / totals[s]
                 for s in labels if totals[s]),
            -len(component_rows[g]), g))
        for group in ordered:
            vector = Counter(row_source[rid] for rid in component_rows[group])
            scored = []
            for fold in range(folds):
                cost = 0.0
                for f in range(folds):
                    for label in labels:
                        value = loads[f][label] + (vector[label] if f == fold else 0)
                        target = totals[label] / folds
                        cost += ((value - target) / max(target, 1.0)) ** 2
                tie = hashlib.sha256(f"{seed}:{group}:{fold}".encode()).hexdigest()
                scored.append((cost, tie, fold))
            chosen = min(scored)[2]
            assignment[group] = chosen
            for label in labels: loads[chosen][label] += vector[label]
        return assignment, loads
    group_assignment, outer_loads = assign(component_rows, 42)
    fold_by_id = {rid: group_assignment[comp_id[rid]] for rid in ids}
    outer_rows = [{"canonical_observation_id": rid, "group_id": comp_id[rid], "outer_fold": fold_by_id[rid]}
                  for rid in ids]
    inner_rows = []
    inner_counts = {}
    inner_folds = 4
    for outer_fold in range(5):
        train_ids = [rid for rid in ids if fold_by_id[rid] != outer_fold]
        train_components = {cid: members for cid, members in component_rows.items()
                            if group_assignment[cid] != outer_fold}
        if len(train_components) < inner_folds: raise ValueError(f"insufficient_inner_groups_outer_{outer_fold}")
        inner_assignment, inner_loads = assign(train_components, 42 + outer_fold + 1, folds=inner_folds)
        inner_counts[str(outer_fold)] = inner_loads
        for rid in train_ids:
            inner_rows.append({"canonical_observation_id": rid, "outer_fold": outer_fold,
                               "inner_fold": inner_assignment[comp_id[rid]],
                               "group_id": comp_id[rid]})
    write_csv(out / "logical_outer_splits.csv", outer_rows, list(outer_rows[0]))
    write_csv(out / "logical_inner_splits.csv", inner_rows, list(inner_rows[0]))
    manifest = {"rows": len(ids), "components": len(group_assignment), "outer_folds": 5,
                "split_seed": 42, "inner_folds": inner_folds, "inner_validation_fold": 0,
                "attempt_rows": len(attempts), "available_rows": sum(r["status"] == "available" for r in attempts),
                "component_rule": "union exact QASM, QPack workflow and parameter-invariant graph/global input digest; transitive closure",
                "assignment_rule": "deterministic greedy source-stratified grouped assignment; source labels only, no target/error used",
                "source_rows_per_outer_fold": {str(f): outer_loads[f] for f in range(5)},
                "source_rows_per_inner_fold": inner_counts,
                "all_row_ids_assigned_once": len(fold_by_id) == len(ids),
                "train_test_group_overlap": False,
                "contract_sha256": digest(CONTRACT.read_bytes())}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


def materialize_model_input_groups(out, canonical, representation, rec_by_id):
    from qiskit import qasm2
    attempts = {r["canonical_observation_id"]: r for r in read(out / "feature_attempts.csv")}
    rep_by_id = {r["canonical_row_id"]: r for r in representation}
    candidate_manifest = json.loads(CANDIDATES.read_text())
    candidates = {r["qasm_sha256"]: r for r in candidate_manifest["records"]}
    cache = {}
    result = []
    for i, c in enumerate(canonical):
        rid = c["canonical_row_id"]
        attempt = attempts[rid]
        row = {"canonical_observation_id": rid, "status": attempt["status"],
               "logical_input_digest": attempt.get("logical_input_digest", ""),
               "model_input_group_digest": ""}
        if attempt["status"] != "available":
            result.append(row)
            continue
        source = c["source_id"]
        if source == "qonductor_single_circuit_ibm":
            item = candidates.get(attempt["source_qasm_sha256"])
            if not item: raise ValueError(f"candidate_missing_for_group_digest:{rid}")
            key = ("qonductor", item["qasm_sha256"])
            if key not in cache:
                data = (CANDIDATES.parent / item["path"]).read_bytes()
                cache[key] = qasm2.loads(data.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        elif source == "mali_real_qpu":
            data_path = c135.MALI_QASM_ROOT / c["qasm_path_or_member"]
            data = data_path.read_bytes()
            if digest(data) != c["qasm_bytes_sha256"]: raise ValueError(f"mali_qasm_hash_mismatch:{rid}")
            key = ("mali", c["qasm_bytes_sha256"])
            if key not in cache: cache[key] = mf.parse_qasm(data)[0]
        else:
            recipe = json.loads(rep_by_id[rid]["source_recipe_json"])
            key = ("qpack", rep_by_id[rid]["source_representation_digest"])
            if key not in cache: cache[key] = c135.qpack_circuit(int(recipe["size"]), int(recipe["p"]))
        circuit = cache[key]
        exact = canonical_instruction_digest(circuit)
        if row["logical_input_digest"] and exact != row["logical_input_digest"]:
            raise ValueError(f"logical_input_digest_changed:{rid}")
        row["model_input_group_digest"] = canonical_instruction_digest(circuit, include_parameters=False)
        result.append(row)
        if i and i % 1000 == 0: print(f"group identities {i}/8767", flush=True)
    write_csv(out / "model_input_groups.csv", result, list(result[0]))
    print(f"model-input identities {sum(bool(r['model_input_group_digest']) for r in result)}/8767")


def validate_outputs(out, canonical):
    import math
    import numpy as np
    feat = read(out / "feature_attempts.csv")
    prior_feature_manifest = json.loads((out / "feature_manifest.json").read_text())
    outer = read(out / "logical_outer_splits.csv")
    inner = read(out / "logical_inner_splits.csv")
    expected = {r["canonical_row_id"] for r in canonical}
    if len(feat) != 8767 or {r["canonical_observation_id"] for r in feat} != expected:
        raise ValueError("feature_denominator_or_ids_mismatch")
    source_counts = Counter((r["source_id"], r["status"]) for r in feat)
    if source_counts != Counter({("mali_real_qpu", "available"): 340,
                                 ("qpack_mcp", "available"): 3945,
                                 ("qonductor_single_circuit_ibm", "available"): 3065,
                                 ("qonductor_single_circuit_ibm", "unavailable"): 1417}):
        raise ValueError(f"feature_source_coverage_mismatch:{source_counts}")
    for r in feat:
        if r["status"] == "available":
            values = [float(r[f"g{i:02d}"]) for i in range(51)]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"nonfinite_global:{r['canonical_observation_id']}")
            graph = out / r["graph_file"]
            if digest(graph.read_bytes()) != r["graph_sha256"]:
                raise ValueError(f"graph_hash_mismatch:{r['canonical_observation_id']}")
            with np.load(graph, allow_pickle=False) as g:
                if len(g["node_type"]) != int(r["node_count"]) or g["edge_index"].shape != (2, int(r["edge_count"])):
                    raise ValueError(f"graph_shape_mismatch:{r['canonical_observation_id']}")
    fold = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in outer}
    groups_by_fold = {f: set() for f in range(5)}
    source_by_fold = {f: Counter() for f in range(5)}
    canonical_by_id = {r["canonical_row_id"]: r for r in canonical}
    group_lookup = {r["canonical_observation_id"]: r["group_id"] for r in outer}
    for r in outer:
        f = int(r["outer_fold"])
        groups_by_fold[f].add(r["group_id"])
        source_by_fold[f][canonical_by_id[r["canonical_observation_id"]]["source_id"]] += 1
    for a in range(5):
        for b in range(a + 1, 5):
            if groups_by_fold[a] & groups_by_fold[b]:
                raise ValueError("outer_group_leakage")
    source_fold_counts = {f: Counter(canonical_by_id[rid]["source_id"] for rid, value in fold.items() if value == f)
                          for f in range(5)}
    if any(any(source_fold_counts[f][source] == 0 for source in
               {r["source_id"] for r in canonical}) for f in range(5)):
        raise ValueError("outer_fold_missing_source")
    for key in ("qasm_bytes_sha256", "workflow_group_hash"):
        identity_to_folds = {}
        for c in canonical:
            value = c.get(key, "")
            if value: identity_to_folds.setdefault(value, set()).add(fold[c["canonical_row_id"]])
        if any(len(v) != 1 for v in identity_to_folds.values()):
            raise ValueError(f"split_identity_leak:{key}")
    logical_to_folds = {}
    model_groups = {r["canonical_observation_id"]: r for r in read(out / "model_input_groups.csv")}
    for r in feat:
        signature = model_groups[r["canonical_observation_id"]]["model_input_group_digest"]
        if signature:
            logical_to_folds.setdefault(signature, set()).add(fold[r["canonical_observation_id"]])
    if any(len(v) != 1 for v in logical_to_folds.values()):
        raise ValueError("split_identity_leak:logical_input_digest")
    inner_fold_count = json.loads((out / "split_manifest.json").read_text())["inner_folds"]
    for f in range(5):
        train_ids = {rid for rid, val in fold.items() if val != f}
        if {r["canonical_observation_id"] for r in inner if int(r["outer_fold"]) == f} != train_ids:
            raise ValueError(f"inner_assignment_not_once_per_fold:{f}")
        val_groups = {k: set() for k in range(inner_fold_count)}
        val_sources = {k: Counter() for k in range(inner_fold_count)}
        for r in inner:
            if int(r["outer_fold"]) == f:
                val_groups[int(r["inner_fold"])].add(r["group_id"])
                val_sources[int(r["inner_fold"])][canonical_by_id[r["canonical_observation_id"]]["source_id"]] += 1
        for a in range(inner_fold_count):
            for b in range(a + 1, inner_fold_count):
                if val_groups[a] & val_groups[b]:
                    raise ValueError(f"inner_group_leakage:{f}")
        if any(val_sources[0][source] == 0 for source in {r["source_id"] for r in canonical}):
            raise ValueError(f"inner_validation_fold0_missing_source:{f}")
    input_paths = [CANON, REP, RECOVERY, CANDIDATES, REGISTRY, QPACK, c135.QONDUCTOR_ZIP,
                   Path(c135.__file__), Path(mf.__file__), CONTRACT]
    input_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path.read_bytes())
                    for path in input_paths}
    feature_manifest = {"status": "PASS", "rows": len(feat), "available": sum(x["status"] == "available" for x in feat),
            "unavailable": sum(x["status"] == "unavailable" for x in feat),
            "source_status_counts": {f"{a}|{b}": n for (a, b), n in source_counts.items()},
            "graphs_referenced": len({x["graph_file"] for x in feat if x["status"] == "available"}),
            "graphs_hash_checked": True, "global_width_finite_all_available": True,
            "source_extractor_sha256": mf.SOURCE_PINS["data_preparation/helper.py"],
            "source_utils_sha256": mf.SOURCE_PINS["data_preparation/utils.py"],
            "feature_materializer_executed_sha256": prior_feature_manifest.get(
                "feature_materializer_executed_sha256", prior_feature_manifest.get("feature_encoder_sha256", "")),
            "feature_encoder_sha256": digest(Path(__file__).read_bytes()),
            "preparation_script_sha256": digest(Path(__file__).read_bytes()),
            "python_version": __import__("platform").python_version(),
            "qiskit_version": __import__("importlib.metadata", fromlist=["version"]).version("qiskit"),
            "numpy_version": __import__("importlib.metadata", fromlist=["version"]).version("numpy"),
            "input_hashes": input_hashes,
            "features_stage": "logical_recipe_pretranspile; graph after barrier/final-measurement removal",
            "T1_T2_mapping": "logical wire i to nominal snapshot qubit i, microseconds",
            "T1_T2_are_synthetic": False, "runtime_target_accessed": False}
    (out / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    manifest = {"status": "PASS", "rows": len(feat), "available": sum(x["status"] == "available" for x in feat),
                "unavailable": sum(x["status"] == "unavailable" for x in feat),
                "source_status_counts": {f"{a}|{b}": n for (a, b), n in source_counts.items()},
                "graphs_referenced": len({x["graph_file"] for x in feat if x["status"] == "available"}),
                "graphs_hash_checked": True, "global_width_finite_all_available": True,
                "outer_split_rows": len(outer), "outer_groups_disjoint": True,
                "source_counts_per_outer_fold": {str(f): dict(source_by_fold[f]) for f in range(5)},
                "source_counts_per_inner_fold0": {str(f): dict(val_sources[0]) for f in range(5)},
                "legacy_capacity_mismatch_rows": sum(bool(r["legacy_capacity_field"]) and
                    r["legacy_capacity_matches_snapshot"] == "false" for r in read(out / "eligibility.csv")),
                "inner_assignment_checked": True,
                "model_visible_input_groups_checked": True,
                "preparation_script_sha256": digest(Path(__file__).read_bytes()),
                "artifacts_sha256": {name: digest((out / name).read_bytes()) for name in
                                      ("eligibility.csv", "feature_attempts.csv", "model_input_groups.csv",
                                       "logical_outer_splits.csv", "logical_inner_splits.csv", "split_manifest.json")},
                "contract_sha256": digest(CONTRACT.read_bytes())}
    (out / "validation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    inner_by_fold = {}
    for r in inner:
        if int(r["inner_fold"]) >= 0:
            inner_by_fold.setdefault((int(r["outer_fold"]), int(r["inner_fold"])), set()).add(r["canonical_observation_id"])
    attempt_by_id = {r["canonical_observation_id"]: r for r in feat}
    available_ids = {rid for rid, r in attempt_by_id.items() if r["status"] == "available"}
    def rowset_sha(row_ids):
        return digest("\n".join(sorted(row_ids)).encode())
    cells = []
    for method in ("mali_full_feature_graph", "mali_matched_global_mlp"):
        for f in range(5):
            for seed in (42, 1234, 31415):
                train = {rid for rid, value in fold.items() if value != f}
                test = {rid for rid, value in fold.items() if value == f}
                valid = inner_by_fold[(f, 0)]
                fit_ok, valid_ok = (train - valid) & available_ids, valid & available_ids
                test_ok = test & available_ids
                cells.append({"method": method, "outer_fold": f, "seed": seed,
                              "fit_available_rows": len(fit_ok), "validation_available_rows": len(valid_ok),
                              "test_rows_all_sources": len(test), "test_available_rows": len(test_ok),
                              "test_unavailable_rows": len(test - available_ids),
                              "fit_ids_sha256": rowset_sha(fit_ok), "validation_ids_sha256": rowset_sha(valid_ok),
                              "test_ids_sha256": rowset_sha(test), "test_available_ids_sha256": rowset_sha(test_ok)})
    for f in range(5):
        train = {rid for rid, value in fold.items() if value != f}
        test = {rid for rid, value in fold.items() if value == f}
        cells.append({"method": "unified_polynomial_v3", "outer_fold": f, "seed": "deterministic",
                      "fit_rows": len(train), "inner_selection_ids_sha256": rowset_sha(inner_by_fold[(f, 0)]),
                      "inner_fold_ids_sha256": {str(k): rowset_sha(inner_by_fold[(f, k)]) for k in range(4)},
                      "test_rows_all_sources": len(test), "fit_ids_sha256": rowset_sha(train),
                      "test_ids_sha256": rowset_sha(test)})
    execution = {"training_authorized": False, "rows": 8767, "cells": cells,
                 "fit_cells": len(cells), "all_methods_share_split": True,
                 "cuda_required_for_neural_fits": True,
                 "method_cards": {
                     "mali_full_feature_graph": {"reader_label": "Ma-Li full-feature logical-recipe adaptation",
                         "fidelity_class": "architecture_adaptation", "protocol": "benchmark_v1/protocol/mali_full_features.json",
                         "protocol_sha256": digest((ROOT / "benchmark_v1/protocol/mali_full_features.json").read_bytes()),
                         "raw_globals": 51, "graph_node_width": 178, "available_rows": 7350,
                         "neural_seeds": [42, 1234, 31415], "target_transform": "identity seconds per the locked full-feature contract"},
                     "mali_matched_global_mlp": {"raw_globals": 51, "same_rows_and_splits_as_graph": True,
                         "protocol_sha256": digest((ROOT / "benchmark_v1/protocol/mali_full_features.json").read_bytes())},
                     "unified_polynomial_v3": {"reader_label": "Qonductor-style unified polynomial adaptation",
                         "protocol": "benchmark_v1/protocol/learned_method_contract_v3.json",
                         "protocol_sha256": digest((ROOT / "benchmark_v1/protocol/learned_method_contract_v3.json").read_bytes()),
                         "features": ["log1p(active_width)", "log1p(structural_depth)", "log1p(two_qubit_count)",
                                      "log1p(swap_like_count)", "log1p(shots)", "circuit_count=1"],
                         "candidate_degrees": [2, 3, 4], "target_transform": "log1p seconds; invert before scoring",
                         "inner_grouped_folds_for_degree_selection": 4}},
                 "validation_sha256": digest((out / "validation.json").read_bytes()),
                 "feature_manifest_sha256": digest((out / "feature_manifest.json").read_bytes()),
                 "split_manifest_sha256": digest((out / "split_manifest.json").read_bytes())}
    (out / "training_cell_manifest.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
