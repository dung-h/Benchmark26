#!/usr/bin/env python3
"""Routine C133--C135 materialization and target preflight for RQPU V3.

This worker only parses archived circuits, inventories pinned FakeBackends and
runs deterministic target/feature preflight samples.  It never reads an
observed runtime as an input, fits a model, runs a QPU or runs a simulator.
The output directory is append-only: an existing directory is refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime import fake_provider


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
QPACK_STRUCT = ROOT / "artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/qpack_mcp_structural_rows.csv"
MALI_QASM_ROOT = Path("/home/server/Documents/Quantum-Execution-Time-Prediction/data/quantum_circuits")
QONDUCTOR_ZIP = Path("/home/server/Documents/Qonductor-SC25/data/database/circuits.zip")
SNAPSHOT_POLICY = ROOT / "benchmark_v1/protocol/snapshot_policy_v3.json"
REPRESENTATION_POLICY = ROOT / "benchmark_v1/protocol/circuit_representation_v3.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
QPACK_REVISION = "9beaf65e951e01181b1e324cbb1b227af10eef96"
ROOT_SEED = "20260925"
TRANSPILE_STREAM = "20260925-transpile"
CONTEXT_ID = "rqpu_c135_target_preflight_v3"

BACKENDS = {
    "ibm_belem": "FakeBelemV2",
    "ibm_brisbane": "FakeBrisbane",
    "ibm_jakarta": "FakeJakartaV2",
    "ibm_kolkata": "FakeKolkataV2",
    "ibm_kyoto": "FakeKyoto",
    "ibm_lagos": "FakeLagosV2",
    "ibm_lima": "FakeLimaV2",
    "ibm_manila": "FakeManilaV2",
    "ibm_nairobi": "FakeNairobiV2",
    "ibm_osaka": "FakeOsaka",
    "ibm_perth": "FakePerth",
    "ibm_quito": "FakeQuitoV2",
}
ALIASES = {
    "ibmq_belem": "ibm_belem",
    "ibmq_jakarta": "ibm_jakarta",
    "ibmq_kolkata": "ibm_kolkata",
    "ibmq_lima": "ibm_lima",
    "ibmq_manila": "ibm_manila",
    "ibmq_quito": "ibm_quito",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_digest(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: "" if row.get(key) is None else row.get(key, "") for key in fields} for row in rows)
    return sha256_file(path)


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def canonical_backend(name: str) -> str:
    return ALIASES.get(name, name)


def seed_for(qasm_digest: str, source: str, backend: str, repetition: int = 0) -> int:
    material = f"{ROOT_SEED}|{TRANSPILE_STREAM}|{qasm_digest}|{CONTEXT_ID}:{source}:{backend}|{repetition}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def qubit_index(circuit: QuantumCircuit, qubit: object) -> int:
    return int(circuit.find_bit(qubit).index)


def qasm_features(circuit: QuantumCircuit) -> dict[str, object]:
    counts = Counter(str(inst.operation.name) for inst in circuit.data)
    active: set[int] = set()
    one_q = two_q = measurement = swap_like = 0
    for inst in circuit.data:
        name = str(inst.operation.name)
        qargs = tuple(qubit_index(circuit, q) for q in inst.qubits)
        if name not in {"barrier", "snapshot"}:
            active.update(qargs)
        if name == "measure":
            measurement += 1
        elif name not in {"barrier", "snapshot"} and len(qargs) == 1:
            one_q += 1
        elif len(qargs) == 2:
            two_q += 1
        if "swap" in name:
            swap_like += 1
    return {
        "register_width": circuit.num_qubits,
        "active_width": len(active),
        "structural_depth": circuit.depth(),
        "one_qubit_count": one_q,
        "two_qubit_count": two_q,
        "swap_like_count": swap_like,
        "measurement_count": measurement,
        "gate_counts_json": json.dumps(dict(sorted(counts.items())), sort_keys=True),
        "operation_names_json": json.dumps(sorted(counts), separators=(",", ":")),
    }


def parse_qasm_bytes(data: bytes) -> QuantumCircuit:
    """Parse QASM2 and QASM3 without silently downgrading either format."""
    text = data.decode("utf-8")
    if text.lstrip().startswith("OPENQASM 3"):
        return qasm3.loads(text)
    return QuantumCircuit.from_qasm_str(text)


def compiled_ir_digest(circuit: QuantumCircuit) -> str:
    """Version-independent digest of the compiled operation IR."""
    operations = []
    for inst in circuit.data:
        operations.append({
            "name": str(inst.operation.name),
            "qargs": [qubit_index(circuit, q) for q in inst.qubits],
            "cargs": [circuit.find_bit(c).index for c in inst.clbits],
            "params": [str(p) for p in inst.operation.params],
        })
    return stable_digest({"num_qubits": circuit.num_qubits, "num_clbits": circuit.num_clbits, "operations": operations})


def regular_graph_edges(size: int) -> list[tuple[int, int]]:
    if size == 2:
        return [(0, 1)]
    if size == 3:
        return [(0, 1), (1, 2), (2, 0)]
    if size == 4:
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges: list[tuple[int, int]] = []
    for index in range(size - 2):
        edges += [(index, index + 1), (index, index + 2)]
    edges += [(size - 2, size - 1), (size - 2, 0), (size - 1, 0), (size - 1, 1)]
    return edges


def qpack_circuit(size: int, p: int) -> QuantumCircuit:
    circuit = QuantumCircuit(size, size)
    circuit.h(range(size))
    for _ in range(p):
        for source, target in regular_graph_edges(size):
            circuit.cx(source, target)
            circuit.rz(0.3, target)
            circuit.cx(source, target)
        circuit.rx(0.2, range(size))
    circuit.measure(range(size), range(size))
    return circuit


def environment_fingerprint() -> dict[str, object]:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "qiskit_version": importlib.metadata.version("qiskit"),
        "qiskit_ibm_runtime_version": importlib.metadata.version("qiskit-ibm-runtime"),
        "pip_freeze_sha256": sha256_bytes(freeze.encode()),
        "pip_freeze": freeze.splitlines(),
        "host": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def metric_value(config: object, key: str) -> object:
    value = getattr(config, key, None)
    return None if value in (None, "None", "null") else value


def snapshot_registry(out: Path, env: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for backend_name, class_name in BACKENDS.items():
        cls = getattr(fake_provider, class_name, None)
        if cls is None:
            errors.append(f"missing_class:{backend_name}:{class_name}")
            continue
        try:
            backend = cls()
            config = backend.configuration()
            props = backend.properties()
            module_path = Path(__import__(cls.__module__, fromlist=["__name__"]).__file__ or "")
            backend_dir = module_path.parent
            conf_path = backend_dir / str(getattr(cls, "conf_filename", ""))
            props_path = backend_dir / str(getattr(cls, "props_filename", ""))
            target_names = sorted(str(x) for x in backend.target.operation_names)
            conf_sha = sha256_file(conf_path) if conf_path.exists() else ""
            props_sha = sha256_file(props_path) if props_path.exists() else ""
            qprop_names = [str(getattr(x, "name", x)) for x in props.qubits[0]] if getattr(props, "qubits", None) else []
            rows.append({
                "snapshot_id": f"qiskit_ibm_runtime_0.49.0::{backend_name}::{getattr(config, 'backend_version', '')}",
                "backend_canonical": backend_name,
                "backend_observed_aliases": ";".join(sorted([k for k, v in ALIASES.items() if v == backend_name] + [backend_name])),
                "fake_backend_class": class_name,
                "class_module_path": str(module_path),
                "class_module_sha256": sha256_file(module_path) if module_path.exists() else "",
                "package_name": "qiskit-ibm-runtime",
                "package_version": env["qiskit_ibm_runtime_version"],
                "qiskit_version": env["qiskit_version"],
                "environment_hash": env["pip_freeze_sha256"],
                "configuration_path": str(conf_path),
                "configuration_sha256": conf_sha,
                "properties_path": str(props_path),
                "properties_sha256": props_sha,
                "defaults_path": "",
                "defaults_sha256": "",
                "snapshot_or_update_date": str(metric_value(config, "online_date") or metric_value(config, "last_update_date") or ""),
                "backend_version": str(getattr(config, "backend_version", "")),
                "num_qubits": getattr(config, "n_qubits", getattr(backend, "num_qubits", "")),
                "clops_h": metric_value(config, "clops_h"),
                "clops_v": metric_value(config, "clops_v"),
                "target_construction": "FakeBackendV2 target from pinned local conf/props package assets",
                "target_operation_names": ";".join(target_names),
                "target_operation_coverage": len(target_names),
                "instruction_duration_coverage": len(getattr(backend.target.durations(), "instructions", [])),
                "error_coverage": len(getattr(props, "gates", []) or []),
                "t1_t2_coverage": ";".join(x for x in ("T1", "T2") if x in qprop_names),
                "readout_coverage": "readout_error" in qprop_names,
                "snapshot_tier": "tier_3_backend_matched_nominal",
                "source_claim": "local pinned FakeBackend asset; not job-day calibration",
            })
        except Exception as exc:  # retain evidence instead of aborting inventory
            errors.append(f"instantiate:{backend_name}:{type(exc).__name__}:{exc}")
    fields = list(rows[0]) if rows else []
    digest = write_csv(out / "c133_snapshot_registry_v3.csv", rows, fields)
    manifest = {
        "artifact_id": "c133-snapshot-registry-v3",
        "contract": "benchmark_v1/protocol/snapshot_policy_v3.json",
        "status": "pass" if len(rows) == len(BACKENDS) and not errors else "fail",
        "backend_count": len(rows),
        "expected_backend_count": len(BACKENDS),
        "errors": errors,
        "environment": "environment_fingerprint.json",
        "registry_sha256": digest,
        "provenance": "local qiskit-ibm-runtime fake-provider conf/props; missing defaults are explicit, not backfilled",
    }
    write_json(out / "c133_snapshot_registry_manifest.json", manifest)
    return rows, manifest


def load_circuits(canonical: list[dict[str, str]]) -> tuple[dict[str, QuantumCircuit], dict[str, bytes], dict[str, str]]:
    qasm_cache: dict[str, QuantumCircuit] = {}
    raw_bytes: dict[str, bytes] = {}
    errors: dict[str, str] = {}
    qond_zip = ZipFile(QONDUCTOR_ZIP)
    try:
        for row in canonical:
            if row["source_id"] == "qpack_mcp":
                continue
            digest = row["qasm_bytes_sha256"]
            if digest in qasm_cache or digest in errors:
                continue
            try:
                if row["source_id"] == "mali_real_qpu":
                    path = MALI_QASM_ROOT / row["qasm_path_or_member"]
                    data = path.read_bytes()
                else:
                    data = qond_zip.read(row["qasm_path_or_member"])
                actual = sha256_bytes(data)
                if actual != digest:
                    errors[digest] = f"qasm_hash_mismatch:{actual}"
                    continue
                raw_bytes[digest] = data
                qasm_cache[digest] = parse_qasm_bytes(data)
            except Exception as exc:
                errors[digest] = f"{type(exc).__name__}:{exc}"
    finally:
        qond_zip.close()
    return qasm_cache, raw_bytes, errors


def materialize_features(out: Path, canonical: list[dict[str, str]], qpack_rows: dict[str, dict[str, str]], qasm_cache: dict[str, QuantumCircuit], parse_errors: dict[str, str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    qpack_cache: dict[tuple[int, int], tuple[QuantumCircuit, str]] = {}
    for raw in canonical:
        source = raw["source_id"]
        base = {
            "canonical_row_id": raw["canonical_row_id"],
            "source_id": source,
            "source_row_index": raw["source_row_index"],
            "backend_observed": raw["backend"],
            "backend_canonical": canonical_backend(raw["backend"]),
            "shots": raw["shots"],
            "circuit_count": 1,
            "target_seconds_present": False,
            "availability_status": "available",
            "terminal_reason": "",
            "target_snapshot_id": "pending_C135",
            "compiled_or_submitted_digest": "",
            "representation_digest": "",
            "lifecycle_stage": "",
            "representation_id": "",
            "register_width": "",
            "active_width": "",
            "structural_depth": "",
            "one_qubit_count": "",
            "two_qubit_count": "",
            "swap_like_count": "",
            "measurement_count": "",
            "gate_counts_json": "",
            "operation_names_json": "",
        }
        if source == "qpack_mcp":
            q = qpack_rows[raw["source_row_index"]]
            key = (int(q["size"]), int(q["p"]))
            if key not in qpack_cache:
                circuit = qpack_circuit(*key)
                recipe_digest = stable_digest({"qpack_revision": QPACK_REVISION, "problem": q["problem"], "size": key[0], "p": key[1]})
                qpack_cache[key] = (circuit, recipe_digest)
            circuit, recipe_digest = qpack_cache[key]
            feats = qasm_features(circuit)
            base.update({"representation_id": "pinned_angle_insensitive_to_target_v3", "lifecycle_stage": "logical_structure_reconstructed_from_recorded_workflow_configuration", "representation_digest": recipe_digest, "compiled_or_submitted_digest": "", "target_snapshot_id": "pending_C135", "provenance_flag": "reconstruction_qualified_no_exact_qasm_no_angles_no_route", "source_recipe": f"MCP|size={q['size']}|p={q['p']}|QPack@{QPACK_REVISION}", **feats})
        else:
            digest = raw["qasm_bytes_sha256"]
            circuit = qasm_cache.get(digest)
            if circuit is None:
                base.update({"representation_id": "exact_logical_to_pinned_target_v3" if source == "mali_real_qpu" else "exact_submitted_physical_v3", "lifecycle_stage": "logical_pre_transpile" if source == "mali_real_qpu" else "submitted_physical", "representation_digest": digest, "provenance_flag": "exact_logical_qasm" if source == "mali_real_qpu" else "exact_submitted_physical_qasm", "availability_status": "unavailable", "terminal_reason": parse_errors.get(digest, "qasm_parse_or_hash_failure")})
            else:
                feats = qasm_features(circuit)
                base.update({"representation_id": "exact_logical_to_pinned_target_v3" if source == "mali_real_qpu" else "exact_submitted_physical_v3", "lifecycle_stage": "logical_pre_transpile" if source == "mali_real_qpu" else "submitted_physical", "representation_digest": digest, "provenance_flag": "exact_logical_qasm" if source == "mali_real_qpu" else "exact_submitted_physical_qasm", **feats})
                if source == "qonductor_single_circuit_ibm":
                    raw_obj = json.loads(raw["raw_row_json"])
                    checks = {
                        "archive_sum_depth_equals_qasm_two_qubit_count": str(raw_obj.get("sum_depth", "")) == str(feats["two_qubit_count"]),
                        "archive_sum_qubits_equals_qasm_structural_depth": str(raw_obj.get("sum_qubits", "")) == str(feats["structural_depth"]),
                        "archive_sum_2q_gates_equals_qasm_active_width": str(raw_obj.get("sum_2q_gates", "")) == str(feats["active_width"]),
                    }
                    audit.append({"canonical_row_id": raw["canonical_row_id"], "source_row_index": raw["source_row_index"], "qasm_digest": digest, "archive_sum_depth": raw_obj.get("sum_depth", ""), "qasm_two_qubit_count": feats["two_qubit_count"], "archive_sum_qubits": raw_obj.get("sum_qubits", ""), "qasm_structural_depth": feats["structural_depth"], "archive_sum_2q_gates": raw_obj.get("sum_2q_gates", ""), "qasm_active_width": feats["active_width"], "mapping_status": "PASS" if all(checks.values()) else "FAIL", "check_json": json.dumps(checks, sort_keys=True)})
        rows.append(base)
    fields = list(rows[0]) if rows else []
    feature_sha = write_csv(out / "c134_feature_sidecar_v3.csv", rows, fields)
    audit_fields = list(audit[0]) if audit else []
    audit_sha = write_csv(out / "c134_qonductor_semantic_audit_v3.csv", audit, audit_fields)
    audit_counts = Counter(x["mapping_status"] for x in audit)
    manifest = {"artifact_id": "c134-feature-sidecar-v3", "contract": "benchmark_v1/protocol/circuit_representation_v3.json", "canonical_sha256": sha256_file(CANON), "rows": len(rows), "source_counts": dict(Counter(r["source_id"] for r in rows)), "feature_sha256": feature_sha, "qonductor_audit_sha256": audit_sha, "qonductor_audit_counts": dict(audit_counts), "parse_error_count": len(parse_errors), "target_leakage": "target_seconds omitted; target_seconds_present is constant false", "status": "pass" if len(rows) == 8767 and audit_counts == Counter({"PASS": 4482}) else "fail"}
    write_json(out / "c134_feature_sidecar_manifest.json", manifest)
    return rows, manifest


def target_supported(target: object, name: str, qargs: tuple[int, ...]) -> bool:
    if name in {"barrier", "snapshot"}:
        return True
    try:
        return bool(target.instruction_supported(name, qargs))
    except Exception:
        return False


def qcre_path_seconds(circuit: QuantumCircuit, backend: object) -> float:
    target = backend.target
    durations = target.durations()
    dag = circuit_to_dag(circuit)
    finish: dict[object, float] = {}
    total = 0.0
    dt = float(target.dt or 0.0)
    if dt <= 0:
        raise ValueError("target_dt_missing")
    for node in dag.topological_op_nodes():
        qargs = tuple(qubit_index(circuit, q) for q in node.qargs)
        name = str(node.op.name)
        if not target_supported(target, name, qargs):
            raise ValueError(f"unsupported_operation:{name}:{qargs}")
        start = max((finish.get(pred, 0.0) for pred in dag.predecessors(node)), default=0.0)
        if name in {"barrier", "snapshot"}:
            duration_dt = 0.0
        else:
            duration_dt = float(durations.get(name, qargs))
        finish[node] = start + duration_dt
        total = max(total, finish[node])
    return total * dt


def calibration_coverage(circuit: QuantumCircuit, backend: object) -> dict[str, object]:
    props = backend.properties()
    active = sorted({qubit_index(circuit, q) for inst in circuit.data for q in inst.qubits})

    def property_names(values: object) -> set[str]:
        """Return calibration names across Qiskit property API variants.

        Qiskit releases expose the entries returned by ``qubit_property`` as
        objects with a ``name`` attribute, while some FakeBackend assets expose
        the name directly as a string.  Treating the latter as if it had
        ``.name`` caused every otherwise valid C135 attempt to fail with an
        accessor error.  This normalization is observational only: it does
        not fill missing calibration values.
        """
        return {str(getattr(value, "name", value)) for value in (values or [])}

    t1_t2_readout = 0
    for q in active:
        names = property_names(props.qubit_property(q))
        if {"T1", "T2", "readout_error"}.issubset(names):
            t1_t2_readout += 1
    duration_error = 0
    for inst in circuit.data:
        name = str(inst.operation.name)
        if name in {"barrier", "snapshot"}:
            continue
        qargs = tuple(qubit_index(circuit, q) for q in inst.qubits)
        try:
            if target_supported(backend.target, name, qargs) and backend.target.durations().get(name, qargs) is not None:
                if name not in {"measure", "reset", "delay"}:
                    props.gate_error(name, qargs)
                duration_error += 1
        except Exception:
            pass
    return {"active_qubits": len(active), "active_qubits_with_T1_T2_readout": t1_t2_readout, "calibration_qubit_fraction": (t1_t2_readout / len(active) if active else 0.0), "duration_error_covered_ops": duration_error}


def select_preflight(feature_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        if row["availability_status"] == "available":
            grouped[(str(row["source_id"]), str(row["backend_canonical"]))].append(row)
    selected: list[dict[str, object]] = []
    for (source, backend), rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: (stable_digest({"seed": ROOT_SEED, "stream": TRANSPILE_STREAM, "context": CONTEXT_ID, "source": source, "backend": backend, "representation": row["representation_digest"]}), str(row["canonical_row_id"])))
        for rank, row in enumerate(ranked[:10], start=1):
            selected.append({**row, "preflight_rank": rank, "preflight_cell": f"{source}|{backend}", "selection_digest": stable_digest({"source": source, "backend": backend, "representation": row["representation_digest"]})})
    return selected


def run_preflight(out: Path, feature_rows: list[dict[str, object]], qpack_rows: dict[str, dict[str, str]], qasm_cache: dict[str, QuantumCircuit]) -> dict[str, object]:
    selected = select_preflight(feature_rows)
    backend_cache = {name: getattr(fake_provider, class_name)() for name, class_name in BACKENDS.items()}
    qpack_cache: dict[tuple[int, int], QuantumCircuit] = {}
    results: list[dict[str, object]] = []
    for row in selected:
        source = str(row["source_id"]); backend_name = str(row["backend_canonical"]); backend = backend_cache.get(backend_name)
        result = {"canonical_row_id": row["canonical_row_id"], "source_id": source, "backend_canonical": backend_name, "representation_id": row["representation_id"], "preflight_cell": row["preflight_cell"], "preflight_rank": row["preflight_rank"], "transpiler_seed": "", "parse_status": "PASS", "target_build_status": "PASS" if backend else "FAIL", "target_operation_coverage_status": "", "transpile_status": "not_run", "compiled_digest": "", "qiskit_duration_seconds": "", "qcre_path_seconds": "", "qcre_vs_qiskit_relative_error": "", "hyb_hanas_calibration_status": "", "checks_status": "FAIL", "failure_reason": ""}
        try:
            if backend is None:
                raise RuntimeError("backend_registry_missing")
            if source == "qpack_mcp":
                q = qpack_rows[str(row["source_row_index"])]
                key = (int(q["size"]), int(q["p"]))
                circuit = qpack_cache.setdefault(key, qpack_circuit(*key))
            else:
                circuit = qasm_cache[str(row["representation_digest"])]
            if source == "mali_real_qpu" or source == "qpack_mcp":
                seed = seed_for(str(row["representation_digest"]), source, backend_name)
                result["transpiler_seed"] = seed
                compiled = transpile(circuit, backend=backend, optimization_level=1, seed_transpiler=seed, layout_method="trivial", routing_method="sabre", translation_method="translator")
                result["transpile_status"] = "PASS"
            else:
                compiled = circuit
                result["transpile_status"] = "not_required_exact_submitted_physical"
                if compiled.num_qubits > backend.num_qubits:
                    raise RuntimeError(f"width_exceeds_target:{compiled.num_qubits}>{backend.num_qubits}")
            unsupported = []
            for inst in compiled.data:
                name = str(inst.operation.name); qargs = tuple(qubit_index(compiled, q) for q in inst.qubits)
                if not target_supported(backend.target, name, qargs):
                    unsupported.append(f"{name}:{qargs}")
            result["target_operation_coverage_status"] = "PASS" if not unsupported else "FAIL"
            if unsupported:
                raise RuntimeError("unsupported_operations:" + ",".join(unsupported[:8]))
            result["compiled_digest"] = compiled_ir_digest(compiled)
            qiskit_duration = float(compiled.estimate_duration(target=backend.target, unit="s"))
            qcre_duration = float(qcre_path_seconds(compiled, backend))
            result["qiskit_duration_seconds"] = qiskit_duration
            result["qcre_path_seconds"] = qcre_duration
            result["qcre_vs_qiskit_relative_error"] = abs(qcre_duration - qiskit_duration) / max(abs(qiskit_duration), 1e-30)
            calibration = calibration_coverage(compiled, backend)
            result["hyb_hanas_calibration_status"] = "PASS" if calibration["calibration_qubit_fraction"] == 1.0 and calibration["duration_error_covered_ops"] >= 0 else "FAIL"
            result["checks_status"] = "PASS" if math.isfinite(qiskit_duration) and math.isfinite(qcre_duration) and result["qcre_vs_qiskit_relative_error"] <= 0.05 and result["hyb_hanas_calibration_status"] == "PASS" else "FAIL"
            if result["checks_status"] != "PASS":
                result["failure_reason"] = "qcre_qiskit_or_calibration_acceptance"
        except Exception as exc:
            result["failure_reason"] = f"{type(exc).__name__}:{exc}"
        results.append(result)
    fields = list(results[0]) if results else []
    result_sha = write_csv(out / "c135_target_preflight_results_v3.csv", results, fields)
    cell_summary: list[dict[str, object]] = []
    by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_cell[str(row["preflight_cell"])].append(row)
    for cell, members in sorted(by_cell.items()):
        cell_summary.append({"preflight_cell": cell, "attempts": len(members), "successes": sum(x["checks_status"] == "PASS" for x in members), "target_coverage_failures": sum(x["target_operation_coverage_status"] == "FAIL" for x in members), "parse_or_transpile_failures": sum(x["transpile_status"] == "FAIL" for x in members), "status": "PASS" if all(x["checks_status"] == "PASS" for x in members) else "FAIL", "unique_representation_count": len({x["representation_id"] for x in members})})
    cell_sha = write_csv(out / "c135_target_preflight_cell_summary_v3.csv", cell_summary, list(cell_summary[0]) if cell_summary else [])
    status = "pass" if results and all(x["checks_status"] == "PASS" for x in results) else "fail"
    manifest = {"artifact_id": "c135-target-preflight-v3", "contract": "benchmark_v1/protocol/snapshot_policy_v3.json", "representation_contract": "benchmark_v1/protocol/circuit_representation_v3.json", "status": status, "sample_selection": "up to 10 deterministic rows per available source/backend cell; repeated QPack structural configurations remain explicitly counted", "attempts": len(results), "cells": len(cell_summary), "result_sha256": result_sha, "cell_summary_sha256": cell_sha, "pass_criteria": {"required_checks": ["parse", "target coverage", "deterministic target transpilation or exact physical compatibility", "finite Qiskit duration", "QCRE critical path within 5%", "Hyb-HANAS calibration coverage"], "full_panel_authorized": False}, "failure_policy": "No fallback target, calibration, circuit or metric; strong review required for any failure."}
    write_json(out / "c135_target_preflight_manifest.json", manifest)
    return manifest


def write_report(out: Path, manifests: dict[str, object]) -> None:
    lines = ["# Routine Wave 0 + C133–C135 recovery batch", "", "No model fitting, timing run, simulator execution or QPU execution was performed.", "", "| Task | Status | Detail |", "|---|---|---|"]
    lines.append(f"| Wave 0 | **recorded in controlling contracts** | Existing V2 artifacts remain immutable; V3 outputs are additive. |")
    lines.append(f"| C133 | **{str(manifests['C133']['status']).upper()}** | {manifests['C133']['backend_count']}/{manifests['C133']['expected_backend_count']} backend records. |")
    lines.append(f"| C134 | **{str(manifests['C134']['status']).upper()}** | {manifests['C134']['rows']} feature rows; Qonductor audit {manifests['C134']['qonductor_audit_counts']}. |")
    lines.append(f"| C135 | **{str(manifests['C135']['status']).upper()}** | {manifests['C135']['attempts']} deterministic preflight attempts across {manifests['C135']['cells']} source/backend cells. |")
    lines += ["", "## Next gate", "", "This routine batch does not authorize C121 analytical execution or unified V3 training. Strong review must adjudicate C122 semantic retention and every C135 failure before creating a full-run manifest.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    for path in (CANON, QPACK_STRUCT, QONDUCTOR_ZIP, SNAPSHOT_POLICY, REPRESENTATION_POLICY, SEED_REGISTRY):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    out.mkdir(parents=True)
    env = environment_fingerprint()
    write_json(out / "environment_fingerprint.json", env)
    canonical = read_csv(CANON)
    qpack_rows = {row["source_row_index"]: row for row in read_csv(QPACK_STRUCT)}
    registry_rows, c133 = snapshot_registry(out, env)
    qasm_cache, _, parse_errors = load_circuits(canonical)
    features, c134 = materialize_features(out, canonical, qpack_rows, qasm_cache, parse_errors)
    c135 = run_preflight(out, features, qpack_rows, qasm_cache)
    manifests = {"C133": c133, "C134": c134, "C135": c135}
    write_json(out / "batch_manifest.json", {"artifact_id": "wave0-c133-c135-routine-batch-v3", "created_utc": datetime.now(timezone.utc).isoformat(), "canonical_sha256": sha256_file(CANON), "canonical_rows": len(canonical), "source_counts": dict(Counter(r["source_id"] for r in canonical)), "contracts": ["benchmark_v1/decisions/snapshot_policy_v3.md", "benchmark_v1/decisions/circuit_representation_v3.md", "benchmark_v1/decisions/learned_method_contract_v3.md"], "manifests": manifests, "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())}})
    write_report(out, manifests)
    print(json.dumps({"output": str(out.relative_to(ROOT)), "C133": c133["status"], "C134": c134["status"], "C135": c135["status"], "attempts": c135["attempts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
