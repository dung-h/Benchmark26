#!/usr/bin/env python3
"""Materialize the frozen compiled-input representation for the 8,767-row QPU panel.

This is a CPU-only input builder. It reads no observed runtime values for feature
generation and performs no fitting, timing, QPU submission, or simulator run.
Ma--Li and QPack use the pinned C133 Target and C135 transpilation recipe;
Qonductor keeps its exact submitted physical QASM unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import pickle
import struct
import subprocess
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1" / "scripts"))
import run_recovery_c133_c135_v3 as c135  # noqa: E402

CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
INNER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv"
C133_DIR = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3"
C133_REGISTRY = C133_DIR / "c133_snapshot_registry_v3.csv"
C133_MANIFEST = C133_DIR / "c133_snapshot_registry_manifest.json"
C134_MANIFEST = C133_DIR / "c134_feature_sidecar_manifest.json"
QPACK_STRUCT = ROOT / "artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/qpack_mcp_structural_rows.csv"
QPACK_REVISION = c135.QPACK_REVISION
MALI_QASM_ROOT = c135.MALI_QASM_ROOT
QONDUCTOR_ZIP = c135.QONDUCTOR_ZIP
DEFAULT_OUT = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
# Captured immediately before the full materialization process was launched.
# The audit-refresh code added later does not retroactively change that run.
MATERIALIZER_EXECUTED_SHA256 = "e92c7ec8aba92bc993fce6781bda98efdcbcd088136e5564774b0ef33e88d38d"
EXPECTED_COUNTS = {
    "mali_real_qpu": 340,
    "qonductor_single_circuit_ibm": 4482,
    "qpack_mcp": 3945,
}
EXPECTED_QISKIT = "2.5.2"
EXPECTED_RUNTIME = "0.49.0"
TRANSPILE_SETTINGS = {
    "optimization_level": 1,
    "layout_method": "trivial",
    "routing_method": "sabre",
    "translation_method": "translator",
    "seed_function": "run_recovery_c133_c135_v3.seed_for(representation_digest, source_id, backend_canonical)",
    "seed_context": c135.CONTEXT_ID,
}
GRAPH_FORMAT = "s71-operation-dag-v1"
DYNAMIC_NAMES = {"if_else", "while_loop", "for_loop", "switch_case"}
SKIP_NAMES = {"barrier", "snapshot"}
GRAPH_MAX_WIRE_ARITY = 4  # The inherited S71 graph schema stores q0..q3.


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def graph_file_sha256_manifest(graph_dir: Path) -> dict[str, str]:
    """Map each serialized graph filename to the SHA256 of its file bytes."""
    return {path.name: sha_file(path) for path in sorted(graph_dir.glob("*.pkl"))}


def stable_digest(value: object) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: "" if row.get(key) is None else row.get(key, "") for key in fields} for row in rows)
    return sha_file(path)


def write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha_file(path)


def environment_fingerprint() -> dict[str, object]:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "qiskit_version": importlib.metadata.version("qiskit"),
        "qiskit_ibm_runtime_version": importlib.metadata.version("qiskit-ibm-runtime"),
        "pip_freeze_sha256": sha_bytes(freeze.encode()),
        "host": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def canonical_snapshot_registry(env: dict[str, object]) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    registry_rows = read_csv(C133_REGISTRY)
    manifest = json.loads(C133_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or len(registry_rows) != len(c135.BACKENDS):
        raise RuntimeError("C133 registry is not a complete PASS")
    if manifest.get("registry_sha256") != sha_file(C133_REGISTRY):
        raise RuntimeError("C133 registry hash does not match its manifest")
    if env["qiskit_version"] != EXPECTED_QISKIT or env["qiskit_ibm_runtime_version"] != EXPECTED_RUNTIME:
        raise RuntimeError(f"pinned environment mismatch: {env['qiskit_version']} / {env['qiskit_ibm_runtime_version']}")
    by_backend = {row["backend_canonical"]: row for row in registry_rows}
    if set(by_backend) != set(c135.BACKENDS):
        raise RuntimeError("C133 backend key mismatch")
    backends: dict[str, object] = {}
    for backend_name, class_name in c135.BACKENDS.items():
        reg = by_backend[backend_name]
        cls = getattr(fake_provider, class_name, None)
        if cls is None:
            raise RuntimeError(f"pinned_fake_backend_class_missing:{class_name}")
        module = Path(__import__(cls.__module__, fromlist=["__name__"]).__file__ or "")
        if not module.is_file() or sha_file(module) != reg["class_module_sha256"]:
            raise RuntimeError(f"C133 backend module hash drift:{backend_name}")
        for path_key, hash_key in (("configuration_path", "configuration_sha256"), ("properties_path", "properties_sha256")):
            asset = Path(reg[path_key])
            if not asset.is_file() or sha_file(asset) != reg[hash_key]:
                raise RuntimeError(f"C133 snapshot asset hash drift:{backend_name}:{path_key}")
        backends[backend_name] = cls()
    return by_backend, backends


def _param_identity(value: object) -> object:
    """Stable enough for the pinned Qiskit representation; reject symbolic values."""
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float_hex", "value": value.hex()}
    try:
        numeric = float(value)  # Qiskit numeric scalar types.
    except (TypeError, ValueError, OverflowError):
        text = str(value)
        if any(token in text for token in ("Parameter(", "ParameterExpression(", "Symbol(")):
            raise ValueError(f"unbound_symbolic_parameter:{text}")
        return {"kind": "text", "value": text}
    return {"kind": "float_hex", "value": numeric.hex()}


def compiled_ir_digest(circuit: QuantumCircuit) -> str:
    operations: list[dict[str, object]] = []
    for inst in circuit.data:
        operations.append({
            "name": str(inst.operation.name),
            "qargs": [circuit.find_bit(qubit).index for qubit in inst.qubits],
            "cargs": [circuit.find_bit(clbit).index for clbit in inst.clbits],
            "params": [_param_identity(value) for value in inst.operation.params],
        })
    return stable_digest({
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "global_phase": _param_identity(circuit.global_phase),
        "operations": operations,
    })


def feature_values(circuit: QuantumCircuit) -> dict[str, object]:
    counts: Counter[str] = Counter()
    active: set[int] = set()
    one = two = swap_like = measurements = 0
    for inst in circuit.data:
        name = str(inst.operation.name)
        qargs = tuple(circuit.find_bit(qubit).index for qubit in inst.qubits)
        counts[name] += 1
        if name not in SKIP_NAMES:
            active.update(qargs)
        if name == "measure":
            measurements += 1
        elif name not in SKIP_NAMES and len(qargs) == 1:
            one += 1
        elif len(qargs) == 2:
            two += 1
        if "swap" in name:
            swap_like += 1
    return {
        "register_width": int(circuit.num_qubits),
        "active_width": len(active),
        "structural_depth": int(circuit.depth() or 0),
        "one_qubit_count": one,
        "two_qubit_count": two,
        "swap_like_count": swap_like,
        "measurement_count": measurements,
        "operation_count": sum(counts.values()) - sum(counts[name] for name in SKIP_NAMES),
        "gate_counts_json": json.dumps(dict(sorted(counts.items())), sort_keys=True, separators=(",", ":")),
    }


def _wire_index(circuit: QuantumCircuit, qubit: object) -> int:
    return int(circuit.find_bit(qubit).index)


def graph_record(circuit: QuantumCircuit) -> tuple[dict[str, object] | None, str, str]:
    names: list[str] = []
    arity = array("B")
    parameter = array("B")
    q_arrays = [array("i") for _ in range(GRAPH_MAX_WIRE_ARITY)]
    edge_src = array("i")
    edge_dst = array("i")
    last_by_qubit: dict[int, int] = {}
    dynamic = False
    high_arity: Counter[int] = Counter()
    graph_hash = hashlib.sha256()
    graph_hash.update(GRAPH_FORMAT.encode() + b"\0")
    graph_hash.update(struct.pack(">QQ", int(circuit.num_qubits), int(circuit.num_clbits)))
    operation_count = 0
    for inst in circuit.data:
        operation = inst.operation
        name = str(operation.name)
        if name in SKIP_NAMES:
            continue
        if name in DYNAMIC_NAMES:
            dynamic = True
        qargs = tuple(_wire_index(circuit, qubit) for qubit in inst.qubits)
        if len(qargs) > GRAPH_MAX_WIRE_ARITY:
            high_arity[len(qargs)] += 1
        node = operation_count
        operation_count += 1
        encoded_name = name.encode("utf-8")
        graph_hash.update(struct.pack(">I", len(encoded_name)))
        graph_hash.update(encoded_name)
        graph_hash.update(struct.pack(">I", len(qargs)))
        for qarg in qargs:
            graph_hash.update(struct.pack(">q", qarg))
        has_parameter = bool(getattr(operation, "params", ()))
        graph_hash.update(b"\x01" if has_parameter else b"\x00")
        names.append(sys.intern(name))
        arity.append(min(len(qargs), 255))
        parameter.append(1 if has_parameter else 0)
        for index, values in enumerate(q_arrays):
            values.append(qargs[index] if index < len(qargs) else -1)
        for qubit in qargs:
            previous = last_by_qubit.get(qubit)
            if previous is not None:
                edge_src.append(previous)
                edge_dst.append(node)
            last_by_qubit[qubit] = node
    graph_digest = graph_hash.hexdigest()
    if dynamic:
        return None, graph_digest, "dynamic_control_not_representable_by_declared_flat_dag_v3"
    if high_arity:
        arities = ",".join(str(key) for key in sorted(high_arity))
        return None, graph_digest, f"operation_arity_exceeds_s71_q0_q3_schema:{arities}"
    return {
        "format": GRAPH_FORMAT,
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "operation_count": operation_count,
        "names": names,
        "arity": arity,
        "parameter": parameter,
        "q0": q_arrays[0],
        "q1": q_arrays[1],
        "q2": q_arrays[2],
        "q3": q_arrays[3],
        "edge_src": edge_src,
        "edge_dst": edge_dst,
    }, graph_digest, ""


def target_support(circuit: QuantumCircuit, backend: object) -> dict[str, object]:
    """Return exact nominal-target support diagnostics without changing the circuit."""
    target = backend.target
    cache: dict[tuple[str, tuple[int, ...]], bool] = {}
    unsupported_counts: Counter[str] = Counter()
    unsupported_instruction_count = 0
    checked_instruction_count = 0
    for inst in circuit.data:
        name = str(inst.operation.name)
        if name in SKIP_NAMES:
            continue
        qargs = tuple(_wire_index(circuit, qubit) for qubit in inst.qubits)
        key = (name, qargs)
        if key not in cache:
            try:
                cache[key] = bool(target.instruction_supported(name, qargs))
            except Exception:
                cache[key] = False
        checked_instruction_count += 1
        if not cache[key]:
            unsupported_instruction_count += 1
            unsupported_counts[name] += 1
    too_wide = int(circuit.num_qubits) > int(getattr(backend, "num_qubits", 0) or 0)
    return {
        "nominal_target_compatibility": "fail" if too_wide or unsupported_instruction_count else "pass",
        "nominal_target_width_exceeds": too_wide,
        "target_checked_instruction_count": checked_instruction_count,
        "target_unsupported_instruction_count": unsupported_instruction_count,
        "target_unsupported_operation_counts_json": json.dumps(dict(sorted(unsupported_counts.items())), separators=(",", ":")),
    }


def load_source_circuit(row: dict[str, str], qpack: dict[str, dict[str, str]], qond_zip: ZipFile,
                        parsed_cache: dict[tuple[str, str], tuple[QuantumCircuit | None, str]],
                        source_digest: str) -> tuple[QuantumCircuit | None, bytes | None, str]:
    source = row["source_id"]
    cache_key = (source, source_digest)
    if cache_key in parsed_cache:
        circuit, error = parsed_cache[cache_key]
        return circuit, None, error
    try:
        if source == "qpack_mcp":
            q = qpack[row["source_row_index"]]
            circuit = c135.qpack_circuit(int(q["size"]), int(q["p"]))
            parsed_cache[cache_key] = (circuit, "")
            return circuit, None, ""
        if source == "mali_real_qpu":
            data = (MALI_QASM_ROOT / row["qasm_path_or_member"]).read_bytes()
        elif source == "qonductor_single_circuit_ibm":
            data = qond_zip.read(row["qasm_path_or_member"])
        else:
            raise ValueError(f"unknown_source_id:{source}")
        actual = sha_bytes(data)
        if actual != row["qasm_bytes_sha256"]:
            raise ValueError(f"source_qasm_hash_mismatch:{actual}")
        circuit = c135.parse_qasm_bytes(data)
        parsed_cache[cache_key] = (circuit, "")
        return circuit, data, ""
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        parsed_cache[cache_key] = (None, error)
        return None, None, error


def representation_identity(row: dict[str, str], qpack: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    source = row["source_id"]
    if source == "qpack_mcp":
        q = qpack[row["source_row_index"]]
        recipe = {
            "qpack_revision": QPACK_REVISION,
            "problem": q["problem"],
            "size": int(q["size"]),
            "p": int(q["p"]),
        }
        return stable_digest(recipe), "pinned_angle_insensitive_to_target_v3", "logical_structure_reconstructed_from_recorded_workflow_configuration"
    if source == "mali_real_qpu":
        return row["qasm_bytes_sha256"], "exact_logical_to_pinned_target_v3", "logical_pre_transpile"
    return row["qasm_bytes_sha256"], "exact_submitted_physical_v3", "submitted_physical"


def source_recipe(row: dict[str, str], qpack: dict[str, dict[str, str]]) -> dict[str, object]:
    if row["source_id"] == "qpack_mcp":
        q = qpack[row["source_row_index"]]
        return {
            "kind": "qpack_reconstruction_qualified",
            "generator_revision": QPACK_REVISION,
            "problem": q["problem"],
            "size": int(q["size"]),
            "p": int(q["p"]),
            "angles": {"rz": 0.3, "rx": 0.2, "status": "representative_not_original_optimizer_angles"},
        }
    return {"kind": row["identity_kind"], "qasm_bytes_sha256": row["qasm_bytes_sha256"]}


def compiler_identity(source: str, representation_digest: str, backend_name: str,
                      registry_row: dict[str, str], env: dict[str, object]) -> tuple[str, int | None, dict[str, object]]:
    if source in {"mali_real_qpu", "qpack_mcp"}:
        seed = c135.seed_for(representation_digest, source, backend_name)
        route = "deterministic_nominal_target_transpile"
        transpiler = {**TRANSPILE_SETTINGS, "seed_transpiler": seed}
    else:
        seed = None
        route = "preserve_exact_submitted_physical_qasm_no_retranspile"
        transpiler = {"route": route, "no_retranspile": True}
    target_identity = {
        "snapshot_id": registry_row["snapshot_id"],
        "fake_backend_class": registry_row["fake_backend_class"],
        "module_sha256": registry_row["class_module_sha256"],
        "configuration_sha256": registry_row["configuration_sha256"],
        "properties_sha256": registry_row["properties_sha256"],
        "environment_hash": registry_row["environment_hash"],
        "package_version": registry_row["package_version"],
        "qiskit_version": registry_row["qiskit_version"],
    }
    identity = {
        "source_id": source,
        "source_representation_digest": representation_digest,
        "target": target_identity,
        "qiskit_version": env["qiskit_version"],
        "qiskit_ibm_runtime_version": env["qiskit_ibm_runtime_version"],
        "route": route,
        "transpiler": transpiler,
    }
    return stable_digest(identity), seed, {"compiler_cache_identity": identity, "compiler_route": route}


def transpile_declared(circuit: QuantumCircuit, source: str, backend: object, seed: int | None) -> QuantumCircuit:
    if source == "qonductor_single_circuit_ibm":
        return circuit
    if seed is None:
        raise ValueError("missing_transpiler_seed")
    return transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=seed,
        layout_method="trivial",
        routing_method="sabre",
        translation_method="translator",
    )


def select_canaries(canonical: list[dict[str, str]], qpack: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    chosen: dict[str, dict[str, str]] = {}
    per_source_backend: dict[tuple[str, str], dict[str, str]] = {}
    for row in canonical:
        key = (row["source_id"], c135.canonical_backend(row["backend"]))
        per_source_backend.setdefault(key, row)
    for row in per_source_backend.values():
        chosen[row["canonical_row_id"]] = row
    qpack_structure_rows: dict[str, dict[str, str]] = {}
    for row in canonical:
        if row["source_id"] == "qpack_mcp":
            digest, _, _ = representation_identity(row, qpack)
            qpack_structure_rows.setdefault(digest, row)
    for row in qpack_structure_rows.values():
        chosen[row["canonical_row_id"]] = row
    return [chosen[key] for key in sorted(chosen)]


def compile_one(row: dict[str, str], qpack: dict[str, dict[str, str]], qond_zip: ZipFile,
                parsed_cache: dict[tuple[str, str], tuple[QuantumCircuit | None, str]],
                env: dict[str, object], registry: dict[str, dict[str, str]], backends: dict[str, object]) -> dict[str, object]:
    source = row["source_id"]
    backend_name = c135.canonical_backend(row["backend"])
    rep_digest, rep_id, lifecycle = representation_identity(row, qpack)
    reg = registry.get(backend_name)
    base = {
        "canonical_row_id": row["canonical_row_id"],
        "source_id": source,
        "source_row_index": row["source_row_index"],
        "source_row_id": row["source_row_id"],
        "identity_kind": row["identity_kind"],
        "source_qasm_sha256": row["qasm_bytes_sha256"],
        "source_recipe_json": json.dumps(source_recipe(row, qpack), sort_keys=True, separators=(",", ":")),
        "source_representation_digest": rep_digest,
        "representation_id": rep_id,
        "lifecycle_stage": lifecycle,
        "source_local_leakage_group_id": row.get("circuit_group_hash", ""),
        "workflow_group_hash": row.get("workflow_group_hash", ""),
        "backend_observed": row["backend"],
        "backend_canonical": backend_name,
        "shots": row["shots"],
        "circuit_count": 1,
        "target_snapshot_id": reg.get("snapshot_id", "") if reg else "",
        "target_registry_row_sha256": stable_digest(reg) if reg else "",
        "target_configuration_sha256": reg.get("configuration_sha256", "") if reg else "",
        "target_properties_sha256": reg.get("properties_sha256", "") if reg else "",
        "snapshot_tier": reg.get("snapshot_tier", "") if reg else "",
        "compiler_cache_key": "",
        "compiler_route": "",
        "transpiler_seed": "",
        "compiled_or_submitted_digest": "",
        "graph_input_digest": "",
        "graph_file": "",
        "compiled_qasm3_file": "",
        "compiled_qasm3_sha256": "",
        "materialization_status": "unavailable",
        "primary_models_eligible": False,
        "terminal_reason": "",
        "target_compatibility_status": "not_evaluated",
        "target_checked_instruction_count": 0,
        "target_unsupported_instruction_count": 0,
        "target_unsupported_operation_counts_json": "{}",
        "register_width": "",
        "active_width": "",
        "structural_depth": "",
        "one_qubit_count": "",
        "two_qubit_count": "",
        "swap_like_count": "",
        "measurement_count": "",
        "operation_count": "",
        "gate_counts_json": "{}",
        "graph_status": "unavailable",
        "graph_terminal_reason": "",
        "outer_fold": "",
        "outer_assignment_origin": "",
        "unified_leakage_group_id": "",
        "source_local_leakage_group_kind": "",
    }
    if reg is None or backend_name not in backends:
        base["terminal_reason"] = f"missing_pinned_target:{backend_name}"
        return base
    cache_key, seed, compiler = compiler_identity(source, rep_digest, backend_name, reg, env)
    base["compiler_cache_key"] = cache_key
    base["compiler_route"] = compiler["compiler_route"]
    base["transpiler_seed"] = "" if seed is None else seed
    circuit, source_bytes, load_error = load_source_circuit(row, qpack, qond_zip, parsed_cache, rep_digest)
    if circuit is None:
        base["terminal_reason"] = f"source_input_unavailable:{load_error}"
        return base
    try:
        output = transpile_declared(circuit, source, backends[backend_name], seed)
        compiled_digest = compiled_ir_digest(output)
        features = feature_values(output)
        graph, graph_digest, graph_error = graph_record(output)
        nominal_support = target_support(output, backends[backend_name])
        if source in {"mali_real_qpu", "qpack_mcp"}:
            target_compatibility = str(nominal_support["nominal_target_compatibility"])
            materialization_error = "" if target_compatibility == "pass" else "transpiled_output_not_supported_by_pinned_target"
        else:
            # Historical submitted QASM is kept unchanged. Current nominal target
            # compatibility is diagnostic only because it is not job-day state.
            target_compatibility = "advisory_" + str(nominal_support["nominal_target_compatibility"])
            materialization_error = ""
        base.update({
            "compiled_or_submitted_digest": compiled_digest,
            "target_compatibility_status": target_compatibility,
            "target_checked_instruction_count": nominal_support["target_checked_instruction_count"],
            "target_unsupported_instruction_count": nominal_support["target_unsupported_instruction_count"],
            "target_unsupported_operation_counts_json": nominal_support["target_unsupported_operation_counts_json"],
            **features,
        })
        if materialization_error:
            base["terminal_reason"] = materialization_error
            base["graph_terminal_reason"] = graph_error
            return base
        if graph is None:
            base["terminal_reason"] = graph_error
            base["graph_terminal_reason"] = graph_error
            base["graph_input_digest"] = graph_digest
            return base
        base.update({
            "graph_input_digest": graph_digest,
            "materialization_status": "available",
            "primary_models_eligible": True,
            "terminal_reason": "",
            "graph_status": "available",
            "graph_terminal_reason": "",
        })
        base["_graph_record"] = graph
        if source in {"mali_real_qpu", "qpack_mcp"}:
            try:
                qasm_text = qasm3.dumps(output)
                base["_compiled_qasm3"] = qasm_text.encode("utf-8")
            except Exception as exc:
                # Graph/features remain exactly recoverable from the emitted record;
                # the optional source-readable export failure is retained in metadata.
                base["_compiled_qasm3_error"] = f"{type(exc).__name__}:{exc}"
        return base
    except Exception as exc:
        base["terminal_reason"] = f"{type(exc).__name__}:{exc}"
        base["graph_terminal_reason"] = base["terminal_reason"]
        return base


def determinism_canary(canonical: list[dict[str, str]], qpack: dict[str, dict[str, str]], env: dict[str, object],
                        registry: dict[str, dict[str, str]], backends: dict[str, object]) -> dict[str, object]:
    chosen = select_canaries(canonical, qpack)
    parsed: dict[tuple[str, str], tuple[QuantumCircuit | None, str]] = {}
    details: list[dict[str, object]] = []
    with ZipFile(QONDUCTOR_ZIP) as qond_zip:
        for row in chosen:
            source = row["source_id"]
            backend_name = c135.canonical_backend(row["backend"])
            rep_digest, _, _ = representation_identity(row, qpack)
            circuit1, _, err1 = load_source_circuit(row, qpack, qond_zip, parsed, rep_digest)
            if circuit1 is None:
                details.append({"canonical_row_id": row["canonical_row_id"], "status": "FAIL", "reason": f"input:{err1}"})
                continue
            seed = c135.seed_for(rep_digest, source, backend_name) if source in {"mali_real_qpu", "qpack_mcp"} else None
            try:
                out_a = transpile_declared(circuit1, source, backends[backend_name], seed)
                digest_a = compiled_ir_digest(out_a)
                graph_a, graph_digest_a, graph_error_a = graph_record(out_a)
            except Exception as exc:
                details.append({"canonical_row_id": row["canonical_row_id"], "status": "FAIL", "reason": f"first_compile:{type(exc).__name__}:{exc}"})
                continue
            # Parse a fresh copy to detect accidental in-place compiler mutation.
            parsed_b: dict[tuple[str, str], tuple[QuantumCircuit | None, str]] = {}
            circuit2, _, err2 = load_source_circuit(row, qpack, qond_zip, parsed_b, rep_digest)
            if circuit2 is None:
                details.append({"canonical_row_id": row["canonical_row_id"], "status": "FAIL", "reason": f"second_input:{err2}"})
                continue
            try:
                out_b = transpile_declared(circuit2, source, backends[backend_name], seed)
                digest_b = compiled_ir_digest(out_b)
                graph_b, graph_digest_b, graph_error_b = graph_record(out_b)
                same = digest_a == digest_b and graph_digest_a == graph_digest_b and graph_error_a == graph_error_b
                details.append({
                    "canonical_row_id": row["canonical_row_id"],
                    "source_id": source,
                    "backend_canonical": backend_name,
                    "source_representation_digest": rep_digest,
                    "seed": "" if seed is None else seed,
                    "first_compiled_ir_sha256": digest_a,
                    "second_compiled_ir_sha256": digest_b,
                    "first_graph_input_sha256": graph_digest_a,
                    "second_graph_input_sha256": graph_digest_b,
                    "first_graph_status": "available" if graph_a is not None else graph_error_a,
                    "second_graph_status": "available" if graph_b is not None else graph_error_b,
                    "status": "PASS" if same else "FAIL",
                    "reason": "" if same else "compiled_or_graph_digest_changed_on_repeat",
                })
            except Exception as exc:
                details.append({"canonical_row_id": row["canonical_row_id"], "status": "FAIL", "reason": f"second_compile:{type(exc).__name__}:{exc}"})
    failed = [row for row in details if row["status"] != "PASS"]
    return {
        "status": "PASS" if details and not failed else "FAIL",
        "canary_count": len(details),
        "failure_count": len(failed),
        "checks": details,
        "selection": "one row per source/backend plus each of the six pinned QPack reconstruction recipes",
    }


def split_and_identity_audit(canonical: list[dict[str, str]], outer_rows: list[dict[str, str]],
                             inner_rows: list[dict[str, str]], materialized: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    canonical_by_id = {row["canonical_row_id"]: row for row in canonical}
    outer_by_id = {row["canonical_observation_id"]: row for row in outer_rows}
    material_by_id = {str(row["canonical_row_id"]): row for row in materialized}
    errors: list[str] = []
    expected_ids = set(canonical_by_id)
    if len(canonical_by_id) != 8767:
        errors.append(f"canonical_unique_ids:{len(canonical_by_id)}")
    if set(outer_by_id) != expected_ids:
        errors.append("outer_split_identity_mismatch")
    if set(material_by_id) != expected_ids:
        errors.append("materialized_identity_mismatch")
    if len(inner_rows) != 8767 * 5:
        errors.append(f"inner_split_rows:{len(inner_rows)}")
    inner_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in inner_rows:
        key = (row["outer_fold"], row["canonical_observation_id"])
        if key in inner_index:
            errors.append(f"duplicate_inner_split:{key[0]}:{key[1]}")
            break
        inner_index[key] = row
    for cid, row in canonical_by_id.items():
        split = outer_by_id.get(cid)
        built = material_by_id.get(cid)
        if split is None or built is None:
            continue
        if split["source_id"] != row["source_id"] or str(split["source_row_index"]) != str(row["source_row_index"]):
            errors.append(f"outer_source_identity_mismatch:{cid}")
        if str(built.get("source_id")) != row["source_id"] or str(built.get("source_row_index")) != str(row["source_row_index"]):
            errors.append(f"materialized_source_identity_mismatch:{cid}")
        if str(built.get("outer_fold")) != split["outer_fold"] or str(built.get("unified_leakage_group_id")) != split["unified_leakage_group_id"]:
            errors.append(f"materialized_frozen_split_mismatch:{cid}")
    folds = sorted({row["outer_fold"] for row in outer_rows}, key=int)
    group_overlap: dict[str, int] = {}
    fold_counts: dict[str, dict[str, int]] = {}
    for fold in folds:
        train_groups = {row["unified_leakage_group_id"] for row in outer_rows if row["outer_fold"] != fold}
        test_groups = {row["unified_leakage_group_id"] for row in outer_rows if row["outer_fold"] == fold}
        overlap = train_groups & test_groups
        group_overlap[fold] = len(overlap)
        if overlap:
            errors.append(f"outer_group_leakage_fold_{fold}:{len(overlap)}")
        fold_counts[fold] = dict(Counter(row["outer_fold"] for row in outer_rows if row["outer_fold"] == fold))
        for inner_fold in folds:
            test_ids = {row["canonical_observation_id"] for row in outer_rows if row["outer_fold"] == fold}
            inner_assignments = [row for row in inner_rows if row["outer_fold"] == fold]
            bad_test = [row for row in inner_assignments if row["canonical_observation_id"] in test_ids and row["inner_fold"] != ""]
            if bad_test:
                errors.append(f"outer_test_has_inner_assignment:{fold}")
                break
        # Check source-local group boundaries within each outer fold's training set.
        train_group_to_inner: dict[str, set[str]] = defaultdict(set)
        for row in inner_rows:
            if row["outer_fold"] == fold and row["inner_fold"] != "":
                train_group_to_inner[row["unified_leakage_group_id"]].add(row["inner_fold"])
        if any(len(assignments) != 1 for assignments in train_group_to_inner.values()):
            errors.append(f"inner_group_leakage_fold_{fold}")
    identity_fold_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    exact_qasm_sources: dict[str, set[str]] = defaultdict(set)
    exact_qasm_folds: dict[str, set[str]] = defaultdict(set)
    workflow_folds: dict[str, set[str]] = defaultdict(set)
    recipe_folds: dict[str, set[str]] = defaultdict(set)
    compiled_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    graph_input_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for cid, row in canonical_by_id.items():
        split = outer_by_id.get(cid, {})
        built = material_by_id.get(cid, {})
        fold = str(split.get("outer_fold", ""))
        source = row["source_id"]
        if source in {"mali_real_qpu", "qonductor_single_circuit_ibm"} and row["qasm_bytes_sha256"]:
            identity_fold_map[(source, row["qasm_bytes_sha256"])].add(fold)
            exact_qasm_sources[row["qasm_bytes_sha256"]].add(source)
            exact_qasm_folds[row["qasm_bytes_sha256"]].add(fold)
        if source == "qpack_mcp" and row["workflow_group_hash"]:
            workflow_folds[row["workflow_group_hash"]].add(fold)
        if built.get("source_representation_digest"):
            recipe_folds[str(built["source_representation_digest"])].add(fold)
        if built.get("compiled_or_submitted_digest"):
            compiled_groups[str(built["compiled_or_submitted_digest"])].append(built)
        if built.get("graph_input_digest") and built.get("primary_models_eligible"):
            graph_input_groups[str(built["model_input_digest"])].append(built)
    source_identity_leaks = [
        {"source_id": source, "source_identity_digest": digest, "folds": sorted(folds, key=int)}
        for (source, digest), folds in identity_fold_map.items() if len(folds) > 1
    ]
    qpack_workflow_leaks = [
        {"workflow_group_hash": digest, "folds": sorted(folds, key=int)}
        for digest, folds in workflow_folds.items() if len(folds) > 1
    ]
    cross_source_exact = [
        {"qasm_sha256": digest, "source_ids": sorted(sources), "folds": sorted(exact_qasm_folds[digest], key=int)}
        for digest, sources in exact_qasm_sources.items() if len(sources) > 1
    ]
    compiled_collision_rows: list[dict[str, object]] = []
    for digest, rows in compiled_groups.items():
        rep_digests = {str(row["source_representation_digest"]) for row in rows}
        sources = {str(row["source_id"]) for row in rows}
        collision_folds = sorted({str(row["outer_fold"]) for row in rows}, key=int)
        if len(rows) <= 1 and len(collision_folds) <= 1:
            continue
        if len(collision_folds) > 1:
            expected_qpack = sources == {"qpack_mcp"} and len(rep_digests) == 1
            classification = "expected_qpack_structure_reuse_across_workflows" if expected_qpack else "cross_fold_compiled_ir_collision_review_required"
        elif len(rep_digests) == 1:
            classification = "same_source_representation_repeated_within_frozen_fold"
        else:
            classification = "compiled_ir_collision_within_fold_distinct_source_recipes"
        compiled_collision_rows.append({
            "collision_kind": "compiled_ir_digest",
            "digest": digest,
            "classification": classification,
            "sources_json": json.dumps(sorted(sources), separators=(",", ":")),
            "source_representation_count": len(rep_digests),
            "row_count": len(rows),
            "fold_counts_json": json.dumps(dict(sorted(Counter(str(row["outer_fold"]) for row in rows).items(), key=lambda x: int(x[0]))), separators=(",", ":")),
            "sample_canonical_ids_json": json.dumps(sorted(str(row["canonical_row_id"]) for row in rows)[:20], separators=(",", ":")),
        })
    for digest, rows in graph_input_groups.items():
        row_ids = {str(row["canonical_row_id"]) for row in rows}
        folds_here = sorted({str(row["outer_fold"]) for row in rows}, key=int)
        sources = {str(row["source_id"]) for row in rows}
        if len(folds_here) <= 1:
            continue
        expected_qpack = sources == {"qpack_mcp"} and all(str(row["workflow_group_hash"]) for row in rows)
        compiled_collision_rows.append({
            "collision_kind": "model_graph_plus_globals_digest",
            "digest": digest,
            "classification": "expected_qpack_structure_reuse_across_workflows" if expected_qpack else "feature_input_collision_across_folds_review",
            "sources_json": json.dumps(sorted(sources), separators=(",", ":")),
            "source_representation_count": len({str(row["source_representation_digest"]) for row in rows}),
            "row_count": len(row_ids),
            "fold_counts_json": json.dumps(dict(sorted(Counter(str(row["outer_fold"]) for row in rows).items(), key=lambda x: int(x[0]))), separators=(",", ":")),
            "sample_canonical_ids_json": json.dumps(sorted(row_ids)[:20], separators=(",", ":")),
        })
    blocking = bool(errors or source_identity_leaks or qpack_workflow_leaks or cross_source_exact)
    audit = {
        "status": "PASS" if not blocking else "FAIL",
        "blocking_training": blocking,
        "errors": errors,
        "canonical_unique_rows": len(canonical_by_id),
        "outer_split_rows": len(outer_rows),
        "inner_split_rows": len(inner_rows),
        "outer_fold_row_counts": {fold: sum(row["outer_fold"] == fold for row in outer_rows) for fold in folds},
        "outer_leakage_group_count": len({row["unified_leakage_group_id"] for row in outer_rows}),
        "outer_group_overlap_by_fold": group_overlap,
        "source_identity_cross_fold_count": len(source_identity_leaks),
        "source_identity_cross_fold_examples": source_identity_leaks[:20],
        "qpack_workflow_cross_fold_count": len(qpack_workflow_leaks),
        "qpack_workflow_cross_fold_examples": qpack_workflow_leaks[:20],
        "cross_source_exact_qasm_collision_count": len(cross_source_exact),
        "cross_source_exact_qasm_collision_examples": cross_source_exact[:20],
        "qpack_reconstruction_recipe_cross_fold_count": sum(len(folds) > 1 for digest, folds in recipe_folds.items() if any(row["source_id"] == "qpack_mcp" and row["source_representation_digest"] == digest for row in materialized)),
        "compiled_ir_collision_count": sum(row["collision_kind"] == "compiled_ir_digest" for row in compiled_collision_rows),
        "compiled_ir_cross_fold_review_count": sum(row["classification"] == "cross_fold_compiled_ir_collision_review_required" for row in compiled_collision_rows),
        "graph_global_feature_cross_fold_collision_count": sum(row["collision_kind"] == "model_graph_plus_globals_digest" for row in compiled_collision_rows),
        "qpack_expected_graph_global_cross_fold_collision_count": sum(row["collision_kind"] == "model_graph_plus_globals_digest" and row["classification"] == "expected_qpack_structure_reuse_across_workflows" for row in compiled_collision_rows),
        "compiled_and_feature_collisions_are_auto_leaks": False,
        "qpack_claim_boundary": "frozen split holds workflow_group_hash out; reconstructed QAOA structures and exact graph+globals may recur across folds; do not claim circuit-OOD",
    }
    audit.update(model_evaluation_collision_gate(compiled_collision_rows))
    return audit, compiled_collision_rows


def model_evaluation_collision_gate(collision_rows: list[dict[str, object]]) -> dict[str, object]:
    """Keep source-split leakage separate from repeated model-facing inputs."""
    review_rows = [
        row for row in collision_rows
        if row.get("collision_kind") == "model_graph_plus_globals_digest"
        and row.get("classification") != "expected_qpack_structure_reuse_across_workflows"
    ]
    return {
        "representation_collision_review_status": "REVIEW_REQUIRED" if review_rows else "PASS",
        "model_evaluation_gate": "BLOCKED_PENDING_REPRESENTATION_COLLISION_REVIEW" if review_rows else "PASS_NO_UNEXPLAINED_CROSS_FOLD_MODEL_INPUT_COLLISIONS",
        "blocking_model_evaluation": bool(review_rows),
        "non_qpack_cross_fold_model_input_collision_count": len(review_rows),
        "non_qpack_cross_fold_model_input_collision_examples": review_rows[:20],
    }


def refresh_existing_audit(output_dir: Path) -> int:
    """Rebuild only R2 audit/manifest sidecars from a completed materialization."""
    required_files = (output_dir / "representation_rows.csv", output_dir / "manifest.json", output_dir / "support_and_leakage_audit.json")
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit("cannot refresh incomplete materialization: " + ", ".join(missing))
    canonical = read_csv(CANONICAL)
    outer_rows = read_csv(OUTER)
    inner_rows = read_csv(INNER)
    rows = read_csv(output_dir / "representation_rows.csv")
    for row in rows:
        row["primary_models_eligible"] = row.get("primary_models_eligible", "").strip().lower() in {"true", "1", "yes"}
    audit, collision_rows = split_and_identity_audit(canonical, outer_rows, inner_rows, rows)
    old_audit = json.loads((output_dir / "support_and_leakage_audit.json").read_text(encoding="utf-8"))
    old_audit.update(audit)
    collision_fields = ["collision_kind", "digest", "classification", "sources_json", "source_representation_count", "row_count", "fold_counts_json", "sample_canonical_ids_json"]
    collision_sha = write_csv(output_dir / "identity_collision_audit.csv", collision_rows, collision_fields)
    audit_sha = write_json(output_dir / "support_and_leakage_audit.json", old_audit)
    graph_hash_sha = write_json(output_dir / "graph_file_hashes.json", graph_file_sha256_manifest(output_dir / "graphs"))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision_audit_sha256"] = collision_sha
    manifest["support_and_leakage_audit_sha256"] = audit_sha
    manifest["graph_file_hash_manifest_sha256"] = graph_hash_sha
    manifest["graph_file_hash_manifest_semantics"] = "filename_to_sha256_of_serialized_pickle_file_bytes"
    manifest["materializer_executed_sha256"] = MATERIALIZER_EXECUTED_SHA256
    manifest["audit_refresh_code_sha256"] = sha_file(Path(__file__).resolve())
    manifest["source_split_audit_status"] = audit["status"]
    manifest["representation_collision_review_status"] = audit["representation_collision_review_status"]
    manifest["model_evaluation_gate"] = audit["model_evaluation_gate"]
    if audit["blocking_training"]:
        manifest["status"] = "BLOCKED_SOURCE_SPLIT_AUDIT_FAILED"
    elif audit["blocking_model_evaluation"]:
        manifest["status"] = "REVIEW_REQUIRED_BEFORE_MODEL_EVALUATION"
    else:
        manifest["status"] = "PASS"
    manifest_sha = write_json(manifest_path, manifest)
    print(json.dumps({
        "audit_refresh_code_sha256": manifest["audit_refresh_code_sha256"],
        "materializer_executed_sha256": manifest["materializer_executed_sha256"],
        "source_split_audit_status": audit["status"],
        "representation_collision_review_status": audit["representation_collision_review_status"],
        "model_evaluation_gate": audit["model_evaluation_gate"],
        "manifest_sha256": manifest_sha,
        "audit_sha256": audit_sha,
        "collision_audit_sha256": collision_sha,
    }, indent=2, sort_keys=True))
    return 0 if not audit["blocking_training"] else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canary-only", action="store_true", help="Run repeat-compilation canaries without creating artifacts.")
    parser.add_argument("--refresh-audit", action="store_true", help="Refresh R2 audit/manifest sidecars after materialization; does not recompile circuits.")
    args = parser.parse_args()
    if args.refresh_audit:
        return refresh_existing_audit(args.output_dir.resolve())
    if not args.canary_only and args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    required = (CANONICAL, OUTER, INNER, C133_REGISTRY, C133_MANIFEST, C134_MANIFEST, QPACK_STRUCT, MALI_QASM_ROOT, QONDUCTOR_ZIP)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required input(s): " + ", ".join(missing))
    env = environment_fingerprint()
    if env["qiskit_version"] != EXPECTED_QISKIT or env["qiskit_ibm_runtime_version"] != EXPECTED_RUNTIME:
        raise SystemExit("R2 requires the pinned Qiskit 2.5.2 / qiskit-ibm-runtime 0.49.0 environment")
    canonical = read_csv(CANONICAL)
    outer_rows = read_csv(OUTER)
    inner_rows = read_csv(INNER)
    qpack_rows_list = read_csv(QPACK_STRUCT)
    qpack = {row["source_row_index"]: row for row in qpack_rows_list}
    if len(canonical) != 8767 or len({row["canonical_row_id"] for row in canonical}) != 8767:
        raise SystemExit("canonical corpus must contain 8,767 unique rows")
    if Counter(row["source_id"] for row in canonical) != Counter(EXPECTED_COUNTS):
        raise SystemExit("canonical source counts differ from the frozen 340/4,482/3,945 panel")
    if len(qpack) != EXPECTED_COUNTS["qpack_mcp"]:
        raise SystemExit("QPack reconstruction sidecar identity/count mismatch")
    if sha_file(CANONICAL) != json.loads(C134_MANIFEST.read_text(encoding="utf-8"))["canonical_sha256"]:
        raise SystemExit("canonical input hash disagrees with existing C134 evidence")
    registry, backends = canonical_snapshot_registry(env)
    baseline_ids = {row["canonical_observation_id"] for row in outer_rows}
    if baseline_ids != {row["canonical_row_id"] for row in canonical}:
        raise SystemExit("frozen outer split does not cover the canonical IDs exactly once")
    outer_by_id = {row["canonical_observation_id"]: row for row in outer_rows}
    qond_zip = ZipFile(QONDUCTOR_ZIP)
    try:
        canary = determinism_canary(canonical, qpack, env, registry, backends)
    finally:
        qond_zip.close()
    print(json.dumps({"determinism_canary": canary}, indent=2, sort_keys=True), flush=True)
    if canary["status"] != "PASS":
        return 2
    if args.canary_only:
        return 0

    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    graph_dir = out / "graphs"
    qasm_dir = out / "compiled_qasm"
    graph_dir.mkdir()
    qasm_dir.mkdir()
    parsed_cache: dict[tuple[str, str], tuple[QuantumCircuit | None, str]] = {}
    result_by_cache: dict[str, dict[str, object]] = {}
    graph_file_by_digest: dict[str, str] = {}
    graph_hash_by_digest: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    qond_zip = ZipFile(QONDUCTOR_ZIP)
    try:
        for index, row in enumerate(canonical, start=1):
            source = row["source_id"]
            backend_name = c135.canonical_backend(row["backend"])
            rep_digest, _, _ = representation_identity(row, qpack)
            cache_key, _, _ = compiler_identity(source, rep_digest, backend_name, registry[backend_name], env)
            if cache_key not in result_by_cache:
                result_by_cache[cache_key] = compile_one(row, qpack, qond_zip, parsed_cache, env, registry, backends)
            result = dict(result_by_cache[cache_key])
            result["canonical_row_id"] = row["canonical_row_id"]
            result["source_row_index"] = row["source_row_index"]
            result["source_row_id"] = row["source_row_id"]
            result["source_local_leakage_group_id"] = row["circuit_group_hash"]
            result["workflow_group_hash"] = row["workflow_group_hash"]
            split = outer_by_id[row["canonical_row_id"]]
            result["outer_fold"] = split["outer_fold"]
            result["outer_assignment_origin"] = split["outer_assignment_origin"]
            result["unified_leakage_group_id"] = split["unified_leakage_group_id"]
            result["source_local_leakage_group_kind"] = split["source_local_leakage_group_kind"]
            graph = result.pop("_graph_record", None)
            qasm_bytes = result.pop("_compiled_qasm3", None)
            qasm_error = result.pop("_compiled_qasm3_error", "")
            if graph is not None:
                digest = str(result["graph_input_digest"])
                if digest not in graph_file_by_digest:
                    graph_name = digest + ".pkl"
                    graph_path = graph_dir / graph_name
                    with graph_path.open("wb") as handle:
                        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    graph_file_by_digest[digest] = graph_name
                    graph_hash_by_digest[digest] = sha_file(graph_path)
                # The digest is computed from the complete model-facing graph
                # stream (ordered names, qargs, parameter bits and register sizes).
                # Identical digests therefore share one immutable graph record.
                result["graph_file"] = graph_file_by_digest[digest]
            if qasm_bytes is not None:
                qasm_name = str(result["compiler_cache_key"]) + ".qasm3"
                qasm_path = qasm_dir / qasm_name
                if not qasm_path.exists():
                    qasm_path.write_bytes(qasm_bytes)
                result["compiled_qasm3_file"] = str(Path("compiled_qasm") / qasm_name)
                result["compiled_qasm3_sha256"] = sha_file(qasm_path)
            elif qasm_error:
                result["compiled_qasm3_file"] = ""
                result["compiled_qasm3_sha256"] = ""
                result["compiled_qasm3_export_error"] = qasm_error
            # Hash the exact graph+declared globals consumed by the graph/MLP
            # input layer. Labels, source IDs, backends and calibration are absent.
            if result.get("primary_models_eligible"):
                globals_raw = {name: int(result[name]) for name in (
                    "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
                    "swap_like_count", "measurement_count", "shots",
                )}
                result["model_input_digest"] = stable_digest({"graph_input_digest": result["graph_input_digest"], "globals": globals_raw})
            else:
                result["model_input_digest"] = ""
            result.pop("compiler_cache_identity", None)
            rows.append(result)
            if index % 500 == 0 or index == len(canonical):
                print(f"materialized {index}/{len(canonical)} rows; unique compile/cache identities={len(result_by_cache)}; graph records={len(graph_file_by_digest)}", flush=True)
    finally:
        qond_zip.close()

    # The materialized rows have no target or other post-execution fields.
    forbidden = {"target_seconds", "target_native", "runtime", "time_taken", "error_rate", "T1", "T2"}
    fields = sorted({key for row in rows for key in row})
    if forbidden & set(fields):
        raise RuntimeError(f"forbidden feature/target fields present:{sorted(forbidden & set(fields))}")
    rows_path = out / "representation_rows.csv"
    rows_sha = write_csv(rows_path, rows, fields)
    audit, collision_rows = split_and_identity_audit(canonical, outer_rows, inner_rows, rows)
    collision_fields = ["collision_kind", "digest", "classification", "sources_json", "source_representation_count", "row_count", "fold_counts_json", "sample_canonical_ids_json"]
    collision_sha = write_csv(out / "identity_collision_audit.csv", collision_rows, collision_fields)

    material_status = Counter(str(row["materialization_status"]) for row in rows)
    graph_status = Counter(str(row["graph_status"]) for row in rows)
    eligibility = Counter(str(bool(row["primary_models_eligible"])) for row in rows)
    source_counts = {source: {
        "assigned": sum(row["source_id"] == source for row in rows),
        "materialized": sum(row["source_id"] == source and row["materialization_status"] == "available" for row in rows),
        "primary_eligible": sum(row["source_id"] == source and row["primary_models_eligible"] for row in rows),
        "unavailable_reasons": dict(Counter(str(row["terminal_reason"]) for row in rows if row["source_id"] == source and row["terminal_reason"])),
    } for source in EXPECTED_COUNTS}
    qond_rows = [row for row in rows if row["source_id"] == "qonductor_single_circuit_ibm"]
    qpack_recipe_fold_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["source_id"] == "qpack_mcp":
            qpack_recipe_fold_counts[str(row["source_representation_digest"])].add(str(row["outer_fold"]))
    c134_semantic = read_csv(C133_DIR / "c134_qonductor_semantic_audit_v3.csv")
    c134_semantic_counts = dict(Counter(row["mapping_status"] for row in c134_semantic))
    audit.update({
        "materialization_rows": len(rows),
        "materialization_status_counts": dict(material_status),
        "graph_status_counts": dict(graph_status),
        "primary_model_eligibility_counts": dict(eligibility),
        "source_counts": source_counts,
        "unique_source_target_compiler_cache_identities": len(result_by_cache),
        "unique_graph_records": len(graph_file_by_digest),
        "graph_record_bytes": sum((graph_dir / name).stat().st_size for name in graph_file_by_digest.values()),
        "compiled_qasm3_export_count": len(list(qasm_dir.glob("*.qasm3"))),
        "qonductor_nominal_target_compatibility": dict(Counter(str(row["target_compatibility_status"]) for row in qond_rows)),
        "qonductor_semantic_audit_from_existing_c134": c134_semantic_counts,
        "qonductor_semantic_audit_sha256": sha_file(C133_DIR / "c134_qonductor_semantic_audit_v3.csv"),
        "qpack_unique_reconstruction_recipes": len(qpack_recipe_fold_counts),
        "qpack_recipes_crossing_workflow_folds": sum(len(value) > 1 for value in qpack_recipe_fold_counts.values()),
    })
    audit_path = out / "support_and_leakage_audit.json"
    audit_sha = write_json(audit_path, audit)
    graph_hash_manifest = {graph_file_by_digest[digest]: graph_hash_by_digest[digest] for digest in graph_file_by_digest}
    graph_hash_sha = write_json(out / "graph_file_hashes.json", graph_hash_manifest)
    canary_path = out / "determinism_canary.json"
    canary_sha = write_json(canary_path, canary)
    inputs = {
        "canonical_observations.csv": sha_file(CANONICAL),
        "unified_outer_split_v2.csv": sha_file(OUTER),
        "unified_inner_split_v2.csv": sha_file(INNER),
        "c133_snapshot_registry_v3.csv": sha_file(C133_REGISTRY),
        "c133_snapshot_registry_manifest.json": sha_file(C133_MANIFEST),
        "c134_feature_sidecar_manifest.json": sha_file(C134_MANIFEST),
        "qpack_mcp_structural_rows.csv": sha_file(QPACK_STRUCT),
    }
    manifest = {
        "artifact_id": "compiled-unified-qpu-input-materialization-v1",
        "status": "PASS" if audit["status"] == "PASS" and canary["status"] == "PASS" else "BLOCKED_TRAINING_REVIEW_REQUIRED",
        "training_performed": False,
        "timing_performed": False,
        "qpu_submitted": False,
        "target_labels_read_for_features": False,
        "protocol": "docs/benchmark_recovery_plan.md#C-real-qpu-compiled-input-correction",
        "representation_contract": "benchmark_v1/decisions/circuit_representation_v3.md",
        "snapshot_contract": "benchmark_v1/decisions/snapshot_policy_v3.md",
        "compiled_feature_contract": [
            "polynomial: active_width, structural_depth, two_qubit_count, swap_like_count, shots; circuit_count=1",
            "graph/metadata MLP: active_width, structural_depth, one_qubit_count, two_qubit_count, swap_like_count, measurement_count, shots",
            "no backend/source/representation/hash/calibration/runtime feature",
        ],
        "source_routes": {
            "mali_real_qpu": "exact logical QASM to pinned nominal FakeBackend Target with C135 deterministic transpilation",
            "qonductor_single_circuit_ibm": "exact submitted physical QASM preserved unchanged; current nominal target support is advisory only",
            "qpack_mcp": "six pinned angle-insensitive QAOA structural reconstructions with representative angles to pinned nominal FakeBackend Target",
        },
        "transpiler_settings": TRANSPILE_SETTINGS,
        "qpack_revision": QPACK_REVISION,
        "snapshot_claim": "tier_3_backend_matched_nominal; not execution-day calibration",
        "frozen_split": "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv and unified_inner_split_v2.csv; reused byte-for-byte",
        "inputs_sha256": inputs,
        "environment": env,
        "row_count": len(rows),
        "representation_rows_sha256": rows_sha,
        "collision_audit_sha256": collision_sha,
        "support_and_leakage_audit_sha256": audit_sha,
        "graph_file_hash_manifest_sha256": graph_hash_sha,
        "determinism_canary_sha256": canary_sha,
        "output_counts": {
            "materialization_status": dict(material_status),
            "graph_status": dict(graph_status),
            "primary_models_eligible": dict(eligibility),
        },
        "unavailable_handling": "retain all 8,767 assigned rows and explicit reason; no imputation or synthetic circuits/labels",
    }
    manifest_sha = write_json(out / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": manifest_sha, "manifest": manifest, "audit": audit}, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
