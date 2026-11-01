"""Pinned Ma--Li full-feature input materialization helpers.

The public entry points deliberately avoid importing the upstream ``helper``
module: that module constructs an IBM provider at import time.  Instead, the
small pinned extractor functions are loaded from ``git show`` after their blob
hashes are checked.  Graph arrays are stored as sparse categorical records;
the exact 178-wide float tensor is expanded only for a fit batch.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/mali_full_features.json"
UPSTREAM_ROOT = ROOT.parent / "Quantum-Execution-Time-Prediction"
COMPILED_ROOT = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
REPRESENTATION = COMPILED_ROOT / "representation_rows.csv"
HASH_MANIFEST = COMPILED_ROOT / "graph_file_hashes.json"
SNAPSHOT_REGISTRY = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"
QONDUCTOR_ZIP = ROOT.parent / "Qonductor-SC25/data/database/circuits.zip"
OUT_ROOT = ROOT / "artifacts/real_qpu/mali_full_features"
ADAPTER_DIR = ROOT / "benchmark_v1/scripts"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
INNER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv"

SOURCE_PINS = {
    "data_preparation/helper.py": "c34cc98df02b016ee9c94dc9ca27efdbbb7e6c6439831f09d3c92ed3a78f74ad",
    "data_preparation/utils.py": "d70db4646f0b9d3deb548d6d960711a43ad818cef344113a6fa1e9f9ae64bcb7",
    "data_preparation/circ_dag_converter.py": "1b29a1c1869baeafdfbf825353f8024b9613b602cfea608e48f6bc23d54e7a85",
    "model/transformer_model.py": "d6d250b2cb3082efc756ae9c5c269d4fe40bbdf49c743c535cddf6a0691c5683",
    "model/parameter/default/config.yaml": "26b0d40480659a0b15bd6aec4815f892b26df61cf416874c40bfc36b59ddceeb",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(body)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pinned_blob(relative: str) -> bytes:
    blob = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "show", f"32c392a6ece276f1ff046d4e30052d0571ff6dc6:{relative}"]
    )
    expected = SOURCE_PINS[relative]
    observed = sha256_bytes(blob)
    if observed != expected:
        raise RuntimeError(f"upstream_blob_hash_mismatch:{relative}:{observed}:{expected}")
    return blob


def _selected_ast(blob: bytes, functions: set[str], classes: set[str]) -> ast.Module:
    module = ast.parse(blob.decode("utf-8"))
    selected = [
        node for node in module.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions)
        or (isinstance(node, ast.ClassDef) and node.name in classes)
    ]
    found_functions = {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found_classes = {node.name for node in selected if isinstance(node, ast.ClassDef)}
    if found_functions != functions or found_classes != classes:
        raise RuntimeError(f"pinned_ast_selection_mismatch:{found_functions}:{found_classes}")
    return ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    ))


def load_pinned_global_extractor() -> tuple[dict[str, Any], dict[str, str]]:
    """Compile only the upstream feature helpers into a side-effect-free namespace."""
    import dataclasses
    from pathlib import Path as _Path

    helper_blob = pinned_blob("data_preparation/helper.py")
    utils_blob = pinned_blob("data_preparation/utils.py")
    util_scope: dict[str, Any] = {"dataclass": dataclasses.dataclass, "np": np}
    util_module = _selected_ast(
        utils_blob,
        {"calc_qubit_index", "calc_supermarq_features"},
        {"SupermarqFeatures"},
    )
    exec(compile(util_module, "<pinned-mali-utils>", "exec"), util_scope)
    helper_scope: dict[str, Any] = {
        "np": np,
        "Path": _Path,
        "QuantumCircuit": QuantumCircuit,
        "PATH_LENGTH": 260,
        "calc_supermarq_features": util_scope["calc_supermarq_features"],
    }
    helper_module = _selected_ast(
        helper_blob,
        {"get_openqasm_gates", "dict_to_featurevector", "create_feature_dict"},
        set(),
    )
    exec(compile(helper_module, "<pinned-mali-helper>", "exec"), helper_scope)
    pinned_supermarq = helper_scope["calc_supermarq_features"]

    def supermarq_public_wire_compat(circuit: QuantumCircuit):
        """Keep pinned formulas while handling OpenQASM3 loose hardware wires.

        The upstream helper maps qubits by walking ``qc.qregs``. Qiskit 2.5.2
        parses hardware identifiers (``$N``) into loose global wires with no
        registers, so its mapping raises even though ``qc.find_bit`` gives the
        exact physical/global index. Use the original function unchanged for
        ordinary named-register circuits and port only its index lookup for
        that documented API case.
        """
        try:
            return pinned_supermarq(circuit)
        except ValueError as exc:
            if circuit.qregs or "Global qubit index for local qubit" not in str(exc):
                raise
        return _supermarq_with_global_wire_indices(circuit)

    helper_scope["calc_supermarq_features"] = supermarq_public_wire_compat
    version = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    ordered_fields = version["global_features"]["ordered_fields"]
    gates = helper_scope["get_openqasm_gates"]()
    if len(gates) != 44 or ordered_fields[:44] != gates:
        raise RuntimeError("pinned_global_order_mismatch")
    return helper_scope, {
        "helper_sha256": sha256_bytes(helper_blob),
        "utils_sha256": sha256_bytes(utils_blob),
    }


def _supermarq_with_global_wire_indices(circuit: QuantumCircuit) -> SimpleNamespace:
    """Pinned SupermarQ equations using Qiskit's public global wire indices."""
    connectivity_collection: list[list[int]] = [[] for _ in range(circuit.num_qubits)]
    liveness_a_matrix = 0
    for instruction in circuit.data:
        if instruction.operation.name in ("barrier", "measure"):
            continue
        qargs = instruction.qubits
        liveness_a_matrix += len(qargs)
        all_indices = [int(circuit.find_bit(q).index) for q in qargs[:1]]
        if len(qargs) == 2:
            all_indices.append(int(circuit.find_bit(qargs[1]).index))
        for qubit_index in all_indices:
            neighbors = all_indices.copy()
            neighbors.remove(qubit_index)
            connectivity_collection[qubit_index].extend(neighbors)

    connectivity = [len(set(neighbors)) for neighbors in connectivity_collection]
    count_ops = circuit.count_ops()
    num_gates = sum(count_ops.values())
    num_gates -= count_ops.get("measure", 0) + count_ops.get("barrier", 0)
    num_multiple_qubit_gates = circuit.num_nonlocal_gates()
    depth = circuit.depth(lambda item: item.operation.name not in ("barrier", "measure"))
    program_communication = sum(connectivity) / (circuit.num_qubits * (circuit.num_qubits - 1))
    if num_multiple_qubit_gates == 0:
        critical_depth = 0.0
    else:
        multi_depth = circuit.depth(
            filter_function=lambda item: len(item.qubits) > 1 and item.operation.name != "barrier"
        )
        critical_depth = multi_depth / num_multiple_qubit_gates
    if num_multiple_qubit_gates > num_gates:
        raise ValueError("unavailable_upstream_metric_range:num_nonlocal_gates_exceeds_gate_count")
    entanglement_ratio = num_multiple_qubit_gates / num_gates
    parallelism = (num_gates / depth - 1) / (circuit.num_qubits - 1)
    liveness = liveness_a_matrix / (depth * circuit.num_qubits)
    values = (program_communication, critical_depth, entanglement_ratio, parallelism, liveness)
    if not all(0 <= value <= 1 for value in values):
        raise ValueError("unavailable_upstream_metric_range:supermarq_value_outside_unit_interval")
    return SimpleNamespace(
        program_communication=program_communication,
        critical_depth=critical_depth,
        entanglement_ratio=entanglement_ratio,
        parallelism=parallelism,
        liveness=liveness,
    )


def extract_global_features(circuit: QuantumCircuit, extractor: dict[str, Any]) -> list[float]:
    """Return the pinned 51-vector or a fail-closed reason."""
    operations = circuit.count_ops()
    gates = extractor["get_openqasm_gates"]()
    g = sum(int(value) for name, value in operations.items() if name not in {"barrier", "measure"})
    if circuit.num_qubits <= 1 or g <= 0:
        raise ValueError("unavailable_degenerate_upstream_metric")
    depth = circuit.depth(lambda item: item[0].name not in ("barrier", "measure"))
    if not depth:
        raise ValueError("unavailable_degenerate_upstream_metric")
    try:
        features = extractor["create_feature_dict"](circuit)
    except AssertionError as exc:
        raise ValueError(f"unavailable_upstream_metric_range:{exc}") from exc
    values = [float(features[name]) for name in gates]
    values.extend(float(features[name]) for name in (
        "num_qubits", "depth", "program_communication", "critical_depth",
        "entanglement_ratio", "parallelism", "liveness",
    ))
    if len(values) != 51 or not all(math.isfinite(value) for value in values):
        raise ValueError("unavailable_nonfinite_or_wrong_width_global_features")
    # The upstream formulas are asserted in the source. Keep an explicit port gate too.
    if any(value < 0 or value > 1 for value in values[46:]) or values[46 + 2] > 1:
        raise ValueError("unavailable_upstream_metric_range")
    return values


def parse_qasm(data: bytes) -> tuple[QuantumCircuit, str]:
    text = data.decode("utf-8")
    if text.lstrip().startswith("OPENQASM 3"):
        return qasm3.loads(text), text
    return QuantumCircuit.from_qasm_str(text), text


def snapshot_t1_t2(row: dict[str, str], registry: dict[str, dict[str, str]]) -> tuple[dict[int, tuple[float, float]], dict[str, Any]]:
    snapshot = registry.get(row["target_snapshot_id"])
    if snapshot is None:
        raise ValueError("unavailable_missing_backend_property:snapshot_id_not_in_registry")
    path = Path(snapshot["properties_path"])
    expected_hash = str(row["target_properties_sha256"])
    if not path.is_file() or sha256_file(path) != expected_hash or snapshot["properties_sha256"] != expected_hash:
        raise ValueError("unavailable_missing_backend_property:properties_hash_mismatch")
    props = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, tuple[float, float]] = {}
    for index, entries in enumerate(props.get("qubits", [])):
        found: dict[str, float] = {}
        raw: dict[str, Any] = {}
        for prop in entries:
            if prop.get("name") not in {"T1", "T2"}:
                continue
            raw[str(prop["name"])] = {"value": prop.get("value"), "unit": prop.get("unit")}
            value = float(prop["value"])
            unit = str(prop.get("unit", "")).strip().lower().replace("μ", "u").replace("µ", "u")
            scale = {"s": 1e6, "ms": 1e3, "us": 1.0, "µs": 1.0, "ns": 1e-3}.get(unit)
            if scale is None:
                raise ValueError(f"unavailable_missing_backend_property:invalid_T_unit:{unit}")
            found[str(prop["name"])] = value * scale
        if set(found) == {"T1", "T2"}:
            result[index] = (found["T1"], found["T2"])
    return result, {
        "snapshot_id": row["target_snapshot_id"],
        "properties_sha256": expected_hash,
        "snapshot_tier": row["snapshot_tier"],
        "source_package": snapshot["package_name"],
        "source_package_version": snapshot["package_version"],
        "t1_t2_unit": "microseconds",
        "raw_units_checked": True,
    }


def compact_graph_record(circuit: QuantumCircuit, physical_t1_t2: dict[int, tuple[float, float]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Port the pinned DAG node encoding on current public Qiskit DAG APIs.

    Node insertion order is checked against the pinned source conventions:
    input/output pairs by global wire order followed by operation nodes in
    circuit order. DAG edges are deduplicated exactly as NetworkX DiGraph does.
    """
    gate_names = [
        "u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx", "cswap", "crx", "cry", "crz",
        "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx", "c4x", "xx_plus_yy", "ecr",
    ]
    gate_lookup = {name: index + 2 for index, name in enumerate(gate_names)}
    clean = RemoveFinalMeasurements()(RemoveBarriers()(circuit))
    for item in clean.data:
        name = str(item.operation.name)
        if getattr(item.operation, "blocks", None) is not None:
            raise ValueError(f"unavailable_graph_control_flow:{name}")
        if name == "measure" or item.clbits or getattr(item.operation, "condition", None) is not None:
            raise ValueError(f"unavailable_graph_unsupported_classical_or_measurement:{name}")
        if name not in gate_lookup:
            raise ValueError(f"unavailable_graph_unknown_gate:{name}")

    dag = circuit_to_dag(clean)
    dag_nodes = list(dag.nodes())
    active_qubits: set[int] = set()
    op_nodes: list[DAGOpNode] = []
    for node in dag_nodes:
        if isinstance(node, DAGOpNode):
            if node.cargs or getattr(node.op, "condition", None) is not None:
                raise ValueError(f"unavailable_graph_classical_operation:{node.name}")
            op_nodes.append(node)
            active_qubits.update(int(clean.find_bit(q).index) for q in node.qargs)

    if not active_qubits:
        raise ValueError("unavailable_graph_no_used_quantum_wire")
    ordered_active = sorted(active_qubits)
    compact_wire = {physical: index for index, physical in enumerate(ordered_active)}

    # Verify the current public DAG node order used by the upstream converter.
    expected_prefix: list[tuple[str, int]] = []
    for physical in range(clean.num_qubits):
        if physical in compact_wire:
            expected_prefix.extend((("in", physical), ("out", physical)))
    observed_prefix: list[tuple[str, int]] = []
    for node in dag_nodes:
        if isinstance(node, DAGOpNode):
            break
        if isinstance(node, (DAGInNode, DAGOutNode)) and node.wire in clean.qubits:
            physical = int(clean.find_bit(node.wire).index)
            if physical in compact_wire:
                observed_prefix.append(("in" if isinstance(node, DAGInNode) else "out", physical))
    if observed_prefix != expected_prefix:
        raise ValueError("unavailable_graph_dag_wire_insertion_order_mismatch")
    if [node for node in dag_nodes if isinstance(node, DAGOpNode)] != op_nodes:
        raise ValueError("unavailable_graph_operation_insertion_order_mismatch")

    num_nodes = 2 * len(ordered_active) + len(op_nodes)
    node_type = np.empty(num_nodes, dtype=np.uint8)
    wire0 = np.full(num_nodes, 255, dtype=np.uint8)
    wire1 = np.full(num_nodes, 255, dtype=np.uint8)
    t1_0 = np.zeros(num_nodes, dtype=np.float32)
    t2_0 = np.zeros(num_nodes, dtype=np.float32)
    t1_1 = np.zeros(num_nodes, dtype=np.float32)
    t2_1 = np.zeros(num_nodes, dtype=np.float32)
    dag_id_to_new = np.full(int(dag.node_counter), -1, dtype=np.int32)

    # Keep input/output nodes interleaved exactly as Qiskit DAG insertion order.
    active_node_idx: dict[int, tuple[int, int]] = {}
    for compact, physical in enumerate(ordered_active):
        in_node, out_node = 2 * compact, 2 * compact + 1
        active_node_idx[physical] = (in_node, out_node)
        node_type[in_node], node_type[out_node] = 0, 1
        wire0[in_node] = wire0[out_node] = compact
        t1, t2 = physical_t1_t2[physical]
        if not (math.isfinite(t1) and math.isfinite(t2) and t1 > 0 and t2 > 0):
            raise ValueError(f"unavailable_missing_backend_property:wire_{physical}_invalid_T1T2")
        t1_0[in_node] = t1_0[out_node] = t1
        t2_0[in_node] = t2_0[out_node] = t2

    op_new_index: dict[int, int] = {}
    for op_index, node in enumerate(op_nodes):
        new_idx = 2 * len(ordered_active) + op_index
        op_new_index[node._node_id] = new_idx
        node_type[new_idx] = gate_lookup[node.name]
        qargs = tuple(int(clean.find_bit(q).index) for q in node.qargs)
        if len(qargs) == 2:
            first, second = qargs
            wire0[new_idx] = compact_wire[first]
            wire1[new_idx] = compact_wire[second]
            first_t = physical_t1_t2[first]
            second_t = physical_t1_t2[second]
            t1_0[new_idx], t2_0[new_idx] = first_t
            t1_1[new_idx], t2_1[new_idx] = second_t
        # Upstream node builder leaves wire/T1T2 slots zero for one- and
        # higher-operand operations; this lossy behavior is preserved.
        dag_id_to_new[node._node_id] = new_idx

    for node in dag_nodes:
        if isinstance(node, (DAGInNode, DAGOutNode)) and node.wire in clean.qubits:
            physical = int(clean.find_bit(node.wire).index)
            if physical in compact_wire:
                mapped = active_node_idx[physical][0 if isinstance(node, DAGInNode) else 1]
                dag_id_to_new[node._node_id] = mapped

    # DAG edges are streamed to keep the 1.4M-operation rows below the memory
    # cap. Sorting unique integer keys implements simple DiGraph deduplication.
    edge_keys: list[int] = []
    for src, dst, _wire in dag.edges():
        src_new = int(dag_id_to_new[src._node_id])
        dst_new = int(dag_id_to_new[dst._node_id])
        if src_new >= 0 and dst_new >= 0:
            edge_keys.append(src_new * num_nodes + dst_new)
    keys = np.unique(np.fromiter(edge_keys, dtype=np.int64, count=len(edge_keys))) if edge_keys else np.empty(0, dtype=np.int64)
    edge_index = np.vstack((keys // num_nodes, keys % num_nodes)).astype(np.int32, copy=False)
    record = {
        "node_type": node_type,
        "wire0": wire0,
        "wire1": wire1,
        "t1_0": t1_0,
        "t2_0": t2_0,
        "t1_1": t1_1,
        "t2_1": t2_1,
        "node_index": np.arange(num_nodes, dtype=np.int32),
        "edge_index": edge_index,
    }
    meta = {
        "node_count": int(num_nodes),
        "edge_count": int(edge_index.shape[1]),
        "operation_count_after_pruning": len(op_nodes),
        "allocated_width": int(circuit.num_qubits),
        "active_width": len(ordered_active),
        "active_physical_wires": ordered_active,
        "wire_map_physical_to_compact": {str(key): value for key, value in compact_wire.items()},
        "node_order": "qiskit_dag_nodes_input_output_wire_order_then_operation_insertion_order",
        "edge_representation": "unique_directed_qiskit_dag_edges; parallel wire edges collapsed as NetworkX DiGraph",
    }
    return record, meta


def saved_s71_graph_signature(record: dict[str, Any]) -> tuple[str, Counter[str], int]:
    """Recompute the compiled-panel graph digest from its pinned S71 pickle.

    This is the same byte protocol used by ``materialize_compiled_unified_qpu_v1``:
    operation order, gate name, ordered global qargs, and parameter-presence bit.
    It lets the fast materializer verify the pickle against the representation row
    without parsing QASM. S71 stores at most four qargs; larger arity is rejected.
    """
    required = {"format", "num_qubits", "num_clbits", "operation_count", "names", "arity", "parameter", "q0", "q1", "q2", "q3"}
    if not required.issubset(record):
        raise ValueError("unavailable_saved_graph_schema_missing_fields")
    if record["format"] != "s71-operation-dag-v1":
        raise ValueError("unavailable_saved_graph_schema_mismatch")
    names = list(record["names"])
    arity = np.asarray(record["arity"], dtype=np.int64)
    parameter = np.asarray(record["parameter"], dtype=np.int64)
    qcols = [np.asarray(record[f"q{index}"], dtype=np.int64) for index in range(4)]
    count = len(names)
    if int(record["operation_count"]) != count or len(arity) != count or len(parameter) != count or any(len(values) != count for values in qcols):
        raise ValueError("unavailable_saved_graph_column_length_mismatch")
    digest = hashlib.sha256()
    digest.update(b"s71-operation-dag-v1\0")
    digest.update(struct.pack(">QQ", int(record["num_qubits"]), int(record["num_clbits"])))
    counts: Counter[str] = Counter()
    for index, name in enumerate(names):
        width = int(arity[index])
        if width > 4:
            raise ValueError(f"unavailable_saved_graph_arity_exceeds_q0_q3_schema:{width}")
        qargs = [int(qcols[qindex][index]) for qindex in range(width)]
        if any(qarg < 0 or qarg >= int(record["num_qubits"]) for qarg in qargs):
            raise ValueError(f"unavailable_saved_graph_qarg_out_of_range:operation_{index}")
        encoded_name = str(name).encode("utf-8")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(qargs)))
        for qarg in qargs:
            digest.update(struct.pack(">q", qarg))
        digest.update(b"\x01" if int(parameter[index]) else b"\x00")
        counts[str(name)] += 1
    return digest.hexdigest(), counts, count


def saved_graph_fast_path_source_gate(source_id: str, representation: dict[str, str]) -> str:
    """Return a source-level proof token for the no-cargs/no-condition fast path.

    The pickle does not serialize cargs or per-instruction conditions. Therefore
    only the two compiled representations whose pinned source construction is
    quantum-only and whose classical operations are terminal measurements may
    use it. Submitted Qonductor circuits always remain on authenticated QASM.
    """
    stage = str(representation.get("lifecycle_stage", ""))
    route = str(representation.get("compiler_route", ""))
    expected = {
        "mali_real_qpu": ("logical_pre_transpile", "canonical_qasm_sha256"),
        "qpack_mcp": ("logical_structure_reconstructed_from_recorded_workflow_configuration", "qpack_reconstruction_qualified"),
    }
    if source_id not in expected or stage != expected[source_id][0]:
        raise ValueError("unavailable_saved_graph_source_classical_semantics_not_certified")
    if route != "deterministic_nominal_target_transpile":
        raise ValueError("unavailable_saved_graph_compiler_route_not_certified")
    try:
        recipe_kind = json.loads(representation.get("source_recipe_json", "{}" )).get("kind")
    except (TypeError, json.JSONDecodeError):
        recipe_kind = None
    if recipe_kind != expected[source_id][1]:
        raise ValueError("unavailable_saved_graph_source_recipe_not_certified")
    return f"pass:{source_id}:{stage}:{route}:{recipe_kind}:static_source_contract_plus_control_op_gate"


def compact_graph_record_from_s71(record: dict[str, Any], physical_t1_t2: dict[int, tuple[float, float]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the exact sparse Ma--Li graph from a verified S71 op record.

    The graph pickle omits barriers/snapshots by contract. It retains measurement
    nodes, so this fast path accepts only terminal-measurement suffixes, which are
    exactly removable by the pinned ``RemoveFinalMeasurements`` pass. Other cases
    must use the QASM path rather than silently changing graph semantics.
    """
    _digest, _counts, operation_count = saved_s71_graph_signature(record)
    names = list(record["names"])
    arity = np.asarray(record["arity"], dtype=np.int64)
    qcols = [np.asarray(record[f"q{index}"], dtype=np.int64) for index in range(4)]
    measurements = [index for index, name in enumerate(names) if str(name) == "measure"]
    first_measurement = min(measurements) if measurements else len(names)
    if any(str(name) != "measure" for name in names[first_measurement:]):
        raise ValueError("unavailable_saved_graph_nonterminal_measurement_requires_qasm")
    retained = [index for index, name in enumerate(names) if str(name) != "measure"]
    kept_names = [str(names[index]) for index in retained]
    kept_qargs = [tuple(int(qcols[qindex][index]) for qindex in range(int(arity[index]))) for index in retained]
    if any(name in {"if_else", "while_loop", "for_loop", "switch_case"} for name in kept_names):
        raise ValueError("unavailable_graph_control_flow:dynamic_control_not_representable_by_static_upstream_encoder")

    gate_names = [
        "u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx", "cswap", "crx", "cry", "crz",
        "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx", "c4x", "xx_plus_yy", "ecr",
    ]
    gate_lookup = {name: index + 2 for index, name in enumerate(gate_names)}
    for name in kept_names:
        if name not in gate_lookup:
            raise ValueError(f"unavailable_graph_unknown_gate:{name}")
    active = sorted({qarg for qargs in kept_qargs for qarg in qargs})
    if not active:
        raise ValueError("unavailable_graph_no_used_quantum_wire")
    compact_wire = {physical: index for index, physical in enumerate(active)}
    for physical in active:
        if physical not in physical_t1_t2:
            raise ValueError(f"unavailable_missing_backend_property:wire_{physical}_invalid_T1T2")

    num_nodes = 2 * len(active) + len(kept_names)
    node_type = np.empty(num_nodes, dtype=np.uint8)
    wire0 = np.full(num_nodes, 255, dtype=np.uint8)
    wire1 = np.full(num_nodes, 255, dtype=np.uint8)
    t1_0 = np.zeros(num_nodes, dtype=np.float32)
    t2_0 = np.zeros(num_nodes, dtype=np.float32)
    t1_1 = np.zeros(num_nodes, dtype=np.float32)
    t2_1 = np.zeros(num_nodes, dtype=np.float32)
    active_node_idx: dict[int, tuple[int, int]] = {}
    for compact, physical in enumerate(active):
        input_node, output_node = 2 * compact, 2 * compact + 1
        active_node_idx[physical] = (input_node, output_node)
        node_type[input_node], node_type[output_node] = 0, 1
        wire0[input_node] = wire0[output_node] = compact
        t1, t2 = physical_t1_t2[physical]
        if not (math.isfinite(t1) and math.isfinite(t2) and t1 > 0 and t2 > 0):
            raise ValueError(f"unavailable_missing_backend_property:wire_{physical}_invalid_T1T2")
        t1_0[input_node] = t1_0[output_node] = t1
        t2_0[input_node] = t2_0[output_node] = t2

    edge_keys: list[int] = []
    last_by_qubit: dict[int, int] = {}
    for operation_index, (name, qargs) in enumerate(zip(kept_names, kept_qargs)):
        new_index = 2 * len(active) + operation_index
        node_type[new_index] = gate_lookup[name]
        if len(qargs) == 2:
            first, second = qargs
            wire0[new_index], wire1[new_index] = compact_wire[first], compact_wire[second]
            first_t, second_t = physical_t1_t2[first], physical_t1_t2[second]
            t1_0[new_index], t2_0[new_index] = first_t
            t1_1[new_index], t2_1[new_index] = second_t
        for physical in qargs:
            previous = last_by_qubit.get(physical, active_node_idx[physical][0])
            edge_keys.append(previous * num_nodes + new_index)
            last_by_qubit[physical] = new_index
    for physical in active:
        edge_keys.append(last_by_qubit[physical] * num_nodes + active_node_idx[physical][1])
    unique_keys = np.unique(np.asarray(edge_keys, dtype=np.int64)) if edge_keys else np.empty(0, dtype=np.int64)
    edge_index = np.vstack((unique_keys // num_nodes, unique_keys % num_nodes)).astype(np.int32, copy=False)
    graph = {
        "node_type": node_type,
        "wire0": wire0,
        "wire1": wire1,
        "t1_0": t1_0,
        "t2_0": t2_0,
        "t1_1": t1_1,
        "t2_1": t2_1,
        "node_index": np.arange(num_nodes, dtype=np.int32),
        "edge_index": edge_index,
    }
    metadata = {
        "node_count": int(num_nodes),
        "edge_count": int(edge_index.shape[1]),
        "operation_count_after_pruning": len(kept_names),
        "allocated_width": int(record["num_qubits"]),
        "active_width": len(active),
        "active_physical_wires": active,
        "wire_map_physical_to_compact": {str(key): value for key, value in compact_wire.items()},
        "node_order": "qiskit_dag_nodes_input_output_wire_order_then_operation_insertion_order",
        "edge_representation": "unique_directed_qiskit_dag_edges; parallel wire edges collapsed as NetworkX DiGraph",
    }
    return graph, metadata


def extract_global_features_from_s71(record: dict[str, Any], representation: dict[str, str], extractor: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """Recreate the exact pinned 51-vector from a verified S71 record + sidecar.

    ``representation_rows`` supplies the compiled-circuit gate counts and full
    Qiskit depth (including barriers); the S71 record supplies ordered qargs for
    SupermarQ connectivity/depth/liveness. If either component is not sufficient,
    callers must fall back to parsing authenticated QASM.
    """
    _digest, observed_counts, operation_count = saved_s71_graph_signature(record)
    gate_counts = json.loads(representation["gate_counts_json"])
    classical_control_names = {"if_else", "while_loop", "for_loop", "switch_case"}
    found_control = classical_control_names.intersection(observed_counts) | classical_control_names.intersection(gate_counts)
    if found_control:
        raise ValueError(f"unavailable_saved_graph_classical_control_not_encoded:{sorted(found_control)}")
    if int(representation["register_width"]) != int(record["num_qubits"]):
        raise ValueError("unavailable_saved_graph_register_width_mismatch")
    if int(representation["operation_count"]) != operation_count:
        raise ValueError("unavailable_saved_graph_operation_count_mismatch")
    if sum(observed_counts.values()) != operation_count:
        raise ValueError("unavailable_saved_graph_operation_count_mismatch")
    if int(gate_counts.get("snapshot", 0)):
        raise ValueError("unavailable_saved_graph_snapshot_not_encoded")
    for name, count in observed_counts.items():
        if int(gate_counts.get(name, 0)) != count:
            raise ValueError(f"unavailable_saved_graph_gate_count_mismatch:{name}")
    extras = set(gate_counts) - set(observed_counts) - {"barrier"}
    if extras:
        raise ValueError(f"unavailable_saved_graph_gate_names_missing:{sorted(extras)}")
    width = int(record["num_qubits"])
    if width <= 1:
        raise ValueError("unavailable_degenerate_upstream_metric")
    gate_names = extractor["get_openqasm_gates"]()
    values = [float(gate_counts.get(name, 0)) for name in gate_names]
    values.extend((float(width), float(representation["structural_depth"])))

    qcols = [np.asarray(record[f"q{index}"], dtype=np.int64) for index in range(4)]
    arity = np.asarray(record["arity"], dtype=np.int64)
    names = list(record["names"])
    connectivity: list[set[int]] = [set() for _ in range(width)]
    liveness_a_matrix = 0
    num_gates = 0
    multi_qubit_gates = 0
    depth_per_wire = [0] * width
    multi_depth_per_wire = [0] * width
    for index, name_value in enumerate(names):
        name = str(name_value)
        count = int(arity[index])
        qargs = tuple(int(qcols[qindex][index]) for qindex in range(count))
        if name in {"barrier", "measure"}:
            continue
        if not qargs:
            raise ValueError(f"unavailable_saved_graph_zero_qarg_operation:{name}")
        liveness_a_matrix += len(qargs)
        num_gates += 1
        if len(qargs) > 1:
            multi_qubit_gates += 1
        # Exact pinned SupermarQ quirk: only 2-operand operations contribute a
        # symmetric connectivity pair; higher-arity operations contribute none.
        connectivity_indices = qargs[:1] + (qargs[1:2] if len(qargs) == 2 else ())
        for qubit in connectivity_indices:
            connectivity[qubit].update(other for other in connectivity_indices if other != qubit)
        next_depth = max(depth_per_wire[qubit] for qubit in qargs) + 1
        for qubit in qargs:
            depth_per_wire[qubit] = next_depth
        if len(qargs) > 1:
            next_multi_depth = max(multi_depth_per_wire[qubit] for qubit in qargs) + 1
            for qubit in qargs:
                multi_depth_per_wire[qubit] = next_multi_depth
    depth = max(depth_per_wire, default=0)
    multi_depth = max(multi_depth_per_wire, default=0)
    if num_gates <= 0 or depth <= 0:
        raise ValueError("unavailable_degenerate_upstream_metric")
    program_communication = sum(len(neighbors) for neighbors in connectivity) / (width * (width - 1))
    critical_depth = multi_depth / multi_qubit_gates if multi_qubit_gates else 0.0
    entanglement_ratio = multi_qubit_gates / num_gates
    parallelism = (num_gates / depth - 1) / (width - 1)
    liveness = liveness_a_matrix / (depth * width)
    tail = [program_communication, critical_depth, entanglement_ratio, parallelism, liveness]
    if not all(math.isfinite(value) for value in values + tail) or any(not 0 <= value <= 1 for value in tail):
        raise ValueError("unavailable_upstream_metric_range")
    values.extend(tail)
    if len(values) != 51:
        raise ValueError("unavailable_nonfinite_or_wrong_width_global_features")
    return values, {
        "signature_source": "sha256-verified S71 graph pickle plus pinned representation_rows gate counts/depth",
        "operation_digest": _digest,
        "operation_count": operation_count,
        "supermarq_filtered_depth": depth,
        "supermarq_multi_qubit_depth": multi_depth,
    }


def expand_dense_x(record: dict[str, np.ndarray]) -> np.ndarray:
    """Expand one sparse record to the exact 178-float node matrix."""
    node_type = record["node_type"]
    count = len(node_type)
    x = np.zeros((count, 178), dtype=np.float32)
    x[np.arange(count), node_type] = 1.0
    wire0 = record["wire0"]
    wire1 = record["wire1"]
    rows = np.flatnonzero(wire0 != 255)
    x[rows, 46 + wire0[rows]] = 1.0
    rows = np.flatnonzero(wire1 != 255)
    x[rows, 46 + wire1[rows]] = 1.0
    x[:, 173] = record["t1_0"]
    x[:, 174] = record["t2_0"]
    x[:, 175] = record["t1_1"]
    x[:, 176] = record["t2_1"]
    x[:, 177] = record["node_index"]
    return x


def load_snapshot_registry() -> dict[str, dict[str, str]]:
    return {row["snapshot_id"]: row for row in read_csv(SNAPSHOT_REGISTRY)}


def load_input_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    canonical = read_csv(CANONICAL)
    representations = read_csv(REPRESENTATION)
    by_representation = {row["canonical_row_id"]: row for row in representations}
    by_canonical = {row["canonical_row_id"]: row for row in canonical}
    if len(canonical) != 8767 or len(representations) != 8767 or set(by_canonical) != set(by_representation):
        raise RuntimeError("canonical_representation_identity_mismatch")
    return canonical, by_canonical, by_representation


def qasm_bytes_for(row: dict[str, str], representation: dict[str, str], qonductor_zip: zipfile.ZipFile) -> tuple[bytes, str, dict[str, Any]]:
    source = row["source_id"]
    if source in {"mali_real_qpu", "qpack_mcp"}:
        relative = representation["compiled_qasm3_file"]
        expected = representation["compiled_qasm3_sha256"]
        path = COMPILED_ROOT / relative
        if not relative or not path.is_file():
            raise ValueError("unavailable_source_input_missing:compiled_qasm3")
        data = path.read_bytes()
        if sha256_bytes(data) != expected:
            raise RuntimeError(f"pinned_compiled_qasm_hash_mismatch:{row['canonical_row_id']}")
        return data, expected, {"path": str(path.relative_to(ROOT)), "stage": representation["lifecycle_stage"]}
    if source == "qonductor_single_circuit_ibm":
        member = row["qasm_path_or_member"]
        data = qonductor_zip.read(member)
        expected = row["qasm_bytes_sha256"]
        if sha256_bytes(data) != expected:
            raise RuntimeError(f"pinned_qonductor_member_hash_mismatch:{row['canonical_row_id']}")
        return data, expected, {"member": member, "stage": "submitted_physical"}
    raise ValueError(f"unknown_source:{source}")


def save_sparse_graph(record: dict[str, np.ndarray], graph_key: str) -> tuple[str, str]:
    OUT_ROOT.joinpath("graphs").mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "graphs" / f"{graph_key}.npz"
    if not path.exists():
        temporary = path.with_suffix(".npz.partial")
        np.savez_compressed(temporary, **record)
        generated = temporary if temporary.is_file() else temporary.with_suffix(temporary.suffix + ".npz")
        if not generated.is_file():
            raise RuntimeError("sparse_graph_write_failed")
        os.replace(generated, path)
    return str(path.relative_to(OUT_ROOT)), sha256_file(path)
