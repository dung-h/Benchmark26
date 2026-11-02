"""Independent, label-free Qonductor extraction tasks for bounded CPU workers."""
from __future__ import annotations

from collections import Counter
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

_STATE: dict[str, Any] = {}
FATAL_PREFIXES = (
    "pinned_input_or_source_hash_drift:", "compiled_identity_mismatch:",
    "matrix_order_mismatch:", "golden_parity_failure:",
    "canonical_or_split_identity_drift:", "group_leakage:",
    "target_in_feature_construction:",
)


def initialize_feature_worker(config: dict[str, Any]) -> None:
    """Load the exact source methods once per process without fitting models."""
    engine = importlib.import_module("materialize_qonductor_native_features")
    for name, pin in config["implementation_pins"].items():
        engine.require_hash(Path(pin["path"]), pin["sha256"], name)
    limit = int(config["native_threads"])
    if limit not in (1, 2):
        raise ValueError("feature workers require one or two native threads")
    extractor, _loader, parser, _metadata = engine.load_pinned_qonductor_methods(
        Path(config["source_root"]), config["source_pin"],
    )
    reference = engine.import_materializer(Path(config["materializer_reference"]))
    if "archive" in _STATE:
        _STATE["archive"].close()
    if "thread_limit" in _STATE:
        _STATE["thread_limit"].restore_original_limits()
    _STATE.clear()
    _STATE.update(
        engine=engine, extractor=extractor, parser=parser, reference=reference,
        archive=zipfile.ZipFile(config["archive"]),
        thread_limit=threadpool_limits(limits=limit),
    )


def extraction_key(source_id: str, qasm_hash: str, width: str, shots: str) -> tuple[str, str, str, str]:
    # Shots are part of the returned vector, including for saved compiled QASM.
    return source_id, qasm_hash, width, str(int(shots))


def bind_feature_metadata(metadata: dict[str, Any], row: dict[str, str], rep: dict[str, str]) -> dict[str, Any]:
    """Rebind duplicate input aliases while checking their saved expectations."""
    bound = dict(metadata)
    if row["source_id"] == "qonductor_single_circuit_ibm":
        bound["member"] = row["qasm_path_or_member"]
    else:
        expected = {
            "register_width": int(rep["register_width"]),
            "graph_input_digest": rep["graph_input_digest"],
            "verified_gate_counts": json.loads(rep["gate_counts_json"]),
            "verified_measurement_count": int(rep["measurement_count"]),
        }
        if any(bound.get(field) != value for field, value in expected.items()):
            raise RuntimeError(f"compiled_identity_mismatch:cached metadata:{row['canonical_row_id']}")
        bound["compiled_qasm3_file"] = rep["compiled_qasm3_file"]
    return bound


def plan_feature_tasks(
    rows: list[dict[str, str]], representation: dict[str, dict[str, str]],
    representation_root: Path, archive_path: Path,
) -> list[dict[str, Any]]:
    """Verify every named input, then group identical extraction requests.

    Only input identities/settings enter worker payloads. Runtime labels and
    source-specific outcomes are never used for grouping or scheduling.
    """
    engine = importlib.import_module("materialize_qonductor_native_features")
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    verified: dict[tuple[str, str], int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for row in rows:
            source = row["source_id"]
            rep = representation[row["canonical_row_id"]]
            try:
                shots = str(int(row["shots"]))
                if int(shots) <= 0:
                    raise ValueError("invalid_canonical_shots")
                if source == "qonductor_single_circuit_ibm":
                    route = "exact_submitted_physical_qasm"
                    member = row["qasm_path_or_member"]
                    expected = row["qasm_bytes_sha256"]
                    locator = (member, expected)
                    if locator not in verified:
                        try:
                            raw = archive.read(member)
                        except KeyError as exc:
                            raise FileNotFoundError(f"source_input_missing:{member}") from exc
                        if engine.sha256_bytes(raw) != expected:
                            raise RuntimeError(f"pinned_input_or_source_hash_drift:{member}")
                        verified[locator] = len(raw)
                    key = extraction_key(source, expected, "", shots)
                    spec = {"member": member}
                else:
                    route = "saved_nominal_target_compiled_qasm3"
                    path = engine.resolve_saved_compiled_qasm_path(representation_root, rep["compiled_qasm3_file"])
                    expected = rep["compiled_qasm3_sha256"]
                    locator = (str(path), expected)
                    if locator not in verified:
                        raw = path.read_bytes()
                        if engine.sha256_bytes(raw) != expected:
                            raise RuntimeError(f"compiled_identity_mismatch:{path}")
                        verified[locator] = len(raw)
                    key = extraction_key(source, expected, str(int(rep["register_width"])), shots)
                    spec = {
                        "path": str(path), "compiled_qasm3_file": rep["compiled_qasm3_file"],
                        "register_width": int(rep["register_width"]),
                        "graph_input_digest": rep["graph_input_digest"],
                        "gate_counts": json.loads(rep["gate_counts_json"]),
                        "measurement_count": int(rep["measurement_count"]),
                    }
                if key in groups:
                    previous = groups[key]["input"]
                    if route == "saved_nominal_target_compiled_qasm3" and any(
                        previous[field] != spec[field] for field in
                        ("register_width", "graph_input_digest", "gate_counts", "measurement_count")
                    ):
                        raise RuntimeError(f"compiled_identity_mismatch:duplicate expectations:{key}")
                    groups[key]["row_ids"].append(row["canonical_row_id"])
                else:
                    size = verified[locator]
                    groups[key] = {
                        "cache_key": list(key), "source_id": source, "shots": int(shots),
                        "source_input_stage": route, "qasm_sha256": expected,
                        "input": spec, "input_bytes": size, "exclusive": size >= 8 * 1024 ** 2,
                        "row_ids": [row["canonical_row_id"]],
                    }
            except (FileNotFoundError, ValueError) as exc:
                key = ("unavailable", row["canonical_row_id"])
                groups[key] = {"row_ids": [row["canonical_row_id"]], "input_error": f"{type(exc).__name__}:{exc}", "input_bytes": 0}
    return sorted(groups.values(), key=lambda item: -item["input_bytes"])


def extract_feature_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run the same parser, AST checks and pinned extractor as the serial route."""
    if "input_error" in task:
        return {"error": task["input_error"], "stage_timings_seconds": {}, "execution_pid": os.getpid()}
    state = _STATE
    engine = state["engine"]
    timings: dict[str, float] = {}
    try:
        source = task["source_id"]
        spec = task["input"]
        shots = int(task["shots"])
        started = time.perf_counter()
        if source == "qonductor_single_circuit_ibm":
            raw = state["archive"].read(spec["member"])
            if engine.sha256_bytes(raw) != task["qasm_sha256"]:
                raise RuntimeError("pinned_input_or_source_hash_drift:worker archive input")
            timings["source_qasm_read_and_hash_seconds"] = time.perf_counter() - started
            vector, metadata, parse_s, extract_s = engine.extract_qonductor_archive_features(
                raw, task["qasm_sha256"], spec["member"], shots, {}, state["parser"], state["extractor"],
            )
            timings["pinned_qasm_parse_seconds"] = parse_s
            timings["pinned_qonductor_extractor_seconds"] = extract_s
        else:
            raw = Path(spec["path"]).read_bytes()
            if engine.sha256_bytes(raw) != task["qasm_sha256"]:
                raise RuntimeError("compiled_identity_mismatch:worker saved input")
            timings["saved_qasm_read_and_hash_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            try:
                restored, restoration = engine.restore_saved_qasm3_hardware_wires(raw.decode("utf-8"), spec["register_width"])
            except ValueError as exc:
                if str(exc).startswith("cannot parse saved QASM3 AST:"):
                    raise
                raise RuntimeError(f"compiled_identity_mismatch:hardware_wire_AST:{exc}") from exc
            timings["ast_restore_parse_rewrite_roundtrip_and_classical_check_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            circuit = engine.qasm3.loads(restored)
            timings["qiskit_qasm3_parse_seconds"] = time.perf_counter() - started
            if circuit.num_qubits != spec["register_width"]:
                raise RuntimeError("compiled_identity_mismatch:allocated_width")
            engine.check_native_condition_support(circuit)
            started = time.perf_counter()
            reference = state["reference"]
            digest = engine.graph_digest_only(circuit, reference.GRAPH_FORMAT, reference.SKIP_NAMES)
            if digest != spec["graph_input_digest"]:
                raise RuntimeError("compiled_identity_mismatch:graph_input_digest")
            timings["streaming_graph_digest_check_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            counts = Counter(str(inst.operation.name) for inst in circuit.data)
            if dict(counts) != spec["gate_counts"]:
                raise RuntimeError("compiled_identity_mismatch:gate_counts_json")
            if counts.get("measure", 0) != spec["measurement_count"]:
                raise RuntimeError("compiled_identity_mismatch:measurement_count")
            timings["gate_and_measurement_count_checks_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            vector = state["extractor"]._extract_features([circuit], shots).tolist()[0]
            timings["pinned_qonductor_extractor_seconds"] = time.perf_counter() - started
            metadata = {
                **restoration, "compiled_qasm3_file": spec["compiled_qasm3_file"],
                "register_width": spec["register_width"], "graph_input_digest": digest,
                "gate_counts_match": True, "measurement_count_match": True,
                "verified_gate_counts": dict(counts), "verified_measurement_count": counts.get("measure", 0),
                "parser_route": "saved QASM3 after AST hardware-wire restoration",
            }
        vector = [float(value) for value in vector]
        if len(vector) != 5 or not all(np.isfinite(value) for value in vector):
            raise RuntimeError("native_extractor_exception:nonfinite_or_malformed_feature_vector")
        if vector[3:] != [float(shots), 1.0]:
            raise RuntimeError("matrix_order_mismatch:shots_or_circuit_count")
        return {"vector": vector, "metadata": metadata, "stage_timings_seconds": timings, "execution_pid": os.getpid()}
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as exc:
        if any(str(exc).startswith(prefix) for prefix in FATAL_PREFIXES):
            raise
        if isinstance(exc, (KeyError, AttributeError, TypeError)):
            raise RuntimeError(f"feature_worker_implementation_error:{type(exc).__name__}:{exc}") from exc
        return {"error": f"{type(exc).__name__}:{exc}", "stage_timings_seconds": timings, "execution_pid": os.getpid()}
