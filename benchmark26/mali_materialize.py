#!/usr/bin/env python3
"""Materialize pinned 51-global and 178-node Ma--Li inputs for the common panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qiskit import qasm3

from benchmark26.paths import ARTIFACTS, NEURAL_PYTHON

ROOT = Path(__file__).resolve().parents[1]
from benchmark26 import mali_features as _mali_features
from benchmark26.mali_features import (
    CANONICAL,
    COMPILED_ROOT,
    HASH_MANIFEST,
    INNER,
    OUT_ROOT,
    OUTER,
    PROTOCOL_PATH,
    QONDUCTOR_ZIP,
    REPRESENTATION,
    SNAPSHOT_REGISTRY,
    SOURCE_PINS,
    UPSTREAM_ROOT,
    compact_graph_record,
    compact_graph_record_from_s71,
    extract_global_features,
    extract_global_features_from_s71,
    load_input_rows,
    load_pinned_global_extractor,
    load_snapshot_registry,
    parse_qasm,
    qasm_bytes_for,
    save_sparse_graph,
    saved_graph_fast_path_source_gate,
    saved_s71_graph_signature,
    sha256_bytes,
    sha256_file,
    snapshot_t1_t2,
    stable_hash,
)


def atomic_json(path: Path, value: Any) -> None:
    """Write a JSON file through a temporary `.partial` file, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def implementation_hashes() -> dict[str, str]:
    """Pin each local code file that can change materialized feature values."""
    paths = {
        "materializer": Path(__file__).resolve(),
        "feature_library": Path(_mali_features.__file__).resolve(),
        "qasm3_hardware_wire_adapter": ROOT / "benchmark26/qasm3_wires.py",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"feature_implementation_source_missing:{name}:{path}")
        result[name] = sha256_file(path)
    return result


def input_manifest() -> dict[str, Any]:
    """Verify the frozen protocol, the execution authorization and the pinned source files, and return the input manifest.

    Raises if a hash differs, the authorization status is not the expected one, or
    the limit on GPU training workers is not 1.
    """
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    auth_path = ARTIFACTS / "unified_runtime_estimation/execution_authorization.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    expected_auth = auth["method_contracts"]["mali_full_features"]
    checks = {
        "protocol": (PROTOCOL_PATH, expected_auth["sha256"]),
        "canonical": (CANONICAL, protocol["input_pins"]["canonical"]["sha256"]),
        "representation_rows": (REPRESENTATION, protocol["input_pins"]["representation_rows"]["sha256"]),
        "representation_manifest": (
            COMPILED_ROOT / "manifest.json",
            protocol["input_pins"]["representation_manifest"]["sha256"],
        ),
        "graph_hash_manifest": (HASH_MANIFEST, protocol["input_pins"]["graph_hash_manifest"]["sha256"]),
        "snapshot_registry": (SNAPSHOT_REGISTRY, protocol["input_pins"]["snapshot_registry"]["sha256"]),
        "outer_split": (OUTER, protocol["input_pins"]["outer_split"]["sha256"]),
        "inner_split": (INNER, protocol["input_pins"]["inner_split"]["sha256"]),
        "execution_authorization": (auth_path, None),
        "qonductor_archive": (QONDUCTOR_ZIP, protocol["external_assets"]["qonductor_archive"]["sha256"]),
    }
    observed = {}
    for name, (path, expected) in checks.items():
        if not path.is_file():
            raise RuntimeError(f"pinned_input_missing:{name}:{path}")
        actual = sha256_file(path)
        if expected and actual != expected:
            raise RuntimeError(f"pinned_input_hash_mismatch:{name}:{actual}:{expected}")
        observed[name] = {"path": os.path.relpath(path, ROOT), "sha256": actual}
    if auth.get("status") != "authorized_implementation_and_execution_subject_to_technical_gates":
        raise RuntimeError("bounded_execution_authorization_missing")
    if auth.get("limits", {}).get("gpu_training_workers") != 1:
        raise RuntimeError("execution_authorization_worker_limit_mismatch")
    source_blobs = {}
    for relative, expected in SOURCE_PINS.items():
        blob = subprocess.check_output(
            ["git", "-C", str(UPSTREAM_ROOT), "show", f"32c392a6ece276f1ff046d4e30052d0571ff6dc6:{relative}"]
        )
        actual = sha256_bytes(blob)
        if actual != expected:
            raise RuntimeError(f"upstream_blob_hash_mismatch:{relative}:{actual}:{expected}")
        source_blobs[relative] = actual
    if (
        not auth.get("scope")
        or "full-feature Ma-Li graph and matched MLP across the frozen 8,767-row QPU panel" not in auth["scope"]
    ):
        raise RuntimeError("authorization_scope_does_not_include_mali_full_panel")
    implementation = implementation_hashes()
    return {
        "status": "PASS",
        "protocol_id": protocol["contract_id"],
        "protocol_sha256": observed["protocol"]["sha256"],
        "execution_authorization_sha256": observed["execution_authorization"]["sha256"],
        "inputs": observed,
        "upstream_revision": protocol["source"]["revision"],
        "upstream_blob_sha256": source_blobs,
        "assigned_observations": 8767,
        "identity_column": "canonical_row_id",
        "target_column_read_by_feature_code": False,
        "split_hashes_verified": True,
        "archive_sha256_verified": True,
        "implementation_hashes": implementation,
        "feature_implementation_sha256": stable_hash(implementation),
    }


def environment_lock() -> dict[str, Any]:
    """Record the Python environments used for features and for neural training.

    The installed packages of each interpreter are frozen into a dependency lock
    file under OUT_ROOT (an existing file is kept). Returns the file hashes and the
    versions of the key packages.
    """
    from importlib import metadata

    def freeze_from_metadata(python: str) -> str:
        """Return sorted `name==version` lines for every package installed in the given Python."""
        code = (
            "import importlib.metadata as m; "
            "items=sorted((d.metadata.get('Name','').lower().replace('_','-'), d.version) "
            "for d in m.distributions() if d.metadata.get('Name')); "
            "print(''.join(f'{name}=={version}\\n' for name,version in items), end='')"
        )
        return subprocess.run([python, "-c", code], check=True, capture_output=True, text=True).stdout

    freeze = freeze_from_metadata(sys.executable)
    packages = {}
    for name in ("numpy", "qiskit", "qiskit-ibm-runtime", "qiskit-qasm3-import", "openqasm3", "networkx", "rustworkx"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    lock_dir = OUT_ROOT
    lock_dir.mkdir(parents=True, exist_ok=True)
    feature_path = lock_dir / "dependency_lock_feature.txt"
    if not feature_path.exists():
        temp = feature_path.with_suffix(".txt.partial")
        temp.write_text(freeze, encoding="utf-8")
        os.replace(temp, feature_path)
    gpu_python = NEURAL_PYTHON
    gpu_freeze = freeze_from_metadata(str(gpu_python))
    gpu_path = lock_dir / "dependency_lock_neural.txt"
    if not gpu_path.exists():
        temp = gpu_path.with_suffix(".txt.partial")
        temp.write_text(gpu_freeze, encoding="utf-8")
        os.replace(temp, gpu_path)
    return {
        "feature_python": sys.executable,
        "feature_python_version": platform.python_version(),
        "feature_packages": packages,
        "feature_lock_sha256": sha256_file(feature_path),
        "neural_python": str(gpu_python),
        "neural_lock_sha256": sha256_file(gpu_path),
        "neural_lock_file": gpu_path.name,
        "feature_lock_file": feature_path.name,
    }


def saved_graph_signature(circuit, repr_row: dict[str, str]) -> dict[str, Any]:
    """Recompute the pinned S71 input digest and operation inventory from QASM."""
    import struct
    from collections import Counter

    skip = {"barrier", "snapshot"}
    digest = hashlib.sha256()
    digest.update(b"s71-operation-dag-v1\0")
    digest.update(struct.pack(">QQ", int(circuit.num_qubits), int(circuit.num_clbits)))
    counts: Counter[str] = Counter()
    measurements = 0
    operations = 0
    for item in circuit.data:
        name = str(item.operation.name)
        counts[name] += 1
        if name == "measure":
            measurements += 1
        if name in skip:
            continue
        qargs = tuple(int(circuit.find_bit(qubit).index) for qubit in item.qubits)
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(qargs)))
        for qarg in qargs:
            digest.update(struct.pack(">q", qarg))
        digest.update(b"\x01" if bool(getattr(item.operation, "params", ())) else b"\x00")
        operations += 1
    expected_counts = json.loads(repr_row["gate_counts_json"])
    if dict(counts) != expected_counts:
        raise RuntimeError(f"feature_signature_gate_failed:gate_counts:{dict(counts)}:{expected_counts}")
    if measurements != int(repr_row["measurement_count"]):
        raise RuntimeError("feature_signature_gate_failed:measurement_count")
    if operations != int(repr_row["operation_count"]):
        raise RuntimeError(f"feature_signature_gate_failed:operation_count:{operations}:{repr_row['operation_count']}")
    observed_digest = digest.hexdigest()
    if observed_digest != repr_row["graph_input_digest"]:
        raise RuntimeError(
            f"feature_signature_gate_failed:graph_input_digest:{observed_digest}:{repr_row['graph_input_digest']}"
        )
    return {
        "graph_input_digest": observed_digest,
        "allocated_width": int(circuit.num_qubits),
        "classical_width": int(circuit.num_clbits),
        "operation_count": operations,
        "measurement_count": measurements,
        "gate_counts": dict(counts),
        "signature_source": "pinned QASM bytes parsed by qiskit 2.5.2; exact S71 operation-dag digest plus full gate counts/classical AST preservation",
    }


def verify_source_graph_file(
    repr_row: dict[str, str], graph_hashes: dict[str, str], verified: set[str]
) -> tuple[str, str]:
    """Check that the saved operation graph of a representation row exists and matches its recorded SHA-256.

    Returns the file name and expected hash, or two empty strings when the row has
    no graph file and none is required. `verified` remembers the files already
    hashed, so each file is read once.
    """
    name = str(repr_row.get("graph_file", ""))
    if not name:
        if repr_row.get("graph_status") == "available":
            raise RuntimeError(f"source_graph_file_missing_from_available_row:{repr_row['canonical_row_id']}")
        return "", ""
    expected = graph_hashes.get(name)
    if not expected:
        raise RuntimeError(f"source_graph_file_missing_from_hash_manifest:{name}")
    if name not in verified:
        path = COMPILED_ROOT / "graphs" / name
        if not path.is_file():
            raise RuntimeError(f"source_graph_file_missing:{name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"source_graph_file_hash_mismatch:{name}:{actual}:{expected}")
        verified.add(name)
    return name, expected


def make_row_attempt(
    row: dict[str, str],
    repr_row: dict[str, str],
    qond_zip,
    extractor,
    registry,
    cache: dict,
    graph_hashes: dict[str, str],
    verified_graph_files: set[str],
    input_route_mode: str = "qasm_parse",
    code_sha256: str = "",
    implementation_hashes_by_name: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Materialize the features of one canonical observation and return its ledger row.

    Reads the circuit bytes, verifies the saved graph, and extracts the global and
    graph features with the pinned extractor. Results are cached by (QASM hash,
    register width, property hash), so identical inputs are computed once; a cache
    hit is marked `feature_cache_reuse`. With `input_route_mode` set to
    `verified_graph_fast_path`, hash-verified saved Ma-Li and QPack operation graphs
    replace the QASM parse and the exact parser is the fallback; Qonductor rows
    always use the parser. A hash or digest violation, a width mismatch or a failed
    signature gate raises an error and never falls back. Other failures are recorded
    as reasons in the row.
    """
    if input_route_mode not in {"qasm_parse", "verified_graph_fast_path"}:
        raise ValueError(f"unknown_input_route_mode:{input_route_mode}")
    canonical_id = row["canonical_row_id"]
    source = row["source_id"]
    qasm_data, qasm_sha, source_ref = qasm_bytes_for(row, repr_row, qond_zip)
    source_graph_file, source_graph_sha = verify_source_graph_file(repr_row, graph_hashes, verified_graph_files)
    expected_width = int(repr_row["register_width"])
    snapshot_id = repr_row["target_snapshot_id"]
    property_hash = repr_row["target_properties_sha256"]
    cache_key = (qasm_sha, expected_width, property_hash)
    base = {
        "canonical_row_id": canonical_id,
        "source_id": source,
        "backend": repr_row["backend_canonical"],
        "source_identity_kind": row["identity_kind"],
        "source_stage": repr_row["lifecycle_stage"],
        "representation_id": repr_row["representation_id"],
        "qasm_input_sha256": qasm_sha,
        "qasm_source_ref_json": json.dumps(source_ref, sort_keys=True, separators=(",", ":")),
        "materialization_route": "qasm_parse",
        "materialization_route_reason": "default_exact_qasm_path"
        if input_route_mode == "qasm_parse"
        else "pending_verified_graph_eligibility",
        "materialization_semantics_gate": "qasm_parser_preserves_operation_cargs_conditions_and_control_flow",
        "materializer_code_sha256": code_sha256,
        "feature_library_code_sha256": (implementation_hashes_by_name or {}).get("feature_library", ""),
        "qasm3_wire_adapter_code_sha256": (implementation_hashes_by_name or {}).get("qasm3_hardware_wire_adapter", ""),
        "register_width": expected_width,
        "snapshot_id": snapshot_id,
        "snapshot_tier": repr_row["snapshot_tier"],
        "properties_sha256": property_hash,
        "source_graph_file": source_graph_file,
        "source_graph_file_sha256": source_graph_sha,
        "global_status": "unavailable",
        "global_reason": "not_attempted",
        "graph_status": "unavailable",
        "graph_reason": "not_attempted",
        "graph_key": "",
        "graph_artifact": "",
        "graph_artifact_sha256": "",
        "raw_global_width": 51,
        "effective_global_width_before_fold_mask": 51,
    }
    if cache_key in cache:
        cached = cache[cache_key]
        base.update(cached)
        base["canonical_row_id"] = canonical_id
        base["source_id"] = source
        base["backend"] = repr_row["backend_canonical"]
        base["source_identity_kind"] = row["identity_kind"]
        base["source_stage"] = repr_row["lifecycle_stage"]
        base["representation_id"] = repr_row["representation_id"]
        base["qasm_input_sha256"] = qasm_sha
        base["qasm_source_ref_json"] = json.dumps(source_ref, sort_keys=True, separators=(",", ":"))
        base["materialization_route"] = "feature_cache_reuse"
        base["materialization_route_reason"] = str(cached.get("materialization_route", "unknown_cached_route"))
        base["materialization_semantics_gate"] = str(cached.get("materialization_semantics_gate", "cached_result"))
        base["materializer_code_sha256"] = code_sha256
        base["feature_library_code_sha256"] = (implementation_hashes_by_name or {}).get("feature_library", "")
        base["qasm3_wire_adapter_code_sha256"] = (implementation_hashes_by_name or {}).get(
            "qasm3_hardware_wire_adapter", ""
        )
        return base

    if input_route_mode == "verified_graph_fast_path":
        # Submitted Qonductor QASM may carry cargs/conditions that are not encoded
        # in the S71 pickle. Keep every Qonductor row on the exact parser path.
        try:
            source_semantics_gate = saved_graph_fast_path_source_gate(source, repr_row)
        except ValueError as exc:
            source_semantics_gate = ""
            source_semantics_reason = str(exc)
        else:
            source_semantics_reason = ""
        if not source_semantics_gate:
            base["materialization_route"] = "qasm_fallback_source_semantics_not_certified"
            base["materialization_route_reason"] = source_semantics_reason
        elif repr_row.get("graph_status") != "available" or not source_graph_file:
            base["materialization_route"] = "qasm_fallback_graph_unavailable"
            base["materialization_route_reason"] = str(
                repr_row.get("graph_terminal_reason", "verified_graph_record_missing")
            )
            base["materialization_semantics_gate"] = "qasm_parser_required_graph_representation_missing_or_unavailable"
        else:
            graph_path = COMPILED_ROOT / "graphs" / source_graph_file
            try:
                with graph_path.open("rb") as handle:
                    saved_graph = pickle.load(handle)
                operation_digest, saved_counts, operation_count = saved_s71_graph_signature(saved_graph)
                expected_digest = str(repr_row.get("graph_input_digest", ""))
                if not expected_digest or operation_digest != expected_digest:
                    raise RuntimeError(
                        f"verified_graph_operation_digest_mismatch:{canonical_id}:{operation_digest}:{expected_digest}"
                    )
                global_values, global_meta = extract_global_features_from_s71(saved_graph, repr_row, extractor)
                physical_t1_t2, calibration = snapshot_t1_t2(repr_row, registry)
                sparse_graph, graph_meta = compact_graph_record_from_s71(saved_graph, physical_t1_t2)
                graph_key = stable_hash(
                    {
                        "graph_input_digest": expected_digest,
                        "properties_sha256": property_hash,
                        "node_schema": "mali-pinned-node-178-v1",
                    }
                )
                relative_path, file_hash = save_sparse_graph(sparse_graph, graph_key)
                sidecar_counts = json.loads(repr_row["gate_counts_json"])
                base.update({f"global_{index:02d}": value for index, value in enumerate(global_values)})
                base.update(
                    {
                        "global_status": "available",
                        "global_reason": "",
                        "graph_status": "available",
                        "graph_reason": "",
                        "graph_key": graph_key,
                        "graph_artifact": relative_path,
                        "graph_artifact_sha256": file_hash,
                        "graph_node_count": graph_meta["node_count"],
                        "graph_edge_count": graph_meta["edge_count"],
                        "graph_active_width": graph_meta["active_width"],
                        "graph_meta_json": json.dumps(graph_meta, sort_keys=True, separators=(",", ":")),
                        "calibration_json": json.dumps(calibration, sort_keys=True, separators=(",", ":")),
                        "feature_signature_json": json.dumps(
                            {
                                "graph_input_digest": operation_digest,
                                "allocated_width": int(saved_graph["num_qubits"]),
                                "classical_width": int(saved_graph["num_clbits"]),
                                "operation_count": operation_count,
                                "measurement_count": int(saved_counts.get("measure", 0)),
                                "gate_counts": sidecar_counts,
                                "signature_source": global_meta["signature_source"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "wire_restoration_json": json.dumps(
                            {
                                "status": "not_replayed_verified_saved_ir",
                                "source_graph_file": source_graph_file,
                                "source_graph_file_sha256": source_graph_sha,
                                "note": "Saved compiled graph carries global hardware indices; QASM3 wire restoration is not replayed on this route.",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "materialization_route": "verified_graph_pickle",
                        "materialization_route_reason": "sha256_and_operation_digest_verified; compiled_unitary_source_contract",
                        "materialization_semantics_gate": source_semantics_gate,
                    }
                )
                cache[cache_key] = {
                    k: v
                    for k, v in base.items()
                    if k
                    not in {
                        "canonical_row_id",
                        "source_id",
                        "backend",
                        "source_identity_kind",
                        "source_stage",
                        "representation_id",
                        "qasm_input_sha256",
                        "qasm_source_ref_json",
                        "materializer_code_sha256",
                        "feature_library_code_sha256",
                        "qasm3_wire_adapter_code_sha256",
                    }
                }
                return base
            except RuntimeError:
                # A hash/digest violation is an integrity failure, never a reason
                # to silently accept a potentially unrelated fallback circuit.
                raise
            except (ValueError, KeyError, TypeError, pickle.UnpicklingError) as exc:
                # The representation is authenticated but does not carry enough
                # semantics for the pinned encoder; the original parser path is
                # retained as the exact fallback.
                base["materialization_route"] = "qasm_fallback_saved_graph_ineligible"
                base["materialization_route_reason"] = f"{type(exc).__name__}:{exc}"
                base["materialization_semantics_gate"] = (
                    "qasm_parser_required_saved_pickle_missing_pinned_encoder_semantics"
                )

    restoration = {}
    try:
        text = qasm_data.decode("utf-8")
        if source in {"mali_real_qpu", "qpack_mcp"} and text.lstrip().startswith("OPENQASM 3"):
            from benchmark26.qasm3_wires import restore_saved_qasm3_hardware_wires

            text, restoration = restore_saved_qasm3_hardware_wires(text, expected_width)
            circuit = qasm3.loads(text)
        else:
            circuit, text = parse_qasm(qasm_data)
        if int(circuit.num_qubits) != expected_width:
            raise RuntimeError(f"source_identity_width_mismatch:{circuit.num_qubits}:{expected_width}")
        signature = saved_graph_signature(circuit, repr_row)
        base["feature_signature_json"] = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        base["wire_restoration_json"] = json.dumps(restoration, sort_keys=True, separators=(",", ":"))
    except ValueError as exc:
        reason = str(exc)
        if reason.startswith(("source_identity_width_mismatch", "feature_signature_gate_failed")):
            raise RuntimeError(f"{canonical_id}:{reason}") from exc
        base["global_reason"] = reason
        base["graph_reason"] = reason
        cache[cache_key] = {
            k: v
            for k, v in base.items()
            if k
            not in {
                "canonical_row_id",
                "source_id",
                "backend",
                "source_identity_kind",
                "source_stage",
                "representation_id",
                "qasm_input_sha256",
                "qasm_source_ref_json",
            }
        }
        return base
    except Exception as exc:
        raise RuntimeError(f"pinned_parse_or_signature_failure:{canonical_id}:{type(exc).__name__}:{exc}") from exc

    has_control_flow = any(
        str(item.operation.name) == "if_else" or getattr(item.operation, "blocks", None) is not None
        for item in circuit.data
    )
    try:
        global_values = extract_global_features(circuit, extractor)
        base.update({f"global_{index:02d}": value for index, value in enumerate(global_values)})
        base["global_status"] = "available"
        base["global_reason"] = ""
    except ValueError as exc:
        reason = str(exc)
        if has_control_flow and "Global qubit index for local qubit" in reason:
            base["global_reason"] = "unavailable_control_flow_unsupported_by_pinned_global_extractor:" + reason
        else:
            base["global_reason"] = reason

    try:
        if source == "qonductor_single_circuit_ibm" and has_control_flow:
            raise ValueError(
                "unavailable_graph_control_flow:dynamic_control_not_representable_by_static_upstream_encoder"
            )
        physical_t1_t2, calibration = snapshot_t1_t2(repr_row, registry)
        sparse_graph, graph_meta = compact_graph_record(circuit, physical_t1_t2)
        graph_key = stable_hash(
            {
                "graph_input_digest": repr_row["graph_input_digest"],
                "properties_sha256": property_hash,
                "node_schema": "mali-pinned-node-178-v1",
            }
        )
        relative_path, file_hash = save_sparse_graph(sparse_graph, graph_key)
        base.update(
            {
                "graph_status": "available",
                "graph_reason": "",
                "graph_key": graph_key,
                "graph_artifact": relative_path,
                "graph_artifact_sha256": file_hash,
                "graph_node_count": graph_meta["node_count"],
                "graph_edge_count": graph_meta["edge_count"],
                "graph_active_width": graph_meta["active_width"],
                "graph_meta_json": json.dumps(graph_meta, sort_keys=True, separators=(",", ":")),
                "calibration_json": json.dumps(calibration, sort_keys=True, separators=(",", ":")),
                "wire_restoration_json": json.dumps(restoration, sort_keys=True, separators=(",", ":")),
            }
        )
    except ValueError as exc:
        base["graph_status"] = "unavailable"
        base["graph_reason"] = str(exc)
    cache[cache_key] = {
        k: v
        for k, v in base.items()
        if k
        not in {
            "canonical_row_id",
            "source_id",
            "backend",
            "source_identity_kind",
            "source_stage",
            "representation_id",
            "qasm_input_sha256",
            "qasm_source_ref_json",
        }
    }
    return base


def feature_fields() -> list[str]:
    """Return the column names of the feature status ledger, in order."""
    return [
        "canonical_row_id",
        "source_id",
        "backend",
        "source_identity_kind",
        "source_stage",
        "representation_id",
        "qasm_input_sha256",
        "qasm_source_ref_json",
        "materialization_route",
        "materialization_route_reason",
        "materialization_semantics_gate",
        "materializer_code_sha256",
        "feature_library_code_sha256",
        "qasm3_wire_adapter_code_sha256",
        "register_width",
        "snapshot_id",
        "snapshot_tier",
        "properties_sha256",
        "global_status",
        "global_reason",
        "graph_status",
        "graph_reason",
        "graph_key",
        "graph_artifact",
        "graph_artifact_sha256",
        "source_graph_file",
        "source_graph_file_sha256",
        "raw_global_width",
        "effective_global_width_before_fold_mask",
        *[f"global_{index:02d}" for index in range(51)],
        "feature_signature_json",
        "wire_restoration_json",
        "graph_node_count",
        "graph_edge_count",
        "graph_active_width",
        "graph_meta_json",
        "calibration_json",
    ]


def selected_canaries(rows: list[dict[str, str]], representations: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Choose the predeclared canary rows.

    For each (source, backend) cell, take the available graph with the fewest
    operations, or the first row if none is available. Add the two special
    Qonductor rows (row39 and row4477) when present.
    """
    chosen: dict[str, dict[str, str]] = {}
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        repr_row = representations[row["canonical_row_id"]]
        by_cell.setdefault((row["source_id"], repr_row["backend_canonical"]), []).append(row)
    for (source, backend), members in sorted(by_cell.items()):
        available = [r for r in members if representations[r["canonical_row_id"]]["graph_status"] == "available"]
        if available:
            selected = min(available, key=lambda r: int(representations[r["canonical_row_id"]]["operation_count"]))
        else:
            selected = members[0]
        chosen[selected["canonical_row_id"]] = selected
    special = {
        "qonductor_single_circuit_ibm|row39",
        "qonductor_single_circuit_ibm|row4477",
    }
    by_id = {row["canonical_row_id"]: row for row in rows}
    for canonical_id in special:
        if canonical_id in by_id:
            chosen[canonical_id] = by_id[canonical_id]
    return sorted(chosen.values(), key=lambda row: row["canonical_row_id"])


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    """Read an existing ledger CSV into a dict by canonical row ID; return {} if the file does not exist."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["canonical_row_id"]: row for row in csv.DictReader(handle)}


def write_ledger(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write the ledger CSV through a temporary file and an fsync, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    """Materialize the features of the canonical observations and write the status ledger.

    Checks the pinned inputs and records the environment, then processes each row
    with `make_row_attempt` and writes feature_status_ledger.csv under the output
    root, with run_manifest.json, input_manifest.json and feature_dictionary.json. `--canary` runs the predeclared canary set; `--resume` continues from an
    existing ledger after checking its IDs; `--max-rows` is a development-only
    subset whose manifest stays partial; `--input-route` chooses between the exact
    QASM parser and the hash-verified saved-graph fast path.
    """
    global OUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run the predeclared source/backend canary set, including the largest row and row4477.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing feature_status_ledger.csv after checking pinned inputs.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Development-only bounded subset; output manifest remains partial."
    )
    parser.add_argument(
        "--input-route",
        choices=("qasm_parse", "verified_graph_fast_path"),
        default="qasm_parse",
        help="Use the exact QASM parser (default) or hash-verified saved MaLi/QPack operation graphs with QASM fallback.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUT_ROOT,
        help="Artifact destination (default: the complete-run artifact directory).",
    )
    args = parser.parse_args()

    OUT_ROOT = args.output_root.resolve()
    _mali_features.OUT_ROOT = OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Check the pinned inputs and record the environment.
    pinned = input_manifest()
    code_sha256 = sha256_file(Path(__file__))
    implementation_pins = pinned["implementation_hashes"]
    if implementation_pins.get("materializer") != code_sha256:
        raise RuntimeError("materializer_implementation_hash_self_check_failed")
    pinned["feature_code_sha256"] = code_sha256
    pinned["input_route_mode"] = args.input_route
    env = environment_lock()
    canonical_rows, canonical_by_id, repr_by_id = load_input_rows()
    if args.canary:
        chosen = selected_canaries(canonical_rows, repr_by_id)
    else:
        chosen = canonical_rows
    if args.max_rows is not None:
        chosen = chosen[: args.max_rows]
    ledger_path = OUT_ROOT / "feature_status_ledger.csv"
    fields = feature_fields()
    existing = read_existing(ledger_path) if args.resume else {}
    unknown_existing = set(existing) - set(canonical_by_id)
    if unknown_existing:
        raise RuntimeError(f"resume_contains_unknown_ids:{sorted(unknown_existing)[:4]}")
    rows_by_id = {**existing}
    extractor, helper_pins = load_pinned_global_extractor()
    registry = load_snapshot_registry()
    graph_hashes = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    verified_graph_files: set[str] = set()
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    # Materialize each row that is not already in the ledger.
    with zipfile.ZipFile(QONDUCTOR_ZIP) as qond_zip:
        for number, row in enumerate(chosen, start=1):
            canonical_id = row["canonical_row_id"]
            if canonical_id in rows_by_id:
                continue
            attempt = make_row_attempt(
                row,
                repr_by_id[canonical_id],
                qond_zip,
                extractor,
                registry,
                cache,
                graph_hashes,
                verified_graph_files,
                args.input_route,
                code_sha256,
                implementation_pins,
            )
            rows_by_id[canonical_id] = attempt
            if number % 2 == 0 or number == len(chosen):
                ordered = sorted(rows_by_id.values(), key=lambda item: item["canonical_row_id"])
                write_ledger(ledger_path, ordered, fields)
                counts = Counter((item["global_status"], item["graph_status"]) for item in rows_by_id.values())
                atomic_json(
                    OUT_ROOT / "run_manifest.json",
                    {
                        "artifact_id": "mali-full-feature-materialization",
                        "status": "partial_canary" if args.canary or args.max_rows is not None else "materializing",
                        "rows_attempted": len(rows_by_id),
                        "assigned_rows": 8767,
                        "global_graph_status_counts": {"|".join(key): value for key, value in counts.items()},
                        "last_completed_canonical_row_id": canonical_id,
                        "elapsed_seconds": time.monotonic() - started,
                        "current_process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / (1024 * 1024),
                        "worker_count": 1,
                        "native_threads": 2,
                        "input_route_mode": args.input_route,
                        "materializer_code_sha256": code_sha256,
                        "implementation_hashes": implementation_pins,
                        "feature_implementation_sha256": pinned["feature_implementation_sha256"],
                        "input_manifest": pinned,
                        "environment": env,
                        "upstream_helper_blob_pins": helper_pins,
                        "fit_started": False,
                        "gpu_used": False,
                    },
                )
                print(f"C2 progress {len(rows_by_id)}/{len(chosen)} rows; last={canonical_id}", flush=True)
    rows = sorted(rows_by_id.values(), key=lambda item: item["canonical_row_id"])
    write_ledger(ledger_path, rows, fields)
    full = len(rows_by_id) == 8767
    atomic_json(
        OUT_ROOT / "input_manifest.json",
        {
            **pinned,
            "canonical_identity_count": len(canonical_rows),
            "feature_ledger_rows": len(rows),
            "full_panel_materialized": full,
            "row_order_sha256": stable_hash([row["canonical_row_id"] for row in canonical_rows]),
            "feature_status_ledger_sha256": sha256_file(ledger_path),
            "representation_rows_sha256": sha256_file(REPRESENTATION),
            "feature_code_sha256": code_sha256,
        },
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    atomic_json(
        OUT_ROOT / "feature_dictionary.json",
        {
            "method_id": "mali_full_features_graph_and_matched_mlp",
            "global_features": protocol["global_features"],
            "graph": protocol["graph"],
            "preprocessing": protocol["preprocessing"],
            "input_stage_by_source": protocol["dataset"]["source_stages"],
            "source_authority": protocol["source"],
            "materialization_routes": {
                "default": "qasm_parse",
                "verified_graph_fast_path": {
                    "scope": ["mali_real_qpu", "qpack_mcp"],
                    "verification": [
                        "source graph file SHA256",
                        "S71 ordered operation digest",
                        "QASM bytes SHA256",
                        "snapshot property SHA256",
                    ],
                    "fallback": "authenticated QASM parse for sources/rows whose graph or semantics are unavailable",
                    "qonductor_policy": "always QASM parse because saved graph does not encode cargs/conditions",
                    "classical_semantics_firewall": (
                        "S71 operation records do not encode classical arguments or instruction conditions. The verified fast "
                        "route is limited to source-authorized Ma-Li logical and QPack workflow-reconstruction inputs with "
                        "deterministic nominal transpilation; the operation digest authenticates the ordered quantum record, "
                        "not classical semantics. Unsupported gates, nonterminal measurements, control-flow operations, or any "
                        "source outside that static unitary-plus-terminal-measurement contract fall back to authenticated QASM."
                    ),
                },
            },
        },
    )
    atomic_json(OUT_ROOT / "method_cards.json", protocol["method_variants"])
    status_counts = Counter((item["global_status"], item["graph_status"]) for item in rows)
    final_status = (
        "features_materialized"
        if full
        else "partial_canary"
        if args.canary or args.max_rows is not None
        else "partial_resumeable"
    )
    atomic_json(
        OUT_ROOT / "run_manifest.json",
        {
            "artifact_id": "mali-full-feature-materialization",
            "status": final_status,
            "rows_attempted": len(rows),
            "assigned_rows": 8767,
            "global_graph_status_counts": {"|".join(key): value for key, value in status_counts.items()},
            "elapsed_seconds": time.monotonic() - started,
            "current_process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
            "worker_count": 1,
            "native_threads": 2,
            "input_route_mode": args.input_route,
            "materializer_code_sha256": code_sha256,
            "implementation_hashes": implementation_pins,
            "feature_implementation_sha256": pinned["feature_implementation_sha256"],
            "input_manifest_sha256": sha256_file(OUT_ROOT / "input_manifest.json"),
            "feature_dictionary_sha256": sha256_file(OUT_ROOT / "feature_dictionary.json"),
            "method_cards_sha256": sha256_file(OUT_ROOT / "method_cards.json"),
            "feature_status_ledger_sha256": sha256_file(ledger_path),
            "fit_started": False,
            "gpu_used": False,
        },
    )
    print(f"C2 status={final_status} rows={len(rows)} counts={dict(status_counts)}", flush=True)


if __name__ == "__main__":
    main()
