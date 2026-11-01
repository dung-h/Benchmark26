#!/usr/bin/env python3
"""Build the five pinned Qonductor features for every canonical QPU row."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import re
import signal
import sys
import struct
import tempfile
import threading
import time
from typing import Any
import zipfile

import numpy as np
from qiskit import qasm3
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark_v1" / "scripts"
RESOURCE_CANARY_ROOT = ROOT / "work" / "benchmark_recovery" / "qonductor_native_resource_canaries"
DEFAULT_C3_OUTPUT = ROOT / "artifacts" / "real_qpu" / "qonductor_native_features"
DEFAULT_C3_CHECKPOINT = ROOT / "work" / "benchmark_recovery" / "qonductor_native_features_checkpoint" / "parallel_feature_rows.jsonl"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qonductor_native_adapter import load_pinned_qonductor_methods
from qasm3_hardware_wire_adapter import restore_saved_qasm3_hardware_wires
from bounded_feature_pool import iter_feature_results
from qonductor_feature_tasks import bind_feature_metadata, extraction_key, initialize_feature_worker, extract_feature_task, plan_feature_tasks


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"pinned input is missing: {label} ({path})")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"pinned input hash mismatch for {label}: {observed}")
    return observed


def import_materializer(path: Path):
    spec = importlib.util.spec_from_file_location("qre_pinned_representation_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned representation materializer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_saved_compiled_qasm_path(representation_root: Path, relative_name: str) -> Path:
    """Resolve representation_rows.compiled_qasm3_file without prefix duplication."""
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts or not relative_name:
        raise FileNotFoundError("source_input_missing:invalid_saved_qasm_path")
    path = (representation_root / relative).resolve()
    try:
        path.relative_to(representation_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError("source_input_missing:path_escapes_saved_qasm_root") from exc
    return path


@contextmanager
def open_archive_once(path: Path):
    """Open the immutable Qonductor archive once and reuse its member reader."""
    with zipfile.ZipFile(path) as archive:
        yield archive.read


def extract_qonductor_archive_features(
    qasm_bytes: bytes,
    observed_qasm_sha: str,
    member: str,
    shots: int,
    feature_cache: dict[tuple[str, str, str], tuple[list[float], dict[str, Any]]],
    parse_qasm_string,
    extractor,
) -> tuple[list[float], dict[str, Any], float, float]:
    """Extract one exact submitted QASM row without retaining its circuit object.

    Only the small feature vector and metadata are cached by source/hash/shots.
    A different shot count reparses the same bytes; this deliberately bounds
    memory when source archives contain extremely large circuits.
    """
    cache_key = ("qonductor_single_circuit_ibm", observed_qasm_sha, str(shots))
    if cache_key in feature_cache:
        vector, metadata = feature_cache[cache_key]
        return list(vector), dict(metadata), 0.0, 0.0

    started = time.perf_counter()
    circuit = parse_qasm_string(qasm_bytes.decode("utf-8"))
    parse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    check_native_condition_support(circuit)
    vector = [float(value) for value in extractor._extract_features([circuit], shots).tolist()[0]]
    extract_seconds = time.perf_counter() - started
    if len(vector) != 5:
        raise RuntimeError("matrix_order_mismatch: upstream extractor did not emit five features")
    metadata = {
        "member": member,
        "top_level_operations": len(circuit.data),
        "top_level_control_flow": [
            inst.operation.name for inst in circuit.data
            if inst.operation.name in {"if_else", "while_loop", "for_loop", "switch_case"}
        ],
        "parser_route": "pinned upstream QASM2-then-QASM3 archive loader",
    }
    del circuit
    feature_cache[cache_key] = (vector, metadata)
    return vector, dict(metadata), parse_seconds, extract_seconds


def check_native_condition_support(circuit) -> None:
    """Preserve upstream tuple-condition arithmetic; classify Expr explicitly."""
    for inst in circuit.data:
        operation = inst.operation
        if getattr(operation, "_directive", False):
            continue
        condition = getattr(operation, "condition", None)
        if condition is not None and not isinstance(condition, tuple):
            raise ValueError("unsupported_condition_expression_not_supported_by_upstream")


def circuit_signature(circuit) -> dict[str, object]:
    operations = []
    for item in circuit.data:
        condition = getattr(item.operation, "condition", None)
        operations.append({
            "name": str(item.operation.name),
            "qargs": [circuit.find_bit(bit).index for bit in item.qubits],
            "cargs": [circuit.find_bit(bit).index for bit in item.clbits],
            "condition": None if condition is None else str(condition),
        })
    return {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "operations": operations,
    }


def graph_digest_only(circuit, graph_format: str, skip_names: set[str]) -> str:
    """Compute the representation digest without building graph arrays.

    This mirrors the exact byte stream used by the pinned representation
    materializer. It does not define a new digest; callers must prove byte-for-
    byte parity against graph_record on canaries and fixtures.
    """
    digest = hashlib.sha256()
    digest.update(graph_format.encode() + b"\0")
    digest.update(struct.pack(">QQ", int(circuit.num_qubits), int(circuit.num_clbits)))
    for inst in circuit.data:
        operation = inst.operation
        name = str(operation.name)
        if name in skip_names:
            continue
        qargs = tuple(int(circuit.find_bit(qubit).index) for qubit in inst.qubits)
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(qargs)))
        for qarg in qargs:
            digest.update(struct.pack(">q", qarg))
        digest.update(b"\x01" if bool(getattr(operation, "params", ())) else b"\x00")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def resolve_resource_canary_output(
    canonical_row_id: str,
    requested_output: Path | None,
    *,
    output_dir_was_supplied: bool,
    checkpoint_was_supplied: bool,
    native_threads: int,
) -> Path:
    """Fail closed: a resource probe can write only to its isolated work folder."""
    match = re.fullmatch(r"qonductor_single_circuit_ibm\|row([0-9]+)", canonical_row_id)
    if not match:
        raise ValueError("resource canary ID must be one exact Qonductor canonical row ID")
    if output_dir_was_supplied or checkpoint_was_supplied:
        raise ValueError("resource-canary mode forbids full-C3 --output-dir/--checkpoint-path")
    if native_threads not in (1, 2):
        raise ValueError("resource-canary mode is limited to 1 or 2 native threads")
    default_output = RESOURCE_CANARY_ROOT / f"row{match.group(1)}"
    output = (requested_output or default_output).resolve()
    root = RESOURCE_CANARY_ROOT.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("resource-canary output must remain under the isolated work canary root") from exc
    if output == root or output == DEFAULT_C3_OUTPUT.resolve():
        raise ValueError("resource-canary output cannot be the canary root or default C3 artifact path")
    if output.parent != root:
        raise ValueError("resource-canary output must be a single row directory directly under its work root")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refuse to overwrite nonempty resource-canary directory: {output}")
    return output


def _linux_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _linux_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _max_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; macOS reports bytes. The benchmark host is Linux.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def run_resource_canary(
    canonical_row_id: str,
    output_dir: Path,
    *,
    source_root: Path,
    archive_path: Path,
    native_threads: int,
    minimum_available_gib: float = 8.0,
    maximum_process_rss_gib: float = 16.0,
) -> int:
    """Time only one pinned Qonductor row; this is never a C3 completion receipt."""
    if minimum_available_gib < 8.0:
        raise ValueError("resource-canary available-memory reserve cannot be lower than 8 GiB")
    if maximum_process_rss_gib <= 0:
        raise ValueError("resource-canary process RSS cap must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "resource_canary.json"
    if receipt_path.exists():
        raise RuntimeError(f"refuse to overwrite resource-canary receipt: {receipt_path}")

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    contract_path = ROOT / "benchmark_v1" / "protocol" / "qonductor_native_features.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    canonical_path = ROOT / contract["input_pins"]["canonical"]["path"]
    canonical_hash = require_hash(canonical_path, contract["input_pins"]["canonical"]["sha256"], "canonical")
    archive_hash = require_hash(
        archive_path,
        contract["external_assets"]["qonductor_archive"]["sha256"],
        "Qonductor archive",
    )
    canonical_rows = read_csv(canonical_path)
    matches = [row for row in canonical_rows if row["canonical_row_id"] == canonical_row_id]
    if len(matches) != 1 or matches[0]["source_id"] != "qonductor_single_circuit_ibm":
        raise RuntimeError("requested resource-canary row is not a unique pinned Qonductor canonical observation")
    row = matches[0]
    if row["qasm_path_or_member"] == "" or row["qasm_bytes_sha256"] == "":
        raise RuntimeError("requested resource-canary row does not have an exact archived QASM identity")
    extractor, _archive_loader, parse_qasm_string, source_metadata = load_pinned_qonductor_methods(
        source_root, contract["source_pin"]
    )
    if source_metadata["matrix_order"] != contract["features"]["actual_matrix_order"]:
        raise RuntimeError("pinned feature order differs from active Qonductor contract")
    before_available = _linux_available_bytes()
    before_rss = _linux_rss_bytes()
    min_available = int(minimum_available_gib * (1024 ** 3))
    rss_cap = int(maximum_process_rss_gib * (1024 ** 3))
    if before_available is not None and before_available < min_available:
        raise RuntimeError("resource-canary preflight refused: available RAM is already below the 8 GiB reserve")

    receipt: dict[str, Any] = {
        "artifact_id": "qonductor-native-single-row-resource-canary",
        "status": "running_resource_canary_only",
        "stage": "resource_canary_only",
        "assigned_rows": 1,
        "canonical_row_id": canonical_row_id,
        "source_id": row["source_id"],
        "source_row_index": row["source_row_index"],
        "started_utc": started_utc,
        "finished_utc": None,
        "model_fits_started": False,
        "full_c3_started": False,
        "resource_policy": {
            "workers": 1,
            "native_threads": native_threads,
            "minimum_available_ram_gib": minimum_available_gib,
            "maximum_process_rss_gib": maximum_process_rss_gib,
            "stop_behavior": "SIGINT on cap; do not begin extractor if parse stage was interrupted",
        },
        "inputs": {
            "canonical_path": str(canonical_path.relative_to(ROOT)),
            "canonical_sha256": canonical_hash,
            "archive_path": str(archive_path),
            "archive_sha256": archive_hash,
            "archive_member": row["qasm_path_or_member"],
            "expected_qasm_sha256": row["qasm_bytes_sha256"],
            "protocol_path": str(contract_path.relative_to(ROOT)),
            "protocol_sha256": sha256_file(contract_path),
            "source_revision": source_metadata["source_revision"],
            "source_blobs": source_metadata["pinned_blobs"],
            "materializer_sha256": sha256_file(Path(__file__)),
            "pinned_adapter_sha256": sha256_file(SCRIPTS / "qonductor_native_adapter.py"),
        },
        "qasm": {"sha256": None, "bytes": None, "top_level_operation_count": None},
        "features": {"order": list(source_metadata["matrix_order"]), "vector": None},
        "timing_seconds": {"qasm_parse": None, "feature_extractor": None},
        "memory": {
            "available_before_bytes": before_available,
            "process_rss_before_bytes": before_rss,
            "minimum_available_observed_bytes": before_available,
            "maximum_process_rss_observed_bytes": before_rss,
            "ru_maxrss_bytes": None,
        },
        "terminal_reason": None,
    }
    atomic_write_json(receipt_path, receipt)

    stop_monitor = threading.Event()
    cap_state: dict[str, Any] = {"reason": None, "stage": None}
    stage_state: dict[str, str] = {"stage": "archive_read"}

    def monitor_resources() -> None:
        while not stop_monitor.wait(0.25):
            available = _linux_available_bytes()
            rss = _linux_rss_bytes()
            if available is not None:
                current_min = receipt["memory"]["minimum_available_observed_bytes"]
                receipt["memory"]["minimum_available_observed_bytes"] = available if current_min is None else min(current_min, available)
            if rss is not None:
                current_max = receipt["memory"]["maximum_process_rss_observed_bytes"]
                receipt["memory"]["maximum_process_rss_observed_bytes"] = rss if current_max is None else max(current_max, rss)
            reason = None
            if available is not None and available < min_available:
                reason = "minimum_available_ram_reserve_reached"
            elif rss is not None and rss > rss_cap:
                reason = "maximum_process_rss_cap_reached"
            if reason is not None:
                cap_state["reason"] = reason
                cap_state["stage"] = stage_state["stage"]
                os.kill(os.getpid(), signal.SIGINT)
                return

    monitor = threading.Thread(target=monitor_resources, name="resource-canary-memory-guard", daemon=True)
    monitor.start()
    try:
        with threadpool_limits(limits=native_threads):
            stage_state["stage"] = "archive_read_and_hash"
            with zipfile.ZipFile(archive_path) as archive:
                qasm_bytes = archive.read(row["qasm_path_or_member"])
            qasm_sha = sha256_bytes(qasm_bytes)
            if qasm_sha != row["qasm_bytes_sha256"]:
                raise RuntimeError("canonical QASM member hash mismatch")
            receipt["qasm"]["sha256"] = qasm_sha
            receipt["qasm"]["bytes"] = len(qasm_bytes)
            stage_state["stage"] = "qasm_parse"
            parse_start = time.perf_counter()
            circuit = parse_qasm_string(qasm_bytes.decode("utf-8"))
            receipt["timing_seconds"]["qasm_parse"] = time.perf_counter() - parse_start
            operation_count = len(circuit.data)
            receipt["qasm"]["top_level_operation_count"] = operation_count
            receipt["qasm"]["allocated_qubits"] = int(circuit.num_qubits)
            receipt["qasm"]["allocated_clbits"] = int(circuit.num_clbits)
            stage_state["stage"] = "pinned_feature_extractor"
            extractor_start = time.perf_counter()
            vector = [float(value) for value in extractor._extract_features([circuit], int(row["shots"])).tolist()[0]]
            receipt["timing_seconds"]["feature_extractor"] = time.perf_counter() - extractor_start
            if len(vector) != 5 or not all(np.isfinite(value) for value in vector):
                raise RuntimeError("pinned extractor returned malformed/nonfinite five-feature vector")
            if int(vector[3]) != int(row["shots"]) or int(vector[4]) != 1:
                raise RuntimeError("pinned extractor shots/circuit_count parity check failed")
            receipt["features"]["vector"] = vector
            receipt["status"] = "complete_resource_canary_only"
            receipt["terminal_reason"] = "single_row_resource_probe_complete_not_full_c3"
            del circuit
            del qasm_bytes
    except KeyboardInterrupt:
        receipt["status"] = "aborted_resource_cap"
        receipt["terminal_reason"] = cap_state["reason"] or "interrupt_received_during_resource_canary"
        receipt["aborted_stage"] = cap_state["stage"] or stage_state["stage"]
    except Exception as exc:
        receipt["status"] = "resource_canary_failed"
        receipt["terminal_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        stop_monitor.set()
        monitor.join(timeout=2.0)
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt["memory"]["ru_maxrss_bytes"] = _max_rss_bytes()
        final_rss = _linux_rss_bytes()
        if final_rss is not None:
            observed = receipt["memory"]["maximum_process_rss_observed_bytes"]
            receipt["memory"]["maximum_process_rss_observed_bytes"] = final_rss if observed is None else max(observed, final_rss)
        atomic_write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "stage": receipt["stage"], "assigned_rows": 1, "receipt": str(receipt_path), "terminal_reason": receipt["terminal_reason"]}, sort_keys=True))
    return 0 if receipt["status"] == "complete_resource_canary_only" else 3


def load_row_checkpoint(checkpoint_path: Path, manifest_path: Path, expected_manifest: dict[str, object]) -> list[dict[str, object]]:
    if not checkpoint_path.exists():
        if manifest_path.exists():
            observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed_manifest != expected_manifest:
                raise RuntimeError("row checkpoint input/source/protocol signature changed; use a new checkpoint path")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.touch()
            return []
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, expected_manifest)
        checkpoint_path.touch()
        return []
    if not manifest_path.exists():
        if checkpoint_path.stat().st_size != 0:
            raise RuntimeError("nonempty row checkpoint is missing its input/source/protocol manifest")
        atomic_write_json(manifest_path, expected_manifest)
        return []
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed_manifest != expected_manifest:
        raise RuntimeError("row checkpoint input/source/protocol signature changed; use a new checkpoint path")
    raw_checkpoint = checkpoint_path.read_bytes()
    if raw_checkpoint and not raw_checkpoint.endswith(b"\n"):
        last_complete = raw_checkpoint.rfind(b"\n") + 1
        with checkpoint_path.open("r+b") as handle:
            handle.truncate(last_complete)
            handle.flush()
            os.fsync(handle.fileno())
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with checkpoint_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed row checkpoint line {line_number}: {exc}") from exc
            if envelope.get("kind") != "row" or not isinstance(envelope.get("record"), dict):
                raise RuntimeError(f"unexpected row checkpoint envelope at line {line_number}")
            row_id = str(envelope["record"].get("canonical_row_id", ""))
            if not row_id or row_id in seen:
                raise RuntimeError(f"empty/duplicate row checkpoint identity at line {line_number}: {row_id}")
            seen.add(row_id)
            rows.append(envelope)
    return rows


def load_verified_legacy_checkpoint(
    rows_path: Path, receipt_path: Path, expected_identity: dict[str, object],
    implementation_hashes: dict[str, str],
) -> list[dict[str, object]]:
    """Import complete old records only through a hash-pinned transition receipt.

    The original checkpoint and its manifest remain immutable. The new
    checkpoint has its own implementation identity and records this receipt.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") == "qonductor-native-worker-scaling-resume":
        return load_verified_parallel_checkpoint(rows_path, receipt_path, expected_identity, implementation_hashes)
    if receipt.get("schema") != "qonductor-native-parallel-resume":
        raise RuntimeError("invalid legacy checkpoint import receipt")
    pins = [receipt["legacy_rows"], receipt["legacy_manifest"], receipt["source_canary"]]
    pins.extend(receipt["source_snapshot"].values())
    for pin in pins:
        require_hash(ROOT / pin["path"], pin["sha256"], "legacy checkpoint transition")
    if rows_path.resolve() != (ROOT / receipt["legacy_rows"]["path"]).resolve():
        raise RuntimeError("legacy checkpoint path does not match transition receipt")
    legacy_identity = json.loads((ROOT / receipt["legacy_manifest"]["path"]).read_text(encoding="utf-8"))
    if legacy_identity != expected_identity:
        raise RuntimeError("legacy checkpoint input/source/protocol signature changed")
    snapshots = receipt["source_snapshot"]
    for name in ("pinned_source_adapter", "hardware_wire_adapter"):
        if snapshots[name]["sha256"] != implementation_hashes[name]:
            raise RuntimeError(f"legacy checkpoint adapter changed: {name}")
    canary = json.loads((ROOT / receipt["source_canary"]["path"]).read_text(encoding="utf-8"))
    for name, old_key in (("native_feature_materializer", "materializer_sha256"), ("pinned_source_adapter", "pinned_adapter_sha256")):
        if canary["inputs"][old_key] != snapshots[name]["sha256"]:
            raise RuntimeError(f"legacy source snapshot does not match canary: {name}")
    raw = rows_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("legacy checkpoint has an incomplete tail; preserve and inspect it before import")
    envelopes = [json.loads(line) for line in raw.splitlines()]
    seen = set()
    for envelope in envelopes:
        if envelope.get("kind") != "row" or not isinstance(envelope.get("record"), dict):
            raise RuntimeError("invalid legacy checkpoint envelope")
        record = envelope["record"]
        row_id = record.get("canonical_row_id")
        if not row_id or row_id in seen:
            raise RuntimeError("duplicate legacy checkpoint identity")
        seen.add(row_id)
        old_key = envelope.get("cache_key")
        if old_key:
            shots = str(int(float(record["shots"])))
            submitted = record["source_id"] == "qonductor_single_circuit_ibm"
            old_tail = shots if submitted else str(int(envelope["cache_metadata"]["register_width"]))
            if list(old_key) != [record["source_id"], record["input_qasm_sha256"], old_tail]:
                raise RuntimeError("legacy checkpoint cache identity mismatch")
            envelope["cache_key"] = list(extraction_key(record["source_id"], record["input_qasm_sha256"], "" if submitted else old_tail, shots))
        envelope["implementation_origin"] = "verified_sequential_checkpoint_import"
        envelope["legacy_resume_receipt_sha256"] = sha256_file(receipt_path)
    if len(envelopes) != receipt["legacy_rows"]["completed_rows"]:
        raise RuntimeError("legacy checkpoint count does not match transition receipt")
    return envelopes


def load_verified_parallel_checkpoint(
    rows_path: Path, receipt_path: Path, expected_identity: dict[str, object],
    implementation_hashes: dict[str, str],
) -> list[dict[str, object]]:
    """Reuse pinned parallel rows while changing only scheduling code/policy.

    Source snapshots must match the old manifest. Extraction, AST restoration
    and graph identity code must still match the current implementation. The
    caller then applies its full canonical/split/per-row metadata validation.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "qonductor-native-worker-scaling-resume":
        raise RuntimeError("invalid parallel checkpoint import receipt")
    pins = [receipt["legacy_rows"], receipt["legacy_manifest"], receipt["prior_transition_receipt"]]
    pins.extend(receipt["source_snapshot"].values())
    for pin in pins:
        require_hash(ROOT / pin["path"], pin["sha256"], "parallel checkpoint transition")
    if rows_path.resolve() != (ROOT / receipt["legacy_rows"]["path"]).resolve():
        raise RuntimeError("parallel checkpoint path does not match transition receipt")
    manifest = json.loads((ROOT / receipt["legacy_manifest"]["path"]).read_text(encoding="utf-8"))
    if manifest.get("schema") != "qonductor-native-feature-row-checkpoint-v2":
        raise RuntimeError("parallel checkpoint has an unsupported identity schema")
    original_identity = dict(manifest)
    old_implementation = original_identity.pop("implementation_sha256")
    prior_receipt_hash = original_identity.pop("legacy_resume_receipt_sha256")
    old_resource_policy = original_identity.pop("resource_policy", None)
    original_identity["schema"] = "qonductor-native-feature-row-checkpoint-v1"
    if original_identity != expected_identity:
        raise RuntimeError("parallel checkpoint input/source/protocol signature changed")
    if old_resource_policy != receipt.get("legacy_resource_policy"):
        raise RuntimeError("parallel checkpoint resource-policy provenance changed")
    if prior_receipt_hash != receipt["prior_transition_receipt"]["sha256"]:
        raise RuntimeError("parallel checkpoint prior transition receipt changed")
    snapshots = receipt["source_snapshot"]
    if set(snapshots) != set(old_implementation):
        raise RuntimeError("parallel checkpoint source snapshot closure is incomplete")
    for name, old_hash in old_implementation.items():
        if snapshots[name]["sha256"] != old_hash:
            raise RuntimeError(f"parallel checkpoint source snapshot changed: {name}")
        if name not in ("native_feature_materializer", "bounded_feature_pool") and implementation_hashes.get(name) != old_hash:
            raise RuntimeError(f"parallel checkpoint extraction implementation changed: {name}")
    raw = rows_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("parallel checkpoint has an incomplete tail; preserve and inspect before import")
    envelopes = [json.loads(line) for line in raw.splitlines()]
    seen = set()
    for envelope in envelopes:
        if envelope.get("kind") != "row" or not isinstance(envelope.get("record"), dict):
            raise RuntimeError("invalid parallel checkpoint envelope")
        record = envelope["record"]
        row_id = record.get("canonical_row_id")
        if not row_id or row_id in seen:
            raise RuntimeError("duplicate parallel checkpoint identity")
        seen.add(row_id)
        key = envelope.get("cache_key")
        if key:
            submitted = record["source_id"] == "qonductor_single_circuit_ibm"
            width = "" if submitted else str(int(envelope["cache_metadata"]["register_width"]))
            expected_key = extraction_key(record["source_id"], record["input_qasm_sha256"], width, str(int(float(record["shots"]))))
            if list(key) != list(expected_key):
                raise RuntimeError("parallel checkpoint cache identity mismatch")
        # Retain the original sequential-import origin and its receipt pin.
        envelope["checkpoint_import_receipt_sha256"] = sha256_file(receipt_path)
        envelope["checkpoint_import_origin"] = "verified_parallel_checkpoint_import"
    if len(envelopes) != receipt["legacy_rows"]["completed_rows"]:
        raise RuntimeError("parallel checkpoint count does not match transition receipt")
    return envelopes


def validate_split_identity(canonical: list[dict[str, str]], outer: list[dict[str, str]], inner: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    canonical_by_id = {row["canonical_row_id"]: row for row in canonical}
    if len(canonical_by_id) != 8767:
        raise RuntimeError(f"canonical IDs are not unique 8,767-row identity: {len(canonical_by_id)}")
    outer_by_id = {row["canonical_observation_id"]: row for row in outer}
    if len(outer_by_id) != len(outer) or set(outer_by_id) != set(canonical_by_id):
        raise RuntimeError("outer split identity is not an exact canonical-row join")
    group_to_outer: dict[str, set[str]] = {}
    for row_id, split_row in outer_by_id.items():
        source_row = canonical_by_id[row_id]
        if split_row["source_id"] != source_row["source_id"] or split_row["source_row_index"] != source_row["source_row_index"]:
            raise RuntimeError(f"outer split source identity mismatch: {row_id}")
        group_to_outer.setdefault(split_row["unified_leakage_group_id"], set()).add(split_row["outer_fold"])
    leakage = {group: folds for group, folds in group_to_outer.items() if len(folds) != 1}
    if leakage:
        raise RuntimeError(f"frozen outer split leaks groups across folds: {len(leakage)}")

    inner_by_outer: dict[int, dict[str, dict[str, str]]] = {}
    for outer_fold in range(5):
        rows = [row for row in inner if row["outer_fold"] == str(outer_fold)]
        mapping = {row["canonical_observation_id"]: row for row in rows}
        if len(mapping) != 8767 or set(mapping) != set(canonical_by_id):
            raise RuntimeError(f"inner split identity mismatch for outer fold {outer_fold}")
        group_to_inner: dict[str, set[str]] = {}
        for row_id, row in mapping.items():
            outer_row = outer_by_id[row_id]
            is_outer_test = outer_row["outer_fold"] == str(outer_fold)
            assignment = row["inner_fold"]
            if is_outer_test and assignment != "":
                raise RuntimeError(f"outer-test row has an inner validation assignment: {row_id}/{outer_fold}")
            if not is_outer_test:
                if assignment not in {"0", "1", "2", "3"}:
                    raise RuntimeError(f"outer-train row has invalid inner assignment: {row_id}/{outer_fold}/{assignment}")
                group_to_inner.setdefault(outer_row["unified_leakage_group_id"], set()).add(assignment)
        if any(len(folds) != 1 for folds in group_to_inner.values()):
            raise RuntimeError(f"frozen inner split leaks groups for outer fold {outer_fold}")
        inner_by_outer[outer_fold] = mapping
    return (
        inner_by_outer,
        {key: value["unified_leakage_group_id"] for key, value in outer_by_id.items()},
        {key: value["outer_fold"] for key, value in outer_by_id.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "Qonductor-SC25")
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "Qonductor-SC25/data/database/circuits.zip")
    parser.add_argument("--superseded-preflight-dir", type=Path, default=None)
    parser.add_argument("--native-threads", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 13), help="maximum CPU extraction processes; memory admission controls actual concurrency")
    parser.add_argument("--aggregate-rss-cap-gib", type=float, default=None, help="explicit resource-only override, up to36GiB; recorded without rewriting frozen scientific cards")
    parser.add_argument("--import-checkpoint", type=Path, default=None, help="immutable sequential checkpoint to reuse through a verified transition receipt")
    parser.add_argument("--import-receipt", type=Path, default=None, help="hash-pinned evidence for the sequential checkpoint import")
    parser.add_argument("--resource-canary-id", type=str, default=None, help="run exactly one Qonductor row as a resource probe only")
    parser.add_argument("--resource-canary-output-dir", type=Path, default=None, help="isolated work/benchmark_recovery destination for the one-row probe")
    parser.add_argument("--canary-min-available-gib", type=float, default=8.0)
    parser.add_argument("--canary-max-process-rss-gib", type=float, default=16.0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.aggregate_rss_cap_gib is not None and (not np.isfinite(args.aggregate_rss_cap_gib) or not 0 < args.aggregate_rss_cap_gib <= 36):
        parser.error("aggregate RSS cap must be finite, positive and at most36GiB")
    if (args.import_checkpoint is None) != (args.import_receipt is None):
        parser.error("--import-checkpoint and --import-receipt must be supplied together")
    if args.resource_canary_id is not None:
        try:
            if args.aggregate_rss_cap_gib is not None:
                raise ValueError("resource-canary mode forbids full-pool resource overrides")
            if args.superseded_preflight_dir is not None:
                raise ValueError("resource-canary mode forbids superseded full-C3 preflight inputs")
            if args.import_checkpoint is not None:
                raise ValueError("resource-canary mode forbids checkpoint imports")
            output = resolve_resource_canary_output(
                args.resource_canary_id,
                args.resource_canary_output_dir,
                output_dir_was_supplied=args.output_dir is not None,
                checkpoint_was_supplied=args.checkpoint_path is not None,
                native_threads=args.native_threads,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return run_resource_canary(
            args.resource_canary_id,
            output,
            source_root=args.source_root,
            archive_path=args.archive,
            native_threads=args.native_threads,
            minimum_available_gib=args.canary_min_available_gib,
            maximum_process_rss_gib=args.canary_max_process_rss_gib,
        )

    args.output_dir = args.output_dir or DEFAULT_C3_OUTPUT
    args.checkpoint_path = args.checkpoint_path or DEFAULT_C3_CHECKPOINT
    superseded_preflight = None
    if args.output_dir.exists():
        prior_manifest_path = args.output_dir / "run_manifest.json"
        if not any(args.output_dir.iterdir()):
            pass
        elif not prior_manifest_path.is_file():
            raise SystemExit(f"refuse to overwrite existing non-run output directory: {args.output_dir}")
        else:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            if prior_manifest.get("artifact_id") != "qonductor-native-features-unified" or prior_manifest.get("status") != "fail" or prior_manifest.get("stage") != "feature_materialization_only":
                raise SystemExit(f"refuse to overwrite existing completed or unrelated output directory: {args.output_dir}")
            names = ["feature_rows.csv", "source_parity.json", "method_card.json", "run_manifest.json"]
            superseded_preflight = {
                "status": "superseded_internal_preflight_failure",
                "reason": "compiled_qasm3_file already contains compiled_qasm/ prefix; first pass joined it under that directory twice",
                "run_manifest_sha256": sha256_file(prior_manifest_path),
                "prior_output_hashes": {
                    name: sha256_file(args.output_dir / name)
                    for name in names if (args.output_dir / name).is_file()
                },
                "observed_qonductor_rows": 4482,
                "observed_row4477": [133.0, 197.0, 12.0, 8192.0, 1.0],
                "model_fits_started": False,
            }
    if args.superseded_preflight_dir is not None:
        preflight_dir = args.superseded_preflight_dir
        preflight_manifest = preflight_dir / "run_manifest.json"
        if not preflight_manifest.is_file():
            raise RuntimeError(f"superseded preflight receipt is missing: {preflight_manifest}")
        prior_manifest = json.loads(preflight_manifest.read_text(encoding="utf-8"))
        if prior_manifest.get("artifact_id") != "qonductor-native-features-unified" or prior_manifest.get("status") != "fail":
            raise RuntimeError("superseded preflight receipt is not the expected failed C3 attempt")
        names = ["feature_rows.csv", "source_parity.json", "method_card.json", "run_manifest.json"]
        superseded_preflight = {
            "status": "superseded_internal_preflight_failure",
            "reason": "saved representation path already includes compiled_qasm/; first pass added the directory twice",
            "receipt_artifact_class": "internal implementation correction; not a scientific circuit failure",
            "prior_output_hashes": {
                name: sha256_file(preflight_dir / name)
                for name in names if (preflight_dir / name).is_file()
            },
            "observed_qonductor_rows": 4482,
            "observed_row4477": [133.0, 197.0, 12.0, 8192.0, 1.0],
            "model_fits_started": False,
        }

    contract_path = ROOT / "benchmark_v1/protocol/qonductor_native_features.json"
    execution_path = ROOT / "benchmark_v1/protocol/estimator_execution.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    rss_cap_gib = args.aggregate_rss_cap_gib if args.aggregate_rss_cap_gib is not None else execution["resources"]["aggregate_rss_budget_gib"]
    authorization_path = ROOT / "artifacts/unified_runtime_estimation/execution_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "authorized_implementation_and_execution_subject_to_technical_gates":
        raise RuntimeError("explicit bounded execution authorization receipt is absent or invalid")
    if contract["method_id"] not in authorization.get("scope", ()) and not any(
        key == "qonductor_native_features" for key in authorization.get("method_contracts", {})
    ):
        raise RuntimeError("authorization does not cover the pinned Qonductor contract")

    # Verify all shared design and environment guards before reading circuits.
    pinned_paths = {
        "canonical": (ROOT / contract["input_pins"]["canonical"]["path"], contract["input_pins"]["canonical"]["sha256"]),
        "outer": (ROOT / contract["input_pins"]["outer"]["path"], contract["input_pins"]["outer"]["sha256"]),
        "inner": (ROOT / contract["input_pins"]["inner"]["path"], contract["input_pins"]["inner"]["sha256"]),
        "representation_rows": (ROOT / contract["input_pins"]["representation_rows"]["path"], contract["input_pins"]["representation_rows"]["sha256"]),
        "materializer_reference": (ROOT / contract["input_pins"]["materializer_reference"]["path"], contract["input_pins"]["materializer_reference"]["sha256"]),
        "seed_registry": (ROOT / execution["input_pins"]["seed_registry"]["path"], execution["input_pins"]["seed_registry"]["sha256"]),
        "CURRENT": (ROOT / execution["input_pins"]["current_guard"]["path"], execution["input_pins"]["current_guard"]["sha256"]),
        "fidelity_registry": (ROOT / execution["input_pins"]["fidelity_guard"]["path"], execution["input_pins"]["fidelity_guard"]["sha256"]),
    }
    input_hashes = {name: require_hash(path, expected, name) for name, (path, expected) in pinned_paths.items()}
    for name, pinned in execution["method_contracts"].items():
        path = ROOT / pinned["path"]
        observed = sha256_file(path)
        if observed != pinned["sha256"]:
            raise RuntimeError(f"shared method contract hash mismatch: {name}/{observed}")
        input_hashes[f"contract:{name}"] = observed
    if sha256_file(execution_path) != authorization["governing_contract"]["sha256"]:
        raise RuntimeError("execution contract differs from authorized receipt")

    archive_expected = contract["external_assets"]["qonductor_archive"]["sha256"]
    input_hashes["qonductor_archive"] = require_hash(args.archive, archive_expected, "Qonductor archive")
    extractor, archive_loader, parse_qasm_string, source_metadata = load_pinned_qonductor_methods(args.source_root, contract["source_pin"])
    materializer_path = ROOT / contract["input_pins"]["materializer_reference"]["path"]
    materializer = import_materializer(materializer_path)

    canonical = read_csv(ROOT / contract["input_pins"]["canonical"]["path"])
    outer = read_csv(ROOT / contract["input_pins"]["outer"]["path"])
    inner = read_csv(ROOT / contract["input_pins"]["inner"]["path"])
    representation = read_csv(ROOT / contract["input_pins"]["representation_rows"]["path"])
    inner_by_outer, group_by_id, outer_fold_by_id = validate_split_identity(canonical, outer, inner)
    expected_sources = contract["corpus"]["source_counts"]
    counts = Counter(row["source_id"] for row in canonical)
    if dict(counts) != expected_sources:
        raise RuntimeError(f"canonical source counts drifted: {dict(counts)}")
    canonical_ids = {row["canonical_row_id"] for row in canonical}
    representation_by_id = {row["canonical_row_id"]: row for row in representation}
    if len(representation_by_id) != len(representation) or set(representation_by_id) != canonical_ids:
        raise RuntimeError("representation rows are not a one-to-one canonical identity map")
    canonical_by_id = {row["canonical_row_id"]: row for row in canonical}

    representation_root = ROOT / contract["representation"]["compiled_qasm_root"]
    source_feature_order = list(contract["features"]["actual_matrix_order"])
    if source_feature_order != ["swap", "depth", "num_qubits", "shots", "circuit_count"]:
        raise RuntimeError(f"actual upstream feature order drifted: {source_feature_order}")
    if args.native_threads > contract["resources_and_outputs"]["native_thread_cap"]:
        raise RuntimeError("native thread request exceeds the frozen cap")
    if args.native_threads > execution["resources"]["native_threads_per_feature_worker"]:
        raise RuntimeError("native thread request exceeds the shared feature-worker cap")
    implementation_hashes = {
        "native_feature_materializer": sha256_file(Path(__file__)),
        "pinned_source_adapter": sha256_file(SCRIPTS / "qonductor_native_adapter.py"),
        "hardware_wire_adapter": sha256_file(SCRIPTS / "qasm3_hardware_wire_adapter.py"),
        "graph_materializer_reference": input_hashes["materializer_reference"],
        "feature_tasks": sha256_file(SCRIPTS / "qonductor_feature_tasks.py"),
        "bounded_feature_pool": sha256_file(SCRIPTS / "bounded_feature_pool.py"),
    }

    feature_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    route_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    unique_inputs: set[tuple[str, str]] = set()
    cache: dict[tuple[str, ...], tuple[list[float], dict[str, Any]]] = {}
    restored_signatures: Counter[str] = Counter()
    row4477_vector: list[float] | None = None
    saved_qasm_checks = 0
    archive_member_checks = 0
    source_prefix = "qonductor_single_circuit_ibm|row"

    checkpoint_path = args.checkpoint_path.resolve()
    output_path = args.output_dir.resolve()
    if checkpoint_path.is_relative_to(output_path):
        raise RuntimeError("row checkpoint must remain outside the semantic artifact directory")
    checkpoint_manifest_path = checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
    checkpoint_progress_path = checkpoint_path.with_name(checkpoint_path.name + ".progress.json")
    checkpoint_identity = {
        "schema": "qonductor-native-feature-row-checkpoint-v1",
        "protocol_sha256": input_hashes["contract:qonductor"],
        "canonical_sha256": input_hashes["canonical"],
        "outer_sha256": input_hashes["outer"],
        "inner_sha256": input_hashes["inner"],
        "representation_rows_sha256": input_hashes["representation_rows"],
        "materializer_reference_sha256": input_hashes["materializer_reference"],
        "qonductor_archive_sha256": input_hashes["qonductor_archive"],
        "source_revision": source_metadata["source_revision"],
        "source_blobs": source_metadata["pinned_blobs"],
        "feature_order": source_feature_order,
        "assigned_rows": len(canonical),
    }
    legacy_identity = dict(checkpoint_identity)
    imported_envelopes = []
    if args.import_checkpoint is not None:
        if args.import_checkpoint.resolve() == checkpoint_path:
            raise RuntimeError("checkpoint continuation must use a different path from immutable sequential evidence")
        imported_envelopes = load_verified_legacy_checkpoint(
            args.import_checkpoint, args.import_receipt, legacy_identity, implementation_hashes,
        )
    checkpoint_identity.update({
        "schema": "qonductor-native-feature-row-checkpoint-v2",
        "implementation_sha256": implementation_hashes,
        "legacy_resume_receipt_sha256": sha256_file(args.import_receipt) if args.import_receipt else None,
        "resource_policy": {
            "workers": args.workers, "native_threads": args.native_threads,
            "task_memory_reservation_gib": 0.0,
            "task_memory_reservation_rule": "saved: min(16,max(1,0.75+0.9*input_MiB)); submitted: min(16,max(0.5,0.25+0.25*input_MiB)); GiB heuristic, not an upper bound",
            "idle_worker_reservation_gib": 0.5,
            "aggregate_rss_cap_gib": rss_cap_gib,
            "backfill_smaller_inputs": True,
            "available_ram_reserve_gib": execution["resources"]["available_ram_reserve_gib"],
            "authority": "User resource-optimization request and measured scaling allowance in estimator_implementation_plan; frozen initial resource policy retained.",
        },
    })
    checkpoint_envelopes = load_row_checkpoint(checkpoint_path, checkpoint_manifest_path, checkpoint_identity)
    importing_now = not checkpoint_envelopes and bool(imported_envelopes)
    if importing_now:
        checkpoint_envelopes = imported_envelopes
    checkpoint_by_id = {
        str(envelope["record"]["canonical_row_id"]): envelope for envelope in checkpoint_envelopes
    }
    if not set(checkpoint_by_id).issubset(canonical_ids):
        raise RuntimeError("row checkpoint contains identities outside the frozen canonical panel")
    feature_rows.extend(envelope["record"] for envelope in checkpoint_envelopes)
    source_checkpoint_counts: Counter[str] = Counter(
        str(envelope["record"]["source_id"]) for envelope in checkpoint_envelopes
    )
    for envelope in checkpoint_envelopes:
        record = envelope["record"]
        row_id = str(record["canonical_row_id"])
        canonical_row = canonical_by_id[row_id]
        if record.get("source_id") != canonical_row["source_id"] or record.get("source_row_index") != canonical_row["source_row_index"]:
            raise RuntimeError(f"row checkpoint source identity changed: {row_id}")
        if record.get("unified_leakage_group_id") != group_by_id[row_id] or record.get("outer_fold") != outer_fold_by_id[row_id]:
            raise RuntimeError(f"row checkpoint frozen split identity changed: {row_id}")
        if record.get("representation_id") != representation_by_id[row_id]["representation_id"]:
            raise RuntimeError(f"row checkpoint representation identity changed: {row_id}")
        if str(record.get("availability_status")).startswith("available"):
            for name in source_feature_order:
                if not np.isfinite(float(record[name])):
                    raise RuntimeError(f"row checkpoint contains nonfinite feature: {row_id}/{name}")
            if [float(record[name]) for name in source_feature_order[3:]] != [float(canonical_row["shots"]), 1.0]:
                raise RuntimeError(f"row checkpoint shots/circuit count mismatch: {row_id}")
            rep = representation_by_id[row_id]
            submitted = canonical_row["source_id"] == "qonductor_single_circuit_ibm"
            expected_hash = canonical_row["qasm_bytes_sha256"] if submitted else rep["compiled_qasm3_sha256"]
            expected_key = extraction_key(canonical_row["source_id"], expected_hash, "" if submitted else rep["register_width"], canonical_row["shots"])
            if record["input_qasm_sha256"] != expected_hash or tuple(envelope["cache_key"]) != expected_key:
                raise RuntimeError(f"row checkpoint input/cache identity mismatch: {row_id}")
            metadata = envelope["cache_metadata"]
            if submitted:
                if metadata["member"] != canonical_row["qasm_path_or_member"]:
                    raise RuntimeError(f"row checkpoint archive member mismatch: {row_id}")
            elif any(metadata[name] != expected for name, expected in (
                ("compiled_qasm3_file", rep["compiled_qasm3_file"]),
                ("register_width", int(rep["register_width"])),
                ("graph_input_digest", rep["graph_input_digest"]),
                ("gate_counts_match", True), ("measurement_count_match", True),
            )):
                raise RuntimeError(f"row checkpoint saved-input metadata mismatch: {row_id}")
            if not submitted:
                if envelope.get("implementation_origin") == "verified_sequential_checkpoint_import":
                    # The pinned sequential source checked these exact counts;
                    # retain that evidence as metadata in the new continuation.
                    metadata.setdefault("verified_gate_counts", json.loads(rep["gate_counts_json"]))
                    metadata.setdefault("verified_measurement_count", int(rep["measurement_count"]))
                if metadata.get("verified_gate_counts") != json.loads(rep["gate_counts_json"]) or metadata.get("verified_measurement_count") != int(rep["measurement_count"]):
                    raise RuntimeError(f"row checkpoint saved counts mismatch: {row_id}")
        route = str(record.get("source_input_stage", ""))
        qasm_hash = str(record.get("input_qasm_sha256", ""))
        if route and qasm_hash:
            route_counts[route] += 1
            unique_inputs.add((route, qasm_hash))
            backend_counts[canonical_by_id[row_id]["backend"]] += 1
        if route == "exact_submitted_physical_qasm" and qasm_hash:
            archive_member_checks += 1
        if route == "saved_nominal_target_compiled_qasm3" and qasm_hash:
            saved_qasm_checks += 1
        restored_signatures[str(record.get("adapter_signature_sha256") or "no_ast_restore")] += 1
        if row_id == f"{source_prefix}4477" and str(record["availability_status"]).startswith("available"):
            row4477_vector = [float(record[name]) for name in source_feature_order]
        if record["availability_status"] == "unavailable":
            failures.append({"canonical_row_id": row_id, "source_id": str(record["source_id"]), "reason": str(record["terminal_reason"])})
        cache_key = envelope.get("cache_key")
        if cache_key and str(record["availability_status"]).startswith("available"):
            cache[tuple(str(value) for value in cache_key)] = (
                [float(record[name]) for name in source_feature_order],
                dict(envelope.get("cache_metadata") or {}),
            )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_stream = checkpoint_path.open("a", encoding="utf-8")
    if importing_now:
        for envelope in checkpoint_envelopes:
            checkpoint_stream.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
        checkpoint_stream.flush()
        os.fsync(checkpoint_stream.fileno())

    # Verify the one-open parser adapter against the pinned path loader on
    # QASM2, QASM3 (when present), and the opaque dynamic-control row.
    qpu_rows = [row for row in canonical if row["source_id"] == "qonductor_single_circuit_ibm"]
    row4477 = next(row for row in qpu_rows if row["canonical_row_id"] == f"{source_prefix}4477")
    parser_canaries: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as canary_archive:
        selected = [qpu_rows[0], row4477]
        seen_formats = set()
        for candidate in qpu_rows:
            raw_candidate = canary_archive.read(candidate["qasm_path_or_member"])
            fmt = "qasm3" if raw_candidate.lstrip().startswith(b"OPENQASM 3") else "qasm2"
            if fmt not in seen_formats:
                selected.append(candidate)
                seen_formats.add(fmt)
            if len(seen_formats) == 2:
                break
        for candidate in selected:
            member = candidate["qasm_path_or_member"]
            raw = canary_archive.read(member)
            fmt = "qasm3" if raw.lstrip().startswith(b"OPENQASM 3") else "qasm2"
            fast = parse_qasm_string(raw.decode("utf-8"))
            pinned = archive_loader(args.archive, member)
            if pinned is None:
                raise RuntimeError(f"pinned archive loader cannot read parser canary {member}")
            if circuit_signature(fast) != circuit_signature(pinned):
                raise RuntimeError(f"pinned QASM parser parity mismatch for {member}")
            shots = int(candidate["shots"])
            fast_vector = extractor._extract_features([fast], shots).tolist()[0]
            pinned_vector = extractor._extract_features([pinned], shots).tolist()[0]
            if fast_vector != pinned_vector:
                raise RuntimeError(f"pinned feature parser parity mismatch for {member}")
            parser_canaries.append({
                "canonical_row_id": candidate["canonical_row_id"],
                "member": member,
                "qasm_format": fmt,
                "source_member_sha256": sha256_bytes(raw),
                "feature_vector": fast_vector,
                "parser_signature_sha256": sha256_bytes(json_canonical(circuit_signature(fast)).encode("utf-8")),
                "matches_upstream_loader": True,
            })

    pending_rows = [row for row in canonical if row["canonical_row_id"] not in checkpoint_by_id]
    tasks = plan_feature_tasks(pending_rows, representation_by_id, representation_root, args.archive)
    for task in tasks:
        # Scheduling metadata only: workers ignore this field during extraction.
        # Four measured ~4.2-MiB saved inputs peaked at ~4.1 GiB per worker.
        # This estimates ~4.62 GiB at4.3 MiB, above observed ~4.12 GiB HWM.
        # It is admission guidance, never a guarantee or a scientific feature.
        task["memory_reservation_gib"] = min(16.0, max(1.0, 0.75 + 0.9 * task["input_bytes"] / 1024**2))
        if task.get("source_input_stage") == "exact_submitted_physical_qasm":
            # Native submitted inputs do not build/round-trip the saved AST.
            task["memory_reservation_gib"] = min(16.0, max(0.5, 0.25 + 0.25 * task["input_bytes"] / 1024**2))
    session_started = time.perf_counter()
    completed_tasks = 0
    worker_pids: set[int] = set()
    memory_observed = {"peak_process_tree_rss_gib": 0.0, "minimum_available_ram_gib": None}
    progress_last_written = 0.0
    last_completed_row_id = feature_rows[-1]["canonical_row_id"] if feature_rows else None

    def write_progress() -> None:
        nonlocal progress_last_written
        atomic_write_json(checkpoint_progress_path, {
            "status": "in_progress", "completed_rows": len(feature_rows), "assigned_rows": len(canonical),
            "last_completed_canonical_row_id": last_completed_row_id,
            "source_completed_rows": dict(source_checkpoint_counts),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "configured_workers": args.workers, "worker_pids_observed": sorted(worker_pids),
            "resource_policy": checkpoint_identity["resource_policy"],
            "completed_input_tasks_this_session": completed_tasks,
            "input_tasks_this_session": len(tasks),
            "elapsed_session_seconds": time.perf_counter() - session_started,
            **memory_observed,
        })
        progress_last_written = time.perf_counter()

    def observe_memory(available: float, rss: float) -> None:
        memory_observed["peak_process_tree_rss_gib"] = max(memory_observed["peak_process_tree_rss_gib"], rss)
        before = memory_observed["minimum_available_ram_gib"]
        memory_observed["minimum_available_ram_gib"] = available if before is None else min(before, available)
        if time.perf_counter() - progress_last_written >= 5.0:
            write_progress()

    def commit_task(task: dict[str, Any], result: dict[str, Any], *, reused: bool = False) -> None:
        nonlocal archive_member_checks, saved_qasm_checks, row4477_vector, completed_tasks, last_completed_row_id
        if result.get("execution_pid") is not None:
            worker_pids.add(int(result["execution_pid"]))
        for index, row_id in enumerate(task["row_ids"]):
            if row_id in checkpoint_by_id:
                raise RuntimeError(f"duplicate completion identity: {row_id}")
            row = canonical_by_id[row_id]
            source = row["source_id"]
            rep = representation_by_id[row_id]
            record = {
                "canonical_row_id": row_id, "source_id": source,
                "source_row_index": row["source_row_index"],
                "unified_leakage_group_id": group_by_id[row_id], "outer_fold": outer_fold_by_id[row_id],
                "source_input_stage": "", "input_qasm_sha256": "", "adapter_signature_sha256": "",
                "representation_id": rep["representation_id"],
                **{field: "" for field in source_feature_order},
                "availability_status": "unavailable", "terminal_reason": result.get("error", ""),
            }
            metadata = {}
            if not result.get("error"):
                vector = list(result["vector"])
                if len(vector) != 5 or not all(np.isfinite(float(value)) for value in vector):
                    raise RuntimeError("native_extractor_exception:malformed worker vector")
                if vector[3:] != [float(row["shots"]), 1.0]:
                    raise RuntimeError(f"matrix_order_mismatch:worker shots:{row_id}")
                metadata = bind_feature_metadata(result["metadata"], row, rep)
                if source == "qonductor_single_circuit_ibm":
                    archive_member_checks += 1
                else:
                    saved_qasm_checks += 1
                record.update(zip(source_feature_order, vector))
                record.update({
                    "source_input_stage": task["source_input_stage"],
                    "input_qasm_sha256": task["qasm_sha256"],
                    "adapter_signature_sha256": metadata.get("normalized_ast_sha256", ""),
                    "availability_status": "available_opaque_control_flow_structural_input"
                    if source == "qonductor_single_circuit_ibm" and metadata.get("top_level_control_flow") else "available",
                    "terminal_reason": "",
                })
                route_counts[record["source_input_stage"]] += 1
                backend_counts[row["backend"]] += 1
                unique_inputs.add((record["source_input_stage"], record["input_qasm_sha256"]))
                restored_signatures[str(record["adapter_signature_sha256"] or "no_ast_restore")] += 1
                if row_id == f"{source_prefix}4477":
                    row4477_vector = vector
            else:
                failures.append({"canonical_row_id": row_id, "source_id": source, "reason": str(result["error"])})
            source_checkpoint_counts[source] += 1
            feature_rows.append(record)
            envelope = {
                "kind": "row", "record": record,
                "cache_key": task.get("cache_key") if not result.get("error") else None,
                "cache_metadata": metadata if not result.get("error") else None,
                "stage_timings_seconds": result.get("stage_timings_seconds", {}) if index == 0 and not reused else {},
                "implementation_origin": "implementation_pinned_process_pool",
            }
            checkpoint_stream.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
            checkpoint_stream.flush()
            os.fsync(checkpoint_stream.fileno())
            checkpoint_by_id[row_id] = envelope
            checkpoint_envelopes.append(envelope)
        completed_tasks += 1
        last_completed_row_id = task["row_ids"][-1]
        write_progress()

    implementation_paths = {
        "native_feature_materializer": Path(__file__),
        "pinned_source_adapter": SCRIPTS / "qonductor_native_adapter.py",
        "hardware_wire_adapter": SCRIPTS / "qasm3_hardware_wire_adapter.py",
        "graph_materializer_reference": materializer_path,
        "feature_tasks": SCRIPTS / "qonductor_feature_tasks.py",
        "bounded_feature_pool": SCRIPTS / "bounded_feature_pool.py",
    }
    worker_config = {
        "source_root": str(args.source_root), "source_pin": contract["source_pin"],
        "archive": str(args.archive), "native_threads": args.native_threads,
        "materializer_reference": str(materializer_path),
        "implementation_pins": {
            name: {"path": str(path), "sha256": implementation_hashes[name]}
            for name, path in implementation_paths.items()
        },
    }
    unresolved_tasks = []
    try:
        write_progress()
        for task in tasks:
            key = tuple(task.get("cache_key", ()))
            if key in cache:
                vector, metadata = cache[key]
                commit_task(task, {"vector": vector, "metadata": metadata}, reused=True)
            elif "input_error" in task:
                commit_task(task, {"error": task["input_error"]})
            else:
                unresolved_tasks.append(task)
        for task, result in iter_feature_results(
            unresolved_tasks, extract_feature_task, workers=args.workers,
            initializer=initialize_feature_worker, initargs=(worker_config,),
            available_ram_reserve_gib=execution["resources"]["available_ram_reserve_gib"],
            aggregate_rss_cap_gib=rss_cap_gib,
            idle_worker_reservation_gib=0.5,
            backfill=True,
            resource_callback=observe_memory,
        ):
            commit_task(task, result)
    finally:
        checkpoint_stream.close()

    # Completion order is an execution detail; reader-facing rows retain the
    # canonical order and identity across both serial and parallel schedules.
    canonical_order = {row["canonical_row_id"]: index for index, row in enumerate(canonical)}
    feature_rows.sort(key=lambda row: canonical_order[row["canonical_row_id"]])

    if len(feature_rows) != 8767 or len({row["canonical_row_id"] for row in feature_rows}) != 8767:
        raise RuntimeError("feature materialization did not preserve exactly 8,767 canonical row identities")
    if row4477_vector != [133.0, 197.0, 12.0, 8192.0, 1.0]:
        raise RuntimeError(f"row4477 pinned top-level feature vector mismatch: {row4477_vector}")
    atomic_write_json(checkpoint_progress_path, {
        "status": "complete",
        "completed_rows": len(feature_rows),
        "assigned_rows": len(canonical),
        "last_completed_canonical_row_id": feature_rows[-1]["canonical_row_id"],
        "source_completed_rows": dict(source_checkpoint_counts),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "configured_workers": args.workers,
        "worker_pids_observed": sorted(worker_pids),
        "elapsed_session_seconds": time.perf_counter() - session_started,
        "completed_input_tasks_this_session": completed_tasks,
        **memory_observed,
    })
    source_status = {
        source: {
            "assigned": sum(1 for row in feature_rows if row["source_id"] == source),
            "available": sum(1 for row in feature_rows if row["source_id"] == source and str(row["availability_status"]).startswith("available")),
            "unavailable": sum(1 for row in feature_rows if row["source_id"] == source and row["availability_status"] == "unavailable"),
        }
        for source in expected_sources
    }

    timings: dict[str, list[float]] = {}
    for envelope in checkpoint_envelopes:
        for stage, value in dict(envelope.get("stage_timings_seconds") or {}).items():
            timings.setdefault(stage, []).append(float(value))
    timing_summary = {
        stage: {
            "distinct_input_count": len(values),
            "sum_seconds": float(sum(values)),
            "median_seconds": float(np.median(values)),
            "p90_seconds": float(np.quantile(values, 0.90)),
            "max_seconds": float(max(values)),
        }
        for stage, values in sorted(timings.items())
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "feature_rows.csv"
    fields = [
        "canonical_row_id", "source_id", "source_row_index", "unified_leakage_group_id", "outer_fold",
        "source_input_stage", "input_qasm_sha256", "adapter_signature_sha256", "representation_id",
        *source_feature_order, "availability_status", "terminal_reason",
    ]
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(feature_rows)

    source_parity = {
        "status": "pass" if not failures and all(item["available"] == item["assigned"] for item in source_status.values()) else "fail",
        "source_revision": source_metadata["source_revision"],
        "source_root_head": source_metadata["source_root_head"],
        "matrix_order": source_feature_order,
        "archive_member_hash_checks": archive_member_checks,
        "saved_compiled_qasm_hash_checks": saved_qasm_checks,
        "parser_canaries": parser_canaries,
        "unique_input_count": len(unique_inputs),
        "source_status": source_status,
        "route_counts": dict(route_counts),
        "backend_counts": dict(backend_counts),
        "implementation_sha256": implementation_hashes,
        "stage_timing_summary": timing_summary,
        "checkpoint": {
            "rows_path": str(checkpoint_path),
            "rows_sha256": sha256_file(checkpoint_path),
            "manifest_path": str(checkpoint_manifest_path),
            "manifest_sha256": sha256_file(checkpoint_manifest_path),
            "progress_path": str(checkpoint_progress_path),
            "progress_sha256": sha256_file(checkpoint_progress_path),
            "legacy_import_receipt_path": str(args.import_receipt) if args.import_receipt else None,
            "legacy_import_receipt_sha256": sha256_file(args.import_receipt) if args.import_receipt else None,
            "legacy_imported_rows": len(imported_envelopes),
        },
        "row4477": {
            "canonical_row_id": "qonductor_single_circuit_ibm|row4477",
            "status": next(row["availability_status"] for row in feature_rows if row["canonical_row_id"] == "qonductor_single_circuit_ibm|row4477"),
            "expected_vector": [133, 197, 12, 8192, 1],
            "observed_vector": row4477_vector,
            "branch_body_expansion": False,
        },
        "failures": failures,
        "feature_rows_sha256": sha256_file(feature_path),
    }
    parity_path = args.output_dir / "source_parity.json"
    parity_path.write_text(json.dumps(source_parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    method_card = {
        "method_id": contract["method_id"],
        "reader_label": contract["reader_label"],
        "fidelity_class": contract["fidelity_class"],
        "evaluation_target_clock": contract["evaluation_target_clock"],
        "method_output_clock": contract["method_output_clock"],
        "claim_boundary": contract["claim_boundary"],
        "source_pin": contract["source_pin"],
        "features": contract["features"],
        "representation_routes": contract["representation"]["source_routes"],
        "source_feature_extractor": source_metadata,
        "trained_model_family": "Qonductor polynomial component only; not upstream's six-family selector",
    }
    card_path = args.output_dir / "method_card.json"
    card_path.write_text(json.dumps(method_card, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "qiskit": importlib.metadata.version("qiskit"),
        "openqasm3": importlib.metadata.version("openqasm3"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
        "native_thread_limit": args.native_threads,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    manifest = {
        "artifact_id": "qonductor-native-features-unified",
        "status": source_parity["status"],
        "stage": "feature_materialization_only",
        "assigned_rows": len(feature_rows),
        "feature_order": source_feature_order,
        "target_seconds_accessed": False,
        "source_status": source_status,
        "source_parity_sha256": sha256_file(parity_path),
        "feature_rows_sha256": sha256_file(feature_path),
        "method_card_sha256": sha256_file(card_path),
        "input_sha256": input_hashes,
        "source_adapter": source_metadata,
        "implementation_sha256": implementation_hashes,
        "stage_timing_summary": timing_summary,
        "checkpoint": source_parity["checkpoint"],
        "environment": environment,
        "resource_policy": {
            **checkpoint_identity["resource_policy"],
            "workers": args.workers,
            "native_threads": args.native_threads,
            "aggregate_rss_cap_gib": rss_cap_gib,
            "available_ram_reserve_gib": execution["resources"]["available_ram_reserve_gib"],
            "observed_memory": memory_observed,
            "worker_pids_observed": sorted(worker_pids),
            "dispatch": "spawn; memory-reserved admission with smaller-input backfill; inputs >=8MiB exclusive; one checkpoint writer",
            "task_grouping": "source, verified QASM hash, allocated width for saved inputs, recorded shots",
            "task_order": "descending input byte size with memory-compatible backfill; no runtime labels used",
            "input_tasks_this_session": len(tasks),
            "elapsed_session_seconds": time.perf_counter() - session_started,
        },
        "command": [sys.executable, *sys.argv],
        "split_validation": {
            "outer_rows": 8767,
            "inner_rows": len(inner),
            "outer_group_leakage": 0,
            "inner_group_leakage": 0,
        },
        "superseded_preflight_attempt": superseded_preflight,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "source_status": source_status, "route_counts": dict(route_counts), "failures": failures[:20], "failures_count": len(failures), "row4477": source_parity["row4477"], "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
