#!/usr/bin/env python3
"""Run the frozen Ma–Li full-feature graph and matched-MLP OOF cells.

Training requires the validation records of the external study bundle. The caller selects the neural environment. This runner does not measure
prediction latency or alter the supplied data and split definitions.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark26.paths import ARTIFACTS, MALI_SOURCE, PROTOCOLS, recorded

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROTOCOLS / "mali_full_features.json"
EXECUTION = PROTOCOLS / "estimator_execution.json"
CANONICAL = ARTIFACTS / "benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ARTIFACTS / "benchmark_v2/real_qpu/unified_outer_split_v2.csv"
INNER = ARTIFACTS / "benchmark_v2/real_qpu/unified_inner_split_v2.csv"
FEATURE_ROOT = ARTIFACTS / "real_qpu/mali_full_features"
OUT_ROOT = FEATURE_ROOT
EXECUTION_RECORD_PATH = ARTIFACTS / "unified_runtime_estimation/execution_record.json"
C1_C3_RECEIPT_PATH = ARTIFACTS / "unified_runtime_estimation/c1_c3_gate_receipt.json"
PREFLIGHT_GPU_PATH = ARTIFACTS / "unified_runtime_estimation/preflight_gpu.json"
TECHNICAL_GATE_RECEIPT_PATH = ARTIFACTS / "unified_runtime_estimation/technical_gate_receipt.json"
INITIAL_CELLS_QA_PATH = ARTIFACTS / "unified_runtime_estimation/initial_cells_qa_receipt.json"
PINNED_MODEL_REVISION = "32c392a6ece276f1ff046d4e30052d0571ff6dc6"
GATE_ID = "mali-full-features-c4-v1"
REQUIRED_C4_CHECKS = {
    "source_and_input_pins",
    "51_global_feature_parity",
    "178_graph_encoding_parity",
    "fold_transform_train_only",
    "matched_fit_ids_and_split_leakage",
    "microbatch_gradient_parity",
    "cuda_largest_graph_disposition",
    "atomic_resume_equivalence",
    "heldout_labels_not_used_for_fit_or_transform",
}
FEATURE_IMPLEMENTATION_PATHS = {
    "materializer": ROOT / "benchmark26/mali_inputs.py",
    "feature_library": ROOT / "benchmark26/mali_features.py",
    "qasm3_hardware_wire_adapter": ROOT / "benchmark26/qasm3_wires.py",
}
METHODS = ("mali_full_features_graph", "mali_full_features_mlp")
SEEDS = (42, 1234, 31415)
NODE_MICROBATCH_LIMIT = 250_000
CHECKPOINT_INTERVAL_EPOCHS = 10

from benchmark26.mali_features import (
    SOURCE_PINS,
    expand_dense_x,
    sha256_bytes,
    sha256_file,
    stable_hash,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into a list of dicts, with the csv field-size limit raised."""
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_bytes(value: Any) -> bytes:
    """Return a value as indented, sorted JSON bytes with a trailing newline; NaN is rejected."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write bytes through a temporary `.partial` file and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    """Write a value as JSON atomically."""
    atomic_bytes(path, json_bytes(value))


def canonical_protocol_hashes() -> tuple[dict[str, Any], str, str]:
    """Return the Ma-Li protocol, its SHA-256 and the SHA-256 of the execution protocol, after checking that they agree."""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    expected = execution["method_contracts"]["mali"]
    protocol_hash = sha256_file(PROTOCOL)
    execution_hash = sha256_file(EXECUTION)
    if protocol_hash != expected["sha256"]:
        raise RuntimeError("mali_protocol_hash_mismatch")
    for name, pin in protocol["input_pins"].items():
        path = recorded(pin["path"])
        if not path.is_file() or sha256_file(path) != pin["sha256"]:
            raise RuntimeError(f"mali_input_pin_mismatch:{name}")
    return protocol, protocol_hash, execution_hash


def validate_technical_gate_receipt(receipt_path: Path, feature_root: Path = FEATURE_ROOT) -> dict[str, Any]:
    """Fail closed unless root's C1-C4 technical gate receipt matches active pins."""
    protocol, protocol_hash, execution_hash = canonical_protocol_hashes()
    path = receipt_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"c4_gate_receipt_missing:{path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_feature_root_binding(receipt, feature_root)
    if receipt.get("overall_status") != "PASS":
        raise RuntimeError("technical_gate_receipt_not_pass")
    gates = receipt.get("gates")
    if not isinstance(gates, dict) or any(gates.get(f"C{index}") != "PASS" for index in range(1, 5)):
        raise RuntimeError("technical_gate_receipt_missing_C1_C4_pass")
    expected_pins = {
        "execution_record_sha256": sha256_file(EXECUTION_RECORD_PATH),
        "execution_protocol_sha256": execution_hash,
        "input_manifest_sha256": sha256_file(feature_root / "input_manifest.json"),
        "feature_status_ledger_sha256": sha256_file(feature_root / "feature_status_ledger.csv"),
        "c6_runner_sha256": sha256_file(Path(__file__)),
    }
    contracts = receipt.get("contract_hashes")
    if not isinstance(contracts, dict) or contracts.get("mali_full_features") != protocol_hash:
        raise RuntimeError("technical_gate_receipt_contract_hash_mismatch:mali_full_features")
    for key, expected in expected_pins.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"c4_gate_receipt_pin_mismatch:{key}")
    checks = receipt.get("c4_checks", receipt.get("checks"))
    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in REQUIRED_C4_CHECKS):
        raise RuntimeError("technical_gate_receipt_missing_required_C4_boolean_checks")
    methods = set(receipt.get("eligible_methods", []))
    if methods != set(METHODS):
        raise RuntimeError("technical_gate_receipt_invalid_method_allowlist")
    if not isinstance(receipt.get("activation_recomputation_enabled"), bool):
        raise RuntimeError("technical_gate_receipt_missing_activation_recomputation_selection")
    return receipt


def feature_root_relative(feature_root: Path) -> str:
    """Return the feature root relative to the repository; raise if it lies outside it."""
    resolved = feature_root.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RuntimeError("feature_root_must_be_inside_repository")
    return str(resolved.relative_to(ROOT.resolve()))


def validate_feature_root_binding(receipt: dict[str, Any], feature_root: Path, required: bool = False) -> None:
    """Bind signed stage receipts to the selected artifact root.

    Historical default-root receipts remain readable if they predate this field;
    any alternate root requires an explicit root-relative path in each receipt.
    """
    expected = feature_root_relative(feature_root)
    observed = receipt.get("feature_root")
    if observed is None:
        if required or feature_root.resolve() != FEATURE_ROOT.resolve():
            raise RuntimeError("receipt_feature_root_pin_missing")
        return
    if observed != expected:
        raise RuntimeError("receipt_feature_root_pin_mismatch")


def validate_neural_environment(input_manifest: dict[str, Any], feature_root: Path) -> dict[str, Any]:
    """Check that the current Python and installed packages are the ones pinned at feature build.

    Uses the environment of the input manifest or, for older feature builds that omitted
    it, the separately pinned C1-C3 receipt. Compares the interpreter path, the dependency
    lock file and the SHA-256 of the current package list. Returns the interpreter and lock
    identity.
    """
    import importlib.metadata as metadata

    environment = input_manifest.get("environment", {})
    if not environment:
        # Completed historical feature builders retained lock files but omitted
        # environment in their final manifest. Use the main's separately
        # pinned receipt; never rewrite the completed feature-build output.
        if not C1_C3_RECEIPT_PATH.is_file():
            raise RuntimeError("materialization_environment_missing_and_c1_c3_receipt_missing")
        receipt = json.loads(C1_C3_RECEIPT_PATH.read_text(encoding="utf-8"))
        validate_feature_root_binding(receipt, feature_root, required=True)
        pins = receipt.get("pins", receipt)
        expected = {
            "input_manifest_sha256": sha256_file(feature_root / "input_manifest.json"),
            "feature_status_ledger_sha256": sha256_file(feature_root / "feature_status_ledger.csv"),
        }
        if (
            receipt.get("overall_status") != "PASS"
            or not isinstance(pins, dict)
            or any(pins.get(name) != value for name, value in expected.items())
            or receipt.get("execution_record_sha256") != sha256_file(EXECUTION_RECORD_PATH)
        ):
            raise RuntimeError("c1_c3_environment_receipt_pin_mismatch")
        environment = receipt.get("neural_environment", {})
    expected_python = environment.get("neural_python")
    current_python = str(Path(sys.executable).resolve())
    if not expected_python or str(Path(expected_python).resolve()) != current_python:
        raise RuntimeError("neural_python_does_not_match_materialization_environment_lock")
    lock_name = environment.get("neural_lock_file")
    lock_path = feature_root / str(lock_name or "")
    if not lock_name or not lock_path.is_file() or sha256_file(lock_path) != environment.get("neural_lock_sha256"):
        raise RuntimeError("neural_dependency_lock_file_missing_or_hash_mismatch")
    actual_freeze = "".join(
        f"{name}=={version}\n"
        for name, version in sorted(
            (dist.metadata.get("Name", "").lower().replace("_", "-"), dist.version)
            for dist in metadata.distributions()
            if dist.metadata.get("Name")
        )
    )
    if sha256_bytes(actual_freeze.encode("utf-8")) != environment.get("neural_lock_sha256"):
        raise RuntimeError("current_neural_packages_differ_from_materialization_lock")
    return {"python": current_python, "lock_file": lock_name, "lock_sha256": environment["neural_lock_sha256"]}


def load_pinned_model():
    """Execute only the hash-pinned upstream model source in an inert namespace."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv, global_mean_pool
    from torch_geometric.nn.norm import LayerNorm

    relative = "model/transformer_model.py"
    expected = SOURCE_PINS[relative]
    blob = subprocess.check_output(["git", "-C", str(MALI_SOURCE), "show", f"{PINNED_MODEL_REVISION}:{relative}"])
    if sha256_bytes(blob) != expected:
        raise RuntimeError("pinned_mali_model_blob_hash_mismatch")
    scope: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "TransformerConv": TransformerConv,
        "global_mean_pool": global_mean_pool,
        "LayerNorm": LayerNorm,
    }
    exec(compile(blob, f"<pinned-mali:{relative}>", "exec"), scope)
    return scope["Simple_Model"], expected


def make_model(model_class, variant: str, global_width: int, checkpoint_layers: bool = False):
    """Build the pinned Ma-Li model for a variant.

    The graph variant uses node features, three graph layers and the global features; the
    MLP variant uses the global features only. With `checkpoint_layers`, the graph layers use
    activation checkpointing, which saves memory by recomputing them in the backward pass.
    """
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from torch_geometric.nn import global_mean_pool

    graph = variant == "mali_full_features_graph"
    args = SimpleNamespace(
        use_graph_features=graph,
        use_global_features=True,
        use_gate_type=True,
        use_qubit_index=True,
        use_T1T2=True,
        use_gate_error=True,
        use_gate_index=True,
        num_layers=3,
    )
    model = model_class(args, length_of_x=178, length_of_gf=global_width)
    if checkpoint_layers and graph:

        def checkpointed_forward(self, data):
            """Forward pass of the graph model with each graph layer under activation checkpointing."""
            x, edge_index, gf = data.x, data.edge_index, data.global_features
            x = x.to(torch.float32)
            gf = gf.to(torch.float32)
            x = x[:, self.mask]
            for layer_index in range(self.args.num_layers):
                layer = getattr(self, f"conv{layer_index}")
                x = checkpoint(
                    lambda current, module=layer, edges=edge_index: F.relu(module(current, edges)),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            x = global_mean_pool(x, data.batch)
            gf = F.relu(self.gf_linear1(gf))
            gf = F.relu(self.gf_linear2(gf))
            x = torch.cat([x, gf], dim=1)
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.relu(self.linear3(x))
            return self.linear4(x).squeeze()

        import types

        model.forward = types.MethodType(checkpointed_forward, model)
    return model


def fit_transform_global(raw_by_id: dict[str, list[float]], fit_ids: list[str]) -> dict[str, Any]:
    """Fit the standardization of the 51 global features on the fit rows."""
    values = __import__("numpy").asarray([raw_by_id[x] for x in fit_ids], dtype=__import__("numpy").float64)
    if values.shape != (len(fit_ids), 51):
        raise RuntimeError("global_fit_matrix_shape_mismatch")
    retained = __import__("numpy").flatnonzero(values.sum(axis=0) > 0)
    if not len(retained):
        raise RuntimeError("global_fit_transform_retained_zero_fields")
    selected = values[:, retained]
    mean = selected.mean(axis=0, dtype=__import__("numpy").float64)
    std = selected.std(axis=0, ddof=1, dtype=__import__("numpy").float64) if len(fit_ids) > 1 else selected[0] * 0
    variable = std > 1e-6
    return {
        "retained_indices": retained.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "variable_mask": variable.tolist(),
        "fit_ids_sha256": stable_hash(fit_ids),
    }


def transform_global(raw: list[float], transform: dict[str, Any]):
    """Apply a fitted global transform: select the retained features and standardize them."""
    import numpy as np

    selected = np.asarray(raw, dtype=np.float32)[transform["retained_indices"]]
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    variable = np.asarray(transform["variable_mask"], dtype=bool)
    result = np.zeros_like(selected)
    result[variable] = (selected[variable] - mean[variable]) / std[variable]
    return result


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Return the random-number generator states of Python, NumPy and PyTorch."""
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the random-number generator states captured by `capture_rng_state`."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint through a temporary `.partial` file and rename it into place."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_gpu_budget(torch, budget_gib: int = 12, reserve_gib: int = 4) -> dict[str, Any]:
    """Check that CUDA is available and that the card leaves the contract reserve, then cap this process.

    The declared budget is 12 GiB and the reserve 4 GiB. The cap accounts for memory used by
    other processes. Returns the device facts and the budgets.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_mali_full_feature_training")
    properties = torch.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    budget_bytes = budget_gib * 1024**3
    if total_bytes <= budget_bytes:
        raise RuntimeError("gpu_total_memory_does_not_leave_contract_reserve")
    free_bytes, reported_total = torch.cuda.mem_get_info(0)
    if reported_total != total_bytes or free_bytes < reserve_gib * 1024**3:
        raise RuntimeError("gpu_free_memory_below_contract_reserve")
    # Account for allocations outside this process before fixing the allocator
    # cap. A nominal 12 GiB cap alone can leave <4 GiB on a busy 16 GiB card.
    effective_budget_bytes = min(
        budget_bytes, int(free_bytes) - reserve_gib * 1024**3 + int(torch.cuda.memory_allocated(0))
    )
    if effective_budget_bytes <= 0:
        raise RuntimeError("gpu_reserve_leaves_no_training_allocation_budget")
    torch.cuda.set_per_process_memory_fraction(effective_budget_bytes / total_bytes, device=0)
    return {
        "name": properties.name,
        "total_bytes": total_bytes,
        "free_before_bytes": int(free_bytes),
        "declared_per_process_budget_bytes": budget_bytes,
        "per_process_budget_bytes": effective_budget_bytes,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def checkpoint_identity(
    method: str,
    fold: int,
    seed: int,
    protocol_hash: str,
    execution_hash: str,
    input_manifest_hash: str,
    ledger_hash: str,
    gpu_run_id: str,
    activation_recomputation_enabled: bool,
    feature_implementation_sha256: str,
) -> dict[str, Any]:
    """Return the identity record stored in a checkpoint.

    It holds the method, fold and seed, the protocol, execution, input-manifest and ledger
    hashes, the feature implementation hash (64 hex characters), the GPU run ID and
    whether activation recomputation is on.
    """
    if len(feature_implementation_sha256) != 64:
        raise RuntimeError("feature_implementation_hash_missing_or_invalid")
    return {
        "method": method,
        "outer_fold": int(fold),
        "seed": int(seed),
        "mali_protocol_sha256": protocol_hash,
        "execution_protocol_sha256": execution_hash,
        "input_manifest_sha256": input_manifest_hash,
        "feature_status_ledger_sha256": ledger_hash,
        "feature_implementation_sha256": feature_implementation_sha256,
        "gpu_run_id": gpu_run_id,
        "activation_recomputation_enabled": bool(activation_recomputation_enabled),
    }


def validate_checkpoint_identity(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    """Raise if the identity stored in a checkpoint differs from the expected one."""
    actual = checkpoint.get("identity")
    if actual != expected:
        raise RuntimeError("checkpoint_identity_mismatch")


def validate_feature_implementation_pins(input_manifest: dict[str, Any]) -> str:
    """Bind training/checkpoints to the exact feature builder and feature helpers."""
    pins = input_manifest.get("implementation_hashes")
    if not isinstance(pins, dict) or set(pins) != set(FEATURE_IMPLEMENTATION_PATHS):
        raise RuntimeError("feature_implementation_hash_set_missing_or_unexpected")
    observed = {}
    for name, path in FEATURE_IMPLEMENTATION_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"feature_implementation_source_missing:{name}")
        observed[name] = sha256_file(path)
    if pins != observed:
        raise RuntimeError("feature_implementation_hash_pin_mismatch")
    combined_hash = stable_hash(observed)
    if input_manifest.get("feature_implementation_sha256") != combined_hash:
        raise RuntimeError("feature_implementation_combined_hash_mismatch")
    return combined_hash


def _node_chunk(record: dict[str, Any], start: int, stop: int):
    """Expand only a bounded slice of the sparse node representation."""
    import numpy as np

    node_type = record["node_type"][start:stop]
    rows = len(node_type)
    x = np.zeros((rows, 178), dtype=np.float32)
    x[np.arange(rows), node_type] = 1.0
    wire0, wire1 = record["wire0"][start:stop], record["wire1"][start:stop]
    selected = np.flatnonzero(wire0 != 255)
    x[selected, 46 + wire0[selected]] = 1.0
    selected = np.flatnonzero(wire1 != 255)
    x[selected, 46 + wire1[selected]] = 1.0
    x[:, 173] = record["t1_0"][start:stop]
    x[:, 174] = record["t2_0"][start:stop]
    x[:, 175] = record["t1_1"][start:stop]
    x[:, 176] = record["t2_1"][start:stop]
    x[:, 177] = record["node_index"][start:stop]
    return x


def fit_node_transform(graph_paths: list[Path], fit_ids: list[str]) -> dict[str, Any]:
    """Stable node-weighted float64 moments without a corpus-wide dense matrix."""
    import numpy as np

    count = 0
    mean = np.zeros(178, dtype=np.float64)
    m2 = np.zeros(178, dtype=np.float64)
    for path in graph_paths:
        with np.load(path, allow_pickle=False) as archive:
            record = {name: archive[name] for name in archive.files}
            total = len(record["node_type"])
            for start in range(0, total, 65_536):
                chunk = _node_chunk(record, start, min(total, start + 65_536)).astype(np.float64)
                chunk_count = chunk.shape[0]
                chunk_mean = chunk.mean(axis=0)
                centered = chunk - chunk_mean
                chunk_m2 = np.einsum("ij,ij->j", centered, centered, optimize=True)
                delta = chunk_mean - mean
                combined = count + chunk_count
                mean += delta * (chunk_count / combined)
                m2 += chunk_m2 + delta * delta * (count * chunk_count / combined)
                count = combined
    if count < 2:
        raise RuntimeError("too_few_fit_graph_nodes_for_sample_std")
    std = np.sqrt(m2 / (count - 1))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "variable_mask": (std > 1e-6).tolist(),
        "fit_node_count": int(count),
        "fit_ids_sha256": stable_hash(fit_ids),
        "graph_paths_sha256": stable_hash([sha256_file(path) for path in graph_paths]),
    }


def load_saved_checkpoint(path: Path, expected_identity: dict[str, Any], torch) -> dict[str, Any]:
    """Load a checkpoint on the CPU and check its identity."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Torch versions predating weights_only.
        payload = torch.load(path, map_location="cpu")
    validate_checkpoint_identity(payload, expected_identity)
    return payload


def _load_graph_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _make_example(
    canonical_id: str,
    variant: str,
    ledger: dict[str, dict[str, str]],
    targets: dict[str, float] | None,
    transform: dict[str, Any],
    feature_root: Path,
    device,
    include_target: bool,
    verified_graph_paths: set[Path] | None = None,
):
    import numpy as np
    import torch
    from torch_geometric.data import Data

    row = ledger[canonical_id]
    raw = [float(row[f"global_{index:02d}"]) for index in range(51)]
    global_features = torch.as_tensor(transform_global(raw, transform["global"]), dtype=torch.float32).reshape(1, -1)
    if variant == "mali_full_features_mlp":
        data = Data(global_features=global_features)
    else:
        if row.get("graph_status") != "available":
            raise RuntimeError(f"graph_test_input_unavailable:{canonical_id}")
        artifact = row.get("graph_artifact", "")
        # Ledgers record bundle paths as ``artifacts/...``; resolve them in the
        # configured bundle rather than under the code checkout.
        in_bundle = artifact.startswith("artifacts/")
        graph_path = (recorded(artifact) if in_bundle else feature_root / artifact).resolve()
        allowed_root = (ARTIFACTS if in_bundle else feature_root).resolve()
        if allowed_root not in graph_path.parents or not graph_path.is_file():
            raise RuntimeError(f"graph_artifact_missing_or_outside_root:{canonical_id}")
        if (verified_graph_paths is None or graph_path not in verified_graph_paths) and sha256_file(
            graph_path
        ) != row.get("graph_artifact_sha256"):
            raise RuntimeError(f"graph_artifact_hash_mismatch:{canonical_id}")
        if verified_graph_paths is not None:
            verified_graph_paths.add(graph_path)
        record = _load_graph_npz(graph_path)
        x = expand_dense_x(record)
        mean = np.asarray(transform["node"]["mean"], dtype=np.float32)
        std = np.asarray(transform["node"]["std"], dtype=np.float32)
        variable = np.asarray(transform["node"]["variable_mask"], dtype=bool)
        x[:, variable] = (x[:, variable] - mean[variable]) / std[variable]
        x[:, ~variable] = 0.0
        edge_index = torch.as_tensor(record["edge_index"].astype(np.int64, copy=False), dtype=torch.long)
        data = Data(x=torch.as_tensor(x, dtype=torch.float32), edge_index=edge_index, global_features=global_features)
    if include_target:
        if targets is None or canonical_id not in targets:
            raise RuntimeError(f"training_target_missing:{canonical_id}")
        data.y = torch.tensor([float(targets[canonical_id])], dtype=torch.float32)
    return data


def _batch_to_device(examples: list[Any], device, graph: bool):
    from torch_geometric.data import Batch

    if graph:
        return Batch.from_data_list(examples).to(device)

    # The upstream global-only branch only consumes global_features.
    class GlobalBatch:
        pass

    batch = GlobalBatch()
    import torch

    batch.global_features = torch.cat([example.global_features for example in examples], dim=0).to(device)
    # The pinned upstream forward reads/casts these attributes before its
    # graph flag. Empty tensors satisfy that API without introducing graph
    # observations or altering the global-only model's inputs/parameters.
    batch.x = torch.empty((0, 178), dtype=torch.float32, device=device)
    batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    return batch


def _prediction(model, batch):
    output = model(batch)
    return output.reshape(-1)


if __name__ == "__main__":
    main()
