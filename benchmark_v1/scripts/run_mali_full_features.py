#!/usr/bin/env python3
"""Run the frozen Ma–Li full-feature graph and matched-MLP OOF cells.

This runner is deliberately inert without both a PASS C4 receipt and an
explicit coordinator GPU handoff ID. It never starts timing or changes frozen
data/splits/protocols. The neural environment is selected by the caller.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "benchmark_v1/scripts"
PROTOCOL = ROOT / "benchmark_v1/protocol/mali_full_features.json"
EXECUTION = ROOT / "benchmark_v1/protocol/estimator_execution.json"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
INNER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv"
FEATURE_ROOT = ROOT / "artifacts/real_qpu/mali_full_features"
OUT_ROOT = FEATURE_ROOT
AUTHORIZATION_PATH = ROOT / "artifacts/unified_runtime_estimation/execution_authorization.json"
C1_C3_RECEIPT_PATH = ROOT / "artifacts/unified_runtime_estimation/c1_c3_gate_receipt.json"
PREFLIGHT_HANDOFF_PATH = ROOT / "artifacts/unified_runtime_estimation/preflight_gpu_handoff.json"
TECHNICAL_GATE_RECEIPT_PATH = ROOT / "artifacts/unified_runtime_estimation/technical_gate_receipt.json"
INITIAL_CELLS_QA_PATH = ROOT / "artifacts/unified_runtime_estimation/initial_cells_qa_receipt.json"
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
    "materializer": ROOT / "benchmark_v1/scripts/materialize_mali_full_features.py",
    "feature_library": ROOT / "benchmark_v1/scripts/mali_full_features.py",
    "qasm3_hardware_wire_adapter": ROOT / "benchmark_v1/scripts/qasm3_hardware_wire_adapter.py",
}
METHODS = ("mali_full_features_graph", "mali_full_features_mlp")
SEEDS = (42, 1234, 31415)
NODE_MICROBATCH_LIMIT = 250_000
CHECKPOINT_INTERVAL_EPOCHS = 10

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mali_full_features import (  # noqa: E402
    SOURCE_PINS,
    expand_dense_x,
    sha256_bytes,
    sha256_file,
    stable_hash,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def canonical_protocol_hashes() -> tuple[dict[str, Any], str, str]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    expected = execution["method_contracts"]["mali"]
    protocol_hash = sha256_file(PROTOCOL)
    execution_hash = sha256_file(EXECUTION)
    if protocol_hash != expected["sha256"]:
        raise RuntimeError("mali_protocol_hash_mismatch")
    for name, pin in protocol["input_pins"].items():
        path = ROOT / pin["path"]
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
        "execution_authorization_sha256": sha256_file(AUTHORIZATION_PATH),
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


def validate_gpu_handoff(handoff_id: str) -> str:
    value = str(handoff_id).strip()
    if not value:
        raise RuntimeError("explicit_coordinator_gpu_handoff_id_required")
    return value


def feature_root_relative(feature_root: Path) -> str:
    resolved = feature_root.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RuntimeError("feature_root_must_be_inside_repository")
    return str(resolved.relative_to(ROOT.resolve()))


def validate_feature_root_binding(receipt: dict[str, Any], feature_root: Path,
                                  required: bool = False) -> None:
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


def resolve_feature_paths(feature_root_arg: Path | None,
                          output_dir_arg: Path | None) -> tuple[Path, Path]:
    selected_root = (feature_root_arg or FEATURE_ROOT).resolve()
    feature_root_relative(selected_root)
    selected_output = (output_dir_arg or selected_root / "training").resolve()
    if selected_root not in selected_output.parents:
        raise RuntimeError("training_output_must_remain_under_selected_feature_root")
    return selected_root, selected_output


def validate_neural_environment(input_manifest: dict[str, Any], feature_root: Path) -> dict[str, Any]:
    import importlib.metadata as metadata
    environment = input_manifest.get("environment", {})
    if not environment:
        # Completed historical materializers retained lock files but omitted
        # environment in their final manifest. Use the coordinator's separately
        # pinned receipt; never rewrite the completed materialization artifact.
        if not C1_C3_RECEIPT_PATH.is_file():
            raise RuntimeError("materialization_environment_missing_and_c1_c3_receipt_missing")
        receipt = json.loads(C1_C3_RECEIPT_PATH.read_text(encoding="utf-8"))
        validate_feature_root_binding(receipt, feature_root, required=True)
        pins = receipt.get("pins", receipt)
        expected = {
            "input_manifest_sha256": sha256_file(feature_root / "input_manifest.json"),
            "feature_status_ledger_sha256": sha256_file(feature_root / "feature_status_ledger.csv"),
        }
        if (receipt.get("overall_status") != "PASS" or not isinstance(pins, dict)
                or any(pins.get(name) != value for name, value in expected.items())
                or receipt.get("execution_authorization_sha256") != sha256_file(AUTHORIZATION_PATH)):
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
            for dist in metadata.distributions() if dist.metadata.get("Name")
        )
    )
    if sha256_bytes(actual_freeze.encode("utf-8")) != environment.get("neural_lock_sha256"):
        raise RuntimeError("current_neural_packages_differ_from_materialization_lock")
    return {"python": current_python, "lock_file": lock_name,
            "lock_sha256": environment["neural_lock_sha256"]}


def validate_initial_qa_receipt(receipt_path: Path, training_dir: Path,
                                technical_receipt_path: Path, protocol_hash: str,
                                execution_hash: str, input_hash: str, ledger_hash: str,
                                feature_root: Path = FEATURE_ROOT) -> dict[str, Any]:
    """Require the independent root review of the exact shared 11-cell gate."""
    if not receipt_path.is_file():
        raise RuntimeError("initial_cell_qa_receipt_missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_feature_root_binding(receipt, feature_root)
    if receipt.get("overall_status") != "PASS":
        raise RuntimeError("initial_cell_qa_receipt_not_pass")
    run_manifest_path = training_dir / "run_manifest.json"
    expected = {
        "technical_gate_receipt_sha256": sha256_file(technical_receipt_path),
        "execution_authorization_sha256": sha256_file(AUTHORIZATION_PATH),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"initial_cell_qa_receipt_pin_mismatch:{key}")
    initial_run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    validate_initial_stage_state(initial_run_manifest, sha256_file(receipt_path))
    cells = receipt.get("initial_cells")
    if not isinstance(cells, dict):
        raise RuntimeError("initial_cell_qa_receipt_cell_set_incomplete")
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    expected_cells = {
        cell["cell_id"]
        for section in ("qpu_cells", "simulator_cells")
        for cell in execution.get(section, [])
        if cell.get("outer_fold") == 0
    }
    if len(expected_cells) != 11:
        raise RuntimeError("execution_contract_initial_cell_set_invalid")
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        unexpected = sorted(set(cells) - expected_cells)
        raise RuntimeError(
            f"initial_cell_qa_receipt_cell_set_mismatch:missing={missing}:unexpected={unexpected}"
        )
    for cell_id in expected_cells:
        details = cells[cell_id]
        if not isinstance(details, dict) or details.get("qa_status") != "PASS":
            raise RuntimeError(f"initial_cell_qa_cell_not_integrity_pass:{cell_id}")
        terminal = details.get("terminal_status")
        allowed_terminals = {"completed", "resource_failure", "representation_unavailable"}
        if cell_id.split("/", 1)[0] == "spirit_sprinters_mps_runtime":
            allowed_terminals.add("failure_nonfinite_prediction")
        if terminal not in allowed_terminals:
            raise RuntimeError(f"initial_cell_qa_invalid_terminal_status:{cell_id}")
        files = details.get("files")
        if not isinstance(files, list):
            raise RuntimeError(f"initial_cell_qa_files_missing:{cell_id}")
        verified_names = set()
        parts = cell_id.split("/")
        method_name = parts[0]
        is_mali_cell = method_name in METHODS
        expected_base = None
        if is_mali_cell:
            _method, fold_name, seed_name = parts
            expected_base = (training_dir / fold_name / method_name / seed_name).resolve()
        verified_paths = set()
        terminal_sources = []
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise RuntimeError(f"initial_cell_qa_file_record_invalid:{cell_id}")
            path = (ROOT / item["path"]).resolve()
            if ROOT not in path.parents or not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise RuntimeError(f"initial_cell_qa_file_hash_mismatch:{cell_id}:{item.get('path')}")
            if expected_base is not None and expected_base not in path.parents:
                raise RuntimeError(f"initial_cell_qa_file_outside_cell_directory:{cell_id}:{item.get('path')}")
            verified_names.add(path.name)
            verified_paths.add(path)
            if path.suffix.lower() == ".json":
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(manifest, dict) and manifest.get("terminal_status") is not None:
                    terminal_sources.append((str(path), manifest["terminal_status"]))
        if not terminal_sources:
            raise RuntimeError(f"initial_cell_qa_terminal_manifest_not_pinned:{cell_id}")
        source_statuses = {status for _path, status in terminal_sources}
        if source_statuses != {terminal}:
            raise RuntimeError(
                f"initial_cell_qa_terminal_status_mismatch:{cell_id}:receipt={terminal}:manifest={sorted(source_statuses)}"
            )
        if is_mali_cell:
            cell_manifest_path = expected_base / "cell_manifest.json"
            predictions_path = expected_base / "predictions.csv"
            if cell_manifest_path not in verified_paths:
                raise RuntimeError(f"initial_cell_qa_manifest_not_pinned:{cell_id}")
            if "cell_manifest.json" not in verified_names:
                raise RuntimeError(f"initial_cell_qa_manifest_not_pinned:{cell_id}")
            if terminal == "completed" and predictions_path not in verified_paths:
                raise RuntimeError(f"initial_cell_qa_predictions_not_pinned:{cell_id}")
    return receipt


def validate_initial_stage_state(manifest: dict[str, Any], receipt_sha256: str) -> None:
    status = manifest.get("status")
    if status == "awaiting_independent_initial_cell_qa":
        return
    if status in {"running", "complete"} and manifest.get("initial_cells_qa_receipt_sha256") == receipt_sha256:
        return
    raise RuntimeError("remaining_folds_require_completed_fold0_and_independent_QA")


def validate_preflight_authorization(handoff_path: Path, handoff_id: str,
                                     protocol_hash: str, execution_hash: str,
                                     input_hash: str, ledger_hash: str,
                                     feature_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not handoff_path.is_file() or not C1_C3_RECEIPT_PATH.is_file():
        raise RuntimeError("preflight_handoff_or_c1_c3_receipt_missing")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if feature_root is not None:
        validate_feature_root_binding(handoff, feature_root)
    if (handoff.get("status") != "authorized"
            or handoff.get("handoff_id") != handoff_id
            or handoff.get("scope") != "technical_cuda_forward_backward_checkpoint_resume_only"
            or handoff.get("allow_scientific_fit") is not False
            or handoff.get("allow_prediction_export") is not False):
        raise RuntimeError("preflight_handoff_scope_or_authority_invalid")
    c1_c3_ref = handoff.get("c1_c3_receipt")
    if (not isinstance(c1_c3_ref, dict) or c1_c3_ref.get("path") != str(C1_C3_RECEIPT_PATH.relative_to(ROOT))
            or c1_c3_ref.get("sha256") != sha256_file(C1_C3_RECEIPT_PATH)):
        raise RuntimeError("preflight_handoff_C1_C3_reference_mismatch")
    if handoff.get("execution_authorization_sha256") != sha256_file(AUTHORIZATION_PATH):
        raise RuntimeError("preflight_handoff_execution_authorization_hash_mismatch")
    receipt = json.loads(C1_C3_RECEIPT_PATH.read_text(encoding="utf-8"))
    if feature_root is not None:
        validate_feature_root_binding(receipt, feature_root)
    if receipt.get("overall_status") != "PASS":
        raise RuntimeError("c1_c3_receipt_not_pass")
    gates = receipt.get("gates", {})
    if not isinstance(gates, dict) or any(gates.get(f"C{index}") != "PASS" for index in range(1, 4)):
        raise RuntimeError("c1_c3_receipt_gate_status_missing")
    expected = {"execution_authorization_sha256": sha256_file(AUTHORIZATION_PATH)}
    contracts = receipt.get("contract_hashes", {})
    if not isinstance(contracts, dict) or contracts.get("mali_full_features") != protocol_hash:
        raise RuntimeError("c1_c3_preflight_contract_hash_mismatch")
    pins = receipt.get("pins", receipt)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"c1_c3_preflight_receipt_pin_mismatch:{key}")
    expected_hashes = {"execution_protocol_sha256": execution_hash,
                       "input_manifest_sha256": input_hash,
                       "feature_status_ledger_sha256": ledger_hash}
    if not isinstance(pins, dict) or any(pins.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("c1_c3_preflight_active_pin_mismatch")
    return handoff, receipt


def load_pinned_model():
    """Execute only the hash-pinned upstream model source in an inert namespace."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv, global_mean_pool
    from torch_geometric.nn.norm import LayerNorm

    relative = "model/transformer_model.py"
    expected = SOURCE_PINS[relative]
    blob = subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "Quantum-Execution-Time-Prediction"), "show",
         f"{PINNED_MODEL_REVISION}:{relative}"]
    )
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
    from types import SimpleNamespace
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import global_mean_pool
    from torch.utils.checkpoint import checkpoint

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


def fold_partitions(outer_rows: list[dict[str, str]], inner_rows: list[dict[str, str]], fold: int) -> dict[str, list[str]]:
    outer_by_id = {row["canonical_observation_id"]: row for row in outer_rows}
    if len(outer_by_id) != len(outer_rows) or len(outer_by_id) != 8767:
        raise RuntimeError("outer_split_identity_not_one_to_one_8767")
    inner_by_id_fold = {}
    for row in inner_rows:
        key = (row["canonical_observation_id"], int(row["outer_fold"]))
        if key in inner_by_id_fold:
            raise RuntimeError("duplicate_inner_split_identity_fold")
        if key[0] not in outer_by_id:
            raise RuntimeError("inner_split_unknown_canonical_identity")
        if int(outer_by_id[key[0]]["outer_fold"]) == key[1]:
            if row["inner_fold"] != "":
                raise RuntimeError("inner_split_outer_test_must_have_blank_inner_assignment")
            inner_by_id_fold[key] = None
        else:
            if row["inner_fold"] == "":
                raise RuntimeError("inner_split_outer_train_assignment_missing")
            inner_by_id_fold[key] = int(row["inner_fold"])
    outer_train = sorted(canonical_id for canonical_id, row in outer_by_id.items() if int(row["outer_fold"]) != fold)
    test = sorted(canonical_id for canonical_id, row in outer_by_id.items() if int(row["outer_fold"]) == fold)
    fit, valid = [], []
    for canonical_id in outer_train:
        inner_fold = inner_by_id_fold.get((canonical_id, fold))
        if inner_fold is None:
            raise RuntimeError(f"inner_split_missing_outer_train_id:{fold}:{canonical_id}")
        if inner_fold == 0:
            valid.append(canonical_id)
        elif inner_fold in {1, 2, 3}:
            fit.append(canonical_id)
        else:
            raise RuntimeError(f"unexpected_inner_fold:{fold}:{canonical_id}:{inner_fold}")
    if not fit or not valid or not test:
        raise RuntimeError(f"empty_outer_fold_partition:{fold}")
    groups = {canonical_id: row["unified_leakage_group_id"] for canonical_id, row in outer_by_id.items()}
    if ({groups[x] for x in fit} & {groups[x] for x in valid + test}
            or {groups[x] for x in valid} & {groups[x] for x in test}):
        raise RuntimeError(f"unified_group_leakage_fold_{fold}")
    return {"fit": fit, "validation": valid, "test": test}


def fit_transform_global(raw_by_id: dict[str, list[float]], fit_ids: list[str]) -> dict[str, Any]:
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
    import numpy as np
    selected = np.asarray(raw, dtype=np.float32)[transform["retained_indices"]]
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    variable = np.asarray(transform["variable_mask"], dtype=bool)
    result = np.zeros_like(selected)
    result[variable] = (selected[variable] - mean[variable]) / std[variable]
    return result


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_gpu_budget(torch, budget_gib: int = 12, reserve_gib: int = 4) -> dict[str, Any]:
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
    effective_budget_bytes = min(budget_bytes, int(free_bytes) - reserve_gib * 1024**3
                                 + int(torch.cuda.memory_allocated(0)))
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


def common_eligible_partitions(partitions: dict[str, list[str]], ledger: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Use identical feature-eligible train/validation IDs for graph and MLP."""
    result: dict[str, list[str]] = {}
    for name in ("fit", "validation"):
        selected = []
        for canonical_id in partitions[name]:
            row = ledger.get(canonical_id)
            if row is None:
                raise RuntimeError(f"feature_ledger_missing_split_id:{canonical_id}")
            if row.get("global_status") == "available" and row.get("graph_status") == "available":
                selected.append(canonical_id)
        if not selected:
            raise RuntimeError(f"no_common_graph_mlp_{name}_rows")
        result[name] = selected
    result["test"] = list(partitions["test"])
    if set(result["fit"]) & set(result["validation"]):
        raise RuntimeError("common_fit_validation_identity_overlap")
    return result


def checkpoint_identity(method: str, fold: int, seed: int, protocol_hash: str,
                        execution_hash: str, input_manifest_hash: str,
                        ledger_hash: str, handoff_id: str,
                        activation_recomputation_enabled: bool,
                        feature_implementation_sha256: str) -> dict[str, Any]:
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
        "gpu_handoff_id": handoff_id,
        "activation_recomputation_enabled": bool(activation_recomputation_enabled),
    }


def validate_checkpoint_identity(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = checkpoint.get("identity")
    if actual != expected:
        raise RuntimeError("checkpoint_identity_mismatch")


def validate_feature_implementation_pins(input_manifest: dict[str, Any]) -> str:
    """Bind training/checkpoints to the exact materializer and feature helpers."""
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


def effective_batch_microbatches(items: list[Any], node_counts: list[int], limit: int = NODE_MICROBATCH_LIMIT) -> list[list[Any]]:
    """Greedy stable partition; oversized graphs stay intact as singletons."""
    if len(items) != len(node_counts) or not items:
        raise ValueError("effective_batch_items_node_counts_mismatch")
    groups: list[list[Any]] = []
    current: list[Any] = []
    nodes = 0
    for item, count_value in zip(items, node_counts):
        count = int(count_value)
        if count < 0:
            raise ValueError("negative_graph_node_count")
        if current and (nodes + count > limit or nodes >= limit):
            groups.append(current)
            current, nodes = [], 0
        current.append(item)
        nodes += count
        if count > limit:
            groups.append(current)
            current, nodes = [], 0
    if current:
        groups.append(current)
    if [id(item) for group in groups for item in group] != [id(item) for item in items]:
        raise RuntimeError("microbatch_partition_dropped_or_reordered_items")
    return groups


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
        "mean": mean.tolist(), "std": std.tolist(),
        "variable_mask": (std > 1e-6).tolist(), "fit_node_count": int(count),
        "fit_ids_sha256": stable_hash(fit_ids),
        "graph_paths_sha256": stable_hash([sha256_file(path) for path in graph_paths]),
    }


def load_saved_checkpoint(path: Path, expected_identity: dict[str, Any], torch) -> dict[str, Any]:
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


def _make_example(canonical_id: str, variant: str, ledger: dict[str, dict[str, str]],
                  targets: dict[str, float] | None, transform: dict[str, Any],
                  feature_root: Path, device, include_target: bool,
                  verified_graph_paths: set[Path] | None = None):
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
        graph_path = (feature_root / artifact).resolve()
        if feature_root.resolve() not in graph_path.parents or not graph_path.is_file():
            raise RuntimeError(f"graph_artifact_missing_or_outside_root:{canonical_id}")
        if (verified_graph_paths is None or graph_path not in verified_graph_paths) and sha256_file(graph_path) != row.get("graph_artifact_sha256"):
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
        data = Data(x=torch.as_tensor(x, dtype=torch.float32), edge_index=edge_index,
                    global_features=global_features)
    if include_target:
        if targets is None or canonical_id not in targets:
            raise RuntimeError(f"training_target_missing:{canonical_id}")
        data.y = torch.tensor([float(targets[canonical_id])], dtype=torch.float32)
    return data


def _example_node_count(data) -> int:
    x = getattr(data, "x", None)
    return int(x.shape[0]) if x is not None else 0


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


def _predict_test_groups(model, groups, device, graph: bool, torch):
    """Return ordered predictions and positions with inference-only CUDA OOM.

    Graph inference uses singleton groups, so an OOM belongs to that held-out
    observation. Fitting or validation OOM still fails the complete fit cell.
    """
    predicted_values = []
    failures = []
    offset = 0
    for group in groups:
        batch = None
        try:
            batch = _batch_to_device(group, device, graph)
            values = _prediction(model, batch).detach().cpu().numpy().reshape(-1)
            if len(values) != len(group) or not __import__("numpy").isfinite(values).all():
                raise RuntimeError("nonfinite_or_wrong_count_test_predictions")
            predicted_values.extend((offset + index, float(value)) for index, value in enumerate(values))
        except torch.cuda.OutOfMemoryError:
            if not graph or len(group) != 1:
                raise
            failures.append(offset)
        finally:
            del batch
        if failures and failures[-1] == offset:
            torch.cuda.empty_cache()
        offset += len(group)
    return predicted_values, failures


def _fit_cell(method: str, fold: int, seed: int, partitions: dict[str, list[str]],
              ledger: dict[str, dict[str, str]], targets: dict[str, float],
              feature_root: Path, output_dir: Path, transform: dict[str, Any],
              identity: dict[str, Any], protocol: dict[str, Any], receipt: dict[str, Any],
              torch, device, resume: bool, checkpoint_layers: bool) -> dict[str, Any]:
    import numpy as np
    import torch.nn.functional as F
    from torch_geometric.data import Batch

    graph = method == "mali_full_features_graph"
    verified_graph_paths: set[Path] = set()
    directory = output_dir / f"fold_{fold}" / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    final_path = directory / "predictions.csv"
    model_class, model_hash = load_pinned_model()
    global_width = len(transform["global"]["retained_indices"])
    model = make_model(model_class, method, global_width, checkpoint_layers=checkpoint_layers).to(device)
    if graph:
        model.mask = model.mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    start_epoch = 0
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    if checkpoint_path.exists() and resume:
        payload = load_saved_checkpoint(checkpoint_path, identity, torch)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["next_epoch"])
        best_loss = float(payload["best_validation_mse"])
        best_epoch = int(payload["best_epoch"])
        best_state = payload["best_model_state"]
    elif checkpoint_path.exists() and not resume:
        raise RuntimeError(f"checkpoint_exists_resume_required:{checkpoint_path}")
    else:
        seed_all(seed)
        model = make_model(model_class, method, global_width, checkpoint_layers=checkpoint_layers).to(device)
        if graph:
            model.mask = model.mask.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)

    epochs = int(protocol["training"]["epochs"])
    started = time.monotonic()
    latest_train_loss = None
    for epoch in range(start_epoch, epochs):
        model.train()
        order_rng = np.random.default_rng(int(stable_hash([fold, seed, epoch])[:16], 16))
        order = order_rng.permutation(len(partitions["fit"]))
        epoch_losses: list[float] = []
        for batch_start in range(0, len(order), 2):
            ids = [partitions["fit"][int(index)] for index in order[batch_start:batch_start + 2]]
            examples = [_make_example(cid, method, ledger, targets, transform, feature_root, device, True, verified_graph_paths) for cid in ids]
            node_counts = [_example_node_count(example) for example in examples]
            if graph:
                groups = effective_batch_microbatches(examples, node_counts)
            else:
                groups = [examples]
            effective_size = len(examples)
            optimizer.zero_grad(set_to_none=True)
            for group in groups:
                batch = _batch_to_device(group, device, graph)
                predicted = _prediction(model, batch)
                actual = torch.cat([example.y.reshape(-1) for example in group]).to(device)
                loss_sum = F.mse_loss(predicted, actual, reduction="sum")
                (loss_sum / effective_size).backward()
                epoch_losses.append(float((loss_sum.detach() / effective_size).cpu()))
                del batch, predicted, actual, loss_sum
            optimizer.step()
            if graph:
                del examples, groups
                torch.cuda.empty_cache()
        scheduler.step()
        latest_train_loss = float(sum(epoch_losses))

        model.eval()
        validation_squared_error = 0.0
        validation_count = 0
        with torch.no_grad():
            for start in range(0, len(partitions["validation"]), 2):
                ids = partitions["validation"][start:start + 2]
                examples = [_make_example(cid, method, ledger, targets, transform, feature_root, device, True, verified_graph_paths) for cid in ids]
                node_counts = [_example_node_count(example) for example in examples]
                groups = effective_batch_microbatches(examples, node_counts) if graph else [examples]
                for group in groups:
                    batch = _batch_to_device(group, device, graph)
                    predicted = _prediction(model, batch)
                    actual = torch.cat([example.y.reshape(-1) for example in group]).to(device)
                    validation_squared_error += float(F.mse_loss(predicted, actual, reduction="sum").cpu())
                    validation_count += len(group)
                    del batch, predicted, actual
                if graph:
                    del examples, groups
                    torch.cuda.empty_cache()
        validation_mse = validation_squared_error / validation_count
        if not math.isfinite(validation_mse):
            raise RuntimeError(f"nonfinite_validation_mse:{method}:{fold}:{seed}:{epoch}")
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        completed = epoch + 1
        if completed % CHECKPOINT_INTERVAL_EPOCHS == 0 or completed == epochs:
            save_checkpoint(checkpoint_path, {
                "identity": identity,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "completed_epoch": completed,
                "next_epoch": completed,
                "best_validation_mse": best_loss,
                "best_epoch": best_epoch,
                "best_model_state": best_state,
                "fit_ids_sha256": stable_hash(partitions["fit"]),
                "validation_ids_sha256": stable_hash(partitions["validation"]),
                "transform": transform,
                "last_train_loss_sum": latest_train_loss,
            })
            atomic_json(directory / "cell_progress.json", {
                "status": "training", "method": method, "outer_fold": fold, "seed": seed,
                "completed_epoch": completed, "best_epoch": best_epoch,
                "best_validation_mse_raw_seconds_squared": best_loss,
                "elapsed_seconds": time.monotonic() - started,
                "gpu_handoff_id": identity["gpu_handoff_id"],
            })

    if best_state is None or best_epoch < 0:
        raise RuntimeError("best_validation_checkpoint_missing")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    inference_failures = []
    with torch.no_grad():
        for start in range(0, len(partitions["test"]), 1 if graph else 256):
            ids = partitions["test"][start:start + (1 if graph else 256)]
            eligible_ids = [cid for cid in ids if ledger[cid].get("global_status") == "available"
                            and (not graph or ledger[cid].get("graph_status") == "available")]
            if not eligible_ids:
                continue
            examples = [_make_example(cid, method, ledger, None, transform, feature_root, device, False, verified_graph_paths) for cid in eligible_ids]
            groups = effective_batch_microbatches(examples, [_example_node_count(x) for x in examples]) if graph else [examples]
            predicted, failed_positions = _predict_test_groups(model, groups, device, graph, torch)
            predictions.extend((eligible_ids[index], value) for index, value in predicted)
            inference_failures.extend({"canonical_row_id": eligible_ids[index], "outer_fold": fold,
                                       "method": method, "seed": seed, "status": "resource_failure",
                                       "reason": "cuda_out_of_memory_during_heldout_inference"}
                                      for index in failed_positions)
            if graph:
                del examples, groups
                torch.cuda.empty_cache()
    # The graph prediction loop uses singleton chunks, while MLP chunks retain order.
    if len(predictions) != len(set(cid for cid, _value in predictions)):
        raise RuntimeError("duplicate_test_prediction_identity")
    rows = []
    for canonical_id, predicted_seconds in predictions:
        rows.append({
            "canonical_row_id": canonical_id, "outer_fold": fold, "method": method, "seed": seed,
            "actual_seconds": targets[canonical_id], "predicted_seconds": predicted_seconds,
            "absolute_error_seconds": abs(predicted_seconds - targets[canonical_id]),
            "negative_prediction": int(predicted_seconds < 0),
        })
    write_csv_atomic(final_path, rows, ["canonical_row_id", "outer_fold", "method", "seed", "actual_seconds",
                                       "predicted_seconds", "absolute_error_seconds", "negative_prediction"])
    failure_path = directory / "test_resource_failures.csv"
    write_csv_atomic(failure_path, inference_failures,
                     ["canonical_row_id", "outer_fold", "method", "seed", "status", "reason"])
    manifest = {
        "qa_status": "PASS", "status": "complete", "terminal_status": "completed",
        "method": method, "outer_fold": fold, "seed": seed,
        "epochs": epochs, "best_epoch": best_epoch, "best_validation_mse_raw_seconds_squared": best_loss,
        "n_fit": len(partitions["fit"]), "n_validation": len(partitions["validation"]),
        "n_test_predicted": len(rows), "n_test_assigned": len(partitions["test"]),
        "n_test_resource_failures": len(inference_failures),
        "fit_ids_sha256": stable_hash(partitions["fit"]),
        "validation_ids_sha256": stable_hash(partitions["validation"]),
        "test_ids_sha256": stable_hash(partitions["test"]),
        "identity": identity, "pinned_upstream_model_sha256": model_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path), "predictions_sha256": sha256_file(final_path),
        "test_resource_failures_sha256": sha256_file(failure_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(directory / "cell_manifest.json", manifest)
    atomic_json(directory / "cell_progress.json", manifest)
    return manifest


def record_resource_failure(method: str, fold: int, seed: int, output_dir: Path,
                            identity: dict[str, Any], partitions: dict[str, list[str]],
                            error: BaseException) -> dict[str, Any]:
    cell_dir = output_dir / f"fold_{fold}" / method / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cell_dir / "checkpoint.pt"
    manifest = {
        "qa_status": "PASS", "status": "resource_failure", "terminal_status": "resource_failure",
        "method": method, "outer_fold": fold, "seed": seed, "identity": identity,
        "reason": "cuda_out_of_memory", "error_type": type(error).__name__, "error": str(error),
        "n_fit": len(partitions["fit"]), "n_validation": len(partitions["validation"]),
        "n_test_assigned": len(partitions["test"]),
        "fit_ids_sha256": stable_hash(partitions["fit"]),
        "validation_ids_sha256": stable_hash(partitions["validation"]),
        "test_ids_sha256": stable_hash(partitions["test"]),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "predictions_sha256": None,
    }
    atomic_json(cell_dir / "cell_manifest.json", manifest)
    atomic_json(cell_dir / "cell_progress.json", manifest)
    return manifest


def _initial_cell_file_index(output_dir: Path) -> dict[str, Any]:
    cells = {}
    for method in METHODS:
        for seed in SEEDS:
            base = output_dir / "fold_0" / method / f"seed_{seed}"
            cell_id = f"{method}/fold_0/seed_{seed}"
            files = []
            for name in ("cell_manifest.json", "predictions.csv", "test_resource_failures.csv", "checkpoint.pt"):
                path = base / name
                if path.is_file():
                    files.append({"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256_file(path)})
            cell_manifest = json.loads((base / "cell_manifest.json").read_text())
            cells[cell_id] = {"qa_status": cell_manifest.get("qa_status", "FAIL"),
                              "terminal_status": cell_manifest.get("terminal_status", cell_manifest.get("status")),
                              "files": files}
    return cells


def _load_all_cell_manifests(output_dir: Path) -> list[dict[str, Any]]:
    cells = []
    expected = {(method, fold, seed) for method in METHODS for fold in range(5) for seed in SEEDS}
    found = set()
    for method, fold, seed in sorted(expected):
        path = output_dir / f"fold_{fold}" / method / f"seed_{seed}" / "cell_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"required_cell_manifest_missing:{method}:{fold}:{seed}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        key = (manifest.get("method"), int(manifest.get("outer_fold", -1)), int(manifest.get("seed", -1)))
        if key != (method, fold, seed) or manifest.get("qa_status") != "PASS":
            raise RuntimeError(f"cell_manifest_integrity_failed:{method}:{fold}:{seed}")
        found.add(key)
        cells.append(manifest)
    if found != expected:
        raise RuntimeError("cell_manifest_product_not_30")
    return cells


def _write_final_outputs(output_dir: Path, cells: list[dict[str, Any]],
                         canonical_by_id: dict[str, dict[str, str]],
                         ledger: dict[str, dict[str, str]], outer_rows: list[dict[str, str]],
                         targets: dict[str, float]) -> None:
    import numpy as np
    seed_rows: list[dict[str, str]] = []
    inference_failures = set()
    for cell in cells:
        if cell.get("terminal_status") != "completed":
            continue
        path = output_dir / f"fold_{cell['outer_fold']}" / cell["method"] / f"seed_{cell['seed']}" / "predictions.csv"
        if not path.is_file() or sha256_file(path) != cell.get("predictions_sha256"):
            raise RuntimeError(f"cell_prediction_hash_mismatch:{cell['method']}:{cell['outer_fold']}:{cell['seed']}")
        seed_rows.extend(read_csv(path))
        if cell.get("test_resource_failures_sha256") is not None:
            failure_path = path.parent / "test_resource_failures.csv"
            if not failure_path.is_file() or sha256_file(failure_path) != cell["test_resource_failures_sha256"]:
                raise RuntimeError("heldout_inference_resource_failure_file_hash_mismatch")
            inference_failures.update((row["canonical_row_id"], cell["method"], int(cell["seed"]))
                                      for row in read_csv(failure_path))
    write_csv_atomic(output_dir / "per_seed_predictions.csv", seed_rows,
                     ["canonical_row_id", "outer_fold", "method", "seed", "actual_seconds",
                      "predicted_seconds", "absolute_error_seconds", "negative_prediction"])
    by_prediction: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for row in seed_rows:
        by_prediction.setdefault((row["canonical_row_id"], int(row["outer_fold"]), row["method"]), []).append(row)
    cell_terminal = {(cell["method"], int(cell["outer_fold"]), int(cell["seed"])): cell["terminal_status"] for cell in cells}
    outer_by_id = {row["canonical_observation_id"]: row for row in outer_rows}
    assigned = []
    for canonical_id, outer in sorted(outer_by_id.items()):
        fold = int(outer["outer_fold"])
        for method in METHODS:
            predictions = by_prediction.get((canonical_id, fold, method), [])
            values = [float(row["predicted_seconds"]) for row in predictions]
            if len(values) == len(SEEDS):
                prediction = float(np.median(values))
                status, reason = "complete", ""
            else:
                terminals = [cell_terminal[(method, fold, seed)] for seed in SEEDS]
                if method == METHODS[0] and ledger[canonical_id].get("graph_status") != "available":
                    status, reason = "unavailable_feature", ledger[canonical_id].get("graph_reason", "graph_unavailable")
                elif ledger[canonical_id].get("global_status") != "available":
                    status, reason = "unavailable_feature", ledger[canonical_id].get("global_reason", "global_unavailable")
                elif "resource_failure" in terminals:
                    status, reason = "resource_failure", "one_or_more_seed_fits_resource_failed"
                elif any((canonical_id, method, seed) in inference_failures for seed in SEEDS):
                    status, reason = "resource_failure", "one_or_more_seed_heldout_inferences_resource_failed"
                else:
                    status, reason = "partial_seed_set", "three_seed_predictions_not_available"
                prediction = ""
            actual = targets[canonical_id]
            assigned.append({"canonical_row_id": canonical_id, "source_id": canonical_by_id[canonical_id]["source_id"],
                             "outer_fold": fold, "method": method, "status": status, "reason": reason,
                             "n_seed_predictions": len(values), "actual_seconds": actual,
                             "predicted_seconds": prediction,
                             "absolute_error_seconds": abs(prediction - actual) if prediction != "" else "",
                             "negative_prediction": int(prediction < 0) if prediction != "" else ""})
    write_csv_atomic(output_dir / "oof_attempts.csv", assigned,
                     ["canonical_row_id", "source_id", "outer_fold", "method", "status", "reason",
                      "n_seed_predictions", "actual_seconds", "predicted_seconds", "absolute_error_seconds",
                      "negative_prediction"])
    metric_rows = []
    source_rows = []
    for method in METHODS:
        successful = [row for row in assigned if row["method"] == method and row["status"] == "complete"]
        if successful:
            actual = np.asarray([float(row["actual_seconds"]) for row in successful], dtype=np.float64)
            predicted = np.asarray([float(row["predicted_seconds"]) for row in successful], dtype=np.float64)
            metric_rows.append({"method": method, **_metric_summary(actual, predicted, np.abs(predicted - actual))})
            for source in sorted({row["source_id"] for row in successful}):
                subset = [row for row in successful if row["source_id"] == source]
                source_actual = np.asarray([float(row["actual_seconds"]) for row in subset], dtype=np.float64)
                source_predicted = np.asarray([float(row["predicted_seconds"]) for row in subset], dtype=np.float64)
                source_rows.append({"method": method, "source_id": source,
                                    **_metric_summary(source_actual, source_predicted,
                                                      np.abs(source_predicted - source_actual))})
    metric_fields = ["method", "n", "MAE_seconds", "MedAE_seconds", "RMSE_seconds", "R2_seconds",
                     "p90_abs_error_seconds", "p99_abs_error_seconds", "max_abs_error_seconds", "negative_prediction_count"]
    write_csv_atomic(output_dir / "metrics.csv", metric_rows, metric_fields)
    write_csv_atomic(output_dir / "source_metrics.csv", source_rows, ["method", "source_id", *metric_fields[1:]])
    successful = {(row["canonical_row_id"], row["method"]): row for row in assigned if row["status"] == "complete"}
    paired = []
    for canonical_id in sorted({key[0] for key in successful}):
        graph = successful.get((canonical_id, METHODS[0]))
        mlp = successful.get((canonical_id, METHODS[1]))
        if graph and mlp:
            paired.append({"canonical_row_id": canonical_id, "source_id": graph["source_id"],
                           "outer_fold": graph["outer_fold"], "actual_seconds": graph["actual_seconds"],
                           "graph_prediction_seconds": graph["predicted_seconds"],
                           "mlp_prediction_seconds": mlp["predicted_seconds"],
                           "graph_absolute_error_seconds": graph["absolute_error_seconds"],
                           "mlp_absolute_error_seconds": mlp["absolute_error_seconds"]})
    write_csv_atomic(output_dir / "paired_intersections.csv", paired,
                     ["canonical_row_id", "source_id", "outer_fold", "actual_seconds", "graph_prediction_seconds",
                      "mlp_prediction_seconds", "graph_absolute_error_seconds", "mlp_absolute_error_seconds"])
    atomic_json(output_dir / "fold_seed_manifests.json", {"cells": cells,
                                                           "cell_count": len(cells),
                                                           "seed_reduction": "row-wise median of all three seeds only"})


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _compare_tensors(left, right, torch, label: str, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    if left is None or right is None:
        if left is not right:
            raise RuntimeError(f"cuda_preflight_gradient_presence_mismatch:{label}")
        return
    if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
        raise RuntimeError(f"cuda_preflight_finite_mask_mismatch:{label}")
    if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
        raise RuntimeError(f"cuda_preflight_parity_failed:{label}")


def _cuda_gradient_parity(torch, model_class, device) -> dict[str, Any]:
    """Compare checkpointed/plain gradients and batch-vs-microbatch gradients."""
    import copy
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data

    seed_all(9137)
    graph_width = 178
    global_width = 51
    first = Data(
        x=torch.randn((7, graph_width), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3], [1, 2, 3, 4, 5, 6, 0, 1]], dtype=torch.long),
        global_features=torch.randn((1, global_width), dtype=torch.float32),
        y=torch.tensor([0.75], dtype=torch.float32),
    )
    second = Data(
        x=torch.randn((5, graph_width), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 4, 0, 1]], dtype=torch.long),
        global_features=torch.randn((1, global_width), dtype=torch.float32),
        y=torch.tensor([1.25], dtype=torch.float32),
    )
    batch = Batch.from_data_list([first, second]).to(device)
    first, second = first.to(device), second.to(device)
    plain = make_model(model_class, METHODS[0], global_width, checkpoint_layers=False).to(device)
    plain.mask = plain.mask.to(device)
    checkpointed = make_model(model_class, METHODS[0], global_width, checkpoint_layers=True).to(device)
    checkpointed.mask = checkpointed.mask.to(device)
    checkpointed.load_state_dict(copy.deepcopy(plain.state_dict()))
    plain.train(); checkpointed.train()
    p_plain = plain(batch).reshape(-1)
    loss_plain = F.mse_loss(p_plain, batch.y.reshape(-1), reduction="sum") / 2
    loss_plain.backward()
    p_checkpointed = checkpointed(batch).reshape(-1)
    loss_checkpointed = F.mse_loss(p_checkpointed, batch.y.reshape(-1), reduction="sum") / 2
    loss_checkpointed.backward()
    _compare_tensors(p_plain.detach(), p_checkpointed.detach(), torch, "checkpointed_output")
    _compare_tensors(loss_plain.detach().reshape(1), loss_checkpointed.detach().reshape(1), torch, "checkpointed_loss")
    plain_parameters = dict(plain.named_parameters())
    checkpointed_parameters = dict(checkpointed.named_parameters())
    for name in plain_parameters:
        _compare_tensors(plain_parameters[name].grad, checkpointed_parameters[name].grad,
                         torch, f"checkpointed_gradient:{name}")

    micro = make_model(model_class, METHODS[0], global_width, checkpoint_layers=False).to(device)
    micro.mask = micro.mask.to(device)
    micro.load_state_dict(copy.deepcopy(plain.state_dict()))
    micro.zero_grad(set_to_none=True)
    for example in (first, second):
        prediction = micro(example).reshape(-1)
        (F.mse_loss(prediction, example.y.reshape(-1), reduction="sum") / 2).backward()
    for name, parameter in micro.named_parameters():
        _compare_tensors(plain_parameters[name].grad, parameter.grad, torch, f"microbatch_gradient:{name}")

    opt_plain = torch.optim.Adam(plain.parameters(), lr=0.0005, weight_decay=0.0001)
    opt_checkpointed = torch.optim.Adam(checkpointed.parameters(), lr=0.0005, weight_decay=0.0001)
    opt_plain.step(); opt_checkpointed.step()
    for name, parameter in plain.named_parameters():
        _compare_tensors(parameter.detach(), dict(checkpointed.named_parameters())[name].detach(),
                         torch, f"checkpointed_adam_step:{name}")
    return {"status": "PASS", "fixture_graphs": 2, "output_loss_gradient_and_adam_parity": True,
            "tolerances": {"rtol": 1e-5, "atol": 1e-6}}


def _cuda_checkpoint_resume_check(torch, model_class, device) -> dict[str, Any]:
    """Exercise atomic checkpoint restore and RNG continuation on a tiny fixture."""
    import copy
    import tempfile
    import torch.nn.functional as F
    from types import SimpleNamespace

    def global_batch(values):
        return _batch_to_device([SimpleNamespace(global_features=row.reshape(1, -1))
                                 for row in values], device, graph=False)

    seed_all(411)
    original = make_model(model_class, METHODS[1], 51).to(device)
    original_optimizer = torch.optim.Adam(original.parameters(), lr=0.0005, weight_decay=0.0001)
    x = torch.randn((2, 51), device=device)
    y = torch.randn((2,), device=device)
    original_optimizer.zero_grad(set_to_none=True)
    F.mse_loss(original(global_batch(x)).reshape(-1), y).backward()
    original_optimizer.step()
    identity = {"preflight_fixture": "mali-upstream-mlp", "seed": 411}
    state = {"identity": identity,
             "model_state": copy.deepcopy({key: value.detach().cpu() for key, value in original.state_dict().items()}),
             "optimizer_state": copy.deepcopy(original_optimizer.state_dict()),
             "rng_state": capture_rng_state(), "next_epoch": 1}
    with tempfile.TemporaryDirectory(prefix="mali-cuda-resume-") as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.pt"
        save_checkpoint(checkpoint_path, state)
        # Original takes the uninterrupted next step using its captured RNG.
        x_next = torch.randn((2, 51), device=device)
        y_next = torch.randn((2,), device=device)
        original_optimizer.zero_grad(set_to_none=True)
        F.mse_loss(original(global_batch(x_next)).reshape(-1), y_next).backward()
        original_optimizer.step()
        resumed = make_model(model_class, METHODS[1], 51).to(device)
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.0005, weight_decay=0.0001)
        payload = load_saved_checkpoint(checkpoint_path, identity, torch)
        resumed.load_state_dict(payload["model_state"])
        resumed_optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload["rng_state"])
        x_resume = torch.randn((2, 51), device=device)
        y_resume = torch.randn((2,), device=device)
        resumed_optimizer.zero_grad(set_to_none=True)
        F.mse_loss(resumed(global_batch(x_resume)).reshape(-1), y_resume).backward()
        resumed_optimizer.step()
        for name, value in original.named_parameters():
            _compare_tensors(value.detach(), dict(resumed.named_parameters())[name].detach(),
                             torch, f"checkpoint_resume_parameter:{name}", rtol=0, atol=0)
    return {"status": "PASS", "atomic_checkpoint": True, "optimizer_and_rng_resume_equivalence": True}


def _cuda_resource_mode_attempts(attempt, torch) -> tuple[list[dict[str, Any]], bool, str]:
    """Try intact plain/recomputed graphs; retain both failed dispositions.

    Leave each exception handler before the next attempt so its traceback does
    not retain failed model activations. No architecture or graph is changed.
    """
    import gc
    modes = []
    for checkpoint_layers in (False, True):
        try:
            result = attempt(checkpoint_layers)
        except torch.cuda.OutOfMemoryError:
            result = {"status": "OOM", "activation_recomputation_enabled": checkpoint_layers,
                      "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}
        modes.append(result)
        gc.collect()
        torch.cuda.empty_cache()
        if result["status"] == "PASS":
            return modes, checkpoint_layers, "available"
    return modes, True, "resource_failure"


def run_cuda_preflight(ledger: dict[str, dict[str, str]], feature_root: Path,
                       input_hash: str, ledger_hash: str, handoff_id: str) -> dict[str, Any]:
    """Resource-only largest-graph and checkpoint probes; never fit/export targets."""
    import torch
    from torch_geometric.data import Data

    sys.path.insert(0, str(SCRIPT_DIR))
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease("mali-full-features-cuda-preflight")
    try:
        resource_info = validate_gpu_budget(torch)
        device = torch.device("cuda:0")
        torch.set_num_threads(2)
        model_class, model_hash = load_pinned_model()
        candidates = [row for row in ledger.values() if row.get("graph_status") == "available"]
        if not candidates:
            raise RuntimeError("preflight_no_available_graphs")
        largest = max(candidates, key=lambda row: int(row.get("graph_node_count", 0)))
        graph_path = (feature_root / largest["graph_artifact"]).resolve()
        if sha256_file(graph_path) != largest.get("graph_artifact_sha256"):
            raise RuntimeError("preflight_largest_graph_hash_mismatch")
        record = _load_graph_npz(graph_path)
        import numpy as np
        x = expand_dense_x(record)
        edge_index = torch.as_tensor(record["edge_index"].astype(np.int64, copy=False), dtype=torch.long)
        data = Data(x=torch.as_tensor(x, dtype=torch.float32), edge_index=edge_index,
                    global_features=torch.zeros((1, 51), dtype=torch.float32),
                    batch=torch.zeros(len(x), dtype=torch.long))
        data = data.to(device)

        def attempt(checkpoint_layers: bool) -> dict[str, Any]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            seed_all(31_415)
            model = make_model(model_class, METHODS[0], 51, checkpoint_layers=checkpoint_layers).to(device)
            model.mask = model.mask.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
            started = time.monotonic()
            output = model(data).reshape(-1)
            loss = output.square().sum()
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise RuntimeError("preflight_largest_graph_nonfinite_gradient")
            optimizer.step()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise RuntimeError("preflight_largest_graph_nonfinite_output_or_loss")
            return {"status": "PASS", "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "activation_recomputation_enabled": checkpoint_layers}

        modes, selected_mode, largest_disposition = _cuda_resource_mode_attempts(attempt, torch)
        largest_identity = {"canonical_row_id": largest["canonical_row_id"],
                            "nodes": int(largest["graph_node_count"]),
                            "edges": int(largest["graph_edge_count"]),
                            "artifact_sha256": largest["graph_artifact_sha256"],
                            "terminal_status": largest_disposition}
        # The giant tensors otherwise remain on CUDA while the independent
        # tiny-fixture parity/resume checks run and distort their resource use.
        del data, x, edge_index, record
        __import__("gc").collect()
        torch.cuda.empty_cache()
        parity = _cuda_gradient_parity(torch, model_class, device)
        resume = _cuda_checkpoint_resume_check(torch, model_class, device)
        return {
            "artifact_id": "mali-full-features-cuda-technical-preflight",
            "status": "PASS", "scientific_fit_started": False,
            "prediction_exported": False, "gpu_handoff_id": handoff_id,
            "input_manifest_sha256": input_hash, "feature_status_ledger_sha256": ledger_hash,
            "pinned_model_sha256": model_hash, "gpu": resource_info,
            "largest_graph": largest_identity,
            "attempts": modes, "activation_recomputation_enabled": selected_mode,
            "gradient_parity": parity, "checkpoint_resume": resume,
            "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
            "resume_identity": {"preflight_fixture": "mali-upstream-mlp", "seed": 411},
        }
    finally:
        lease.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run the separately authorized technical CUDA resource/parity/resume probe only.")
    parser.add_argument("--preflight-handoff-id")
    parser.add_argument("--gpu-handoff-id")
    parser.add_argument("--feature-root", type=Path, default=None,
                        help="Feature/materialization root; defaults to the original protocol path.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Training output directory; defaults to <feature-root>/training.")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=[0],
                        help="Initial stage [0], or all folds after independent initial-cell QA receipt.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and (args.validate_only or args.no_resume or args.folds != [0]):
        raise SystemExit("--preflight-only cannot be combined with training-stage flags")
    if args.folds not in ([0], [0, 1, 2, 3, 4]):
        raise SystemExit("fold selection must be [0] or all five folds; fold 0 gate cannot be bypassed")
    feature_root, output_dir = resolve_feature_paths(args.feature_root, args.output_dir)
    if args.preflight_only:
        handoff_id = validate_gpu_handoff(args.preflight_handoff_id or "")
    else:
        handoff_id = validate_gpu_handoff(args.gpu_handoff_id or "")

    protocol, protocol_hash, execution_hash = canonical_protocol_hashes()
    input_path = feature_root / "input_manifest.json"
    ledger_path = feature_root / "feature_status_ledger.csv"
    materialization_path = feature_root / "run_manifest.json"
    for path in (input_path, ledger_path, materialization_path):
        if not path.is_file():
            raise RuntimeError(f"full_feature_materialization_missing:{path.name}")
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    ledger_hash = sha256_file(ledger_path)
    input_hash = sha256_file(input_path)
    feature_implementation_sha256 = validate_feature_implementation_pins(input_manifest)
    if input_manifest.get("full_panel_materialized") is not True or input_manifest.get("canonical_identity_count") != 8767:
        raise RuntimeError("full_8767_feature_materialization_not_complete")
    if input_manifest.get("feature_status_ledger_sha256") != ledger_hash or materialization.get("status") != "features_materialized":
        raise RuntimeError("feature_materialization_manifest_ledger_mismatch")
    if input_manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("feature_materialization_protocol_pin_mismatch")
    auth_input = input_manifest.get("inputs", {}).get("execution_authorization", {})
    if auth_input.get("sha256") != sha256_file(AUTHORIZATION_PATH):
        raise RuntimeError("feature_materialization_authorization_pin_mismatch")
    neural_environment = validate_neural_environment(input_manifest, feature_root)
    canonical_rows = read_csv(CANONICAL)
    outer_rows = read_csv(OUTER)
    inner_rows = read_csv(INNER)
    ledger_rows = read_csv(ledger_path)
    canonical_by_id = {row["canonical_row_id"]: row for row in canonical_rows}
    ledger = {row["canonical_row_id"]: row for row in ledger_rows}
    if len(canonical_by_id) != 8767 or len(ledger) != 8767 or set(canonical_by_id) != set(ledger):
        raise RuntimeError("feature_ledger_canonical_identity_mismatch")
    if {row["canonical_observation_id"] for row in outer_rows} != set(canonical_by_id):
        raise RuntimeError("outer_split_canonical_identity_mismatch")
    if {row["canonical_observation_id"] for row in inner_rows} != set(canonical_by_id):
        raise RuntimeError("inner_split_canonical_identity_mismatch")
    dynamic_id = protocol["dataset"].get("dynamic_row")
    if dynamic_id in ledger:
        dynamic = ledger[dynamic_id]
        if (dynamic.get("global_status") != "available" or dynamic.get("graph_status") != "unavailable"
                or "control_flow" not in dynamic.get("graph_reason", "")):
            raise RuntimeError("row4477_must_keep_valid_global_and_explicit_dynamic_graph_unavailable")

    if args.preflight_only:
        _handoff, c1_c3 = validate_preflight_authorization(
            PREFLIGHT_HANDOFF_PATH, handoff_id, protocol_hash, execution_hash, input_hash, ledger_hash,
            feature_root=feature_root)
        result = run_cuda_preflight(ledger, feature_root, input_hash, ledger_hash, handoff_id)
        result["feature_implementation_sha256"] = feature_implementation_sha256
        result["implementation_hashes"] = input_manifest["implementation_hashes"]
        result["execution_authorization_sha256"] = sha256_file(AUTHORIZATION_PATH)
        result["c1_c3_receipt_sha256"] = sha256_file(C1_C3_RECEIPT_PATH)
        result["c1_c3_receipt_status"] = c1_c3["overall_status"]
        result["c6_runner_sha256"] = sha256_file(Path(__file__))
        result["neural_environment"] = neural_environment
        result_path = feature_root / "preflight" / "preflight_report.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("gpu_handoff_id") != handoff_id:
                raise RuntimeError("preflight_report_exists_for_different_handoff")
        atomic_json(result_path, result)
        print(json.dumps({"status": "preflight_complete_no_scientific_fit", "report": str(result_path),
                          "report_sha256": sha256_file(result_path),
                          "activation_recomputation_enabled": result["activation_recomputation_enabled"]}, indent=2))
        return

    technical = validate_technical_gate_receipt(TECHNICAL_GATE_RECEIPT_PATH, feature_root)
    preflight_path = feature_root / "preflight" / "preflight_report.json"
    if not preflight_path.is_file() or technical.get("preflight_report_sha256") != sha256_file(preflight_path):
        raise RuntimeError("technical_gate_receipt_preflight_report_pin_mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "PASS" or preflight.get("scientific_fit_started") is not False
            or preflight.get("input_manifest_sha256") != input_hash
            or preflight.get("feature_status_ledger_sha256") != ledger_hash
            or preflight.get("feature_implementation_sha256") != feature_implementation_sha256
            or preflight.get("implementation_hashes") != input_manifest["implementation_hashes"]
            or preflight.get("c6_runner_sha256") != sha256_file(Path(__file__))):
        raise RuntimeError("technical_cuda_preflight_report_invalid_or_stale")
    if technical.get("activation_recomputation_enabled") != preflight.get("activation_recomputation_enabled"):
        raise RuntimeError("technical_gate_recomputation_selection_mismatch")
    if set(technical.get("eligible_methods", [])) != set(METHODS):
        raise RuntimeError("technical_gate_does_not_authorize_graph_MLP_pair")

    run_manifest_path = output_dir / "run_manifest.json"
    identity_root = {
        "feature_root": feature_root_relative(feature_root),
        "mali_protocol_sha256": protocol_hash,
        "execution_protocol_sha256": execution_hash,
        "execution_authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "neural_environment": neural_environment,
        "input_manifest_sha256": input_hash,
        "feature_status_ledger_sha256": ledger_hash,
        "feature_implementation_sha256": feature_implementation_sha256,
        "implementation_hashes": input_manifest["implementation_hashes"],
        "technical_gate_receipt_sha256": sha256_file(TECHNICAL_GATE_RECEIPT_PATH),
        "gpu_handoff_id": handoff_id,
        "folds": [0, 1, 2, 3, 4], "seeds": list(SEEDS), "methods": list(METHODS),
        "activation_recomputation_enabled": technical["activation_recomputation_enabled"],
    }
    existing_manifest = None
    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("identity") != identity_root:
            raise RuntimeError("training_run_identity_mismatch_refusing_resume")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("training_output_dir_nonempty_without_run_manifest")

    if args.folds == [0, 1, 2, 3, 4]:
        initial = validate_initial_qa_receipt(INITIAL_CELLS_QA_PATH, output_dir,
                                              TECHNICAL_GATE_RECEIPT_PATH, protocol_hash,
                                              execution_hash, input_hash, ledger_hash,
                                              feature_root=feature_root)
        if existing_manifest is None:
            raise RuntimeError("remaining_folds_require_completed_fold0_and_independent_QA")
        validate_initial_stage_state(existing_manifest, sha256_file(INITIAL_CELLS_QA_PATH))
    else:
        initial = None
    if args.validate_only:
        checks = []
        for fold in args.folds:
            parts = fold_partitions(outer_rows, inner_rows, fold)
            common = common_eligible_partitions(parts, ledger)
            checks.append({"fold": fold, "fit": len(common["fit"]), "validation": len(common["validation"]),
                           "test_assigned": len(common["test"])})
        print(json.dumps({"status": "validated_no_fit", "methods": list(METHODS), "folds": checks,
                          "initial_qa_receipt_verified": initial is not None,
                          "technical_gate_receipt_sha256": sha256_file(TECHNICAL_GATE_RECEIPT_PATH)}, indent=2))
        return

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required_no_CPU_fallback")
    torch.set_num_threads(2)
    sys.path.insert(0, str(SCRIPT_DIR))
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease("mali-full-features-graph-and-mlp")
    try:
        gpu = validate_gpu_budget(torch)
        device = torch.device("cuda:0")
        targets = {canonical_id: float(row["target_seconds"]) for canonical_id, row in canonical_by_id.items()}
        if not math.isfinite(sum(targets.values())):
            raise RuntimeError("nonfinite_canonical_targets")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_state = {"artifact_id": "mali-full-feature-neural-oof", "status": "running",
                     "identity": identity_root, "gpu": gpu,
                     "completed_cells": int(existing_manifest.get("completed_cells", 0)) if existing_manifest else 0,
                     "expected_cells": 30, "requested_stage_folds": args.folds,
                     "preflight_report_sha256": sha256_file(preflight_path)}
        if initial is not None:
            run_state["initial_cells_qa_receipt_sha256"] = sha256_file(INITIAL_CELLS_QA_PATH)
        atomic_json(run_manifest_path, run_state)
        global_raw = {canonical_id: [float(row[f"global_{index:02d}"]) for index in range(51)]
                      for canonical_id, row in ledger.items() if row.get("global_status") == "available"}
        all_cells: list[dict[str, Any]] = []
        for fold in args.folds:
            parts = fold_partitions(outer_rows, inner_rows, fold)
            eligible = common_eligible_partitions(parts, ledger)
            global_transform = fit_transform_global(global_raw, eligible["fit"])
            graph_paths = [(feature_root / ledger[cid]["graph_artifact"]).resolve() for cid in eligible["fit"]]
            node_transform = fit_node_transform(graph_paths, eligible["fit"])
            transform = {"global": global_transform, "node": node_transform, "outer_fold": fold,
                         "fit_ids_sha256": stable_hash(eligible["fit"]), "protocol_sha256": protocol_hash,
                         "input_manifest_sha256": input_hash}
            transform_path = output_dir / f"fold_{fold}" / "transform.json"
            if transform_path.exists():
                saved_transform = json.loads(transform_path.read_text(encoding="utf-8"))
                if saved_transform != transform:
                    raise RuntimeError(f"fold_transform_identity_mismatch:{fold}")
            else:
                atomic_json(transform_path, transform)
            fold_cells: list[dict[str, Any]] = []
            for method in METHODS:
                for seed in SEEDS:
                    relative = Path(f"fold_{fold}") / method / f"seed_{seed}"
                    cell_dir = output_dir / relative
                    cell_manifest_file = cell_dir / "cell_manifest.json"
                    expected_identity = checkpoint_identity(method, fold, seed, protocol_hash, execution_hash,
                                                           input_hash, ledger_hash, handoff_id,
                                                           technical["activation_recomputation_enabled"],
                                                           feature_implementation_sha256)
                    if cell_manifest_file.is_file():
                        cell = json.loads(cell_manifest_file.read_text(encoding="utf-8"))
                        if cell.get("identity") != expected_identity or cell.get("fit_ids_sha256") != stable_hash(eligible["fit"]):
                            raise RuntimeError(f"completed_cell_identity_mismatch:{method}:{fold}:{seed}")
                    else:
                        try:
                            cell = _fit_cell(method, fold, seed, eligible, ledger, targets, feature_root,
                                             output_dir, transform, expected_identity, protocol, technical, torch, device,
                                             resume=not args.no_resume,
                                             checkpoint_layers=technical["activation_recomputation_enabled"])
                        except torch.cuda.OutOfMemoryError as exc:
                            torch.cuda.empty_cache()
                            cell = record_resource_failure(method, fold, seed, output_dir,
                                                           expected_identity, eligible, exc)
                    fold_cells.append(cell)
                    all_cells.append(cell)
                    run_state.update({"completed_cells": len(all_cells),
                                      "last_cell": {"method": method, "outer_fold": fold, "seed": seed}})
                    atomic_json(run_manifest_path, run_state)
            if len(fold_cells) != 6:
                raise RuntimeError(f"fold_{fold}_expected_six_method_seed_cells")
            if any(cell.get("fit_ids_sha256") != stable_hash(eligible["fit"])
                   or cell.get("validation_ids_sha256") != stable_hash(eligible["validation"])
                   for cell in fold_cells):
                raise RuntimeError(f"fold_{fold}_matched_graph_mlp_partition_qa_failed")
            if fold == 0:
                fold_qa = {"status": "PASS", "outer_fold": 0, "cell_count": 6,
                           "fit_ids_sha256": stable_hash(eligible["fit"]),
                           "validation_ids_sha256": stable_hash(eligible["validation"]),
                           "terminal_statuses": {f"{c['method']}/seed_{c['seed']}": c["status"] for c in fold_cells}}
                atomic_json(output_dir / "fold_0_technical_qa.json", fold_qa)
                atomic_json(output_dir / "fold_seed_manifests.json", {"cells": all_cells, "fold_0_qa": fold_qa})
                run_state["status"] = "awaiting_independent_initial_cell_qa" if args.folds == [0] else "running"
                run_state["initial_cells"] = _initial_cell_file_index(output_dir)
                atomic_json(run_manifest_path, run_state)
                if args.folds == [0]:
                    print(json.dumps({"status": run_state["status"], "completed_cells": 6,
                                      "run_manifest": str(run_manifest_path)}, indent=2))
                    return

        # Aggregate only after all five folds have completed or have a valid
        # terminal resource-failure receipt; unavailable input rows stay assigned.
        all_cells = _load_all_cell_manifests(output_dir)
        _write_final_outputs(output_dir, all_cells, canonical_by_id, ledger, outer_rows, targets)
        run_state.update({"status": "complete", "completed_cells": len(all_cells), "expected_cells": 30,
                          "outputs": {name: sha256_file(output_dir / name) for name in
                                      ("per_seed_predictions.csv", "oof_attempts.csv", "metrics.csv",
                                       "source_metrics.csv", "paired_intersections.csv", "fold_seed_manifests.json")}})
        atomic_json(run_manifest_path, run_state)
        print(json.dumps({"status": "complete", "completed_cells": len(all_cells), "output_dir": str(output_dir)}, indent=2))
    finally:
        lease.release()


def _metric_summary(actual, predicted, error) -> dict[str, Any]:
    import numpy as np
    if len(actual) == 0:
        raise RuntimeError("empty_metric_denominator")
    residual = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    r2 = 1.0 - float(np.square(residual).sum()) / denominator if denominator > 0 else None
    return {
        "n": int(len(actual)), "MAE_seconds": float(np.mean(error)),
        "MedAE_seconds": float(np.median(error)), "RMSE_seconds": float(np.sqrt(np.mean(np.square(residual)))),
        "R2_seconds": r2, "p90_abs_error_seconds": float(np.quantile(error, .90)),
        "p99_abs_error_seconds": float(np.quantile(error, .99)),
        "max_abs_error_seconds": float(np.max(error)),
        "negative_prediction_count": int(np.count_nonzero(predicted < 0)),
    }
    if args.validate_only:
        print(json.dumps({"status": "validated_no_fit", "handoff_id": handoff_id,
                          "eligible_methods": receipt["eligible_methods"]}, indent=2))
        return
    # The rest of the run is added below; the lease is intentionally acquired
    # only after all C4, source, data, split and handoff gates are checked.
    raise RuntimeError("training_execution_not_ready: feature materialization or C6 implementation incomplete")


if __name__ == "__main__":
    main()
