#!/usr/bin/env python3
"""Create immutable pre-outcome code/data/protocol pins for expansion v1."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS, recorded
from pathlib import Path
import datetime as dt, hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"


def sha(p):
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    """Read a JSON file."""
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    """Create the immutable execution lock before any outcome is seen.

    Requires the previous lock, the protocol amendment and the passed feature, split, classical and batch
    receipts. Records the hashes of the code, the inputs and every graph file, with the git state, in
    execution_lock_v2.json. Refuses to overwrite an existing lock.
    """
    lock_path = OUT / "execution_lock_v2.json"
    if lock_path.exists():
        raise RuntimeError("protocol_already_locked_refuse_overwrite")
    previous_lock = OUT / "execution_lock.json"
    if not previous_lock.is_file():
        raise RuntimeError("pre_amendment_lock_missing")
    amendment = OUT / "PROTOCOL_AMENDMENT_01.md"
    if not amendment.is_file():
        raise RuntimeError("label_isolation_amendment_missing")
    feature = read(OUT / "cohort/feature_preflight_receipt.json")
    split = read(OUT / "splits/split_receipt.json")
    split_validation = read(OUT / "splits/split_validation_receipt.json")
    classic = read(OUT / "classical/preflight_receipt.json")
    batch = read(OUT / "profiling/batch_profile.json")
    if (
        feature.get("status") != "PASS_TARGET_FREE_FEATURE_PREFLIGHT"
        or feature.get("observations") != 3443
        or feature.get("groups") != 203
    ):
        raise RuntimeError("feature_preflight_not_pass")
    if (
        split.get("status") != "PASS_TARGET_BLIND_SPLITS_PREPARED"
        or split.get("target_values_loaded_or_used") is not False
    ):
        raise RuntimeError("split_preflight_not_target_blind_pass")
    if (
        split_validation.get("status") != "PASS_TARGET_FREE_SPLIT_VALIDATION"
        or split_validation.get("targets_read") is not False
    ):
        raise RuntimeError("split_independent_validation_not_pass")
    if any(
        split_validation.get("evaluations", {}).get(ev, {}).get("unique_rows_tested_once") != 3443
        for ev in ("E1", "E2", "E3")
    ):
        raise RuntimeError("split_oof_coverage_incomplete")
    if (
        classic.get("status") != "PASS_CLASSICAL_INPUT_AND_SELECTOR_PREFLIGHT"
        or classic.get("no_fit_or_score") is not True
    ):
        raise RuntimeError("classical_preflight_not_pass")
    if batch.get("status") != "PASS_TARGET_FREE_BATCH_PROFILE" or batch.get("chosen_common_effective_batch") != 32:
        raise RuntimeError("batch_profile_not_pass_or_choice_changed")
    if batch.get("trainer_source_sha256") != sha(ROOT / "experiments/qpu/train_neural.py") or batch.get(
        "core_source_sha256"
    ) != sha(ROOT / "experiments/qpu/core.py"):
        raise RuntimeError("batch_profile_code_identity_stale")
    code = [
        "experiments/qpu/core.py",
        "experiments/qpu/prepare_cohort.py",
        "experiments/qpu/prepare_splits.py",
        "experiments/qpu/audit_classical_inputs.py",
        "experiments/qpu/profile_batch.py",
        "experiments/qpu/train_neural.py",
        "experiments/qpu/run_classical.py",
        "experiments/qpu/lock_protocol.py",
        "experiments/qpu/validate_splits.py",
        "experiments/qpu/validate_first_cell.py",
        "benchmark26/shots.py",
        "benchmark26/mali_features.py",
        "benchmark26/mali_training.py",
        "benchmark26/mali_batching.py",
        "benchmark26/gpu_lease.py",
        "benchmark26/qonductor_regression.py",
        "benchmark26/qonductor_models.py",
        "benchmark26/qonductor_materialize.py",
    ]
    inputs = [
        "docs/final_completion/qpu_workflow_mean_expansion_v1/PLAN.md",
        "docs/final_completion/qpu_workflow_mean_expansion_v1/proposed_observations.csv",
        "docs/final_completion/qpu_workflow_mean_expansion_v1/inventory_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/PROTOCOL.md",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/PROTOCOL_AMENDMENT_01.md",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/execution_lock.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/targets.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/native_baseline_features.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/classical_feature_provenance.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/classical_input_audit.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/qpack_feature_source_map.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_collision_audit.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_preflight_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E1_outer.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E1_inner.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E2_outer.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E2_inner.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E3_outer.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E3_inner.csv",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/split_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/split_validation_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/profiling/batch_profile.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/classical/preflight_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/training/canary_receipt.json",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/training/canary/fold_0/G1/seed_42/checkpoint.pt",
        "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/training/canary/fold_0/G1/seed_42/epoch_history.csv",
        "artifacts/real_qpu/qonductor_native_features/feature_rows.csv",
        "artifacts/real_qpu/qualified_logical_panel/candidates.json",
        "benchmark_v1/protocol/qonductor_native_features.json",
        "benchmark_v1/protocol/qualified_logical_benchmark.json",
    ]
    for p in code + inputs:
        if not (recorded(p)).is_file():
            raise RuntimeError(f"missing_pin:{p}")
    graph_hashes = {}
    for r in __import__("csv").DictReader((OUT / "cohort/feature_ledger.csv").open(newline="", encoding="utf-8")):
        rel = r["graph_artifact"]
        path = recorded(rel)
        digest = sha(path)
        if digest != r["graph_artifact_sha256"]:
            raise RuntimeError(f"graph_hash_mismatch:{rel}")
        graph_hashes[rel] = digest
    if len(graph_hashes) == 0:
        raise RuntimeError("graph_hash_inventory_empty")
    dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "study": "postfreeze_qpu_workflow_mean_expansion_v1",
        "status": "LOCKED_PRE_OUTCOME",
        "locked_at_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": head,
        "branch": branch,
        "preexisting_and_current_dirty_paths_at_lock": dirty,
        "protocol_markdown_sha256": sha(OUT / "PROTOCOL.md"),
        "protocol_amendment_sha256": sha(amendment),
        "supersedes_lock_sha256": sha(previous_lock),
        "code_hashes": {p: sha(recorded(p)) for p in code},
        "input_hashes": {p: sha(recorded(p)) for p in inputs},
        "unique_graph_hashes": graph_hashes,
        "cohort": {
            "rows": 3443,
            "groups": 203,
            "sources": {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 3058, "qpack_mcp": 45},
            "targets": "source-specific observed runtime seconds; QPack one arithmetic-mean label per reconciled workflow",
        },
        "evaluations": {
            "E1": {"outer_folds": 5, "group_disjoint": True},
            "E2": {"outer_folds": 5, "source_stratified": True, "group_overlap_intentional": True},
            "E3": {"outer_cases": 10, "heldout_backend_label_only": True, "known_circuit_new_context_allowed": True},
        },
        "neural": {
            "treatments": ["M0", "M1", "G0", "G1"],
            "seeds": [42, 1234, 31415],
            "planned_cells": 240,
            "epochs_max": 500,
            "selected_checkpoint": "earliest minimum inner-validation raw-seconds MSE",
            "loss": "unweighted raw-seconds MSE",
            "optimizer": {"name": "Adam", "lr": 0.0005, "weight_decay": 0.0001},
            "dtype": "float32",
            "amp": False,
            "effective_batch": 32,
            "graph_node_budget": 250000,
            "gpu_budget_gib": 12,
            "gpu_reserve_gib": 4,
            "cpu_rss_limit_gib": 12,
            "shot_standardization": "fit inner-fit only; raw shots; sample std ddof=1",
            "seed_reduction": "median three held-out predictions per row",
        },
        "classical": {
            "families": classic["families"],
            "candidate_counts": classic["candidate_counts"],
            "selection_metric": "mean inner raw-seconds R2",
            "input_order": classic["feature_names"],
            "input_route_is_source_provenance_specific": True,
        },
        "baseline": {"C0": "median target from outer training observations only"},
        "primary_metric": "group-balanced MAE seconds",
        "bootstrap": {
            "replicates": 10000,
            "seed": 42,
            "unit": "structural group",
            "E1_E2_CI_levels": [0.95, 0.99375],
            "E3_CI_levels": [0.95, 0.99875],
        },
        "test_labels": {
            "not_used_for_split_or_preprocessing": True,
            "neural_function_receives_outer_train_targets_only": True,
            "classical_selector_receives_outer_train_targets_only": True,
            "join_test_targets_after_prediction_export": True,
        },
        "restrictions": {
            "no_qpu_jobs": True,
            "no_simulator_runs": True,
            "no_frozen_artifact_changes": True,
            "no_commit_or_push": True,
        },
        "batch_profile_sha256": sha(OUT / "profiling/batch_profile.json"),
        "feature_preflight_sha256": sha(OUT / "cohort/feature_preflight_receipt.json"),
        "split_receipt_sha256": sha(OUT / "splits/split_receipt.json"),
        "split_validation_sha256": sha(OUT / "splits/split_validation_receipt.json"),
        "classical_preflight_sha256": sha(OUT / "classical/preflight_receipt.json"),
    }
    lock_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "study": payload["study"],
                "lock_sha256": sha(lock_path),
                "protocol_sha256": payload["protocol_markdown_sha256"],
                "code_files": len(code),
                "input_files": len(inputs),
                "unique_graph_files": len(graph_hashes),
                "dirty_paths_at_lock": len(dirty),
                "head": head,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
