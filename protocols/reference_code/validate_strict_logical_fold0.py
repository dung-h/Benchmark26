#!/usr/bin/env python3
"""Independently check the six strict Ma-Li CUDA fold-0 cells."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/strict_logical_panel"
OUTPUT = PANEL / "mali"
PARENT = ROOT / "benchmark_v1/protocol/strict_logical_benchmark.json"
CONTRACT = ROOT / "benchmark_v1/protocol/strict_method_completion.json"
AUTH = PANEL / "execution_record.json"
METHODS = ("strict_mali_graph", "strict_mali_global_mlp")
SEEDS = (42, 1234, 31415)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
    import run_mali_full_features as base
    base.atomic_json(path, value)


def validate() -> dict[str, Any]:
    auth = json.loads(AUTH.read_text())
    binds = auth["bound_inputs"]
    expected_binds = {"parent_protocol_sha256": sha(PARENT),
                      "completion_protocol_sha256": sha(CONTRACT),
                      "panel_manifest_sha256": sha(PANEL / "manifest.json")}
    if any(binds.get(k) != v for k,v in expected_binds.items()):
        raise RuntimeError("fold0_execution_record_pin_mismatch")
    panel = read_csv(PANEL / "logical_features.csv")
    outer = {r["canonical_observation_id"]: int(r["outer_fold"])
             for r in read_csv(PANEL / "outer_splits.csv")}
    if len(panel) != 340 or len(outer) != 340:
        raise RuntimeError("fold0_panel_cardinality_mismatch")
    expected_test = sorted(i for i,f in outer.items() if f == 0)
    expected_test_set = set(expected_test)
    training_gate_path = OUTPUT / "fold_0_training_gate.json"
    if not training_gate_path.is_file():
        raise RuntimeError("fold0_training_gate_missing")
    training_gate = json.loads(training_gate_path.read_text())
    if training_gate.get("status") != "awaiting_independent_technical_validation":
        raise RuntimeError("fold0_training_gate_wrong_status")
    if (training_gate.get("execution_record_sha256") != sha(AUTH)
            or training_gate.get("parent_protocol_sha256") != sha(PARENT)
            or training_gate.get("completion_protocol_sha256") != sha(CONTRACT)
            or training_gate.get("profile_sha256") != sha(OUTPUT / "cuda_profile.json")):
        raise RuntimeError("fold0_training_gate_hash_mismatch")
    profile = json.loads((OUTPUT / "cuda_profile.json").read_text())
    for check in ("heldout_perturbation_invariance", "checkpoint_rng_resume_equivalence"):
        if profile.get(check) != "PASS":
            raise RuntimeError(f"fold0_canary_failed:{check}")
    if profile.get("microbatch_gradient_adam_parity", {}).get("status") != "PASS":
        raise RuntimeError("fold0_microbatch_parity_not_pass")
    if profile.get("largest_intact_graph_canary", {}).get("status") != "PASS":
        raise RuntimeError("fold0_largest_graph_canary_not_pass")
    cells = []
    for method in METHODS:
        for seed in SEEDS:
            folder = OUTPUT / "fold_0" / method / f"seed_{seed}"
            manifest_path, prediction_path, checkpoint_path = folder / "manifest.json", folder / "predictions.csv", folder / "checkpoint.pt"
            if not all(p.is_file() for p in (manifest_path, prediction_path, checkpoint_path)):
                raise RuntimeError(f"fold0_cell_artifact_missing:{method}:{seed}")
            manifest = json.loads(manifest_path.read_text())
            identity = manifest.get("identity", {})
            if (manifest.get("status") != "complete" or manifest.get("epochs") != 500
                    or not 0 <= int(manifest.get("best_epoch", -1)) < 500):
                raise RuntimeError(f"fold0_cell_manifest_incomplete:{method}:{seed}")
            if (identity.get("method") != method or int(identity.get("outer_fold", -1)) != 0
                    or int(identity.get("seed", -1)) != seed
                    or identity.get("execution_record_sha256") != sha(AUTH)
                    or identity.get("parent_protocol_sha256") != sha(PARENT)
                    or identity.get("completion_protocol_sha256") != sha(CONTRACT)
                    or identity.get("profile_sha256") != sha(OUTPUT / "cuda_profile.json")):
                raise RuntimeError(f"fold0_cell_identity_mismatch:{method}:{seed}")
            for name, expected in manifest.get("output_hashes", {}).items():
                if sha(folder / name) != expected:
                    raise RuntimeError(f"fold0_cell_output_hash_mismatch:{method}:{seed}:{name}")
            predictions = read_csv(prediction_path)
            ids = {r["canonical_observation_id"] for r in predictions}
            if len(predictions) != len(expected_test) or ids != expected_test_set:
                raise RuntimeError(f"fold0_prediction_test_ids_mismatch:{method}:{seed}")
            for row in predictions:
                value = float(row["predicted_seconds"])
                if not math.isfinite(value) or row["status"] != "predicted":
                    raise RuntimeError(f"fold0_invalid_prediction:{method}:{seed}:{row['canonical_observation_id']}")
            cells.append({"method": method, "seed": seed, "status": "PASS",
                "test_rows": len(predictions), "test_ids_sha256": hashlib.sha256("\n".join(expected_test).encode()).hexdigest(),
                "manifest_sha256": sha(manifest_path), "predictions_sha256": sha(prediction_path),
                "checkpoint_sha256": sha(checkpoint_path), "best_epoch": manifest["best_epoch"],
                "elapsed_seconds": manifest["elapsed_seconds"]})
    # Independently check the six CPU estimator families on the exact same
    # fold-0 train/validation/test IDs before any later fold can be opened.
    import numpy as np
    import collections
    qonductor_dir = PANEL / "qonductor"
    candidate_path = qonductor_dir / "frozen_candidates.json"
    score_path = qonductor_dir / "candidate_inner_scores.csv"
    selection_path = qonductor_dir / "fold_selection.json"
    prediction_path = qonductor_dir / "predictions.csv"
    if not all(p.is_file() for p in (candidate_path, score_path, selection_path, prediction_path)):
        raise RuntimeError("fold0_qonductor_artifact_missing")
    candidates = json.loads(candidate_path.read_text())
    score_rows = read_csv(score_path)
    fold0_scores = [r for r in score_rows if int(r["outer_fold"]) == 0]
    expected_families = ("Extra Trees", "Random Forest", "Gradient Boosting", "AdaBoost",
                         "Histogram Gradient Boosting", "Polynomial Regression")
    expected_score_count = sum(len(candidates["candidates"][family]) for family in expected_families) * 4
    score_keys = [(r["model_family"], int(r["candidate_index"]), int(r["inner_fold"])) for r in fold0_scores]
    if len(fold0_scores) != expected_score_count or len(set(score_keys)) != expected_score_count:
        raise RuntimeError("fold0_qonductor_inner_score_count_or_identity_mismatch")
    selected_expected = {}
    for family in expected_families:
        family_scores = [r for r in fold0_scores if r["model_family"] == family]
        expected_candidates = len(candidates["candidates"][family])
        if len(family_scores) != expected_candidates * 4:
            raise RuntimeError(f"fold0_qonductor_candidate_score_count_mismatch:{family}")
        ranked = []
        for candidate_index in range(expected_candidates):
            parts = sorted((r for r in family_scores if int(r["candidate_index"]) == candidate_index),
                           key=lambda r: int(r["inner_fold"]))
            if len(parts) != 4 or [int(r["inner_fold"]) for r in parts] != [0, 1, 2, 3]:
                raise RuntimeError(f"fold0_qonductor_inner_fold_missing:{family}:{candidate_index}")
            if any(r["status"] != "scored" or not math.isfinite(float(r["r2_seconds"])) for r in parts):
                raise RuntimeError(f"fold0_qonductor_inner_score_invalid:{family}:{candidate_index}")
            mean_score = float(np.mean([float(r["r2_seconds"]) for r in parts]))
            ranked.append((mean_score, candidate_index, [float(r["r2_seconds"]) for r in parts]))
        best = max(ranked, key=lambda item: (item[0], -item[1]))
        selected_expected[family] = best
    selection = json.loads(selection_path.read_text())
    selected_actual = selection.get("folds", {}).get("0", {}).get("selected", {})
    for family, best in selected_expected.items():
        actual = selected_actual.get(family, {})
        if (int(actual.get("candidate_index", -1)) != best[1]
                or not math.isclose(float(actual.get("mean_inner_r2_seconds", float("nan"))), best[0], rel_tol=1e-12, abs_tol=1e-12)
                or len(actual.get("inner_r2_seconds", [])) != 4):
            raise RuntimeError(f"fold0_qonductor_inner_selection_mismatch:{family}")
    expected_selector = max(expected_families, key=lambda name: (selected_expected[name][0], -expected_families.index(name)))
    if selection.get("folds", {}).get("0", {}).get("selector_family") != expected_selector:
        raise RuntimeError("fold0_qonductor_selector_not_inner_selected")
    q_predictions = read_csv(prediction_path)
    fold0_predictions = [r for r in q_predictions if int(r["outer_fold"]) == 0]
    expected_methods = {"strict_qonductor_" + name.lower().replace(" ", "_") for name in expected_families}
    expected_methods.add("strict_qonductor_six_family_budgeted_selector")
    method_counts = collections.Counter(r["method_id"] for r in fold0_predictions)
    if len(fold0_predictions) != 7 * len(expected_test) or set(method_counts) != expected_methods:
        raise RuntimeError("fold0_qonductor_prediction_method_or_count_mismatch")
    q_ids_by_method = {}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in panel}
    for method in expected_methods:
        method_rows = [r for r in fold0_predictions if r["method_id"] == method]
        ids = {r["canonical_observation_id"] for r in method_rows}
        if len(method_rows) != len(expected_test) or ids != expected_test_set:
            raise RuntimeError(f"fold0_qonductor_test_id_mismatch:{method}")
        for row in method_rows:
            prediction = float(row["predicted_seconds"])
            actual = float(row["actual_seconds"])
            if row["status"] != "predicted" or not math.isfinite(prediction) or not math.isclose(actual, targets[row["canonical_observation_id"]], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"fold0_qonductor_prediction_invalid:{method}:{row['canonical_observation_id']}")
        q_ids_by_method[method] = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if len(set(q_ids_by_method.values())) != 1:
        raise RuntimeError("fold0_qonductor_methods_not_paired")
    qonductor_gate = {"status": "PASS", "family_count": len(expected_families),
        "selector": "strict_qonductor_six_family_budgeted_selector", "selector_family": expected_selector,
        "candidate_inner_score_rows": len(fold0_scores), "prediction_rows": len(fold0_predictions),
        "test_rows_per_method": len(expected_test), "test_ids_sha256": q_ids_by_method[next(iter(expected_methods))],
        "candidate_manifest_sha256": sha(candidate_path), "candidate_inner_scores_sha256": sha(score_path),
        "fold_selection_sha256": sha(selection_path), "predictions_sha256": sha(prediction_path),
        "all_inner_scores_finite": True, "all_six_selections_recomputed": True,
        "selector_uses_inner_validation_only": True, "all_methods_use_identical_fold0_test_ids": True}
    fold_gate = {"artifact_id": "strict-logical-mali-fold0-independent-gate",
        "status": "PASS", "outer_fold": 0, "cell_count": 6, "test_rows": len(expected_test),
        "cells": cells, "qonductor_fold0": qonductor_gate, "training_gate_sha256": sha(training_gate_path),
        "cuda_profile_sha256": sha(OUTPUT / "cuda_profile.json"),
        "execution_record_sha256": sha(AUTH), "parent_protocol_sha256": sha(PARENT),
        "completion_protocol_sha256": sha(CONTRACT), "validator_sha256": sha(Path(__file__)),
        "technical_checks": {"all_six_cells_complete": True, "500_epochs_each": True,
            "frozen_fold0_test_ids_exact": True, "predictions_finite": True,
            "training_and_validation_partitions_pinned": True,
            "cuda_profile_parity_resource_and_resume_canaries_pass": True}}
    target = OUTPUT / "fold_0_gate_receipt.json"
    if target.exists():
        if json.loads(target.read_text()) != fold_gate:
            raise RuntimeError("existing_fold0_gate_receipt_differs_refuse_overwrite")
    else:
        write_json(target, fold_gate)
    return fold_gate


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True), flush=True)
