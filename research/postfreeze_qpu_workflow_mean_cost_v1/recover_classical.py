#!/usr/bin/env python3
"""Reconstruct exactly the 20 already-selected R0 pipelines and verify parity."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from common import ART, ROOT, SRC, atomic_json, load_native, rows, sha, stable, verify_plan_pins

sys.path.insert(0, str(ROOT / "research/postfreeze_qpu_workflow_mean_expansion_v1"))
from core import load_context, load_targets, parts_for

sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
from run_strict_logical_qonductor import source_grids_and_models


def build_amendment() -> dict:
    native_path = SRC / "cohort/native_baseline_features.csv"
    cases = []
    for evaluation in ("E1", "E2", "E3"):
        context, _, _ = load_context(evaluation, include_targets=False)
        for fold in range(context["case_count"]):
            folder = SRC / "classical" / evaluation / f"fold_{fold}"
            result = json.loads((folder / "regression_manifest.json").read_text())
            parts = parts_for(context, fold)
            if result["selected_family"] not in {"Extra Trees", "Histogram Gradient Boosting", "Polynomial Regression"}:
                raise RuntimeError(f"unexpected_selected_family:{evaluation}:{fold}")
            cases.append({
                "evaluation": evaluation, "fold": fold,
                "heldout_backend": context["cases"][fold] if evaluation == "E3" else "",
                "selected_family": result["selected_family"],
                "candidate_index": int(result["selected_candidate_index"]),
                "params_json": result["family_selections"][result["selected_family"]]["params_json"],
                "train_ids_sha256": stable(sorted(parts["train"])),
                "test_ids_sha256": stable(sorted(parts["test"])),
                "saved_prediction_sha256": sha(folder / "regression_predictions.csv"),
                "train_rows": len(parts["train"]), "test_rows": len(parts["test"]),
            })
    return {
        "status": "LOCKED_FIXED_WINNER_STATE_RECONSTRUCTION",
        "purpose": "one fitted serving state per previously selected outer case; no candidate search or reselection",
        "authorization_basis": "user instructed to accept and implement the detailed plan; this amendment limits implementation to the fixed-winner recovery described in that plan",
        "source_study_protocol_sha256": sha(SRC / "PROTOCOL.md"),
        "source_study_lock_sha256": sha(SRC / "execution_lock_v2.json"),
        "source_code_sha256": {
            "research/postfreeze_qpu_workflow_mean_expansion_v1/run_classical.py": sha(ROOT / "research/postfreeze_qpu_workflow_mean_expansion_v1/run_classical.py"),
            "benchmark_v1/scripts/run_qonductor_grouped_reproduction.py": sha(ROOT / "benchmark_v1/scripts/run_qonductor_grouped_reproduction.py"),
            "research/postfreeze_qpu_workflow_mean_cost_v1/recover_classical.py": sha(Path(__file__)),
        },
        "input_hashes": {
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/native_baseline_features.csv": sha(native_path),
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/targets.csv": sha(SRC / "cohort/targets.csv"),
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E1_outer.csv": sha(SRC / "splits/E1_outer.csv"),
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E2_outer.csv": sha(SRC / "splits/E2_outer.csv"),
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E3_outer.csv": sha(SRC / "splits/E3_outer.csv"),
        },
        "feature_order": ["swap", "depth", "num_qubits", "shots", "circuit_count"],
        "input_dtype": "numpy.float64",
        "train_row_order": "sorted(canonical_observation_id)",
        "target": "raw target_seconds, outer-training IDs only",
        "estimator": "one candidate already selected in saved inner-validation manifest; constructor and parameters unchanged",
        "random_state": 0,
        "forest_n_jobs": 1,
        "fit_threadpool_limit": 1,
        "serialization": "joblib, then fresh-process reload and complete saved test-prediction replay",
        "parity_rule": "max absolute replay difference <= 1e-9 seconds; no post-fit tolerance changes",
        "no_neural_training": True,
        "no_hpo_or_candidate_selection": True,
        "test_labels_read_for_fit": False,
        "plan_pins": verify_plan_pins().get("evidence_hashes", {}),
        "cases": cases,
    }


def fit_cases() -> None:
    import joblib
    import sklearn
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits

    native = load_native()
    templates, _, _ = source_grids_and_models()
    pin = json.loads((ART / "recovery_amendment_01.json").read_text())
    for case in pin["cases"]:
        ev, fold = case["evaluation"], int(case["fold"])
        output = ART / "recovered_classical" / ev / f"fold_{fold}"
        output.mkdir(parents=True, exist_ok=True)
        state_path, receipt_path = output / "estimator.joblib", output / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("amendment_sha256") != sha(ART / "recovery_amendment_01.json") or sha(state_path) != receipt["state_sha256"]:
                raise RuntimeError(f"recovery_resume_identity_mismatch:{ev}:{fold}")
            continue
        context, _, _ = load_context(ev, include_targets=False)
        parts = parts_for(context, fold)
        train = sorted(parts["train"])
        if stable(train) != case["train_ids_sha256"] or stable(sorted(parts["test"])) != case["test_ids_sha256"]:
            raise RuntimeError(f"recovery_split_identity_changed:{ev}:{fold}")
        train_targets = load_targets(train)
        x = np.stack([native[cid] for cid in train]).astype(np.float64, copy=False)
        y = np.asarray([train_targets[cid] for cid in train], dtype=np.float64)
        estimator = clone(templates[case["selected_family"]]).set_params(**json.loads(case["params_json"]))
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            estimator.fit(x, y)
        fit_seconds = time.perf_counter() - start
        joblib.dump(estimator, state_path, compress=3)
        loaded = joblib.load(state_path)
        test_ids = sorted(parts["test"])
        test_x = np.stack([native[cid] for cid in test_ids]).astype(np.float64, copy=False)
        actual = np.asarray(loaded.predict(test_x), dtype=np.float64)
        saved_rows = rows(SRC / "classical" / ev / f"fold_{fold}" / "regression_predictions.csv")
        saved = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in saved_rows}
        expected = np.asarray([saved[cid] for cid in test_ids], dtype=np.float64)
        if not np.isfinite(actual).all() or actual.shape != expected.shape:
            raise RuntimeError(f"recovery_prediction_invalid:{ev}:{fold}")
        max_abs = float(np.max(np.abs(actual - expected)))
        if max_abs > 1e-9:
            raise RuntimeError(f"recovery_prediction_parity_failed:{ev}:{fold}:{max_abs:.17g}")
        receipt = {
            "status": "PASS_PREDICTION_PARITY_VALIDATED_RECONSTRUCTION",
            "evaluation": ev, "fold": fold, "heldout_backend": case["heldout_backend"],
            "selected_family": case["selected_family"], "candidate_index": case["candidate_index"],
            "params_json": case["params_json"], "train_rows": len(train), "test_rows": len(test_ids),
            "train_ids_sha256": stable(train), "test_ids_sha256": stable(test_ids),
            "native_features_sha256": sha(SRC / "cohort/native_baseline_features.csv"),
            "saved_predictions_sha256": case["saved_prediction_sha256"],
            "state_sha256": sha(state_path), "state_bytes": state_path.stat().st_size,
            "max_abs_replay_difference_seconds": max_abs, "replay_tolerance_seconds": 1e-9,
            "fit_seconds_maintenance_only": fit_seconds,
            "python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "random_state": 0, "threadpool_limit": 1,
            "test_target_values_read": False,
            "caveat": "prediction parity on saved held-out inputs does not prove byte identity or behavior on every unseen input",
            "amendment_sha256": sha(ART / "recovery_amendment_01.json"),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps({"evaluation": ev, "fold": fold, "family": case["selected_family"],
                          "fit_seconds": fit_seconds, "max_abs_replay_difference_seconds": max_abs}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-amendment", action="store_true")
    ap.add_argument("--fit-fixed-winners", action="store_true")
    args = ap.parse_args()
    path = ART / "recovery_amendment_01.json"
    if args.lock_amendment:
        payload = build_amendment()
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise RuntimeError("existing_recovery_amendment_differs_refuse_overwrite")
        else:
            atomic_json(path, payload)
        print(json.dumps({"status": payload["status"], "cases": len(payload["cases"]), "sha256": sha(path)}, indent=2))
        return
    if args.fit_fixed_winners:
        if not path.is_file():
            raise RuntimeError("recovery_amendment_must_be_locked_before_fixed_fit")
        pin = json.loads(path.read_text())
        if pin.get("status") != "LOCKED_FIXED_WINNER_STATE_RECONSTRUCTION" or len(pin.get("cases", [])) != 20:
            raise RuntimeError("invalid_locked_recovery_amendment")
        fit_cases()
        return
    ap.error("choose --lock-amendment or --fit-fixed-winners")


if __name__ == "__main__":
    main()
