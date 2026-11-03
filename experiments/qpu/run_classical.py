#!/usr/bin/env python3
"""Run the inherited six-family Qonductor-style selector and train-median baseline."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS
from pathlib import Path
import argparse, concurrent.futures, json, math, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from core import (
    OUT,
    RUN,
    atomic_csv,
    atomic_json,
    execution_identity,
    load_context,
    load_targets,
    parts_for,
    rows,
    sha,
    stable,
)
from benchmark26.qonductor_regression import fit_one_candidate, source_grids_and_models

NATIVE = OUT / "cohort/native_baseline_features.csv"
FEATURES = ("swap", "depth", "num_qubits", "shots", "circuit_count")


def load_matrix(context):
    records = {r["canonical_observation_id"]: r for r in rows(NATIVE)}
    if set(records) != context["all_ids"]:
        raise RuntimeError("classical_input_identity_mismatch")
    matrix = {}
    for cid, r in records.items():
        x = np.asarray([float(r[k]) for k in FEATURES], dtype=np.float64)
        if x.shape != (5,) or not np.isfinite(x).all():
            raise RuntimeError(f"invalid_classical_input:{cid}")
        matrix[cid] = x
    return matrix


def run_case(ev, fold, context, matrix, lock_sha, templates, candidates, families):
    parts = parts_for(context, fold)
    folder = RUN / "classical" / ev / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    train = parts["train"]
    test = parts["test"]
    train_targets = load_targets(train)
    if set(train_targets) & set(test):
        raise RuntimeError("test_target_entered_classical_fit")
    train_order = sorted(train)
    idx = {cid: i for i, cid in enumerate(train_order)}
    xtrain = np.stack([matrix[c] for c in train_order])
    y = np.asarray([train_targets[c] for c in train_order], dtype=np.float64)
    x = {"matrix": xtrain, "id_to_index": idx}
    splits = parts["inner_cv"]
    identity = {
        "evaluation": ev,
        "fold": fold,
        "heldout_backend": context["cases"][fold] if ev == "E3" else "",
        "train_ids_sha256": stable(train_order),
        "fit_ids_sha256": stable(parts["fit"]),
        "validation_ids_sha256": stable(parts["validation"]),
        "test_ids_sha256": stable(test),
        "feature_file_sha256": sha(NATIVE),
        "feature_order": list(FEATURES),
        "protocol_lock_sha256": lock_sha,
        "candidate_counts": {k: len(candidates[k]) for k in families},
    }
    manifest = folder / "regression_manifest.json"
    predpath = folder / "regression_predictions.csv"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved.get("identity") != identity or sha(predpath) != saved.get("predictions_sha256"):
            raise RuntimeError(f"classical_resume_identity_mismatch:{ev}:{fold}")
        regression = saved
    else:
        import concurrent.futures
        from threadpoolctl import threadpool_limits

        selections = {}
        details = {}
        for family in families:
            sp = folder / f"{family.lower().replace(' ', '_')}_candidate_scores.json"
            sid = {**identity, "family": family}
            if sp.exists():
                item = json.loads(sp.read_text())
                if item.get("identity") != sid:
                    raise RuntimeError(f"candidate_resume_mismatch:{ev}:{fold}:{family}")
                scores = item["scores"]
            else:
                start = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    fs = [
                        pool.submit(fit_one_candidate, family, candidate, x, y, splits, templates)
                        for candidate in candidates[family]
                    ]
                    scores = [f.result() for f in fs]
                atomic_json(sp, {"identity": sid, "scores": scores, "elapsed_seconds": time.monotonic() - start})
            good = [r for r in scores if r["status"] == "scored" and math.isfinite(float(r["mean_inner_r2"]))]
            if not good:
                raise RuntimeError(f"no_classical_candidates_scored:{ev}:{fold}:{family}")
            best = max(good, key=lambda r: (float(r["mean_inner_r2"]), -int(r["candidate_index"])))
            selections[family] = best
            details[family] = {
                "candidate_index": best["candidate_index"],
                "mean_inner_raw_r2": best["mean_inner_r2"],
                "params_json": best["params_json"],
                "valid_candidates": len(good),
                "candidate_count": len(scores),
            }
        winner = max(families, key=lambda family: (float(selections[family]["mean_inner_r2"]), -families.index(family)))
        best = selections[winner]
        from sklearn.base import clone

        estimator = clone(templates[winner]).set_params(**json.loads(best["params_json"]))
        with threadpool_limits(limits=1):
            estimator.fit(xtrain, y)
            pred = np.asarray(estimator.predict(np.stack([matrix[c] for c in test])), dtype=np.float64)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"classical_nonfinite:{ev}:{fold}")
        out = [
            {
                "canonical_observation_id": cid,
                "fold": fold,
                "evaluation": ev,
                "treatment": "R0",
                "selected_family": winner,
                "predicted_seconds": float(v),
            }
            for cid, v in zip(test, pred)
        ]
        atomic_csv(predpath, out)
        regression = {
            "status": "complete",
            "identity": identity,
            "selected_family": winner,
            "selected_candidate_index": best["candidate_index"],
            "selection_inner_metric": "mean raw-seconds R2 across 4 inner folds",
            "family_selections": details,
            "fit_rows": len(parts["fit"]),
            "validation_rows": len(parts["validation"]),
            "outer_train_rows": len(train),
            "test_rows": len(test),
            "test_labels_passed_to_selector_or_fit": False,
            "predictions_sha256": sha(predpath),
        }
        atomic_json(manifest, regression)
    # Training-median constant baseline is calculated from outer training labels only.
    constant_path = folder / "constant_predictions.csv"
    constant_manifest = folder / "constant_manifest.json"
    constant = float(np.median(list(train_targets.values())))
    const_identity = {
        "evaluation": ev,
        "fold": fold,
        "train_ids_sha256": stable(train_order),
        "test_ids_sha256": stable(test),
        "train_targets_sha256": stable(train_targets),
        "protocol_lock_sha256": lock_sha,
    }
    const_rows = [
        {
            "canonical_observation_id": cid,
            "fold": fold,
            "evaluation": ev,
            "treatment": "C0",
            "predicted_seconds": constant,
        }
        for cid in test
    ]
    if constant_manifest.exists():
        saved = json.loads(constant_manifest.read_text())
        if saved.get("identity") != const_identity or sha(constant_path) != saved.get("predictions_sha256"):
            raise RuntimeError(f"constant_resume_mismatch:{ev}:{fold}")
    else:
        atomic_csv(constant_path, const_rows)
        atomic_json(
            constant_manifest,
            {
                "status": "complete",
                "identity": const_identity,
                "outer_train_median_seconds": constant,
                "fit_rows": len(train),
                "test_rows": len(test),
                "test_labels_used": False,
                "predictions_sha256": sha(constant_path),
            },
        )
    return {
        "evaluation": ev,
        "fold": fold,
        "regression": regression["status"],
        "selected_family": regression["selected_family"],
        "regression_test_rows": regression["test_rows"],
        "constant_test_rows": len(test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preflight", "train"], required=True)
    ap.add_argument("--evaluation", choices=["E1", "E2", "E3"])
    args = ap.parse_args()
    if args.stage == "preflight":
        for ev in ("E1", "E2", "E3"):
            ctx, _, rec = load_context(ev, include_targets=False)
            matrix = load_matrix(ctx)
            if len(matrix) != 3443:
                raise RuntimeError("classical_preflight_row_mismatch")
        templates, grids, _ = source_grids_and_models()
        candidates_doc = json.loads((ARTIFACTS / "real_qpu/qualified_logical_panel/candidates.json").read_text())
        receipt = {
            "status": "PASS_CLASSICAL_INPUT_AND_SELECTOR_PREFLIGHT",
            "rows": 3443,
            "feature_names": list(FEATURES),
            "families": list(templates),
            "candidate_counts": {k: len(v) for k, v in candidates_doc["candidates"].items()},
            "feature_provenance_sha256": sha(OUT / "cohort/classical_feature_provenance.csv"),
            "no_target_values_read": True,
            "no_fit_or_score": True,
            "selector_source_revision": "pinned Qonductor-SC25 adapter already audited in strict logical benchmark",
        }
        atomic_json(RUN / "classical/preflight_receipt.json", receipt)
        print(json.dumps(receipt, indent=2))
        return
    _lock, lock_sha = execution_identity()
    evals = [args.evaluation] if args.evaluation else ["E1", "E2", "E3"]
    templates, _, _ = source_grids_and_models()
    doc = json.loads((ARTIFACTS / "real_qpu/qualified_logical_panel/candidates.json").read_text())
    candidates = doc["candidates"]
    families = list(templates)
    if set(candidates) != set(families):
        raise RuntimeError("classical_family_manifest_mismatch")
    completed = []
    for ev in evals:
        context, _targets, _ = load_context(ev, include_targets=False)
        matrix = load_matrix(context)
        for fold in range(context["case_count"]):
            result = run_case(ev, fold, context, matrix, lock_sha, templates, candidates, families)
            completed.append(result)
            atomic_json(
                RUN / "classical/run_progress.json",
                {
                    "status": "running",
                    "completed_cases": len(completed),
                    "planned_cases": 20,
                    "last_case": result,
                    "protocol_lock_sha256": lock_sha,
                },
            )
            print(json.dumps(result), flush=True)
    atomic_json(
        RUN / "classical/completion_receipt.json",
        {
            "status": "COMPLETE_20_OF_20",
            "cases": completed,
            "protocol_lock_sha256": lock_sha,
            "test_labels_passed_to_selector_or_fit": False,
        },
    )


if __name__ == "__main__":
    main()
