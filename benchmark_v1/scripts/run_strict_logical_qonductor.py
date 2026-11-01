#!/usr/bin/env python3
"""Execute the frozen six-family Qonductor-style adaptation on the strict panel.

The input is the 340 Ma-Li observations only. Existing Qonductor feature rows
are reused only after exact logical/compiled-QASM, backend, shot and identity
joins against the frozen strict panel. No labels enter feature construction.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/strict_logical_panel"
CONTRACT = ROOT / "benchmark_v1/protocol/strict_method_completion.json"
PARENT = ROOT / "benchmark_v1/protocol/strict_logical_benchmark.json"
AUTH = PANEL / "execution_authorization.json"
FEATURES = ROOT / "artifacts/real_qpu/qonductor_native_features/feature_rows.csv"
REPRESENTATIONS = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
SOURCE_ROOT = ROOT.parent / "Qonductor-SC25"
OUTPUT = PANEL / "qonductor"
SOURCE_REVISION = "5d1ac8a90cd574a23e7544e1044681641354ff67"
SOURCE_FILE = "src/execution_time/regression_estimator.py"
SOURCE_SHA256 = "d44ab196a56fba444f5cebfe9dee8737ad3b6ab6f39081336a41d5811fdb828a"
FEATURE_ORDER = ["swap", "depth", "num_qubits", "shots", "circuit_count"]
FAMILIES = ["Extra Trees", "Random Forest", "Gradient Boosting", "AdaBoost",
            "Histogram Gradient Boosting", "Polynomial Regression"]
EXPECTED_COUNTS = [64, 64, 64, 48, 64, 3]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, obj: Any) -> None:
    payload = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, payload)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def json_sha(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected_json_object:{path}")
    return value


def check_authority() -> dict[str, Any]:
    auth = json_sha(AUTH)
    expected = {
        "parent_protocol_sha256": sha_file(PARENT),
        "completion_protocol_sha256": sha_file(CONTRACT),
        "panel_manifest_sha256": sha_file(PANEL / "manifest.json"),
    }
    bound = auth.get("bound_inputs", {})
    for key, value in expected.items():
        if bound.get(key) != value:
            raise RuntimeError(f"execution_authorization_pin_mismatch:{key}")
    if "C1 strict-context adapters and frozen candidate configurations" not in auth.get("scope", []):
        raise RuntimeError("execution_scope_missing_c1_c6")
    return auth


def load_panel() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    check_authority()
    panel = read_csv(PANEL / "logical_features.csv")
    features = read_csv(FEATURES)
    representations = read_csv(REPRESENTATIONS)
    outer = read_csv(PANEL / "outer_splits.csv")
    inner = read_csv(PANEL / "inner_splits.csv")
    if len(panel) != 340 or len({r["canonical_observation_id"] for r in panel}) != 340:
        raise RuntimeError("strict_panel_not_exactly_340_unique_rows")
    ids = {r["canonical_observation_id"] for r in panel}
    by_feature = {r["canonical_row_id"]: r for r in features if r["source_id"] == "mali_real_qpu"}
    by_representation = {r["canonical_row_id"]: r for r in representations if r["source_id"] == "mali_real_qpu"}
    if len(by_feature) != 340 or len(by_representation) != 340 or set(by_feature) != ids or set(by_representation) != ids:
        raise RuntimeError("strict_panel_feature_or_representation_join_not_340_to_340")
    for row in panel:
        identity = row["canonical_observation_id"]
        f, rep = by_feature[identity], by_representation[identity]
        checks = {
            "logical_qasm_sha256": (row["logical_qasm_sha256"], rep["source_qasm_sha256"]),
            "compiled_qasm_sha256": (rep["compiled_qasm3_sha256"], f["input_qasm_sha256"]),
            "backend": (row["backend"], rep["backend_canonical"]),
            "shots": (str(int(float(row["shots"]))), str(int(float(f["shots"])))),
        }
        for name, pair in checks.items():
            if pair[0] != pair[1]:
                raise RuntimeError(f"strict_feature_join_mismatch:{identity}:{name}:{pair}")
        if f["availability_status"] != "available":
            raise RuntimeError(f"qonductor_feature_unavailable:{identity}:{f['terminal_reason']}")
        if float(f["circuit_count"]) != 1.0:
            raise RuntimeError(f"not_one_circuit:{identity}")
    outer_by_id = {r["canonical_observation_id"]: r for r in outer}
    if len(outer_by_id) != len(outer) or set(outer_by_id) != ids:
        raise RuntimeError("frozen_outer_split_identity_mismatch")
    if len(inner) != 1360:
        raise RuntimeError("frozen_inner_split_row_count_mismatch")
    group = {r["canonical_observation_id"]: r["group_id"] for r in panel}
    inner_by_fold: dict[int, dict[str, dict[str, str]]] = {}
    for fold in range(5):
        test = {identity for identity, row in outer_by_id.items() if int(row["outer_fold"]) == fold}
        assignments = [r for r in inner if int(r["outer_fold"]) == fold]
        assigned = {r["canonical_observation_id"]: r for r in assignments}
        if len(assigned) != len(assignments) or set(assigned) != ids - test:
            raise RuntimeError(f"inner_outer_train_identity_mismatch:{fold}")
        if {int(r["inner_fold"]) for r in assignments} != {0, 1, 2, 3}:
            raise RuntimeError(f"inner_fold_values_mismatch:{fold}")
        for inner_fold in range(4):
            valid = {i for i, r in assigned.items() if int(r["inner_fold"]) == inner_fold}
            fit = set(assigned) - valid
            if ({group[i] for i in fit} & ({group[i] for i in valid} | {group[i] for i in test})
                    or {group[i] for i in valid} & {group[i] for i in test}):
                raise RuntimeError(f"group_leakage:{fold}:{inner_fold}")
        inner_by_fold[fold] = assigned
    return panel, by_feature, outer_by_id, inner_by_fold


def source_grids_and_models():
    """Use the audited transcription of the pinned upstream constructor/grid."""
    source_blob = __import__("subprocess").run(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{SOURCE_REVISION}:{SOURCE_FILE}"],
        check=True, capture_output=True,
    ).stdout
    if sha_bytes(source_blob) != SOURCE_SHA256:
        raise RuntimeError("pinned_qonductor_source_hash_mismatch")
    scripts = ROOT / "benchmark_v1/scripts"
    sys.path.insert(0, str(scripts))
    from run_qonductor_grouped_reproduction import base_models, source_grids
    return base_models(), source_grids(), source_blob


def freeze_candidates() -> dict[str, Any]:
    check_authority()
    models, grids, source_blob = source_grids_and_models()
    if list(models) != FAMILIES:
        raise RuntimeError("qonductor_model_family_order_drift")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for family, expected_count in zip(FAMILIES, EXPECTED_COUNTS):
        count = min(64, int(np.prod([len(v) for v in grids[family].values()])))
        sampled = list(ParameterSampler(grids[family], n_iter=count, random_state=0))
        if len(sampled) != expected_count:
            raise RuntimeError(f"candidate_count_mismatch:{family}:{len(sampled)}")
        checked = []
        for index, params in enumerate(sampled):
            # This checks estimator API/value compatibility without fitting to
            # any benchmark labels. Raw sampled and effective params are kept.
            estimator = clone(models[family]).set_params(**params)
            validate = getattr(estimator, "_validate_params", None)
            if validate is not None:
                validate()
            checked.append({"candidate_index": index, "raw_params": params,
                            "effective_params": {key: estimator.get_params(deep=True)[key] for key in params}})
        candidates[family] = checked
    payload = {
        "artifact_id": "strict-logical-qonductor-frozen-candidates",
        "status": "PASS",
        "method_class": "budgeted_qonductor_style_unified_adaptation",
        "source_revision": SOURCE_REVISION,
        "source_file_sha256": sha_bytes(source_blob),
        "candidate_source_script_sha256": sha_file(Path(__file__).with_name("run_qonductor_grouped_reproduction.py")),
        "candidate_sampler": "sklearn.model_selection.ParameterSampler; sklearn=" + sklearn.__version__ + "; n_iter=min(64,grid_size); random_state=0",
        "feature_order": FEATURE_ORDER,
        "feature_rows_sha256": sha_file(FEATURES),
        "representation_rows_sha256": sha_file(REPRESENTATIONS),
        "panel_manifest_sha256": sha_file(PANEL / "manifest.json"),
        "candidate_counts": {name: len(candidates[name]) for name in FAMILIES},
        "candidates": candidates,
    }
    target = OUTPUT / "frozen_candidates.json"
    if target.exists():
        if json_sha(target) != payload:
            raise RuntimeError("existing_frozen_candidates_differ_refuse_overwrite")
    else:
        atomic_json(target, payload)
    return payload


def split_ids(panel, outer, inner_by_fold, fold):
    all_ids = {r["canonical_observation_id"] for r in panel}
    test = sorted(i for i, r in outer.items() if int(r["outer_fold"]) == fold)
    train = all_ids - set(test)
    assignment = inner_by_fold[fold]
    output = {}
    for inner_fold in range(4):
        valid = sorted(i for i, r in assignment.items() if int(r["inner_fold"]) == inner_fold)
        fit = sorted(train - set(valid))
        output[inner_fold] = (fit, valid)
    return train, test, output


def fit_one_candidate(family, candidate, x, y, splits, model_templates):
    results = []
    for inner_fold, (fit_ids, valid_ids) in splits.items():
        fit_idx = np.asarray([x["id_to_index"][i] for i in fit_ids], dtype=int)
        valid_idx = np.asarray([x["id_to_index"][i] for i in valid_ids], dtype=int)
        estimator = clone(model_templates[family]).set_params(**candidate["raw_params"])
        with threadpool_limits(limits=1):
            estimator.fit(x["matrix"][fit_idx], y[fit_idx])
            prediction = np.asarray(estimator.predict(x["matrix"][valid_idx]), dtype=np.float64)
        if not np.isfinite(prediction).all():
            results.append({"inner_fold": inner_fold, "score": "", "status": "numeric_failure_nonfinite_prediction"})
        else:
            results.append({"inner_fold": inner_fold,
                            "score": float(r2_score(y[valid_idx], prediction)), "status": "scored"})
    status = "scored" if all(r["status"] == "scored" for r in results) else "numeric_failure"
    return {"family": family, "candidate_index": int(candidate["candidate_index"]),
            "params_json": json.dumps(candidate["raw_params"], sort_keys=True, separators=(",", ":")),
            "status": status, "inner": results,
            "mean_inner_r2": (float(np.mean([r["score"] for r in results])) if status == "scored" else "")}


def fit_all(folds: list[int], workers: int = 4) -> dict[str, Any]:
    from run_qonductor_grouped_reproduction import base_models

    if workers < 1 or workers > 4:
        raise ValueError("workers_must_be_between_1_and_4")
    if any(fold != 0 for fold in folds):
        gate_path = PANEL / "mali" / "fold_0_gate_receipt.json"
        if not gate_path.is_file():
            raise RuntimeError("strict_fold0_independent_gate_required_before_continuation")
        gate = json_sha(gate_path)
        if (gate.get("status") != "PASS"
                or gate.get("authorization_sha256") != sha_file(AUTH)
                or gate.get("parent_protocol_sha256") != sha_file(PARENT)
                or gate.get("completion_protocol_sha256") != sha_file(CONTRACT)
                or gate.get("validator_sha256") != sha_file(ROOT / "benchmark_v1/scripts/validate_strict_logical_fold0.py")):
            raise RuntimeError("strict_fold0_independent_gate_pin_mismatch")
    panel, feature_by_id, outer, inner_by_fold = load_panel()
    frozen = freeze_candidates()
    templates, _grids, _blob = source_grids_and_models()
    ids = [r["canonical_observation_id"] for r in panel]
    id_to_index = {identity: index for index, identity in enumerate(ids)}
    xmat = np.asarray([[float(feature_by_id[i][name]) for name in FEATURE_ORDER]
                       for i in ids], dtype=np.float64)
    y = np.asarray([float(next(r for r in panel if r["canonical_observation_id"] == i)["target_seconds"])
                    for i in ids], dtype=np.float64)
    if xmat.shape != (340, 5) or not np.isfinite(xmat).all() or not np.isfinite(y).all():
        raise RuntimeError("strict_qonductor_design_matrix_invalid")
    x = {"matrix": xmat, "id_to_index": id_to_index}
    fields = ["outer_fold", "model_family", "candidate_index", "inner_fold", "params_json", "status", "r2_seconds"]
    score_path = OUTPUT / "candidate_inner_scores.csv"
    existing = read_csv(score_path) if score_path.exists() else []
    score_map = {(int(r["outer_fold"]), r["model_family"], int(r["candidate_index"]), int(r["inner_fold"])): r
                 for r in existing}
    if len(score_map) != len(existing):
        raise RuntimeError("duplicate_candidate_inner_score_rows")
    all_predictions: list[dict[str, Any]] = []
    prediction_path = OUTPUT / "predictions.csv"
    if prediction_path.exists():
        all_predictions = read_csv(prediction_path)
    fold_manifest_path = OUTPUT / "fold_selection.json"
    fold_records_raw = json_sha(fold_manifest_path).get("folds", {}) if fold_manifest_path.exists() else {}
    fold_records = {int(key): value for key, value in fold_records_raw.items()}
    identity_signature = {
        "authorization_sha256": sha_file(AUTH),
        "parent_protocol_sha256": sha_file(PARENT),
        "completion_protocol_sha256": sha_file(CONTRACT),
        "panel_manifest_sha256": sha_file(PANEL / "manifest.json"),
        "logical_features_sha256": sha_file(PANEL / "logical_features.csv"),
        "outer_splits_sha256": sha_file(PANEL / "outer_splits.csv"),
        "inner_splits_sha256": sha_file(PANEL / "inner_splits.csv"),
        "candidate_manifest_sha256": sha_file(OUTPUT / "frozen_candidates.json"),
        "feature_ledger_sha256": sha_file(FEATURES),
        "representation_rows_sha256": sha_file(REPRESENTATIONS),
        "sklearn_version": sklearn.__version__,
    }
    for fold in folds:
        if not 0 <= fold <= 4:
            raise ValueError(f"invalid_outer_fold:{fold}")
        if fold in fold_records:
            if fold_records[fold].get("identity_signature") != identity_signature:
                raise RuntimeError(f"completed_fold_identity_mismatch:{fold}")
            existing_fold = [r for r in all_predictions if int(r["outer_fold"]) == fold]
            expected_fold_rows = 7 * sum(int(row["outer_fold"]) == fold for row in outer.values())
            if len(existing_fold) != expected_fold_rows:
                raise RuntimeError(f"completed_fold_prediction_count_mismatch:{fold}")
            print(json.dumps({"fold": fold, "status": "already_complete"}), flush=True)
            continue
        train_ids, test_ids, splits = split_ids(panel, outer, inner_by_fold, fold)
        for inner_fold, (_fit_ids, valid_ids) in splits.items():
            validation_values = [y[id_to_index[i]] for i in valid_ids]
            if len(validation_values) < 2 or len(set(validation_values)) < 2:
                raise RuntimeError(f"invalid_inner_r2_partition:{fold}:{inner_fold}")
        candidate_tasks = []
        for family in FAMILIES:
            for candidate in frozen["candidates"][family]:
                needed = [(fold, family, int(candidate["candidate_index"]), inner_fold) for inner_fold in range(4)]
                if all(key in score_map for key in needed):
                    continue
                candidate_tasks.append((family, candidate))
        print(json.dumps({"fold": fold, "status": "fit_start", "candidate_jobs": len(candidate_tasks),
                          "workers": workers, "inner_fits_total": sum(len(frozen['candidates'][f]) * 4 for f in FAMILIES)}), flush=True)
        def submit(candidate_task):
            return fit_one_candidate(candidate_task[0], candidate_task[1], x, y, splits, templates)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(submit, candidate_tasks):
                for part in result["inner"]:
                    key = (fold, result["family"], result["candidate_index"], int(part["inner_fold"]))
                    row = {"outer_fold": fold, "model_family": result["family"],
                           "candidate_index": result["candidate_index"], "inner_fold": part["inner_fold"],
                           "params_json": result["params_json"], "status": part["status"],
                           "r2_seconds": part["score"]}
                    score_map[key] = row
                atomic_csv(score_path, sorted(score_map.values(), key=lambda r: (
                    int(r["outer_fold"]), FAMILIES.index(r["model_family"]), int(r["candidate_index"]), int(r["inner_fold"]))), fields)
                print(json.dumps({"fold": fold, "family": result["family"],
                                  "candidate": result["candidate_index"], "status": result["status"],
                                  "mean_inner_r2": result["mean_inner_r2"]}), flush=True)
        selected_family_scores = {}
        fold_predictions = []
        for family in FAMILIES:
            family_candidates = []
            for candidate in frozen["candidates"][family]:
                rows = [score_map.get((fold, family, int(candidate["candidate_index"]), inner_fold))
                        for inner_fold in range(4)]
                if any(row is None for row in rows):
                    raise RuntimeError(f"candidate_inner_scores_incomplete:{fold}:{family}:{candidate['candidate_index']}")
                statuses = [row["status"] for row in rows]
                if all(status == "scored" for status in statuses):
                    scores = [float(row["r2_seconds"]) for row in rows]
                    family_candidates.append((float(np.mean(scores)), int(candidate["candidate_index"]), candidate, scores))
            if not family_candidates:
                raise RuntimeError(f"no_valid_candidate:{fold}:{family}")
            # Stable tie: earliest candidate in the frozen list.
            best = max(family_candidates, key=lambda item: (item[0], -item[1]))
            score, candidate_index, candidate, scores = best
            fit_idx = np.asarray([id_to_index[i] for i in sorted(train_ids)], dtype=int)
            test_idx = np.asarray([id_to_index[i] for i in test_ids], dtype=int)
            estimator = clone(templates[family]).set_params(**candidate["raw_params"])
            with threadpool_limits(limits=1):
                estimator.fit(xmat[fit_idx], y[fit_idx])
                prediction = np.asarray(estimator.predict(xmat[test_idx]), dtype=np.float64)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"nonfinite_outer_prediction:{fold}:{family}")
            selected_family_scores[family] = {"candidate_index": candidate_index,
                "params": candidate["raw_params"], "mean_inner_r2_seconds": score,
                "inner_r2_seconds": scores, "refit_rows": len(fit_idx)}
            for idx, value in zip(test_idx.tolist(), prediction.tolist()):
                identity = ids[idx]
                fold_predictions.append({"canonical_observation_id": identity, "outer_fold": fold,
                    "method_id": "strict_qonductor_" + family.lower().replace(" ", "_"),
                    "model_family": family, "selected_candidate_index": candidate_index,
                    "actual_seconds": y[idx], "predicted_seconds": value,
                    "negative_prediction": int(value < 0), "status": "predicted"})
        selector = max(FAMILIES, key=lambda name: (selected_family_scores[name]["mean_inner_r2_seconds"], -FAMILIES.index(name)))
        selector_rows = [r for r in fold_predictions if r["model_family"] == selector]
        for row in selector_rows:
            fold_predictions.append({**row, "method_id": "strict_qonductor_six_family_budgeted_selector"})
        all_predictions = [row for row in all_predictions if int(row["outer_fold"]) != fold] + fold_predictions
        atomic_csv(prediction_path, sorted(all_predictions, key=lambda r: (
            int(r["outer_fold"]), str(r["canonical_observation_id"]), str(r["method_id"]))),
            ["canonical_observation_id", "outer_fold", "method_id", "model_family", "selected_candidate_index",
             "actual_seconds", "predicted_seconds", "negative_prediction", "status"])
        fold_records[fold] = {"status": "complete", "identity_signature": identity_signature,
                              "outer_test_ids": test_ids, "selected": selected_family_scores,
                              "selector_family": selector, "prediction_rows": len(fold_predictions)}
        atomic_json(fold_manifest_path, {"artifact_id": "strict-logical-qonductor-fold-selection",
            "status": "running" if len(fold_records) < 5 else "complete", "identity_signature": identity_signature,
            "folds": fold_records})
    if set(fold_records) != set(range(5)):
        return {"status": "partial", "completed_folds": sorted(fold_records),
                "remaining_folds": sorted(set(range(5)) - set(fold_records)),
                "completed_inner_score_rows": len(score_map),
                "expected_inner_score_rows": sum(EXPECTED_COUNTS) * 4 * len(fold_records)}
    return aggregate(identity_signature)


def metric_rows(predictions):
    methods = sorted({row["method_id"] for row in predictions})
    out = []
    for method in methods:
        rows = [row for row in predictions if row["method_id"] == method and row["status"] == "predicted"]
        y = np.asarray([float(r["actual_seconds"]) for r in rows], dtype=np.float64)
        p = np.asarray([float(r["predicted_seconds"]) for r in rows], dtype=np.float64)
        if len(y) == 0:
            continue
        abs_error = np.abs(y-p)
        out.append({"method_id": method, "assigned_rows": 340, "predicted_rows": len(y),
                    "coverage": len(y)/340, "mae_seconds": float(abs_error.mean()),
                    "medae_seconds": float(np.median(abs_error)), "rmse_seconds": float(np.sqrt(np.mean((y-p)**2))),
                    "r2_seconds": float(r2_score(y,p)), "p90_absolute_error_seconds": float(np.quantile(abs_error,.9)),
                    "p99_absolute_error_seconds": float(np.quantile(abs_error,.99)),
                    "max_absolute_error_seconds": float(abs_error.max()),
                    "negative_prediction_count": sum(float(r["predicted_seconds"]) < 0 for r in rows)})
    return out


def aggregate(identity_signature=None):
    path = OUTPUT / "predictions.csv"
    if not path.exists():
        raise RuntimeError("qonductor_predictions_missing")
    predictions = read_csv(path)
    if len(predictions) != 340 * 7 or len({(r["canonical_observation_id"], r["method_id"]) for r in predictions}) != len(predictions):
        raise RuntimeError("qonductor_oof_identity_count_mismatch")
    metrics = metric_rows(predictions)
    atomic_csv(OUTPUT / "metrics.csv", metrics, list(metrics[0]))
    group_by_id = {r["canonical_observation_id"]: r["group_id"] for r in read_csv_path(PANEL / "logical_features.csv")}
    # Paired logical-group bootstrap for the six-family selector and each family.
    rng = np.random.default_rng(42)
    groups = sorted(set(group_by_id.values()))
    bootstrap_rows = []
    for metric in metrics:
        method = metric["method_id"]
        by_group: dict[str, list[dict[str, str]]] = {}
        for row in predictions:
            if row["method_id"] == method:
                by_group.setdefault(group_by_id[row["canonical_observation_id"]], []).append(row)
        observed = float(metric["mae_seconds"])
        reps = []
        for _ in range(10000):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            err = [abs(float(r["actual_seconds"])-float(r["predicted_seconds"]))
                   for group_id in sampled for r in by_group[group_id]]
            reps.append(float(np.mean(err)))
        bootstrap_rows.append({"method_id": method, "observed_mae_seconds": observed,
            "bootstrap_mean_mae_seconds": float(np.mean(reps)),
            "bootstrap_ci_low_seconds": float(np.quantile(reps,.025)),
            "bootstrap_ci_high_seconds": float(np.quantile(reps,.975)),
            "resamples": 10000, "seed": 42, "cluster": "backend_independent_model_signature"})
    atomic_csv(OUTPUT / "group_bootstrap_mae.csv", bootstrap_rows, list(bootstrap_rows[0]))
    if identity_signature is None:
        identity_signature = json_sha(OUTPUT / "fold_selection.json")["identity_signature"]
    manifest = {"artifact_id": "strict-logical-qonductor-oof", "status": "complete",
        "assigned_rows": 340, "method_count": 7, "attempt_rows": len(predictions),
        "evaluation_target_clock": "archived_observed_mean_result_time_taken_seconds",
        "feature_stage": "offline_nominal_compiled_views_of_the_same_340_Ma-Li_logical_observations",
        "fidelity": "budgeted Qonductor-style unified adaptation; not full-grid paper reproduction",
        "identity_signature": identity_signature,
        "outputs": {name: sha_file(OUTPUT/name) for name in
                    ("frozen_candidates.json", "candidate_inner_scores.csv", "fold_selection.json",
                     "predictions.csv", "metrics.csv", "group_bootstrap_mae.csv")}}
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    return {"status": "complete", "metrics": metrics}


def read_csv_path(path):
    return read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "freeze", "fit", "aggregate"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "preflight":
        panel, features, outer, inner = load_panel()
        result = {"status": "PASS", "panel_rows": len(panel), "feature_rows": len(features),
                  "outer_folds": len(set(int(r["outer_fold"]) for r in outer.values())),
                  "inner_assignments_by_outer_fold": {str(k): len(v) for k,v in inner.items()},
                  "feature_order": FEATURE_ORDER, "authority_sha256": sha_file(AUTH)}
    elif args.stage == "freeze":
        result = freeze_candidates()
    elif args.stage == "fit":
        result = fit_all(args.folds, args.workers)
    else:
        result = aggregate()
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
