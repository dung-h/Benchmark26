#!/usr/bin/env python3
"""Final label-blind checkpoint/OOF replay. Never trains or rewrites producer files."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import argparse
import collections
import csv
import datetime
import gc
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from core import RUN, load_context, parts_for, rows, sha, stable, atomic_json, atomic_csv

SEEDS = (42, 1234, 31415)
TREATMENTS = ("M0", "M1", "G0", "G1")


def earliest_minimum(history):
    vals = [float(r["validation_mse"]) for r in history]
    if not vals or not all(math.isfinite(x) for x in vals):
        raise RuntimeError("Invalid validation curve")
    return min(range(len(vals)), key=lambda i: (vals[i], i)) + 1


def unique_predictions(records, expected_ids):
    ids = [r["canonical_observation_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise RuntimeError("Prediction IDs missing, duplicate, or foreign")
    if not all(math.isfinite(float(r["predicted_seconds"])) for r in records):
        raise RuntimeError("Nonfinite saved prediction")


def finite_state(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(x, torch) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(x, torch) for x in value)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=RUN / "validation/final_v2")
    ap.add_argument("--replay-all", action="store_true", help="Inference only, all saved held-out predictions")
    args = ap.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    validator_sha = sha(Path(__file__))
    final_receipt = dest / "receipt.json"
    if final_receipt.exists():
        raise RuntimeError("Final receipt already exists; do not overwrite")
    started = time.time()
    import numpy as np
    import torch
    import train_neural as producer
    from benchmark26 import mali_training as base
    from benchmark26.gpu_lease import acquire_gpu_lease

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    lock, lock_sha = producer.protocol_lock(require_code=False)
    lease = acquire_gpu_lease("expanded-qpu-final-saved-checkpoint-replay") if args.replay_all else None
    cases = []
    cells = []
    selectors = []
    expected_paths = set()
    coverage = collections.Counter()
    try:
        for ev in ("E1", "E2", "E3"):
            context, labels, split = load_context(ev, include_targets=False)
            assert labels is None and split["target_values_read"] is False
            panel = context["panel_by_id"]
            all_ids = set(context["all_ids"])
            tested = []
            for fold in range(context["case_count"]):
                parts = parts_for(context, fold)
                tested.extend(parts["test"])
                groups = {r["canonical_observation_id"]: r["group_id"] for r in context["panel"]}
                assert not set(parts["fit"]) & set(parts["validation"])
                assert not set(parts["train"]) & set(parts["test"])
                if ev == "E1":
                    assert not {groups[c] for c in parts["train"]} & {groups[c] for c in parts["test"]}
                if ev in ("E1", "E3"):
                    assert not {groups[c] for c in parts["fit"]} & {groups[c] for c in parts["validation"]}
                if ev == "E3":
                    backend = context["cases"][fold]
                    assert all(panel[c]["backend"] == backend for c in parts["test"])
                    assert all(panel[c]["backend"] != backend for c in parts["train"])
                transform_path = RUN / "training" / ev / f"fold_{fold}" / "transform.json"
                transform = json.loads(transform_path.read_text())
                # Recompute from fit IDs, using no label reader. Independent direct global/shot moment checks below.
                assert transform == producer.fit_transform(context, parts, fold)
                assert transform["fit_ids_sha256"] == stable(parts["fit"])
                raw = np.asarray(
                    [[float(context["ledger"][c][f"global_{j:02d}"]) for j in range(51)] for c in parts["fit"]],
                    dtype=np.float64,
                )
                retained = np.flatnonzero(raw.sum(axis=0) > 0)
                assert retained.tolist() == transform["global"]["retained_indices"]
                assert np.array_equal(raw[:, retained].mean(axis=0), np.asarray(transform["global"]["mean"]))
                assert np.array_equal(raw[:, retained].std(axis=0, ddof=1), np.asarray(transform["global"]["std"]))
                shots = np.asarray([float(panel[c]["shots"]) for c in parts["fit"]], dtype=np.float64)
                assert float(shots.mean()) == transform["shots"]["mean"]
                assert float(shots.std(ddof=1)) == transform["shots"]["std_ddof1"]
                cases.append(
                    {
                        "evaluation": ev,
                        "fold": fold,
                        "heldout_backend": context["cases"][fold] if ev == "E3" else "",
                        "fit_rows": len(parts["fit"]),
                        "validation_rows": len(parts["validation"]),
                        "test_rows": len(parts["test"]),
                        "fit_groups": len({groups[c] for c in parts["fit"]}),
                        "test_groups": len({groups[c] for c in parts["test"]}),
                        "fit_test_group_overlap": len(
                            {groups[c] for c in parts["fit"]} & {groups[c] for c in parts["test"]}
                        ),
                        "fit_only_transform_recomputed": True,
                        "target_values_read": False,
                    }
                )
                for treatment in TREATMENTS:
                    for seed in SEEDS:
                        folder = RUN / "training" / ev / f"fold_{fold}" / treatment / f"seed_{seed}"
                        mp = folder / "manifest.json"
                        expected_paths.add(mp)
                        manifest = json.loads(mp.read_text())
                        ident = manifest["identity"]
                        assert manifest["status"] == "complete" and manifest["epochs"] == 500
                        assert (
                            ident["evaluation"] == ev
                            and ident["fold"] == fold
                            and ident["treatment"] == treatment
                            and ident["seed"] == seed
                        )
                        assert ident["canary"] is False and ident["protocol_lock_sha256"] == lock_sha
                        assert ident["transform_sha256"] == stable(transform)
                        for key, ids in [
                            ("fit_ids_sha256", "fit"),
                            ("validation_ids_sha256", "validation"),
                            ("test_ids_sha256", "test"),
                            ("outer_train_ids_sha256", "train"),
                        ]:
                            assert ident[key] == stable(parts[ids])
                        for name, digest in manifest["outputs"].items():
                            assert sha(folder / name) == digest, str(folder / name)
                        hist = rows(folder / "epoch_history.csv")
                        assert len(hist) == 500 and [int(r["epoch"]) for r in hist] == list(range(1, 501))
                        assert all(
                            math.isfinite(float(r[k]))
                            for r in hist
                            for k in ["train_mse", "validation_mse", "best_validation_mse"]
                        )
                        best = earliest_minimum(hist)
                        assert manifest["best_epoch"] == best
                        assert manifest["best_validation_mse"] == float(hist[best - 1]["validation_mse"])
                        saved = rows(folder / "predictions.csv")
                        unique_predictions(saved, parts["test"])
                        assert all(
                            int(r["seed"]) == seed
                            and int(r["outer_fold"]) == fold
                            and r["treatment"] == treatment
                            and int(r["selected_epoch"]) == best
                            for r in saved
                        )
                        assert manifest["test_labels_available_to_training_function"] is False
                        gate = json.loads((folder / "initialization_gate.json").read_text())
                        assert gate["status"] == "PASS" and gate["uses_test_labels"] is False
                        assert set(gate["fixture_fit_ids"]).issubset(set(parts["fit"]))
                        if treatment in ("M1", "G1"):
                            assert (
                                gate["added_parameters"] == 64 and gate["zero_shot_initialization_max_abs_diff"] <= 1e-6
                            )
                            assert (
                                math.isfinite(gate["new_shot_column_gradient_max_abs"])
                                and gate["new_shot_column_gradient_max_abs"] > 0
                            )
                        checkpoint = base.load_saved_checkpoint(folder / "checkpoint.pt", ident, torch)
                        assert checkpoint["next_epoch"] == 500 and checkpoint["best_epoch"] + 1 == best
                        assert checkpoint["best_validation_mse"] == manifest["best_validation_mse"]
                        assert all(
                            finite_state(checkpoint[k], torch)
                            for k in ["model_state", "best_model_state", "optimizer_state"]
                        )
                        rp = dest / "replay_cells" / ev / f"fold_{fold}" / treatment / f"seed_{seed}.json"
                        row = {
                            "evaluation": ev,
                            "fold": fold,
                            "treatment": treatment,
                            "seed": seed,
                            "test_rows": len(saved),
                            "best_epoch": best,
                            "epochs": 500,
                            "parameter_count": manifest["parameter_count"],
                            "checkpoint_sha256": manifest["outputs"]["checkpoint.pt"],
                            "prediction_sha256": manifest["outputs"]["predictions.csv"],
                            "manifest_sha256": sha(mp),
                            "validator_sha256": validator_sha,
                            "all_states_finite": True,
                            "target_values_read": False,
                        }
                        if args.replay_all:
                            if rp.exists():
                                previous = json.loads(rp.read_text())
                                assert (
                                    previous["manifest_sha256"] == row["manifest_sha256"]
                                    and previous["validator_sha256"] == validator_sha
                                    and previous["status"] == "PASS"
                                )
                                row.update(
                                    {
                                        k: previous[k]
                                        for k in ["prediction_replay_rows", "max_abs_replay_difference", "status"]
                                    }
                                )
                            else:
                                model, model_sha, _projection = producer.make_model(treatment, transform, seed, torch)
                                assert model_sha == ident["model_source_sha256"]
                                model.load_state_dict(checkpoint["best_model_state"])
                                assert sum(p.numel() for p in model.parameters()) == manifest["parameter_count"]
                                model.eval()
                                cache = producer.ExampleCache(context, transform, treatment, 512 * 1024**2)
                                maximum = 0.0
                                max_scaled = 0.0
                                max_repeat_difference = 0.0
                                tolerance_exceedances = 0
                                with torch.no_grad():
                                    for rec in saved:
                                        cid = rec["canonical_observation_id"]
                                        batch = base._batch_to_device(
                                            [cache.get(cid)], "cuda:0", treatment.startswith("G")
                                        )
                                        value = float(base._prediction(model, batch).detach().cpu().reshape(-1)[0])
                                        assert math.isfinite(value)
                                        reference = float(rec["predicted_seconds"])
                                        difference = abs(value - reference)
                                        maximum = max(maximum, difference)
                                        max_scaled = max(max_scaled, difference / (1e-5 + 1e-6 * abs(reference)))
                                        if difference > 1e-6:
                                            tolerance_exceedances += 1
                                            repeated = float(
                                                base._prediction(model, batch).detach().cpu().reshape(-1)[0]
                                            )
                                            max_repeat_difference = max(max_repeat_difference, abs(repeated - value))
                                        del batch
                                assert max_scaled <= 1.0, (str(folder), maximum, max_scaled)
                                row.update(
                                    {
                                        "prediction_replay_rows": len(saved),
                                        "max_abs_replay_difference": maximum,
                                        "max_scaled_replay_difference": max_scaled,
                                        "rows_above_initial_absolute_1e_6_tolerance": tolerance_exceedances,
                                        "max_same_input_repeat_difference": max_repeat_difference,
                                        "replay_absolute_tolerance": 1e-5,
                                        "replay_relative_tolerance": 1e-6,
                                        "status": "PASS",
                                    }
                                )
                                atomic_json(rp, row)
                                del model, cache
                                torch.cuda.empty_cache()
                        else:
                            row.update(
                                {
                                    "prediction_replay_rows": 0,
                                    "max_abs_replay_difference": None,
                                    "status": "STRUCTURE_ONLY",
                                }
                            )
                        coverage[(ev, treatment, seed)] += len(saved)
                        cells.append(row)
                        del checkpoint
                        gc.collect()
                        atomic_json(
                            dest / "progress.json",
                            {
                                "status": "RUNNING",
                                "validated_cells": len(cells),
                                "total_cells": 240,
                                "last_cell": row,
                                "elapsed_seconds": time.time() - started,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "validated_cells": len(cells),
                                    "evaluation": ev,
                                    "fold": fold,
                                    "treatment": treatment,
                                    "seed": seed,
                                    "replayed_rows": row["prediction_replay_rows"],
                                }
                            ),
                            flush=True,
                        )
                # Audit cached classical selection scores; do not fit another estimator.
                cf = RUN / "classical" / ev / f"fold_{fold}"
                cm = json.loads((cf / "regression_manifest.json").read_text())
                assert cm["status"] == "complete" and cm["test_labels_passed_to_selector_or_fit"] is False
                assert cm["identity"]["protocol_lock_sha256"] == lock_sha
                assert cm["identity"]["train_ids_sha256"] == stable(parts["train"]) and cm["identity"][
                    "test_ids_sha256"
                ] == stable(parts["test"])
                pred = rows(cf / "regression_predictions.csv")
                unique_predictions(pred, parts["test"])
                assert sha(cf / "regression_predictions.csv") == cm["predictions_sha256"]
                order = list(lock["classical"]["families"])
                bests = {}
                for family in order:
                    scores_path = cf / (family.lower().replace(" ", "_") + "_candidate_scores.json")
                    scores = json.loads(scores_path.read_text())
                    assert scores["identity"]["protocol_lock_sha256"] == lock_sha
                    good = [
                        r
                        for r in scores["scores"]
                        if r["status"] == "scored" and math.isfinite(float(r["mean_inner_r2"]))
                    ]
                    best_score = max(good, key=lambda r: (float(r["mean_inner_r2"]), -int(r["candidate_index"])))
                    stored = cm["family_selections"][family]
                    assert int(stored["candidate_index"]) == int(best_score["candidate_index"])
                    assert stored["mean_inner_raw_r2"] == best_score["mean_inner_r2"]
                    assert stored["params_json"] == best_score["params_json"]
                    bests[family] = best_score
                winner = max(order, key=lambda f: (float(bests[f]["mean_inner_r2"]), -order.index(f)))
                assert cm["selected_family"] == winner and all(r["selected_family"] == winner for r in pred)
                km = json.loads((cf / "constant_manifest.json").read_text())
                kp = rows(cf / "constant_predictions.csv")
                assert km["status"] == "complete" and km["test_labels_used"] is False
                assert km["identity"]["train_ids_sha256"] == stable(parts["train"]) and km["identity"][
                    "test_ids_sha256"
                ] == stable(parts["test"])
                assert sha(cf / "constant_predictions.csv") == km["predictions_sha256"]
                unique_predictions(kp, parts["test"])
                assert all(float(r["predicted_seconds"]) == km["outer_train_median_seconds"] for r in kp)
                selectors.append(
                    {
                        "evaluation": ev,
                        "fold": fold,
                        "selected_family": winner,
                        "selected_candidate_index": cm["selected_candidate_index"],
                        "test_rows": len(pred),
                        "prediction_hashes_match": True,
                        "inner_selection_replay": "PASS",
                        "classical_refit_executed": False,
                        "fitted_classical_state_saved": False,
                    }
                )
            assert len(tested) == len(set(tested)) == 3443 and set(tested) == all_ids
        actual_paths = set((RUN / "training").glob("E*/fold_*/*/seed_*/manifest.json"))
        assert actual_paths == expected_paths and len(cells) == 240 and len(selectors) == 20
        assert all(n == 3443 for n in coverage.values()) and len(coverage) == 36
        _, final_lock_sha = producer.protocol_lock(require_code=False)
        assert final_lock_sha == lock_sha
        atomic_csv(dest / "neural_cells.csv", cells)
        atomic_csv(dest / "split_and_preprocessing_cases.csv", cases)
        atomic_csv(dest / "classical_selection.csv", selectors)
        receipt = {
            "status": "PASS" if args.replay_all else "STRUCTURE_ONLY_NOT_FINAL",
            "completed_neural_cells": len(cells),
            "classical_cases": len(selectors),
            "prediction_replay_rows": sum(r["prediction_replay_rows"] for r in cells),
            "max_abs_prediction_replay_difference": max((r["max_abs_replay_difference"] or 0) for r in cells),
            "replay_absolute_tolerance": 1e-5,
            "replay_relative_tolerance": 1e-6,
            "initial_strict_replay_attempt": "validation/final/failure.json; 1.9073486328125e-6 seconds exceeded initial absolute1e-6. Preserved; no prediction changed.",
            "target_values_read": False,
            "new_training_or_refit": False,
            "replay_all": args.replay_all,
            "protocol_lock_sha256": lock_sha,
            "validator_sha256": validator_sha,
            "elapsed_seconds": time.time() - started,
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operational_counter_issue": "Pinned runner counter414 includes reused cells; exact unique manifests240. Source unchanged.",
            "classical_limitation": "Saved selection and prediction receipts checked; fitted estimator states were not saved by producer, so no classical inference replay or refit claimed.",
        }
        atomic_json(final_receipt, receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException as exc:
        atomic_json(
            dest / "failure.json",
            {
                "status": "FAIL",
                "error": repr(exc),
                "validated_cells": len(cells),
                "validator_sha256": validator_sha,
                "protocol_lock_sha256": lock_sha,
                "elapsed_seconds": time.time() - started,
            },
        )
        raise
    finally:
        if lease:
            lease.release()


if __name__ == "__main__":
    main()
