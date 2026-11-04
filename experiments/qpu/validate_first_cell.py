#!/usr/bin/env python3
"""Label-blind replay of the first full cell's selected checkpoint/predictions."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from pathlib import Path
import csv, json, math, sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
import torch
from benchmark26 import mali_training as base
from core import OUT, RUN, atomic_json, load_context, parts_for, rows, sha, stable
from train_neural import ExampleCache, make_model, protocol_lock


def main():
    """Replay the first full cell (E1, fold 0, M0, seed 42) without labels and compare with its saved predictions.

    Checks the manifest, the lock, the epoch selection, the transform identity and the initialization gate,
    reloads the saved checkpoint, recomputes the test predictions and requires that they agree with the
    saved ones to 1e-6 seconds. Writes a receipt.
    """
    context, targets, split = load_context("E1", include_targets=False)
    if targets is not None or split.get("target_values_read") is not False:
        raise RuntimeError("validator_must_not_read_target_values")
    parts = parts_for(context, 0)
    folder = RUN / "training/E1/fold_0/M0/seed_42"
    manifest = json.loads((folder / "manifest.json").read_text())
    identity = manifest["identity"]
    lock, lock_sha = protocol_lock(require_code=False)
    if manifest.get("status") != "complete" or identity.get("protocol_lock_sha256") != lock_sha:
        raise RuntimeError("first_cell_or_lock_not_complete")
    if (
        identity.get("fit_ids_sha256") != base.stable_hash(parts["fit"])
        or identity.get("validation_ids_sha256") != base.stable_hash(parts["validation"])
        or identity.get("test_ids_sha256") != base.stable_hash(parts["test"])
    ):
        raise RuntimeError("first_cell_partition_identity_mismatch")
    history = rows(folder / "epoch_history.csv")
    vals = [float(r["validation_mse"]) for r in history]
    if not vals or any(not math.isfinite(v) for v in vals):
        raise RuntimeError("invalid_validation_curve")
    best = min(range(len(vals)), key=lambda i: (vals[i], i)) + 1
    if manifest.get("best_epoch") != best or identity.get("canary") is not False:
        raise RuntimeError("checkpoint_selection_rule_mismatch")
    transform = json.loads((RUN / "training/E1/fold_0/transform.json").read_text())
    if transform["fit_ids_sha256"] != base.stable_hash(parts["fit"]):
        raise RuntimeError("preprocess_not_fit_only")
    saved = rows(folder / "predictions.csv")
    expected = set(parts["test"])
    got = {r["canonical_observation_id"] for r in saved}
    if (
        len(saved) != len(expected)
        or got != expected
        or any(not math.isfinite(float(r["predicted_seconds"])) for r in saved)
    ):
        raise RuntimeError("first_cell_prediction_coverage_or_finiteness_failure")
    gate = json.loads((folder / "initialization_gate.json").read_text())
    if gate.get("status") != "PASS" or gate.get("uses_test_labels") is not False:
        raise RuntimeError("first_cell_initialization_gate_failed")
    ckpt = base.load_saved_checkpoint(folder / "checkpoint.pt", identity, torch)
    model, model_sha, _ = make_model("M0", transform, 42, torch)
    if model_sha != identity["model_source_sha256"]:
        raise RuntimeError("pinned_model_identity_mismatch")
    model.load_state_dict(ckpt["best_model_state"])
    model.eval()
    cache = ExampleCache(context, transform, "M0", 256 * 1024**2)
    predicted = {}
    with torch.no_grad():
        for cid in sorted(expected):
            batch = base._batch_to_device([cache.get(cid)], "cuda:0", False)
            predicted[cid] = float(base._prediction(model, batch).detach().cpu().reshape(-1)[0])
            del batch
    saved_map = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in saved}
    maximum = max(abs(predicted[c] - saved_map[c]) for c in expected)
    if maximum > 1e-6:
        raise RuntimeError(f"prediction_replay_mismatch:{maximum}")
    receipt = {
        "status": "PASS_FIRST_CELL_LABEL_BLIND_REPLAY",
        "evaluation": "E1",
        "fold": 0,
        "treatment": "M0",
        "seed": 42,
        "test_rows": len(expected),
        "fit_ids_sha256": base.stable_hash(parts["fit"]),
        "validation_ids_sha256": base.stable_hash(parts["validation"]),
        "test_ids_sha256": base.stable_hash(parts["test"]),
        "best_epoch_from_validation_curve": best,
        "best_validation_mse": vals[best - 1],
        "prediction_replay_max_abs_difference": maximum,
        "target_values_read": False,
        "test_targets_passed_to_training": False,
        "prediction_sha256": sha(folder / "predictions.csv"),
        "checkpoint_sha256": sha(folder / "checkpoint.pt"),
        "cell_manifest_sha256": sha(folder / "manifest.json"),
        "protocol_lock_sha256": lock_sha,
        "split_receipt_sha256": sha(OUT / "splits/split_validation_receipt.json"),
    }
    atomic_json(RUN / "training/first_cell_validation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
