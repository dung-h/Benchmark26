#!/usr/bin/env python3
"""Predict the test rows again from saved graph-ablation checkpoints and compare with the saved predictions.

The script reads a finished run directory and writes nothing into it. For each
chosen training run (evaluation setting, fold, model, seed) it rebuilds the fold
transform with the current code and checks it against the saved `transform.json`.
It then loads the best-validation weights from `checkpoint.pt`, predicts every
test row and compares the result with `predictions.csv`. A code change that only
renames things, or removes code that the run does not reach, should reproduce the
saved predictions up to floating-point noise.

Needs the study files (`B26_ARTIFACTS`), the pinned Ma–Li checkout
(`B26_MALI_SOURCE`), `B26_RUN_DIR` set to the finished run, PyTorch and a CUDA GPU.

    python experiments/graph_ablation/replay_checkpoints.py --output /tmp/replay.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "experiments/qpu/graph_ablation_runner.py"

# Two evaluation settings x four models, spread over folds and seeds.
DEFAULT_CELLS = (
    ("E1", 0, "M1", 42),
    ("E1", 1, "G1", 1234),
    ("E1", 2, "G1e", 31415),
    ("E1", 3, "G1p", 42),
    ("E2", 4, "M1", 1234),
    ("E2", 0, "G1", 31415),
    ("E2", 1, "G1e", 42),
    ("E2", 2, "G1p", 1234),
)
TOLERANCE_SECONDS = 1e-4


def load_runner():
    spec = importlib.util.spec_from_file_location("graph_ablation_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER_PATH.parent))
    spec.loader.exec_module(module)
    return module


def replay_cell(runner, torch, contexts, evaluation, fold, model_code, seed):
    if evaluation not in contexts:
        contexts[evaluation] = runner.load_context(evaluation, include_targets=False)[0]
    context = contexts[evaluation]
    parts = runner.parts_for(context, fold)
    folder = runner.cell_paths(evaluation, model_code, fold, seed)
    saved_transform = json.loads((folder.parent.parent / "transform.json").read_text(encoding="utf-8"))
    transform = runner.fit_transform(context, parts, fold)
    if transform != saved_transform:
        return {"status": "FAIL", "reason": "fold transform differs from the saved transform.json"}
    model, _model_sha, _projection = runner.make_model(model_code, transform, seed, torch)
    try:
        payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    except TypeError:  # Torch versions predating weights_only.
        payload = torch.load(folder / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(payload["best_model_state"])
    model.eval()
    with (folder / "predictions.csv").open(newline="") as handle:
        saved = {row["canonical_observation_id"]: float(row["predicted_seconds"]) for row in csv.DictReader(handle)}
    if not all(math.isfinite(value) for value in saved.values()):
        return {"status": "FAIL", "reason": "a saved prediction is not finite"}
    if list(saved) != list(parts["test"]):
        return {"status": "FAIL", "reason": "test rows differ from the saved predictions"}
    spec = runner.METHODS[model_code]
    cache = runner.ExampleCache(context, transform, model_code, 2 * 1024**3)
    largest = 0.0
    with torch.no_grad():
        for run_id in parts["test"]:
            batch = runner.base._batch_to_device([cache.get(run_id)], "cuda:0", spec["graph"])
            value = float(runner.base._prediction(model, batch).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value):
                return {"status": "FAIL", "reason": f"non-finite prediction for {run_id}"}
            largest = max(largest, abs(value - saved[run_id]))
    del model, cache
    torch.cuda.empty_cache()
    return {
        "status": "PASS" if largest <= TOLERANCE_SECONDS else "FAIL",
        "test_rows": len(saved),
        "largest_difference_seconds": largest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="receipt path, outside the run directory")
    args = parser.parse_args()
    runner = load_runner()
    if args.output.resolve().is_relative_to(runner.RUN.resolve()):
        raise SystemExit("write the receipt outside the run directory")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    contexts, results = {}, []
    for evaluation, fold, model_code, seed in DEFAULT_CELLS:
        result = replay_cell(runner, torch, contexts, evaluation, fold, model_code, seed)
        result.update({"evaluation": evaluation, "fold": fold, "model": model_code, "seed": seed})
        print(json.dumps(result), flush=True)
        results.append(result)
    receipt = {
        "status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
        "run_directory": str(runner.RUN),
        "tolerance_seconds": TOLERANCE_SECONDS,
        "largest_difference_seconds": max(r.get("largest_difference_seconds", math.inf) for r in results),
        "cells": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "largest_difference_seconds")}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
