#!/usr/bin/env python3
"""Retrain each neural model on the whole outer-training partition.

train_neural.py fits a model on inner folds 1-3 and keeps the epoch with the lowest
error on inner fold 0. This script trains the same architecture with the same seed on
all four inner folds for that number of epochs, refits the feature transforms on
those rows and predicts the outer test rows with the final weights. The neural errors
in the paper come from these models. Models with shot count are retrained for E1, E2
and E3, models without it for E1 and E2: 180 cells.

    python3 experiments/qpu/retrain_neural.py --output /path/to/retrained

The selected epochs are read from training/ of the study bundle, or of B26_RUN_DIR
when it is set. Completed cells are skipped. Test labels are never read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repository root
sys.path.insert(0, str(HERE))
import train_neural as tn  # noqa: E402
from core import atomic_csv, atomic_json, load_context, load_targets, parts_for, rows, sha  # noqa: E402

CACHE_GIB = 6
TREATMENTS = {"E1": ("M0", "M1", "G0", "G1"), "E2": ("M0", "M1", "G0", "G1"), "E3": ("M1", "G1")}


def all_cells():
    """Return every (evaluation, fold, treatment, seed) that is retrained."""
    return [
        (evaluation, fold, treatment, seed)
        for evaluation, treatments in TREATMENTS.items()
        for fold in range(10 if evaluation == "E3" else 5)
        for treatment in treatments
        for seed in tn.SEEDS
    ]


def disable_activation_checkpointing() -> None:
    """Build graph models without activation checkpointing (faster, more GPU memory)."""
    build = tn.base.make_model

    def without_checkpointing(model_class, variant, global_width, checkpoint_layers=False):
        return build(model_class, variant, global_width, checkpoint_layers=False)

    tn.base.make_model = without_checkpointing


def retrain_cell(context, fold, treatment, seed, output, torch):
    """Train one cell on the outer-training rows for its selected epochs and save its test predictions."""
    import numpy as np

    evaluation = context["evaluation"]
    relative = Path(evaluation) / f"fold_{fold}" / treatment / f"seed_{seed}"
    folder = output / relative
    done = folder / "manifest.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    selected = tn.RUN / "training" / relative / "manifest.json"
    epochs = int(json.loads(selected.read_text(encoding="utf-8"))["best_epoch"])
    parts = parts_for(context, fold)
    ids = list(parts["train"])
    if set(ids) & set(parts["test"]):
        raise RuntimeError("test_row_in_training_rows")
    targets = load_targets(ids)
    transform = tn.fit_transform(context, {"fit": ids}, fold)
    spec = tn.METHODS[treatment]
    model, model_sha, _ = tn.make_model(treatment, transform, seed, torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    cache = tn.ExampleCache(context, transform, treatment, CACHE_GIB * 1024**3)
    node_limit = tn.NODE_BUDGET if spec["graph"] else 0
    folder.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(epochs):
        order = np.random.default_rng(int(tn.base.stable_hash([fold, seed, epoch])[:16], 16)).permutation(len(ids))
        ordered = [ids[int(j)] for j in order]
        losses = [
            tn.batching.loss_step(
                model, optimizer, ordered[start : start + tn.EFFECTIVE_BATCH], spec["internal"], node_limit, cache,
                targets, torch,
            )
            for start in range(0, len(ordered), tn.EFFECTIVE_BATCH)
        ]
        if not all(math.isfinite(value) for value in losses):
            raise RuntimeError(f"nonfinite_loss_epoch_{epoch + 1}")
    model.eval()
    predictions = []
    with torch.no_grad():
        for cid in parts["test"]:
            batch = tn.base._batch_to_device([cache.get(cid)], "cuda:0", spec["graph"])
            value = float(tn.base._prediction(model, batch).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value):
                raise RuntimeError("nonfinite_prediction")
            predictions.append(
                {
                    "canonical_observation_id": cid,
                    "outer_fold": fold,
                    "treatment": treatment,
                    "seed": seed,
                    "predicted_seconds": value,
                    "epochs": epochs,
                }
            )
    atomic_csv(folder / "predictions.csv", predictions)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, folder / "weights.pt")
    result = {
        "evaluation": evaluation,
        "fold": fold,
        "treatment": treatment,
        "seed": seed,
        "epochs": epochs,
        "training_rows": len(ids),
        "test_rows": len(parts["test"]),
        "model_source_sha256": model_sha,
        "selected_epoch_manifest_sha256": sha(selected),
        "torch": torch.__version__,
        "predictions_sha256": sha(folder / "predictions.csv"),
        "weights_sha256": sha(folder / "weights.pt"),
    }
    atomic_json(done, result)
    return result


def collect(output) -> list[dict]:
    """Return one row per observation and cell: the three seed predictions and their median."""
    table = []
    for evaluation, treatments in TREATMENTS.items():
        for treatment in treatments:
            seeds: dict[str, dict[int, str]] = {}
            folds: dict[str, str] = {}
            for fold in range(10 if evaluation == "E3" else 5):
                for seed in tn.SEEDS:
                    path = output / evaluation / f"fold_{fold}" / treatment / f"seed_{seed}" / "predictions.csv"
                    for row in rows(path):
                        cid = row["canonical_observation_id"]
                        seeds.setdefault(cid, {})[seed] = row["predicted_seconds"]
                        folds[cid] = row["outer_fold"]
            for cid in sorted(seeds):
                values = seeds[cid]
                table.append(
                    {
                        "evaluation": evaluation,
                        "treatment": treatment,
                        "canonical_observation_id": cid,
                        "outer_fold": folds[cid],
                        **{f"seed_{seed}": values[seed] for seed in tn.SEEDS},
                        "predicted_seconds": repr(sorted(float(values[seed]) for seed in tn.SEEDS)[1]),
                    }
                )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="directory for the retrained cells")
    parser.add_argument("--cell", action="append", help="train only this cell, written E1:0:M1:42 (repeatable)")
    parser.add_argument("--collect", type=Path, help="write the predictions of all finished cells to this CSV")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.collect:
        atomic_csv(args.collect, collect(output))
        return 0
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required")
    disable_activation_checkpointing()
    cells = all_cells()
    if args.cell:
        wanted = []
        for text in args.cell:
            evaluation, fold, treatment, seed = text.split(":")
            wanted.append((evaluation, int(fold), treatment, int(seed)))
        if not set(wanted) <= set(cells):
            raise SystemExit("unknown cell")
        cells = wanted
    contexts = {}
    for evaluation, fold, treatment, seed in cells:
        if evaluation not in contexts:
            contexts[evaluation] = load_context(evaluation, include_targets=False)[0]
        result = retrain_cell(contexts[evaluation], fold, treatment, seed, output, torch)
        print(json.dumps({"done": [evaluation, fold, treatment, seed], "epochs": result["epochs"]}), flush=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
