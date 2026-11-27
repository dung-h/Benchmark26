#!/usr/bin/env python3
"""Retrain the graph-ablation models on the whole outer-training partition.

graph_ablation_runner.py fits each model on inner folds 1-3 and keeps the epoch with
the lowest error on inner fold 0. This script trains the same model with the same seed
on all four inner folds for that number of epochs, refits the feature transforms on
those rows and predicts the outer test rows with the final weights. The errors of the
graph model with and without edges in the paper come from these models. MLP, graph and
graph without edges, E1 and E2: 90 cells.

    B26_RUN_DIR=/path/to/finished/graph_ablation_run \
        python3 experiments/qpu/retrain_graph_ablation.py --output /path/to/out

The selected epochs are read from training/ of that run. Completed cells are skipped.
Test labels are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repository root
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import graph_ablation_runner as ga  # noqa: E402
from core import atomic_csv, atomic_json, load_context, load_targets, parts_for  # noqa: E402

TREATMENTS = ("M1", "G1", "G1e")
EVALUATIONS = ("E1", "E2")
CACHE_GIB = 5


def retrain_cell(context, fold: int, treatment: str, seed: int, output: Path, torch) -> None:
    """Train one cell on the outer-training rows for its selected epochs and save its test predictions."""
    evaluation = context["evaluation"]
    relative = Path(evaluation) / f"fold_{fold}" / treatment / f"seed_{seed}"
    folder = output / relative
    if (folder / "predictions.csv").exists():
        return
    selected = Path(os.environ["B26_RUN_DIR"]) / "training" / relative / "manifest.json"
    epochs = int(json.loads(selected.read_text(encoding="utf-8"))["best_epoch"])
    parts = parts_for(context, fold)
    ids, test = list(parts["train"]), list(parts["test"])
    if set(ids) & set(test):
        raise RuntimeError("test_row_in_training_rows")
    targets = load_targets(ids)
    transform = ga.fit_transform(context, {"fit": ids}, fold)
    spec = ga.METHODS[treatment]
    model, _, _ = ga.make_model(treatment, transform, seed, torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    cache = ga.ExampleCache(context, transform, treatment, CACHE_GIB * 1024**3)
    node_limit = ga.NODE_BUDGET if spec["graph"] else 0
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(int(ga.base.stable_hash([fold, seed, epoch])[:16], 16)).permutation(len(ids))
        ordered = [ids[int(j)] for j in order]
        losses = [
            ga.batching.loss_step(model, optimizer, ordered[start : start + ga.EFFECTIVE_BATCH], spec["internal"],
                                  node_limit, cache, targets, torch)
            for start in range(0, len(ordered), ga.EFFECTIVE_BATCH)
        ]
        if not all(math.isfinite(value) for value in losses):
            raise RuntimeError(f"nonfinite_loss_epoch_{epoch + 1}")
    model.eval()
    predictions = []
    with torch.no_grad():
        for cid in test:
            batch = ga.base._batch_to_device([cache.get(cid)], "cuda:0", spec["graph"])
            value = float(ga.base._prediction(model, batch).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value):
                raise RuntimeError("nonfinite_prediction")
            predictions.append({"canonical_observation_id": cid, "outer_fold": fold, "treatment": treatment,
                                "seed": seed, "predicted_seconds": value, "epochs": epochs})
    folder.mkdir(parents=True, exist_ok=True)
    atomic_csv(folder / "predictions.csv", predictions)
    atomic_json(folder / "manifest.json", {"epochs": epochs, "training_rows": len(ids), "test_rows": len(test)})


def collect(output: Path) -> list[dict]:
    """Return one row per observation and cell: the three seed predictions and their median."""
    table = []
    for evaluation in EVALUATIONS:
        for treatment in TREATMENTS:
            seeds: dict[str, dict[int, str]] = {}
            folds: dict[str, str] = {}
            for fold in range(5):
                for seed in ga.SEEDS:
                    path = output / evaluation / f"fold_{fold}" / treatment / f"seed_{seed}" / "predictions.csv"
                    with path.open(newline="", encoding="utf-8") as stream:
                        for row in csv.DictReader(stream):
                            cid = row["canonical_observation_id"]
                            seeds.setdefault(cid, {})[seed] = row["predicted_seconds"]
                            folds[cid] = row["outer_fold"]
            for cid in sorted(seeds):
                values = seeds[cid]
                table.append({
                    "evaluation": evaluation, "treatment": treatment, "canonical_observation_id": cid,
                    "outer_fold": folds[cid], **{f"seed_{seed}": values[seed] for seed in ga.SEEDS},
                    "predicted_seconds": repr(sorted(float(values[seed]) for seed in ga.SEEDS)[1]),
                })
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

    torch.set_num_threads(4)
    cells = [(e, f, t, s) for e in EVALUATIONS for f in range(5) for t in TREATMENTS for s in ga.SEEDS]
    if args.cell:
        cells = [(e, int(f), t, int(s)) for e, f, t, s in (text.split(":") for text in args.cell)]
    contexts = {}
    for evaluation, fold, treatment, seed in cells:
        if evaluation not in contexts:
            contexts[evaluation] = load_context(evaluation, include_targets=False)[0]
        retrain_cell(contexts[evaluation], fold, treatment, seed, output, torch)
        print(json.dumps({"done": [evaluation, fold, treatment, seed]}), flush=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
