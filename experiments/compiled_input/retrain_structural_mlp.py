#!/usr/bin/env python3
"""Retrain the compiled-feature MLP on the whole outer-training partition.

train_structural_mlp.py fits the model on inner folds 1-3 and keeps the best epoch on
inner fold 0. This script trains the same network with the same seed on all four inner
folds for that number of epochs, refits the feature standardization on those rows and
predicts the outer test rows with the final weights. The compiled-feature MLP errors in
the paper come from these models (E1 and E2, 30 cells).

    B26_RUN_DIR=/path/to/compiled_run python3 experiments/compiled_input/retrain_structural_mlp.py --output /path/to/out

B26_RUN_DIR holds the compiled inputs and the training/ folder of train_structural_mlp.py.
Completed cells are skipped. Test labels are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/qpu")]

import numpy as np  # noqa: E402

from core import atomic_csv, atomic_json, load_context, load_targets, parts_for  # noqa: E402

SEEDS = (42, 1234, 31415)
NAMES = [
    "count_rz", "count_sx", "count_x", "count_ecr", "count_cz", "count_measure", "count_reset", "count_delay",
    "count_barrier", "count_other", "count_total", "count_two_qubit", "active_qubits", "depth_1q", "depth_2q",
    "unitary_depth", "hyperq_full_seconds", "hyperq_unitary_seconds",
]


def feature_matrix(run: Path, context) -> tuple[np.ndarray, dict[str, int]]:
    """Return the 121 inputs of every observation: logical and compiled features, descriptors, log shots."""
    mapping = json.loads((run / "physical_mapping.json").read_text())
    with (run / "structural_features_draft.csv").open() as stream:
        draft = {row["canonical_observation_id"]: row for row in csv.DictReader(stream)}
    compiled = {}
    for key in set(mapping.values()):
        with np.load(run / "physical_inputs" / f"{key}.npz", allow_pickle=False) as data:
            compiled[key] = data["compiled_global"].astype(np.float64)
    ids = sorted(context["all_ids"])
    matrix = []
    for cid in ids:
        logical = [float(context["ledger"][cid][f"global_{j:02d}"]) for j in range(51)]
        extra = [np.log1p(float(draft[cid][name])) for name in NAMES]
        shots = np.log1p(float(context["runs_by_id"][cid]["shots"]))
        matrix.append(logical + compiled[mapping[cid]].tolist() + extra + [shots])
    return np.asarray(matrix, dtype=np.float64), {cid: i for i, cid in enumerate(ids)}


def retrain_cell(run: Path, raw, index, context, fold: int, seed: int, output: Path, torch) -> None:
    """Train one cell on the outer-training rows for its selected epochs and save its test predictions."""
    from torch import nn
    from torch_geometric.data import Data

    from benchmark26 import mali_training as base

    evaluation = context["evaluation"]
    folder = output / evaluation / f"fold_{fold}" / f"seed_{seed}"
    if (folder / "predictions.csv").exists():
        return
    selected = run / "training" / evaluation / f"fold_{fold}" / "TM_structural" / f"seed_{seed}" / "complete.json"
    epochs = int(json.loads(selected.read_text())["best_epoch"])
    parts = parts_for(context, fold)
    ids, test = list(parts["train"]), list(parts["test"])
    if set(ids) & set(test):
        raise RuntimeError("test_row_in_training_rows")
    targets = load_targets(ids)
    rows = np.array([index[c] for c in ids])
    x = raw.copy()
    mean, std = x[rows].mean(0), x[rows].std(0, ddof=1)
    keep = std > 1e-6
    x[:, keep] = (x[:, keep] - mean[keep]) / std[keep]
    x[:, ~keep] = 0
    x = x.astype(np.float32)

    def batch(cids):
        values = torch.from_numpy(x[[index[c] for c in cids]]).cuda()
        return Data(global_features=values, x=torch.empty((0, 178), device=values.device),
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=values.device))

    base.seed_all(seed)
    model_class, _ = base.load_pinned_model()
    model = base.make_model(model_class, "mali_full_features_mlp", x.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(int(base.stable_hash([fold, seed, epoch])[:16], 16)).permutation(len(ids))
        for start in range(0, len(order), 32):
            cids = [ids[int(k)] for k in order[start : start + 32]]
            truth = torch.tensor([targets[c] for c in cids], dtype=torch.float32, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch(cids)).reshape(-1), truth)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test), 128):
            cids = test[start : start + 128]
            values = model(batch(cids)).reshape(-1).cpu().numpy()
            predictions += [
                {"canonical_observation_id": c, "evaluation": evaluation, "outer_fold": fold, "seed": seed,
                 "predicted_seconds": float(v)}
                for c, v in zip(cids, values)
            ]
    folder.mkdir(parents=True, exist_ok=True)
    atomic_csv(folder / "predictions.csv", predictions)
    atomic_json(folder / "manifest.json", {"epochs": epochs, "training_rows": len(ids), "test_rows": len(test)})


def collect(output: Path) -> list[dict]:
    """Return one row per observation and evaluation: the three seed predictions and their median."""
    table = []
    for evaluation in ("E1", "E2"):
        seeds: dict[str, dict[int, str]] = {}
        folds: dict[str, str] = {}
        for fold in range(5):
            for seed in SEEDS:
                with (output / evaluation / f"fold_{fold}" / f"seed_{seed}" / "predictions.csv").open() as stream:
                    for row in csv.DictReader(stream):
                        seeds.setdefault(row["canonical_observation_id"], {})[seed] = row["predicted_seconds"]
                        folds[row["canonical_observation_id"]] = row["outer_fold"]
        for cid in sorted(seeds):
            values = seeds[cid]
            table.append({
                "evaluation": evaluation, "treatment": "TM_structural", "canonical_observation_id": cid,
                "outer_fold": folds[cid], **{f"seed_{seed}": values[seed] for seed in SEEDS},
                "predicted_seconds": repr(sorted(float(values[seed]) for seed in SEEDS)[1]),
            })
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="directory for the retrained cells")
    parser.add_argument("--cell", action="append", help="train only this cell, written E1:0:42 (repeatable)")
    parser.add_argument("--collect", type=Path, help="write the predictions of all finished cells to this CSV")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.collect:
        atomic_csv(args.collect, collect(output))
        return 0
    import torch

    torch.set_num_threads(4)
    run = Path(os.environ["B26_RUN_DIR"])
    cells = [(e, f, s) for e in ("E1", "E2") for f in range(5) for s in SEEDS]
    if args.cell:
        cells = [(e, int(f), int(s)) for e, f, s in (text.split(":") for text in args.cell)]
    loaded = {}
    for evaluation, fold, seed in cells:
        if evaluation not in loaded:
            context = load_context(evaluation, include_targets=False)[0]
            loaded[evaluation] = (context, *feature_matrix(run, context))
        context, raw, index = loaded[evaluation]
        retrain_cell(run, raw, index, context, fold, seed, output, torch)
        print(json.dumps({"done": [evaluation, fold, seed]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
