#!/usr/bin/env python3
"""Offline final aggregation of immutable saved predictions. Never trains."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import record_name
import argparse
import collections
import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from core import OUT, RUN, ROOT, rows, sha, stable, atomic_json, atomic_csv, load_context, parts_for
from analysis_core import aligned, metrics, group_errors, bootstrap_draws, paired_intervals, CONTRASTS

METHODS = ("M0", "M1", "G0", "G1", "R0", "C0")
SEEDS = (42, 1234, 31415)
NAMES = {
    "M0": "MLP",
    "M1": "MLP + shots",
    "G0": "Graph+MLP",
    "G1": "Graph+MLP + shots",
    "R0": "Selected classical pipeline",
    "C0": "Outer-training median",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=RUN / "analysis_v1")
    ap.add_argument("--validation", type=Path, default=RUN / "validation/final_v2/receipt.json")
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError("Analysis output exists; refuse overwrite")
    receipt = json.loads(args.validation.read_text())
    assert receipt["status"] == "PASS" and receipt["completed_neural_cells"] == 240 and receipt["classical_cases"] == 20
    assert receipt["prediction_replay_rows"] == 123948
    dest = args.output
    dest.mkdir(parents=True)
    panel = rows(OUT / "cohort/panel.csv")
    ids = sorted(r["canonical_observation_id"] for r in panel)
    panelmap = {r["canonical_observation_id"]: r for r in panel}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in rows(OUT / "cohort/targets.csv")}
    assert len(panelmap) == len(ids) == len(targets) == 3443 and set(ids) == set(targets)
    assert len({r["group_id"] for r in panel}) == 203
    source_counts = dict(collections.Counter(r["source_id"] for r in panel))
    assert sorted(source_counts.values()) == [45, 340, 3058]
    assert all(np.isfinite(v) and v > 0 for v in targets.values())
    y = np.array([targets[c] for c in ids])
    groups = np.array([panelmap[c]["group_id"] for c in ids])
    sources = np.array([panelmap[c]["source_id"] for c in ids])
    backends = np.array([panelmap[c]["backend"] for c in ids])
    pooled = []
    sliced = []
    seed_metrics = []
    reduced = []
    raw_seed = []
    gm = []
    comparisons = []
    fold_deltas = []
    tails = []
    provenance = []
    for ev in ("E1", "E2", "E3"):
        context, _, _ = load_context(ev, include_targets=False)
        collection = {m: [] for m in METHODS}
        byseed = {(m, s): [] for m in METHODS[:4] for s in SEEDS}
        foldbyid = {}
        for fold in range(context["case_count"]):
            part = parts_for(context, fold)
            foldbyid.update({c: fold for c in part["test"]})
            for m in METHODS[:4]:
                records = []
                for seed in SEEDS:
                    path = RUN / "training" / ev / f"fold_{fold}" / m / f"seed_{seed}" / "predictions.csv"
                    current = aligned(rows(path), part["test"])
                    records.append(current)
                    byseed[m, seed].extend(current)
                    provenance.append({"path": record_name(path), "sha256": sha(path)})
                    raw_seed.extend(
                        {
                            **r,
                            "evaluation": ev,
                            "source_id": panelmap[r["canonical_observation_id"]]["source_id"],
                            "group_id": panelmap[r["canonical_observation_id"]]["group_id"],
                            "target_seconds": targets[r["canonical_observation_id"]],
                        }
                        for r in current
                    )
                for i, c in enumerate(part["test"]):
                    pv = [float(records[j][i]["predicted_seconds"]) for j in range(3)]
                    collection[m].append(
                        {
                            "canonical_observation_id": c,
                            "predicted_seconds": float(np.median(pv)),
                            "outer_fold": fold,
                            "seed_min_seconds": min(pv),
                            "seed_max_seconds": max(pv),
                            "seed_reduction": "median_of_three_predictions",
                        }
                    )
            for m, filename in [("R0", "regression_predictions.csv"), ("C0", "constant_predictions.csv")]:
                path = RUN / "classical" / ev / f"fold_{fold}" / filename
                records = aligned(rows(path), part["test"])
                if m == "C0":
                    expected = float(np.median([targets[c] for c in part["train"]]))
                    assert all(float(r["predicted_seconds"]) == expected for r in records)
                collection[m].extend(records)
                provenance.append({"path": record_name(path), "sha256": sha(path)})
        ps = {}
        for m in METHODS:
            rec = aligned(collection[m], ids)
            p = np.array([float(r["predicted_seconds"]) for r in rec])
            ps[m] = p
            pooled.append({"evaluation": ev, "treatment": m, "method": NAMES[m], **metrics(y, p, groups, sources)})
            for c, r in zip(ids, rec):
                reduced.append(
                    {
                        "evaluation": ev,
                        "treatment": m,
                        "canonical_observation_id": c,
                        "outer_fold": foldbyid[c],
                        "source_id": panelmap[c]["source_id"],
                        "group_id": panelmap[c]["group_id"],
                        "backend": panelmap[c]["backend"],
                        "shots": panelmap[c]["shots"],
                        "target_seconds": targets[c],
                        "predicted_seconds": float(r["predicted_seconds"]),
                        "signed_error_seconds": float(r["predicted_seconds"]) - targets[c],
                        "absolute_error_seconds": abs(float(r["predicted_seconds"]) - targets[c]),
                        "seed_reduction": "median_of_three_predictions"
                        if m in METHODS[:4]
                        else "single_deterministic_pipeline",
                        "seed_min_seconds": r.get("seed_min_seconds", ""),
                        "seed_max_seconds": r.get("seed_max_seconds", ""),
                    }
                )
            for scope, arr in [
                ("source", sources),
                ("backend", backends),
                ("outer_fold", np.array([str(foldbyid[c]) for c in ids])),
            ]:
                for key in sorted(set(arr)):
                    ix = arr == key
                    sliced.append(
                        {
                            "evaluation": ev,
                            "treatment": m,
                            "scope": scope,
                            "scope_value": key,
                            **metrics(y[ix], p[ix], groups[ix], sources[ix]),
                        }
                    )
            for ix in np.argsort(-np.abs(p - y), kind="stable")[:25]:
                tails.append(
                    {
                        "evaluation": ev,
                        "treatment": m,
                        "canonical_observation_id": ids[ix],
                        "source_id": sources[ix],
                        "group_id": groups[ix],
                        "backend": backends[ix],
                        "target_seconds": y[ix],
                        "predicted_seconds": p[ix],
                        "signed_error_seconds": p[ix] - y[ix],
                        "absolute_error_seconds": abs(p[ix] - y[ix]),
                    }
                )
        for (m, seed), records in byseed.items():
            p = np.array([float(r["predicted_seconds"]) for r in aligned(records, ids)])
            seed_metrics.append({"evaluation": ev, "treatment": m, "seed": seed, **metrics(y, p, groups, sources)})
            for fold in range(context["case_count"]):
                ix = np.array([foldbyid[c] == fold for c in ids])
                seed_metrics.append(
                    {
                        "evaluation": ev,
                        "treatment": m,
                        "seed": seed,
                        "outer_fold": fold,
                        **metrics(y[ix], p[ix], groups[ix], sources[ix]),
                    }
                )
        order, gs, values = group_errors(y, ps, groups, sources)
        for i, g in enumerate(order):
            for j, m in enumerate(METHODS):
                gm.append(
                    {
                        "evaluation": ev,
                        "treatment": m,
                        "group_id": g,
                        "source_id": gs[i],
                        "rows": int((groups == g).sum()),
                        "mae_seconds": values[i, j],
                    }
                )
        scopes = (
            [("all", np.ones(len(y), dtype=bool))]
            if ev != "E3"
            else [(b, backends == b) for b in sorted(set(backends))]
        )
        for scope, ix in scopes:
            subset_ps = {m: p[ix] for m, p in ps.items()}
            go, ss, gvals = group_errors(y[ix], subset_ps, groups[ix], sources[ix])
            draws = bootstrap_draws(ss, ev != "E3")
            drawpath = dest / "bootstrap" / f"{ev}_{scope}.npz"
            drawpath.parent.mkdir(exist_ok=True)
            np.savez_compressed(drawpath, draws=draws, groups=np.asarray(go), sources=np.asarray(ss))
            for row in paired_intervals(gvals, list(METHODS), draws, 0.99375 if ev != "E3" else 0.99875):
                comparisons.append(
                    {
                        "evaluation": ev,
                        "scope": scope,
                        "groups": len(go),
                        "rows": int(ix.sum()),
                        "source_stratified": ev != "E3",
                        "bootstrap_seed": 42,
                        "inference_limitation": "Single group: bootstrap cannot estimate between-group uncertainty"
                        if len(go) < 2
                        else "",
                        "bootstrap_draws_sha256": sha(drawpath),
                        **row,
                    }
                )
        for fold in range(context["case_count"]):
            ix = np.array([foldbyid[c] == fold for c in ids])
            for a, b in CONTRASTS:
                ma = metrics(y[ix], ps[a][ix], groups[ix], sources[ix])
                mb = metrics(y[ix], ps[b][ix], groups[ix], sources[ix])
                fold_deltas.append(
                    {
                        "evaluation": ev,
                        "outer_fold": fold,
                        "heldout_backend": context["cases"][fold] if ev == "E3" else "",
                        "contrast": f"{a}-{b}",
                        "rows": int(ix.sum()),
                        "groups": ma["groups"],
                        "delta_group_balanced_mae_seconds": ma["group_balanced_mae_seconds"]
                        - mb["group_balanced_mae_seconds"],
                        "delta_pooled_mae_seconds": ma["pooled_mae_seconds"] - mb["pooled_mae_seconds"],
                    }
                )
    assert len(raw_seed) == 123948 and len(reduced) == 61974 and len(comparisons) == 48
    for name, data in [
        ("pooled_metrics", pooled),
        ("sliced_metrics", sliced),
        ("seed_metrics", seed_metrics),
        ("reduced_predictions", reduced),
        ("per_seed_predictions", raw_seed),
        ("group_metrics", gm),
        ("paired_comparisons", comparisons),
        ("fold_differences", fold_deltas),
        ("tail_errors", tails),
        ("input_prediction_hashes", provenance),
    ]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        atomic_csv(dest / f"{name}.csv", data, fields)
    summary = {
        "status": "AGGREGATED_PENDING_INDEPENDENT_REPLAY",
        "cohort_rows": 3443,
        "groups": 203,
        "source_counts": source_counts,
        "neural_cells": 240,
        "classical_cases": 20,
        "per_seed_neural_rows": len(raw_seed),
        "reduced_rows": len(reduced),
        "prespecified_comparisons": 48,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "primary_metric": "group_balanced_mae_seconds",
        "neural_seed_reduction": "median_of_three_predictions",
        "no_training_refit_or_simulation": True,
        "no_prediction_clipping_or_tail_removal": True,
        "checkpoint_validation_receipt": record_name(args.validation),
        "checkpoint_validation_sha256": sha(args.validation),
        "aggregator_sha256": sha(Path(__file__)),
        "analysis_core_sha256": sha(Path(__file__).with_name("analysis_core.py")),
        "target_sha256": sha(OUT / "cohort/targets.csv"),
        "panel_sha256": sha(OUT / "cohort/panel.csv"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json(dest / "aggregation_receipt.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
