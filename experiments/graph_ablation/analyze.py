#!/usr/bin/env python3
"""Aggregate the graph-ablation predictions with the prespecified metric and contrasts.

Reads labels only after all 120 cells are complete and the baseline safety check
passed. The estimand is fixed by ``PROTOCOL.md``: each observation's prediction
is the median over the three seeds; GB-MAE is the mean over the 203 groups of
each group's mean absolute error. Contrasts are paired over groups and use
10,000 whole-group bootstrap draws (seed 42) resampled within source strata,
with Bonferroni-adjusted two-sided 99.1667% intervals (six contrasts).

The run was analyzed by the locked script whose SHA-256 is
``LOCKED_ANALYSIS_SHA256``. That script differs from this one only in how it
found the bundle (a fixed local path) and where it wrote; the computations,
random-number stream and output fields are the same. ``--check`` compares the
result with a saved summary.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.paths import ARTIFACTS, RUN_DIR

STUDY = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
ARMS = ("M1", "G1", "G1e", "G1p")
SEEDS = (42, 1234, 31415)
EVALUATIONS = ("E1", "E2")
FOLDS = 5
CONTRASTS = (("G1", "G1e"), ("G1", "G1p"), ("G1e", "M1"))
DRAWS = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05 / 6
LOCKED_ANALYSIS_SHA256 = "8d48c0a4e1fced2cd81cb92653e152f73109a21c362fb8cdd894a0ee338e9e65"


def sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_errors(predictions: dict[str, dict[int, float]], truth: dict[str, float], groups: dict[str, list[str]]):
    """Per-group MAE of the median-of-seeds prediction, and per-seed GB-MAE."""
    median_error = {}
    per_seed = {seed: [] for seed in SEEDS}
    for group, ids in groups.items():
        median_error[group] = float(
            np.mean([abs(float(np.median([predictions[c][s] for s in SEEDS])) - truth[c]) for c in ids])
        )
        for seed in SEEDS:
            per_seed[seed].append(float(np.mean([abs(predictions[c][seed] - truth[c]) for c in ids])))
    return median_error, {str(seed): float(np.mean(values)) for seed, values in per_seed.items()}


def stratified_draws(names: list[str], source: dict[str, str], draws: int = DRAWS, seed: int = BOOTSTRAP_SEED):
    """Group indices resampled with replacement within each source, keeping each source's group count."""
    strata = {s: np.array([i for i, g in enumerate(names) if source[g] == s]) for s in sorted(set(source.values()))}
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.choice(index, size=(draws, len(index)), replace=True) for index in strata.values()], axis=1
    )


def paired_contrast(treatment: dict[str, float], control: dict[str, float], names: list[str], draws) -> dict:
    """Return the mean paired difference of group errors and its Bonferroni-adjusted bootstrap interval."""
    delta = np.array([treatment[g] - control[g] for g in names])
    sample = delta[draws].mean(axis=1)
    low, high = np.quantile(sample, [ALPHA / 2, 1 - ALPHA / 2])
    return {"delta": float(delta.mean()), "lower": float(low), "upper": float(high)}


def analyze(run: Path) -> dict:
    """Aggregate the 120 completed cells into metrics, contrasts and per-group errors.

    Requires complete training, a passed baseline check and the analysis lock. For E1 and
    E2 it takes the median of the three seed predictions per observation, averages the
    absolute error within groups and over groups, and computes the three paired contrasts.
    Returns the per-group rows and the summary.
    """
    progress = json.loads((run / "training/run_progress.json").read_text())
    if progress["status"] != "COMPLETE_120_OF_120" or progress["completed_cells"] != 120:
        raise RuntimeError("training_incomplete")
    sanity = json.loads((run / "training/baseline_sanity.json").read_text())
    if sanity["status"] != "PASS":
        raise RuntimeError("baseline_sanity_not_passed")
    lock = json.loads((run / "analysis_lock.json").read_text())
    if (
        lock["analysis_sha256"] != LOCKED_ANALYSIS_SHA256
        or sha(run / "ablation_lock.json") != lock["ablation_lock_sha256"]
    ):
        raise RuntimeError("analysis_lock_mismatch")

    panel = {r["canonical_observation_id"]: r for r in read(STUDY / "cohort/panel.csv")}
    truth = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in read(STUDY / "cohort/targets.csv")}
    if set(panel) != set(truth) or len(panel) != 3443:
        raise RuntimeError("cohort_mismatch")

    metrics, contrasts, group_rows = [], [], []
    for evaluation in EVALUATIONS:
        groups, source = collections.defaultdict(list), {}
        for cid, row in panel.items():
            group = row["group_id"]
            groups[group].append(cid)
            if source.setdefault(group, row["source_id"]) != row["source_id"]:
                raise RuntimeError(f"group_spans_sources:{group}")
        names = sorted(groups)
        if len(names) != 203:
            raise RuntimeError("group_count_mismatch")
        errors = {}
        for arm in ARMS:
            predictions = collections.defaultdict(dict)
            for fold in range(FOLDS):
                for seed in SEEDS:
                    path = run / "training" / evaluation / f"fold_{fold}" / arm / f"seed_{seed}" / "predictions.csv"
                    for row in read(path):
                        cid = row["canonical_observation_id"]
                        if seed in predictions[cid] or int(row["outer_fold"]) != fold:
                            raise RuntimeError(f"duplicate_or_misplaced_prediction:{evaluation}:{arm}:{cid}")
                        predictions[cid][seed] = float(row["predicted_seconds"])
            if set(predictions) != set(panel) or any(set(v) != set(SEEDS) for v in predictions.values()):
                raise RuntimeError(f"prediction_coverage:{evaluation}:{arm}")
            errors[arm], per_seed = group_errors(predictions, truth, groups)
            for scope in ["all"] + sorted(set(source.values())):
                chosen = [g for g in names if scope == "all" or source[g] == scope]
                metrics.append(
                    {
                        "evaluation": evaluation,
                        "arm": arm,
                        "scope": scope,
                        "groups": len(chosen),
                        "rows": sum(len(groups[g]) for g in chosen),
                        "group_balanced_mae_seconds": float(np.mean([errors[arm][g] for g in chosen])),
                        "seed_group_balanced_mae_seconds": per_seed if scope == "all" else None,
                    }
                )
        for index, group in enumerate(names):
            group_rows.append(
                {
                    "evaluation": evaluation,
                    "group": f"g{index:03d}",
                    "source": source[group],
                    "observations": len(groups[group]),
                    **{f"mae_{arm}": errors[arm][group] for arm in ARMS},
                }
            )
        draws = stratified_draws(names, source)
        for treatment, control in CONTRASTS:
            result = paired_contrast(errors[treatment], errors[control], names, draws)
            contrasts.append(
                {
                    "evaluation": evaluation,
                    "contrast": f"{treatment}-{control}",
                    "delta_group_balanced_mae_seconds": result["delta"],
                    "adjusted_confidence_level": 1 - ALPHA,
                    "lower": result["lower"],
                    "upper": result["upper"],
                    "bootstrap_draws": DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    return group_rows, {
        "status": "COMPLETE_DESCRIPTIVE_AND_PREDECLARED_CONTRASTS",
        "scope": "fixed fitted predictions; bootstrap does not include retraining variation",
        "metrics": metrics,
        "contrasts": contrasts,
        "baseline_sanity": sanity,
    }


def main() -> int:
    """Run the analysis and, as requested, write the summary, export per-group errors, or compare with a saved summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=RUN_DIR, help="run directory (default: $B26_RUN_DIR)")
    parser.add_argument("--output", type=Path, help="write the summary here (refuses to overwrite)")
    parser.add_argument("--check", type=Path, help="compare with a saved summary; exit 1 on any difference")
    parser.add_argument(
        "--export-groups",
        type=Path,
        help="write anonymous per-group errors (groups numbered in the resampling order) for verify_summary.py",
    )
    args = parser.parse_args()
    if args.run is None:
        parser.error("set B26_RUN_DIR or pass --run")
    group_rows, summary = analyze(args.run)
    if args.export_groups:
        if args.export_groups.exists():
            raise SystemExit(f"refusing to overwrite {args.export_groups}")
        args.export_groups.parent.mkdir(parents=True, exist_ok=True)
        with args.export_groups.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(group_rows)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.check:
        saved = json.loads(args.check.read_text())
        same = saved == summary
        print(json.dumps({"identical_to_saved_summary": same}))
        return 0 if same else 1
    if not args.output:
        print(json.dumps(summary["contrasts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
