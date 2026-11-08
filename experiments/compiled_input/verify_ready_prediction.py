#!/usr/bin/env python3
"""Recompute the device-ready timing summary and intervals from anonymous group means."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results/compiled_input/ready_prediction_latency"
ARMS = ("TM_structural", "TG_structural", "TGF_structural")


def rows(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    means = rows("group_means.csv")
    checks = 0
    for row in rows("summary.csv"):
        chosen = [
            r for r in means
            if r["evaluation"] == row["evaluation"] and r["arm"] == row["arm"]
            and (row["source"] == "all" or r["source"] == row["source"])
        ]
        values = np.array([float(r["mean_ms"]) for r in chosen])
        weights = np.array([int(r["observations"]) for r in chosen])
        assert len(chosen) == int(row["groups"]) and weights.sum() == int(row["observations"])
        assert abs(values.mean() - float(row["group_balanced_mean_ms"])) < 1e-9
        assert abs(np.average(values, weights=weights) - float(row["pooled_mean_ms"])) < 1e-9
        checks += 2
    source = {int(r["group_index"]): r["source"] for r in means}
    groups = sorted(source)
    table = {(r["evaluation"], r["arm"], int(r["group_index"])): float(r["mean_ms"]) for r in means}
    contrasts = rows("contrasts.csv")
    for evaluation in ("E1", "E2"):
        rng = np.random.default_rng(42)
        picks = []
        for name in sorted(set(source.values())):
            members = np.array([g for g in groups if source[g] == name])
            picks.append(members[rng.integers(0, len(members), size=(10000, len(members)))])
        pick = np.concatenate(picks, axis=1)
        for row in (r for r in contrasts if r["evaluation"] == evaluation):
            delta = np.zeros(len(groups))
            delta[groups] = [table[evaluation, row["first"], g] - table[evaluation, row["second"], g] for g in groups]
            draws = delta[pick].mean(axis=1)
            alpha = 1 - float(row["ci_level"])
            low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
            assert abs(delta.mean() - float(row["difference_ms"])) < 1e-9
            assert abs(low - float(row["ci_low_ms"])) < 1e-9 and abs(high - float(row["ci_high_ms"])) < 1e-9
            checks += 3
    print(json.dumps({"status": "PASS", "checks": checks, "scope": "Group summaries and paired intervals from anonymous group means; does not repeat the timing."}))


if __name__ == "__main__":
    main()
