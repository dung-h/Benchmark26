#!/usr/bin/env python3
"""Independent consistency checks for completed inference-cost summaries."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/postfreeze_qpu_workflow_mean_cost_v1/v2"
SRC = ROOT / "artifacts/postfreeze_qpu_workflow_mean_expansion_v1"
OUT = ART / "summary"
METHODS = {"M0", "M1", "G0", "G1", "R0", "C0"}


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    checks = {}
    receipt = json.loads((OUT / "aggregation_receipt.json").read_text())
    assert receipt["status"] == "PASS_AGGREGATION"
    fingerprint = json.loads((ART / "timing_manifest.json").read_text())["fingerprint"]
    assert receipt["timing_fingerprint"] == fingerprint
    checks["fingerprint_matches_locked_manifest"] = True

    coverage = read("coverage_matrix.csv")
    assert len(coverage) == 72
    assert all(r["status"] == "PASS" and int(r["assigned_rows"]) == 3443 and int(r["observed_rows"]) == 3443 for r in coverage)
    checks["72_method_clock_session_cells_complete"] = True

    raw = read("row_level_latency.csv")
    assert len(raw) == 3443 * 6 * 3 * 2 * 2
    raw_keys = [(r["evaluation"], r["clock"], r["session"], r["canonical_observation_id"], r["method"]) for r in raw]
    assert len(raw_keys) == len(set(raw_keys))
    assert {r["method"] for r in raw} == METHODS
    assert all(np.isfinite(float(r["median_latency_ms_10_reps"])) and float(r["median_latency_ms_10_reps"]) >= 0 for r in raw)
    checks["row_level_latency_keys_and_values_valid"] = True

    summary = read("latency_summary.csv")
    assert len(summary) == 144
    comparison = read("paired_comparisons.csv")
    assert len(comparison) == 72
    draw_data = np.load(OUT / "bootstrap_draws.npz")
    assert len(draw_data.files) == 72
    for row in comparison:
        key = f"{row['evaluation']}_{row['scope']}_{row['contrast'].split('-')[0]}_minus_{row['contrast'].split('-')[1]}"
        arr = draw_data[key]
        n = int(row["bootstrap_replicates"])
        assert len(arr) == n
        if n:
            alpha = float(row["per_comparison_alpha"])
            assert abs(float(np.quantile(arr, alpha / 2)) - float(row["ci_low_adjusted_ms"])) < 1e-9
            assert abs(float(np.quantile(arr, 1 - alpha / 2)) - float(row["ci_high_adjusted_ms"])) < 1e-9
    checks["paired_intervals_replay_from_saved_group_bootstrap_draws"] = True

    acc = read("saved_accuracy_metrics.csv")
    pooled = {(r["evaluation"], r["method"]): r for r in acc if r["scope"] == "all"}
    saved = {(r["evaluation"], r["treatment"]): r for r in csv.DictReader(open(SRC / "analysis_v1/pooled_metrics.csv", newline="", encoding="utf-8"))}
    assert len(pooled) == 18
    for key, row in pooled.items():
        source = saved[key]
        assert row["group_balanced_mae_seconds"] == source["group_balanced_mae_seconds"]
        assert row["raw_seconds_r2"] == source["raw_seconds_r2"]
    checks["accuracy_values_match_saved_source_artifact_exactly"] = True

    resources = read("resource_summary.csv")
    assert len(resources) == 12
    assert all(int(r["observations"]) == 3443 for r in resources)
    checks["resource_rows_cover_all_request_records"] = True

    out = {"status": "PASS_INDEPENDENT_SUMMARY_VALIDATION", "timing_fingerprint": fingerprint,
           "checks": checks, "row_level_latency_rows": len(raw),
           "coverage_cells": len(coverage), "paired_comparison_rows": len(comparison),
           "accuracy_source": "saved expansion pooled_metrics.csv and sliced_metrics.csv",
           "no_model_retraining_or_metric_recomputation": True}
    (OUT / "validation_receipt.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
