#!/usr/bin/env python3
"""Aggregate locked, target-blind inference timings after both sessions finish.

This script deliberately reads only timing receipts/records plus saved accuracy
summary CSVs. It never fits or scores a model against runtime labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/postfreeze_qpu_workflow_mean_cost_v1/v2"
SRC = ROOT / "artifacts/postfreeze_qpu_workflow_mean_expansion_v1"
METHODS = ("M0", "M1", "G0", "G1", "R0", "C0")
CONTRASTS = (("G0", "M0"), ("G1", "M1"), ("M1", "M0"),
             ("G1", "G0"), ("R0", "M1"), ("R0", "G1"))
SEED = 42
N_BOOT = 10_000


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def expected_ids(evaluation, case):
    if evaluation in ("E1", "E2"):
        path = SRC / f"splits/{evaluation}_outer.csv"
        return {r["canonical_observation_id"] for r in read_csv(path)
                if int(r["outer_fold"]) == int(case)}
    path = SRC / "splits/E3_outer.csv"
    backends = sorted({r["heldout_backend"] for r in read_csv(path)})
    backend = backends[int(case)]
    return {r["canonical_observation_id"] for r in read_csv(path)
            if r["heldout_backend"] == backend and r["partition"] == "test"}


def collect(evaluation, clock, session):
    base = ART / "measure" / evaluation / f"clock_{clock}" / f"session_{session}"
    receipt_path = base / "session_{}_receipt.json".format(session)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS_TIMING_SESSION":
        raise RuntimeError(f"session_not_pass:{receipt_path}")
    if receipt.get("completed_units") != 3443 or receipt.get("assigned_units") != 3443:
        raise RuntimeError(f"session_coverage_not_3443:{receipt_path}")
    out = {}
    for case in range(5 if evaluation in ("E1", "E2") else 10):
        case_dir = base / f"case_{case}"
        files = list(case_dir.glob("*.json"))
        expected = expected_ids(evaluation, case)
        observed_ids = set()
        for file in files:
            d = json.loads(file.read_text())
            if d.get("fingerprint") != receipt.get("fingerprint"):
                raise RuntimeError(f"fingerprint_mismatch:{file}")
            cid = d["canonical_observation_id"]
            if cid in observed_ids:
                raise RuntimeError(f"duplicate_observation:{evaluation}:{clock}:{session}:{case}:{cid}")
            observed_ids.add(cid)
            timed = [r for r in d["records"] if r.get("sample_type") == "timed"]
            for method in METHODS:
                vals = [r["elapsed_ns"] for r in timed if r["method"] == method]
                if len(vals) != 10 or any(v < 0 for v in vals):
                    raise RuntimeError(f"timed_rep_count:{file}:{method}:{len(vals)}")
                if any(r.get("parity") != "PASS" for r in timed if r["method"] == method):
                    raise RuntimeError(f"parity_failure:{file}:{method}")
                out[(cid, method)] = statistics.median(vals) / 1e6
        if observed_ids != expected:
            raise RuntimeError(f"case_coverage_mismatch:{evaluation}:{case}:got={len(observed_ids)}:expected={len(expected)}")
    if len(out) != 3443 * len(METHODS):
        raise RuntimeError(f"method_row_coverage:{len(out)}")
    return out


def paired_group_values(la, lb, panel, evaluation, case=None):
    subset = {cid for cid in panel if case is None or panel[cid]["case"] == case}
    by_group = defaultdict(list)
    for cid in subset:
        by_group[panel[cid]["group_id"]].append(la[cid] - lb[cid])
    grp_delta = {g: float(np.mean(v)) for g, v in by_group.items()}
    grp_source = {
        g: "+".join(sorted({panel[cid]["source_id"] for cid in subset
                            if panel[cid]["group_id"] == g}))
        for g in grp_delta
    }
    return grp_delta, grp_source


def bootstrap_ci(grp_delta, grp_source=None, draws=N_BOOT, alpha=0.05, seed=SEED):
    groups = sorted(grp_delta)
    if len(groups) < 2:
        return None, None, 0, np.asarray([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    if grp_source is None:
        vals = np.asarray([grp_delta[g] for g in groups])
        idx = rng.integers(0, len(groups), size=(draws, len(groups)))
        samples[:] = vals[idx].mean(axis=1)
    else:
        strata = defaultdict(list)
        for g in groups:
            strata[grp_source[g]].append(g)
        accum = np.zeros(draws, dtype=np.float64)
        total = 0
        for source in sorted(strata):
            gs = strata[source]
            vals = np.asarray([grp_delta[g] for g in gs])
            idx = rng.integers(0, len(gs), size=(draws, len(gs)))
            accum += vals[idx].sum(axis=1)
            total += len(gs)
        samples[:] = accum / total
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2)), draws, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ART / "summary"))
    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    panel_rows = read_csv(SRC / "cohort/panel.csv")
    panel = {r["canonical_observation_id"]: {**r, "case": ""} for r in panel_rows}
    timings = {}
    for ev in ("E1", "E2", "E3"):
        if ev in ("E1", "E2"):
            cases = range(5)
            assignments = {r["canonical_observation_id"]: int(r["outer_fold"])
                           for r in read_csv(SRC / f"splits/{ev}_outer.csv")}
        else:
            split = read_csv(SRC / "splits/E3_outer.csv")
            be = sorted({r["heldout_backend"] for r in split})
            assignments = {r["canonical_observation_id"]: be.index(r["heldout_backend"])
                           for r in split if r["partition"] == "test"}
            cases = range(10)
        for cid, case in assignments.items():
            panel[cid]["case"] = int(case)
        for clock in ("A", "B"):
            for session in (1, 2):
                timings[(ev, clock, session)] = collect(ev, clock, session)

    # Save row-level session medians and paired two-session aggregate.
    raw_rows, paired = [], {}
    for ev in ("E1", "E2", "E3"):
        ids = sorted({cid for cid, _ in timings[(ev, "A", 1)]})
        for clock in ("A", "B"):
            for cid in ids:
                for method in METHODS:
                    a = timings[(ev, clock, 1)][(cid, method)]
                    b = timings[(ev, clock, 2)][(cid, method)]
                    med = statistics.median((a, b))
                    paired[(ev, clock, cid, method)] = med
                    for session, value in ((1, a), (2, b)):
                        raw_rows.append({"evaluation": ev, "clock": clock, "session": session,
                                         "canonical_observation_id": cid, "group_id": panel[cid]["group_id"],
                                         "source_id": panel[cid]["source_id"], "backend": panel[cid]["backend"],
                                         "method": method, "median_latency_ms_10_reps": value,
                                         "two_session_median_latency_ms": med})
    write_csv(outdir / "row_level_latency.csv", raw_rows, list(raw_rows[0]))

    drift_rows = []
    for ev in ("E1", "E2", "E3"):
        ids = sorted({cid for cid, _ in timings[(ev, "A", 1)]})
        for clock in ("A", "B"):
            for method in METHODS:
                x = np.asarray([timings[(ev, clock, 1)][(cid, method)] for cid in ids], dtype=np.float64)
                y = np.asarray([timings[(ev, clock, 2)][(cid, method)] for cid in ids], dtype=np.float64)
                denom = np.maximum((x + y) / 2, np.finfo(float).tiny)
                rel = np.abs(y - x) / denom
                corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
                drift_rows.append({"evaluation": ev, "clock": clock, "method": method,
                                   "rows": len(ids), "session1_median_ms": float(np.median(x)),
                                   "session2_median_ms": float(np.median(y)),
                                   "median_absolute_relative_session_change": float(np.median(rel)),
                                   "p90_absolute_relative_session_change": float(np.quantile(rel, .90)),
                                   "pearson_session_latency_correlation_descriptive": corr})
    write_csv(outdir / "session_drift_summary.csv", drift_rows, list(drift_rows[0]))

    # Descriptive summaries and paired structural-group bootstrap contrasts.
    summary = []
    comparisons = []
    bootstrap_draws = {}
    for ev in ("E1", "E2", "E3"):
        all_ids = sorted({cid for cid, _ in timings[(ev, "A", 1)]})
        heldout_cases = sorted({panel[cid]["case"] for cid in all_ids})
        scopes = [("ALL", all_ids)] if ev != "E3" else [(str(c), [cid for cid in all_ids if panel[cid]["case"] == c]) for c in heldout_cases]
        for clock in ("A", "B"):
            for scope, ids in scopes:
                for method in METHODS:
                    vals = [paired[(ev, clock, cid, method)] for cid in ids]
                    groups = defaultdict(list)
                    for cid in ids:
                        groups[panel[cid]["group_id"]].append(paired[(ev, clock, cid, method)])
                    gb = float(np.mean([np.mean(v) for v in groups.values()]))
                    summary.append({"evaluation": ev, "scope": scope, "clock": clock,
                                    "method": method, "rows": len(ids), "groups": len(groups),
                                    "mean_latency_ms": float(np.mean(vals)),
                                    "median_latency_ms": float(np.median(vals)),
                                    "p90_latency_ms": float(np.quantile(vals, .90)),
                                    "p99_latency_ms": float(np.quantile(vals, .99)),
                                    "group_balanced_mean_latency_ms": gb})
                if clock == "B":
                    for left, right in CONTRASTS:
                        la = {cid: paired[(ev, clock, cid, left)] for cid in ids}
                        lb = {cid: paired[(ev, clock, cid, right)] for cid in ids}
                        grp, src = paired_group_values(la, lb, panel, ev,
                            None if ev != "E3" else int(scope))
                        delta = float(np.mean(list(grp.values()))) if grp else math.nan
                        if ev == "E3":
                            adjusted_alpha = 0.05 / 60
                            bootstrap_seed = SEED + heldout_cases.index(int(scope)) * 100 + len(comparisons)
                            lo, hi, n, samples = bootstrap_ci(grp, draws=N_BOOT, alpha=adjusted_alpha,
                                                              seed=bootstrap_seed)
                        else:
                            adjusted_alpha = 0.05 / 12
                            bootstrap_seed = SEED + len(comparisons)
                            lo, hi, n, samples = bootstrap_ci(grp, src, draws=N_BOOT,
                                                              alpha=adjusted_alpha,
                                                              seed=bootstrap_seed)
                        bootstrap_draws[f"{ev}_{scope}_{left}_minus_{right}"] = samples
                        comparisons.append({"evaluation": ev, "scope": scope, "clock": clock,
                                            "contrast": f"{left}-{right}", "delta_group_balanced_ms": delta,
                                            "ci_low_adjusted_ms": lo, "ci_high_adjusted_ms": hi,
                                            "per_comparison_alpha": adjusted_alpha,
                                            "bootstrap_replicates": n,
                                            "resampling_unit": "whole structural group",
                                            "bootstrap_seed": bootstrap_seed})
    write_csv(outdir / "latency_summary.csv", summary, list(summary[0]))
    write_csv(outdir / "paired_comparisons.csv", comparisons, list(comparisons[0]))

    coverage_rows = []
    for ev in ("E1", "E2", "E3"):
        ncases = 5 if ev in ("E1", "E2") else 10
        for clock in ("A", "B"):
            for session in (1, 2):
                receipt = json.loads((ART / "measure" / ev / f"clock_{clock}" /
                                      f"session_{session}" / f"session_{session}_receipt.json").read_text())
                for method in METHODS:
                    coverage_rows.append({"evaluation": ev, "clock": clock, "session": session,
                                          "method": method, "assigned_rows": 3443,
                                          "observed_rows": receipt["completed_units"],
                                          "case_count": ncases,
                                          "status": "PASS" if receipt["status"] == "PASS_TIMING_SESSION" and receipt["completed_units"] == 3443 else "FAIL"})
    write_csv(outdir / "coverage_matrix.csv", coverage_rows, list(coverage_rows[0]))

    np.savez_compressed(outdir / "bootstrap_draws.npz", **bootstrap_draws)

    # Source/backend slices use the same row-level latency observations.
    slice_rows = []
    for ev in ("E1", "E2", "E3"):
        ids = sorted({cid for cid, _ in timings[(ev, "A", 1)]})
        for clock in ("A", "B"):
            for dimension in ("source_id", "backend"):
                buckets = defaultdict(list)
                for cid in ids:
                    buckets[panel[cid][dimension]].append(cid)
                for value, bucket in sorted(buckets.items()):
                    for method in METHODS:
                        vals = [paired[(ev, clock, cid, method)] for cid in bucket]
                        slice_rows.append({"evaluation": ev, "clock": clock,
                                           "slice_dimension": dimension, "slice_value": value,
                                           "method": method, "rows": len(bucket),
                                           "groups": len({panel[cid]["group_id"] for cid in bucket}),
                                           "mean_latency_ms": float(np.mean(vals)),
                                           "median_latency_ms": float(np.median(vals)),
                                           "p90_latency_ms": float(np.quantile(vals, .90)),
                                           "p99_latency_ms": float(np.quantile(vals, .99))})
    write_csv(outdir / "source_backend_summary.csv", slice_rows, list(slice_rows[0]))

    # Resource values are process-level high-water marks sampled after each request.
    resource_rows = []
    for ev in ("E1", "E2", "E3"):
        for clock in ("A", "B"):
            for session in (1, 2):
                rss, alloc, reserv = [], [], []
                base = ART / "measure" / ev / f"clock_{clock}" / f"session_{session}"
                for file in base.glob("case_*/*.json"):
                    d = json.loads(file.read_text())
                    r = d.get("resources_after_unit", {})
                    if r:
                        rss.append(int(r.get("process_peak_rss_bytes", 0)))
                        alloc.append(int(r.get("cuda_max_allocated_bytes", 0)))
                        reserv.append(int(r.get("cuda_max_reserved_bytes", 0)))
                resource_rows.append({"evaluation": ev, "clock": clock, "session": session,
                                      "observations": len(rss),
                                      "max_process_peak_rss_bytes": max(rss, default=0),
                                      "max_cuda_allocated_bytes": max(alloc, default=0),
                                      "max_cuda_reserved_bytes": max(reserv, default=0),
                                      "note": "per-process high-water mark sampled after each request; not OS-sampled transient peak"})
    write_csv(outdir / "resource_summary.csv", resource_rows, list(resource_rows[0]))

    # Join accuracy from the saved expansion result, without recalculating it.
    acc = read_csv(SRC / "analysis_v1/pooled_metrics.csv")
    accuracy = [{"evaluation": r["evaluation"], "scope": "all", "scope_value": "ALL",
                 "method": r["treatment"],
                 "group_balanced_mae_seconds": r["group_balanced_mae_seconds"],
                 "source_macro_mae_seconds": r["source_macro_mae_seconds"],
                 "raw_seconds_r2": r["raw_seconds_r2"], "accuracy_source": "saved pooled_metrics.csv"}
                for r in acc]
    sliced = [r for r in read_csv(SRC / "analysis_v1/sliced_metrics.csv")
              if r["scope"] == "backend"]
    accuracy.extend({"evaluation": r["evaluation"], "scope": r["scope"],
                     "scope_value": r["scope_value"], "method": r["treatment"],
                     "group_balanced_mae_seconds": r["group_balanced_mae_seconds"],
                     "source_macro_mae_seconds": r["source_macro_mae_seconds"],
                     "raw_seconds_r2": r["raw_seconds_r2"],
                     "accuracy_source": "saved sliced_metrics.csv"}
                    for r in sliced)
    write_csv(outdir / "saved_accuracy_metrics.csv", accuracy, list(accuracy[0]))
    accuracy_cost = []
    for r in summary:
        if r["evaluation"] in ("E1", "E2") and r["scope"] == "ALL" and r["clock"] in ("A", "B"):
            ma = next(a for a in accuracy if a["evaluation"] == r["evaluation"] and a["scope"] == "all" and a["method"] == r["method"])
            accuracy_cost.append({"evaluation": r["evaluation"], "clock": r["clock"],
                                  "scope": "all", "scope_value": "ALL",
                                  "method": r["method"],
                                  "group_balanced_mean_latency_ms": r["group_balanced_mean_latency_ms"],
                                  "saved_group_balanced_mae_seconds": ma["group_balanced_mae_seconds"],
                                  "saved_raw_seconds_r2": ma["raw_seconds_r2"],
                                  "accuracy_artifact": "analysis_v1/pooled_metrics.csv"})
        elif r["evaluation"] == "E3" and r["clock"] == "B":
            split = read_csv(SRC / "splits/E3_outer.csv")
            backends = sorted({x["heldout_backend"] for x in split})
            backend = backends[int(r["scope"])]
            ma = next(a for a in accuracy if a["evaluation"] == "E3" and
                      a["scope"] == "backend" and a["scope_value"] == backend and
                      a["method"] == r["method"])
            accuracy_cost.append({"evaluation": "E3", "clock": "B",
                                  "scope": "backend", "scope_value": backend,
                                  "method": r["method"],
                                  "group_balanced_mean_latency_ms": r["group_balanced_mean_latency_ms"],
                                  "saved_group_balanced_mae_seconds": ma["group_balanced_mae_seconds"],
                                  "saved_raw_seconds_r2": ma["raw_seconds_r2"],
                                  "accuracy_artifact": "analysis_v1/sliced_metrics.csv"})
    write_csv(outdir / "accuracy_cost_points.csv", accuracy_cost, list(accuracy_cost[0]))
    receipt = {"status": "PASS_AGGREGATION", "timing_fingerprint": json.loads((ART / "timing_manifest.json").read_text())["fingerprint"],
               "evaluations": ["E1", "E2", "E3"], "clocks": ["A", "B"],
               "sessions": [1, 2], "rows_per_evaluation": 3443,
               "methods": list(METHODS), "timed_repetitions_per_session": 10,
               "bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED,
               "uses_saved_accuracy_without_recalculation": True,
               "files": ["row_level_latency.csv", "latency_summary.csv", "source_backend_summary.csv",
                         "resource_summary.csv", "session_drift_summary.csv", "coverage_matrix.csv", "paired_comparisons.csv", "accuracy_cost_points.csv",
                         "saved_accuracy_metrics.csv", "bootstrap_draws.npz"]}
    (outdir / "aggregation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
