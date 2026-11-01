#!/usr/bin/env python3
"""Render a concise auditable report from completed cost-study summaries."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/postfreeze_qpu_workflow_mean_cost_v1/v2/summary"
METHODS = {"M0": "MLP", "M1": "MLP + shots", "G0": "Graph+MLP",
           "G1": "Graph+MLP + shots", "R0": "Selected classical pipeline",
           "C0": "Outer-training median"}


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, n=3):
    return f"{float(x):.{n}f}"


def main():
    latency = read("latency_summary.csv")
    accuracy = read("saved_accuracy_metrics.csv")
    comps = read("paired_comparisons.csv")
    resources = read("resource_summary.csv")
    drift = read("session_drift_summary.csv")
    cover = read("coverage_matrix.csv")
    points = read("accuracy_cost_points.csv")
    receipt = json.loads((OUT / "aggregation_receipt.json").read_text())

    lines = [
        "# QPU workflow-mean estimator inference cost: results",
        "",
        "## Scope and measurement boundary",
        "",
        "This post-freeze study measures six already-selected predictors on the separate expanded cohort of 3,443 observations and 203 structural groups (Ma–Li 340, Qonductor 3,058, QPack 45 workflow means). It does not replace or alter the frozen 4,515-row benchmark.",
        "",
        "Clock A starts with the method's ready CPU-side input and measures one prediction request, including singleton batch formation, transfer when needed, all three neural seeds, median reduction, scalar return, and CUDA synchronization. Clock B starts with saved raw feature inputs in CPU memory and also measures each method's saved feature transforms and graph/tensor construction. B is the primary practical endpoint for a request when raw features are already available.",
        "",
        "Neither clock includes reading a QASM file, reconstructing a circuit from source, a new Qiskit transpilation, network/service latency, or QPU execution. Therefore these measurements are prediction compute cost, not end-to-end job time. Clock C (circuit-bytes-to-prediction) and cold-start Clock D are outside this completed A/B run and must not be inferred from A/B.",
        "",
        "E1 is group-disjoint evaluation, E2 is source-stratified random-row evaluation with intended structural-group overlap, and E3 is backend-context holdout. For every evaluation and clock, two fresh sessions measured each observation-method pair 10 times, with three warmups and saved-output parity checks. Reported latency is the median of 10 within each session, then the median across the two sessions per observation. Accuracy is copied from the existing saved pooled/sliced metric artifacts and is not recomputed here.",
        "",
        "## Coverage",
        "",
        f"Aggregation fingerprint: `{receipt['timing_fingerprint']}`. Each of the six methods has 3,443/3,443 rows in each evaluation × clock × session cell. Coverage matrix is `coverage_matrix.csv`; all {sum(r['status']=='PASS' for r in cover)}/{len(cover)} method-session cells passed.",
        "",
        "## Primary accuracy–cost points (Clock B)",
        "",
        "Latency is group-balanced mean milliseconds per observation; error is the saved group-balanced MAE in seconds. Values remain separate dimensions and are not combined into a synthetic score.",
        "",
        "| Evaluation | Method | Latency (ms) | Saved GB-MAE (s) | Saved raw R² |",
        "|---|---|---:|---:|---:|",
    ]
    for ev in ("E1", "E2"):
        for r in points:
            if r["evaluation"] == ev and r["clock"] == "B":
                lines.append(f"| {ev} | {METHODS[r['method']]} | {fmt(r['group_balanced_mean_latency_ms'])} | {fmt(r['saved_group_balanced_mae_seconds'], 4)} | {fmt(r['saved_raw_seconds_r2'], 4)} |")
    lines += ["", "Clock B intervals for the six preregistered comparisons are in `paired_comparisons.csv`. E1/E2 intervals use source-stratified whole-group resampling and Bonferroni-adjusted confidence levels across the 12 planned comparisons. Negative delta means the first named method is faster. These are conditional on this cohort and two sessions on this host.", ""]

    # Clock A is a ready-input diagnostic: show latency only so it is not
    # mistaken for a second accuracy-cost endpoint with a different boundary.
    lines += ["## Ready-input latency diagnostic (Clock A)", "", "Clock A begins after method-specific input preparation. It measures model execution, median-of-three neural aggregation and return of the scalar prediction. It excludes raw-feature transforms and graph/tensor construction, which Clock B includes. Accuracy is therefore not repeated in this table.", "", "| Evaluation | Method | Group-balanced mean latency (ms) | Median request (ms) | P90 (ms) | P99 (ms) |", "|---|---|---:|---:|---:|---:|"]
    for ev in ("E1", "E2"):
        for r in latency:
            if r["evaluation"] == ev and r["scope"] == "ALL" and r["clock"] == "A":
                mean = "<0.001" if r["method"] == "C0" else fmt(r["group_balanced_mean_latency_ms"])
                median = "<0.001" if r["method"] == "C0" else fmt(r["median_latency_ms"])
                p90 = "<0.001" if r["method"] == "C0" else fmt(r["p90_latency_ms"])
                p99 = "<0.001" if r["method"] == "C0" else fmt(r["p99_latency_ms"])
                lines.append(f"| {ev} | {METHODS[r['method']]} | {mean} | {median} | {p90} | {p99} |")
    lines += ["", "Clock A is diagnostic; Clock B remains the primary warm-request boundary. The constant lookup is below the report's 0.001 ms display precision and close to harness overhead; the displayed bound is not a precision claim.", ""]

    lines += ["## Backend-context slices (E3)", "", "Each row below belongs to its own held-out-backend test population; these are separate tasks, not one pooled leaderboard.", "", "| Backend | Method | Latency (ms) | Saved GB-MAE (s) | Saved raw R² |", "|---|---|---:|---:|---:|"]
    for r in points:
        if r["evaluation"] == "E3" and r["clock"] == "B":
            lines.append(f"| {r['scope_value']} | {METHODS[r['method']]} | {fmt(r['group_balanced_mean_latency_ms'])} | {fmt(r['saved_group_balanced_mae_seconds'], 4)} | {fmt(r['saved_raw_seconds_r2'], 4)} |")

    lines += ["", "Per-source and per-backend latency summaries are in `source_backend_summary.csv`. Session-to-session shifts are in `session_drift_summary.csv`; row-level session medians remain in `row_level_latency.csv`. The E3 paired intervals in `paired_comparisons.csv` use whole structural groups and a family-wise Bonferroni level across 60 planned backend contrasts. A backend with fewer than two structural groups cannot support a between-group interval; the comparison file records this as unavailable.", "", "## Resources and operational cost", "", "Process RSS and CUDA allocation/reservation high-water marks are in `resource_summary.csv`. They are the values visible to this process after requests, not OS-sampled transient peaks. Model loading, classical winner-state reconstruction, training, feature-cache construction, and QPU acquisition are preparation costs and are not included in per-request latency. The recovered classical states are prediction-parity validated reconstructions, not byte-identical recovered historical estimators. The median baseline C0 is a saved scalar lookup; very short times are close to the harness/call-overhead floor and should not be read as general service latency.", "", "Plots are `accuracy_cost_E1_primary.png`, `accuracy_cost_E2_primary.png`, separate ready-input diagnostic plots, and one E3 plot per held-out backend. Pareto lines are descriptive only; uncertainty and context must be considered before claiming a method dominates another.", "", "## Interpretation and limits", "", "The comparison answers how long each saved predictor takes under the measured input boundary, and places that cost beside its already-saved accuracy on the same evaluation population. It does not establish universal deployment speed: measurements come from one host, two sessions, one GPU/CPU setup, singleton requests, and fixed software versions. Low GPU utilization is expected for latency-focused single-item inference; maximizing utilization with large batches would answer throughput, a different question.", "", "No neural model was trained, no hyperparameters or splits were changed, no new simulator or QPU execution was performed, and no saved accuracy metric or historical artifact was overwritten.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
