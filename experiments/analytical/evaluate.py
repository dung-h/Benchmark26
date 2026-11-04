#!/usr/bin/env python3
"""Score frozen analytical routes on the unified dataset and frozen splits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from benchmark26.integrity import sha256_file as sha
from benchmark26.paths import ARTIFACTS, record_name
from benchmark26.tables import read_csv as read

ROOT = Path(__file__).resolve().parents[2]
NS = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
STUDY = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
AN = STUDY / "analysis_v1"


def write(path, records):
    """Write records as a CSV file; raise if there are none."""
    if not records:
        raise RuntimeError("empty_output:" + str(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def fnum(v):
    """Convert a value to float."""
    return float(v)


def metric(y, p, g, s):
    """Return the error metrics of predictions p against targets y, with group labels g and source labels s.

    Undefined values are empty strings.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    g = np.asarray(g)
    s = np.asarray(s)
    if not len(y) or y.shape != p.shape or len(g) != len(y) or len(s) != len(y):
        raise ValueError("metric_alignment")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not (y > 0).all():
        raise ValueError("metric_nonfinite_or_target")
    err = p - y
    ae = np.abs(err)
    den = float(np.square(y - y.mean()).sum())
    rawr2 = 1 - float(np.square(err).sum()) / den if den > 0 and len(y) > 1 else ""
    log_status = "UNAVAILABLE_NONPOSITIVE_PREDICTION"
    logr2 = ""
    if (p > 0).all():
        ly = np.log1p(y)
        lp = np.log1p(p)
        ld = float(np.square(ly - ly.mean()).sum())
        if ld > 0 and len(y) > 1:
            logr2 = 1 - float(np.square(ly - lp).sum()) / ld
            log_status = "DEFINED"
        else:
            log_status = "UNDEFINED_CONSTANT_OR_SINGLE_TARGET"
    groups = sorted(set(g))
    sources = sorted(set(s))
    return {
        "rows": len(y),
        "groups": len(groups),
        "sources": len(sources),
        "group_balanced_mae_seconds": float(np.mean([ae[g == x].mean() for x in groups])),
        "pooled_mae_seconds": float(ae.mean()),
        "source_macro_mae_seconds": float(np.mean([ae[s == x].mean() for x in sources])),
        "medae_seconds": float(np.median(ae)),
        "rmse_seconds": float(np.sqrt(np.square(err).mean())),
        "raw_seconds_r2": rawr2,
        "log1p_seconds_r2": logr2,
        "log_r2_status": log_status,
        "bias_seconds": float(err.mean()),
        "p90_absolute_error_seconds": float(np.quantile(ae, 0.90)),
        "p95_absolute_error_seconds": float(np.quantile(ae, 0.95)),
        "p99_absolute_error_seconds": float(np.quantile(ae, 0.99)),
        "max_absolute_error_seconds": float(ae.max()),
        "zero_predictions": int((p == 0).sum()),
        "overprediction_fraction": float((err > 0).mean()),
        "underprediction_fraction": float((err < 0).mean()),
        "target_mean_seconds": float(y.mean()),
        "target_std_seconds": float(y.std()),
        "prediction_mean_seconds": float(p.mean()),
    }


def corr(x, y):
    """Return (Pearson, Spearman) correlations, or ('', '') for fewer than 3 points, non-finite values or a constant input."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if (
        len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or len(np.unique(x)) < 2
        or len(np.unique(y)) < 2
    ):
        return ("", "")
    return (float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic))


def main():
    """Score the frozen analytical routes on the 3,443-run dataset with the frozen splits.

    Checks the dataset and lineage, then for E1, E2 and each E3 held-out backend computes
    metrics, source, backend and provenance slices, fold metrics, correlations and
    coverage. The Qonductor selector predictions and the outer-training medians are
    combined without refitting. Unitless NoTaDS rows are correlation-only. Paired
    whole-group intervals reuse the frozen draw arrays. Route parity and exactly-once
    shot multiplication are checked from the stored features. Writes the metric,
    sliced, fold, correlation, paired-comparison and coverage tables and a receipt
    with the input hashes.
    """
    run_rows = read(STUDY / "cohort/panel.csv")
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in read(STUDY / "cohort/targets.csv")}
    ids = sorted(r["canonical_observation_id"] for r in run_rows)
    if len(ids) != 3443 or len(set(ids)) != 3443 or set(ids) != set(targets):
        raise RuntimeError("cohort_identity")
    runs_by_id = {r["canonical_observation_id"]: r for r in run_rows}
    lin = {r["canonical_observation_id"]: r for r in read(NS / "lineage.csv")}
    if set(lin) != set(ids):
        raise RuntimeError("lineage_identity")
    prov = {
        i: {"structure_provenance": lin[i]["structure_provenance"], "timing_provenance": lin[i]["timing_provenance"]}
        for i in ids
    }
    rawpred = read(NS / "predictions.csv")
    idx = {}
    for r in rawpred:
        key = (r["evaluation"], r["case"], r["canonical_observation_id"], r["method_id"])
        if key in idx:
            raise RuntimeError("duplicate_analytical_prediction:" + str(key))
        cid = r["canonical_observation_id"]
        if cid not in runs_by_id or float(r["target_seconds"]) != targets[cid]:
            raise RuntimeError("target_mismatch:" + cid)
        for k in ("group_id", "source_id", "backend", "shots"):
            if r[k] != runs_by_id[cid][k] and not (k == "shots" and float(r[k]) == float(runs_by_id[cid][k])):
                raise RuntimeError("metadata_mismatch:" + cid + ":" + k)
        idx[key] = r
    reduced = read(AN / "reduced_predictions.csv")
    base = {}
    for r in reduced:
        if r["treatment"] in ("R0", "C0"):
            key = (r["evaluation"], r["canonical_observation_id"], r["treatment"])
            if key in base:
                raise RuntimeError("duplicate_saved_baseline:" + str(key))
            base[key] = float(r["predicted_seconds"])
    for ev in ("E1", "E2", "E3"):
        for tr in ("R0", "C0"):
            if sum((e == ev and t == tr) for e, _, t in base) != 3443:
                raise RuntimeError("baseline_coverage:" + ev + ":" + tr)
    # One test prediction per observation per evaluation; E3 is segmented by held-backend case.
    methods = sorted({k[3] for k in idx})
    expected_methods = {
        "QISKIT_SINGLE",
        "QISKIT_SINGLE_SHOT_SCALED",
        "QISKIT_SINGLE_AFFINE",
        "QISKIT_SINGLE_SHOT_SCALED_AFFINE",
        "QCRE_UNITARY_SINGLE",
        "QCRE_UNITARY_SINGLE_SHOT_SCALED",
        "QCRE_UNITARY_SINGLE_AFFINE",
        "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE",
        "HYPERQ_PATH_SINGLE",
        "HYPERQ_PATH_SINGLE_SHOT_SCALED",
        "HYPERQ_PATH_SINGLE_AFFINE",
        "HYPERQ_PATH_SINGLE_SHOT_SCALED_AFFINE",
        "NOTADS_PAPER_LEVEL_SCORE",
        "NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED",
        "NOTADS_PAPER_LEVEL_SCORE_AFFINE",
        "NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE",
        "NOTADS_CODE_FILTERED_DEPTH",
        "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED",
        "NOTADS_CODE_FILTERED_DEPTH_AFFINE",
        "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE",
        "IBM_QUICK_USAGE_DIAGNOSTIC",
        "IBM_USAGE_DURATION_DIAGNOSTIC",
    }
    if set(methods) != expected_methods:
        raise RuntimeError("method_inventory_mismatch:" + repr(methods))
    cases = {}
    foldmap = {
        ev: {r["canonical_observation_id"]: str(int(r["outer_fold"])) for r in read(STUDY / f"splits/{ev}_outer.csv")}
        for ev in ("E1", "E2")
    }
    if any(set(foldmap[ev]) != set(ids) for ev in foldmap):
        raise RuntimeError("fold_identity")
    for ev in ("E1", "E2"):
        cases[(ev, "ALL")] = ids
        for method in methods:
            rows = [r for (e, c, i, m), r in idx.items() if e == ev and m == method]
            got = sorted(r["canonical_observation_id"] for r in rows)
            if got != ids:
                raise RuntimeError(f"prediction_coverage:{ev}:{method}:{len(got)}")
    e3contexts = sorted({c for e, c, i, m in idx if e == "E3"})
    for context in e3contexts:
        testr = sorted({i for e, c, i, m in idx if e == "E3" and c == context})
        if not testr:
            raise RuntimeError("empty_E3_context:" + context)
        cases[("E3", context)] = testr
        for method in methods:
            got = sorted(i for e, c, i, m in idx if e == "E3" and c == context and m == method)
            if got != testr:
                raise RuntimeError(f"E3_method_coverage:{context}:{method}")
    sec_methods = []
    score_methods = []
    for m in methods:
        unit = next(idx[(e, c, i, m)]["output_unit"] for e, c, i, mm in idx if mm == m)
        (score_methods if unit == "relative_score" else sec_methods).append(m)

    # Combine the frozen Qonductor selector and outer-training median predictions without refitting.
    def getrow(ev, case, cid, m):
        """Return the saved prediction row for (evaluation, case, observation, method).

        For E1 and E2 with case ALL, the case is the observation's fold.
        """
        if ev in ("E1", "E2") and case == "ALL":
            case = foldmap[ev][cid]
        return idx[(ev, case, cid, m)]

    def getpred(ev, case, cid, m):
        """Return a method's prediction in seconds.

        R0 and C0 come from the reduced baseline; any other method must have a prediction in seconds.
        """
        if m in ("R0", "C0"):
            return base[(ev, cid, m)]
        r = getrow(ev, case, cid, m)
        if r["output_unit"] not in ("seconds", "seconds_diagnostic") or r["prediction_seconds"] == "":
            raise ValueError("non_seconds_method:" + m)
        return float(r["prediction_seconds"])

    metric_rows = []
    slice_rows = []
    fold_rows = []
    corr_rows = []
    coverage_rows = []
    for (ev, case), caseids in cases.items():
        candidate_methods = sec_methods + ["R0", "C0"]
        if ev == "E3" and case == "ALL":
            continue
        for m in candidate_methods:
            rs = [runs_by_id[i] for i in caseids]
            yy = [targets[i] for i in caseids]
            pp = [getpred(ev, case, i, m) for i in caseids]
            mm = metric(yy, pp, [x["group_id"] for x in rs], [x["source_id"] for x in rs])
            outunit = "seconds_diagnostic" if m.endswith("_DIAGNOSTIC") else "seconds"
            metric_rows.append({"evaluation": ev, "case": case, "method_id": m, "output_unit": outunit, **mm})
            coverage_rows.append(
                {
                    "evaluation": ev,
                    "case": case,
                    "method_id": m,
                    "rows": len(rs),
                    "groups": len({x["group_id"] for x in rs}),
                    "sources": len({x["source_id"] for x in rs}),
                    "status": "FULL_COMMON_COVERAGE",
                }
            )
            if ev in ("E1", "E2"):
                for fold in sorted({foldmap[ev][i] for i in caseids}):
                    foldids = [i for i in caseids if foldmap[ev][i] == fold]
                    fr = [runs_by_id[i] for i in foldids]
                    fm = metric(
                        [targets[i] for i in foldids],
                        [getpred(ev, case, i, m) for i in foldids],
                        [x["group_id"] for x in fr],
                        [x["source_id"] for x in fr],
                    )
                    fold_rows.append(
                        {
                            "evaluation": ev,
                            "outer_fold": fold,
                            "method_id": m,
                            "output_unit": "seconds_diagnostic" if m.endswith("_DIAGNOSTIC") else "seconds",
                            **fm,
                        }
                    )
            for scope in ("overall", "source", "backend", "structure_provenance", "timing_provenance"):
                values = (
                    ["ALL"]
                    if scope == "overall"
                    else sorted(
                        {
                            (
                                runs_by_id[i][
                                    {
                                        "source": "source_id",
                                        "backend": "backend",
                                        "structure_provenance": "",
                                        "timing_provenance": "",
                                    }.get(scope, "")
                                ]
                                if scope in ("source", "backend")
                                else prov[i][scope]
                            )
                            for i in caseids
                        }
                    )
                )
                for v in values:
                    subset = [
                        i
                        for i in caseids
                        if scope == "overall"
                        or (
                            runs_by_id[i]["source_id"] == v
                            if scope == "source"
                            else runs_by_id[i]["backend"] == v
                            if scope == "backend"
                            else prov[i][scope] == v
                        )
                    ]
                    if not subset:
                        continue
                    subrows = [runs_by_id[i] for i in subset]
                    sy = [targets[i] for i in subset]
                    sp = [getpred(ev, case, i, m) for i in subset]
                    z = metric(sy, sp, [x["group_id"] for x in subrows], [x["source_id"] for x in subrows])
                    slice_rows.append(
                        {
                            "evaluation": ev,
                            "case": case,
                            "scope": scope,
                            "scope_value": v,
                            "method_id": m,
                            "output_unit": "seconds_diagnostic" if m.endswith("_DIAGNOSTIC") else "seconds",
                            **z,
                        }
                    )
            if m not in ("R0", "C0"):
                vals = [getpred(ev, case, i, m) for i in caseids]
                unit = "seconds_diagnostic" if m.endswith("_DIAGNOSTIC") else "seconds"
                pr, sr = corr(vals, yy)
                corr_rows.append(
                    {
                        "evaluation": ev,
                        "case": case,
                        "scope": "overall",
                        "scope_value": "ALL",
                        "method_id": m,
                        "input_unit": unit,
                        "pearson_target": pr,
                        "spearman_target": sr,
                        "rows": len(vals),
                        "interpretation": "descriptive association; not a mechanism or accuracy metric",
                    }
                )
                if ev in ("E1", "E2"):
                    for source in sorted({runs_by_id[i]["source_id"] for i in caseids}):
                        subset = [i for i in caseids if runs_by_id[i]["source_id"] == source]
                        pr, sr = corr([getpred(ev, case, i, m) for i in subset], [targets[i] for i in subset])
                        corr_rows.append(
                            {
                                "evaluation": ev,
                                "case": case,
                                "scope": "source",
                                "scope_value": source,
                                "method_id": m,
                                "input_unit": unit,
                                "pearson_target": pr,
                                "spearman_target": sr,
                                "rows": len(subset),
                                "interpretation": "descriptive association; not a mechanism or accuracy metric",
                            }
                        )
    # Unitless NoTaDS rows are correlation-only (raw and shot-scaled); affine versions were scored above.
    for (ev, case), caseids in cases.items():
        if ev == "E3" and case == "ALL":
            continue
        for m in score_methods:
            coverage_rows.append(
                {
                    "evaluation": ev,
                    "case": case,
                    "method_id": m,
                    "rows": len(caseids),
                    "groups": len({runs_by_id[i]["group_id"] for i in caseids}),
                    "sources": len({runs_by_id[i]["source_id"] for i in caseids}),
                    "status": "FULL_COMMON_COVERAGE_SCORE_ONLY_UNITLESS",
                }
            )
            for scope in ("overall", "source"):
                values = ["ALL"] if scope == "overall" else sorted({runs_by_id[i]["source_id"] for i in caseids})
                for v in values:
                    subset = [i for i in caseids if scope == "overall" or runs_by_id[i]["source_id"] == v]
                    rr = [getrow(ev, case, i, m) for i in subset]
                    xx = [float(r["raw_input_value"]) for r in rr]
                    yy = [targets[i] for i in subset]
                    pr, sr = corr(xx, yy)
                    corr_rows.append(
                        {
                            "evaluation": ev,
                            "case": case,
                            "scope": scope,
                            "scope_value": v,
                            "method_id": m,
                            "input_unit": "relative_score",
                            "pearson_target": pr,
                            "spearman_target": sr,
                            "rows": len(subset),
                            "interpretation": "unitless score association only; no seconds MAE",
                        }
                    )
    # Paired whole-group intervals use the exact frozen draw arrays for E1/E2 and each E3 backend.
    comp_rows = []
    compare_pairs = []
    for m in sec_methods:
        if m.endswith("_DIAGNOSTIC"):
            continue
        compare_pairs.extend([(m, "R0"), (m, "C0")])
    route_groups = [
        ("QISKIT_SINGLE", "QISKIT_SINGLE_SHOT_SCALED"),
        ("QISKIT_SINGLE", "QISKIT_SINGLE_AFFINE"),
        ("QISKIT_SINGLE_SHOT_SCALED", "QISKIT_SINGLE_SHOT_SCALED_AFFINE"),
        ("QISKIT_SINGLE_AFFINE", "QISKIT_SINGLE_SHOT_SCALED_AFFINE"),
        ("QCRE_UNITARY_SINGLE", "QCRE_UNITARY_SINGLE_SHOT_SCALED"),
        ("QCRE_UNITARY_SINGLE", "QCRE_UNITARY_SINGLE_AFFINE"),
        ("QCRE_UNITARY_SINGLE_SHOT_SCALED", "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE"),
        ("QCRE_UNITARY_SINGLE_AFFINE", "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE"),
        ("HYPERQ_PATH_SINGLE", "HYPERQ_PATH_SINGLE_SHOT_SCALED"),
        ("HYPERQ_PATH_SINGLE", "HYPERQ_PATH_SINGLE_AFFINE"),
        ("HYPERQ_PATH_SINGLE_SHOT_SCALED", "HYPERQ_PATH_SINGLE_SHOT_SCALED_AFFINE"),
        ("HYPERQ_PATH_SINGLE_AFFINE", "HYPERQ_PATH_SINGLE_SHOT_SCALED_AFFINE"),
        ("NOTADS_PAPER_LEVEL_SCORE_AFFINE", "NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE"),
        ("NOTADS_CODE_FILTERED_DEPTH_AFFINE", "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE"),
        ("QISKIT_SINGLE", "HYPERQ_PATH_SINGLE"),
        ("QCRE_UNITARY_SINGLE", "QISKIT_SINGLE"),
    ]
    compare_pairs.extend(route_groups)
    for (ev, case), caseids in cases.items():
        if ev == "E3" and case == "ALL":
            continue
        bootpath = AN / "bootstrap" / ((f"E3_{case}.npz") if ev == "E3" else ev + "_all.npz")
        if not bootpath.exists():
            raise RuntimeError("missing_frozen_bootstrap:" + str(bootpath))
        with np.load(bootpath, allow_pickle=False) as z:
            draws = z["draws"]
            bgroups = list(z["groups"].astype(str))
        group_order = sorted({runs_by_id[i]["group_id"] for i in caseids})
        if bgroups != group_order:
            raise RuntimeError(f"bootstrap_group_alignment:{ev}:{case}:{len(bgroups)}:{len(group_order)}")
        group_rows = defaultdict(list)
        for i in caseids:
            group_rows[runs_by_id[i]["group_id"]].append(i)
        for a, b in compare_pairs:
            per_group = []
            for g in group_order:
                ii = group_rows[g]
                err_a = np.mean([abs(getpred(ev, case, i, a) - targets[i]) for i in ii])
                err_b = np.mean([abs(getpred(ev, case, i, b) - targets[i]) for i in ii])
                per_group.append(err_a - err_b)
            d = np.asarray(per_group, float)
            boot = d[draws].mean(axis=1)
            comp_rows.append(
                {
                    "evaluation": ev,
                    "case": case,
                    "contrast": a + " - " + b,
                    "treatment": a,
                    "control": b,
                    "delta_group_balanced_mae_seconds": float(d.mean()),
                    "ci95_lower_seconds": float(np.quantile(boot, 0.025)),
                    "ci95_upper_seconds": float(np.quantile(boot, 0.975)),
                    "empirical_bootstrap_fraction_delta_below_zero": float(np.mean(boot < 0)),
                    "valid_replicates": len(boot),
                    "resampling_unit": "structural_group",
                    "bootstrap_draws_path": record_name(bootpath),
                    "bootstrap_draws_sha256": sha(bootpath),
                    "fraction_is_p_value": False,
                    "multiplicity": "descriptive/exploratory; no confirmatory superiority claims",
                }
            )
    # Check route parity and exactly-once shot multiplication from stored features.
    rawfeat = {(r["graph_file"], r["snapshot_id"]): r for r in read(NS / "raw_features.csv")}
    for i in ids:
        l = lin[i]
        x = rawfeat[(l["graph_file"], l["snapshot_id"])]
        sh = float(runs_by_id[i]["shots"])
        for route, field in [
            ("QISKIT_SINGLE", "hyperq_full_seconds"),
            ("HYPERQ_PATH_SINGLE", "hyperq_full_seconds"),
            ("QCRE_UNITARY_SINGLE", "hyperq_unitary_seconds"),
        ]:
            r = idx[("E1", foldmap["E1"][i], i, route)]
            if not math.isclose(float(r["raw_input_value"]), float(x[field]), rel_tol=1e-13, abs_tol=1e-15):
                raise RuntimeError("raw_feature_prediction_mismatch:" + i + ":" + route)
        q = idx[("E1", foldmap["E1"][i], i, "QISKIT_SINGLE")]
        h = idx[("E1", foldmap["E1"][i], i, "HYPERQ_PATH_SINGLE")]
        qs = idx[("E1", foldmap["E1"][i], i, "QISKIT_SINGLE_SHOT_SCALED")]
        hs = idx[("E1", foldmap["E1"][i], i, "HYPERQ_PATH_SINGLE_SHOT_SCALED")]
        if not math.isclose(float(q["raw_input_value"]), float(h["raw_input_value"]), rel_tol=0, abs_tol=0):
            raise RuntimeError("qiskit_hyperq_not_identical:" + i)
        if not math.isclose(
            float(qs["raw_input_value"]), float(q["raw_input_value"]) * sh, rel_tol=1e-13, abs_tol=1e-15
        ):
            raise RuntimeError("qiskit_shot_multiplier:" + i)
        if not math.isclose(
            float(hs["raw_input_value"]), float(h["raw_input_value"]) * sh, rel_tol=1e-13, abs_tol=1e-15
        ):
            raise RuntimeError("hyperq_shot_multiplier:" + i)
    # Per-row routes may have different test fold labels; the checks above use E1's frozen case.
    write(NS / "metrics.csv", metric_rows)
    write(NS / "sliced_metrics.csv", slice_rows)
    write(NS / "fold_metrics.csv", fold_rows)
    write(NS / "correlations.csv", corr_rows)
    write(NS / "paired_comparisons.csv", comp_rows)
    write(NS / "coverage.csv", coverage_rows)
    # Keep the service-usage formula visibly separate from same-boundary estimator ranking.
    boundary = [r for r in metric_rows if r["method_id"].endswith("_DIAGNOSTIC")]
    write(
        NS / "boundary_diagnostics.csv",
        [
            {
                "evaluation": r["evaluation"],
                "case": r["case"],
                "method_id": r["method_id"],
                "rows": r["rows"],
                "groups": r["groups"],
                "diagnostic_mae_seconds": r["pooled_mae_seconds"],
                "diagnostic_bias_seconds": r["bias_seconds"],
                "target_boundary_warning": "Different source-specific target boundaries; diagnostic only, not an interchangeable-runtime leaderboard.",
            }
            for r in boundary
        ],
    )
    # Checksums are for comparators before writing this analysis.
    baseline_sha = sha(AN / "reduced_predictions.csv")
    run_table_sha = sha(STUDY / "cohort/panel.csv")
    target_sha = sha(STUDY / "cohort/targets.csv")
    receipt = {
        "status": "PASS_ANALYTICAL_SCORING",
        "rows": len(ids),
        "groups": len({r["group_id"] for r in run_rows}),
        "sources": 3,
        "evaluations": ["E1", "E2", "E3"],
        "method_ids": methods,
        "seconds_scored_methods": sec_methods,
        "unitless_score_methods_correlation_only": score_methods,
        "E3_backend_cases": e3contexts,
        "metric_rows": len(metric_rows),
        "slice_rows": len(slice_rows),
        "fold_metric_rows": len(fold_rows),
        "correlation_rows": len(corr_rows),
        "paired_comparisons": len(comp_rows),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "resampling_unit": "structural_group",
        "no_model_retraining": True,
        "no_jobs_run": True,
        "raw_NoTaDS_scores_given_seconds_MAE": False,
        "IBM_usage_formulas_same_boundary_ranking": False,
        "Qiskit_and_HyperQ_full_path_raw_inputs_exactly_equal": True,
        "shot_scaling_applied_once": True,
        "saved_reduced_predictions_sha256_before_analysis": baseline_sha,
        "panel_sha256": run_table_sha,
        "targets_sha256": target_sha,
        "metrics_sha256": sha(NS / "metrics.csv"),
        "sliced_metrics_sha256": sha(NS / "sliced_metrics.csv"),
        "fold_metrics_sha256": sha(NS / "fold_metrics.csv"),
        "correlations_sha256": sha(NS / "correlations.csv"),
        "paired_comparisons_sha256": sha(NS / "paired_comparisons.csv"),
        "coverage_sha256": sha(NS / "coverage.csv"),
        "boundary_diagnostics_sha256": sha(NS / "boundary_diagnostics.csv"),
    }
    (NS / "evaluation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
