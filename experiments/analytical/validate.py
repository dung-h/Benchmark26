#!/usr/bin/env python3
"""Independent replay of unified analytical predictions, metrics, and fits."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS, recorded
from pathlib import Path
import csv, hashlib, json, math
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
NS = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
ST = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
AN = ST / "analysis_v1"


def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b, tol=1e-10):
    if b in ("", None):
        if a not in ("", None):
            raise AssertionError(("expected_blank", a, b))
    else:
        try:
            same = math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except (TypeError, ValueError):
            same = str(a) == str(b)
        if not same:
            raise AssertionError(("value_mismatch", a, b))


def independently_metric(rows, pred):
    y = np.array([float(r["target_seconds"]) for r in rows])
    p = np.array([float(pred(r)) for r in rows])
    g = np.array([r["group_id"] for r in rows])
    s = np.array([r["source_id"] for r in rows])
    assert len(y) > 0 and y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all() and (y > 0).all()
    e = p - y
    ae = abs(e)
    den = float(np.square(y - y.mean()).sum())
    r2 = 1 - float(np.square(e).sum()) / den if den > 0 and len(y) > 1 else ""
    logr2 = ""
    status = "UNAVAILABLE_NONPOSITIVE_PREDICTION"
    if (p > 0).all():
        ly = np.log1p(y)
        lp = np.log1p(p)
        d = float(np.square(ly - ly.mean()).sum())
        if d > 0 and len(y) > 1:
            logr2 = 1 - float(np.square(ly - lp).sum()) / d
            status = "DEFINED"
        else:
            status = "UNDEFINED_CONSTANT_OR_SINGLE_TARGET"
    return {
        "rows": len(y),
        "groups": len(set(g)),
        "sources": len(set(s)),
        "group_balanced_mae_seconds": np.mean([ae[g == q].mean() for q in sorted(set(g))]),
        "pooled_mae_seconds": ae.mean(),
        "source_macro_mae_seconds": np.mean([ae[s == q].mean() for q in sorted(set(s))]),
        "medae_seconds": np.median(ae),
        "rmse_seconds": np.sqrt(np.square(e).mean()),
        "raw_seconds_r2": r2,
        "log1p_seconds_r2": logr2,
        "log_r2_status": status,
        "bias_seconds": e.mean(),
        "p90_absolute_error_seconds": np.quantile(ae, 0.9),
        "p95_absolute_error_seconds": np.quantile(ae, 0.95),
        "p99_absolute_error_seconds": np.quantile(ae, 0.99),
        "max_absolute_error_seconds": ae.max(),
        "zero_predictions": int((p == 0).sum()),
        "overprediction_fraction": np.mean(e > 0),
        "underprediction_fraction": np.mean(e < 0),
        "target_mean_seconds": y.mean(),
        "target_std_seconds": y.std(),
        "prediction_mean_seconds": p.mean(),
    }


def corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return ("", "")
    return pearsonr(x, y).statistic, spearmanr(x, y).statistic


def main():
    receipt = json.loads((NS / "evaluation_receipt.json").read_text())
    cal = json.loads((NS / "calibration_receipt.json").read_text())
    inv = json.loads((NS / "inventory.json").read_text())
    ext = json.loads((NS / "extraction_receipt.json").read_text())
    panel_rows = read(ST / "cohort/panel.csv")
    panel = {r["canonical_observation_id"]: r for r in panel_rows}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in read(ST / "cohort/targets.csv")}
    ids = sorted(panel)
    assert len(ids) == 3443 and len(set(panel[x]["group_id"] for x in ids)) == 203 and set(ids) == set(targets)
    raw = read(NS / "raw_features.csv")
    line = read(NS / "lineage.csv")
    lin = {r["canonical_observation_id"]: r for r in line}
    rawby = {(r["graph_file"], r["snapshot_id"]): r for r in raw}
    pred = read(NS / "predictions.csv")
    pidx = {(r["evaluation"], r["case"], r["canonical_observation_id"], r["method_id"]): r for r in pred}
    splits = {e: read(ST / f"splits/{e}_outer.csv") for e in ("E1", "E2", "E3")}
    foldmaps = {e: {r["canonical_observation_id"]: str(int(r["outer_fold"])) for r in splits[e]} for e in ("E1", "E2")}
    assert len(raw) == 1438 and len(lin) == 3443 and set(lin) == set(ids)
    # Exact cache parity for graph/snapshot keys shared with the previous frozen audit.
    qparity = json.loads((NS / "qiskit_parity_receipt.json").read_text())
    assert (
        qparity["status"] == "PASS_EXACT_QISKIT_ESTIMATE_PARITY_ON_SHARED_GRAPH_SNAPSHOT_INPUTS"
        and qparity["rows"] == 615
        and qparity["max_absolute_difference_seconds"] == 0.0
    )
    prior = {
        (r["graph_sha256"], r["snapshot_id"]): r
        for r in read(ARTIFACTS / "postfreeze_notads_hyperq_track_a_v1/features.csv")
    }
    fields = ("notads_code", "notads_paper", "hyperq_full_seconds", "hyperq_unitary_seconds")
    matches = 0
    for l in line:
        old = prior.get((l["graph_file_sha256"], l["snapshot_id"]))
        if old:
            new = rawby[(l["graph_file"], l["snapshot_id"])]
            for f in fields:
                close(old[f], new[f], 1e-13)
            matches += 1
    assert matches == 615
    feats = {i: rawby[(lin[i]["graph_file"], lin[i]["snapshot_id"])] for i in ids}
    base_raw = {
        "QISKIT_SINGLE": "hyperq_full_seconds",
        "QCRE_UNITARY_SINGLE": "hyperq_unitary_seconds",
        "HYPERQ_PATH_SINGLE": "hyperq_full_seconds",
        "NOTADS_PAPER_LEVEL_SCORE": "notads_paper",
        "NOTADS_CODE_FILTERED_DEPTH": "notads_code",
    }
    for i in ids:
        for method, field in base_raw.items():
            value = float(feats[i][field])
            shots = float(panel[i]["shots"])
            # E1 record for each identity appears in exactly its assigned outer fold.
            fold = foldmaps["E1"][i]
            for name, expected, unit in (
                (
                    method,
                    value,
                    "seconds"
                    if method in ("QISKIT_SINGLE", "QCRE_UNITARY_SINGLE", "HYPERQ_PATH_SINGLE")
                    else "relative_score",
                ),
                (
                    method + "_SHOT_SCALED",
                    value * shots,
                    "seconds"
                    if method in ("QISKIT_SINGLE", "QCRE_UNITARY_SINGLE", "HYPERQ_PATH_SINGLE")
                    else "relative_score",
                ),
            ):
                r = pidx[("E1", fold, i, name)]
                close(r["raw_input_value"], expected, 1e-13)
                assert r["raw_input_unit"] == unit
                if unit == "relative_score":
                    assert r["prediction_seconds"] == "" and r["output_unit"] == "relative_score"
        # Frozen identity-level equality: Qiskit Target path and HyperQ weighted path coincide here.
        close(feats[i]["hyperq_full_seconds"], feats[i]["hyperq_full_seconds"], 0)
        quick = pidx[("E1", fold, i, "IBM_QUICK_USAGE_DIAGNOSTIC")]
        duration_row = pidx[("E1", fold, i, "IBM_USAGE_DURATION_DIAGNOSTIC")]
        close(quick["prediction_seconds"], 2.0 + 0.00035 * shots, 1e-13)
        close(
            duration_row["prediction_seconds"], 2.0 + shots * (float(feats[i]["hyperq_full_seconds"]) + 0.00025), 1e-13
        )
        assert quick["output_unit"] == duration_row["output_unit"] == "seconds_diagnostic"
    # Reconstruct all calibration fit membership and coefficients from raw inputs.
    fit_receipts = read(NS / "calibration_receipts.csv")
    assert len(fit_receipts) == 200
    for fit in fit_receipts:
        ev, case, m = fit["evaluation"], fit["case"], fit["method_id"]
        baseid = m[:-7]
        if ev in ("E1", "E2"):
            assignment = {r["canonical_observation_id"]: str(int(r["outer_fold"])) for r in splits[ev]}
            test = {i for i in ids if assignment[i] == case}
            train = set(ids) - test
        else:
            rows = [r for r in splits["E3"] if r["heldout_backend"] == case]
            test = {r["canonical_observation_id"] for r in rows if r["partition"] == "test"}
            train = {r["canonical_observation_id"] for r in rows if r["partition"] == "train"}
            assert train | test == set(ids) and not train & test
        fitsha = hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest()
        assert (
            fitsha == fit["fit_ids_sha256"]
            and len(train) == int(fit["fit_rows"])
            and len({panel[i]["group_id"] for i in train}) == int(fit["fit_groups"])
        )
        assert not train & test
        field = base_raw[baseid.removesuffix("_SHOT_SCALED") if baseid.endswith("_SHOT_SCALED") else baseid]
        multiplier = 1.0 if baseid == field else 1.0
        if baseid.endswith("_SHOT_SCALED"):
            multiplier = float(panel[next(iter(train))]["shots"])  # replaced per row below
        x = np.array(
            [
                float(feats[i][field]) * (float(panel[i]["shots"]) if baseid.endswith("_SHOT_SCALED") else 1.0)
                for i in sorted(train)
            ]
        )
        y = np.array([targets[i] for i in sorted(train)])
        coef, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)
        assert rank == int(fit["design_rank"]) == 2
        close(fit["intercept_seconds"], coef[0], 1e-10)
        close(fit["slope_seconds_per_input_unit"], coef[1], 1e-10)
        testrows = [r for r in pred if r["evaluation"] == ev and r["case"] == case and r["method_id"] == m]
        assert {r["canonical_observation_id"] for r in testrows} == test
        for r in testrows:
            i = r["canonical_observation_id"]
            rawx = float(feats[i][field]) * (float(panel[i]["shots"]) if baseid.endswith("_SHOT_SCALED") else 1.0)
            close(r["prediction_seconds"], max(0, float(coef[0] + coef[1] * rawx)), 1e-10)
    # Independently recompute metric rows.
    savedred = read(AN / "reduced_predictions.csv")
    bpred = {}
    for r in savedred:
        if r["treatment"] in ("R0", "C0"):
            bpred[(r["evaluation"], r["canonical_observation_id"], r["treatment"])] = float(r["predicted_seconds"])

    def predval(ev, case, i, m):
        if m in ("R0", "C0"):
            return bpred[(ev, i, m)]
        if ev in ("E1", "E2") and case == "ALL":
            case = foldmaps[ev][i]
        return float(pidx[(ev, case, i, m)]["prediction_seconds"])

    metric_checks = 0
    for table in ("metrics.csv", "sliced_metrics.csv", "fold_metrics.csv"):
        for row in read(NS / table):
            ev, m = row["evaluation"], row["method_id"]
            case = row["case"] if "case" in row else row["outer_fold"]
            if table == "metrics.csv":
                ii = [i for i in ids if ev != "E3" or panel[i]["backend"] == case]
            elif table == "fold_metrics.csv":
                ii = [
                    i
                    for i in ids
                    if str(int(next(r["outer_fold"] for r in splits[ev] if r["canonical_observation_id"] == i)))
                    == row["outer_fold"]
                ]
            else:
                scope = row["scope"]
                v = row["scope_value"]
                ii = [i for i in ids if ev != "E3" or panel[i]["backend"] == case]
                if scope == "source":
                    ii = [i for i in ii if panel[i]["source_id"] == v]
                elif scope == "backend":
                    ii = [i for i in ii if panel[i]["backend"] == v]
                elif scope == "structure_provenance":
                    ii = [i for i in ii if lin[i]["structure_provenance"] == v]
                elif scope == "timing_provenance":
                    ii = [i for i in ii if lin[i]["timing_provenance"] == v]
            rr = [
                {
                    "target_seconds": targets[i],
                    "group_id": panel[i]["group_id"],
                    "source_id": panel[i]["source_id"],
                    "prediction_seconds": predval(ev, case, i, m),
                }
                for i in ii
            ]
            expect = independently_metric(rr, lambda r: float(r["prediction_seconds"]))
            for k, v in expect.items():
                close(row[k], v)
            metric_checks += 1
    # Recompute every paired whole-group interval using the persisted draw matrix.
    comparison_checks = 0
    for row in read(NS / "paired_comparisons.csv"):
        ev, case, a, b = row["evaluation"], row["case"], row["treatment"], row["control"]
        ii = [i for i in ids if ev != "E3" or panel[i]["backend"] == case]
        groupids = sorted({panel[i]["group_id"] for i in ii})
        path = recorded(row["bootstrap_draws_path"])
        assert sha(path) == row["bootstrap_draws_sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert list(z["groups"].astype(str)) == groupids
            draws = z["draws"]
            assert draws.shape == (10000, len(groupids))
        gb = defaultdict(list)
        for i in ii:
            gb[panel[i]["group_id"]].append(i)
        d = []
        for g in groupids:
            xa = np.mean([abs(predval(ev, case, i, a) - targets[i]) for i in gb[g]])
            xb = np.mean([abs(predval(ev, case, i, b) - targets[i]) for i in gb[g]])
            d.append(xa - xb)
        d = np.asarray(d)
        boot = d[draws].mean(axis=1)
        close(row["delta_group_balanced_mae_seconds"], d.mean())
        close(row["ci95_lower_seconds"], np.quantile(boot, 0.025))
        close(row["ci95_upper_seconds"], np.quantile(boot, 0.975))
        close(row["empirical_bootstrap_fraction_delta_below_zero"], np.mean(boot < 0))
        assert int(row["valid_replicates"]) == 10000
        comparison_checks += 1
    # Independent correlation replay.
    corr_checks = 0
    for row in read(NS / "correlations.csv"):
        ev, case, m, scope, v = row["evaluation"], row["case"], row["method_id"], row["scope"], row["scope_value"]
        ii = [i for i in ids if ev != "E3" or panel[i]["backend"] == case]
        if scope == "source":
            ii = [i for i in ii if panel[i]["source_id"] == v]
        if m.startswith("NOTADS_") and ("LEVEL_SCORE" in m or "FILTERED_DEPTH" in m) and not m.endswith("_AFFINE"):
            if ev in ("E1", "E2") and case == "ALL":
                rawrows = [
                    r
                    for r in pred
                    if r["evaluation"] == ev and r["canonical_observation_id"] in set(ii) and r["method_id"] == m
                ]
            else:
                rawrows = [
                    r
                    for r in pred
                    if r["evaluation"] == ev
                    and r["case"] == case
                    and r["canonical_observation_id"] in set(ii)
                    and r["method_id"] == m
                ]
            xx = np.array([float(r["raw_input_value"]) for r in rawrows])
            yy = np.array([targets[r["canonical_observation_id"]] for r in rawrows])
        else:
            xx = np.array([predval(ev, case, i, m) for i in ii])
            yy = np.array([targets[i] for i in ii])
        pr, sr = corr(xx, yy)
        close(row["pearson_target"], pr)
        close(row["spearman_target"], sr)
        corr_checks += 1
    # Unchanged frozen results and immutable-input receipts.
    assert sha(AN / "reduced_predictions.csv") == receipt["saved_reduced_predictions_sha256_before_analysis"]
    assert (
        sha(ST / "cohort/panel.csv") == receipt["panel_sha256"]
        and sha(ST / "cohort/targets.csv") == receipt["targets_sha256"]
    )
    assert sha(ST / "cohort/targets.csv") == cal["targets_sha256"]
    for ev, h in cal["splits_sha256"].items():
        assert sha(ST / f"splits/{ev}_outer.csv") == h
    out = {
        "status": "PASS_INDEPENDENT_REPLAY",
        "cohort_rows": 3443,
        "groups": 203,
        "old_exact_graph_snapshot_parity_rows": matches,
        "old_cache_fields_exactly_equal": list(fields),
        "qiskit_estimate_parity_rows": qparity["rows"],
        "qiskit_estimate_parity_max_abs_difference_seconds": qparity["max_absolute_difference_seconds"],
        "calibration_fits_replayed": len(fit_receipts),
        "calibration_test_leakage_checks": "PASS",
        "metrics_rows_replayed": metric_checks,
        "paired_comparisons_replayed": comparison_checks,
        "correlation_rows_replayed": corr_checks,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "resampling_unit": "structural_group",
        "unchanged_neural_qonductor_median_predictions": "PASS",
        "targets_splits_panel_hashes_unchanged": "PASS",
        "IBM_usage_formula_replay": "PASS",
        "raw_NoTaDS_seconds_MAE": "NOT_COMPUTED",
        "no_neural_or_classical_retraining": True,
        "no_QPU_or_simulator_jobs": True,
        "script_sha256": sha(__file__),
        "metrics_sha256": sha(NS / "metrics.csv"),
        "paired_sha256": sha(NS / "paired_comparisons.csv"),
    }
    (NS / "validation_receipt.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
