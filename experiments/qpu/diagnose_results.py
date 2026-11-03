#!/usr/bin/env python3
"""Post-score support/error diagnostics from existing inputs only."""

import collections
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from core import OUT, ROOT, rows, atomic_csv, atomic_json, sha, load_context, parts_for
from analysis_core import metrics


def main():
    dest = OUT / "analysis_v1/diagnostics"
    if dest.exists():
        raise RuntimeError("Refuse diagnostics overwrite")
    dest.mkdir(parents=True)
    pred = rows(OUT / "analysis_v1/reduced_predictions.csv")
    panel = {r["canonical_observation_id"]: r for r in rows(OUT / "cohort/panel.csv")}
    support = {}
    case_rows = []
    history = []
    corrs = []
    breakdown = []
    for ev in ("E1", "E2", "E3"):
        context, _, _ = load_context(ev, include_targets=False)
        for fold in range(context["case_count"]):
            parts = parts_for(context, fold)
            fit = [panel[c] for c in parts["fit"]]
            train = [panel[c] for c in parts["train"]]
            fg = {r["group_id"] for r in fit}
            tg = {r["group_id"] for r in train}
            fb = {(r["group_id"], r["backend"]) for r in fit}
            fbs = {(r["group_id"], r["backend"], float(r["shots"])) for r in fit}
            fshots = {float(r["shots"]) for r in fit}
            raw = {c: tuple(context["ledger"][c][f"global_{i:02d}"] for i in range(51)) for c in context["all_ids"]}
            fit_vectors = {raw[c] for c in parts["fit"]}
            full_fit = {(raw[c], context["ledger"][c]["graph_artifact_sha256"]) for c in parts["fit"]}
            sc = collections.Counter()
            for c in parts["test"]:
                r = panel[c]
                flag = {
                    "fit_same_group": r["group_id"] in fg,
                    "outer_train_same_group": r["group_id"] in tg,
                    "fit_same_group_backend": (r["group_id"], r["backend"]) in fb,
                    "fit_same_group_backend_shots": (r["group_id"], r["backend"], float(r["shots"])) in fbs,
                    "fit_seen_shot_value": float(r["shots"]) in fshots,
                    "fit_exact_raw_global51": raw[c] in fit_vectors,
                    "fit_exact_raw_globals_and_graph_bytes": (raw[c], context["ledger"][c]["graph_artifact_sha256"])
                    in full_fit,
                }
                support[ev, c] = flag
                sc.update(k for k, v in flag.items() if v)
            case_rows.append(
                {
                    "evaluation": ev,
                    "outer_fold": fold,
                    "heldout_backend": context["cases"][fold] if ev == "E3" else "",
                    "fit_rows": len(fit),
                    "fit_groups": len(fg),
                    "test_rows": len(parts["test"]),
                    **{f"test_rows_{k}": sc[k] for k in flag},
                }
            )
            for m in ("M0", "M1", "G0", "G1"):
                for s in (42, 1234, 31415):
                    folder = OUT / "training" / ev / f"fold_{fold}" / m / f"seed_{s}"
                    man = json.loads((folder / "manifest.json").read_text())
                    h = rows(folder / "epoch_history.csv")
                    history.append(
                        {
                            "evaluation": ev,
                            "outer_fold": fold,
                            "treatment": m,
                            "seed": s,
                            "selected_epoch": man["best_epoch"],
                            "parameter_count": man["parameter_count"],
                            "first_train_mse": float(h[0]["train_mse"]),
                            "last_train_mse": float(h[-1]["train_mse"]),
                            "first_validation_mse": float(h[0]["validation_mse"]),
                            "selected_validation_mse": man["best_validation_mse"],
                            "last_validation_mse": float(h[-1]["validation_mse"]),
                            "cell_elapsed_seconds": float(h[-1]["elapsed_seconds"]),
                        }
                    )
    for ev in ("E1", "E2", "E3"):
        for m in ("M0", "M1", "G0", "G1", "R0", "C0"):
            records = [r for r in pred if r["evaluation"] == ev and r["treatment"] == m]
            for scope in ["all"] + sorted({r["source_id"] for r in records}):
                rr = records if scope == "all" else [r for r in records if r["source_id"] == scope]
                y = np.array([float(r["target_seconds"]) for r in rr])
                p = np.array([float(r["predicted_seconds"]) for r in rr])
                valid = len(y) > 2 and np.std(y) > 0 and np.std(p) > 0
                corrs.append(
                    {
                        "evaluation": ev,
                        "treatment": m,
                        "source": scope,
                        "rows": len(rr),
                        "pearson_r": float(pearsonr(y, p).statistic) if valid else "",
                        "spearman_rho": float(spearmanr(y, p).statistic) if valid else "",
                        "status": "DEFINED_DESCRIPTIVE_REPEATED_ROWS"
                        if valid
                        else "UNDEFINED_CONSTANT_OR_SMALL_SAMPLE",
                        "not_independence_adjusted_inference": True,
                    }
                )
            for feature in next(iter(support.values())):
                for yes in (True, False):
                    rr = [r for r in records if support[ev, r["canonical_observation_id"]][feature] == yes]
                    if not rr:
                        continue
                    y = [float(r["target_seconds"]) for r in rr]
                    p = [float(r["predicted_seconds"]) for r in rr]
                    breakdown.append(
                        {
                            "evaluation": ev,
                            "treatment": m,
                            "support_feature": feature,
                            "support_present": yes,
                            **metrics(y, p, [r["group_id"] for r in rr], [r["source_id"] for r in rr]),
                        }
                    )
    atomic_csv(dest / "case_input_support.csv", case_rows)
    atomic_csv(dest / "training_curve_summary.csv", history)
    atomic_csv(dest / "correlations.csv", corrs)
    atomic_csv(dest / "support_error_slices.csv", breakdown)
    atomic_json(
        dest / "receipt.json",
        {
            "status": "COMPLETE_DESCRIPTIVE_ONLY",
            "script_sha256": sha(__file__),
            "cases": len(case_rows),
            "training_histories": len(history),
            "correlation_rows": len(corrs),
            "no_new_features_fit_models_or_labels": True,
            "identity_scope": "Exact graph-byte equality is conservative; equivalent graphs in different byte encodings may be missed. Raw global51 overlap differs from fold-retained standardized features.",
            "correlations": "Pooled repeated rows, no correlation p-values or causal attribution.",
        },
    )


if __name__ == "__main__":
    main()
