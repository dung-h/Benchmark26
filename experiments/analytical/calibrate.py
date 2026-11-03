#!/usr/bin/env python3
"""Apply frozen scalar methods and fit outer-train-only affine adaptations."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
P = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"


def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(p, rs):
    if not rs:
        raise RuntimeError("empty_output")
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rs)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw = read(NS / "raw_features.csv")
    lin = read(NS / "lineage.csv")
    target_rows = read(P / "cohort/targets.csv")
    panel = read(P / "cohort/panel.csv")
    ids = {r["canonical_observation_id"] for r in panel}
    targets = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in target_rows}
    if len(ids) != 3443 or set(ids) != set(targets) or len(lin) != 3443:
        raise RuntimeError("cohort_or_target_alignment")
    feat = {(r["graph_file"], r["snapshot_id"]): r for r in raw}
    line = {r["canonical_observation_id"]: r for r in lin}
    meta = {r["canonical_observation_id"]: r for r in panel}
    if len(feat) != len(raw):
        raise RuntimeError("duplicate_raw_graph_snapshot")
    inputs = {
        cid: {
            **meta[cid],
            **line[cid],
            **{
                k: v
                for k, v in feat[(line[cid]["graph_file"], line[cid]["snapshot_id"])].items()
                if k not in ("graph_file", "snapshot_id", "graph_file_sha256")
            },
        }
        for cid in ids
    }
    for cid, x in inputs.items():
        if (
            x["group_id"] != meta[cid]["group_id"]
            or x["backend"] != meta[cid]["backend"]
            or float(x["shots"]) != float(meta[cid]["shots"])
        ):
            raise RuntimeError("joined_metadata_mismatch:" + cid)
        if not math.isfinite(targets[cid]) or targets[cid] <= 0:
            raise RuntimeError("invalid_target:" + cid)
    values = {}
    raw_base_ids = {
        "QISKIT_RAW": "QISKIT_SINGLE",
        "QCRE_RAW": "QCRE_UNITARY_SINGLE",
        "HYPERQ_RAW": "HYPERQ_PATH_SINGLE",
        "NOTADS_PAPER": "NOTADS_PAPER_LEVEL_SCORE",
        "NOTADS_CODE": "NOTADS_CODE_FILTERED_DEPTH",
    }
    for cid, x in inputs.items():
        sh = float(x["shots"])
        vals = {
            "QISKIT_RAW": (float(x["hyperq_full_seconds"]), "seconds"),
            "QCRE_RAW": (float(x["hyperq_unitary_seconds"]), "seconds"),
            "HYPERQ_RAW": (float(x["hyperq_full_seconds"]), "seconds"),
            "NOTADS_PAPER": (float(x["notads_paper"]), "relative_score"),
            "NOTADS_CODE": (float(x["notads_code"]), "relative_score"),
        }
        for mid, (v, unit) in vals.items():
            if not math.isfinite(v) or v < 0:
                raise RuntimeError(f"invalid_raw:{cid}:{mid}")
            route = raw_base_ids[mid]
            values[cid, route] = (v, unit)
            values[cid, route + "_SHOT_SCALED"] = (v * sh, unit)
        ibm = 2.0 + 0.00035 * sh
        values[cid, "IBM_USAGE_QUICK"] = (ibm, "seconds_diagnostic")
        ibm_duration = 2.0 + sh * (values[cid, "QISKIT_SINGLE"][0] + 0.00025)
        values[cid, "IBM_USAGE_DURATION"] = (ibm_duration, "seconds_diagnostic")
    split = {e: read(P / f"splits/{e}_outer.csv") for e in ("E1", "E2", "E3")}
    # Each case supplies disjoint outer-train and test identity sets.
    cases = []
    for e in ("E1", "E2"):
        d = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in split[e]}
        if len(d) != 3443 or set(d) != ids:
            raise RuntimeError(f"split_identity:{e}")
        for fold in sorted(set(d.values())):
            test = {i for i, f in d.items() if f == fold}
            train = ids - test
            if len(train & test) or not test:
                raise RuntimeError(f"outer_split_invalid:{e}:{fold}")
            cases.append((e, str(fold), train, test))
    erows = split["E3"]
    contexts = sorted({r["heldout_backend"] for r in erows})
    for context in contexts:
        rr = [r for r in erows if r["heldout_backend"] == context]
        train = {r["canonical_observation_id"] for r in rr if r["partition"] == "train"}
        test = {r["canonical_observation_id"] for r in rr if r["partition"] == "test"}
        if train & test or train | test != ids or not test:
            raise RuntimeError(f"E3_partition_invalid:{context}")
        cases.append(("E3", context, train, test))
    bases = [name for route in raw_base_ids.values() for name in (route, route + "_SHOT_SCALED")]
    predictions = []
    calreceipts = []
    for evaluation, case, train, test in cases:
        yfit = np.array([targets[i] for i in sorted(train)], dtype=float)
        for base in bases:
            xx = np.array([values[i, base][0] for i in sorted(train)], dtype=float)
            unit = values[next(iter(test)), base][1]
            if not np.isfinite(xx).all() or not np.isfinite(yfit).all():
                raise RuntimeError(f"calibration_nonfinite:{evaluation}:{case}:{base}")
            coef, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(xx)), xx]), yfit, rcond=None)
            if rank < 2 or not np.isfinite(coef).all():
                raise RuntimeError(f"calibration_rank_failure:{evaluation}:{case}:{base}")
            fitid = hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest()
            calreceipts.append(
                {
                    "evaluation": evaluation,
                    "case": case,
                    "method_id": base + "_AFFINE",
                    "fit_rows": len(train),
                    "fit_groups": len({meta[i]["group_id"] for i in train}),
                    "fit_ids_sha256": fitid,
                    "intercept_seconds": float(coef[0]),
                    "slope_seconds_per_input_unit": float(coef[1]),
                    "design_rank": int(rank),
                    "target_used": "outer_train_only",
                    "rule": "max(0,a+b*x)",
                }
            )
            for cid in sorted(test):
                x = inputs[cid]
                v, unit0 = values[cid, base]
                pred = max(0.0, float(coef[0] + coef[1] * v))
                predictions.append(
                    {
                        "evaluation": evaluation,
                        "case": case,
                        "canonical_observation_id": cid,
                        "group_id": x["group_id"],
                        "source_id": x["source_id"],
                        "backend": x["backend"],
                        "shots": x["shots"],
                        "target_seconds": targets[cid],
                        "method_id": base + "_AFFINE",
                        "prediction_seconds": pred,
                        "raw_input_value": v,
                        "raw_input_unit": unit0,
                        "output_unit": "seconds",
                        "fit_rows": len(train),
                        "fit_ids_sha256": fitid,
                    }
                )
        for cid in sorted(test):
            x = inputs[cid]
            for base in bases:
                v, u = values[cid, base]
                unit_out = "seconds" if u == "seconds" else "relative_score"
                predictions.append(
                    {
                        "evaluation": evaluation,
                        "case": case,
                        "canonical_observation_id": cid,
                        "group_id": x["group_id"],
                        "source_id": x["source_id"],
                        "backend": x["backend"],
                        "shots": x["shots"],
                        "target_seconds": targets[cid],
                        "method_id": base,
                        "prediction_seconds": v if unit_out == "seconds" else "",
                        "raw_input_value": v,
                        "raw_input_unit": u,
                        "output_unit": unit_out,
                        "fit_rows": 0,
                        "fit_ids_sha256": "",
                    }
                )
            v, u = values[cid, "IBM_USAGE_QUICK"]
            predictions.append(
                {
                    "evaluation": evaluation,
                    "case": case,
                    "canonical_observation_id": cid,
                    "group_id": x["group_id"],
                    "source_id": x["source_id"],
                    "backend": x["backend"],
                    "shots": x["shots"],
                    "target_seconds": targets[cid],
                    "method_id": "IBM_QUICK_USAGE_DIAGNOSTIC",
                    "prediction_seconds": v,
                    "raw_input_value": v,
                    "raw_input_unit": u,
                    "output_unit": "seconds_diagnostic",
                    "fit_rows": 0,
                    "fit_ids_sha256": "",
                }
            )
            v, u = values[cid, "IBM_USAGE_DURATION"]
            predictions.append(
                {
                    "evaluation": evaluation,
                    "case": case,
                    "canonical_observation_id": cid,
                    "group_id": x["group_id"],
                    "source_id": x["source_id"],
                    "backend": x["backend"],
                    "shots": x["shots"],
                    "target_seconds": targets[cid],
                    "method_id": "IBM_USAGE_DURATION_DIAGNOSTIC",
                    "prediction_seconds": v,
                    "raw_input_value": v,
                    "raw_input_unit": u,
                    "output_unit": "seconds_diagnostic",
                    "fit_rows": 0,
                    "fit_ids_sha256": "",
                }
            )
    write(NS / "calibration_receipts.csv", calreceipts)
    write(NS / "predictions.csv", predictions)
    rec = {
        "status": "PASS_OUTER_TRAIN_CALIBRATION",
        "observations": len(ids),
        "groups": len(set(r["group_id"] for r in panel)),
        "cases": len(cases),
        "calibration_fits": len(calreceipts),
        "prediction_rows": len(predictions),
        "test_labels_used_for_feature_extraction": False,
        "fit_rule": "ordinary least squares a+b*x on outer-train only; max(0,a+b*x) at prediction",
        "raw_NoTaDS_scored_in_seconds": False,
        "no_neural_or_classical_retraining": True,
        "no_QPU_or_simulator_jobs": True,
        "targets_sha256": sha(P / "cohort/targets.csv"),
        "splits_sha256": {e: sha(P / f"splits/{e}_outer.csv") for e in ("E1", "E2", "E3")},
        "predictions_sha256": sha(NS / "predictions.csv"),
        "calibration_receipts_sha256": sha(NS / "calibration_receipts.csv"),
    }
    (NS / "calibration_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
