#!/usr/bin/env python3
"""Read cohort features and labels separately, and construct evaluation partitions."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
SPLITS = OUT / "splits"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_targets(allowed_ids):
    allowed = set(allowed_ids)
    if not allowed:
        raise RuntimeError("refuse_empty_authorized_target_partition")
    target_path = OUT / "cohort/targets.csv"
    found = {}
    with target_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r["canonical_observation_id"]
            if cid not in allowed:
                continue
            y = float(r["target_seconds"])
            if not math.isfinite(y) or y <= 0:
                raise RuntimeError(f"invalid_authorized_target:{cid}")
            if cid in found:
                raise RuntimeError(f"duplicate_authorized_target:{cid}")
            found[cid] = y
    if set(found) != allowed:
        raise RuntimeError(f"authorized_target_population_incomplete:{len(found)}/{len(allowed)}")
    return found


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        import os

        os.fsync(f.fileno())
    import os

    os.replace(tmp, path)


def atomic_csv(path, values, fieldnames=None):
    if not values:
        raise RuntimeError(f"refuse_empty_csv:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(values[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(values)
        f.flush()
        import os

        os.fsync(f.fileno())
    import os

    os.replace(tmp, path)


def load_context(evaluation: str, include_targets: bool = True):
    if evaluation not in {"E1", "E2", "E3"}:
        raise ValueError(f"unknown_evaluation:{evaluation}")
    cohort = OUT / "cohort"
    panel = rows(cohort / "panel.csv")
    ledger_rows = rows(cohort / "feature_ledger.csv")
    panel_by_id = {r["canonical_observation_id"]: r for r in panel}
    ledger = {}
    for p, l in zip(panel, ledger_rows):
        if p["canonical_observation_id"] != l["canonical_observation_id"]:
            raise RuntimeError("panel_ledger_order_or_identity_mismatch")
        cid = p["canonical_observation_id"]
        ledger[cid] = {
            **l,
            "graph_file": l["graph_artifact"],
            "graph_artifact": l["graph_artifact"],
            "graph_artifact_sha256": l["graph_artifact_sha256"],
            "graph_node_count": l["graph_node_count"],
            "graph_edge_count": l["graph_edge_count"],
            "graph_status": "available",
            "global_status": "available",
            **{f"global_{i:02d}": l[f"global_{i:02d}"] for i in range(51)},
        }
    if len(panel) != 3443 or len(ledger) != 3443 or len(panel_by_id) != 3443:
        raise RuntimeError("expanded_panel_identity_mismatch")
    targets = None
    if include_targets:
        # Kept only for offline full-target validation/reporting. Training paths use
        # load_targets(outer_train_ids) and never materialize held-out values.
        targets = load_targets(set(ledger))
    if any(float(r["shots"]) <= 0 for r in panel):
        raise RuntimeError("invalid_shots")
    panel_ids = sorted(ledger)
    base = {
        "panel": panel,
        "panel_by_id": panel_by_id,
        "ledger": ledger,
        "feature_root": ROOT,
        "all_ids": set(panel_ids),
        "evaluation": evaluation,
        "pins": {
            "panel_sha256": sha(cohort / "panel.csv"),
            "target_sha256": sha(cohort / "targets.csv"),
            "feature_ledger_sha256": sha(cohort / "feature_ledger.csv"),
            "native_features_sha256": sha(cohort / "native_baseline_features.csv"),
            "feature_receipt_sha256": sha(cohort / "feature_preflight_receipt.json"),
            "split_receipt_sha256": sha(SPLITS / "split_receipt.json"),
        },
    }
    if evaluation in {"E1", "E2"}:
        outer = rows(SPLITS / f"{evaluation}_outer.csv")
        inner = rows(SPLITS / f"{evaluation}_inner.csv")
        om = {r["canonical_observation_id"]: r for r in outer}
        if len(om) != 3443 or set(om) != set(ledger):
            raise RuntimeError(f"split_outer_id_mismatch:{evaluation}")
        for fold in range(5):
            ir = [r for r in inner if int(r["outer_fold"]) == fold]
            expected = {cid for cid in ledger if int(om[cid]["outer_fold"]) != fold}
            if {r["canonical_observation_id"] for r in ir} != expected:
                raise RuntimeError(f"split_inner_train_identity_mismatch:{evaluation}:{fold}")
        base.update({"outer": om, "inner": inner, "case_count": 5})
    else:
        outer = rows(SPLITS / "E3_outer.csv")
        inner = rows(SPLITS / "E3_inner.csv")
        cases = sorted({r["heldout_backend"] for r in outer})
        if len(cases) != 10 or len(outer) != 3443 * 10:
            raise RuntimeError("E3_outer_case_identity_mismatch")
        base.update({"outer": outer, "inner": inner, "cases": cases, "case_count": len(cases)})
    receipt = {
        "evaluation": evaluation,
        "rows": len(ledger),
        "cases": base["case_count"],
        "target_values_read": bool(include_targets),
        "target_hash_pinned_but_labels_separate": True,
        "outer_split_sha256": sha(SPLITS / f"{evaluation}_outer.csv"),
        "inner_split_sha256": sha(SPLITS / f"{evaluation}_inner.csv"),
        "pins": base["pins"],
    }
    return base, targets, receipt


def parts_for(context, fold):
    ev = context["evaluation"]
    all_ids = context["all_ids"]
    group = {r["canonical_observation_id"]: r["group_id"] for r in context["panel"]}
    if ev in {"E1", "E2"}:
        outer = context["outer"]
        test = sorted(cid for cid, r in outer.items() if int(r["outer_fold"]) == fold)
        train = sorted(all_ids - set(test))
        ir = [r for r in context["inner"] if int(r["outer_fold"]) == fold]
        assignment = {r["canonical_observation_id"]: int(r["inner_fold"]) for r in ir}
    else:
        b = context["cases"][fold]
        outer = [r for r in context["outer"] if r["heldout_backend"] == b]
        test = sorted(r["canonical_observation_id"] for r in outer if r["partition"] == "test")
        train = sorted(r["canonical_observation_id"] for r in outer if r["partition"] == "train")
        ir = [r for r in context["inner"] if r["heldout_backend"] == b]
        assignment = {r["canonical_observation_id"]: int(r["inner_fold"]) for r in ir}
    if set(train) & set(test) or set(train) | set(test) != all_ids:
        raise RuntimeError(f"outer_partition_invalid:{ev}:{fold}")
    if set(assignment) != set(train) or set(assignment.values()) != {0, 1, 2, 3}:
        raise RuntimeError(f"inner_assignment_invalid:{ev}:{fold}")
    fit = sorted(c for c in train if assignment[c] != 0)
    validation = sorted(c for c in train if assignment[c] == 0)
    if set(fit) & set(validation) or set(fit) & set(test) or set(validation) & set(test):
        raise RuntimeError(f"id_overlap:{ev}:{fold}")
    if ev in {"E1", "E3"} and {group[c] for c in fit} & {group[c] for c in validation}:
        raise RuntimeError(f"inner_group_leakage:{ev}:{fold}")
    if ev == "E1" and {group[c] for c in train} & {group[c] for c in test}:
        raise RuntimeError(f"outer_group_leakage:{ev}:{fold}")
    inner_cv = {
        k: (sorted(c for c in train if assignment[c] != k), sorted(c for c in train if assignment[c] == k))
        for k in range(4)
    }
    return {
        "train": train,
        "fit": fit,
        "validation": validation,
        "test": test,
        "assignments": assignment,
        "inner_cv": inner_cv,
    }
