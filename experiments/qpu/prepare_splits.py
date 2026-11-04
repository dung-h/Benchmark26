#!/usr/bin/env python3
"""Create target-blind E1 logical-holdout, E2 source-random and E3 backend-holdout folds."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS, record_name
from pathlib import Path
import csv, hashlib, json, os, platform, subprocess
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
DATASET_DIR = OUT / "cohort"
SPLITS = OUT / "splits"
SEED = 42


def sha(path):
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    """Read a UTF-8 CSV file into a list of dicts."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rs):
    """Write rows as a CSV file; raise if there are none."""
    if not rs:
        raise RuntimeError(f"empty_split:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)


def digest_ids(ids):
    """Return the SHA-256 of the sorted IDs joined by newlines."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def assign_groups(group_rows, source_by_id, folds, seed):
    """Source-row-balanced deterministic greedy group assignment; labels only."""
    labels = sorted(set(source_by_id.values()))
    totals = Counter(source_by_id[rid] for ids in group_rows.values() for rid in ids)
    loads = [{s: 0 for s in labels} for _ in range(folds)]
    assignment = {}
    ordered = sorted(
        group_rows,
        key=lambda g: (
            -max(Counter(source_by_id[rid] for rid in group_rows[g])[s] / totals[s] for s in labels if totals[s]),
            -len(group_rows[g]),
            g,
        ),
    )
    for group in ordered:
        vec = Counter(source_by_id[rid] for rid in group_rows[group])
        choices = []
        for target_fold in range(folds):
            cost = 0.0
            for f in range(folds):
                for s in labels:
                    value = loads[f][s] + (vec[s] if f == target_fold else 0)
                    ideal = totals[s] / folds
                    cost += ((value - ideal) / max(ideal, 1.0)) ** 2
            tie = hashlib.sha256(f"{seed}:{group}:{target_fold}".encode()).hexdigest()
            choices.append((cost, tie, target_fold))
        f = min(choices)[2]
        assignment[group] = f
        for s in labels:
            loads[f][s] += vec[s]
    return assignment, loads


def make_grouped(eligible_ids, group_by_id, source_by_id, folds, seed, tag):
    """Assign whole groups to folds with `assign_groups`, balanced across sources, using a seed.

    Returns the fold of each observation, the loads and the groups. Raises if there are fewer groups than
    folds.
    """
    groups = defaultdict(list)
    for cid in eligible_ids:
        groups[group_by_id[cid]].append(cid)
    if len(groups) < folds:
        raise RuntimeError(f"not_enough_groups:{tag}:{len(groups)}")
    assign, loads = assign_groups(groups, source_by_id, folds, seed)
    return {cid: assign[group_by_id[cid]] for cid in eligible_ids}, loads, groups


def main():
    """Create the target-blind folds of the study.

    E1 holds out logical-circuit groups, E2 holds out random observations stratified by source, and E3 holds
    out backends. E1 and E2 have 5 outer folds and E3 has one case per held-out backend; all have inner folds. Writes the
    outer and inner split files and a receipt. Refuses to run if the splits already exist.
    """
    if (SPLITS / "E1_outer.csv").exists():
        raise RuntimeError("refuse_overwrite_locked_splits")
    run_rows = rows(DATASET_DIR / "panel.csv")
    ids = [r["canonical_observation_id"] for r in run_rows]
    if len(ids) != 3443 or len(set(ids)) != 3443:
        raise RuntimeError("cohort_id_count_mismatch")
    group = {r["canonical_observation_id"]: r["group_id"] for r in run_rows}
    source = {r["canonical_observation_id"]: r["source_id"] for r in run_rows}
    backend = {r["canonical_observation_id"]: r["backend"] for r in run_rows}
    source_counts = Counter(source.values())
    groups = defaultdict(list)
    for cid, gid in group.items():
        groups[gid].append(cid)
    if len(groups) != 203:
        raise RuntimeError("final_component_count_mismatch")
    out_receipts = {}
    all_outputs = []

    # E1: held-out logical/group components.
    e1, loads, _ = make_grouped(ids, group, source, 5, 42, "E1")
    e1_outer = [
        {
            "canonical_observation_id": cid,
            "group_id": group[cid],
            "source_id": source[cid],
            "backend": backend[cid],
            "outer_fold": e1[cid],
        }
        for cid in sorted(ids)
    ]
    e1_inner = []
    e1_receipts = []
    for fold in range(5):
        train = [cid for cid in ids if e1[cid] != fold]
        inner, iloads, igroups = make_grouped(train, group, source, 4, 43 + fold, f"E1-inner-{fold}")
        test = [cid for cid in ids if e1[cid] == fold]
        fit = [cid for cid in train if inner[cid] != 0]
        val = [cid for cid in train if inner[cid] == 0]
        if set(fit) & set(val) or set(train) & set(test):
            raise RuntimeError(f"E1_leakage:{fold}")
        for cid in train:
            e1_inner.append(
                {
                    "canonical_observation_id": cid,
                    "group_id": group[cid],
                    "outer_fold": fold,
                    "inner_fold": inner[cid],
                    "source_id": source[cid],
                }
            )
        if {group[cid] for cid in fit} & {group[cid] for cid in val}:
            raise RuntimeError(f"E1_inner_group_overlap:{fold}")
        if {group[cid] for cid in train} & {group[cid] for cid in test}:
            raise RuntimeError(f"E1_outer_group_overlap:{fold}")
        e1_receipts.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(val),
                "test_rows": len(test),
                "fit_groups": len({group[c] for c in fit}),
                "validation_groups": len({group[c] for c in val}),
                "test_groups": len({group[c] for c in test}),
                "fit_source_rows": dict(Counter(source[c] for c in fit)),
                "validation_source_rows": dict(Counter(source[c] for c in val)),
                "test_source_rows": dict(Counter(source[c] for c in test)),
                "fit_ids_sha256": digest_ids(fit),
                "validation_ids_sha256": digest_ids(val),
                "test_ids_sha256": digest_ids(test),
            }
        )
    write(SPLITS / "E1_outer.csv", e1_outer)
    write(SPLITS / "E1_inner.csv", e1_inner)
    out_receipts["E1"] = {
        "estimand": "held-out logical structural groups",
        "outer_folds": 5,
        "outer_seed": 42,
        "inner_folds": 4,
        "inner_seeds": [43, 44, 45, 46, 47],
        "group_disjoint": True,
        "folds": e1_receipts,
        "source_row_loads": loads,
    }
    all_outputs += [SPLITS / "E1_outer.csv", SPLITS / "E1_inner.csv"]

    # E2: random held-out observations, stratified only by source.
    order = sorted(ids)
    labels = np.asarray([source[i] for i in order])
    e2 = {}
    for f, (_, test_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(order)), labels)
    ):
        for j in test_idx:
            e2[order[int(j)]] = f
    e2_outer = [
        {
            "canonical_observation_id": cid,
            "group_id": group[cid],
            "source_id": source[cid],
            "backend": backend[cid],
            "outer_fold": e2[cid],
        }
        for cid in order
    ]
    e2_inner = []
    e2_receipts = []
    for fold in range(5):
        train = sorted(cid for cid in ids if e2[cid] != fold)
        test = sorted(cid for cid in ids if e2[cid] == fold)
        inner = {}
        for k, (fitidx, validx) in enumerate(
            StratifiedKFold(n_splits=4, shuffle=True, random_state=43 + fold).split(
                np.zeros(len(train)), np.asarray([source[c] for c in train])
            )
        ):
            for j in validx:
                inner[train[int(j)]] = k
        fit = [c for c in train if inner[c] != 0]
        val = [c for c in train if inner[c] == 0]
        for cid in train:
            e2_inner.append(
                {
                    "canonical_observation_id": cid,
                    "group_id": group[cid],
                    "outer_fold": fold,
                    "inner_fold": inner[cid],
                    "source_id": source[cid],
                }
            )
        e2_receipts.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(val),
                "test_rows": len(test),
                "fit_source_rows": dict(Counter(source[c] for c in fit)),
                "validation_source_rows": dict(Counter(source[c] for c in val)),
                "test_source_rows": dict(Counter(source[c] for c in test)),
                "test_groups_seen_in_fit": len({group[c] for c in test} & {group[c] for c in fit}),
                "group_overlap_intentional": True,
                "fit_ids_sha256": digest_ids(fit),
                "validation_ids_sha256": digest_ids(val),
                "test_ids_sha256": digest_ids(test),
            }
        )
    write(SPLITS / "E2_outer.csv", e2_outer)
    write(SPLITS / "E2_inner.csv", e2_inner)
    out_receipts["E2"] = {
        "estimand": "source-stratified random observation interpolation",
        "outer_folds": 5,
        "outer_seed": 42,
        "inner_folds": 4,
        "inner_seeds": [43, 44, 45, 46, 47],
        "stratification": "source_id only",
        "structural_group_disjoint": False,
        "folds": e2_receipts,
    }
    all_outputs += [SPLITS / "E2_outer.csv", SPLITS / "E2_inner.csv"]

    # E3: leave one backend out. Inner selection remains group-disjoint.
    e3outer = []
    e3inner = []
    e3_receipts = []
    for bi, b in enumerate(sorted(set(backend.values()))):
        tag = bi
        test = sorted(cid for cid in ids if backend[cid] == b)
        train = sorted(set(ids) - set(test))
        if not test or not train:
            raise RuntimeError(f"empty_E3_partition:{b}")
        inner, iloads, igroups = make_grouped(train, group, source, 4, 100 + bi, f"E3-{b}")
        fit = [c for c in train if inner[c] != 0]
        val = [c for c in train if inner[c] == 0]
        e3outer.extend(
            {
                "canonical_observation_id": c,
                "group_id": group[c],
                "source_id": source[c],
                "backend": backend[c],
                "heldout_backend": b,
                "outer_fold": tag,
                "partition": "test" if c in set(test) else "train",
            }
            for c in ids
        )
        e3inner.extend(
            {
                "canonical_observation_id": c,
                "group_id": group[c],
                "source_id": source[c],
                "heldout_backend": b,
                "outer_fold": tag,
                "inner_fold": inner[c],
            }
            for c in train
        )
        test_groups = {group[c] for c in test}
        train_groups = {group[c] for c in train}
        e3_receipts.append(
            {
                "heldout_backend": b,
                "fold": tag,
                "fit_rows": len(fit),
                "validation_rows": len(val),
                "test_rows": len(test),
                "fit_groups": len({group[c] for c in fit}),
                "validation_groups": len({group[c] for c in val}),
                "test_groups": len(test_groups),
                "same_logical_groups_in_train_and_test": len(train_groups & test_groups),
                "test_source_rows": dict(Counter(source[c] for c in test)),
                "train_source_rows": dict(Counter(source[c] for c in train)),
                "fit_ids_sha256": digest_ids(fit),
                "validation_ids_sha256": digest_ids(val),
                "test_ids_sha256": digest_ids(test),
                "backend_label_used_for_split_only": True,
                "heldout_target_used": False,
            }
        )
    write(SPLITS / "E3_outer.csv", e3outer)
    write(SPLITS / "E3_inner.csv", e3inner)
    out_receipts["E3"] = {
        "estimand": "leave-one-backend-out context transfer; same structures on other backends allowed",
        "heldout_backends": sorted(set(backend.values())),
        "outer_cases": 10,
        "inner_folds": 4,
        "inner_group_disjoint": True,
        "folds": e3_receipts,
    }
    all_outputs += [SPLITS / "E3_outer.csv", SPLITS / "E3_inner.csv"]

    # Check exactly one outer test prediction opportunity for every observation in E1/E2/E3.
    for name, assignment in (("E1", e1), ("E2", e2)):
        if set(assignment) != set(ids) or set(assignment.values()) != set(range(5)):
            raise RuntimeError(f"outer_coverage:{name}")
    if len(e3outer) != len(ids) * 10 or len(e3inner) != sum(
        len(ids) - len([c for c in ids if backend[c] == b]) for b in set(backend.values())
    ):
        raise RuntimeError("E3_assignment_row_count_mismatch")
    run_table_sha256 = sha(DATASET_DIR / "panel.csv")
    feature_sha = sha(DATASET_DIR / "feature_ledger.csv")
    receipt = {
        "status": "PASS_TARGET_BLIND_SPLITS_PREPARED",
        "rows": len(ids),
        "groups": len(groups),
        "source_counts": dict(source_counts),
        "backend_count": len(set(backend.values())),
        "outer_test_assignments_per_observation": {"E1": 1, "E2": 1, "E3": "one per row's backend"},
        "target_values_loaded_or_used": False,
        "group_definition_source": "target-blind audited cohort panel; logical signature/retained group/physical hash union plus no cross-group full feature collisions",
        "split_seed": 42,
        "studies": out_receipts,
        "input_sha256": {"panel.csv": run_table_sha256, "feature_ledger.csv": feature_sha},
        "outputs_sha256": {record_name(p): sha(p) for p in all_outputs},
        "code_sha256": sha(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    p = SPLITS / "split_receipt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "rows": len(ids),
                "groups": len(groups),
                "source_counts": dict(source_counts),
                "backend_count": receipt["backend_count"],
                "E1": e1_receipts,
                "E2": e2_receipts,
                "E3": e3_receipts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
