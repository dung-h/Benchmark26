#!/usr/bin/env python3
"""Recheck every locked fold without reading any runtime target values."""

from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from core import OUT, atomic_json, load_context, parts_for, sha


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=OUT / "splits/split_validation_receipt.json",
        help="receipt path (default: inside the study bundle)",
    )
    receipt = parser.parse_args().receipt
    audit = {"status": "PASS_TARGET_FREE_SPLIT_VALIDATION", "targets_read": False, "evaluations": {}}
    for ev in ("E1", "E2", "E3"):
        ctx, _targets, _ = load_context(ev, include_targets=False)
        all_test = []
        cases = []
        group = {r["canonical_observation_id"]: r["group_id"] for r in ctx["panel"]}
        source = {r["canonical_observation_id"]: r["source_id"] for r in ctx["panel"]}
        backend = {r["canonical_observation_id"]: r["backend"] for r in ctx["panel"]}
        for fold in range(ctx["case_count"]):
            p = parts_for(ctx, fold)
            all_test.extend(p["test"])
            if set(p["fit"]) & set(p["validation"]):
                raise RuntimeError(f"fit_validation_overlap:{ev}:{fold}")
            if set(p["fit"]) & set(p["test"]) or set(p["validation"]) & set(p["test"]):
                raise RuntimeError(f"test_overlap:{ev}:{fold}")
            if ev in {"E1", "E3"} and {group[c] for c in p["fit"]} & {group[c] for c in p["validation"]}:
                raise RuntimeError(f"inner_group_overlap:{ev}:{fold}")
            if ev == "E1" and {group[c] for c in p["train"]} & {group[c] for c in p["test"]}:
                raise RuntimeError(f"outer_group_overlap:{fold}")
            if ev == "E3":
                held = ctx["cases"][fold]
                if any(backend[c] != held for c in p["test"]) or any(backend[c] == held for c in p["train"]):
                    raise RuntimeError(f"backend_holdout_partition_error:{held}")
            cases.append(
                {
                    "fold": fold,
                    "fit_rows": len(p["fit"]),
                    "validation_rows": len(p["validation"]),
                    "test_rows": len(p["test"]),
                    "fit_groups": len({group[c] for c in p["fit"]}),
                    "validation_groups": len({group[c] for c in p["validation"]}),
                    "test_groups": len({group[c] for c in p["test"]}),
                    "same_group_fit_test": len({group[c] for c in p["fit"]} & {group[c] for c in p["test"]}),
                    "test_source_rows": dict(Counter(source[c] for c in p["test"])),
                    "heldout_backend": ctx["cases"][fold] if ev == "E3" else "",
                }
            )
        if ev in {"E1", "E2"} and (len(all_test) != 3443 or len(set(all_test)) != 3443):
            raise RuntimeError(f"oof_identity_coverage_error:{ev}")
        if ev == "E3":
            if len(all_test) != 3443 or len(set(all_test)) != 3443:
                raise RuntimeError("E3_each_row_not_tested_once_by_own_backend")
        audit["evaluations"][ev] = {
            "case_count": len(cases),
            "cases": cases,
            "outer_test_rows_total": len(all_test),
            "unique_rows_tested_once": len(set(all_test)),
            "outer_fold_file_sha256": sha(OUT / "splits" / (f"{ev}_outer.csv")),
            "inner_fold_file_sha256": sha(OUT / "splits" / (f"{ev}_inner.csv")),
        }
    atomic_json(receipt, audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "targets_read": False,
                "evaluations": {
                    k: {"cases": v["case_count"], "rows_tested_once": v["unique_rows_tested_once"]}
                    for k, v in audit["evaluations"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
