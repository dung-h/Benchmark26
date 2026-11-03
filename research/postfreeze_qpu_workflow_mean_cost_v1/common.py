#!/usr/bin/env python3
"""Pinned readers and identity helpers for expanded-cohort inference timing."""
from __future__ import annotations

import os as _b26_os
from pathlib import Path

import csv
import hashlib
import json
import math
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_expansion_v1'
ART = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1'
PLAN = ROOT / "docs/final_completion/expanded_qpu_inference_cost_v1"
SEEDS = (42, 1234, 31415)
METHODS = ("M0", "M1", "G0", "G1", "R0", "C0")
FEATURES = ("swap", "depth", "num_qubits", "shots", "circuit_count")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_panel() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    panel = rows(SRC / "cohort/panel.csv")
    ledger_rows = rows(SRC / "cohort/feature_ledger.csv")
    native_rows = rows(SRC / "cohort/native_baseline_features.csv")
    panel_by_id = {r["canonical_observation_id"]: r for r in panel}
    ledger = {r["canonical_observation_id"]: r for r in ledger_rows}
    native = {r["canonical_observation_id"]: r for r in native_rows}
    if len(panel_by_id) != 3443 or len(ledger) != 3443 or len(native) != 3443:
        raise RuntimeError("expanded_cohort_row_or_unique_id_count_changed")
    if set(panel_by_id) != set(ledger) or set(panel_by_id) != set(native):
        raise RuntimeError("expanded_cohort_feature_identity_mismatch")
    return panel, panel_by_id, {cid: {**ledger[cid], **{f"global_{i:02d}": ledger[cid][f"global_{i:02d}"] for i in range(51)}} for cid in panel_by_id}


def load_native() -> dict[str, np.ndarray]:
    out = {}
    for row in rows(SRC / "cohort/native_baseline_features.csv"):
        cid = row["canonical_observation_id"]
        x = np.asarray([float(row[k]) for k in FEATURES], dtype=np.float64)
        if x.shape != (5,) or not np.isfinite(x).all():
            raise RuntimeError(f"invalid_native_input:{cid}")
        out[cid] = x
    if len(out) != 3443:
        raise RuntimeError("native_input_row_count_changed")
    return out


def load_split(evaluation: str):
    if evaluation in ("E1", "E2"):
        split = rows(SRC / f"splits/{evaluation}_outer.csv")
        assignment = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in split}
        if len(assignment) != 3443:
            raise RuntimeError(f"{evaluation}_outer_split_count_changed")
        return {fold: sorted(cid for cid, value in assignment.items() if value == fold) for fold in range(5)}
    split = rows(SRC / "splits/E3_outer.csv")
    cases = sorted({r["heldout_backend"] for r in split})
    by_backend = {b: sorted(r["canonical_observation_id"] for r in split if r["heldout_backend"] == b and r["partition"] == "test") for b in cases}
    if len(cases) != 10 or sum(map(len, by_backend.values())) != 3443:
        raise RuntimeError("E3_outer_split_count_changed")
    return {i: by_backend[b] for i, b in enumerate(cases)}


def verify_plan_pins() -> dict:
    evidence = json.loads((PLAN / "EVIDENCE_INVENTORY.json").read_text())
    for rel, expected in evidence.get("evidence_hashes", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"pinned_source_artifact_changed:{rel}")
    return evidence
