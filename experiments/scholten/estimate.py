#!/usr/bin/env python3
"""Score a Scholten-style throughput estimate on the 3,443-run dataset.

Scholten et al. (2023, arXiv:2307.04980, Eq. 6) estimate the runtime of a job of
M circuits as M * shots * D_eff / CLOPS. Here every job holds one circuit
(M = 1), so

    prediction_seconds = shots * unitary_depth / CLOPS

Two choices make this an adaptation, not the original estimator:

- ``unitary_depth`` is the layer depth of the saved compiled circuit, not the
  paper's effective depth, which divides by the depth of a quantum-volume
  circuit. Deep circuits are therefore charged more than in the paper.
- CLOPS comes from ``clops.csv``: the ``clops_v`` or ``clops_h`` field of the
  qiskit-ibm-runtime 0.49.0 fake-backend configuration. ``tests/test_scholten.py``
  compares the table with the installed package when it is present. belem, lima
  and quito have neither field, so their 1,154 runs get no estimate.

Nothing is fitted, so the held-out-group (E1) and random-run (E2) settings give
the same values. The script reports error and correlation on the covered runs,
the same scores of five saved estimators on exactly those runs, and the median
ratio of prediction to label by source and shot range. It reads the study files
(``B26_ARTIFACTS``) and refuses to write into an existing folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark26.integrity import sha256_file
from benchmark26.paths import ARTIFACTS
from benchmark26.tables import read_csv as read

HERE = Path(__file__).resolve().parent
ANALYTICAL = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
STUDY = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
INPUTS = {
    "compiled_features": ANALYTICAL / "raw_features.csv",
    "run_to_circuit": ANALYTICAL / "lineage.csv",
    "labels": STUDY / "cohort/targets.csv",
    "neural_predictions": STUDY / "analysis_v1/reduced_predictions.csv",
    "analytical_predictions": ANALYTICAL / "predictions.csv",
}
# Saved estimators scored on the same runs, with their names in the study files.
COMPARED = {"C0": "median_baseline", "R0": "qonductor", "M1": "mlp_shots", "G1": "graph_mlp_shots"}
QISKIT_X_SHOTS = "QISKIT_SINGLE_SHOT_SCALED"
SHOT_RANGES = ((1, 99), (100, 500), (501, 2000), (2001, 8000), (8001, None))
SOURCE_NAMES = {"mali_real_qpu": "ma_li", "qonductor_single_circuit_ibm": "qonductor", "qpack_mcp": "qpack"}


def load_clops(path: Path = HERE / "clops.csv") -> dict[str, tuple[str, float] | None]:
    """Backend name -> (CLOPS type, value), or None when the backend reports no CLOPS."""
    return {r["backend"]: (r["clops_type"], float(r["clops"])) if r["clops"] else None for r in read(path)}


def backend_name(snapshot_id: str) -> str:
    """'...::ibm_osaka...' -> 'osaka'."""
    return snapshot_id.split("::")[1].replace("ibmq_", "").replace("ibm_", "")


def estimate(shots: float, depth: float, clops: float) -> float:
    """Scholten-style job runtime in seconds for one circuit."""
    return shots * depth / clops


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def scores(rows: list[dict], key: str) -> dict:
    """Group-balanced MAE, pooled MAE, R^2 in seconds, Pearson and Spearman correlation with the label."""
    y = [r["label"] for r in rows]
    p = [r[key] for r in rows]
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(abs(r[key] - r["label"]))
    mean_label = st.mean(y)
    return {
        "runs": len(rows),
        "groups": len(by_group),
        "gb_mae": st.mean(st.mean(v) for v in by_group.values()),
        "pooled_mae": st.mean(abs(a - b) for a, b in zip(p, y)),
        "r2": 1 - sum((a - b) ** 2 for a, b in zip(p, y)) / sum((b - mean_label) ** 2 for b in y),
        "pearson": _pearson(p, y),
        "spearman": _pearson(_ranks(p), _ranks(y)),
        "mean_prediction": st.mean(p),
        "mean_label": mean_label,
    }


def shot_range(shots: float) -> str:
    for low, high in SHOT_RANGES:
        if shots >= low and (high is None or shots <= high):
            return f"{low}-{high}" if high else f">{low - 1}"
    raise ValueError(shots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="new folder for the results")
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)

    clops = load_clops()
    depth = {(r["graph_file"], r["snapshot_id"]): float(r["unitary_depth"]) for r in read(INPUTS["compiled_features"])}
    label = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in read(INPUTS["labels"])}
    rows, uncovered = [], defaultdict(int)
    for r in read(INPUTS["run_to_circuit"]):
        backend = backend_name(r["snapshot_id"])
        if clops[backend] is None:
            uncovered[backend] += 1
            continue
        kind, value = clops[backend]
        d = depth[(r["graph_file"], r["snapshot_id"])]
        rows.append({
            "run_id": r["canonical_observation_id"], "group": r["group_id"],
            "source": SOURCE_NAMES[r["source_id"]], "backend": backend, "clops_type": kind,
            "shots": float(r["shots"]), "label": label[r["canonical_observation_id"]],
            "scholten": estimate(float(r["shots"]), d, value),
        })
    assert len(rows) + sum(uncovered.values()) == 3443, "the dataset has 3,443 runs"
    covered = {r["run_id"] for r in rows}

    # Saved predictions of the other estimators, restricted to the covered runs.
    other = defaultdict(dict)
    for r in read(INPUTS["neural_predictions"]):
        if r["treatment"] in COMPARED and r["canonical_observation_id"] in covered:
            run_id = r["canonical_observation_id"]
            other[(r["evaluation"], COMPARED[r["treatment"]])][run_id] = float(r["predicted_seconds"])
    for r in read(INPUTS["analytical_predictions"]):
        if r["method_id"] == QISKIT_X_SHOTS and r["canonical_observation_id"] in covered:
            run_id = r["canonical_observation_id"]
            other[(r["evaluation"], "qiskit_duration_x_shots")][run_id] = float(r["prediction_seconds"])

    results = []

    def add(method: str, setting: str, scope: str, value: str, subset: list[dict], key: str) -> None:
        if subset:
            names = {"method": method, "setting": setting, "scope": scope, "scope_value": value}
            results.append({**names, **scores(subset, key)})

    add("scholten_style", "E1=E2", "all", "covered", rows, "scholten")
    for kind in ("clops_v", "clops_h"):
        add("scholten_style", "E1=E2", "clops_type", kind, [r for r in rows if r["clops_type"] == kind], "scholten")
    for source in sorted({r["source"] for r in rows}):
        add("scholten_style", "E1=E2", "source", source, [r for r in rows if r["source"] == source], "scholten")
    for (setting, method), predictions in sorted(other.items()):
        assert len(predictions) == len(rows), (setting, method, len(predictions))
        add(method, setting, "all", "covered", [dict(r, other=predictions[r["run_id"]]) for r in rows], "other")

    ratios = defaultdict(list)
    for r in rows:
        ratios[(r["source"], shot_range(r["shots"]))].append(r["scholten"] / r["label"])
    ratio_rows = [
        {"source": source, "shot_range": span, "runs": len(v), "median_ratio": st.median(v),
         "q25_ratio": st.quantiles(v, n=4)[0] if len(v) > 1 else v[0],
         "q75_ratio": st.quantiles(v, n=4)[2] if len(v) > 1 else v[0]}
        for (source, span), v in sorted(ratios.items())
    ]

    def write(name: str, table: list[dict]) -> None:
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(table)

    write("scores.csv", results)
    write("ratio_by_shots.csv", ratio_rows)
    receipt = {
        "covered_runs": len(rows),
        "uncovered_runs_by_backend": dict(sorted(uncovered.items())),
        "clops_table_sha256": sha256_file(HERE / "clops.csv"),
        "inputs_sha256": {name: sha256_file(path) for name, path in INPUTS.items()},
    }
    (out / "run_summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("covered_runs", "uncovered_runs_by_backend")}))


if __name__ == "__main__":
    main()
