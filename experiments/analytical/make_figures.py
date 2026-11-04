#!/usr/bin/env python3
"""Render concise diagnostic figures from saved unified predictions/metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import csv
from pathlib import Path

import matplotlib
import numpy as np

from benchmark26.paths import ARTIFACTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
N = ARTIFACTS / "postfreeze_unified_analytical_completion_v1"
D = N / "figures"
D.mkdir(exist_ok=True)


def read(p):
    """Read a CSV file into a list of dicts."""
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


metrics = read(N / "metrics.csv")
pred = read(N / "predictions.csv")
base = read(ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1/analysis_v1/reduced_predictions.csv")


def metric(ev, case, m):
    """Return the metric row of (evaluation, case, method)."""
    return next(r for r in metrics if r["evaluation"] == ev and r["case"] == case and r["method_id"] == m)


labels = [
    ("Qonductor (R0)", "R0"),
    ("Train median (C0)", "C0"),
    ("Target path raw", "QISKIT_SINGLE"),
    ("Target path × shots", "QISKIT_SINGLE_SHOT_SCALED"),
    ("Target path × shots + affine", "QISKIT_SINGLE_SHOT_SCALED_AFFINE"),
    ("QCRE unitary × shots + affine", "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE"),
    ("NoTaDS code × shots + affine", "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE"),
]
fig, ax = plt.subplots(figsize=(13, 5.6))
x = np.arange(len(labels))
w = 0.36
for off, ev, color in [(-w / 2, "E1", "#2878B5"), (w / 2, "E2", "#E07A32")]:
    vals = [float(metric(ev, "ALL", m)["group_balanced_mae_seconds"]) for _, m in labels]
    ax.bar(
        x + off, vals, w, label=ev + (" group-disjoint" if ev == "E1" else " source-stratified random-row"), color=color
    )
ax.set_xticks(x, [l for l, _ in labels], rotation=18, ha="right")
ax.set_ylabel("Group-balanced MAE (seconds; lower is better)")
ax.set_title("Analytical routes: raw timing, shot scaling and train-only calibration")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(D / "unified_accuracy_main.pdf")
fig.savefig(D / "unified_accuracy_main.png", dpi=180)
plt.close(fig)

# Per-backend E3 diagnostic. Small held-out contexts are shown with exact test counts.
backends = sorted({r["case"] for r in metrics if r["evaluation"] == "E3"})
mnames = [
    "R0",
    "QISKIT_SINGLE_SHOT_SCALED_AFFINE",
    "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE",
    "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE",
    "NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE",
]
mlabel = [
    "Qonductor (R0)",
    "Target path × shots + affine",
    "QCRE unitary × shots + affine",
    "NoTaDS code score × shots + affine",
    "NoTaDS paper score × shots + affine",
]
arr = np.array([[float(metric("E3", b, m)["group_balanced_mae_seconds"]) for m in mnames] for b in backends])
counts = [int(metric("E3", b, "R0")["rows"]) for b in backends]
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=0)
ax.set_xticks(np.arange(len(mnames)), mlabel, rotation=28, ha="right")
ax.set_yticks(np.arange(len(backends)), [f"{b} (n={n})" for b, n in zip(backends, counts)])
ax.set_title("E3 backend-context holdouts: group-balanced MAE by held-out backend")
cb = fig.colorbar(im, ax=ax)
cb.set_label("seconds (lower is better)")
for i in range(len(backends)):
    for j in range(len(mnames)):
        v = arr[i, j]
        ax.text(
            j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > np.median(arr) else "black", fontsize=8
        )
fig.tight_layout()
fig.savefig(D / "e3_backend_heatmap.pdf")
fig.savefig(D / "e3_backend_heatmap.png", dpi=180)
plt.close(fig)

# Same held-out rows, a visual check of scale and systematic bias under E2.
eval_rows = [
    r
    for r in pred
    if r["evaluation"] == "E2"
    and r["method_id"]
    in (
        "QISKIT_SINGLE_SHOT_SCALED_AFFINE",
        "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE",
        "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE",
    )
]
lookup = {r["canonical_observation_id"]: r for r in base if r["evaluation"] == "E2" and r["treatment"] == "R0"}
methods = [
    ("Qonductor (R0)", None),
    ("Target path × shots + affine", "QISKIT_SINGLE_SHOT_SCALED_AFFINE"),
    ("QCRE unitary × shots + affine", "QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE"),
    ("NoTaDS code score × shots + affine", "NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE"),
]
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
for ax, (name, m) in zip(axs.flat, methods):
    if m is None:
        rows = [r for r in lookup.values()]
        pairs = [(float(r["target_seconds"]), float(r["predicted_seconds"])) for r in rows]
    else:
        rows = [r for r in eval_rows if r["method_id"] == m]
        pairs = [(float(r["target_seconds"]), float(r["prediction_seconds"])) for r in rows]
    y = np.array([a for a, b in pairs])
    p = np.array([b for a, b in pairs])
    ax.scatter(y, p, s=8, alpha=0.22, color="#2878B5", edgecolors="none")
    ax.plot([0, 30], [0, 30], color="#C23B22", lw=1, ls="--")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Observed target (s)")
    ax.set_ylabel("Prediction (s)")
    ax.set_title(name)
    ax.grid(alpha=0.2)
    ax.text(0.03, 0.96, f"n={len(y)}", transform=ax.transAxes, va="top")
fig.suptitle("E2: held-out observations, predicted versus observed runtime", y=1.01)
fig.tight_layout()
fig.savefig(D / "e2_prediction_scatter.pdf", bbox_inches="tight")
fig.savefig(D / "e2_prediction_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("figures_written", len(list(D.glob("*.pdf"))), "pdfs", len(list(D.glob("*.png"))), "pngs")
