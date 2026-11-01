#!/usr/bin/env python3
"""Plot saved accuracy against measured latency for E1/E2, separately."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/postfreeze_qpu_workflow_mean_cost_v1/v2/summary"
LABELS = {
    "M0": "MLP", "M1": "MLP + shots", "G0": "Graph+MLP",
    "G1": "Graph+MLP + shots", "R0": "Selected classical", "C0": "Train median",
}


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pareto(points):
    keep = []
    for method, x, y in points:
        if not any((x2 <= x and y2 <= y and (x2 < x or y2 < y))
                   for _, x2, y2 in points):
            keep.append((method, x, y))
    return sorted(keep, key=lambda p: p[1])


def main():
    lat = read(SUMMARY / "latency_summary.csv")
    acc = read(SUMMARY / "saved_accuracy_metrics.csv")
    points = read(SUMMARY / "accuracy_cost_points.csv")
    acc_map = {(r["evaluation"], r["method"]): float(r["group_balanced_mae_seconds"])
               for r in acc}
    for ev in ("E1", "E2"):
        for clock, suffix in (("B", "primary"), ("A", "ready_input_diagnostic")):
            rows = [r for r in lat if r["evaluation"] == ev and r["scope"] == "ALL"
                    and r["clock"] == clock]
            pts = [(r["method"], float(r["group_balanced_mean_latency_ms"]),
                    acc_map[(ev, r["method"])]) for r in rows]
            fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
            for method, x, y in pts:
                ax.scatter(x, y, s=55)
                ax.annotate(LABELS[method], (x, y), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
            front = pareto(pts)
            if len(front) > 1:
                ax.plot([p[1] for p in front], [p[2] for p in front],
                        color="0.45", linestyle="--", linewidth=1)
            ax.set_xlabel("Group-balanced mean prediction latency (ms per observation)")
            ax.set_ylabel("Saved group-balanced MAE (s)")
            ax.set_title(f"{ev}: accuracy–cost ({'raw features to prediction' if clock == 'B' else 'ready input'})")
            ax.grid(alpha=.25)
            out = SUMMARY / f"accuracy_cost_{ev}_{suffix}.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)

    for backend in sorted({r["scope_value"] for r in points if r["evaluation"] == "E3"}):
        backend_rows = [r for r in points if r["evaluation"] == "E3" and r["scope_value"] == backend]
        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for r in backend_rows:
            x = float(r["group_balanced_mean_latency_ms"])
            y = float(r["saved_group_balanced_mae_seconds"])
            ax.scatter(x, y, s=55)
            ax.annotate(LABELS[r["method"]], (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel("Group-balanced mean prediction latency (ms per observation)")
        ax.set_ylabel("Saved group-balanced MAE (s)")
        ax.set_title(f"E3 backend-context holdout: {backend}")
        ax.grid(alpha=.25)
        safe = backend.replace("/", "_")
        fig.savefig(SUMMARY / f"accuracy_cost_E3_{safe}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
