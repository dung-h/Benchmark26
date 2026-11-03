"""Add matched stages per observation, then calculate weighted summaries."""

import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.stages.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No stage records supplied")
    seen = set()
    for row in rows:
        identity = (row["id"], row["method"])
        if identity in seen:
            raise ValueError(f"Duplicate observation/method: {identity}")
        seen.add(identity)
        values = [float(row[key]) for key in ("parse_compile_ns", "method_stage_ns", "inference_ns")]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(f"Invalid duration for {identity}")
        # Shared compilation belongs once in each route's per-input total.
        row["composite_ms"] = sum(values) / 1e6
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        groups = {}
        for row in selected:
            groups.setdefault(row["group_id"], []).append(row["composite_ms"])
        group_means = [np.mean(values) for values in groups.values()]
        totals = [row["composite_ms"] for row in selected]
        print(
            method,
            "group_mean_ms",
            np.mean(group_means),
            "median_ms",
            np.median(totals),
            "p90_ms",
            np.quantile(totals, 0.9),
        )


if __name__ == "__main__":
    main()
