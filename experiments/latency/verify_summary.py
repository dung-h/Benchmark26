"""Reconstruct exported group means and paired group-bootstrap intervals."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(results):
    def read(name):
        with (results / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    manifest = json.loads((results / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = results / Path(name).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name

    sessions = read("neural_session_group_means.csv")
    by_group = collections.defaultdict(dict)
    for row in sessions:
        key = (row["source_id"], row["group_id"], row["method"])
        session = int(row["session"])
        assert session not in by_group[key], (key, session)
        by_group[key][session] = float(row["request_median_ms"])
    assert len(sessions) == 3 * 203 * 5
    assert all(set(values) == {1, 2, 3} for values in by_group.values())
    groups = read("neural_group_means.csv")
    assert len(groups) == 203 * 5
    identities = [(r["source_id"], r["group_id"], r["method"]) for r in groups]
    assert len(set(identities)) == len(identities)
    for row in groups:
        key = (row["source_id"], row["group_id"], row["method"])
        assert np.isclose(np.mean(list(by_group[key].values())), float(row["latency_ms"]), atol=1e-10, rtol=1e-12)
    for row in read("neural_summary.csv"):
        selected = [
            r
            for r in groups
            if r["method"] == row["method"] and (row["scope"] == "all" or r["source_id"] == row["scope"])
        ]
        assert len(selected) == int(row["groups"])
        assert np.isclose(
            np.mean([float(r["latency_ms"]) for r in selected]),
            float(row["group_balanced_mean_ms"]),
            atol=1e-10,
            rtol=1e-12,
        )

    rng = np.random.default_rng(42)
    pairs = read("neural_paired_contrasts.csv")
    contrasts = [
        ("MLP shots minus no shots", "M1", "M0"),
        ("Graph shots minus no shots", "G1", "G0"),
        ("Graph plus shots minus MLP plus shots", "G1", "M1"),
    ]
    for name, left_method, right_method in contrasts:

        def values(method):
            return {(r["source_id"], r["group_id"]): float(r["latency_ms"]) for r in groups if r["method"] == method}

        left, right = values(left_method), values(right_method)
        keys = sorted(left)
        assert keys == sorted(right)
        # Resample within each source, retaining that source's group count.
        sources = sorted({key[0] for key in keys})
        arrays = [np.array([left[k] - right[k] for k in keys if k[0] == source]) for source in sources]
        draws = []
        for _ in range(10000):
            sampled = [v[rng.integers(0, len(v), len(v))] for v in arrays]
            draws.append(np.mean(np.concatenate(sampled)))
        row = next(r for r in pairs if r["contrast"] == name)
        assert np.isclose(
            np.mean([left[k] - right[k] for k in keys]),
            float(row["delta_group_balanced_mean_ms"]),
            atol=1e-10,
            rtol=1e-12,
        )
        assert np.allclose(
            np.quantile(draws, [0.025, 0.975]),
            [float(row["ci95_lower_ms"]), float(row["ci95_upper_ms"])],
            atol=1e-10,
            rtol=1e-12,
        )
    print(
        "PASS: session/source group means and 30,000 paired bootstrap draws "
        "recomputed. Request quantiles and composite raw-stage timings "
        "require external measurement records."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parents[2] / "results/prediction_latency"
    )
    verify(parser.parse_args().results)
