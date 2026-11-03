"""Tests for benchmark metric aggregation functions."""

import numpy as np
from collections import defaultdict


def compute_benchmark_metrics(records):
    """Reference implementation of metric computation matching analysis_core."""
    y = np.array([float(r["target_seconds"]) for r in records])
    p = np.array([float(r["predicted_seconds"]) for r in records])
    ae = np.abs(p - y)

    groups = defaultdict(list)
    sources = defaultdict(list)
    for r, e in zip(records, ae):
        groups[r["group_id"]].append(e)
        sources[r["source_id"]].append(e)

    pooled_mae = float(np.mean(ae))
    group_balanced_mae = float(np.mean([np.mean(v) for v in groups.values()]))
    source_macro_mae = float(np.mean([np.mean(v) for v in sources.values()]))
    medae = float(np.median(ae))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((p - y) ** 2)
    r2_raw = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None

    return {
        "pooled_mae": pooled_mae,
        "group_balanced_mae": group_balanced_mae,
        "source_macro_mae": source_macro_mae,
        "medae": medae,
        "rmse": rmse,
        "r2_raw": r2_raw,
        "negative_predictions": int(np.sum(p < 0)),
    }


def test_metrics_perfect_prediction():
    records = [
        {"target_seconds": 1.0, "predicted_seconds": 1.0, "group_id": "g1", "source_id": "s1"},
        {"target_seconds": 2.0, "predicted_seconds": 2.0, "group_id": "g1", "source_id": "s1"},
        {"target_seconds": 3.0, "predicted_seconds": 3.0, "group_id": "g2", "source_id": "s2"},
    ]
    res = compute_benchmark_metrics(records)
    assert res["pooled_mae"] == 0.0
    assert res["group_balanced_mae"] == 0.0
    assert res["source_macro_mae"] == 0.0
    assert res["r2_raw"] == 1.0
    assert res["negative_predictions"] == 0


def test_metrics_group_balanced_weighting():
    # Group 1 has 9 observations with error 1.0
    # Group 2 has 1 observation with error 11.0
    records = []
    for _ in range(9):
        records.append({"target_seconds": 10.0, "predicted_seconds": 11.0, "group_id": "g1", "source_id": "s1"})
    records.append({"target_seconds": 10.0, "predicted_seconds": 21.0, "group_id": "g2", "source_id": "s1"})

    res = compute_benchmark_metrics(records)
    # Pooled MAE is (9*1 + 1*11)/10 = 20/10 = 2.0
    assert np.isclose(res["pooled_mae"], 2.0)
    # Group-balanced MAE is (1.0 + 11.0)/2 = 6.0
    assert np.isclose(res["group_balanced_mae"], 6.0)
    assert res["group_balanced_mae"] != res["pooled_mae"]


def test_negative_predictions_detection():
    records = [
        {"target_seconds": 0.5, "predicted_seconds": -0.2, "group_id": "g1", "source_id": "s1"},
        {"target_seconds": 1.0, "predicted_seconds": 0.9, "group_id": "g1", "source_id": "s1"},
    ]
    res = compute_benchmark_metrics(records)
    assert res["negative_predictions"] == 1
