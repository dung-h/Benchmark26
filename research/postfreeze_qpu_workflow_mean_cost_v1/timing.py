#!/usr/bin/env python3
"""Timers for raw-seconds inference; no targets are accepted by these calls."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from common import SEEDS


def neural_predict(models, data, builder, *, graph: bool, device="cuda:0", prepare_fn=None):
    """Time optional raw-input transforms, batching/transfer, 3 forwards, median and host return."""
    with torch.inference_mode():
        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            raise RuntimeError("inference_mode_not_active")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        if prepare_fn is not None:
            data = prepare_fn(data)
        batch = builder(data, graph=graph, device=device)
        seeds = {}
        for seed in SEEDS:
            output = models[seed](batch).reshape(-1)
            seeds[str(seed)] = float(output.detach().cpu()[0].item())
        prediction = float(np.median([seeds[str(s)] for s in SEEDS]))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter_ns() - start
    if not math.isfinite(prediction) or not all(math.isfinite(v) for v in seeds.values()):
        raise RuntimeError("nonfinite_neural_prediction")
    return prediction, seeds, elapsed


def predict_r0(estimator, raw_native5):
    start = time.perf_counter_ns()
    value = float(estimator.predict(np.asarray(raw_native5, dtype=np.float64).reshape(1, 5))[0])
    elapsed = time.perf_counter_ns() - start
    if not math.isfinite(value):
        raise RuntimeError("nonfinite_R0_prediction")
    return value, elapsed


def predict_c0(value):
    start = time.perf_counter_ns()
    result = float(value)
    elapsed = time.perf_counter_ns() - start
    if not math.isfinite(result):
        raise RuntimeError("nonfinite_C0_prediction")
    return result, elapsed


def check_prediction(actual, expected, method, per_seed=None):
    if method in {"M0", "M1", "G0", "G1"}:
        if set(per_seed or {}) != set(map(str, SEEDS)):
            raise RuntimeError("neural_seed_coverage_mismatch")
        for seed in SEEDS:
            ref = float(expected["seeds"][str(seed)])
            val = float(per_seed[str(seed)])
            if abs(val - ref) > 1e-5 + 1e-6 * abs(ref):
                raise RuntimeError(f"neural_seed_replay_mismatch:{method}:{seed}:{val-ref}")
        median = float(np.median([per_seed[str(s)] for s in SEEDS]))
        if abs(float(actual) - median) > 1e-12:
            raise RuntimeError("wrong_three_seed_median")
    else:
        ref = float(expected["prediction"])
        tolerance = 1e-9 if method == "R0" else 0.0
        if abs(float(actual) - ref) > tolerance:
            raise RuntimeError(f"saved_prediction_replay_mismatch:{method}:{actual-ref}")
