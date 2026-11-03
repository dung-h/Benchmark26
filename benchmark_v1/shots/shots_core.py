"""Small, testable helpers for the post-freeze Ma–Li shots ablation."""
from __future__ import annotations

import copy
import math
from typing import Any


def fit_shot_transform(values: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Fit the plan-locked float64 mean/sample standard deviation (ddof=1)."""
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("shots_fit_values_must_be_finite_positive_vector")
    mean = float(x.mean(dtype=np.float64))
    std = float(x.std(ddof=1, dtype=np.float64)) if x.size > 1 else 0.0
    constant = not math.isfinite(std) or std <= 1e-6
    return {"mean": mean, "std_ddof1": std, "constant_support": constant,
            "fit_count": int(x.size), "dtype": "float64", "ddof": 1}


def transform_shots(value: float, transform: dict[str, Any]) -> float:
    import numpy as np
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("shots_value_must_be_finite_positive")
    std = float(transform["std_ddof1"])
    if transform["constant_support"]:
        return 0.0
    out = (x - float(transform["mean"])) / std
    if not math.isfinite(out):
        raise ValueError("nonfinite_standardized_shots")
    return float(np.float32(out))


def expand_global_projection(model, torch):
    """Append one zero-weight input to gf_linear1 without changing other state."""
    old = model.gf_linear1
    new = torch.nn.Linear(old.in_features + 1, old.out_features, bias=old.bias is not None,
                          device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_features].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.gf_linear1 = new
    return {"old_input_width": int(old.in_features), "new_input_width": int(new.in_features),
            "added_trainable_parameters": int(new.out_features),
            "old_weight_columns_copied": int(old.in_features), "new_column_initialized_zero": True}


def append_shots(example, standardized_shots: float):
    """Return a per-observation copy with z_shots appended to the global row."""
    import torch
    out = copy.copy(example)
    out.global_features = torch.cat(
        (example.global_features, torch.tensor([[standardized_shots]], dtype=torch.float32)), dim=1
    )
    return out


def initialized_outputs_equal(old_model, new_model, old_batch, new_batch, torch,
                              rtol: float = 1e-6, atol: float = 1e-6) -> float:
    old_model.eval()
    new_model.eval()
    with torch.no_grad():
        a = old_model(old_batch).reshape(-1)
        b = new_model(new_batch).reshape(-1)
        difference = float((a - b).abs().max().cpu()) if a.numel() else 0.0
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise RuntimeError(f"zero_shots_column_forward_parity_failed:max_abs={difference}")
    return difference


def verify_shots_gradient(model, batch, targets, torch) -> float:
    """Require a finite, non-zero gradient on the new input column."""
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch).reshape(-1)
    loss = torch.nn.functional.mse_loss(prediction, targets.reshape(-1))
    loss.backward()
    gradient = model.gf_linear1.weight.grad[:, -1]
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("shots_column_gradient_missing_or_nonfinite")
    magnitude = float(gradient.abs().max().detach().cpu())
    if magnitude <= 0:
        raise RuntimeError("shots_column_gradient_is_zero")
    return magnitude
