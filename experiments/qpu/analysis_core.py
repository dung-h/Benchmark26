"""Prespecified offline metrics; no estimator fitting or model selection."""

from __future__ import annotations
import numpy as np

CONTRASTS = (("G0", "M0"), ("G1", "M1"), ("M1", "M0"), ("G1", "G0"))


def aligned(records, expected_ids):
    indexed = {r["canonical_observation_id"]: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(expected_ids):
        raise ValueError("Missing, duplicate, or foreign prediction identity")
    return [indexed[c] for c in expected_ids]


def metrics(y, p, groups, sources):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    groups, sources = np.asarray(groups), np.asarray(sources)
    if not len(y) or y.shape != p.shape or len(groups) != len(y) or len(sources) != len(y):
        raise ValueError("Metric alignment/shape error")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not (y > 0).all():
        raise ValueError("Nonfinite values or invalid target")
    signed = p - y
    ae = np.abs(signed)
    sst = np.square(y - y.mean()).sum()
    raw_r2 = float(1 - np.square(signed).sum() / sst) if sst > 0 and len(y) > 1 else None
    log_r2 = None
    log_status = "UNAVAILABLE_NONPOSITIVE_PREDICTION"
    if (p > 0).all():
        ly, lp = np.log1p(y), np.log1p(p)
        denominator = np.square(ly - ly.mean()).sum()
        log_r2 = float(1 - np.square(ly - lp).sum() / denominator) if denominator > 0 and len(y) > 1 else None
        log_status = "DEFINED" if log_r2 is not None else "UNDEFINED_CONSTANT_OR_SINGLE_TARGET"
    return dict(
        rows=len(y),
        groups=len(set(groups)),
        sources=len(set(sources)),
        group_balanced_mae_seconds=float(np.mean([ae[groups == g].mean() for g in sorted(set(groups))])),
        pooled_mae_seconds=float(ae.mean()),
        source_macro_mae_seconds=float(np.mean([ae[sources == s].mean() for s in sorted(set(sources))])),
        medae_seconds=float(np.median(ae)),
        rmse_seconds=float(np.sqrt(np.square(signed).mean())),
        raw_seconds_r2=raw_r2,
        log1p_seconds_r2=log_r2,
        log_r2_status=log_status,
        bias_seconds=float(signed.mean()),
        p90_absolute_error_seconds=float(np.quantile(ae, 0.90)),
        p99_absolute_error_seconds=float(np.quantile(ae, 0.99)),
        max_absolute_error_seconds=float(ae.max()),
        negative_predictions=int((p < 0).sum()),
        zero_predictions=int((p == 0).sum()),
        overprediction_fraction=float((signed > 0).mean()),
        underprediction_fraction=float((signed < 0).mean()),
        target_mean_seconds=float(y.mean()),
        target_std_seconds=float(y.std()),
        prediction_mean_seconds=float(p.mean()),
    )


def group_errors(y, predictions, groups, sources):
    groups, sources = np.asarray(groups), np.asarray(sources)
    order = sorted(set(groups))
    source_order = []
    values = np.empty((len(order), len(predictions)), dtype=np.float64)
    for i, g in enumerate(order):
        ix = groups == g
        gs = set(sources[ix])
        if len(gs) != 1:
            raise ValueError("Source-stratified bootstrap requires source-pure groups")
        source_order.append(next(iter(gs)))
        for j, p in enumerate(predictions.values()):
            values[i, j] = np.abs(np.asarray(p)[ix] - np.asarray(y)[ix]).mean()
    return order, source_order, values


def bootstrap_draws(sources, stratified, replicates=10000, seed=42):
    """Return whole-group indices. Within-source counts are fixed when stratified."""
    sources = np.asarray(sources)
    if not len(sources):
        raise ValueError("No groups")
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(sources == s) for s in sorted(set(sources))] if stratified else [np.arange(len(sources))]
    return np.concatenate([rng.choice(ix, size=(replicates, len(ix)), replace=True) for ix in strata], axis=1)


def paired_intervals(group_values, methods, draws, adjusted_level):
    values = np.asarray(group_values)
    result = []
    for a, b in CONTRASTS:
        delta = values[:, methods.index(a)] - values[:, methods.index(b)]
        boot = delta[draws].mean(axis=1)
        q = (1 - adjusted_level) / 2
        lo, hi = np.quantile(boot, [0.025, 0.975])
        alo, ahi = np.quantile(boot, [q, 1 - q])
        result.append(
            dict(
                contrast=f"{a}-{b}",
                treatment=a,
                control=b,
                delta_group_balanced_mae_seconds=float(delta.mean()),
                ci95_lower_seconds=float(lo),
                ci95_upper_seconds=float(hi),
                adjusted_level=adjusted_level,
                adjusted_ci_lower_seconds=float(alo),
                adjusted_ci_upper_seconds=float(ahi),
                empirical_bootstrap_fraction_delta_below_zero=float((boot < 0).mean()),
                valid_replicates=len(boot),
                resampling_unit="structural_group",
                improvement_direction="negative",
                fraction_is_p_value=False,
            )
        )
    return result
