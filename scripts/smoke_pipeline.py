"""Exercise grouped splitting, fitting and reporting with synthetic observations."""
import csv
import json
from pathlib import Path
import numpy as np


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    if config.get("dataset") != "synthetic" or config.get("model") != "least_squares":
        raise ValueError("Smoke config requires dataset=synthetic and model=least_squares")
    groups = int(config["groups"])
    per_group = int(config["observations_per_group"])
    test_groups = int(config["test_groups"])
    if not 0 < test_groups < groups or per_group < 1:
        raise ValueError("Invalid group or split counts")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(int(config["seed"]))
    widths = rng.integers(2, 20, groups)
    depths = rng.integers(5, 100, groups)
    test = set(rng.choice(groups, test_groups, replace=False).tolist())
    rows = []
    for group in range(groups):
        for repeat in range(per_group):
            shots = int(rng.choice([128, 512, 1024]))
            target = .02 * widths[group] + .003 * depths[group] + .0002 * shots
            rows.append(dict(id=f"g{group}_r{repeat}", group_id=f"g{group}",
                             width=int(widths[group]), depth=int(depths[group]), shots=shots,
                             target_seconds=float(target), split="test" if group in test else "train"))
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] == "test"]
    assert {r['group_id'] for r in train}.isdisjoint({r['group_id'] for r in heldout})

    def features(selected):
        return np.asarray([[r[k] for k in ("width", "depth", "shots")] for r in selected], dtype=float)

    # Fit preprocessing on train only, then reuse it on held-out groups.
    x = features(train)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    def design(selected):
        normalized = (features(selected) - mean) / scale
        return np.column_stack([np.ones(len(selected)), normalized])
    coef = np.linalg.lstsq(design(train), [r['target_seconds'] for r in train], rcond=None)[0]
    prediction = design(heldout) @ coef
    predictions = [{**r, 'predicted_seconds': float(v),
                    'absolute_error_seconds': abs(float(v) - r['target_seconds'])}
                   for r, v in zip(heldout, prediction)]
    group_errors = {}
    for r in predictions:
        group_errors.setdefault(r['group_id'], []).append(r['absolute_error_seconds'])
    metrics = dict(dataset="synthetic", model="least_squares", train_rows=len(train),
                   test_rows=len(heldout), train_groups=groups-test_groups, test_groups=test_groups,
                   group_overlap=0, pooled_mae_seconds=float(np.mean([r['absolute_error_seconds'] for r in predictions])),
                   group_balanced_mae_seconds=float(np.mean([np.mean(v) for v in group_errors.values()])))
    output.mkdir(parents=True)
    for name, data in [("observations.csv", rows), ("predictions.csv", predictions)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for name, data in [("config.json", config), ("metrics.json", metrics),
                       ("model.json", dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coef.tolist()))]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    (output / 'report.md').write_text('# Synthetic smoke run\n\n'
        f"Train: {len(train)} observations; test: {len(heldout)} observations in {test_groups} held-out groups. "
        'The run exercises splitting, train-only preprocessing, fitting and evaluation. '
        'Its scores are software checks on a synthetic linear target, not QPU benchmark results.\n')
    print(json.dumps(metrics, indent=2))
    return metrics
