"""Verify the runnable smoke path without external data, sklearn or a GPU."""

import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_benchmark_pipeline.py"


def test_smoke_consumes_config_and_produces_grouped_predictions(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(dataset="synthetic", model="least_squares", seed=7, groups=12, observations_per_group=3, test_groups=3)
        )
    )
    output = tmp_path / "result"
    result = subprocess.run(
        [sys.executable, str(CLI), "--stage", "smoke", "--config", str(config), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metric = json.loads((output / "metrics.json").read_text())
    assert metric["train_rows"] == 27 and metric["test_rows"] == 9
    assert metric["test_groups"] == 3 and metric["group_overlap"] == 0
    assert metric["pooled_mae_seconds"] < 1e-10
    with (output / "observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    train = {r["group_id"] for r in rows if r["split"] == "train"}
    test = {r["group_id"] for r in rows if r["split"] == "test"}
    assert train.isdisjoint(test)
    assert (output / "model.json").is_file() and (output / "report.md").is_file()


def test_status_rejects_ignored_config(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CLI), "--status", "--config", "anything.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "consumed only" in result.stderr


def test_default_cli_runs_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(CLI)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["mode"] == "input_inventory"
