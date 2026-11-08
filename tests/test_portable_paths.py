"""Keep personal paths out of executable source and check external-root relocation."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_personal_interpreter_or_home_paths():
    pattern = re.compile(r"/home/[^/\s]+/|/mnt/windows/|\.venv-[^\s/]+/bin/python")
    for directory in ("experiments", "benchmark26", "scripts", "configs"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in {".py", ".json", ".yaml"}:
                assert not pattern.search(path.read_text()), str(path)


def test_primary_study_uses_configured_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("B26_ARTIFACTS", str(tmp_path))
    import benchmark26.paths

    importlib.reload(benchmark26.paths)
    path = ROOT / "experiments/qpu/core.py"
    spec = importlib.util.spec_from_file_location("relocated_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUT == tmp_path / "postfreeze_qpu_workflow_mean_expansion_v1"


def test_status_checks_configured_external_root(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    artifacts = tmp_path / "external"
    run_table_path = artifacts / "postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv"
    run_table_path.parent.mkdir(parents=True)
    run_table_path.write_text("header\n")
    monkeypatch.setenv("B26_ARTIFACTS", str(artifacts))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_benchmark_pipeline.py"), "--status"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    inventory = json.loads(result.stdout)["inputs"]
    assert inventory["artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv"] is True


def test_experiment_source_paths_resolve_after_reorganization():
    import ast

    for directory in ("experiments", "benchmark26", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (
                    isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name)
                    and node.left.id in {"ROOT", "R"}
                    and isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, str)
                ):
                    continue
                value = node.right.value
                if value.startswith(("experiments/", "benchmark26/", "protocols/")):
                    assert (ROOT / value).exists(), (path, value)
