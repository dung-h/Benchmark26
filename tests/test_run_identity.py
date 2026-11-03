"""New-run preparation keeps the historical lock intact and pins the current code.

Tests that mutate code run on a disposable copy of ``benchmark26/`` and
``experiments/qpu/`` so the working tree is never changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QPU = ROOT / "experiments/qpu"
STUDY = "postfreeze_qpu_workflow_mean_expansion_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    study = artifacts / STUDY
    (study / "cohort").mkdir(parents=True)
    (study / "PROTOCOL.md").write_text("protocol\n")
    (study / "cohort/panel.csv").write_text("canonical_observation_id\nx\n")
    lock = {
        "status": "LOCKED_PRE_OUTCOME",
        "protocol_markdown_sha256": sha(study / "PROTOCOL.md"),
        "input_hashes": {f"artifacts/{STUDY}/cohort/panel.csv": sha(study / "cohort/panel.csv")},
        "unique_graph_hashes": {},
        # The original code differed from this repository's adapted file.
        "code_hashes": {f"research/{STUDY}/core.py": "0" * 64},
        "neural": {"effective_batch": 32},
    }
    (study / "execution_lock_v2.json").write_text(json.dumps(lock))
    return artifacts


def run(code: str, env: dict, qpu: Path = QPU) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=qpu, env={**os.environ, **env}, capture_output=True, text=True, check=False
    )


def copy_code(tmp_path: Path) -> Path:
    """Disposable copy of the code a new run executes; returns its experiments/qpu."""
    repo = tmp_path / "repo"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "benchmark26", repo / "benchmark26", ignore=ignore)
    shutil.copytree(QPU, repo / "experiments/qpu", ignore=ignore)
    return repo / "experiments/qpu"


def prepare(env: dict, qpu: Path = QPU) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(qpu / "prepare_run.py")],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


IDENTITY = "import core; lock, digest = core.execution_identity(); print(digest)"


def test_adapted_code_cannot_extend_original_run(tmp_path):
    artifacts = make_bundle(tmp_path)
    env = {"B26_ARTIFACTS": str(artifacts)}
    result = run(IDENTITY, {**env, "B26_RUN_DIR": ""})
    assert result.returncode != 0
    assert "historical_code_lock_mismatch" in result.stderr
    # Validation of saved outputs checks data identity only.
    ok = run("import core; print(core.execution_identity(require_code=False)[1])", {**env, "B26_RUN_DIR": ""})
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.strip() == sha(artifacts / STUDY / "execution_lock_v2.json")


def test_prepared_new_run_pins_current_code(tmp_path):
    artifacts = make_bundle(tmp_path)
    qpu = copy_code(tmp_path)
    repo = qpu.parents[1]
    lock_before = (artifacts / STUDY / "execution_lock_v2.json").read_bytes()
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(tmp_path / "run")}
    missing = run(IDENTITY, env, qpu)
    assert missing.returncode != 0 and "new_run_not_prepared" in missing.stderr

    prepared = prepare(env, qpu)
    assert prepared.returncode == 0, prepared.stderr
    manifest_path = tmp_path / "run/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "NEW_RUN_ADAPTED_CODE"
    assert manifest["code_hashes"] == {f"research/{STUDY}/core.py": sha(qpu / "core.py")}
    assert manifest["code_differing_from_historical_lock"] == [f"research/{STUDY}/core.py"]
    # The current-code inventory also pins files the historical lock never named.
    deps = manifest["code_dependency_hashes"]
    for relative in ("benchmark26/paths.py", "experiments/qpu/train_neural.py", "experiments/qpu/run_classical.py"):
        assert deps[relative] == sha(repo / relative), relative
    assert "benchmark26/paths.py" not in manifest["code_hashes"]

    accepted = run(IDENTITY, env, qpu)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == sha(manifest_path)
    assert (artifacts / STUDY / "execution_lock_v2.json").read_bytes() == lock_before

    # A changed input invalidates both the historical and the new-run identity.
    (artifacts / STUDY / "cohort/panel.csv").write_text("changed\n")
    changed = run(IDENTITY, env, qpu)
    assert changed.returncode != 0 and "locked_inputs_differ" in changed.stderr


def test_prepared_run_rejects_change_to_dependency_not_in_historical_lock(tmp_path):
    artifacts = make_bundle(tmp_path)
    qpu = copy_code(tmp_path)
    paths = qpu.parents[1] / "benchmark26/paths.py"
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(tmp_path / "run")}
    assert prepare(env, qpu).returncode == 0
    original = paths.read_bytes()

    paths.write_bytes(original + b"\n# changed after the run was prepared\n")
    rejected = run(IDENTITY, env, qpu)
    assert rejected.returncode != 0
    assert "code_dependency_changed_since_run_manifest:benchmark26/paths.py" in rejected.stderr
    # Validating saved outputs checks data identity only, as before.
    saved = run("import core; print(core.execution_identity(require_code=False)[1])", env, qpu)
    assert saved.returncode == 0, saved.stderr
    # Re-preparing the same directory with changed code is refused as well.
    again = prepare(env, qpu)
    assert again.returncode == 1 and "benchmark26/paths.py" in again.stderr

    paths.write_bytes(original)
    assert run(IDENTITY, env, qpu).returncode == 0


def test_manifest_without_dependency_inventory_is_rejected(tmp_path):
    artifacts = make_bundle(tmp_path)
    qpu = copy_code(tmp_path)
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(tmp_path / "run")}
    assert prepare(env, qpu).returncode == 0
    manifest_path = tmp_path / "run/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["code_dependency_hashes"]
    manifest_path.write_text(json.dumps(manifest))
    result = run(IDENTITY, env, qpu)
    assert result.returncode != 0 and "run_manifest_lacks_code_dependency_hashes" in result.stderr


def test_run_dir_inside_original_study_is_rejected(tmp_path):
    artifacts = make_bundle(tmp_path)
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(artifacts / STUDY / "new")}
    result = prepare(env)
    assert result.returncode == 2
    assert "outside the original study directory" in result.stderr
