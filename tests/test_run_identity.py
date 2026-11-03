"""New-run preparation keeps the historical lock intact and pins the current code."""

from __future__ import annotations

import hashlib
import json
import os
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


def run(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=QPU, env={**os.environ, **env}, capture_output=True, text=True, check=False
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
    lock_before = (artifacts / STUDY / "execution_lock_v2.json").read_bytes()
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(tmp_path / "run")}
    missing = run(IDENTITY, env)
    assert missing.returncode != 0 and "new_run_not_prepared" in missing.stderr

    prepared = subprocess.run(
        [sys.executable, str(QPU / "prepare_run.py")],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    manifest_path = tmp_path / "run/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "NEW_RUN_ADAPTED_CODE"
    assert manifest["code_hashes"] == {f"research/{STUDY}/core.py": sha(QPU / "core.py")}
    assert manifest["code_differing_from_historical_lock"] == [f"research/{STUDY}/core.py"]

    accepted = run(IDENTITY, env)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == sha(manifest_path)
    assert (artifacts / STUDY / "execution_lock_v2.json").read_bytes() == lock_before

    # A changed input invalidates both the historical and the new-run identity.
    (artifacts / STUDY / "cohort/panel.csv").write_text("changed\n")
    changed = run(IDENTITY, env)
    assert changed.returncode != 0 and "locked_inputs_differ" in changed.stderr


def test_run_dir_inside_original_study_is_rejected(tmp_path):
    artifacts = make_bundle(tmp_path)
    env = {"B26_ARTIFACTS": str(artifacts), "B26_RUN_DIR": str(artifacts / STUDY / "new")}
    result = subprocess.run(
        [sys.executable, str(QPU / "prepare_run.py")],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "outside the original study directory" in result.stderr
