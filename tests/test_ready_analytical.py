"""The published ready-input timing aggregates reproduce from the anonymous group means."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_means_and_manifest_verify():
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/latency/verify_ready_analytical.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
