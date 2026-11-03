"""Exported reference files must match the SHA-256 pins in the execution contract."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = json.loads((ROOT / "benchmark_v1/protocol/estimator_execution.json").read_text())
EXPORTED = ("seed_registry", "current_guard", "fidelity_guard")


@pytest.mark.parametrize("name", EXPORTED)
def test_registry_matches_pin(name):
    pin = EXECUTION["input_pins"][name]
    path = ROOT / pin["path"]
    assert path.is_file(), pin["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]


def test_feature_implementation_sources_exist():
    for name in ("materialize_mali_full_features", "mali_full_features", "qasm3_hardware_wire_adapter"):
        assert (ROOT / "benchmark_v1/scripts" / f"{name}.py").is_file(), name
