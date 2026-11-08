"""Frozen protocols, registries and reference programs must match their pins."""

import pytest

from benchmark26.integrity import check_frozen_pins
from benchmark26.paths import ROOT, recorded


@pytest.mark.parametrize("pin,status", check_frozen_pins(), ids=lambda value: getattr(value, "path", value))
def test_frozen_file_matches_pin(pin, status):
    assert status == "match", f"{pin.path} {status} (pinned by {pin.source})"


def test_original_layout_paths_resolve():
    assert recorded("benchmark_v1/protocol/mali_full_features.json") == ROOT / "protocols/mali_full_features.json"
    assert recorded("benchmark_v1/registry/seed_registry.json") == ROOT / "protocols/registry/seed_registry.json"
    assert recorded("benchmark_v1/scripts/run_mali_full_features.py") == ROOT / "benchmark26/mali_training.py"
    assert recorded("benchmark_v1/path_estimators/run.py") == ROOT / "benchmark26/path_estimators.py"


def test_feature_implementation_sources_exist():
    from benchmark26.paths import ORIGINAL_MODULES

    for module in ORIGINAL_MODULES.values():
        assert (ROOT / "benchmark26" / f"{module}.py").is_file(), module
