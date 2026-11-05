"""The Scholten-style estimate, its CLOPS table and the shared results."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


estimate = _load("scholten_estimate", ROOT / "experiments/scholten/estimate.py")
verify = _load("scholten_verify", ROOT / "experiments/scholten/verify_summary.py")


def test_formula_is_shots_times_depth_over_clops():
    assert estimate.estimate(1000, 50, 2500) == pytest.approx(20.0)


def test_backend_name_drops_the_ibm_prefix():
    assert estimate.backend_name("x::ibm_osaka::2024") == "osaka"
    assert estimate.backend_name("x::ibmq_lima::2023") == "lima"


def test_shot_ranges_cover_every_shot_count():
    assert estimate.shot_range(1) == "1-99"
    assert estimate.shot_range(500) == "100-500"
    assert estimate.shot_range(1024) == "501-2000"
    assert estimate.shot_range(20000) == ">8000"


def test_three_backends_have_no_clops():
    clops = estimate.load_clops()
    assert {name for name, value in clops.items() if value is None} == {"belem", "lima", "quito"}


def test_clops_table_matches_installed_qiskit_ibm_runtime():
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    folder = Path(fake.__file__).parent / "backends"
    for name, value in estimate.load_clops().items():
        conf = json.loads((folder / name / f"conf_{name}.json").read_text())
        found = next(((k, float(conf[k])) for k in ("clops_v", "clops_h")
                      if isinstance(conf.get(k), (int, float)) and conf[k] > 0), None)
        assert found == value, name


def test_shared_results_pass_their_check():
    assert verify.problems() == []
