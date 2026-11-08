"""The shared Qonductor model-choice tables pass their check."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qonductor_selection_verify",
                                              ROOT / "experiments/qonductor_selection/verify_summary.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_shared_tables_pass_their_check():
    assert verify.problems() == []
