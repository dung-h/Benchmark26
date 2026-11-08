#!/usr/bin/env python3
"""Check the code release and report which external study inputs are available.

The default mode checks only what the repository contains: Python syntax,
protocol JSON and the hash chain between frozen protocol files. It succeeds
without the external study bundle. ``--require-study-inputs`` also fails when
representative bundle files are missing.
"""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from benchmark26.integrity import check_frozen_pins
from benchmark26.paths import ARTIFACTS, PROTOCOLS, ROOT, recorded

OPTIONAL_PACKAGES = ("numpy", "scipy", "sklearn", "pandas", "qiskit", "qiskit_ibm_runtime", "torch", "torch_geometric")
STUDY_INPUTS = (
    "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/execution_lock_v2.json",
    "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv",
    "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv",
    "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/splits/E1_outer.csv",
)


def main() -> int:
    """Check the code release and report which external study inputs are available.

    Parses every Python and protocol file, verifies the frozen protocol hash chain
    (exit 1 if it is broken), lists optional packages and study-bundle files, and exits
    with 2 when `--require-study-inputs` is set and an input is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--require-study-inputs", action="store_true", help="exit with status 2 if a study input is missing"
    )
    args = parser.parse_args()

    sources = [p for p in ROOT.rglob("*.py") if not {".git", ".venv", "work", "inputs"} & set(p.parts)]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    protocols = sorted(PROTOCOLS.rglob("*.json"))
    for path in protocols:
        json.loads(path.read_text(encoding="utf-8"))
    print(f"Parsed {len(sources)} Python files and {len(protocols)} protocol files.")

    broken = [(pin, status) for pin, status in check_frozen_pins() if status != "match"]
    for pin, status in broken:
        print(f"  protocol pin {status}: {pin.path} (pinned by {pin.source})")
    if broken:
        print("FAIL: frozen protocol files no longer match their pins.")
        return 1
    print("Frozen protocol hash chain: intact.")

    print("Optional packages (presence only):")
    for module in OPTIONAL_PACKAGES:
        print(f"  {module}: {'available' if importlib.util.find_spec(module) else 'missing'}")

    missing = [name for name in STUDY_INPUTS if not recorded(name).is_file()]
    print(f"Study bundle at {ARTIFACTS}:")
    for name in STUDY_INPUTS:
        print(f"  {name}: {'missing' if name in missing else 'present'}")
    if missing:
        print("The bundle is not included in this repository; see docs/REPRODUCTION.md.")
    return 2 if args.require_study_inputs and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
