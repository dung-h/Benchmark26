#!/usr/bin/env python3
"""Canonical entrypoint for the Quantum Runtime Estimation Benchmark pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_stage_prerequisites():
    """Inspect availability of inputs for each pipeline stage."""
    stages = {
        "1_acquire": {
            "name": "Upstream Source Acquisition",
            "ready": (ROOT / "upstream/data_sources.json").is_file(),
            "runner": "scripts/fetch_upstream_data.py",
            "notes": "Downloads pinned source files and verifies SHA256 hashes.",
        },
        "2_screen": {
            "name": "Qonductor Recipe Qualification",
            "ready": (ROOT / "upstream/qonductor_recipe_screen.csv").is_file(),
            "runner": "upstream/qualification_provenance.json",
            "notes": "4,482 candidate screening decisions (MQT 1.0.3 prepared readout state).",
        },
        "3_split": {
            "name": "Group-Disjoint Split Construction",
            "ready": (ROOT / "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv").is_file(),
            "runner": "research/postfreeze_qpu_workflow_mean_expansion_v1/prepare_splits.py",
            "notes": "Constructs E1 (group holdout), E2 (random), E3 (backend holdout) folds.",
        },
        "4_train": {
            "name": "Neural & Classical Model Fitting",
            "ready": (ROOT / "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv").is_file(),
            "runner": "research/postfreeze_qpu_workflow_mean_expansion_v1/train_neural.py",
            "notes": "Fits G1, G2, M0, M1 (neural) and R0 selector across outer folds.",
        },
        "5_analytical": {
            "name": "Analytical Duration Evaluation",
            "ready": (ROOT / "research/postfreeze_notads_hyperq_track_a_v1/path_kernel.cpp").is_file(),
            "runner": "scripts/verify_analytical_kernel.py",
            "notes": "Evaluates Qiskit Target, QCRE unitary, and NoTaDS depth metrics.",
        },
        "6_report": {
            "name": "Metric Aggregation & Reporting",
            "ready": (ROOT / "artifacts/estimator_cost/warm_accuracy_cost.csv").is_file(),
            "runner": "scripts/verify_cost.py",
            "notes": "Aggregates pooled, source-balanced, and group-balanced MAE and latency.",
        },
    }
    return stages


def print_status():
    stages = check_stage_prerequisites()
    print("=" * 80)
    print("Benchmark26 — Pipeline Stage Readiness")
    print("=" * 80)
    print(f"{'Stage':<16} | {'Status':<12} | {'Runner / Target':<35} | {'Notes'}")
    print("-" * 80)
    for key, info in sorted(stages.items()):
        status = "[READY]" if info["ready"] else "[REQUIRES BUNDLE]"
        print(f"{info['name']:<16} | {status:<12} | {info['runner']:<35} | {info['notes']}")
    print("-" * 80)
    print("Note: Stages marked [REQUIRES BUNDLE] depend on external study tables.")
    print("See docs/DATA_SOURCES.md and docs/REPRODUCTION.md for instructions.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Unified runner for the Quantum Runtime Estimation Benchmark pipeline."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the readiness status of all pipeline stages and exit.",
    )
    parser.add_argument(
        "--stage",
        choices=["acquire", "screen", "analytical", "status"],
        default="status",
        help="Pipeline stage to execute (default: status).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/unified_benchmark.yaml",
        help="Path to study configuration file (default: configs/unified_benchmark.yaml).",
    )

    args = parser.parse_args()

    if args.status or args.stage == "status":
        print_status()
        return 0

    if args.stage == "acquire":
        print("To download upstream sources, use:")
        print("  python3 scripts/fetch_upstream_data.py --source all --destination /path/outside/inputs")
        return 0

    if args.stage == "screen":
        print("Validating recipe screen provenance...")
        import subprocess
        return subprocess.run([sys.executable, "-m", "pytest", "tests/test_qonductor_recipe_screen.py"]).returncode

    if args.stage == "analytical":
        print("Running analytical recurrence tests...")
        import subprocess
        return subprocess.run([sys.executable, "-m", "pytest", "tests/test_analytical_path_kernel.py"]).returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
