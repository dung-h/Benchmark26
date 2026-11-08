#!/usr/bin/env python3
"""Inspect study inputs or run the synthetic smoke pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import argparse
import json
import sys
from pathlib import Path

from benchmark26.paths import ARTIFACTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Run one pipeline stage (`--stage`, default `status`).

    - `status`: print which study inputs are present (presence only, not readiness).
    - `smoke`: run the synthetic example; needs `--output`, and `--config` is accepted only here.

    Returns the exit code of the stage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Inspect external input availability")
    parser.add_argument("--stage", choices=["status", "smoke"], default="status")
    parser.add_argument("--config", type=Path, help="JSON config consumed by --stage smoke")
    parser.add_argument("--output", type=Path, help="New output directory for --stage smoke")
    args = parser.parse_args()
    if args.config and args.stage != "smoke":
        parser.error("--config is consumed only by --stage smoke")
    stage = "status" if args.status else args.stage
    if stage == "status":
        relative = [
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv",
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv",
            "artifacts/postfreeze_qpu_workflow_mean_expansion_v1/execution_lock_v2.json",
        ]
        print(
            json.dumps(
                {
                    "mode": "input_inventory",
                    "inputs": {p: (ARTIFACTS / Path(p).relative_to("artifacts")).is_file() for p in relative},
                    "note": "File presence is not a full execution-readiness check. See docs/REPRODUCTION.md for experiment entrypoints.",
                },
                indent=2,
            )
        )
        return 0
    if stage == "smoke":
        if not args.output:
            parser.error("--stage smoke requires --output NEW_DIRECTORY")
        from smoke_pipeline import run

        run(args.config or ROOT / "configs/smoke.json", args.output)
        return 0
    return 1  # unreachable: argparse restricts the stage


if __name__ == "__main__":
    raise SystemExit(main())
