#!/usr/bin/env python3
"""Inspect study inputs or run the synthetic smoke pipeline and review checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Inspect external input availability')
    parser.add_argument('--stage', choices=['status', 'smoke', 'screen', 'analytical', 'latency', 'acquire'], default='status')
    parser.add_argument('--config', type=Path, help='JSON config consumed by --stage smoke')
    parser.add_argument('--output', type=Path, help='New output directory for --stage smoke')
    parser.add_argument('--inputs', type=Path, help='Destination directory for upstream acquisition')
    args = parser.parse_args()
    if args.config and args.stage != 'smoke':
        parser.error('--config is consumed only by --stage smoke')
    stage = 'status' if args.status else args.stage
    if stage == 'status':
        relative = ['artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv',
                    'artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv',
                    'artifacts/postfreeze_qpu_workflow_mean_expansion_v1/execution_lock_v2.json']
        print(json.dumps({'mode': 'input_inventory', 'inputs': {p: (Path(os.environ.get('B26_ARTIFACTS', str(ROOT/'artifacts')))/Path(p).relative_to('artifacts')).is_file() for p in relative},
                          'note': 'File presence is not a full execution-readiness check. See docs/CODE_MAP.md for experiment entrypoints.'}, indent=2))
        return 0
    if stage == 'smoke':
        if not args.output:
            parser.error('--stage smoke requires --output NEW_DIRECTORY')
        from smoke_pipeline import run
        run(args.config or ROOT/'configs/smoke.json', args.output)
        return 0
    if stage == 'acquire':
        if not args.inputs:
            parser.error('--stage acquire requires --inputs DESTINATION')
        command = [sys.executable, str(ROOT/'scripts/fetch_upstream_data.py'), '--source', 'all', '--destination', str(args.inputs)]
    elif stage == 'latency':
        command = [sys.executable, str(ROOT/'experiments/latency/verify_summary.py')]
    else:
        test = 'test_qonductor_recipe_screen.py' if stage == 'screen' else 'test_analytical_path_kernel.py'
        command = [sys.executable, '-m', 'pytest', str(ROOT/'tests'/test)]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
