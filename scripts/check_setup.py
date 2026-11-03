#!/usr/bin/env python3
"""Check code-release syntax and list external inputs without executing experiments."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-study-inputs', action='store_true',
                        help='Exit unsuccessfully when representative study inputs are missing.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    py_files = sorted(root.glob('**/*.py'))
    for path in py_files:
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted((root / 'benchmark_v1/protocol').glob('*.json')):
        json.loads(path.read_text())
    print(f'Parsed {len(py_files)} Python files and all component protocols.')
    print('Optional environment packages (presence only; versions/compatibility not checked):')
    for module in ('numpy', 'scipy', 'sklearn', 'qiskit', 'qiskit_ibm_runtime',
                   'torch', 'torch_geometric', 'openqasm3'):
        print(f'  {module}: {"available" if importlib.util.find_spec(module) else "missing"}')
    inputs = (
        'artifacts/postfreeze_qpu_workflow_mean_expansion_v1/execution_lock_v2.json',
        'artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv',
        'artifacts/postfreeze_qpu_workflow_mean_expansion_v1/cohort/feature_ledger.csv',
        'provenance/cost_source_index.json',
    )
    def resolve(name):
        if name.startswith('artifacts/'):
            return Path(os.environ.get('B26_ARTIFACTS', str(root/'artifacts'))) / Path(name).relative_to('artifacts')
        return root/name
    missing = [name for name in inputs if not resolve(name).is_file()]
    print('Representative external study inputs (not an exhaustive dependency inventory):')
    for name in inputs:
        print(f'  {name}: {"missing" if name in missing else "present"}')
    print('See docs/DATA_SOURCES.md for the execution boundary.')
    return 2 if args.require_study_inputs and missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
