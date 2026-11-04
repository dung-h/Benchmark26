#!/usr/bin/env python3
"""Prepare a directory for a new fitting run of the main study with this code.

The original execution lock pins the protocol, inputs, graphs and code of the
saved study. This code differs from the locked code, so it may only start a
new run, written outside the original study directory. This script checks the
locked protocol, inputs and graphs against the bundle and records in
``$B26_RUN_DIR/run_manifest.json`` the hashes of (a) the current version of each
code file named in the lock and (b) every repository file the fitting code can
execute (``core.code_dependency_inventory``), including files added after the
lock such as ``benchmark26/paths.py``. ``train_neural.py`` and
``run_classical.py`` then accept exactly this code and write under
``$B26_RUN_DIR``. The original lock and outputs are not modified.

The run inherits every scientific setting of the original lock (dataset, splits,
seeds, epochs, candidate sets). Its results are a new experiment, not a check
of the saved one.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from core import (  # noqa: E402
    LOCK,
    NEW_RUN,
    PROTOCOL,
    RUN,
    RUN_MANIFEST,
    DEPENDENCY_RULE,
    atomic_json,
    code_dependency_inventory,
    current_code_hashes,
    historical_code_differences,
    load_lock,
    locked_input_problems,
    run_manifest_problems,
    sha,
    _run_dir_problem,
)

ROOT = Path(__file__).resolve().parents[2]


def git_revision() -> str | None:
    """Return the git revision of the repository, with `-dirty` if the working tree has changes, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    """Prepare a directory for a new fitting run with the current code.

    Checks that `B26_RUN_DIR` is set and outside the study, verifies the locked protocol, inputs and
    graphs, and writes run_manifest.json with the hashes of the current code. An existing manifest must
    match. Returns 0 on success.
    """
    if not NEW_RUN:
        print("B26_RUN_DIR is not set. Set it to a new directory for the run's outputs.", file=sys.stderr)
        return 2
    problem = _run_dir_problem()
    if problem:
        print(problem, file=sys.stderr)
        return 2
    lock = load_lock()
    problems = locked_input_problems(lock)
    if problems:
        print("The bundle does not match the locked protocol, inputs or graphs:", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 1
    current = current_code_hashes(lock)
    # Some files named in the original lock are internal process steps and are not distributed.
    not_distributed = sorted(k for k, v in current.items() if v is None)
    if RUN_MANIFEST.is_file():
        remaining = run_manifest_problems(lock, current)
        if remaining:
            print("An existing run manifest does not match this code:", file=sys.stderr)
            for p in remaining:
                print("  " + p, file=sys.stderr)
            return 1
        print(f"Run already prepared: {RUN_MANIFEST}")
        return 0
    differ = historical_code_differences(lock, current)
    manifest = {
        "status": "NEW_RUN_ADAPTED_CODE",
        "purpose": "new fitting run with adapted code; does not continue or replace the original study outputs",
        "historical_lock": str(LOCK),
        "historical_lock_sha256": sha(LOCK),
        "protocol_markdown_sha256": sha(PROTOCOL),
        "locked_inputs_verified": len(lock.get("input_hashes", {})),
        "locked_graphs_verified": len(lock.get("unique_graph_hashes", {})),
        "code_hashes": current,
        "code_dependency_hashes": code_dependency_inventory(lock),
        "code_dependency_rule": DEPENDENCY_RULE,
        "code_identical_to_historical_lock": sorted(set(current) - set(differ)),
        "code_not_distributed": not_distributed,
        "code_differing_from_historical_lock": differ,
        "inherited_settings": {k: lock[k] for k in ("neural", "classical", "evaluations") if k in lock},
        "repository_revision": git_revision(),
        "python": platform.python_version(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RUN.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "run_manifest": str(RUN_MANIFEST),
                "run_manifest_sha256": sha(RUN_MANIFEST),
                "locked_inputs_verified": manifest["locked_inputs_verified"],
                "locked_graphs_verified": manifest["locked_graphs_verified"],
                "code_files_differing_from_historical_lock": len(differ),
                "code_dependency_files_pinned": len(manifest["code_dependency_hashes"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
