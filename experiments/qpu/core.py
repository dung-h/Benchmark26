#!/usr/bin/env python3
"""Read cohort features and labels separately, and construct evaluation partitions."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS, RUN_DIR, recorded
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Inputs of the study: cohort, splits, protocol and execution lock.
OUT = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"
SPLITS = OUT / "splits"
PROTOCOL = OUT / "PROTOCOL.md"
LOCK = OUT / "execution_lock_v2.json"
# Fitting outputs (training/, classical/, validation/, analysis_v1/). Without
# B26_RUN_DIR these are the original study outputs.
RUN = RUN_DIR if RUN_DIR is not None else OUT
NEW_RUN = RUN_DIR is not None
RUN_MANIFEST = RUN / "run_manifest.json"
PREPARE_HINT = "set B26_RUN_DIR to a new directory and run experiments/qpu/prepare_run.py"


def sha(path):
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(value):
    """Return the SHA-256 of the canonical JSON of a value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    """Read a UTF-8 CSV file into a list of dicts."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_targets(allowed_ids):
    """Return the target seconds of the allowed observation IDs.

    Raises if the set is empty, a target is not positive and finite, an ID is duplicated, or a target is
    missing.
    """
    allowed = set(allowed_ids)
    if not allowed:
        raise RuntimeError("refuse_empty_authorized_target_partition")
    target_path = OUT / "cohort/targets.csv"
    found = {}
    with target_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r["canonical_observation_id"]
            if cid not in allowed:
                continue
            y = float(r["target_seconds"])
            if not math.isfinite(y) or y <= 0:
                raise RuntimeError(f"invalid_authorized_target:{cid}")
            if cid in found:
                raise RuntimeError(f"duplicate_authorized_target:{cid}")
            found[cid] = y
    if set(found) != allowed:
        raise RuntimeError(f"authorized_target_population_incomplete:{len(found)}/{len(allowed)}")
    return found


def atomic_json(path, value):
    """Write a value as JSON through a temporary `.partial` file and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        import os

        os.fsync(f.fileno())
    import os

    os.replace(tmp, path)


def atomic_csv(path, values, fieldnames=None):
    """Write rows as a CSV file atomically; raise if there are none."""
    if not values:
        raise RuntimeError(f"refuse_empty_csv:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(values[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(values)
        f.flush()
        import os

        os.fsync(f.fileno())
    import os

    os.replace(tmp, path)


def load_context(evaluation: str, include_targets: bool = True):
    """Load the cohort and splits of an evaluation (E1, E2 or E3) and verify them.

    Checks the 3,443 observations and their feature ledger, the positive shot counts, and the outer and
    inner splits. Targets are read only if `include_targets` is true. Returns the context, the targets
    and a receipt.
    """
    if evaluation not in {"E1", "E2", "E3"}:
        raise ValueError(f"unknown_evaluation:{evaluation}")
    cohort = OUT / "cohort"
    panel = rows(cohort / "panel.csv")
    ledger_rows = rows(cohort / "feature_ledger.csv")
    panel_by_id = {r["canonical_observation_id"]: r for r in panel}
    ledger = {}
    for p, l in zip(panel, ledger_rows):
        if p["canonical_observation_id"] != l["canonical_observation_id"]:
            raise RuntimeError("panel_ledger_order_or_identity_mismatch")
        cid = p["canonical_observation_id"]
        ledger[cid] = {
            **l,
            "graph_file": l["graph_artifact"],
            "graph_artifact": l["graph_artifact"],
            "graph_artifact_sha256": l["graph_artifact_sha256"],
            "graph_node_count": l["graph_node_count"],
            "graph_edge_count": l["graph_edge_count"],
            "graph_status": "available",
            "global_status": "available",
            **{f"global_{i:02d}": l[f"global_{i:02d}"] for i in range(51)},
        }
    if len(panel) != 3443 or len(ledger) != 3443 or len(panel_by_id) != 3443:
        raise RuntimeError("expanded_panel_identity_mismatch")
    targets = None
    if include_targets:
        # Kept only for offline full-target validation/reporting. Training paths use
        # load_targets(outer_train_ids) and never materialize held-out values.
        targets = load_targets(set(ledger))
    if any(float(r["shots"]) <= 0 for r in panel):
        raise RuntimeError("invalid_shots")
    panel_ids = sorted(ledger)
    base = {
        "panel": panel,
        "panel_by_id": panel_by_id,
        "ledger": ledger,
        "feature_root": ROOT,
        "all_ids": set(panel_ids),
        "evaluation": evaluation,
        "pins": {
            "panel_sha256": sha(cohort / "panel.csv"),
            "target_sha256": sha(cohort / "targets.csv"),
            "feature_ledger_sha256": sha(cohort / "feature_ledger.csv"),
            "native_features_sha256": sha(cohort / "native_baseline_features.csv"),
            "feature_receipt_sha256": sha(cohort / "feature_preflight_receipt.json"),
            "split_receipt_sha256": sha(SPLITS / "split_receipt.json"),
        },
    }
    if evaluation in {"E1", "E2"}:
        outer = rows(SPLITS / f"{evaluation}_outer.csv")
        inner = rows(SPLITS / f"{evaluation}_inner.csv")
        om = {r["canonical_observation_id"]: r for r in outer}
        if len(om) != 3443 or set(om) != set(ledger):
            raise RuntimeError(f"split_outer_id_mismatch:{evaluation}")
        for fold in range(5):
            ir = [r for r in inner if int(r["outer_fold"]) == fold]
            expected = {cid for cid in ledger if int(om[cid]["outer_fold"]) != fold}
            if {r["canonical_observation_id"] for r in ir} != expected:
                raise RuntimeError(f"split_inner_train_identity_mismatch:{evaluation}:{fold}")
        base.update({"outer": om, "inner": inner, "case_count": 5})
    else:
        outer = rows(SPLITS / "E3_outer.csv")
        inner = rows(SPLITS / "E3_inner.csv")
        cases = sorted({r["heldout_backend"] for r in outer})
        if len(cases) != 10 or len(outer) != 3443 * 10:
            raise RuntimeError("E3_outer_case_identity_mismatch")
        base.update({"outer": outer, "inner": inner, "cases": cases, "case_count": len(cases)})
    receipt = {
        "evaluation": evaluation,
        "rows": len(ledger),
        "cases": base["case_count"],
        "target_values_read": bool(include_targets),
        "target_hash_pinned_but_labels_separate": True,
        "outer_split_sha256": sha(SPLITS / f"{evaluation}_outer.csv"),
        "inner_split_sha256": sha(SPLITS / f"{evaluation}_inner.csv"),
        "pins": base["pins"],
    }
    return base, targets, receipt


def parts_for(context, fold):
    """Return the fit, validation and test IDs of a fold, with the inner cross-validation folds.

    For E1 and E2 the fold is an outer fold; for E3 it is a held-out backend. Inner fold 0 is validation
    and the others are fit. Raises on overlapping IDs, on group leakage between fit and validation (E1,
    E3) and between train and test (E1).
    """
    ev = context["evaluation"]
    all_ids = context["all_ids"]
    group = {r["canonical_observation_id"]: r["group_id"] for r in context["panel"]}
    if ev in {"E1", "E2"}:
        outer = context["outer"]
        test = sorted(cid for cid, r in outer.items() if int(r["outer_fold"]) == fold)
        train = sorted(all_ids - set(test))
        ir = [r for r in context["inner"] if int(r["outer_fold"]) == fold]
        assignment = {r["canonical_observation_id"]: int(r["inner_fold"]) for r in ir}
    else:
        b = context["cases"][fold]
        outer = [r for r in context["outer"] if r["heldout_backend"] == b]
        test = sorted(r["canonical_observation_id"] for r in outer if r["partition"] == "test")
        train = sorted(r["canonical_observation_id"] for r in outer if r["partition"] == "train")
        ir = [r for r in context["inner"] if r["heldout_backend"] == b]
        assignment = {r["canonical_observation_id"]: int(r["inner_fold"]) for r in ir}
    if set(train) & set(test) or set(train) | set(test) != all_ids:
        raise RuntimeError(f"outer_partition_invalid:{ev}:{fold}")
    if set(assignment) != set(train) or set(assignment.values()) != {0, 1, 2, 3}:
        raise RuntimeError(f"inner_assignment_invalid:{ev}:{fold}")
    fit = sorted(c for c in train if assignment[c] != 0)
    validation = sorted(c for c in train if assignment[c] == 0)
    if set(fit) & set(validation) or set(fit) & set(test) or set(validation) & set(test):
        raise RuntimeError(f"id_overlap:{ev}:{fold}")
    if ev in {"E1", "E3"} and {group[c] for c in fit} & {group[c] for c in validation}:
        raise RuntimeError(f"inner_group_leakage:{ev}:{fold}")
    if ev == "E1" and {group[c] for c in train} & {group[c] for c in test}:
        raise RuntimeError(f"outer_group_leakage:{ev}:{fold}")
    inner_cv = {
        k: (sorted(c for c in train if assignment[c] != k), sorted(c for c in train if assignment[c] == k))
        for k in range(4)
    }
    return {
        "train": train,
        "fit": fit,
        "validation": validation,
        "test": test,
        "assignments": assignment,
        "inner_cv": inner_cv,
    }


# Execution identity -------------------------------------------------------
#
# The original study fixed its protocol, inputs, graphs and code by SHA-256 in
# LOCK before any outcome was seen. This repository's code was adapted after
# that lock (portable paths, formatting), so it cannot continue the original
# run. It can start a new run: prepare_run.py checks the locked inputs and
# records the hashes of the current code in RUN/run_manifest.json. Fitting
# scripts then accept exactly that code and write only under RUN.


def load_lock() -> dict:
    """Load the execution lock; raise if the protocol or the lock does not exist."""
    if not PROTOCOL.is_file() or not LOCK.is_file():
        raise RuntimeError("expansion_protocol_not_locked")
    return json.loads(LOCK.read_text(encoding="utf-8"))


def locked_input_problems(lock: dict) -> list[str]:
    """Differences between the bundle and the locked protocol, inputs and graphs."""
    problems = []
    if lock.get("status") != "LOCKED_PRE_OUTCOME":
        problems.append(f"lock_status:{lock.get('status')}")
    if lock.get("protocol_markdown_sha256") != sha(PROTOCOL):
        problems.append("protocol_markdown_hash")
    for kind in ("input_hashes", "unique_graph_hashes"):
        for relative, digest in lock.get(kind, {}).items():
            path = recorded(relative)
            if not path.is_file():
                problems.append(f"missing:{relative}")
            elif sha(path) != digest:
                problems.append(f"hash_mismatch:{relative}")
    return problems


def current_code_hashes(lock: dict) -> dict:
    """SHA-256 of this repository's version of every code file named in the lock."""
    out = {}
    for relative in lock.get("code_hashes", {}):
        path = recorded(relative)
        out[relative] = sha(path) if path.is_file() else None
    return out


# Entry points of a new fitting run. Their repository-local dependency closure is
# pinned in the run manifest in addition to the files named by the historical lock.
RUN_ENTRYPOINTS = (
    "experiments/qpu/core.py",
    "experiments/qpu/prepare_run.py",
    "experiments/qpu/train_neural.py",
    "experiments/qpu/run_classical.py",
    "experiments/qpu/graph_ablation_runner.py",
)
SOURCE_SUFFIXES = (".cpp", ".c", ".h")
DEPENDENCY_RULE = (
    "repository files reachable from RUN_ENTRYPOINTS and the files named by the historical lock through "
    "static Python imports (module level and function level), the parent package __init__.py files, and "
    "C/C++ sources named by a string literal in the same directory"
)


def _module_files(name: str, bases: tuple[Path, ...]) -> list[Path]:
    """Repository files a dotted module name can resolve to (module, package, parents)."""
    found = []
    parts = name.split(".")
    for base in bases:
        for depth in range(1, len(parts) + 1):
            stem = base.joinpath(*parts[:depth])
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file():
                    found.append(candidate)
    return found


def _direct_dependencies(path: Path) -> set[Path]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bases = (path.parent, ROOT)
    deps: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.update(_module_files(alias.name, bases))
        elif isinstance(node, ast.ImportFrom):
            base = path.parent if node.level else None
            for _ in range(max(node.level - 1, 0)):
                base = base.parent
            roots = (base,) if base is not None else bases
            module = node.module or ""
            if module:
                deps.update(_module_files(module, roots))
            for alias in node.names:
                deps.update(_module_files(f"{module}.{alias.name}" if module else alias.name, roots))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(SOURCE_SUFFIXES):
            candidate = path.parent / Path(node.value).name
            if candidate.is_file():
                deps.add(candidate)
    return deps


def code_dependency_inventory(lock: dict) -> dict:
    """SHA-256 of every repository file a new fitting run can execute, keyed by repository path.

    Deterministic: the closure of RUN_ENTRYPOINTS and of the lock's code files
    under DEPENDENCY_RULE. Unlike :func:`current_code_hashes`, it includes files
    introduced after the original lock, such as ``benchmark26/paths.py``.
    """
    root = ROOT.resolve()
    pending = [ROOT / p for p in RUN_ENTRYPOINTS]
    pending += [recorded(p) for p in lock.get("code_hashes", {})]
    seen: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(root):
            continue
        seen.add(path)
        if path.suffix == ".py":
            pending.extend(_direct_dependencies(path))
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(seen)}


def historical_code_differences(lock: dict, current: dict | None = None) -> list[str]:
    """Return the code files whose current hash differs from the one in the lock."""
    current = current_code_hashes(lock) if current is None else current
    return sorted(k for k, v in lock.get("code_hashes", {}).items() if current.get(k) != v)


def _run_dir_problem() -> str | None:
    if not NEW_RUN:
        return None
    run, study = RUN.resolve(), OUT.resolve()
    if run == study or study in run.parents or run in study.parents:
        return f"B26_RUN_DIR must be outside the original study directory {study}"
    return None


def run_manifest_problems(lock: dict, current: dict | None = None, check_code: bool = True) -> list[str]:
    """Why RUN/run_manifest.json does not authorize fitting with the current code."""
    problem = _run_dir_problem()
    if problem:
        return [problem]
    if not RUN_MANIFEST.is_file():
        return [f"missing {RUN_MANIFEST}; {PREPARE_HINT}"]
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("status") != "NEW_RUN_ADAPTED_CODE":
        problems.append(f"run_manifest_status:{manifest.get('status')}")
    if manifest.get("historical_lock_sha256") != sha(LOCK):
        problems.append("run_manifest_names_a_different_historical_lock")
    if not check_code:
        return problems
    current = current_code_hashes(lock) if current is None else current
    changed = sorted(k for k, v in current.items() if manifest.get("code_hashes", {}).get(k) != v)
    if changed:
        problems.append(f"code_changed_since_run_manifest:{','.join(changed)}; use a new B26_RUN_DIR")
    pinned = manifest.get("code_dependency_hashes")
    if not isinstance(pinned, dict) or not pinned:
        problems.append(f"run_manifest_lacks_code_dependency_hashes; {PREPARE_HINT} in a new directory")
        return problems
    inventory = code_dependency_inventory(lock)
    differ = sorted(k for k in set(pinned) | set(inventory) if pinned.get(k) != inventory.get(k))
    if differ:
        problems.append(f"code_dependency_changed_since_run_manifest:{','.join(differ)}; use a new B26_RUN_DIR")
    return problems


def execution_identity(require_code: bool = True) -> tuple[dict, str]:
    """Return ``(lock, identity_sha256)`` for fitting or for validating fitted outputs.

    The locked protocol, inputs and graphs must always match. With
    ``require_code`` (fitting), the code must also match: the original lock for
    the original study outputs, or RUN/run_manifest.json for a new run. The
    identity hash is the one that fitted outputs record as
    ``protocol_lock_sha256``.
    """
    lock = load_lock()
    problems = locked_input_problems(lock)
    if problems:
        raise RuntimeError("locked_inputs_differ:" + ";".join(problems[:5]))
    if NEW_RUN:
        problems = run_manifest_problems(lock, check_code=require_code)
        if problems:
            raise RuntimeError("new_run_not_prepared:" + ";".join(problems))
        return lock, sha(RUN_MANIFEST)
    if require_code:
        differ = historical_code_differences(lock)
        if differ:
            raise RuntimeError(
                f"historical_code_lock_mismatch:{differ[0]} (+{len(differ) - 1} more). "
                f"This code cannot extend the original run; {PREPARE_HINT}"
            )
    return lock, sha(LOCK)
