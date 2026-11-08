"""Repository and external-input locations.

Every location that depends on the local machine is read from one environment
variable here; see ``configs/paths.env.example``. Frozen protocol files record
paths in the layout of the original study (for example
``benchmark_v1/protocol/...`` or ``artifacts/...``); :func:`recorded` maps such
a path to its location in this repository or in the external study bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols"
REGISTRY = PROTOCOLS / "registry"
REFERENCE_CODE = PROTOCOLS / "reference_code"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Prepared study bundle (dataset tables, splits, features, predictions, receipts).
ARTIFACTS = _env_path("B26_ARTIFACTS", ROOT / "artifacts")
# Input tables of the dataset: the list of runs assembled from the sources, and its inventories.
DATASET_INPUTS = _env_path("B26_DATASET_INPUTS", ROOT / "inputs/dataset_inputs")
# Scratch area for new outputs.
WORK = _env_path("B26_WORK", ROOT / "work")
# Output directory of a new fitting run with this code (see experiments/qpu/prepare_run.py).
# Unset: fitting scripts address the original study outputs and require the original code.
RUN_DIR = Path(os.environ["B26_RUN_DIR"]).expanduser() if os.environ.get("B26_RUN_DIR") else None

# Upstream source checkouts and archives.
MALI_SOURCE = _env_path("B26_MALI_SOURCE", ROOT / "inputs/mali")
MALI_QASM = _env_path("B26_MALI_QASM", MALI_SOURCE / "data/quantum_circuits")
QONDUCTOR_SOURCE = _env_path("B26_QONDUCTOR_SOURCE", ROOT / "inputs/qonductor")
QONDUCTOR_ARCHIVE = _env_path("B26_QONDUCTOR_ARCHIVE", QONDUCTOR_SOURCE / "data/database/circuits.zip")
QONDUCTOR_DATABASE = _env_path("B26_QONDUCTOR_DATABASE", QONDUCTOR_SOURCE / "data/database/quantum_scheduler.db")
QONDUCTOR_TABLE = _env_path("B26_QONDUCTOR_TABLE", ROOT / "inputs/qonductor_observations.parquet")

# Interpreter whose package versions are recorded for neural training.
NEURAL_PYTHON = _env_path("B26_NEURAL_PYTHON", Path(sys.executable))

# Original-study module names of the shared code in ``benchmark26/``.
ORIGINAL_MODULES = {
    "bounded_feature_pool": "feature_pool",
    "gpu_lease": "gpu_lease",
    "mali_full_features": "mali_features",
    "materialize_mali_full_features": "mali_inputs",
    "run_mali_full_features": "mali_training",
    "run_mali_batched": "mali_batching",
    "qasm3_hardware_wire_adapter": "qasm3_wires",
    "qonductor_native_adapter": "qonductor_features",
    "qonductor_feature_tasks": "qonductor_feature_tasks",
    "materialize_qonductor_native_features": "qonductor_inputs",
    "run_qonductor_grouped_reproduction": "qonductor_models",
    "run_strict_logical_qonductor": "qonductor_regression",
    "run_recovery_c133_c135_v3": "circuit_representation",
    "prepare_logical_benchmark": "logical_inputs",
    "shots_core": "shots",
}

# Scripts of the original study that this repository renamed (original name -> current name).
RENAMED_QPU_SCRIPTS = {"prepare_cohort.py": "prepare_dataset.py"}


def recorded(path: str | Path) -> Path:
    """Resolve a path written in a frozen protocol or receipt."""
    text = str(path).replace("\\", "/")
    if Path(text).is_absolute():
        return Path(text)
    head, _, rest = text.partition("/")
    if head == "artifacts":
        return ARTIFACTS / rest
    if head == "work":
        return WORK / rest
    if text.startswith("benchmark_v1/protocol/"):
        return PROTOCOLS / text.removeprefix("benchmark_v1/protocol/")
    if text.startswith("benchmark_v1/registry/"):
        return REGISTRY / text.removeprefix("benchmark_v1/registry/")
    if text.startswith(("benchmark_v1/scripts/", "benchmark_v1/shots/")):
        name = Path(text).stem
        if name in ORIGINAL_MODULES:
            return ROOT / "benchmark26" / f"{ORIGINAL_MODULES[name]}.py"
        return REFERENCE_CODE / Path(text).name
    if text.startswith("docs/final_completion/qpu_workflow_mean_expansion_v1/"):
        return DATASET_INPUTS / Path(text).name
    if text.startswith("research/postfreeze_qpu_workflow_mean_expansion_v1/"):
        name = Path(text).name
        return ROOT / "experiments/qpu" / RENAMED_QPU_SCRIPTS.get(name, name)
    if text == "research/postfreeze_mali_shots_ablation_v1/shots_core.py":
        return ROOT / "benchmark26/shots.py"
    if text.startswith("benchmark_v1/path_estimators/"):
        name = Path(text).name
        return ROOT / "benchmark26" / ("path_estimators.py" if name == "run.py" else name)
    return ROOT / text


def record_name(path: str | Path) -> str:
    """Name a file the way the original study recorded it in receipts (inverse of :func:`recorded`)."""
    target = Path(path).resolve()
    bases = (
        (ARTIFACTS, "artifacts"),
        (WORK, "work"),
        (DATASET_INPUTS, "docs/final_completion/qpu_workflow_mean_expansion_v1"),
        (REGISTRY, "benchmark_v1/registry"),
        (REFERENCE_CODE, "benchmark_v1/scripts"),
        (PROTOCOLS, "benchmark_v1/protocol"),
    )
    for base, prefix in bases:
        if target.is_relative_to(base.resolve()):
            return f"{prefix}/{target.relative_to(base.resolve()).as_posix()}"
    package = (ROOT / "benchmark26").resolve()
    if target.parent == package:
        original = {new: old for old, new in ORIGINAL_MODULES.items()}
        if target.name == "path_estimators.py":
            return "benchmark_v1/path_estimators/run.py"
        if target.name == "path_kernel.cpp":
            return "benchmark_v1/path_estimators/path_kernel.cpp"
        if target.name == "shots.py":
            return "benchmark_v1/shots/shots_core.py"
        if target.stem in original:
            return f"benchmark_v1/scripts/{original[target.stem]}.py"
    if target.is_relative_to(ROOT.resolve()):
        return target.relative_to(ROOT.resolve()).as_posix()
    return str(target)
