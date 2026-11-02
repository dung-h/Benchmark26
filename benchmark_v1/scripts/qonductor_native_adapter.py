"""Side-effect-free AST adapter for the pinned Qonductor feature and loader methods."""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import logging
from pathlib import Path
import subprocess
import zipfile
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Clbit


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(source_root: Path, revision: str, relative_path: str, expected_sha256: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "show", f"{revision}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    blob = completed.stdout
    observed = sha256_bytes(blob)
    if observed != expected_sha256:
        raise RuntimeError(f"pinned source blob hash mismatch for {relative_path}: {observed}")
    return blob


def _compile_methods(module: ast.Module, method_names: set[str], class_name: str) -> type:
    original = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "RegressionEstimator"
    )
    methods = [
        node for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names
    ]
    if {node.name for node in methods} != method_names:
        raise RuntimeError(f"pinned Qonductor methods missing: {method_names - {n.name for n in methods}}")
    adapter_class = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=methods, decorator_list=[]
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[adapter_class], type_ignores=[]))
    scope: dict[str, Any] = {
        "np": np,
        "defaultdict": defaultdict,
        "Clbit": Clbit,
        "QuantumCircuit": QuantumCircuit,
    }
    exec(compile(isolated, "<pinned-Qonductor-feature-methods>", "exec"), scope)
    return scope[class_name]


def _compile_archive_loader(module: ast.Module) -> Any:
    loader = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_circuit_from_archive"
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[loader], type_ignores=[]))
    scope: dict[str, Any] = {
        "pathlib": __import__("pathlib"),
        "zipfile": zipfile,
        "QuantumCircuit": QuantumCircuit,
        "qasm3": qasm3,
        "logger": logging.getLogger("pinned_qonductor_archive_loader"),
        # Execution always uses the upstream default unpack=False path.
        "unpack_archive": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("upstream unpack path is outside this benchmark adapter")
        ),
    }
    exec(compile(isolated, "<pinned-Qonductor-archive-loader>", "exec"), scope)
    return scope["load_circuit_from_archive"]


def _compile_qasm_string_parser(module: ast.Module) -> Any:
    """Compile the exact QASM2-then-QASM3 try/except from the pinned loader."""
    loader = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_circuit_from_archive"
    )
    parser_try = next(
        node for node in ast.walk(loader)
        if isinstance(node, ast.Try)
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "from_qasm_str"
            for candidate in ast.walk(node)
        )
    )

    class RenameCircuitString(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id == "circuit_string":
                return ast.copy_location(ast.Name(id="qasm_string", ctx=node.ctx), node)
            return node

    parser_try = RenameCircuitString().visit(ast.fix_missing_locations(parser_try))
    parser = ast.FunctionDef(
        name="parse_pinned_qasm_string",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="qasm_string")], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[parser_try, ast.Return(value=ast.Name(id="circuit", ctx=ast.Load()))],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[parser], type_ignores=[]))
    scope: dict[str, Any] = {
        "QuantumCircuit": QuantumCircuit,
        "qasm3": qasm3,
        "logger": logging.getLogger("pinned_qonductor_qasm_parser"),
    }
    exec(compile(isolated, "<pinned-Qonductor-QASM2-QASM3-parser>", "exec"), scope)
    return scope["parse_pinned_qasm_string"]


def load_pinned_qonductor_methods(source_root: Path, source_pin: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Compile only upstream extractor methods and the exact archive loader.

    The estimator constructor and its model-training/file-writing methods are
    never imported or executed. Pinned Git blobs are hash checked before use.
    """
    revision = str(source_pin["revision"])
    files = source_pin["files"]
    estimator_path = "src/execution_time/regression_estimator.py"
    archive_path = "src/utils/circuit_archive.py"
    pinned_blobs = {
        relative_path: _git_blob(source_root, revision, relative_path, expected_sha)
        for relative_path, expected_sha in files.items()
    }
    estimator_blob = pinned_blobs[estimator_path]
    archive_blob = pinned_blobs[archive_path]
    estimator_tree = ast.parse(estimator_blob.decode("utf-8"), filename=estimator_path)
    archive_tree = ast.parse(archive_blob.decode("utf-8"), filename=archive_path)
    extractor_class = _compile_methods(
        estimator_tree,
        {"_extract_features", "_extract_circuit_features", "_get_feature_names"},
        "PinnedQonductorExtractor",
    )
    loader = _compile_archive_loader(archive_tree)
    parser = _compile_qasm_string_parser(archive_tree)
    metadata = {
        "source_revision": revision,
        "source_root_head": subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "pinned_blobs": {
            relative_path: sha256_bytes(blob)
            for relative_path, blob in pinned_blobs.items()
        },
        "feature_method_names": ["_extract_features", "_extract_circuit_features", "_get_feature_names"],
        "archive_parser_method": "AST-extracted QASM2 from_qasm_str then QASM3 qasm3.loads fallback from pinned loader",
        "matrix_order": ["swap", "depth", "num_qubits", "shots", "circuit_count"],
        "side_effects": "Only pinned AST methods are compiled. Estimator constructor, fitting, model persistence and archive unpacking are not run.",
    }
    return extractor_class, loader, parser, metadata
