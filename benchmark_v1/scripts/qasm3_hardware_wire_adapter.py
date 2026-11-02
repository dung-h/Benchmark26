"""Restore saved compiled OpenQASM 3 hardware-wire identifiers by pinned width.

This helper is only for saved Ma-Li/QPack compiled QASM3. Submitted Qonductor
archive circuits must keep the pinned upstream QASM2/QASM3 loader semantics.
The returned source preserves the AST except for replacing each hardware
identifier ``$N`` with a declared register element at the same physical index.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import re
from typing import Any


_HARDWARE_ID = re.compile(r"\$([0-9]+)\Z")
_DEFAULT_REGISTER = "__benchmark_physical"


@dataclass(frozen=True)
class RestorationMetadata:
    api: str
    register_name: str | None
    allocated_width: int
    hardware_indices_in_source_order: tuple[int, ...]
    replacement_count: int
    normalized_ast_sha256: str
    restored_qasm_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "register_name": self.register_name,
            "allocated_width": self.allocated_width,
            "hardware_indices_in_source_order": list(self.hardware_indices_in_source_order),
            "replacement_count": self.replacement_count,
            "normalized_ast_sha256": self.normalized_ast_sha256,
            "restored_qasm_sha256": self.restored_qasm_sha256,
        }


def restore_saved_qasm3_hardware_wires(qasm_text: str, register_width: int) -> tuple[str, dict[str, Any]]:
    """Return AST-restored QASM3 plus a stable receipt.

    Args:
        qasm_text: Original, hash-verified saved QASM3 text.
        register_width: Allocated width pinned in representation_rows.

    Returns:
        ``(restored_qasm_text, metadata_dict)``. The helper does not verify the
        input file digest or the method-specific graph/feature signature; the
        caller must perform those gates before accepting extracted features.

    Raises:
        ValueError: On malformed AST, out-of-range hardware index, unexpected
            rewrite/round-trip changes, or failure to preserve identifier order.
    """
    if not isinstance(qasm_text, str):
        raise TypeError("qasm_text must be str")
    if isinstance(register_width, bool) or not isinstance(register_width, int) or register_width < 0:
        raise ValueError("register_width must be a non-negative integer")

    try:
        import openqasm3
        from openqasm3 import ast as qasm_ast
    except ImportError as exc:  # pragma: no cover - environment-specific diagnostic
        raise RuntimeError("openqasm3 is required for AST hardware-wire restoration") from exc

    try:
        tree = openqasm3.parse(qasm_text)
    except Exception as exc:
        raise ValueError(f"cannot parse saved QASM3 AST: {type(exc).__name__}: {exc}") from exc

    def walk(node: Any):
        if isinstance(node, list):
            for child in node:
                yield from walk(child)
        elif isinstance(node, tuple):
            for child in node:
                yield from walk(child)
        elif is_dataclass(node):
            yield node
            for field in fields(node):
                if field.name != "span":
                    yield from walk(getattr(node, field.name))

    identifiers = [node for node in walk(tree) if isinstance(node, qasm_ast.Identifier)]
    names = {node.name for node in identifiers}
    hardware_indices: list[int] = []
    for identifier in identifiers:
        match = _HARDWARE_ID.fullmatch(identifier.name)
        if match:
            physical_index = int(match.group(1))
            if physical_index >= register_width:
                raise ValueError(
                    f"hardware index {physical_index} outside pinned width {register_width}"
                )
            hardware_indices.append(physical_index)

    if not hardware_indices:
        # Named-register QASM is already explicit; leave it byte-identical and
        # let the caller validate its allocated width and semantic signature.
        digest = hashlib.sha256(qasm_text.encode("utf-8")).hexdigest()
        metadata = RestorationMetadata(
            api="restore_saved_qasm3_hardware_wires_v1",
            register_name=None,
            allocated_width=register_width,
            hardware_indices_in_source_order=(),
            replacement_count=0,
            normalized_ast_sha256=_stream_ast_hash(tree),
            restored_qasm_sha256=digest,
        )
        return qasm_text, metadata.as_dict()

    register_name = _DEFAULT_REGISTER
    suffix = 0
    while register_name in names:
        suffix += 1
        register_name = f"{_DEFAULT_REGISTER}_{suffix}"

    def replace_hardware_identifiers(node: Any) -> Any:
        if isinstance(node, qasm_ast.Identifier):
            match = _HARDWARE_ID.fullmatch(node.name)
            if match:
                index = int(match.group(1))
                return qasm_ast.IndexedIdentifier(
                    name=qasm_ast.Identifier(name=register_name),
                    indices=[[qasm_ast.IntegerLiteral(value=index)]],
                )
            return node
        if isinstance(node, list):
            return [replace_hardware_identifiers(child) for child in node]
        if isinstance(node, tuple):
            return tuple(replace_hardware_identifiers(child) for child in node)
        if is_dataclass(node):
            for field in fields(node):
                if field.name != "span":
                    setattr(node, field.name, replace_hardware_identifiers(getattr(node, field.name)))
        return node

    tree = replace_hardware_identifiers(tree)
    declaration = qasm_ast.QubitDeclaration(
        qubit=qasm_ast.Identifier(name=register_name),
        size=qasm_ast.IntegerLiteral(value=register_width),
    )
    statements = tree.statements
    insertion = 0
    while insertion < len(statements) and type(statements[insertion]).__name__ in {"Version", "Include"}:
        insertion += 1
    statements.insert(insertion, declaration)

    try:
        restored_text = openqasm3.dumps(tree)
        reparsed = openqasm3.parse(restored_text)
    except Exception as exc:
        raise ValueError(f"restored QASM3 failed serialization/parse: {type(exc).__name__}: {exc}") from exc

    restored_signature, expected_indices, restored_indices = _compare_and_hash_ast(
        tree, reparsed, qasm_ast, register_name, register_width
    )
    if expected_indices != hardware_indices or restored_indices != hardware_indices:
        raise ValueError(
            "restored hardware identifier occurrence order changed: "
            f"source {hardware_indices}, transformed {expected_indices}, serialized {restored_indices}"
        )

    metadata = RestorationMetadata(
        api="restore_saved_qasm3_hardware_wires_v1",
        register_name=register_name,
        allocated_width=register_width,
        hardware_indices_in_source_order=tuple(hardware_indices),
        replacement_count=len(hardware_indices),
        normalized_ast_sha256=restored_signature,
        restored_qasm_sha256=hashlib.sha256(restored_text.encode("utf-8")).hexdigest(),
    )
    return restored_text, metadata.as_dict()


def _feed_ast_hash(hasher: Any, node: Any) -> None:
    """Hash AST incrementally without materializing a huge nested JSON object."""
    if isinstance(node, list):
        hasher.update(b"list[")
        hasher.update(str(len(node)).encode("ascii"))
        hasher.update(b"]")
        for child in node:
            _feed_ast_hash(hasher, child)
        hasher.update(b"/list")
    elif isinstance(node, tuple):
        hasher.update(b"tuple[")
        hasher.update(str(len(node)).encode("ascii"))
        hasher.update(b"]")
        for child in node:
            _feed_ast_hash(hasher, child)
        hasher.update(b"/tuple")
    elif is_dataclass(node):
        hasher.update(type(node).__name__.encode("utf-8") + b"{")
        for field in fields(node):
            if field.name == "span":
                continue
            hasher.update(field.name.encode("utf-8") + b"=")
            _feed_ast_hash(hasher, getattr(node, field.name))
        hasher.update(b"}")
    else:
        hasher.update(type(node).__name__.encode("utf-8") + b":" + repr(node).encode("utf-8") + b";")


def _stream_ast_hash(tree: Any) -> str:
    hasher = hashlib.sha256()
    _feed_ast_hash(hasher, tree)
    return hasher.hexdigest()


def _compare_and_hash_ast(
    expected: Any, observed: Any, qasm_ast: Any, register_name: str, register_width: int
) -> tuple[str, list[int], list[int]]:
    """Compare ASTs field-by-field while streaming a signature and wire order."""
    hasher = hashlib.sha256()
    expected_indices: list[int] = []
    observed_indices: list[int] = []

    def compare(left: Any, right: Any, location: str) -> None:
        if type(left) is not type(right):
            raise ValueError(f"AST node type changed at {location}: {type(left).__name__} != {type(right).__name__}")
        if isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(f"AST list length changed at {location}: {len(left)} != {len(right)}")
            hasher.update(b"list[" + str(len(left)).encode("ascii") + b"]")
            for index, (left_child, right_child) in enumerate(zip(left, right, strict=True)):
                compare(left_child, right_child, f"{location}[{index}]")
            hasher.update(b"/list")
            return
        if isinstance(left, tuple):
            if len(left) != len(right):
                raise ValueError(f"AST tuple length changed at {location}")
            hasher.update(b"tuple[" + str(len(left)).encode("ascii") + b"]")
            for index, (left_child, right_child) in enumerate(zip(left, right, strict=True)):
                compare(left_child, right_child, f"{location}[{index}]")
            hasher.update(b"/tuple")
            return
        if is_dataclass(left):
            hasher.update(type(left).__name__.encode("utf-8") + b"{")
            for field in fields(left):
                if field.name == "span":
                    continue
                hasher.update(field.name.encode("utf-8") + b"=")
                compare(getattr(left, field.name), getattr(right, field.name), f"{location}.{field.name}")
            hasher.update(b"}")
            if (
                isinstance(left, qasm_ast.IndexedIdentifier)
                and isinstance(left.name, qasm_ast.Identifier)
                and left.name.name == register_name
            ):
                if len(left.indices) != 1 or len(left.indices[0]) != 1:
                    raise ValueError(f"restored hardware operand has an invalid shape at {location}")
                literal_left = left.indices[0][0]
                literal_right = right.indices[0][0]
                if not isinstance(literal_left, qasm_ast.IntegerLiteral) or not isinstance(literal_right, qasm_ast.IntegerLiteral):
                    raise ValueError(f"restored hardware index is not an integer literal at {location}")
                left_index, right_index = int(literal_left.value), int(literal_right.value)
                if not 0 <= left_index < register_width:
                    raise ValueError(f"restored hardware index {left_index} outside width {register_width}")
                expected_indices.append(left_index)
                observed_indices.append(right_index)
            return
        if left != right:
            raise ValueError(f"AST value changed at {location}: {left!r} != {right!r}")
        hasher.update(type(left).__name__.encode("utf-8") + b":" + repr(left).encode("utf-8") + b";")

    compare(expected, observed, "program")
    return hasher.hexdigest(), expected_indices, observed_indices
