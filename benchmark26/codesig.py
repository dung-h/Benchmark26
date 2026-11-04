"""A signature of what a Python file executes, ignoring docstrings, comments and layout.

Two versions of a file have the same signature exactly when they have the same
syntax tree after docstrings are removed. The signature does not change with the
Python version (3.10 to 3.14 were compared) because it serializes only the node
types and fields that exist in all of them.
"""

from __future__ import annotations

import ast
import hashlib
import json

_SKIPPED_FIELDS = {"type_params", "kind", "type_comment"}
_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _serialize(node):
    if isinstance(node, ast.AST):
        fields = [[name, _serialize(getattr(node, name, None))] for name in node._fields if name not in _SKIPPED_FIELDS]
        return [type(node).__name__, *fields]
    if isinstance(node, list):
        return [_serialize(item) for item in node]
    return repr(node)


def _undo_renames(tree, names: dict[str, str], strings: dict[str, str]) -> None:
    """Map renamed identifiers and whole string literals back to their old spelling, in place.

    The old spellings must no longer occur, otherwise a new name and a surviving old one would
    both map to the old one and a changed reference could pass as a rename.
    """
    old_names, old_strings = set(names.values()), set(strings.values())
    for node in ast.walk(tree):
        for field in ("id", "attr", "arg", "name", "asname"):
            if getattr(node, field, None) in old_names:
                raise ValueError(f"old name still present: {getattr(node, field)}")
        if isinstance(node, (ast.Global, ast.Nonlocal)) and old_names.intersection(node.names):
            raise ValueError("old name still present in a global or nonlocal statement")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in old_strings:
            raise ValueError(f"old string still present: {node.value!r}")
    for node in ast.walk(tree):
        for field in ("id", "attr", "arg", "name", "asname"):
            value = getattr(node, field, None)
            if isinstance(value, str) and value in names:
                setattr(node, field, names[value])
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            node.names = [names.get(name, name) for name in node.names]
        if isinstance(node, ast.ImportFrom) and node.module:
            node.module = ".".join(names.get(part, part) for part in node.module.split("."))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in strings:
            node.value = strings[node.value]


def code_signature(source: str, renames: dict | None = None) -> str:
    """Return the SHA-256 hex digest of the syntax tree of `source`, without docstrings.

    `renames` may give `names` and `strings`, each mapping a new spelling to the old one. They
    are applied before hashing, so a file whose only change is those renames keeps its signature.
    """
    tree = ast.parse(source)
    if renames:
        _undo_renames(tree, renames.get("names", {}), renames.get("strings", {}))
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS) and node.body:
            first = node.body[0]
            is_docstring = (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
            )
            if is_docstring:
                node.body = node.body[1:] or [ast.Pass()]
    return hashlib.sha256(json.dumps(_serialize(tree), separators=(",", ":")).encode()).hexdigest()
