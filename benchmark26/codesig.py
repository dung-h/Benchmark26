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


def code_signature(source: str) -> str:
    """Return the SHA-256 hex digest of the syntax tree of `source`, without docstrings."""
    tree = ast.parse(source)
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
