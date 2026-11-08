"""The rename-aware code signature accepts pure renames and nothing else."""

import pytest

from benchmark26.codesig import code_signature

RENAMES = {"names": {"run_rows": "panel"}, "strings": {"run_rows": "panel"}}
ORIGINAL = 'panel = [1]\nother = [2]\nprint(panel, context["panel"])\n'


def test_pure_rename_keeps_the_signature():
    renamed = 'run_rows = [1]\nother = [2]\nprint(run_rows, context["run_rows"])\n'
    assert code_signature(renamed, RENAMES) == code_signature(ORIGINAL)


def test_changed_reference_is_detected():
    changed = 'run_rows = [1]\nother = [2]\nprint(other, context["run_rows"])\n'
    assert code_signature(changed, RENAMES) != code_signature(ORIGINAL)


def test_surviving_old_name_is_rejected():
    with pytest.raises(ValueError):
        code_signature('run_rows = [1]\npanel = [2]\nprint(panel, context["run_rows"])\n', RENAMES)
