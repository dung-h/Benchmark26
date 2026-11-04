"""Read the CSV tables of the study bundle."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into a list of dicts, one per row."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
