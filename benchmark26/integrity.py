"""Check that the frozen protocol files still match the hashes that pin them.

The study protocols pin each other, the reference registries and two reference
programs by SHA-256. A runner that verifies a contract against the original
study bundle fails if any of these files changes, so they are kept
byte-identical in ``protocols/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from benchmark26.paths import PROTOCOLS, recorded


@dataclass(frozen=True)
class Pin:
    source: str
    path: str
    sha256: str

    @property
    def local(self) -> Path:
        """Return the location of the recorded path on this machine."""
        return recorded(self.path)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((PROTOCOLS / name).read_text(encoding="utf-8"))


def frozen_pins() -> list[Pin]:
    """Pins between files that this repository distributes."""
    execution = _load("estimator_execution.json")
    pins = [
        Pin("estimator_execution.json", spec["path"], spec["sha256"])
        for key, spec in execution["input_pins"].items()
        if key in {"seed_registry", "current_guard", "fidelity_guard"}
    ]
    pins += [
        Pin(f"estimator_execution.json:method_contracts.{name}", spec["path"], spec["sha256"])
        for name, spec in execution["method_contracts"].items()
    ]
    qonductor = _load("qonductor_native_features.json")
    reference = qonductor["input_pins"]["materializer_reference"]
    pins.append(Pin("qonductor_native_features.json:materializer_reference", reference["path"], reference["sha256"]))
    return pins


def check_frozen_pins() -> list[tuple[Pin, str]]:
    """Return ``(pin, status)`` with status ``match``, ``differs`` or ``missing``."""
    results = []
    for pin in frozen_pins():
        path = pin.local
        if not path.is_file():
            status = "missing"
        else:
            status = "match" if sha256_file(path) == pin.sha256 else "differs"
        results.append((pin, status))
    return results
