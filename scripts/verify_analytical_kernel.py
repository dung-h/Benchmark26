"""Compile the archived scalar kernel and compare with explicit-edge recurrence.

Requires NumPy and g++; no Qiskit, unpickling, QPU/simulator execution or labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from benchmark26.paths import ARTIFACTS, recorded

R = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
fields = [
    "depth_1q",
    "depth_2q",
    "unitary_depth",
    "paper_kappa1",
    "paper_kappa2",
    "notads_code",
    "notads_paper",
    "hyperq_full_seconds",
    "hyperq_unitary_seconds",
    "measurement_count",
]
manifest = json.loads((ARTIFACTS / "analytical_kernel_validation/manifest.json").read_text())
kernel = R / "benchmark26/path_kernel.cpp"
with tempfile.TemporaryDirectory(prefix="qre-kernel-") as td:
    lib = Path(td) / "kernel.so"
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(kernel), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    fn = dll.paths
    ptr = np.ctypeslib.ndpointer
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ptr(np.uint8, flags="C_CONTIGUOUS"),
        ptr(np.int32, flags="C_CONTIGUOUS"),
        ptr(np.int32, flags="C_CONTIGUOUS"),
        ptr(np.uint8, flags="C_CONTIGUOUS"),
        ptr(np.float64, flags="C_CONTIGUOUS"),
        ptr(np.float64, flags="C_CONTIGUOUS"),
    ]
    fn.restype = ctypes.c_int
    ops = 0
    checks = 0
    for entry in manifest["fixtures"]:
        path = recorded(entry["fixture"])
        assert sha(path) == entry["fixture_sha256"]
        with np.load(path, allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
        n = len(z["arity"])
        ops += n
        got = np.empty(10, dtype=np.float64)
        assert fn(n, int(z["num_qubits"]), z["arity"], z["q0"], z["q1"], z["measure"], z["durations"], got) == 0
        predecessors = [[] for _ in range(n)]
        for s, t in zip(z["edge_src"], z["edge_dst"]):
            assert 0 <= s < t < n
            predecessors[t].append(s)
        val = np.zeros((n, 5))
        has_two = set()
        for i in range(n):
            start = val[predecessors[i]].max(axis=0) if predecessors[i] else np.zeros(5)
            unit = not z["measure"][i]
            a = z["arity"][i]
            d = z["durations"][i]
            val[i] = start + [unit and a == 1, unit and a == 2, unit, d, d if unit else 0]
            if unit and a == 2:
                has_two.add(val[i, 2])
        d1, d2, depth, full, unit = val.max(axis=0)
        expected = [
            d1,
            d2,
            depth,
            depth - len(has_two),
            len(has_two),
            d1 + 10 * d2,
            depth + 9 * len(has_two),
            full,
            unit,
            z["measure"].sum(),
        ]
        for k, a, b in zip(fields, got, expected):
            assert np.isclose(a, b, rtol=1e-10, atol=1e-12), (entry["snapshot_id"], k, a, b)
            assert np.isclose(a, entry["expected"][k], rtol=1e-10, atol=1e-12), (
                entry["snapshot_id"],
                k,
                a,
                entry["expected"][k],
            )
            checks += 2
print(
    json.dumps(
        {
            "status": "PASS_ARCHIVED_KERNEL_FIXTURES",
            "fixtures": len(manifest["fixtures"]),
            "operations": ops,
            "scalar_checks": checks,
            "kernel_sha256": sha(kernel),
            "scope": "Archived graph edges and nominal durations; sample replay, not full 1438-task extraction or historical calibration validation",
            "training_or_quantum_execution": False,
        },
        indent=2,
    )
)
