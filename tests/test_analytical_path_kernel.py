"""Tests for analytical critical path DAG recurrence logic."""

import numpy as np


def python_paths_reference(num_qubits, arity, q0, q1, measure, durations):
    """Pure-Python reference of the C++ path_kernel.cpp recurrence."""
    full = np.zeros(num_qubits, dtype=np.float64)
    unitary = np.zeros(num_qubits, dtype=np.float64)
    level = np.zeros(num_qubits, dtype=np.int32)

    cp_full = 0.0
    cp_unitary = 0.0
    depth = 0

    for i in range(len(arity)):
        a = arity[i]
        x = q0[i]
        y = q1[i]
        dur = durations[i]

        f = (max(full[x], full[y]) if a == 2 else full[x]) + dur
        full[x] = f
        if a == 2:
            full[y] = f
        cp_full = max(cp_full, f)

        if measure[i]:
            continue

        l = (max(level[x], level[y]) if a == 2 else level[x]) + 1
        g = (max(unitary[x], unitary[y]) if a == 2 else unitary[x]) + dur

        level[x] = l
        unitary[x] = g
        if a == 2:
            level[y] = l
            unitary[y] = g

        depth = max(depth, l)
        cp_unitary = max(cp_unitary, g)

    return {
        "depth": depth,
        "cp_full": cp_full,
        "cp_unitary": cp_unitary,
    }


def test_serial_single_qubit_gates():
    # 3 single qubit gates on wire 0: durations 10, 20, 30
    arity = np.array([1, 1, 1], dtype=np.uint8)
    q0 = np.array([0, 0, 0], dtype=np.int32)
    q1 = np.array([-1, -1, -1], dtype=np.int32)
    measure = np.array([0, 0, 0], dtype=np.uint8)
    durations = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    res = python_paths_reference(2, arity, q0, q1, measure, durations)
    assert res["depth"] == 3
    assert res["cp_unitary"] == 60.0
    assert res["cp_full"] == 60.0


def test_parallel_independent_wires():
    # Gate on wire 0 (dur 10), Gate on wire 1 (dur 25)
    arity = np.array([1, 1], dtype=np.uint8)
    q0 = np.array([0, 1], dtype=np.int32)
    q1 = np.array([-1, -1], dtype=np.int32)
    measure = np.array([0, 0], dtype=np.uint8)
    durations = np.array([10.0, 25.0], dtype=np.float64)

    res = python_paths_reference(2, arity, q0, q1, measure, durations)
    assert res["depth"] == 1
    assert res["cp_unitary"] == 25.0  # max of the two wires


def test_two_qubit_synchronization():
    # Gate on wire 0 (10), Gate on wire 1 (20), then CX(0, 1) (50)
    arity = np.array([1, 1, 2], dtype=np.uint8)
    q0 = np.array([0, 1, 0], dtype=np.int32)
    q1 = np.array([-1, -1, 1], dtype=np.int32)
    measure = np.array([0, 0, 0], dtype=np.uint8)
    durations = np.array([10.0, 20.0, 50.0], dtype=np.float64)

    res = python_paths_reference(2, arity, q0, q1, measure, durations)
    assert res["depth"] == 2
    # Wire 1 had 20, wire 0 had 10. CX starts at max(10, 20) = 20, then + 50 = 70.
    assert res["cp_unitary"] == 70.0
