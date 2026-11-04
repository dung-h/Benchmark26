"""Graph controls for the circuit-runtime ablation.

The permutation changes node features relative to fixed edges; it is not a
permutation of the graph's node identifiers. Inputs are never mutated.
"""

from __future__ import annotations
import hashlib
import numpy as np
import torch

ARMS = ("M1r", "G1r", "G1e", "G1p", "G1e-noidx")
DOMAIN = "graph-ablation-v1"


def transform_graph(data, arm: str, observation_id: str):
    """Return a copy of the example with the graph control of an arm applied.

    `G1e` and `G1e-noidx` remove all edges (the second also zeroes the position column), `G1p` permutes the
    node feature rows deterministically for the observation, and `M1r` and `G1r` are unchanged.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    result = data.clone()
    if arm in ("M1r", "G1r"):
        return result
    if not hasattr(result, "x") or result.x is None or result.edge_index is None:
        raise ValueError("graph features and edge_index are required")
    if result.x.ndim != 2 or result.x.shape[1] != 178 or result.edge_index.shape[0] != 2:
        raise ValueError("expected node width 178 and edge_index shape (2, E)")
    if arm in ("G1e", "G1e-noidx"):
        result.edge_index = result.edge_index.new_empty((2, 0))
        if arm == "G1e-noidx":
            result.x[:, 177] = 0
    else:
        seed = int.from_bytes(hashlib.sha256((observation_id + DOMAIN).encode()).digest()[:8], "big")
        permutation = np.random.default_rng(seed).permutation(result.x.shape[0])
        result.x = result.x[torch.as_tensor(permutation, dtype=torch.long, device=result.x.device)]
    return result
