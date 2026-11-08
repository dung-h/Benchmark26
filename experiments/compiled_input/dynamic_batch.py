"""Order-preserving, resource-bounded graph microbatches and atomic gradient steps."""
from __future__ import annotations
import random
import gc
import numpy as np


def partition(items, sizes, node_budget, edge_budget):
    """Yield ordered groups; oversized individual graphs remain intact and are tried alone."""
    if node_budget < 1 or edge_budget < 1:
        raise ValueError('budgets must be positive')
    group, nodes, edges = [], 0, 0
    for item in items:
        n, e = sizes[item]
        if n < 0 or e < 0:
            raise ValueError('negative graph size')
        if group and (nodes + n > node_budget or edges + e > edge_budget):
            yield group
            group, nodes, edges = [], 0, 0
        group.append(item)
        nodes += n
        edges += e
        if n > node_budget or e > edge_budget:
            yield group
            group, nodes, edges = [], 0, 0
    if group:
        yield group


def train_step(model, optimizer, items, sizes, node_budget, edge_budget, forward_sum, torch):
    """Run one effective batch. On CUDA OOM, retry the entire batch with smaller groups.

    forward_sum(group) returns the sum of per-observation losses. This function
    never steps the optimizer before all microbatches complete. The original RNG
    state is restored before each retry. A single-graph OOM is propagated.
    """
    if not items:
        raise ValueError('empty effective batch')
    groups = list(partition(items, sizes, node_budget, edge_budget))
    py_state = random.getstate()
    np_state = np.random.get_state()
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    attempts = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        try:
            total = 0.0
            for group in groups:
                loss_sum = forward_sum(group)
                loss = loss_sum / len(items)
                loss.backward()
                total += float(loss.detach().cpu())
                del loss_sum, loss
            optimizer.step()
            return {'loss': total, 'retries': attempts, 'microbatch_sizes': [len(g) for g in groups]}
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_sum = None
            gc.collect()
            torch.cuda.empty_cache()
            if all(len(g) == 1 for g in groups):
                raise
            groups = [part for g in groups for part in ((g[:len(g)//2], g[len(g)//2:]) if len(g) > 1 else (g,)) if part]
            attempts += 1
