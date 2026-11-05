"""Controls change only the intended graph inputs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from experiments.graph_ablation.transforms import transform_graph


def fixture():
    return Data(
        x=torch.arange(6 * 178, dtype=torch.float32).reshape(6, 178),
        edge_index=torch.tensor([[0, 1, 2, 4], [1, 2, 3, 5]]),
        global_features=torch.ones(1, 4),
    )


def test_empty_and_position_mask_do_not_mutate_input():
    g = fixture()
    x = g.x.clone()
    edges = g.edge_index.clone()
    empty = transform_graph(g, "G1e", "id")
    masked = transform_graph(g, "G1e-noidx", "id")
    assert empty.edge_index.shape == (2, 0) and torch.equal(empty.x, x)
    assert torch.equal(masked.x[:, :177], x[:, :177]) and not masked.x[:, 177].any()
    assert torch.equal(g.x, x) and torch.equal(g.edge_index, edges)


def test_shuffle_preserves_rows_and_fixed_edges_deterministically():
    g = fixture()
    a = transform_graph(g, "G1p", "id")
    b = transform_graph(g, "G1p", "id")
    assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, g.edge_index)
    assert sorted(map(tuple, a.x.tolist())) == sorted(map(tuple, g.x.tolist()))
    assert not torch.equal(a.x, g.x)
