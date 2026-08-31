"""A node the network has never seen gets a real embedding from its neighbours.

Built on a synthetic graph rather than PaySim so the test is cheap, but the
model is the real EdgeEnhancedGraphSAGE and the code path is the serving one.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphsage.inference.live import LiveModel
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE


def _live(seed: int = 0) -> LiveModel:
    """A LiveModel over a 40-node graph, bypassing checkpoint loading."""
    torch.manual_seed(seed)
    n, e = 40, 120
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    data = Data(edge_index=torch.stack([src, dst]),
                edge_attr=torch.rand(e, 6))
    data.num_nodes = n

    m = LiveModel.__new__(LiveModel)
    m.device, m.data, m.k, m.max_nodes = torch.device("cpu"), data, 2, 4000
    import threading
    m._lock = threading.Lock()
    m.x = torch.rand(n, 12)
    m.model = EdgeEnhancedGraphSAGE(in_dim=12, edge_dim=6, hidden_dim=64)
    m.model.eval()
    m.inferences, m.total_ms = 0, 0.0
    return m


def test_scores_a_node_that_does_not_exist():
    m = _live()
    feats = np.random.rand(12).astype(np.float32)
    p, ms, prov = m.score_new_node(feats, [], [(3, np.random.rand(6))])
    assert 0.0 <= p <= 1.0
    assert prov["attached_edges"] == 1
    assert prov["neighbourhood_accounts"] > 0
    assert ms >= 0


def test_score_depends_on_which_neighbours_it_attaches_to():
    """The inductive claim: identical node, different company, different score."""
    m = _live()
    feats = np.random.rand(12).astype(np.float32)
    attr = np.random.rand(6).astype(np.float32)

    scores = {n: m.score_new_node(feats, [], [(n, attr)])[0] for n in (3, 11, 27)}
    assert len({round(s, 6) for s in scores.values()}) > 1, (
        f"same score from different neighbourhoods: {scores} — the embedding "
        "is not actually depending on the neighbours"
    )


def test_the_served_graph_is_not_mutated():
    m = _live()
    before_e = m.data.edge_index.clone()
    before_x = m.x.clone()
    m.score_new_node(np.random.rand(12), [(5, np.random.rand(6))], [])
    assert torch.equal(m.data.edge_index, before_e)
    assert torch.equal(m.x, before_x)
    assert m.data.num_nodes == 40


def test_refuses_a_node_with_no_edges():
    m = _live()
    with pytest.raises(ValueError, match="at least one edge"):
        m.score_new_node(np.random.rand(12), [], [])
