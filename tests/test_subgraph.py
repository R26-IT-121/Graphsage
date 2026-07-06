"""Unit tests for the Suspicious Subgraph extractor (Novelty 3).

Builds a tiny synthetic hub-and-spoke mule ring:

    A, B, C (fresh senders) --> M (mule sink) --> X (cash-out relay target)
    L1 --> L2 (legitimate pair, 2 hops away via nothing — disconnected)

and checks extraction, sink identification, roles, pattern, and the
contract-required JSON shape.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from graphsage.extraction.subgraph import SuspiciousSubgraphExtractor

# node ids:      0    1    2    3    4    5     6
NAMES = ["C_A", "C_B", "C_C", "C_M", "C_X", "C_L1", "C_L2"]


def _amount_log(amount: float) -> float:
    return float(torch.log1p(torch.tensor(amount)))


def build_ring() -> Data:
    # edges: A->M, B->M, C->M (similar amounts), M->X, L1->L2
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 5], [3, 3, 3, 4, 6]], dtype=torch.int64
    )
    #                 amount_log            drain src_dr dst_empty gap transfer
    edge_attr = torch.tensor(
        [
            [_amount_log(9800.0), 1.0, 1.0, 1.0, -1.0, 1.0],
            [_amount_log(9900.0), 1.0, 1.0, 0.0, 2.0, 1.0],
            [_amount_log(10100.0), 1.0, 1.0, 0.0, 1.0, 1.0],
            [_amount_log(29800.0), 1.0, 1.0, 0.0, 3.0, 0.0],
            [_amount_log(50.0), 0.1, 0.0, 0.0, 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # x: [in_degree, out_degree, mean_in, mean_out, max_in] (global)
    x = torch.zeros(7, 5)
    x[:, 1] = torch.tensor([1, 1, 1, 1, 0, 1, 0], dtype=torch.float32)  # out_deg
    x[:, 0] = torch.tensor([0, 0, 0, 3, 1, 0, 1], dtype=torch.float32)  # in_deg
    y = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.int8)
    edge_step = torch.tensor([100, 102, 103, 105, 200], dtype=torch.int16)
    edge_isFraud = torch.tensor([1, 1, 1, 0, 0], dtype=torch.int8)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_step=edge_step,
        edge_isFraud=edge_isFraud,
    )


def make_extractor(**kwargs) -> SuspiciousSubgraphExtractor:
    return SuspiciousSubgraphExtractor(build_ring(), NAMES, **kwargs)


PROBS = torch.tensor([0.2, 0.15, 0.22, 0.97, 0.55, 0.01, 0.02])
ATTN = torch.tensor([0.9, 0.88, 0.91, 0.85, 0.05])


def test_find_trigger_edge():
    ex = make_extractor()
    assert ex.find_trigger_edge("C_A", "C_M", 100) == 0
    # wrong step falls back to the pair match
    assert ex.find_trigger_edge("C_A", "C_M", 999) == 0
    assert ex.find_trigger_edge("C_A", "C_B") is None
    assert ex.find_trigger_edge("C_UNKNOWN", "C_M") is None


def test_extract_ring_shape_and_sink():
    ex = make_extractor(risk_threshold=0.5)
    out = ex.extract(0, PROBS, ATTN)

    # Contract top-level keys (§3.2)
    for key in (
        "k_hop", "node_count", "edge_count", "nodes", "edges",
        "sink_account", "pattern", "pattern_confidence", "structural_evidence",
    ):
        assert key in out, f"missing contract key {key}"

    assert out["k_hop"] == 2
    # 2-hop ball around A and M reaches A,B,C,M,X but not L1/L2
    got_accounts = {n["account_id"] for n in out["nodes"]}
    assert got_accounts == {"C_A", "C_B", "C_C", "C_M", "C_X"}
    assert out["node_count"] == 5
    assert out["edge_count"] == 4

    assert out["sink_account"] == "C_M"
    roles = {n["account_id"]: n["role"] for n in out["nodes"]}
    assert roles["C_M"] == "MULE_CENTRAL"
    assert roles["C_A"] == "FRESH_SENDER"
    assert roles["C_B"] == "FRESH_SENDER"


def test_pattern_smurfing_on_structured_amounts():
    ex = make_extractor(risk_threshold=0.5)
    out = ex.extract(0, PROBS, ATTN)
    # 3 senders converging with near-identical amounts -> SMURFING
    assert out["pattern"] == "SMURFING"
    assert out["structural_evidence"]["convergence_count"] == 3
    assert out["structural_evidence"]["mules_in_subgraph"] == 2  # M and X
    assert 0.0 <= out["pattern_confidence"] <= 1.0


def test_trigger_edge_marked():
    ex = make_extractor()
    out = ex.extract(1, PROBS, ATTN)
    trigger_edges = [e for e in out["edges"] if e["is_trigger_edge"]]
    assert len(trigger_edges) == 1
    assert trigger_edges[0]["src"] == "C_B"
    assert trigger_edges[0]["dst"] == "C_M"
    # amount round-trips through log1p/expm1
    assert abs(trigger_edges[0]["amount"] - 9900.0) < 1.0
    assert trigger_edges[0]["edge_attention_weight"] == 0.88


def test_max_edges_cap_keeps_trigger():
    ex = make_extractor(max_edges=2)
    out = ex.extract(4, PROBS, ATTN)  # trigger = lowest-attention edge L1->L2
    assert out["edge_count"] <= 2
    assert any(e["is_trigger_edge"] for e in out["edges"])


def test_json_serializable():
    import json

    ex = make_extractor()
    out = ex.extract(0, PROBS, ATTN)
    json.dumps(out)  # raises if any numpy/tensor types leak through
