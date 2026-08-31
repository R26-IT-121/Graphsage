"""The demo endpoints, exercised against a small stand-in graph.

Loading PaySim to test request handling would cost gigabytes and prove nothing
extra: what can break here is the handler, not the data.
"""

import io
import threading

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch_geometric.data import Data

from graphsage.api.app import app
from graphsage.inference.live import LiveModel
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE


class StubExtractor:
    def __init__(self, names): self.name_to_id = {n: i for i, n in enumerate(names)}


class StubPredictor:
    """Everything the demo handlers touch, and nothing else."""
    def __init__(self):
        torch.manual_seed(0)
        n, e = 30, 90
        self.node_names = np.array([f"C{i:04d}" for i in range(n)])
        data = Data(edge_index=torch.stack([torch.randint(0, n, (e,)),
                                            torch.randint(0, n, (e,))]),
                    edge_attr=torch.rand(e, 6))
        data.num_nodes = n
        self.probs = torch.rand(n)
        self.meta = {"step_range": (1, 743)}
        self.stage = "stage_3b_v2"
        self.extractor = StubExtractor(self.node_names)
        self.risk_bands = {"medium": 0.2, "high": 0.4, "critical": 0.8}

        m = LiveModel.__new__(LiveModel)
        m.device, m.data, m.k, m.max_nodes = torch.device("cpu"), data, 2, 4000
        m._lock = threading.Lock()
        m.x = torch.rand(n, 12)
        m.model = EdgeEnhancedGraphSAGE(in_dim=12, edge_dim=6, hidden_dim=64)
        m.model.eval()
        m.inferences, m.total_ms = 0, 0.0
        self.live = m


@pytest.fixture
def client():
    app.state.predictor = StubPredictor()
    return TestClient(app)


TXN = dict(step=705, type="TRANSFER", amount=404394.04,
           nameOrig="C0003", nameDest="NEWACCOUNT",
           oldbalanceOrg=404394.04, newbalanceOrig=0.0,
           oldbalanceDest=0.0, newbalanceDest=404394.04)


def test_scores_an_account_that_is_not_in_the_graph(client):
    r = client.post("/api/graph/demo/score-account",
                    json={"account": "NEWACCOUNT", "transactions": [TXN]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["in_graph"] is False
    assert 0.0 <= b["raw_score"] <= 1.0
    assert b["calibrated"] is False          # never claims a calibrated number
    assert len(b["features"]) == 12
    assert b["neighbours"][0]["account"] == "C0003"
    assert b["provenance"]["attached_edges"] == 1


def test_the_score_moves_with_the_neighbourhood(client):
    """The demo's whole claim, through the HTTP surface."""
    scores = []
    for neighbour in ("C0003", "C0011", "C0022"):
        r = client.post("/api/graph/demo/score-account", json={
            "account": "NEWACCOUNT",
            "transactions": [{**TXN, "nameOrig": neighbour}]})
        scores.append(r.json()["raw_score"])
    assert len(set(scores)) > 1, f"neighbourhood ignored: {scores}"


def test_rejects_an_account_with_no_known_counterparty(client):
    r = client.post("/api/graph/demo/score-account", json={
        "account": "NEWACCOUNT",
        "transactions": [{**TXN, "nameOrig": "ALSO_UNKNOWN"}]})
    assert r.status_code == 422
    assert r.json()["error"] == "NoKnownNeighbours"


def test_csv_scores_known_and_unknown_destinations(client):
    csv = ("step,type,amount,nameOrig,nameDest,oldbalanceOrg,newbalanceOrig,"
           "oldbalanceDest,newbalanceDest,isFraud\n"
           "705,TRANSFER,404394.04,C0003,C0007,404394.04,0,0,404394.04,1\n"
           "706,TRANSFER,120000.0,C0011,BRANDNEW,120000.0,0,0,120000.0,0\n"
           "707,TRANSFER,5000.0,UNKNOWN_A,UNKNOWN_B,5000.0,0,0,5000.0,0\n")
    r = client.post("/api/graph/demo/score-csv",
                    files={"file": ("d.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["scored_by"] == "graph_only"
    assert [x["source"] for x in b["rows"]] == ["precomputed", "inductive", "unscored"]
    assert b["counts"] == {"precomputed": 1, "inductive": 1, "unscored": 1}
    assert b["rows"][1]["score"] is not None      # the new account still got scored
