"""API contract tests for /api/graph/analyze using a synthetic graph.

Injects a lightweight predictor built on the test_subgraph ring fixture so the
suite never loads the 204 MB production graph.
"""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from graphsage.api.app import create_app
from graphsage.extraction.subgraph import SuspiciousSubgraphExtractor
from tests.test_subgraph import ATTN, NAMES, PROBS, build_ring


class FakePredictor:
    """Duck-typed stand-in for GraphPredictor over the synthetic ring."""

    def __init__(self):
        self.data = build_ring()
        self.threshold = 0.5
        self.graph_version = "synthetic_test_ring"
        self.probs = PROBS
        self.edge_attention = ATTN
        self.extractor = SuspiciousSubgraphExtractor(
            self.data, NAMES, risk_threshold=0.5
        )

    def is_applicable(self, txn_type: str) -> bool:
        return txn_type in {"TRANSFER", "CASH_OUT"}

    def analyze(self, name_orig, name_dest, step):
        trigger = self.extractor.find_trigger_edge(name_orig, name_dest, step)
        if trigger is None:
            return None
        dst_id = self.extractor.name_to_id[name_dest]
        score = float(self.probs[dst_id])
        return {
            "relational_risk_score": round(score, 4),
            "confidence": round(max(score, 1.0 - score), 4),
            "suspicious_subgraph": self.extractor.extract(
                trigger, self.probs, self.edge_attention
            ),
        }


def make_request(**overrides) -> dict:
    base = {
        "transaction_id": "TX_TEST_001",
        "step": 100,
        "type": "TRANSFER",
        "amount": 9800.0,
        "nameOrig": "C_A",
        "nameDest": "C_M",
        "oldbalanceOrg": 9800.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 9800.0,
        "isFlaggedFraud": 0,
    }
    base.update(overrides)
    return base


client = TestClient(create_app(predictor=FakePredictor()))


def test_analyze_happy_path():
    res = client.post("/api/graph/analyze", json=make_request())
    assert res.status_code == 200
    body = res.json()
    assert body["transaction_id"] == "TX_TEST_001"
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    sg = body["suspicious_subgraph"]
    assert sg["sink_account"] == "C_M"
    assert sg["pattern"] == "SMURFING"
    assert any(e["is_trigger_edge"] for e in sg["edges"])
    assert body["metadata"]["inference_latency_ms"] < 500


def test_not_applicable_type():
    res = client.post("/api/graph/analyze", json=make_request(type="PAYMENT"))
    assert res.status_code == 200
    body = res.json()
    assert body["risk_level"] == "NOT_APPLICABLE"
    assert body["relational_risk_score"] == 0.0
    assert body["suspicious_subgraph"] is None


def test_unknown_pair_is_404_contract_error():
    res = client.post(
        "/api/graph/analyze", json=make_request(nameOrig="C_NOPE", nameDest="C_M")
    )
    assert res.status_code == 404
    body = res.json()
    assert body["error"] == "NotFound"
    assert body["transaction_id"] == "TX_TEST_001"


def test_validation_negative_amount_is_422():
    res = client.post("/api/graph/analyze", json=make_request(amount=-5.0))
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "BadRequest"
    assert "amount" in body["message"]


def test_validation_bad_type_is_422():
    res = client.post("/api/graph/analyze", json=make_request(type="WIRE"))
    assert res.status_code == 422


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["graph_version"] == "synthetic_test_ring"
