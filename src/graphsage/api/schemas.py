"""Pydantic request/response schemas for the /api/graph/analyze endpoint.

This module defines the JSON contract consumed by Member 4's fusion engine.
Lock these field names early — renaming requires coordinating with Member 4.

Response shape (per proposal Section 4.2):
{
    "transaction_id": str,
    "relational_risk_score": float,
    "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "suspicious_subgraph": {
        "nodes": [str, ...],
        "edges": [{"src": str, "dst": str, "weight": float}, ...],
        "sink_account": str,
        "pattern": str
    }
}
"""
