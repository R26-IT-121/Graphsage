"""Pydantic request/response schemas for the /api/graph/analyze endpoint.

Field names implement the locked contract in docs/integration/graph_api_contract.md
(v0.3, locked at PP1). Renaming any field is a breaking change requiring
coordination with Member 4's fusion engine — see contract §7.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TxnType(str, Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    PAYMENT = "PAYMENT"
    DEBIT = "DEBIT"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Pattern(str, Enum):
    HUB_AND_SPOKE = "HUB_AND_SPOKE"
    SMURFING = "SMURFING"
    LAYERING = "LAYERING"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    UNKNOWN = "UNKNOWN"


class NodeRole(str, Enum):
    MULE_CENTRAL = "MULE_CENTRAL"
    FRESH_SENDER = "FRESH_SENDER"
    RELAY = "RELAY"
    LEGITIMATE = "LEGITIMATE"
    TRIGGER_PARTICIPANT = "TRIGGER_PARTICIPANT"


# --------------------------------------------------------------------- #
# Request (contract §2)
# --------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    step: int = Field(ge=1, le=10_000, description="PaySim hour (1-743 in v1 data)")
    type: TxnType
    amount: float = Field(ge=0)
    nameOrig: str = Field(min_length=1, max_length=64)
    nameDest: str = Field(min_length=1, max_length=64)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    isFlaggedFraud: int = Field(ge=0, le=1)

    @field_validator("nameOrig", "nameDest")
    @classmethod
    def strip_account_ids(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("account id must be non-empty")
        return v


# --------------------------------------------------------------------- #
# Response (contract §3)
# --------------------------------------------------------------------- #


class EdgeFeatures(BaseModel):
    amount_log: float
    drain_ratio: float
    src_drained: float
    dst_was_empty: float
    time_gap: float
    type_is_transfer: float


class SubgraphNode(BaseModel):
    account_id: str
    role: NodeRole
    node_risk_score: float = Field(ge=0, le=1)
    in_degree: int = Field(ge=0)
    out_degree: int = Field(ge=0)
    first_seen_step: int
    last_seen_step: int
    fraud_count_received: int = Field(ge=0)
    total_received_amount: float = Field(ge=0)


class SubgraphEdge(BaseModel):
    src: str
    dst: str
    amount: float = Field(ge=0)
    step: int
    edge_attention_weight: float = Field(ge=0, le=1)
    edge_features: EdgeFeatures
    is_trigger_edge: bool
    is_fraud_predicted: bool


class StructuralEvidence(BaseModel):
    convergence_count: int = Field(ge=0)
    fresh_sender_ratio: float = Field(ge=0, le=1)
    mean_drain_ratio: float
    mules_in_subgraph: int = Field(ge=0)


class SuspiciousSubgraph(BaseModel):
    k_hop: int
    node_count: int
    edge_count: int
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
    sink_account: str
    pattern: Pattern
    pattern_confidence: float = Field(ge=0, le=1)
    structural_evidence: StructuralEvidence


class ResponseMetadata(BaseModel):
    inference_latency_ms: int
    graph_version: str
    extraction_method: str = "k_hop_subgraph_pyg"
    ablation_stage: int = 3


class AnalyzeResponse(BaseModel):
    transaction_id: str
    timestamp: str
    model_version: str
    stage: str
    relational_risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    input_transaction: AnalyzeRequest
    suspicious_subgraph: SuspiciousSubgraph | None
    metadata: ResponseMetadata


class ErrorResponse(BaseModel):
    """Contract §4 — Member 4 treats any non-200 as 'GraphSAGE unavailable'."""

    transaction_id: str | None
    error: str
    message: str
    fallback_score: float | None = None
