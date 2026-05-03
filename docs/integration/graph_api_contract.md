# GraphSAGE Component — API Contract for Fusion Engine

**Component:** Edge-Enhanced GraphSAGE Relational Fraud Detector
**Consumed by:** Fusion Engine + LLM Forensic Reporting Layer
**Status:** Draft v0.3 — sample JSON for mock data generation
**Locked at:** May 11 (Proposal Presentation 1) — final contract

This document defines the JSON contract between the GraphSAGE component (Member 1) and the Fusion Engine (Member 4). Sample responses are in [`examples/api_responses/`](../../examples/api_responses/).

---

## 1. Service overview

| Property | Value |
|---|---|
| HTTP method | `POST` |
| Path | `/api/graph/analyze` |
| Content type | `application/json` |
| Target latency (NFR) | **p95 < 500 ms** per request |
| Auth | None (internal trusted network) |
| Concurrency | Stateless — Fusion Engine fires requests in parallel |

The Fusion Engine sends one request per transaction. The response must include `transaction_id` echoed back so Member 4 can join responses from the three upstream models (graph, behavioral, temporal).

---

## 2. Request schema

```json
{
  "transaction_id": "TX_2026_05_01_00045821",
  "step": 723,
  "type": "TRANSFER",
  "amount": 187543.22,
  "nameOrig": "C2029041842",
  "nameDest": "C964377943",
  "oldbalanceOrg": 187543.22,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 187543.22,
  "isFlaggedFraud": 0
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `transaction_id` | string | yes | Unique ID; echoed in response |
| `step` | int | yes | PaySim hour (1–743) |
| `type` | string | yes | One of: `TRANSFER`, `CASH_OUT`, `CASH_IN`, `PAYMENT`, `DEBIT` |
| `amount` | float | yes | Transaction amount |
| `nameOrig` | string | yes | Sender account ID |
| `nameDest` | string | yes | Receiver account ID |
| `oldbalanceOrg` | float | yes | Sender balance before |
| `newbalanceOrig` | float | yes | Sender balance after |
| `oldbalanceDest` | float | yes | Receiver balance before |
| `newbalanceDest` | float | yes | Receiver balance after |
| `isFlaggedFraud` | int (0/1) | yes | Legacy rule output |

**Note:** if `type` is not `TRANSFER` or `CASH_OUT`, the GraphSAGE model is not applicable (PaySim contains zero fraud in other types — see EDA report §3). The endpoint will still respond with a valid JSON but with `risk_level: NOT_APPLICABLE` and `relational_risk_score: 0.0`.

---

## 3. Response schema

### 3.1 Top-level structure

```json
{
  "transaction_id": "TX_2026_05_01_00045821",
  "timestamp": "2026-05-01T03:42:17.123Z",
  "model_version": "graphsage-edge-mlp-v0.3.0",
  "stage": "stage_3_full",

  "relational_risk_score": 0.94,
  "risk_level": "CRITICAL",
  "confidence": 0.91,

  "input_transaction": { /* echo of request fields */ },
  "suspicious_subgraph": { /* see 3.2 */ },
  "metadata": { /* see 3.3 */ }
}
```

| Field | Type | Notes |
|---|---|---|
| `transaction_id` | string | Echoed from request |
| `timestamp` | ISO 8601 string | Server time the response was generated |
| `model_version` | string | Semver of the GraphSAGE model used |
| `stage` | string | One of: `stage_1_baseline`, `stage_2_edge_mlp`, `stage_3_full` |
| `relational_risk_score` | float [0,1] | **Primary input to Logistic Regression meta-classifier.** Higher = more suspicious. |
| `risk_level` | string | One of: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `NOT_APPLICABLE`. Thresholds in [`configs/model_config.yaml`](../../configs/model_config.yaml) |
| `confidence` | float [0,1] | Model's confidence in its own prediction (calibrated probability) |

### 3.2 `suspicious_subgraph` object

This is the structural evidence the Fusion Engine's LLM uses for the forensic narrative.

```json
"suspicious_subgraph": {
  "k_hop": 2,
  "node_count": 8,
  "edge_count": 11,
  "nodes": [ /* see node schema below */ ],
  "edges": [ /* see edge schema below */ ],
  "sink_account": "C964377943",
  "pattern": "HUB_AND_SPOKE",
  "pattern_confidence": 0.88,
  "structural_evidence": {
    "convergence_count": 3,
    "fresh_sender_ratio": 0.75,
    "mean_drain_ratio": 0.97,
    "mules_in_subgraph": 1
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `k_hop` | int | Always 2 for v0.3 (proposal §4.2 Component 4) |
| `node_count` | int | Number of nodes in the extracted subgraph |
| `edge_count` | int | Number of edges in the extracted subgraph |
| `nodes` | array | See node schema |
| `edges` | array | See edge schema |
| `sink_account` | string | Identified terminal node (mule) — **PRIMARY FORENSIC TARGET** |
| `pattern` | string | One of: `HUB_AND_SPOKE`, `SMURFING`, `LAYERING`, `ACCOUNT_TAKEOVER`, `UNKNOWN` |
| `pattern_confidence` | float [0,1] | Confidence in pattern classification |
| `structural_evidence.convergence_count` | int | Distinct senders converging on the sink |
| `structural_evidence.fresh_sender_ratio` | float | % of senders with no prior history |
| `structural_evidence.mean_drain_ratio` | float | Avg `amount/oldbalanceOrg` across edges in subgraph |
| `structural_evidence.mules_in_subgraph` | int | Number of nodes labeled mule by the model |

**Empty case:** if the trigger transaction is not in TRANSFER/CASH_OUT, or if there are no neighbors, `suspicious_subgraph` may be `null`.

### 3.2.1 Node schema (inside `nodes`)

```json
{
  "account_id": "C964377943",
  "role": "MULE_CENTRAL",
  "node_risk_score": 0.97,
  "in_degree": 4,
  "out_degree": 0,
  "first_seen_step": 300,
  "last_seen_step": 723,
  "fraud_count_received": 3,
  "total_received_amount": 729382.45
}
```

| Field | Type | Notes |
|---|---|---|
| `account_id` | string | PaySim account name (e.g. `C964377943`) |
| `role` | string | `MULE_CENTRAL`, `FRESH_SENDER`, `RELAY`, `LEGITIMATE`, `TRIGGER_PARTICIPANT` |
| `node_risk_score` | float [0,1] | Per-node risk from the GraphSAGE classifier |
| `in_degree` | int | Number of edges pointing to this account in the subgraph |
| `out_degree` | int | Number of edges from this account in the subgraph |
| `first_seen_step` | int | Earliest step this account appears |
| `last_seen_step` | int | Latest step in the extracted subgraph |
| `fraud_count_received` | int | Optional — how many fraud edges hit this node |
| `total_received_amount` | float | Optional — sum of inbound amounts |

### 3.2.2 Edge schema (inside `edges`)

```json
{
  "src": "C2029041842",
  "dst": "C964377943",
  "amount": 187543.22,
  "step": 723,
  "edge_attention_weight": 0.93,
  "edge_features": {
    "amount_log": 12.14,
    "drain_ratio": 1.0,
    "src_drained": 1,
    "dst_was_empty": 0,
    "time_gap": -1,
    "type_is_transfer": 1
  },
  "is_trigger_edge": true,
  "is_fraud_predicted": true
}
```

| Field | Type | Notes |
|---|---|---|
| `src` | string | Sender account |
| `dst` | string | Receiver account |
| `amount` | float | Transaction amount |
| `step` | int | PaySim step |
| `edge_attention_weight` | float [0,1] | Output of Edge-MLP (Novelty 1) — high means this edge dominated message passing |
| `edge_features` | object | The 6 raw edge features fed to the model |
| `is_trigger_edge` | bool | True if this is the transaction that triggered analysis |
| `is_fraud_predicted` | bool | True if the model predicts this specific edge is fraudulent |

### 3.3 `metadata` object

Diagnostic info — Member 4 may log but does not need to render.

```json
"metadata": {
  "inference_latency_ms": 47,
  "graph_version": "paysim_2026_05_01",
  "extraction_method": "k_hop_subgraph_pyg",
  "ablation_stage": 3
}
```

---

## 4. Error responses

If the GraphSAGE service fails (model not loaded, malformed input, etc.), it returns HTTP 4xx/5xx with this body:

```json
{
  "transaction_id": "TX_2026_05_01_00045821",
  "error": "BadRequest",
  "message": "Field 'amount' must be non-negative",
  "fallback_score": null
}
```

Per Fusion Engine FR2 (graceful degradation), Member 4 treats any non-200 response as "GraphSAGE unavailable" and proceeds with the other 2 modalities.

---

## 5. Risk level thresholds (current placeholders)

```yaml
# configs/model_config.yaml
inference:
  risk_thresholds:
    low:      0.25
    medium:   0.50
    high:     0.75
    critical: 0.90
```

`relational_risk_score < 0.25` → `LOW`
`0.25 <= score < 0.50` → `MEDIUM`
`0.50 <= score < 0.75` → `HIGH`
`0.75 <= score < 0.90` → `HIGH` (still high)
`score >= 0.90` → `CRITICAL`

These will be calibrated post-training (T7).

---

## 6. Sample responses (for mock generation)

See [`examples/api_responses/`](../../examples/api_responses/):

- `critical_fraud_hub_and_spoke.json` — high-confidence fraud, full subgraph
- `medium_risk_ambiguous.json` — borderline case, partial subgraph
- `low_risk_legitimate.json` — non-fraud, minimal subgraph
- `not_applicable_payment.json` — unsupported transaction type
- `error_bad_request.json` — error response example

The Fusion Engine's mock generator should rotate through these to cover all branches of its fusion logic.

---

## 7. Versioning policy

- The contract version is in `model_version` field (semver)
- **Breaking changes** (renaming a field, removing a field, changing types) trigger a major bump and require coordination with Member 4
- **Additive changes** (new optional fields) trigger a minor bump and are backward-compatible

Lock the major version after Proposal Presentation 1 (May 11, 2026). Subsequent changes go through a 1-week notice via the team chat.
