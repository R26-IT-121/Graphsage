# Integrating the GraphSAGE component

Everything a consumer needs to call this service. Read the
[full contract](graph_api_contract.md) for field-by-field detail.

---

## 1. What this component does

Given one transaction, it answers two questions:

1. **How risky is the receiving account?** — `relational_risk_score`, an
   isotonic-calibrated probability that the account is a money mule, derived
   from the structure of the transaction network rather than the transaction's
   own attributes.
2. **What does the surrounding money network look like?** —
   `suspicious_subgraph`, the 2-hop ring around the transaction with forensic
   roles, a FATF pattern label, and per-edge attention weights.

The second part is the one that matters for report generation: it supplies the
concrete facts ("9 senders, 100% of them brand-new accounts, all converging on
C197117096") that a narrative can cite.

---

## 2. Quick start

```bash
# 1. Start the service (needs data/graph/serving_bundle.pt)
docker compose up --build          # or: python scripts/serve_api.py

# 2. Check it is alive and see the live risk bands
curl -s localhost:8000/health | python -m json.tool

# 3. Score a transaction
curl -s -X POST localhost:8000/api/graph/analyze \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON' | python -m json.tool
{
  "transaction_id": "TX_001", "step": 1, "type": "CASH_OUT",
  "amount": 1277212.25, "nameOrig": "C467632528", "nameDest": "C716083600",
  "oldbalanceOrg": 1277212.25, "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0, "newbalanceDest": 1277212.25,
  "isFlaggedFraud": 0
}
JSON

# 4. Prove the contract holds (run this in your CI)
python scripts/contract_test.py --url http://localhost:8000
```

Ready-made request bodies covering every response branch live in
[`data/demo/demo_transactions.json`](../../data/demo/demo_transactions.json).

---

## 3. The five response branches to handle

| Situation | HTTP | What you get |
|---|---|---|
| Scored normally | 200 | full payload, `risk_level` in LOW/MEDIUM/HIGH/CRITICAL |
| Type outside TRANSFER/CASH_OUT | 200 | `risk_level: NOT_APPLICABLE`, `score: 0.0`, `suspicious_subgraph: null` |
| Accounts/edge unknown to the graph | 404 | `{error: "NotFound", message, transaction_id}` |
| Request fails validation | 422 | `{error: "BadRequest", message}` |
| Service down or timing out | — | treat as GraphSAGE unavailable, fuse without the graph signal |

Two of these are easy to get wrong:

- **`NOT_APPLICABLE` is not "low risk".** It means the model has no opinion —
  PaySim contains zero fraud outside TRANSFER and CASH_OUT. Excluding the graph
  signal from the fusion is more correct than treating it as a 0.0 vote.
- **404 is normal, not an outage.** The graph is a fixed snapshot; a transaction
  between accounts it has never seen cannot be anchored to a subgraph.

---

## 4. Using the payload for report generation

The fields worth putting in a prompt, in rough order of usefulness:

- `suspicious_subgraph.pattern` — FATF typology (`HUB_AND_SPOKE`, `SMURFING`,
  `LAYERING`, `ACCOUNT_TAKEOVER`, `UNKNOWN`). Maps directly to typology
  documents in a retrieval corpus.
- `structural_evidence` — the quantitative claims to cite:
  `convergence_count` (how many senders feed the sink), `fresh_sender_ratio`
  (share with no prior history), `mean_drain_ratio`, `mules_in_subgraph`.
- `sink_account` — the account to name as the collection point.
- `nodes[].role` — `MULE_CENTRAL`, `FRESH_SENDER`, `RELAY`,
  `TRIGGER_PARTICIPANT`, `LEGITIMATE`.
- `pattern_scores` — runner-up typologies with scores, for hedged wording when
  the top pattern is not dominant.
- `edges[].edge_attention_weight` — which transactions the model weighted most.

**Grounding rule:** every number in a generated narrative should come from this
payload. `structural_evidence` exists precisely so claims are checkable.

---

## 5. Two things that are not hard-coded

**Risk bands.** `relational_risk_score` is a calibrated probability on a ~4.7%
base-rate population, so it tops out near 0.25 — an absolute `>= 0.9` rule would
never fire. Use the `risk_level` we return, or read live edges from `/health`
→ `risk_bands`. See contract Section 5.

**Model identity.** `stage` and `model_meta` in `/health` report which
configuration is being served (e.g. `stage_3b_v2`, isotonic calibration). Log
these alongside your fused decisions so a result can be traced to a model.

---

## 6. Embedding the graph view

The service serves an interactive investigation page at `/demo`, framework-free
so it can be iframed or linked from a dashboard. Deep-link straight to a
transaction:

```
http://localhost:8000/demo?nameOrig=C467632528&nameDest=C716083600&step=1&autorun=1
```

It renders the extracted ring — sink centred, nodes coloured by role, edge width
by attention — plus a plain-English verdict. If the dashboard prefers its own
rendering, take `suspicious_subgraph.nodes` / `.edges` and draw them directly;
the payload is designed to drop into any force-directed layout.

---

## 7. Operational notes

- **Latency:** p95 well under 500 ms (NFR1). No inference happens per request —
  scores are precomputed in the serving bundle, so a request is a graph lookup
  plus subgraph extraction.
- **Startup:** the service reads a ~170 MB bundle at boot; give it ~60-90 s
  before the first call. The compose healthcheck already waits.
- **Statelessness:** no session state, safe to run multiple replicas.
- **CORS:** `GET`/`POST` allowed from any origin (contract Section 1 assumes an
  internal trusted network — put it behind the gateway before any public use).
