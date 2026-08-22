"""Curate the streamlined demo dataset from the SERVING BUNDLE.

Mines scenarios using exactly the scores the API serves, so what the demo page
shows can never drift from what the model actually predicts. (An earlier version
read the raw graph plus a separate score cache; when the served model changed,
the "fraud ring" scenario silently became LOW risk.)

Scenarios, chosen to exercise every branch the contract defines:
  CRITICAL       highest-scoring fraud sink that has a ring of senders
  HIGH           a flagged account just above the tuned threshold
  LOW            a genuinely low-scoring destination
  NOT_APPLICABLE a PAYMENT-type transaction (out of model scope)
  INVALID        negative amount — demonstrates request validation live

Each entry carries the exact request-body fields of contract §2. Balances are
reconstructed to be consistent with drain_ratio, since the graph stores
engineered features rather than raw balances.

Usage:
    python scripts/make_demo_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from graphsage.extraction.subgraph import (
    EF_AMOUNT_LOG,
    EF_DRAIN_RATIO,
    EF_TYPE_IS_TRANSFER,
    load_node_names,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO_ROOT / "data" / "graph" / "serving_bundle.pt"
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
NAMES_CACHE = REPO_ROOT / "data" / "graph" / "node_names.npy"
OUT_PATH = REPO_ROOT / "data" / "demo" / "demo_transactions.json"


def txn_entry(label, description, b, names, edge_id, txn_id) -> dict:
    src = int(b["edge_index"][0, edge_id])
    dst = int(b["edge_index"][1, edge_id])
    attr = b["edge_attr"][edge_id]
    amount = round(float(torch.expm1(attr[EF_AMOUNT_LOG])), 2)
    drain = float(attr[EF_DRAIN_RATIO])
    old_org = round(amount / drain, 2) if drain > 1e-6 else round(amount * 4, 2)
    txn_type = "TRANSFER" if float(attr[EF_TYPE_IS_TRANSFER]) >= 0.5 else "CASH_OUT"
    return {
        "label": label,
        "description": description,
        "request": {
            "transaction_id": txn_id,
            "step": int(b["edge_step"][edge_id]),
            "type": txn_type,
            "amount": amount,
            "nameOrig": str(names[src]),
            "nameDest": str(names[dst]),
            "oldbalanceOrg": old_org,
            "newbalanceOrig": round(max(old_org - amount, 0.0), 2),
            "oldbalanceDest": 0.0,
            "newbalanceDest": amount,
            "isFlaggedFraud": 0,
        },
    }


def pick_edge(b, mask: torch.Tensor, scores: torch.Tensor, mode: str) -> int:
    """Pick the edge whose DESTINATION best represents `mode`."""
    ids = mask.nonzero(as_tuple=True)[0]
    if ids.numel() == 0:
        raise SystemExit(f"no candidate edge for '{mode}'")
    dst_scores = scores[b["edge_index"][1, ids]]
    idx = torch.argmax(dst_scores) if mode == "max" else torch.argmin(dst_scores)
    return int(ids[idx])


def main() -> None:
    if not BUNDLE_PATH.exists():
        raise SystemExit(f"{BUNDLE_PATH} not found — run export_serving_bundle.py")
    print("Loading serving bundle...")
    b = torch.load(BUNDLE_PATH, weights_only=False, map_location="cpu")
    names = load_node_names(PARQUET_PATH, cache_path=NAMES_CACHE)
    scores = b["node_scores"]
    threshold = float(b["threshold"])

    # Same band edges the API derives (see GraphPredictor.risk_bands).
    receivers = b["node_degrees"][:, 0] > 0
    critical = max(float(torch.quantile(scores[receivers].float(), 0.995)), threshold)
    medium = 0.5 * threshold
    print(f"bands: medium {medium:.4f} | high {threshold:.4f} | critical {critical:.4f}")

    ei = b["edge_index"]
    in_deg = torch.bincount(ei[1], minlength=int(b["num_nodes"]))
    fraud = b["edge_isFraud"] == 1
    dst_score = scores[ei[1]]

    # CRITICAL — a real fraud sink with a ring of senders that the model scores high.
    crit_edge = pick_edge(b, fraud & (in_deg[ei[1]] >= 4) & (dst_score >= critical),
                          scores, "max")
    # HIGH — flagged, but below the critical tail.
    high_edge = pick_edge(b, fraud & (dst_score >= threshold) & (dst_score < critical),
                          scores, "max")
    # LOW — non-fraud destination the model is confident about.
    low_edge = pick_edge(b, (~fraud) & (dst_score < medium) & (in_deg[ei[1]] >= 2),
                         scores, "min")

    for tag, e in (("critical", crit_edge), ("high", high_edge), ("low", low_edge)):
        print(f"  {tag:9} {names[int(ei[1, e])]}  score={float(scores[ei[1, e]]):.4f}"
              f"  in_degree={int(in_deg[ei[1, e]])}")

    scenarios = [
        txn_entry("Fraud ring (hub-and-spoke)",
                  "Many senders converging on one mule sink — expect CRITICAL with a large subgraph",
                  b, names, crit_edge, "TX_DEMO_HUB_001"),
        txn_entry("Suspicious account",
                  "Above the tuned decision threshold but not extreme — expect HIGH",
                  b, names, high_edge, "TX_DEMO_HIGH_002"),
        txn_entry("Legitimate transfer",
                  "Routine transfer to an established account — expect LOW",
                  b, names, low_edge, "TX_DEMO_LEGIT_003"),
    ]
    scenarios.append({
        "label": "Out-of-scope type (PAYMENT)",
        "description": "No fraud exists outside TRANSFER/CASH_OUT — expect NOT_APPLICABLE",
        "request": {**scenarios[2]["request"],
                    "transaction_id": "TX_DEMO_NA_004", "type": "PAYMENT"},
    })
    scenarios.append({
        "label": "Invalid input (negative amount)",
        "description": "Demonstrates request validation — expect 422 BadRequest",
        "request": {**scenarios[0]["request"],
                    "transaction_id": "TX_DEMO_BAD_005", "amount": -500.0},
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"scenarios": scenarios}, indent=2))
    print(f"Wrote {len(scenarios)} scenarios to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
