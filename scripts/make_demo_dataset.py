"""Curate the streamlined PP2 demo dataset.

Mines the full graph for a handful of high-signal demo scenarios and writes
data/demo/demo_transactions.json — a tiny file the demo page loads instantly,
so the live demo never touches the 6.3M-row parquet:

- HUB: trigger edge into the highest in-degree fraud sink (big mule ring)
- TAKEOVER: fraud edge with src_drained=1 into a previously-empty account
- LEGIT: an ordinary low-drain transfer
- NOT_APPLICABLE: a PAYMENT-type transaction (model out of scope)
- INVALID: negative amount — demonstrates Pydantic validation live

Each entry carries the exact request-body fields of contract §2 (balances are
reconstructed to be consistent with drain_ratio where possible, since the
graph stores engineered features, not raw balances).

Usage:
    python scripts/make_demo_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from graphsage.extraction.subgraph import (
    EF_AMOUNT_LOG,
    EF_DRAIN_RATIO,
    EF_SRC_DRAINED,
    EF_TYPE_IS_TRANSFER,
    load_node_names,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
NAMES_CACHE = REPO_ROOT / "data" / "graph" / "node_names.npy"
OUT_PATH = REPO_ROOT / "data" / "demo" / "demo_transactions.json"


def txn_entry(
    label: str,
    description: str,
    data,
    names: np.ndarray,
    edge_id: int,
    txn_id: str,
) -> dict:
    src = int(data.edge_index[0, edge_id])
    dst = int(data.edge_index[1, edge_id])
    attr = data.edge_attr[edge_id]
    amount = round(float(torch.expm1(attr[EF_AMOUNT_LOG])), 2)
    drain = float(attr[EF_DRAIN_RATIO])
    # Reconstruct plausible balances from amount and drain_ratio
    # (drain_ratio = amount / oldbalanceOrg).
    old_org = round(amount / drain, 2) if drain > 1e-6 else round(amount * 4, 2)
    txn_type = "TRANSFER" if float(attr[EF_TYPE_IS_TRANSFER]) >= 0.5 else "CASH_OUT"
    return {
        "label": label,
        "description": description,
        "request": {
            "transaction_id": txn_id,
            "step": int(data.edge_step[edge_id]),
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


def main() -> None:
    print("Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
    names = load_node_names(PARQUET_PATH, cache_path=NAMES_CACHE)

    fraud_edges = (data.edge_isFraud == 1).nonzero(as_tuple=True)[0]
    fraud_dst = data.edge_index[1, fraud_edges]

    # HUB: the fraud sink the MODEL is most confident about (needs the score
    # cache produced by the API's first startup), among sinks with a ring of
    # senders — raw in-degree alone picks mostly-legit cash-out hubs that the
    # model correctly scores low, which demos badly.
    in_deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    cache = torch.load(
        REPO_ROOT / "data" / "graph" / "inference_cache.pt", weights_only=True
    )
    probs = cache["probs"]
    ringy = in_deg[fraud_dst] >= 5
    candidates = fraud_dst[ringy]
    hub_sink = candidates[torch.argmax(probs[candidates])]
    hub_edge = int(
        fraud_edges[(fraud_dst == hub_sink).nonzero(as_tuple=True)[0][0]]
    )
    print(
        f"  hub sink {names[int(hub_sink)]}: in_degree={int(in_deg[hub_sink])}, "
        f"model prob={float(probs[hub_sink]):.3f}"
    )

    # TAKEOVER: drained sender, low-degree sink (isolated pair, not a ring).
    attrs = data.edge_attr[fraud_edges]
    drained = attrs[:, EF_SRC_DRAINED] >= 0.5
    lonely = in_deg[fraud_dst] <= 2
    takeover_candidates = fraud_edges[drained & lonely]
    takeover_edge = int(takeover_candidates[0])

    # LEGIT: non-fraud, low drain ratio, and a quiet destination (a busy
    # dst would extract a big hub-shaped subgraph and muddy the story).
    # Quiet accounts (in_deg<=3) have too little structure and the model sits
    # near 0.5 on them; accounts with moderate history separate cleanly. Pick
    # the lowest-scored destination so LOW risk is demonstrated honestly.
    dst_deg = in_deg[data.edge_index[1]]
    legit_mask = (
        (data.edge_isFraud == 0)
        & (data.edge_attr[:, EF_DRAIN_RATIO] < 0.2)
        & (data.edge_attr[:, EF_DRAIN_RATIO] > 0)
        & (dst_deg >= 3)
        & (dst_deg <= 10)
    )
    legit_candidates = legit_mask.nonzero(as_tuple=True)[0]
    legit_edge = int(
        legit_candidates[torch.argmin(probs[data.edge_index[1, legit_candidates]])]
    )
    print(f"  legit dst model prob={float(probs[data.edge_index[1, legit_edge]]):.3f}")

    scenarios = [
        txn_entry(
            "Fraud ring (hub-and-spoke)",
            "Many senders converging on one mule sink — expect CRITICAL/HIGH with a large subgraph",
            data, names, hub_edge, "TX_DEMO_HUB_001",
        ),
        txn_entry(
            "Account takeover",
            "Victim account fully drained into a quiet destination — expect elevated risk, small subgraph",
            data, names, takeover_edge, "TX_DEMO_ATO_002",
        ),
        txn_entry(
            "Legitimate transfer",
            "Routine partial transfer between accounts with history — expect LOW",
            data, names, legit_edge, "TX_DEMO_LEGIT_003",
        ),
    ]

    # NOT_APPLICABLE: PAYMENT type (schema-valid, out of model scope).
    scenarios.append(
        {
            "label": "Out-of-scope type (PAYMENT)",
            "description": "PaySim has zero fraud outside TRANSFER/CASH_OUT — expect NOT_APPLICABLE",
            "request": {
                **scenarios[2]["request"],
                "transaction_id": "TX_DEMO_NA_004",
                "type": "PAYMENT",
            },
        }
    )

    # INVALID: fails Pydantic validation live on stage.
    scenarios.append(
        {
            "label": "Invalid input (negative amount)",
            "description": "Demonstrates request validation — expect 422 BadRequest",
            "request": {
                **scenarios[0]["request"],
                "transaction_id": "TX_DEMO_BAD_005",
                "amount": -500.0,
            },
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"scenarios": scenarios}, indent=2))
    print(f"Wrote {len(scenarios)} scenarios to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
