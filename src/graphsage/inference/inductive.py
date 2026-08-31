"""Scoring an account the trained network has never seen.

This is the part of GraphSAGE that a transductive model cannot do at all. A
GCN learns one embedding per node in the training graph; ask it about an
account that did not exist then and it has nothing to return. GraphSAGE learns
the *aggregator* instead — a function from a node's neighbourhood to its
embedding — so a node that appeared after training gets a real embedding by
running that function over whoever it connects to.

The demo built on this module makes the claim checkable rather than asserted:
attach a brand-new account to different neighbours and the score moves, because
the only thing the model can see about it is the company it keeps.

Nothing here mutates the served graph. A node exists for the duration of one
request and is gone afterwards.

The feature derivation below mirrors `graphsage.data.temporal._node_features_v2`
column for column. That is not an aesthetic preference: the shipped checkpoint
was trained on features built by that function, so a new node whose features
are computed even slightly differently is being scored in a space the model was
never fitted on, and the number that comes back would be meaningless. The
equivalence is pinned by a test that runs both paths over the same rows.
"""

from __future__ import annotations

import numpy as np

# Column order of data.edge_attr — graph_builder.EDGE_FEATURE_COLS.
EDGE_COLS = ("amount_log", "drain_ratio", "src_drained",
             "dst_was_empty", "time_gap", "type_is_transfer")

# Column order of the 12-dim v2 node features.
NODE_COLS = ("in_degree", "out_degree", "mean_in_amount_log",
             "mean_out_amount_log", "max_in_amount_log", "std_in_amount_log",
             "distinct_senders", "distinct_receivers", "mean_drain_out",
             "mean_dst_was_empty_in", "txn_velocity", "first_step_norm")

# Count-like columns are log1p-scaled so degrees do not drown out ratios.
_LOG1P_COLS = (0, 1, 6, 7, 10)


def edge_features(txn: dict) -> np.ndarray:
    """The six edge attributes for one transaction — scripts/prepare_features.py.

    `time_gap` is the caller's to supply: it is defined as the gap since the
    *destination's* previous inbound transfer, which is a property of the
    stream rather than of this row. -1 is the training sentinel for "first
    inbound seen for this destination", and is the honest value for a
    destination whose history we are not being given.
    """
    amount = float(txn.get("amount", 0.0) or 0.0)
    old_org = float(txn.get("oldbalanceOrg", 0.0) or 0.0)
    drain = min(max(amount / old_org, 0.0), 1.0) if old_org > 0 else 0.0
    return np.array([
        np.log1p(amount),
        drain,
        1.0 if float(txn.get("newbalanceOrig", 0.0) or 0.0) == 0 else 0.0,
        1.0 if float(txn.get("oldbalanceDest", 0.0) or 0.0) == 0 else 0.0,
        float(txn.get("time_gap", -1.0)),
        1.0 if str(txn.get("type", "")).upper() == "TRANSFER" else 0.0,
    ], dtype=np.float32)


def derive_node_features(out_txns: list[dict], in_txns: list[dict],
                         horizon: int) -> np.ndarray:
    """The 12 v2 features for one account, from only its own transactions.

    `out_txns` are the ones it sent, `in_txns` the ones it received. Every
    column is an aggregate over those two lists — which is what makes the demo
    honest: the operator supplies transactions, not a feature vector, and has
    no way to hand the model a number that did not come from a transaction.
    """
    x = np.zeros(12, dtype=np.float32)

    in_amt = np.array([np.log1p(float(t.get("amount", 0.0) or 0.0))
                       for t in in_txns], dtype=np.float32)
    out_amt = np.array([np.log1p(float(t.get("amount", 0.0) or 0.0))
                        for t in out_txns], dtype=np.float32)

    x[0] = len(in_txns)
    x[1] = len(out_txns)
    x[2] = in_amt.mean() if in_amt.size else 0.0
    x[3] = out_amt.mean() if out_amt.size else 0.0
    x[4] = in_amt.max() if in_amt.size else 0.0
    # pandas .std() is sample standard deviation (ddof=1) and is NaN for a
    # single row; training ran nan_to_num over exactly that case.
    x[5] = in_amt.std(ddof=1) if in_amt.size > 1 else 0.0
    x[6] = len({t.get("nameOrig") for t in in_txns})
    x[7] = len({t.get("nameDest") for t in out_txns})

    if out_txns:
        drains = []
        for t in out_txns:
            amount = float(t.get("amount", 0.0) or 0.0)
            old_org = float(t.get("oldbalanceOrg", 0.0) or 0.0)
            drains.append(min(max(amount / old_org, 0.0), 1.0) if old_org > 0 else 0.0)
        x[8] = float(np.mean(drains))
    if in_txns:
        x[9] = float(np.mean([
            1.0 if float(t.get("oldbalanceDest", 0.0) or 0.0) == 0 else 0.0
            for t in in_txns
        ]))

    steps = [int(t.get("step", 0)) for t in (out_txns + in_txns)]
    if steps:
        first, last = min(steps), max(steps)
        active = max(last - first + 1.0, 1.0)
        x[10] = (x[0] + x[1]) / active          # raw degrees, before log1p
        x[11] = first / max(horizon, 1)
    else:
        # np.inf first_step in training fell through to the -1 sentinel.
        x[10] = 0.0
        x[11] = -1.0

    for col in _LOG1P_COLS:
        x[col] = np.log1p(x[col])
    return x
