"""The demo's feature derivation must match the one the model was trained on.

If these drift, a new node is scored in a feature space the network was never
fitted on and the number it returns is meaningless — while still looking
perfectly plausible. So the check is not "does it run" but "is it the same
arithmetic", run against the real training function over the same rows.
"""

import numpy as np
import pandas as pd
import pytest

from graphsage.data.temporal import _node_features_v2
from graphsage.inference.inductive import derive_node_features, edge_features

HORIZON = 743


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["amount_log"] = np.log1p(df["amount"]).astype("float32")
    df["drain_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        np.clip(df["amount"] / df["oldbalanceOrg"], 0.0, 1.0), 0.0,
    ).astype("float32")
    df["dst_was_empty"] = (df["oldbalanceDest"] == 0).astype("int8")
    return df


def _training_features(rows: list[dict], target: str) -> np.ndarray:
    """Run the real training builder and pull out the target account's row."""
    df = _frame(rows)
    names = pd.Index(sorted(set(df["nameOrig"]) | set(df["nameDest"])))
    name_to_id = pd.Series(range(len(names)), index=names)
    x = _node_features_v2(df, name_to_id, len(names), HORIZON)
    return x[name_to_id[target]].numpy()


CASES = {
    "receives from three senders": ([
        dict(step=700, type="TRANSFER", amount=100_000.0, nameOrig="A", nameDest="MULE",
             oldbalanceOrg=100_000.0, newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=100_000.0),
        dict(step=701, type="TRANSFER", amount=250_000.0, nameOrig="B", nameDest="MULE",
             oldbalanceOrg=250_000.0, newbalanceOrig=0.0, oldbalanceDest=100_000.0, newbalanceDest=350_000.0),
        dict(step=702, type="TRANSFER", amount=75_000.0, nameOrig="C", nameDest="MULE",
             oldbalanceOrg=90_000.0, newbalanceOrig=15_000.0, oldbalanceDest=350_000.0, newbalanceDest=425_000.0),
    ], "MULE"),
    "single inbound (std is NaN in pandas)": ([
        dict(step=705, type="TRANSFER", amount=404_394.04, nameOrig="A", nameDest="NEW",
             oldbalanceOrg=404_394.04, newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=404_394.04),
    ], "NEW"),
    "sends and receives": ([
        dict(step=700, type="TRANSFER", amount=500_000.0, nameOrig="X", nameDest="HUB",
             oldbalanceOrg=500_000.0, newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=500_000.0),
        dict(step=703, type="CASH_OUT", amount=500_000.0, nameOrig="HUB", nameDest="Y",
             oldbalanceOrg=500_000.0, newbalanceOrig=0.0, oldbalanceDest=10_000.0, newbalanceDest=510_000.0),
    ], "HUB"),
    "only sends": ([
        dict(step=710, type="CASH_OUT", amount=20_000.0, nameOrig="SENDER", nameDest="Z",
             oldbalanceOrg=80_000.0, newbalanceOrig=60_000.0, oldbalanceDest=5_000.0, newbalanceDest=25_000.0),
    ], "SENDER"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_training_builder(name):
    rows, target = CASES[name]
    expected = _training_features(rows, target)
    got = derive_node_features(
        [r for r in rows if r["nameOrig"] == target],
        [r for r in rows if r["nameDest"] == target],
        horizon=HORIZON,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_edge_features_match_prepare_features():
    txn = dict(step=705, type="TRANSFER", amount=404_394.04,
               oldbalanceOrg=404_394.04, newbalanceOrig=0.0,
               oldbalanceDest=0.0, newbalanceDest=404_394.04)
    got = edge_features(txn)
    assert got[0] == pytest.approx(np.log1p(404_394.04))
    assert got[1] == pytest.approx(1.0)      # fully drained
    assert got[2] == 1.0                     # src_drained
    assert got[3] == 1.0                     # dst_was_empty
    assert got[4] == -1.0                    # first inbound sentinel
    assert got[5] == 1.0                     # TRANSFER
