"""Leakage-free temporal snapshots of the PaySim graph.

The original pipeline (graph_builder + splits) computes node features and mule
labels over the ENTIRE timeline, then splits nodes by first appearance. That
leaks future information twice:

1. Feature leakage — a node evaluated in the test window carries in-degree /
   amount aggregates that already include test-window (future) transactions.
2. Label leakage — a train-window node whose fraud arrives at step 720 is
   trained as a positive with a label that was not observable at train time.

This module replaces that with the standard temporal snapshot protocol: to
evaluate at time t, the model may only see the graph as it existed up to t.

    snapshot     feature/message edges   labeled positives      evaluated on
    train        step <= train_end       fraud in [1,train_end] active nodes, [1,train_end]
    val          step <= val_end         fraud in (train_end,val_end]   nodes active in window,
                                                                        not already positive
    test         step <= max_step        fraud in (val_end,max_step]    nodes active in window,
                                                                        not already positive

Node ids share one global mapping across snapshots (accounts not yet seen have
zero feature rows and are never evaluated), so checkpoints and extractors stay
index-compatible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphsage.data.graph_builder import EDGE_FEATURE_COLS


def _node_features(
    df: pd.DataFrame, name_to_id: pd.Series, num_nodes: int
) -> torch.Tensor:
    """Same 5 structural aggregates as graph_builder, over a time-bounded df."""
    out_stats = df.groupby("nameOrig", observed=True).agg(
        out_degree=("amount_log", "size"), mean_out=("amount_log", "mean")
    )
    in_stats = df.groupby("nameDest", observed=True).agg(
        in_degree=("amount_log", "size"),
        mean_in=("amount_log", "mean"),
        max_in=("amount_log", "max"),
    )
    x = np.zeros((num_nodes, 5), dtype=np.float32)
    out_idx = name_to_id.loc[out_stats.index.values].to_numpy()
    in_idx = name_to_id.loc[in_stats.index.values].to_numpy()
    x[in_idx, 0] = in_stats["in_degree"].to_numpy()
    x[out_idx, 1] = out_stats["out_degree"].to_numpy()
    x[in_idx, 2] = in_stats["mean_in"].to_numpy()
    x[out_idx, 3] = out_stats["mean_out"].to_numpy()
    x[in_idx, 4] = in_stats["max_in"].to_numpy()
    return torch.from_numpy(x)


def _node_features_v2(
    df: pd.DataFrame, name_to_id: pd.Series, num_nodes: int, horizon: int
) -> torch.Tensor:
    """12-dim behavioural feature set (team-repo commit 3c03a89, May 17),
    recomputed per snapshot so it stays leakage-free.

    Columns 0-4 match v1; 5-11 add mule signatures: amount uniformity
    (smurfing), fan-in/out distinctness, outgoing drain, fresh-receiver ratio,
    transaction velocity, and normalised first appearance. Count-like columns
    (0,1,6,7,10) are log1p-scaled so degrees don't drown out ratios.
    """
    out_stats = df.groupby("nameOrig", observed=True).agg(
        out_degree=("amount_log", "size"),
        mean_out=("amount_log", "mean"),
        distinct_recv=("nameDest", "nunique"),
        mean_drain_out=("drain_ratio", "mean"),
        first_step_out=("step", "min"),
        last_step_out=("step", "max"),
    )
    in_stats = df.groupby("nameDest", observed=True).agg(
        in_degree=("amount_log", "size"),
        mean_in=("amount_log", "mean"),
        max_in=("amount_log", "max"),
        std_in=("amount_log", "std"),
        distinct_send=("nameOrig", "nunique"),
        mean_dst_empty_in=("dst_was_empty", "mean"),
        first_step_in=("step", "min"),
        last_step_in=("step", "max"),
    )
    x = np.zeros((num_nodes, 12), dtype=np.float32)
    out_idx = name_to_id.loc[out_stats.index.values].to_numpy()
    in_idx = name_to_id.loc[in_stats.index.values].to_numpy()

    x[in_idx, 0] = in_stats["in_degree"].to_numpy()
    x[out_idx, 1] = out_stats["out_degree"].to_numpy()
    x[in_idx, 2] = in_stats["mean_in"].to_numpy()
    x[out_idx, 3] = out_stats["mean_out"].to_numpy()
    x[in_idx, 4] = in_stats["max_in"].to_numpy()
    x[in_idx, 5] = np.nan_to_num(in_stats["std_in"].to_numpy(), nan=0.0)
    x[in_idx, 6] = in_stats["distinct_send"].to_numpy()
    x[out_idx, 7] = out_stats["distinct_recv"].to_numpy()
    x[out_idx, 8] = out_stats["mean_drain_out"].to_numpy()
    x[in_idx, 9] = in_stats["mean_dst_empty_in"].to_numpy()

    first_step = np.full(num_nodes, np.inf, dtype=np.float32)
    last_step = np.full(num_nodes, -np.inf, dtype=np.float32)
    first_step[out_idx] = np.minimum(
        first_step[out_idx], out_stats["first_step_out"].to_numpy()
    )
    last_step[out_idx] = np.maximum(
        last_step[out_idx], out_stats["last_step_out"].to_numpy()
    )
    first_step[in_idx] = np.minimum(
        first_step[in_idx], in_stats["first_step_in"].to_numpy()
    )
    last_step[in_idx] = np.maximum(
        last_step[in_idx], in_stats["last_step_in"].to_numpy()
    )
    active_steps = np.maximum(last_step - first_step + 1.0, 1.0)
    total_deg = x[:, 0] + x[:, 1]
    x[:, 10] = np.where(np.isfinite(active_steps), total_deg / active_steps, 0.0)
    # Normalised by the snapshot horizon (not the full timeline) — leakage-free.
    x[:, 11] = np.where(
        np.isfinite(first_step), first_step / max(horizon, 1), -1.0
    ).astype(np.float32)

    for col in (0, 1, 6, 7, 10):
        x[:, col] = np.log1p(x[:, col])
    return torch.from_numpy(x)


def _active_nodes(
    df: pd.DataFrame, name_to_id: pd.Series, num_nodes: int
) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    for col in ("nameOrig", "nameDest"):
        ids = name_to_id.loc[df[col].unique()].to_numpy()
        mask[ids] = True
    return mask


def _mules(
    df: pd.DataFrame, name_to_id: pd.Series, num_nodes: int
) -> torch.Tensor:
    y = torch.zeros(num_nodes, dtype=torch.bool)
    fraud_dst = df.loc[df["isFraud"] == 1, "nameDest"].unique()
    if len(fraud_dst):
        y[name_to_id.loc[fraud_dst].to_numpy()] = True
    return y


def build_temporal_snapshots(
    parquet_path: str | Path,
    train_end: int = 600,
    val_end: int = 700,
    feature_version: str = "v1",
) -> tuple[dict[str, Data], dict]:
    """Build train/val/test snapshot Data objects with a shared id space.

    Each snapshot carries:
        x, edge_index, edge_attr, edge_step, edge_isFraud  — graph as of its horizon
        y          — positives observable in ITS label window (bool)
        eval_mask  — nodes this snapshot is scored on
    """
    df = pd.read_parquet(parquet_path)
    max_step = int(df["step"].max())

    all_accounts = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
    name_to_id = pd.Series(np.arange(len(all_accounts)), index=all_accounts)
    num_nodes = int(len(all_accounts))

    windows = {
        "train": (train_end, 0, train_end),
        "val": (val_end, train_end, val_end),
        "test": (max_step, val_end, max_step),
    }

    if feature_version not in ("v1", "v2"):
        raise ValueError(f"feature_version must be v1 or v2, got {feature_version!r}")

    snapshots: dict[str, Data] = {}
    stats: dict = {
        "num_nodes": num_nodes,
        "train_end": train_end,
        "val_end": val_end,
        "feature_version": feature_version,
    }
    for name, (feat_end, label_start, label_end) in windows.items():
        past = df[df["step"] <= feat_end]
        window = df[(df["step"] > label_start) & (df["step"] <= label_end)]

        if feature_version == "v2":
            x = _node_features_v2(past, name_to_id, num_nodes, horizon=feat_end)
        else:
            x = _node_features(past, name_to_id, num_nodes)
        src = name_to_id.loc[past["nameOrig"].values].to_numpy()
        dst = name_to_id.loc[past["nameDest"].values].to_numpy()
        data = Data(
            x=x,
            edge_index=torch.from_numpy(np.stack([src, dst])).to(torch.int64),
            edge_attr=torch.from_numpy(
                past[EDGE_FEATURE_COLS].to_numpy()
            ).to(torch.float32),
            edge_step=torch.from_numpy(past["step"].to_numpy()).to(torch.int16),
            edge_isFraud=torch.from_numpy(past["isFraud"].to_numpy()).to(torch.int8),
        )
        data.num_nodes = num_nodes

        positives = _mules(window, name_to_id, num_nodes)
        active = _active_nodes(window, name_to_id, num_nodes)
        if label_start > 0:
            # Accounts already known to be mules before the window are not
            # prediction targets (they'd be blocked in deployment).
            already = _mules(df[df["step"] <= label_start], name_to_id, num_nodes)
            eval_mask = active & ~already
        else:
            eval_mask = active
        data.y = (positives & eval_mask).to(torch.int8)
        data.eval_mask = eval_mask
        # The sampler expects train_mask on the training snapshot.
        if name == "train":
            data.train_mask = eval_mask

        snapshots[name] = data
        stats[name] = {
            "feature_horizon": feat_end,
            "label_window": [label_start + 1, label_end],
            "edges": int(data.edge_index.shape[1]),
            "eval_nodes": int(eval_mask.sum()),
            "positives": int(data.y.sum()),
        }

    return snapshots, stats
