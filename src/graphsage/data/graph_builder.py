"""DataFrame -> torch_geometric.data.Data conversion.

Reads the engineered features parquet (produced by scripts/prepare_features.py)
and assembles a PyG graph object suitable for GraphSAGE training.

Conventions:
- Nodes: every account that appears as nameOrig OR nameDest (~3.27M).
- Edges: directed transfers (~2.77M, TRANSFER + CASH_OUT only).
- Edge features: 6 columns from the parquet (see EDA report Section 11).
- Node features: 5 structural aggregates derivable from edges (no leakage).
- Node labels y: 1 if the account received any fraud transaction (mule),
  else 0. Justification (EDA report Sections 7 + 9): senders are one-shot
  disposable accounts; the persistent structural element is the mule.
- edge_step and edge_isFraud are carried as auxiliaries for the time-based
  split and edge-level metrics later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

EDGE_FEATURE_COLS: list[str] = [
    "amount_log",
    "drain_ratio",
    "src_drained",
    "dst_was_empty",
    "time_gap",
    "type_is_transfer",
]

NODE_FEATURE_NAMES: list[str] = [
    "in_degree",
    "out_degree",
    "mean_in_amount_log",
    "mean_out_amount_log",
    "max_in_amount_log",
]

# v2 behavioural set (May 2026): the five above plus seven mule-signature
# aggregates. Selected with build_paysim_graph(feature_version="v2"); the
# serving path and the shipped checkpoints still use v1.
NODE_FEATURE_NAMES_V2: list[str] = NODE_FEATURE_NAMES + [
    "std_in_amount_log",       # 5  — uniformity → smurfing signal
    "distinct_senders",        # 6  — fan-in distinctness → mule signature
    "distinct_receivers",      # 7  — fan-out distinctness
    "mean_drain_out",          # 8  — outgoing drain → mule clearing inflows
    "mean_dst_was_empty_in",   # 9  — incoming-to-fresh-account ratio
    "txn_velocity",            # 10 — (in+out) / active_steps
    "first_step_norm",         # 11 — normalised first-appearance step
]


@dataclass
class GraphStats:
    num_nodes: int
    num_edges: int
    num_mules: int
    num_fraud_edges: int

    def __str__(self) -> str:
        return (
            f"GraphStats(nodes={self.num_nodes:,}, edges={self.num_edges:,}, "
            f"mules={self.num_mules:,}, fraud_edges={self.num_fraud_edges:,})"
        )


def build_paysim_graph(
    parquet_path: str | Path, feature_version: str = "v1"
) -> tuple[Data, GraphStats]:
    """Build a PyG Data object from the processed features parquet.

    Parameters
    ----------
    feature_version : {"v1", "v2"}
        v1 = 5 structural aggregates (default; matches the shipped checkpoints
        and the serving path). v2 = 12-dim behavioural set, for the feature
        ablation arm.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with x, edge_index, edge_attr, y, edge_step, edge_isFraud.
    stats : GraphStats
        Sanity numbers for logging and metadata.
    """
    if feature_version not in ("v1", "v2"):
        raise ValueError(f"feature_version must be v1 or v2, got {feature_version!r}")
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Features parquet not found at {parquet_path}")

    print(f"[1/6] Loading {parquet_path.name}...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"      {len(df):,} edges loaded in {time.time() - t0:.1f}s")

    # Node ID mapping: account name (string) -> contiguous integer
    print("[2/6] Mapping account names to integer IDs...")
    t0 = time.time()
    all_accounts = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
    name_to_id = pd.Series(np.arange(len(all_accounts)), index=all_accounts)
    num_nodes = int(len(all_accounts))
    print(f"      {num_nodes:,} unique nodes in {time.time() - t0:.1f}s")

    # edge_index [2, num_edges]
    print("[3/6] Building edge_index...")
    src_ids = name_to_id.loc[df["nameOrig"].values].to_numpy()
    dst_ids = name_to_id.loc[df["nameDest"].values].to_numpy()
    edge_index = torch.from_numpy(np.stack([src_ids, dst_ids])).to(torch.int64)
    print(f"      shape={tuple(edge_index.shape)}")

    # edge_attr [num_edges, 6]
    print("[4/6] Building edge_attr (6 features)...")
    edge_attr = torch.from_numpy(df[EDGE_FEATURE_COLS].to_numpy()).to(torch.float32)
    print(f"      shape={tuple(edge_attr.shape)}")

    # Node features x [num_nodes, 5 or 12]
    n_cols = 12 if feature_version == "v2" else 5
    print(f"[5/6] Building node features x ({n_cols} columns, {feature_version})...")
    t0 = time.time()
    out_aggs = {
        "out_degree": ("amount_log", "size"),
        "mean_out": ("amount_log", "mean"),
    }
    in_aggs = {
        "in_degree": ("amount_log", "size"),
        "mean_in": ("amount_log", "mean"),
        "max_in": ("amount_log", "max"),
    }
    if feature_version == "v2":
        out_aggs.update(
            distinct_recv=("nameDest", "nunique"),
            mean_drain_out=("drain_ratio", "mean"),
            first_step_out=("step", "min"),
            last_step_out=("step", "max"),
        )
        in_aggs.update(
            std_in=("amount_log", "std"),
            distinct_send=("nameOrig", "nunique"),
            mean_dst_empty_in=("dst_was_empty", "mean"),
            first_step_in=("step", "min"),
            last_step_in=("step", "max"),
        )
    out_stats = df.groupby("nameOrig", observed=True).agg(**out_aggs)
    in_stats = df.groupby("nameDest", observed=True).agg(**in_aggs)

    x = np.zeros((num_nodes, n_cols), dtype=np.float32)
    out_idx = name_to_id.loc[out_stats.index.values].to_numpy()
    in_idx = name_to_id.loc[in_stats.index.values].to_numpy()
    # Column order must match NODE_FEATURE_NAMES[_V2]
    x[in_idx, 0] = in_stats["in_degree"].to_numpy()
    x[out_idx, 1] = out_stats["out_degree"].to_numpy()
    x[in_idx, 2] = in_stats["mean_in"].to_numpy()
    x[out_idx, 3] = out_stats["mean_out"].to_numpy()
    x[in_idx, 4] = in_stats["max_in"].to_numpy()

    if feature_version == "v2":
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
        x[:, 10] = np.where(
            np.isfinite(active_steps), (x[:, 0] + x[:, 1]) / active_steps, 0.0
        )
        max_step = float(df["step"].max())
        x[:, 11] = np.where(
            np.isfinite(first_step), first_step / max(max_step, 1.0), -1.0
        )
        # log1p count-like columns so degrees don't drown out ratios.
        for col in (0, 1, 6, 7, 10):
            x[:, col] = np.log1p(x[:, col])

    x_t = torch.from_numpy(x)
    print(f"      shape={tuple(x_t.shape)} (built in {time.time() - t0:.1f}s)")

    # Node labels y (mule = received any fraud)
    print("[6/6] Building node labels y (mules)...")
    fraud_dst_names = df.loc[df["isFraud"] == 1, "nameDest"].unique()
    fraud_dst_ids = name_to_id.loc[fraud_dst_names].to_numpy()
    y = torch.zeros(num_nodes, dtype=torch.int8)
    y[fraud_dst_ids] = 1
    num_mules = int(y.sum().item())
    print(f"      {num_mules:,} mule nodes ({num_mules / num_nodes * 100:.4f}%)")

    # Auxiliaries for time-based split + edge-level metrics
    edge_step = torch.from_numpy(df["step"].to_numpy()).to(torch.int16)
    edge_isFraud = torch.from_numpy(df["isFraud"].to_numpy()).to(torch.int8)

    data = Data(
        x=x_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_step=edge_step,
        edge_isFraud=edge_isFraud,
    )

    stats = GraphStats(
        num_nodes=num_nodes,
        num_edges=int(edge_index.shape[1]),
        num_mules=num_mules,
        num_fraud_edges=int(edge_isFraud.sum().item()),
    )

    return data, stats
