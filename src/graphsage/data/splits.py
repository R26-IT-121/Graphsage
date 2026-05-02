"""Time-based train / val / test split for the PaySim graph.

Splits NODES (not edges) chronologically by the earliest step at which they
appear as either sender or receiver.

Why time-based, not random:
    Real fraud detection deploys models that predict future fraud from past
    transactions. Random splits leak future information into training and
    inflate F1. Time-based splits mirror deployment and produce honest
    metrics that survive panel scrutiny.

Default split (PaySim 743 steps = ~31 days):
    train: step 1 - 600    (~80 percent, 25 days)
    val:   step 601 - 700  (~13 percent, 4 days)
    test:  step 701 - 743  (~6 percent, 2 days)

Usage:
    from graphsage.data.splits import make_time_split
    data = torch.load("data/graph/paysim_graph.pt")
    data = make_time_split(data, train_end=600, val_end=700)
    # data.train_mask, data.val_mask, data.test_mask now populated
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class SplitStats:
    train_nodes: int
    val_nodes: int
    test_nodes: int
    train_mules: int
    val_mules: int
    test_mules: int

    def __str__(self) -> str:
        total = self.train_nodes + self.val_nodes + self.test_nodes
        return (
            f"SplitStats(\n"
            f"  train: {self.train_nodes:>10,} nodes  ({self.train_nodes/total*100:.2f}%)  "
            f"with {self.train_mules:>5,} mules\n"
            f"  val:   {self.val_nodes:>10,} nodes  ({self.val_nodes/total*100:.2f}%)  "
            f"with {self.val_mules:>5,} mules\n"
            f"  test:  {self.test_nodes:>10,} nodes  ({self.test_nodes/total*100:.2f}%)  "
            f"with {self.test_mules:>5,} mules\n"
            f")"
        )


def make_time_split(
    data: Data,
    train_end: int = 600,
    val_end: int = 700,
) -> tuple[Data, SplitStats]:
    """Add train_mask, val_mask, test_mask, first_step to the graph.

    A node belongs to the split that contains its EARLIEST incident edge
    (whether as sender or receiver):
        train if first_step <= train_end
        val   if train_end < first_step <= val_end
        test  if first_step > val_end
    """
    if not (1 <= train_end < val_end <= int(data.edge_step.max().item())):
        raise ValueError(
            f"Invalid split bounds: train_end={train_end}, val_end={val_end}, "
            f"max_step={int(data.edge_step.max().item())}"
        )

    num_nodes = data.num_nodes

    # Combine src and dst endpoints with their corresponding step
    all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
    edge_step_f = data.edge_step.to(torch.float32)
    all_steps = torch.cat([edge_step_f, edge_step_f])

    # For each node, take the minimum step across all its incident edges.
    # Initial value of +inf so any real step replaces it.
    first_step = torch.full((num_nodes,), float("inf"), dtype=torch.float32)
    first_step.scatter_reduce_(0, all_nodes, all_steps, reduce="amin", include_self=True)

    # Sanity: no node should be left at +inf (every node has at least one edge)
    if torch.isinf(first_step).any():
        n_orphan = int(torch.isinf(first_step).sum().item())
        raise RuntimeError(f"{n_orphan} nodes have no incident edge (graph is malformed)")

    # Boolean masks
    train_mask = first_step <= train_end
    val_mask = (first_step > train_end) & (first_step <= val_end)
    test_mask = first_step > val_end

    # Persist as int16 (fits 1-743) to save space vs float32
    data.first_step = first_step.to(torch.int16)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Sanity stats
    y = data.y.to(torch.bool)
    stats = SplitStats(
        train_nodes=int(train_mask.sum().item()),
        val_nodes=int(val_mask.sum().item()),
        test_nodes=int(test_mask.sum().item()),
        train_mules=int((train_mask & y).sum().item()),
        val_mules=int((val_mask & y).sum().item()),
        test_mules=int((test_mask & y).sum().item()),
    )

    return data, stats
