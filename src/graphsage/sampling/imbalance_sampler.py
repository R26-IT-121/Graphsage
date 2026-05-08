"""GraphAwareImbalanceSampler — Novelty 2 of the research.

For every training step, this sampler builds a balanced mini-batch by:
  1. Drawing N_pos fraud nodes uniformly from the training-mule pool.
  2. Drawing N_neg legit nodes via Hard Negative Mining: prefer legit nodes
     that are TOPOLOGICALLY similar to fraud (high in-degree, large total
     received amount) to make the model discriminate fraud from "look-alike"
     mules.
  3. Extracting the union of their k-hop subgraphs intact (PyG's
     k_hop_subgraph), preserving the hub-and-spoke shape that EDA Section 9
     identified as the actual fraud topology in PaySim.

The seed nodes are the labeled prediction targets; surrounding context nodes
provide message-passing context but contribute no loss. This is a strict
upgrade over SMOTE-style oversampling, which destroys topology by injecting
synthetic disconnected nodes.

Replaces full-batch training under severe imbalance, where the aggregate
weight of 3.2M negatives dominates the gradient regardless of per-example
class weighting (empirically demonstrated by Stage 3a's failure to converge).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


@dataclass
class SubgraphBatch:
    """One mini-batch produced by the sampler."""
    x: Tensor                  # Node features for the subgraph nodes
    edge_index: Tensor         # Relabeled edge_index inside the subgraph
    edge_attr: Tensor          # Edge features for the subgraph edges
    y: Tensor                  # Labels for SEED nodes only (length n_seeds)
    seed_local_idx: Tensor     # Positions of seed nodes inside the relabeled subgraph
    n_pos: int
    n_neg: int


class GraphAwareImbalanceSampler:
    """Balanced k-hop subgraph sampler with optional hard-negative mining.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The full graph. Must have x, edge_index, edge_attr, y, train_mask.
    k_hop : int
        Number of hops for subgraph extraction (default 2).
    pos_per_batch : int
        Number of fraud seed nodes per batch (default 64).
    neg_per_batch : int
        Number of legit seed nodes per batch (default 64).
    hard_negative_ratio : float
        Fraction of legit seeds chosen as hard negatives (high in-degree).
        The rest are sampled uniformly. 1.0 = all hard, 0.0 = all uniform.
        Default 0.5.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        data: Data,
        k_hop: int = 2,
        pos_per_batch: int = 64,
        neg_per_batch: int = 64,
        hard_negative_ratio: float = 0.5,
        seed: int = 42,
    ):
        if not hasattr(data, "train_mask"):
            raise ValueError("data must have train_mask. Run scripts/make_splits.py first.")

        self.data = data
        self.k_hop = int(k_hop)
        self.pos_per_batch = int(pos_per_batch)
        self.neg_per_batch = int(neg_per_batch)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.generator = torch.Generator().manual_seed(seed)

        # Train fraud and train legit pools (CPU index tensors)
        y_bool = data.y.to(torch.bool)
        train = data.train_mask
        self.train_pos_idx = (train & y_bool).nonzero(as_tuple=True)[0].cpu()
        self.train_neg_idx = (train & ~y_bool).nonzero(as_tuple=True)[0].cpu()

        if len(self.train_pos_idx) < pos_per_batch:
            raise ValueError(
                f"Only {len(self.train_pos_idx)} train mules available, "
                f"but pos_per_batch={pos_per_batch}"
            )

        # Hard-negative ranking: train legit nodes sorted by in_degree (col 0 of x)
        # descending. The top-K are "hard negatives" — legit nodes that look
        # structurally like mules.
        if hard_negative_ratio > 0:
            in_degree = data.x[:, 0].cpu()  # column 0 is in_degree per graph_builder
            neg_in_deg = in_degree[self.train_neg_idx]
            sorted_local = torch.argsort(neg_in_deg, descending=True)
            self.hard_neg_pool = self.train_neg_idx[sorted_local]
        else:
            self.hard_neg_pool = self.train_neg_idx

    @property
    def num_pos_train(self) -> int:
        return int(self.train_pos_idx.numel())

    @property
    def num_neg_train(self) -> int:
        return int(self.train_neg_idx.numel())

    def steps_per_epoch(self) -> int:
        """How many batches to process to roughly cover all train fraud nodes."""
        return max(1, self.num_pos_train // self.pos_per_batch)

    def sample(self) -> SubgraphBatch:
        """Sample one balanced batch and extract its k-hop subgraph."""
        # ---- Sample positive seeds (uniform from train fraud) ----
        pos_perm = torch.randperm(self.num_pos_train, generator=self.generator)
        pos_seeds = self.train_pos_idx[pos_perm[: self.pos_per_batch]]

        # ---- Sample negative seeds (hard negatives + uniform mix) ----
        n_hard = int(self.neg_per_batch * self.hard_negative_ratio)
        n_uniform = self.neg_per_batch - n_hard

        neg_seeds_list = []
        # Top-pool size for sampling hard negatives (10x batch hard count)
        hard_pool_size = min(len(self.hard_neg_pool), n_hard * 10) if n_hard > 0 else 0
        if n_hard > 0 and hard_pool_size > 0:
            # Pick from the top-K hardest, but with some randomness to avoid
            # always picking the same nodes.
            hard_pick = torch.randperm(hard_pool_size, generator=self.generator)[:n_hard]
            neg_seeds_list.append(self.hard_neg_pool[hard_pick])
        if n_uniform > 0:
            uni_perm = torch.randperm(self.num_neg_train, generator=self.generator)
            neg_seeds_list.append(self.train_neg_idx[uni_perm[:n_uniform]])

        neg_seeds = torch.cat(neg_seeds_list)
        seeds = torch.cat([pos_seeds, neg_seeds])

        # ---- Extract the union k-hop subgraph (PyG built-in) ----
        # subset: original node IDs included in subgraph (sorted)
        # ei_relabeled: edge_index inside subgraph using local IDs (0..len(subset)-1)
        # mapping: positions of `seeds` inside `subset` (length len(seeds))
        # edge_mask: bool mask over the original edges
        subset, ei_relabeled, mapping, edge_mask = k_hop_subgraph(
            seeds,
            num_hops=self.k_hop,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        sub_x = self.data.x[subset]
        sub_edge_attr = self.data.edge_attr[edge_mask]
        sub_y = self.data.y[seeds].float()  # labels for seed nodes only

        return SubgraphBatch(
            x=sub_x,
            edge_index=ei_relabeled,
            edge_attr=sub_edge_attr,
            y=sub_y,
            seed_local_idx=mapping,
            n_pos=int(self.pos_per_batch),
            n_neg=int(self.neg_per_batch),
        )
