"""Stage 1 — Baseline GraphSAGE.

Vanilla 2-layer GraphSAGE (mean aggregator, node features only, no edge
features). Establishes the "before" number for the ablation study.

Stage 1 (this) -> add Edge-MLP attention = Stage 2 -> add Imbalance Sampler
+ Focal Loss = Stage 3 (full system).

The model is intentionally simple: every line of complexity must justify
itself in the ablation table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BaselineGraphSAGE(nn.Module):
    """Two-layer GraphSAGE for binary node classification.

    Forward signature matches PyG conventions: model(x, edge_index) -> logits.

    Parameters
    ----------
    in_dim : int
        Number of node feature columns (5 for our PaySim graph).
    hidden_dim : int
        Width of the hidden layer (default 64 from configs/model_config.yaml).
    dropout : float
        Dropout probability between layers (default 0.3).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [num_nodes, 1]. Apply sigmoid for probability."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)
