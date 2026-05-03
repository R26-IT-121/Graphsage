"""Stage 2 — Edge-Enhanced GraphSAGE.

Same 2-layer GraphSAGE architecture as Stage 1, but with SAGEConv replaced by
EdgeEnhancedSAGEConv (Novelty 1). The only difference between Stage 1 and
Stage 2 is the convolution layer — this isolates the contribution of edge
attention in the ablation table.

Stage 1 (baseline.py) -> Stage 2 (THIS) -> Stage 3 (Focal Loss + Sampler)

Stage 2 also serves as the architecture for Stage 3 — Stage 3 only changes
the training procedure (loss + sampler), not the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphsage.models.layers import EdgeEnhancedSAGEConv


class EdgeEnhancedGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with edge-attention convolutions.

    Forward signature: model(x, edge_index, edge_attr) -> logits.

    Parameters
    ----------
    in_dim : int
        Number of input node feature columns (5 for our PaySim graph).
    edge_dim : int
        Number of edge feature columns (6 for our PaySim graph).
    hidden_dim : int
        Hidden width (default 64 from configs/model_config.yaml).
    edge_mlp_hidden : int
        Width of the edge attention MLP hidden layer (default 32).
    dropout : float
        Dropout probability between layers (default 0.3).
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        edge_mlp_hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = EdgeEnhancedSAGEConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            edge_mlp_hidden=edge_mlp_hidden,
        )
        self.conv2 = EdgeEnhancedSAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            edge_mlp_hidden=edge_mlp_hidden,
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits of shape [num_nodes]. Apply sigmoid for probability."""
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)

    def forward_with_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward and also return per-layer edge attention weights.

        Used by the Suspicious Subgraph extractor (Novelty 3) to populate
        edge_attention_weight in the JSON output to Member 4's Fusion Engine.

        Returns
        -------
        logits : Tensor [num_nodes]
        attentions : list of two Tensors [num_edges] (conv1, conv2 attention)
        """
        h, attn1 = self.conv1(x, edge_index, edge_attr, return_attention=True)
        h = F.relu(h)
        h, attn2 = self.conv2(h, edge_index, edge_attr, return_attention=True)
        h = F.relu(h)
        logits = self.classifier(h).squeeze(-1)
        return logits, [attn1, attn2]
