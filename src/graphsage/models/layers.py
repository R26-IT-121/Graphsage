"""EdgeEnhancedSAGEConv — Novelty 1 of the research.

Custom GraphSAGE convolution that injects a small MLP into the message-passing
step to compute a dynamic attention weight per edge from transaction features,
instead of treating all edges equally. This is the architectural fix for the
"Underutilization of Edge Features in Message Passing" gap (proposal Section 1.3).

Standard SAGEConv:
    h_i = W_self * x_i + W_neigh * MEAN(x_j for j in N(i))

This layer (Edge-Enhanced):
    edge_weight_ij = sigmoid(EdgeMLP(edge_features_ij))
    h_i = W_self * x_i + W_neigh * SUM(edge_weight_ij * x_j for j in N(i))

The aggregator becomes a weighted sum where suspicious edges (high MLP score)
dominate the message; routine edges contribute little.

For the Suspicious Subgraph extractor (Novelty 3), the per-edge attention
weights are also exposed via `forward(..., return_attention=True)`. They become
the `edge_attention_weight` field in the JSON output to Member 4's Fusion Engine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


class EdgeEnhancedSAGEConv(MessagePassing):
    """GraphSAGE convolution with MLP-based edge attention.

    Parameters
    ----------
    in_channels : int
        Number of input node feature channels.
    out_channels : int
        Number of output node feature channels.
    edge_dim : int
        Number of edge feature columns (6 for our PaySim graph).
    edge_mlp_hidden : int
        Hidden width of the edge attention MLP. Default 32 (per
        configs/model_config.yaml).
    bias : bool
        Whether the linear projections have a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        edge_mlp_hidden: int = 32,
        bias: bool = True,
    ):
        # aggr='add' because we'll be doing a WEIGHTED SUM, not a mean.
        # Mean would normalise away the attention; sum lets attention amplify.
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        # Two separate linear maps, GraphSAGE-style: self vs. neighbor.
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neighbor = nn.Linear(in_channels, out_channels, bias=bias)

        # Novelty 1: per-edge attention MLP.
        # Takes 6 edge features -> hidden -> 1 attention scalar.
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_self.reset_parameters()
        self.lin_neighbor.reset_parameters()
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply edge-enhanced convolution.

        Parameters
        ----------
        x : Tensor [num_nodes, in_channels]
            Node feature matrix.
        edge_index : Tensor [2, num_edges]
            Source and destination indices of edges.
        edge_attr : Tensor [num_edges, edge_dim]
            Per-edge feature matrix.
        return_attention : bool
            If True, also return the per-edge attention weights for downstream
            extraction (used by Novelty 3's Suspicious Subgraph JSON).
        """
        # Compute per-edge attention scalars in [0, 1] via sigmoid.
        # Shape: [num_edges, 1]. We squeeze later in `message`.
        edge_logit = self.edge_mlp(edge_attr)
        edge_weight = torch.sigmoid(edge_logit)

        # Self transformation (W_self * x_i)
        out_self = self.lin_self(x)

        # Aggregate neighbors with attention weights, then transform.
        # propagate dispatches to message() then aggregates with self.aggr ('add').
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out_neigh = self.lin_neighbor(agg)

        out = out_self + out_neigh

        if return_attention:
            return out, edge_weight.squeeze(-1)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """Message function: scale neighbor features by their edge attention.

        x_j has shape [num_edges, in_channels]: features of source nodes.
        edge_weight has shape [num_edges, 1]: per-edge attention weight.
        Broadcasting multiplies each neighbor feature by its edge weight.
        """
        return x_j * edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels} -> {self.out_channels}, "
            f"edge_dim={self.edge_dim})"
        )
