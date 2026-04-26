"""Suspicious Subgraph extractor — Novelty 3.

For every node flagged above the risk threshold, runs torch_geometric.utils.k_hop_subgraph
(k=2) to extract the implicated mule ring, identifies the sink account (terminal node),
and serializes nodes + edges + edge weights + pattern label as a JSON payload for the
Fusion Engine (Member 4). Payload schema is defined in graphsage.api.schemas.
"""
