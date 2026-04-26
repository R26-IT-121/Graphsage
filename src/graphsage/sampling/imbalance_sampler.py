"""GraphAwareImbalanceSampler — Novelty 2.

For every known fraud node, extracts its intact k-hop subgraph using
torch_geometric.utils.k_hop_subgraph, then mines hard negatives — topologically
similar but legitimate corporate accounts. Replaces SMOTE, which destroys
network shape under severe imbalance (773:1 in PaySim).
"""
