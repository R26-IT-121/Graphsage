"""EdgeEnhancedSAGEConv — Novelty 1.

Custom GraphSAGE convolution that injects an MLP into the message-passing step to
compute a dynamic attention weight per edge from transaction features, instead of
treating all edges equally. Subclasses torch_geometric.nn.MessagePassing.
"""
