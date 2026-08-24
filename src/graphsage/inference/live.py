"""Live inductive inference — the trained network, loaded and running.

The default serving path answers from precomputed scores, which is correct for
a fixed historical snapshot but cannot score an account it has never seen. This
module loads the actual weights and runs a forward pass per request over the
k-hop neighbourhood of the transaction.

Why a subgraph and not the whole graph: a full-graph forward over 3.3M nodes is
what exhausted memory during development. GraphSAGE is inductive precisely so
it does not need one — an account's embedding depends only on its k-hop
neighbourhood, so scoring one transaction means propagating over a few hundred
nodes. That is the architecture's central claim, and this is what demonstrates
it rather than asserting it.

Scores stay comparable to the precomputed path because the same isotonic
calibrator is applied afterwards.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE

logger = logging.getLogger(__name__)


class LiveModel:
    """Holds the weights in memory and scores nodes on demand."""

    def __init__(
        self,
        checkpoint: str | Path,
        node_features: str | Path,
        data,
        k: int = 2,
        max_nodes: int = 4000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.data = data
        self.k = k
        # A hub can pull an enormous neighbourhood; cap it so one pathological
        # account cannot stall the request loop.
        self.max_nodes = max_nodes
        self._lock = threading.Lock()

        ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
        self.meta = {
            "stage": ckpt.get("stage"),
            "features": ckpt.get("features"),
            "seed": ckpt.get("seed"),
            "protocol": ckpt.get("protocol"),
        }
        hp = ckpt.get("hyperparameters", {})

        self.x = torch.load(node_features, weights_only=True, map_location="cpu").float()
        self.model = EdgeEnhancedGraphSAGE(
            in_dim=self.x.shape[1],
            edge_dim=int(data.edge_attr.shape[1]),
            hidden_dim=int(hp.get("hidden_dim", 64)),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.loaded_at = time.time()
        self.inferences = 0
        self.total_ms = 0.0
        self.params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Live model ready: {self.params:,} params, in_dim={self.x.shape[1]}"
        )

    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> dict:
        return {
            "loaded": True,
            "parameters": self.params,
            "in_dim": int(self.x.shape[1]),
            "k_hop": self.k,
            "inferences": self.inferences,
            "mean_latency_ms": round(self.total_ms / self.inferences, 1) if self.inferences else None,
            "uptime_seconds": round(time.time() - self.loaded_at, 1),
            **self.meta,
        }

    def score_node(self, node_id: int) -> tuple[float, float]:
        """Return (raw_probability, latency_ms) for one account.

        Runs a genuine forward pass. Thread-locked because a single eval-mode
        model is not safe to run concurrently from multiple request threads.
        """
        t0 = time.time()
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            int(node_id),
            self.k,
            self.data.edge_index,
            relabel_nodes=True,
            num_nodes=int(self.data.num_nodes),
        )

        if subset.numel() > self.max_nodes:
            # Keep the target plus a bounded sample of its neighbourhood.
            keep = torch.cat([mapping, torch.randperm(subset.numel())[: self.max_nodes]])
            keep = torch.unique(keep)
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                int(node_id), 1, self.data.edge_index,
                relabel_nodes=True, num_nodes=int(self.data.num_nodes),
            )

        with self._lock, torch.no_grad():
            logits = self.model(
                self.x[subset].to(self.device),
                edge_index.to(self.device),
                self.data.edge_attr[edge_mask].to(self.device),
            )
            raw = float(torch.sigmoid(logits[mapping]).item())

        ms = (time.time() - t0) * 1000
        self.inferences += 1
        self.total_ms += ms
        return raw, ms
