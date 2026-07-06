"""Inference engine behind /api/graph/analyze.

Loads the graph, node names, and the Stage 3 checkpoint once, runs a single
full-graph forward pass with attention, and caches node probabilities and
edge attention to disk (the graph is static, so scores never change between
requests). Per-request work is then only: resolve the trigger edge, extract
the k=2 suspicious subgraph, and serialize — which is what keeps p95 under
the 500 ms budget (NFR1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from graphsage.extraction.subgraph import SuspiciousSubgraphExtractor, load_node_names
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE

MODEL_VERSION = "graphsage-edge-mlp-v0.3.0"
STAGE_NAME = "stage_3_full"

# Fraud exists only in TRANSFER and CASH_OUT (EDA report §3).
APPLICABLE_TYPES = {"TRANSFER", "CASH_OUT"}


class GraphPredictor:
    """Owns all model state; constructed once at API startup."""

    def __init__(
        self,
        repo_root: str | Path,
        max_subgraph_edges: int = 200,
        device: str = "cpu",
    ):
        repo_root = Path(repo_root)
        graph_path = repo_root / "data" / "graph" / "paysim_graph.pt"
        ckpt_path = repo_root / "checkpoints" / "stage3_full.pt"
        parquet_path = repo_root / "data" / "processed" / "features.parquet"
        names_cache = repo_root / "data" / "graph" / "node_names.npy"
        scores_cache = repo_root / "data" / "graph" / "inference_cache.pt"
        ablation_path = repo_root / "reports" / "ablation_tuned.json"

        t0 = time.time()
        self.data = torch.load(graph_path, weights_only=False, map_location=device)
        self.node_names = load_node_names(parquet_path, cache_path=names_cache)
        self.graph_version = f"paysim_steps_{int(self.data.edge_step.min())}-{int(self.data.edge_step.max())}"

        self.threshold = 0.5
        if ablation_path.exists():
            report = json.loads(ablation_path.read_text())
            self.threshold = float(
                report.get("stage_3", {}).get("best_threshold", 0.5)
            )

        raw_probs, self.edge_attention = self._load_or_compute_scores(
            ckpt_path, scores_cache, device
        )

        # Focal Loss (alpha=0.95) inflates raw sigmoid outputs — mule and
        # non-mule raw probabilities overlap heavily even though the RANKING
        # is good (AUROC 0.94). Calibrate by percentile rank (ECDF) over
        # receiving accounts, the candidate-mule population. This is a
        # monotone transform: ordering, AUROC, and the tuned decision set are
        # all preserved; only the [0,1] scale becomes interpretable, which is
        # what contract §5's risk_level bands assume ("calibrated
        # post-training").
        receivers = self.data.x[:, 0] > 0  # in_degree > 0
        reference = torch.sort(raw_probs[receivers]).values
        self.raw_probs = raw_probs
        self.probs = torch.searchsorted(reference, raw_probs.contiguous()).to(
            torch.float32
        ) / len(reference)
        self.raw_threshold = self.threshold
        self.threshold = float(
            torch.searchsorted(reference, torch.tensor(self.threshold)) / len(reference)
        )

        self.extractor = SuspiciousSubgraphExtractor(
            self.data,
            self.node_names,
            k=2,
            risk_threshold=self.threshold,
            max_edges=max_subgraph_edges,
        )
        self.startup_seconds = time.time() - t0

    def _load_or_compute_scores(
        self, ckpt_path: Path, cache_path: Path, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ckpt_mtime = ckpt_path.stat().st_mtime
        if cache_path.exists():
            cache = torch.load(cache_path, weights_only=True, map_location=device)
            if cache.get("ckpt_mtime") == ckpt_mtime:
                return cache["probs"], cache["edge_attention"]

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        hp = ckpt.get("hyperparameters", {})
        model = EdgeEnhancedGraphSAGE(
            in_dim=self.data.x.shape[1],
            edge_dim=self.data.edge_attr.shape[1],
            hidden_dim=hp.get("hidden_dim", 64),
            edge_mlp_hidden=hp.get("edge_mlp_hidden", 32),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            logits, attentions = model.forward_with_attention(
                self.data.x, self.data.edge_index, self.data.edge_attr
            )
            probs = torch.sigmoid(logits)
            edge_attention = torch.stack(attentions).mean(dim=0)

        torch.save(
            {"probs": probs, "edge_attention": edge_attention, "ckpt_mtime": ckpt_mtime},
            cache_path,
        )
        return probs, edge_attention

    # ------------------------------------------------------------------ #

    def is_applicable(self, txn_type: str) -> bool:
        return txn_type in APPLICABLE_TYPES

    def analyze(self, name_orig: str, name_dest: str, step: int | None) -> dict | None:
        """Score a transaction and extract its suspicious subgraph.

        Returns None when no matching edge exists in the graph (the service
        maps that to an error response per contract §4). The risk score is
        the calibrated (percentile-rank) mule score of the receiving account
        — the persistent structural element of PaySim fraud (EDA §7/§9);
        senders are disposable one-shot accounts.
        """
        trigger = self.extractor.find_trigger_edge(name_orig, name_dest, step)
        if trigger is None:
            return None

        dst_id = self.extractor.name_to_id[name_dest]
        score = float(self.probs[dst_id])
        subgraph = self.extractor.extract(trigger, self.probs, self.edge_attention)
        return {
            "relational_risk_score": round(score, 4),
            "confidence": round(max(score, 1.0 - score), 4),
            "suspicious_subgraph": subgraph,
        }
