"""Inference engine behind /api/graph/analyze.

Loads a serving bundle produced by scripts/export_serving_bundle.py — graph
tensors plus precomputed, isotonic-calibrated node scores and per-edge
attention. Nothing is inferred at request time and no model is instantiated at
all, so startup is a single file read and per-request work is only: resolve
the trigger edge, extract the k=2 subgraph, serialize. That is what keeps p95
under the 500 ms budget (NFR1) and lets the demo run on a laptop.

The bundle comes from the leakage-free temporal protocol (features and message
passing restricted to past edges), so the scores served here are the same ones
reported in reports/temporal/statistics.json — the demo and the dissertation
cannot drift apart.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from graphsage.extraction.subgraph import SuspiciousSubgraphExtractor, load_node_names

MODEL_VERSION = "graphsage-temporal-v0.4.0"

# Fraud exists only in TRANSFER and CASH_OUT (EDA report §3).
APPLICABLE_TYPES = {"TRANSFER", "CASH_OUT"}

BUNDLE_HELP = (
    "Build it in Colab with scripts/export_serving_bundle.py and copy it to "
    "data/graph/serving_bundle.pt"
)


class GraphPredictor:
    """Owns all serving state; constructed once at API startup."""

    def __init__(
        self,
        repo_root: str | Path,
        max_subgraph_edges: int = 200,
        device: str = "cpu",
    ):
        repo_root = Path(repo_root)
        bundle_path = repo_root / "data" / "graph" / "serving_bundle.pt"
        parquet_path = repo_root / "data" / "processed" / "features.parquet"
        names_cache = repo_root / "data" / "graph" / "node_names.npy"

        if not bundle_path.exists():
            raise FileNotFoundError(f"{bundle_path} not found. {BUNDLE_HELP}")

        t0 = time.time()
        bundle = torch.load(bundle_path, weights_only=False, map_location=device)
        self.meta = bundle["meta"]
        self.stage = f"stage_{self.meta['stage']}_{self.meta['features']}"
        self.threshold = float(bundle["threshold"])
        self.probs = bundle["node_scores"]
        self.edge_attention = bundle["edge_attention"]

        # The extractor only reads x[:, 0] (in_degree) and x[:, 1] (out_degree),
        # so the bundle ships those two columns rather than the full matrix.
        self.data = Data(
            x=bundle["node_degrees"],
            edge_index=bundle["edge_index"],
            edge_attr=bundle["edge_attr"],
            edge_step=bundle["edge_step"],
            edge_isFraud=bundle["edge_isFraud"],
        )
        self.data.num_nodes = int(bundle["num_nodes"])

        # Risk bands (contract §5). The fixed 0.25/0.5/0.75/0.9 cutoffs in
        # model_config.yaml were placeholders written for raw sigmoid output.
        # Isotonic-calibrated scores are true probabilities on a 4.7%-positive
        # population, so they top out well below 0.9 and those cutoffs would
        # never fire. Anchor the bands to quantities that mean something here:
        # HIGH is exactly the val-tuned decision threshold (the flag/no-flag
        # boundary we defend), CRITICAL the extreme tail of scored accounts.
        receivers = self.data.x[:, 0] > 0  # in_degree > 0
        scored = self.probs[receivers]
        self.risk_bands = {
            "medium": 0.5 * self.threshold,
            "high": self.threshold,
            "critical": max(
                float(torch.quantile(scored.float(), 0.995)), self.threshold
            ),
        }

        lo, hi = self.meta["step_range"]
        self.graph_version = f"paysim_steps_{lo}-{hi}_{self.meta['protocol']}"
        self.node_names = load_node_names(parquet_path, cache_path=names_cache)

        self.extractor = SuspiciousSubgraphExtractor(
            self.data,
            self.node_names,
            k=2,
            risk_threshold=self.threshold,
            max_edges=max_subgraph_edges,
        )
        # ── Live inductive inference (optional) ─────────────────────────
        # When the weights and the full feature matrix are present, the
        # network is loaded and kept in memory so accounts absent from the
        # precomputed table can still be scored — which is the whole point of
        # an inductive model, and what makes this a service rather than a
        # lookup.
        self.live = None
        self.live_error = None
        ckpt = repo_root / "checkpoints" / "temporal_stage3b_v2_seed0.pt"
        feats = repo_root / "data" / "graph" / "node_features_v2.pt"
        if ckpt.exists() and feats.exists():
            try:
                from graphsage.inference.live import LiveModel

                self.live = LiveModel(ckpt, feats, self.data)
            except Exception as exc:                    # noqa: BLE001
                self.live_error = f"{type(exc).__name__}: {exc}"
        else:
            self.live_error = "weights or node_features_v2.pt not present"

        self.startup_seconds = time.time() - t0

    # ------------------------------------------------------------------ #

    def sample_transactions(self, n: int = 20, fraud_ratio: float = 0.08) -> list[dict]:
        """Draw real edges from the graph as contract-shaped transactions.

        The live monitor needs a stream of genuine records rather than invented
        ones: these are actual PaySim transfers between actual accounts, so the
        scores the monitor shows are the model's real output on real input.

        `fraud_ratio` over-samples known-fraud edges relative to the 0.25% base
        rate. Left at the true rate a demo would screen thousands of clean
        transactions before anything happened; the alternative is inventing
        fraud, which is worse. The response says which is which.
        """
        import random

        ei = self.data.edge_index
        n_edges = int(ei.shape[1])
        n_fraud = max(0, min(n, int(round(n * fraud_ratio))))

        fraud_pool = getattr(self, "_fraud_edge_ids", None)
        if fraud_pool is None:
            fraud_pool = (self.data.edge_isFraud == 1).nonzero(as_tuple=True)[0]
            self._fraud_edge_ids = fraud_pool

        picks: list[int] = []
        if n_fraud and fraud_pool.numel():
            picks += [
                int(fraud_pool[random.randrange(fraud_pool.numel())])
                for _ in range(n_fraud)
            ]
        picks += [random.randrange(n_edges) for _ in range(n - len(picks))]
        random.shuffle(picks)

        out = []
        for eid in picks:
            attr = self.data.edge_attr[eid]
            amount = round(float(torch.expm1(attr[0]).clamp(min=0)), 2)
            drain = float(attr[1])
            old_org = round(amount / drain, 2) if drain > 1e-6 else round(amount * 4, 2)
            out.append({
                "transaction_id": f"TX_LIVE_{int(eid)}",
                "step": int(self.data.edge_step[eid]),
                "type": "TRANSFER" if float(attr[5]) >= 0.5 else "CASH_OUT",
                "amount": amount,
                "nameOrig": str(self.node_names[int(ei[0, eid])]),
                "nameDest": str(self.node_names[int(ei[1, eid])]),
                "oldbalanceOrg": old_org,
                "newbalanceOrig": round(max(old_org - amount, 0.0), 2),
                "oldbalanceDest": 0.0,
                "newbalanceDest": amount,
                "isFlaggedFraud": 0,
                # Ground truth, for measuring the monitor — never shown as a
                # model output.
                "_is_fraud": bool(self.data.edge_isFraud[eid] == 1),
            })
        return out

    def is_applicable(self, txn_type: str) -> bool:
        return txn_type in APPLICABLE_TYPES

    def analyze(self, name_orig: str, name_dest: str, step: int | None) -> dict | None:
        """Score a transaction and extract its suspicious subgraph.

        Returns None when no matching edge exists in the graph (the service
        maps that to an error response per contract §4). The risk score is the
        calibrated mule probability of the RECEIVING account — the persistent
        structural element of PaySim fraud (EDA §7/§9); senders are disposable
        one-shot accounts.
        """
        trigger = self.extractor.find_trigger_edge(name_orig, name_dest, step)
        if trigger is None:
            return None

        dst_id = self.extractor.name_to_id[name_dest]
        score = float(self.probs[dst_id])
        source = "precomputed"

        # An account with no precomputed score is one the snapshot never saw.
        # Rather than refuse, run the network over its neighbourhood.
        if self.live is not None and score <= 0.0:
            try:
                raw, _ms = self.live.score_node(dst_id)
                score = raw
                source = "live_inference"
            except Exception:                           # noqa: BLE001
                pass
        subgraph = self.extractor.extract(trigger, self.probs, self.edge_attention)
        return {
            "relational_risk_score": round(score, 4),
            "confidence": round(max(score, 1.0 - score), 4),
            "score_source": source,
            "suspicious_subgraph": subgraph,
        }
