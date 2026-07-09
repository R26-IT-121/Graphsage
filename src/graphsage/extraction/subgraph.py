"""Suspicious Subgraph extractor — Novelty 3.

For a trigger transaction flagged by the model, extracts the k=2 hop
neighborhood (the implicated mule ring), identifies the sink account,
classifies the money-flow pattern, assigns forensic roles to every account,
and serializes the result as the `suspicious_subgraph` JSON object defined in
docs/integration/graph_api_contract.md §3.2. The payload is consumed by the
Fusion Engine's LLM to write the Chain-of-Evidence forensic narrative.

Design notes:
- Neighborhood discovery is undirected (money can flow through a ring in
  either direction relative to the trigger), but the serialized edges keep
  their original direction.
- The saved graph tensor does not carry account name strings; the name<->id
  mapping is rebuilt from features.parquet with the exact `pd.unique` ordering
  used by graph_builder.build_paysim_graph, and cached to node_names.npy.
- amount is recovered from amount_log via expm1 (prepare_features uses log1p).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from graphsage.extraction.pattern_classifier import classify_pattern

# Column order of data.edge_attr — must match graph_builder.EDGE_FEATURE_COLS.
EF_AMOUNT_LOG = 0
EF_DRAIN_RATIO = 1
EF_SRC_DRAINED = 2
EF_DST_WAS_EMPTY = 3
EF_TIME_GAP = 4
EF_TYPE_IS_TRANSFER = 5

EDGE_FEATURE_NAMES = [
    "amount_log",
    "drain_ratio",
    "src_drained",
    "dst_was_empty",
    "time_gap",
    "type_is_transfer",
]

# data.x columns (graph_builder.NODE_FEATURE_NAMES)
NF_IN_DEGREE = 0
NF_OUT_DEGREE = 1


def load_node_names(
    parquet_path: str | Path,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Rebuild the node-id -> account-name array used at graph build time.

    Must produce the identical ordering to graph_builder.build_paysim_graph:
    pd.unique over the raveled [nameOrig, nameDest] columns. The result is
    cached as .npy because the parquet scan takes ~10s.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False)

    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["nameOrig", "nameDest"])
    names = pd.unique(df[["nameOrig", "nameDest"]].values.ravel()).astype(str)

    if cache_path is not None:
        np.save(cache_path, names)
    return names


class SuspiciousSubgraphExtractor:
    """Extracts the forensic mule-ring payload around a trigger transaction.

    Parameters
    ----------
    data : Data
        The full PaySim graph (x, edge_index, edge_attr, y, edge_step).
    node_names : sequence of str
        node id -> PaySim account name, aligned with data.x rows.
    k : int
        Hop count for the neighborhood walk (contract fixes 2 for v0.3).
    risk_threshold : float
        Node probability above which an account counts as a predicted mule.
        Should be the tuned threshold from reports/ablation_tuned.json.
    max_edges : int
        Serialization cap. High-degree hubs can pull thousands of edges into
        a 2-hop ball; we keep the trigger edge plus the highest-attention
        edges so the payload stays within Member 4's latency budget.
    """

    def __init__(
        self,
        data: Data,
        node_names: Sequence[str],
        k: int = 2,
        risk_threshold: float = 0.5,
        max_edges: int = 200,
    ):
        if len(node_names) != data.num_nodes:
            raise ValueError(
                f"node_names has {len(node_names)} entries but graph has "
                f"{data.num_nodes} nodes"
            )
        self.data = data
        self.node_names = np.asarray(node_names)
        self.name_to_id = {name: i for i, name in enumerate(self.node_names)}
        self.k = k
        self.risk_threshold = risk_threshold
        self.max_edges = max_edges

        # Undirected view for neighborhood discovery only.
        self._undirected_edge_index = torch.cat(
            [data.edge_index, data.edge_index.flip(0)], dim=1
        )

    # ------------------------------------------------------------------ #
    # Trigger resolution
    # ------------------------------------------------------------------ #

    def find_trigger_edge(
        self, name_orig: str, name_dest: str, step: int | None = None
    ) -> int | None:
        """Locate the graph edge matching a request transaction.

        Matches on (src, dst) and, when given, the PaySim step. Falls back to
        ignoring the step (same account pair anchors the same ring). Returns
        None if either account is unknown or no edge exists between them.
        """
        src = self.name_to_id.get(name_orig)
        dst = self.name_to_id.get(name_dest)
        if src is None or dst is None:
            return None

        ei = self.data.edge_index
        pair_mask = (ei[0] == src) & (ei[1] == dst)
        if step is not None:
            step_mask = pair_mask & (self.data.edge_step.to(torch.int64) == int(step))
            candidates = step_mask.nonzero(as_tuple=True)[0]
            if candidates.numel() > 0:
                return int(candidates[0].item())
        candidates = pair_mask.nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            return None
        return int(candidates[0].item())

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    def extract(
        self,
        trigger_edge_id: int,
        node_probs: Tensor,
        edge_attention: Tensor,
    ) -> dict:
        """Build the suspicious_subgraph payload around one trigger edge.

        Parameters
        ----------
        trigger_edge_id : int
            Edge id of the transaction that triggered analysis.
        node_probs : Tensor [num_nodes]
            Calibrated mule probabilities from the model (sigmoid of logits).
        edge_attention : Tensor [num_edges]
            Per-edge attention from EdgeEnhancedGraphSAGE.forward_with_attention
            (average the two layers before passing in).
        """
        data = self.data
        trigger_src = int(data.edge_index[0, trigger_edge_id])
        trigger_dst = int(data.edge_index[1, trigger_edge_id])

        # k-hop ball around both endpoints, undirected discovery.
        subset, _, _, _ = k_hop_subgraph(
            torch.tensor([trigger_src, trigger_dst]),
            self.k,
            self._undirected_edge_index,
            num_nodes=data.num_nodes,
        )
        in_subset = torch.zeros(data.num_nodes, dtype=torch.bool)
        in_subset[subset] = True

        # Original directed edges with both endpoints inside the ball.
        edge_mask = in_subset[data.edge_index[0]] & in_subset[data.edge_index[1]]
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]

        # Cap payload size: always keep the trigger edge, then highest attention.
        if edge_ids.numel() > self.max_edges:
            attn = edge_attention[edge_ids]
            order = torch.argsort(attn, descending=True)
            keep = edge_ids[order[: self.max_edges]]
            if trigger_edge_id not in keep:
                keep = torch.cat([keep[:-1], torch.tensor([trigger_edge_id])])
            edge_ids = keep
        # Restrict nodes to those actually touching a serialized edge.
        node_ids = torch.unique(data.edge_index[:, edge_ids])

        return self._serialize(
            node_ids, edge_ids, trigger_edge_id, node_probs, edge_attention
        )

    # ------------------------------------------------------------------ #
    # Serialization internals
    # ------------------------------------------------------------------ #

    def _serialize(
        self,
        node_ids: Tensor,
        edge_ids: Tensor,
        trigger_edge_id: int,
        node_probs: Tensor,
        edge_attention: Tensor,
    ) -> dict:
        data = self.data
        node_list = node_ids.tolist()
        node_pos = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        src = data.edge_index[0, edge_ids]
        dst = data.edge_index[1, edge_ids]
        steps = data.edge_step[edge_ids].to(torch.int64)
        attrs = data.edge_attr[edge_ids]
        amounts = torch.expm1(attrs[:, EF_AMOUNT_LOG]).clamp(min=0)
        is_fraud_edge = data.edge_isFraud[edge_ids].to(torch.bool)

        # Per-node aggregates within the subgraph.
        in_deg = np.zeros(n, dtype=int)
        out_deg = np.zeros(n, dtype=int)
        first_seen = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        last_seen = np.zeros(n, dtype=np.int64)
        fraud_recv = np.zeros(n, dtype=int)
        total_recv = np.zeros(n, dtype=float)
        for j in range(len(edge_ids)):
            s, d = node_pos[int(src[j])], node_pos[int(dst[j])]
            step = int(steps[j])
            out_deg[s] += 1
            in_deg[d] += 1
            for p in (s, d):
                first_seen[p] = min(first_seen[p], step)
                last_seen[p] = max(last_seen[p], step)
            total_recv[d] += float(amounts[j])
            if bool(is_fraud_edge[j]):
                fraud_recv[d] += 1

        probs = node_probs[node_ids].detach().cpu().numpy().astype(float)
        predicted_mule = probs >= self.risk_threshold
        mules_in_subgraph = int(predicted_mule.sum())

        trigger_src = int(data.edge_index[0, trigger_edge_id])
        trigger_dst = int(data.edge_index[1, trigger_edge_id])

        # Sink: predicted mules first, then in-degree, then probability.
        rank = predicted_mule.astype(float) * 1e6 + in_deg * 1e3 + probs
        sink_pos = int(np.argmax(rank))
        sink_id = node_list[sink_pos]

        # Roles. Fresh sender = no inbound history anywhere in the full graph
        # (global in_degree 0) and few global sends.
        global_in = data.x[node_ids, NF_IN_DEGREE].numpy()
        global_out = data.x[node_ids, NF_OUT_DEGREE].numpy()
        roles = []
        for i, nid in enumerate(node_list):
            if nid == sink_id:
                roles.append("MULE_CENTRAL")
            elif in_deg[i] > 0 and out_deg[i] > 0:
                roles.append("RELAY")
            elif out_deg[i] > 0 and global_in[i] == 0 and global_out[i] <= 3:
                roles.append("FRESH_SENDER")
            elif nid in (trigger_src, trigger_dst):
                roles.append("TRIGGER_PARTICIPANT")
            else:
                roles.append("LEGITIMATE")

        # Structural evidence, computed around the sink.
        sender_mask = dst == sink_id
        sender_positions = [node_pos[int(s)] for s in src[sender_mask].tolist()]
        distinct_senders = sorted(set(sender_positions))
        convergence_count = len(distinct_senders)
        fresh_sender_ratio = (
            float(np.mean([roles[p] == "FRESH_SENDER" for p in distinct_senders]))
            if distinct_senders
            else 0.0
        )
        mean_drain_ratio = (
            float(attrs[:, EF_DRAIN_RATIO].mean()) if len(edge_ids) else 0.0
        )

        # FATF typology scoring (pattern_classifier.py) — needs the subgraph
        # in local indexing with the sink as the flagged node.
        local_ei = torch.stack(
            [
                torch.tensor([node_pos[int(s)] for s in src.tolist()]),
                torch.tensor([node_pos[int(d)] for d in dst.tolist()]),
            ]
        )
        pattern_result = classify_pattern(
            sub_edge_index=local_ei,
            sub_edge_attr=attrs,
            flagged_local=sink_pos,
            num_sub_nodes=n,
        )

        nodes_json = [
            {
                "account_id": str(self.node_names[nid]),
                "role": roles[i],
                "node_risk_score": round(float(probs[i]), 4),
                "in_degree": int(in_deg[i]),
                "out_degree": int(out_deg[i]),
                "first_seen_step": int(first_seen[i]),
                "last_seen_step": int(last_seen[i]),
                "fraud_count_received": int(fraud_recv[i]),
                "total_received_amount": round(total_recv[i], 2),
            }
            for i, nid in enumerate(node_list)
        ]

        edges_json = []
        for j, eid in enumerate(edge_ids.tolist()):
            a = attrs[j]
            edges_json.append(
                {
                    "src": str(self.node_names[int(src[j])]),
                    "dst": str(self.node_names[int(dst[j])]),
                    "amount": round(float(amounts[j]), 2),
                    "step": int(steps[j]),
                    "edge_attention_weight": round(float(edge_attention[eid]), 4),
                    "edge_features": {
                        name: round(float(a[idx]), 4)
                        for idx, name in enumerate(EDGE_FEATURE_NAMES)
                    },
                    "is_trigger_edge": eid == trigger_edge_id,
                    "is_fraud_predicted": bool(predicted_mule[node_pos[int(dst[j])]]),
                }
            )

        return {
            "k_hop": self.k,
            "node_count": n,
            "edge_count": len(edges_json),
            "nodes": nodes_json,
            "edges": edges_json,
            "sink_account": str(self.node_names[sink_id]),
            "pattern": pattern_result.pattern,
            "pattern_confidence": round(pattern_result.confidence, 2),
            # Additive contract extension (v0.3.x): per-pattern score breakdown
            # for Member 4's RAG prompt; existing consumers can ignore it.
            "pattern_scores": pattern_result.scores,
            "structural_evidence": {
                "convergence_count": convergence_count,
                "fresh_sender_ratio": round(fresh_sender_ratio, 4),
                "mean_drain_ratio": round(mean_drain_ratio, 4),
                "mules_in_subgraph": mules_in_subgraph,
            },
        }

