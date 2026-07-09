"""FATF typology classifier for Suspicious Subgraphs (Novelty 3 sub-component).

Given a k-hop subgraph around a flagged account, scores four crime patterns and
returns the most likely typology with a confidence in [0, 1] plus a breakdown.

Patterns (from the FATF "Money Laundering & Terrorist Financing Typologies" reports):
    - HUB_AND_SPOKE   — many distinct senders converge into one collector account.
    - SMURFING        — many similar-sized small txns from many sources (structuring).
    - LAYERING        — funds chained through intermediaries to obscure origin.
    - ACCOUNT_TAKEOVER— legitimate account suddenly drained; funds land at a mule.

Pure heuristic / rule-based — no model weights, so deterministic and auditable.
Member 4's RAG prompt cites the same evidence keys this module emits.

Edge-feature column convention (must match graph_builder.py):
    0: amount_log
    1: drain_ratio
    2: src_drained
    3: dst_was_empty
    4: time_gap
    5: type_is_transfer
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatternResult:
    pattern: str
    confidence: float
    scores: dict[str, float]
    evidence: dict[str, float | int]


def _safe_std(t: torch.Tensor) -> float:
    if t.numel() < 2:
        return 0.0
    return float(t.std().item())


def classify_pattern(
    sub_edge_index: torch.Tensor,
    sub_edge_attr: torch.Tensor,
    flagged_local: int,
    num_sub_nodes: int,
) -> PatternResult:
    """Score the four FATF patterns and return the dominant one.

    Parameters
    ----------
    sub_edge_index : LongTensor [2, E_sub]
        Edges of the k-hop subgraph in **local** (relabelled) indexing.
    sub_edge_attr : FloatTensor [E_sub, 6]
        Edge features for the subgraph edges, in the order defined in this module's
        docstring.
    flagged_local : int
        Local index of the flagged (sink) account within the subgraph.
    num_sub_nodes : int
        Number of nodes in the subgraph.
    """
    src, dst = sub_edge_index[0], sub_edge_index[1]

    # ---- Structural metrics on the flagged node ----
    in_mask = dst == flagged_local
    out_mask = src == flagged_local
    in_deg = int(in_mask.sum().item())
    out_deg = int(out_mask.sum().item())
    distinct_senders = int(torch.unique(src[in_mask]).numel()) if in_deg > 0 else 0

    # ---- Intermediaries: subgraph nodes with both incoming and outgoing edges ----
    has_in = torch.zeros(num_sub_nodes, dtype=torch.bool)
    has_out = torch.zeros(num_sub_nodes, dtype=torch.bool)
    has_in[dst.unique()] = True
    has_out[src.unique()] = True
    intermediaries = int((has_in & has_out).sum().item())

    # ---- Edge-feature aggregates over INCOMING edges to the flagged node ----
    if in_deg > 0:
        in_amounts = sub_edge_attr[in_mask, 0]
        in_drains = sub_edge_attr[in_mask, 1]
        in_src_drained = sub_edge_attr[in_mask, 2]
        in_dst_empty = sub_edge_attr[in_mask, 3]
        mean_in_amount = float(in_amounts.mean().item())
        amount_std = _safe_std(in_amounts)
        mean_in_drain = float(in_drains.mean().item())
        src_drained_ratio = float(in_src_drained.mean().item())
        dst_empty_ratio = float(in_dst_empty.mean().item())
    else:
        mean_in_amount = amount_std = mean_in_drain = 0.0
        src_drained_ratio = dst_empty_ratio = 0.0

    # ---- Pattern scoring ---------------------------------------------------
    # HUB_AND_SPOKE: many distinct senders converge, flagged node mostly collects.
    fan_in_score = min(distinct_senders / 5.0, 1.0)
    collector_score = 1.0 - min(out_deg / max(in_deg + 1, 1), 0.6) / 0.6
    hub_score = fan_in_score * (0.5 + 0.5 * collector_score)

    # SMURFING: many small senders, deliberately uniform amounts (structuring).
    if in_deg >= 4:
        smurf_count = min(distinct_senders / 6.0, 1.0)
        # std of log-amount: 0 → score 1.0; 0.5 → 0.0. Real structuring is very tight.
        smurf_uniform = max(0.0, 1.0 - amount_std / 0.5)
        smurf_small = max(0.0, 1.0 - mean_in_amount / 10.0)  # log-amount ≈ 10 → ~$22k
        smurf_score = smurf_count * 0.35 + smurf_uniform * 0.5 + smurf_small * 0.15
    else:
        smurf_score = 0.0

    # LAYERING: several intermediaries relative to subgraph size,
    # and the flagged node itself is a pass-through (both in and out).
    if num_sub_nodes >= 4:
        chain_score = min(intermediaries / 4.0, 1.0)
        passthrough = 1.0 if (in_deg > 0 and out_deg > 0) else 0.4
        layer_score = chain_score * passthrough
    else:
        layer_score = 0.0

    # ACCOUNT_TAKEOVER: incoming edges show source accounts being fully drained,
    # often into a previously empty receiver. In-degree is typically low (1–3).
    if in_deg >= 1:
        ato_drain = src_drained_ratio
        ato_empty = dst_empty_ratio
        ato_focus = 1.0 - min(distinct_senders / 8.0, 1.0)  # ATO is concentrated, not fan-in
        ato_score = 0.5 * ato_drain + 0.3 * ato_empty + 0.2 * ato_focus
    else:
        ato_score = 0.0

    scores = {
        "HUB_AND_SPOKE": round(float(hub_score), 4),
        "SMURFING": round(float(smurf_score), 4),
        "LAYERING": round(float(layer_score), 4),
        "ACCOUNT_TAKEOVER": round(float(ato_score), 4),
    }

    # Tie-break: if everything is weak, return the structurally most informative one.
    best_pattern = max(scores, key=scores.get)
    best_conf = scores[best_pattern]
    if best_conf < 0.15:
        best_pattern = "HUB_AND_SPOKE" if in_deg >= out_deg else "LAYERING"
        best_conf = max(best_conf, 0.15)
    # SMURFING preferred over HUB_AND_SPOKE when amount uniformity is high — the
    # specific diagnosis (structuring) wins over the general one (collection).
    elif (
        best_pattern == "HUB_AND_SPOKE"
        and in_deg >= 4
        and amount_std < 0.2
        and scores["SMURFING"] >= 0.55
    ):
        best_pattern = "SMURFING"
        best_conf = scores["SMURFING"]

    evidence = {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "distinct_senders": distinct_senders,
        "intermediaries": intermediaries,
        "mean_incoming_amount_log": round(mean_in_amount, 3),
        "incoming_amount_std": round(amount_std, 3),
        "mean_drain_ratio": round(mean_in_drain, 3),
        "src_drained_ratio": round(src_drained_ratio, 3),
        "dst_was_empty_ratio": round(dst_empty_ratio, 3),
    }

    return PatternResult(
        pattern=best_pattern,
        confidence=round(best_conf, 3),
        scores=scores,
        evidence=evidence,
    )
