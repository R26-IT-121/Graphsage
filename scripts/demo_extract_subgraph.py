"""Demo of Novelty 3 — extract a suspicious subgraph around a real fraud edge.

Loads the full PaySim graph and the Stage 3 checkpoint, runs one forward pass
with attention, picks a trigger transaction (a real fraud edge by default, or
--src/--dst/--step to choose one), and prints the suspicious_subgraph JSON
exactly as it will be returned by POST /api/graph/analyze.

Usage:
    python scripts/demo_extract_subgraph.py                  # first test-window fraud edge
    python scripts/demo_extract_subgraph.py --edge-id 12345
    python scripts/demo_extract_subgraph.py --src C123 --dst C456 --step 700
    python scripts/demo_extract_subgraph.py --out examples/api_responses/subgraph_demo.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from graphsage.extraction.subgraph import SuspiciousSubgraphExtractor, load_node_names
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
NAMES_CACHE = REPO_ROOT / "data" / "graph" / "node_names.npy"
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
CKPT_PATH = REPO_ROOT / "checkpoints" / "stage3_full.pt"
ABLATION_PATH = REPO_ROOT / "reports" / "ablation_tuned.json"


def tuned_threshold(default: float = 0.5) -> float:
    if ABLATION_PATH.exists():
        report = json.loads(ABLATION_PATH.read_text())
        return float(report.get("stage_3", {}).get("best_threshold", default))
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-id", type=int, default=None)
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[1/4] Loading graph from {GRAPH_PATH.name}...")
    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
    names = load_node_names(PARQUET_PATH, cache_path=NAMES_CACHE)
    print(f"      {data.num_nodes:,} nodes, {data.num_edges:,} edges in {time.time()-t0:.1f}s")

    print(f"[2/4] Loading checkpoint {CKPT_PATH.name}...")
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    hp = ckpt.get("hyperparameters", {})
    model = EdgeEnhancedGraphSAGE(
        in_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        hidden_dim=hp.get("hidden_dim", 64),
        edge_mlp_hidden=hp.get("edge_mlp_hidden", 32),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("[3/4] Forward pass with attention (full graph, CPU)...")
    t0 = time.time()
    with torch.no_grad():
        logits, attentions = model.forward_with_attention(
            data.x, data.edge_index, data.edge_attr
        )
        probs = torch.sigmoid(logits)
        edge_attention = torch.stack(attentions).mean(dim=0)
    print(f"      done in {time.time()-t0:.1f}s")

    threshold = tuned_threshold()
    extractor = SuspiciousSubgraphExtractor(
        data, names, k=2, risk_threshold=threshold, max_edges=args.max_edges
    )

    if args.edge_id is not None:
        trigger = args.edge_id
    elif args.src and args.dst:
        trigger = extractor.find_trigger_edge(args.src, args.dst, args.step)
        if trigger is None:
            raise SystemExit(f"No edge found for {args.src} -> {args.dst}")
    else:
        # Default: first fraud edge in the test window (last 20% of steps).
        test_start = int(torch.quantile(data.edge_step.float(), 0.8).item())
        mask = (data.edge_isFraud == 1) & (data.edge_step >= test_start)
        trigger = int(mask.nonzero(as_tuple=True)[0][0].item())
        print(f"      trigger: first test-window fraud edge (id={trigger})")

    print(f"[4/4] Extracting k=2 subgraph (threshold={threshold:.4f})...")
    t0 = time.time()
    payload = extractor.extract(trigger, probs, edge_attention)
    latency_ms = (time.time() - t0) * 1000
    print(f"      extraction latency: {latency_ms:.1f} ms")

    print("\n" + "=" * 60)
    print(
        f"pattern={payload['pattern']} (conf {payload['pattern_confidence']}), "
        f"sink={payload['sink_account']}, "
        f"nodes={payload['node_count']}, edges={payload['edge_count']}, "
        f"mules={payload['structural_evidence']['mules_in_subgraph']}"
    )
    print("=" * 60)
    print(json.dumps(payload, indent=2)[:3000])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nFull payload written to {args.out}")


if __name__ == "__main__":
    main()
