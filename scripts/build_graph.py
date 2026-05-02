"""Build the PaySim PyG graph and save it to disk.

Reads:  data/processed/features.parquet
Writes: data/graph/paysim_graph.pt          (~250 MB tensor object)
        data/graph/graph_metadata.json      (sanity stats)

Usage:
    python scripts/build_graph.py
"""

import json
import time
from pathlib import Path

import torch

from graphsage.data.graph_builder import (
    EDGE_FEATURE_COLS,
    NODE_FEATURE_NAMES,
    build_paysim_graph,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
OUT_DIR = REPO_ROOT / "data" / "graph"
OUT_GRAPH = OUT_DIR / "paysim_graph.pt"
OUT_META = OUT_DIR / "graph_metadata.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data, stats = build_paysim_graph(PARQUET_PATH)
    print(f"\nGraph built in {time.time() - t0:.1f}s")

    # Save graph tensor
    print(f"\nSaving graph tensor to {OUT_GRAPH}...")
    t0 = time.time()
    torch.save(data, OUT_GRAPH)
    size_mb = OUT_GRAPH.stat().st_size / 1024**2
    print(f"  {size_mb:.1f} MB written in {time.time() - t0:.1f}s")

    # Metadata for sanity / downstream loaders
    metadata = {
        "num_nodes": stats.num_nodes,
        "num_edges": stats.num_edges,
        "num_mules": stats.num_mules,
        "num_fraud_edges": stats.num_fraud_edges,
        "num_node_features": int(data.x.shape[1]),
        "num_edge_features": int(data.edge_attr.shape[1]),
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_COLS,
        "labeling_rule": "y[node] = 1 iff node received any fraud transaction (mule)",
        "edge_step_min": int(data.edge_step.min().item()),
        "edge_step_max": int(data.edge_step.max().item()),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2))
    print(f"  Metadata written to {OUT_META.name}")

    # Console summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {data}")
    print(f"  Mules:        {stats.num_mules:>10,}  ({stats.num_mules / stats.num_nodes * 100:.4f}%)")
    print(f"  Fraud edges:  {stats.num_fraud_edges:>10,}  ({stats.num_fraud_edges / stats.num_edges * 100:.4f}%)")
    print(f"  Step range:   {metadata['edge_step_min']} -> {metadata['edge_step_max']}")
    print()
    print("Node feature columns: " + ", ".join(NODE_FEATURE_NAMES))
    print("Edge feature columns: " + ", ".join(EDGE_FEATURE_COLS))


if __name__ == "__main__":
    main()
