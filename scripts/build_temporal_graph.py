"""Build the leakage-free temporal snapshots and save them to disk.

Reads:  data/processed/features.parquet
Writes: data/graph/paysim_temporal.pt      (train/val/test snapshot dict)
        data/graph/temporal_metadata.json

Usage:
    python scripts/build_temporal_graph.py
"""

import json
import time
from pathlib import Path

import torch

from graphsage.data.temporal import build_temporal_snapshots

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
OUT_GRAPH = REPO_ROOT / "data" / "graph" / "paysim_temporal.pt"
OUT_META = REPO_ROOT / "data" / "graph" / "temporal_metadata.json"


def main() -> None:
    t0 = time.time()
    snapshots, stats = build_temporal_snapshots(PARQUET_PATH)
    print(f"Snapshots built in {time.time() - t0:.1f}s")
    for name in ("train", "val", "test"):
        s = stats[name]
        print(
            f"  {name:5}: edges<= step {s['feature_horizon']:>3}  "
            f"({s['edges']:>9,} edges)  label window {s['label_window']}  "
            f"eval nodes {s['eval_nodes']:>9,}  positives {s['positives']:>5,}"
        )

    torch.save(snapshots, OUT_GRAPH)
    OUT_META.write_text(json.dumps(stats, indent=2))
    print(f"Saved {OUT_GRAPH.name} ({OUT_GRAPH.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
