"""Add train / val / test masks to paysim_graph.pt and save back.

Reads:  data/graph/paysim_graph.pt
Writes: same file, now with train_mask, val_mask, test_mask, first_step
        data/graph/split_metadata.json

Usage:
    python scripts/make_splits.py
"""

import json
import time
from pathlib import Path

import torch

from graphsage.data.splits import make_time_split

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
META_PATH = REPO_ROOT / "data" / "graph" / "split_metadata.json"

TRAIN_END = 600
VAL_END = 700


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{GRAPH_PATH} not found. Run scripts/build_graph.py first."
        )

    print(f"Loading {GRAPH_PATH.name}...")
    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  {data}")

    print(f"\nAdding time-based split (train<={TRAIN_END}, val<={VAL_END})...")
    t0 = time.time()
    data, stats = make_time_split(data, train_end=TRAIN_END, val_end=VAL_END)
    print(f"  Split computed in {time.time() - t0:.1f}s")
    print(stats)

    # Sanity: masks must be disjoint and cover all nodes
    overlap_tv = (data.train_mask & data.val_mask).sum().item()
    overlap_tt = (data.train_mask & data.test_mask).sum().item()
    overlap_vt = (data.val_mask & data.test_mask).sum().item()
    coverage = (data.train_mask | data.val_mask | data.test_mask).sum().item()
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, "Masks overlap!"
    assert coverage == data.num_nodes, f"Coverage {coverage} != {data.num_nodes}"
    print(f"\nSanity OK: masks disjoint, all {data.num_nodes:,} nodes assigned.")

    # Save back
    print(f"\nSaving updated graph to {GRAPH_PATH.name}...")
    t0 = time.time()
    torch.save(data, GRAPH_PATH)
    size_mb = GRAPH_PATH.stat().st_size / 1024**2
    print(f"  {size_mb:.1f} MB written in {time.time() - t0:.1f}s")

    # Metadata
    metadata = {
        "train_end_step": TRAIN_END,
        "val_end_step": VAL_END,
        "train_nodes": stats.train_nodes,
        "val_nodes": stats.val_nodes,
        "test_nodes": stats.test_nodes,
        "train_mules": stats.train_mules,
        "val_mules": stats.val_mules,
        "test_mules": stats.test_mules,
        "split_strategy": "node assigned by earliest incident edge step (sender or receiver)",
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"  Metadata written to {META_PATH.name}")

    print("\n=== Final ===")
    print(f"  {data}")


if __name__ == "__main__":
    main()
