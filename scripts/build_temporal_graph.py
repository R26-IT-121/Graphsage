"""Build the leakage-free temporal snapshots and save them to disk.

Reads:  data/processed/features.parquet
Writes: data/graph/paysim_temporal.pt      (v1: 5 structural features)
        data/graph/paysim_temporal_v2.pt   (--features v2: 12 behavioural)
        data/graph/temporal_metadata[_v2].json

Usage:
    python scripts/build_temporal_graph.py [--features v1|v2]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from graphsage.data.temporal import build_temporal_snapshots

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    suffix = "_v2" if args.features == "v2" else ""
    out_graph = REPO_ROOT / "data" / "graph" / f"paysim_temporal{suffix}.pt"
    out_meta = REPO_ROOT / "data" / "graph" / f"temporal_metadata{suffix}.json"

    t0 = time.time()
    snapshots, stats = build_temporal_snapshots(
        PARQUET_PATH, feature_version=args.features
    )
    print(f"Snapshots built in {time.time() - t0:.1f}s")
    for name in ("train", "val", "test"):
        s = stats[name]
        print(
            f"  {name:5}: edges<= step {s['feature_horizon']:>3}  "
            f"({s['edges']:>9,} edges)  label window {s['label_window']}  "
            f"eval nodes {s['eval_nodes']:>9,}  positives {s['positives']:>5,}"
        )

    torch.save(snapshots, out_graph)
    out_meta.write_text(json.dumps(stats, indent=2))
    print(f"Saved {out_graph.name} ({out_graph.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
