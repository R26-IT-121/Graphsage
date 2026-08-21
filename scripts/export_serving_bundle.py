"""Export a compact serving bundle from a trained temporal checkpoint.

Runs in Colab (needs the temporal snapshots + a checkpoint) and writes ONE
file that the FastAPI service loads directly. After this, serving requires no
model, no PyG forward pass, and no 3.3M-node inference at startup — the API
just memory-maps precomputed tensors, which is what makes the demo runnable on
a laptop.

What goes in the bundle:
    edge_index / edge_attr / edge_step / edge_isFraud   graph structure
    node_degrees          x[:, :2] only — the extractor uses in/out degree
    node_scores           ISOTONIC-calibrated mule probability per node
    edge_attention        mean per-edge attention from the Edge-MLP layers
    threshold             val-tuned decision threshold, mapped through the
                          same isotonic transform so it stays comparable

Why isotonic: the calibration study (scripts/calibration_study.py) shows raw
focal-loss sigmoids have ECE 0.80 and the ECDF percentile transform only gets
to 0.48 — a percentile is a rank, not a probability. Isotonic regression fitted
on the validation window reaches ECE 0.02 while being monotone, so ranking,
AUROC and the tuned operating point are all unchanged.

Reads:  data/graph/paysim_temporal_v2.pt
        checkpoints/temporal_stage3b_v2_seed0.pt
Writes: data/graph/serving_bundle.pt

Usage:
    python scripts/export_serving_bundle.py
    python scripts/export_serving_bundle.py --stage 3b --features v2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from graphsage.models.baseline import BaselineGraphSAGE
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.training.threshold_tuning import find_best_threshold_for_f1

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "graph" / "serving_bundle.pt"

# Stages whose model has no edge-attention layer (mirrors train_temporal.py).
NO_EDGE_MLP = ("1", "3c")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="3b")
    parser.add_argument("--features", choices=("v1", "v2"), default="v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    suffix = "_v2" if args.features == "v2" else ""
    snap_path = REPO_ROOT / "data" / "graph" / f"paysim_temporal{suffix}.pt"
    tag = f"stage{args.stage}{suffix}_seed{args.seed}"
    ckpt_path = REPO_ROOT / "checkpoints" / f"temporal_{tag}.pt"
    for p in (snap_path, ckpt_path):
        if not p.exists():
            raise SystemExit(f"{p} not found — run scripts/train_temporal.py first.")

    t0 = time.time()
    print(f"Loading {snap_path.name}...")
    snaps = torch.load(snap_path, weights_only=False, map_location="cpu")
    val, test = snaps["val"], snaps["test"]
    # The test snapshot's feature horizon is the last step, so its edge set is
    # the complete transaction history — the right graph to serve.
    graph = test

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if args.stage in NO_EDGE_MLP:
        model = BaselineGraphSAGE(
            in_dim=graph.x.shape[1], hidden_dim=args.hidden_dim
        )
        has_attention = False
    else:
        model = EdgeEnhancedGraphSAGE(
            in_dim=graph.x.shape[1],
            edge_dim=graph.edge_attr.shape[1],
            hidden_dim=args.hidden_dim,
        )
        has_attention = True
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path.name} (stage {args.stage}, {args.features})")

    def infer(snap):
        with torch.no_grad():
            if has_attention:
                logits, attns = model.forward_with_attention(
                    snap.x, snap.edge_index, snap.edge_attr
                )
                return logits, torch.stack(attns).mean(dim=0)
            return model(snap.x, snap.edge_index), None

    print("Scoring validation window (for calibration)...")
    val_logits, _ = infer(val)
    print("Scoring full graph...")
    logits, edge_attention = infer(graph)

    # Threshold tuned on validation, then mapped through the same calibrator.
    vmask = val.eval_mask
    raw_threshold, val_f1 = find_best_threshold_for_f1(
        val_logits[vmask], val.y[vmask].to(torch.float32)
    )

    print("Fitting isotonic calibration on the validation window...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(
        torch.sigmoid(val_logits[vmask]).numpy(),
        val.y[vmask].numpy().astype(float),
    )
    node_scores = torch.from_numpy(
        iso.predict(torch.sigmoid(logits).numpy()).astype(np.float32)
    )
    threshold = float(iso.predict(np.array([raw_threshold]))[0])

    if edge_attention is None:
        # No Edge-MLP: fall back to uniform weights so the subgraph payload
        # stays schema-valid (the demo then shows equal-width edges).
        edge_attention = torch.full(
            (graph.edge_index.shape[1],), 0.5, dtype=torch.float32
        )

    bundle = {
        "edge_index": graph.edge_index,
        "edge_attr": graph.edge_attr,
        "edge_step": graph.edge_step,
        "edge_isFraud": graph.edge_isFraud,
        "node_degrees": graph.x[:, :2].contiguous(),  # in_degree, out_degree
        "node_scores": node_scores,
        "edge_attention": edge_attention.to(torch.float32),
        "threshold": threshold,
        "raw_threshold": float(raw_threshold),
        "num_nodes": int(graph.num_nodes),
        "meta": {
            "stage": args.stage,
            "features": args.features,
            "seed": args.seed,
            "protocol": "temporal_snapshots_leakage_free",
            "calibration": "isotonic_fitted_on_validation_window",
            "has_edge_attention": has_attention,
            "val_f1_at_tuned_threshold": round(float(val_f1), 4),
            "step_range": [int(graph.edge_step.min()), int(graph.edge_step.max())],
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1024**2

    print(json.dumps(bundle["meta"], indent=2))
    print(
        f"\nthreshold: raw {raw_threshold:.4f} -> calibrated {threshold:.4f}"
        f"  (val F1 {val_f1:.4f})"
    )
    print(f"wrote {OUT_PATH.name}  ({size_mb:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
