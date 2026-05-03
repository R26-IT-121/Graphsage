"""Re-evaluate existing checkpoints with threshold tuning.

Loads Stage 1 and Stage 2 checkpoints, runs forward pass on val + test,
finds the optimal threshold on val, reports test metrics at that threshold.

This is the standard fix for the inflated-recall / low-precision pattern
seen with high pos_weight under severe class imbalance.

Reads:  checkpoints/stage1_baseline.pt
        checkpoints/stage2_edge_mlp.pt
        data/graph/paysim_graph.pt
Writes: reports/ablation_tuned.json
        prints clean comparison table

Usage:
    python scripts/eval_with_tuned_threshold.py
"""

import json
from pathlib import Path

import torch

from graphsage.models.baseline import BaselineGraphSAGE
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.training.threshold_tuning import evaluate_with_tuned_threshold
from graphsage.training.trainer import select_device

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
S1_CKPT = REPO_ROOT / "checkpoints" / "stage1_baseline.pt"
S2_CKPT = REPO_ROOT / "checkpoints" / "stage2_edge_mlp.pt"
OUT_PATH = REPO_ROOT / "reports" / "ablation_tuned.json"


def load_logits(model: torch.nn.Module, data, use_edge_attr: bool, device) -> torch.Tensor:
    """Run forward pass and return per-node logits (CPU tensor)."""
    model = model.to(device).eval()
    with torch.no_grad():
        if use_edge_attr:
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
    return logits.cpu()


def main() -> None:
    device = select_device()
    print(f"Device: {device}\n")

    # Load graph
    print(f"Loading {GRAPH_PATH.name}...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  {data}\n")

    # Move data to device
    try:
        data = data.to(device)
    except (RuntimeError, MemoryError):
        device = torch.device("cpu")
        data = data.to(device)
        print(f"  Fell back to CPU.\n")

    in_dim = int(data.x.shape[1])
    edge_dim = int(data.edge_attr.shape[1])

    results = {}

    # ---- Stage 1 ----
    if S1_CKPT.exists():
        print("=" * 60)
        print("Stage 1 — Baseline GraphSAGE")
        print("=" * 60)
        ckpt = torch.load(S1_CKPT, weights_only=False, map_location=device)
        hp = ckpt.get("hyperparameters", {})
        m = BaselineGraphSAGE(
            in_dim=in_dim,
            hidden_dim=hp.get("hidden_dim", 64),
            dropout=hp.get("dropout", 0.3),
        )
        m.load_state_dict(ckpt["state_dict"])
        logits = load_logits(m, data, use_edge_attr=False, device=device)

        eval_out = evaluate_with_tuned_threshold(
            val_logits=logits[data.val_mask.cpu()],
            val_y=data.y[data.val_mask].cpu(),
            test_logits=logits[data.test_mask.cpu()],
            test_y=data.y[data.test_mask].cpu(),
        )
        results["stage_1"] = eval_out
        _print_eval("Stage 1", eval_out)
    else:
        print(f"⚠️  {S1_CKPT.name} not found - skip Stage 1")

    # ---- Stage 2 ----
    if S2_CKPT.exists():
        print("\n" + "=" * 60)
        print("Stage 2 — Edge-Enhanced GraphSAGE")
        print("=" * 60)
        ckpt = torch.load(S2_CKPT, weights_only=False, map_location=device)
        hp = ckpt.get("hyperparameters", {})
        m = EdgeEnhancedGraphSAGE(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hp.get("hidden_dim", 64),
            edge_mlp_hidden=hp.get("edge_mlp_hidden", 32),
            dropout=hp.get("dropout", 0.3),
        )
        m.load_state_dict(ckpt["state_dict"])
        logits = load_logits(m, data, use_edge_attr=True, device=device)

        eval_out = evaluate_with_tuned_threshold(
            val_logits=logits[data.val_mask.cpu()],
            val_y=data.y[data.val_mask].cpu(),
            test_logits=logits[data.test_mask.cpu()],
            test_y=data.y[data.test_mask].cpu(),
        )
        results["stage_2"] = eval_out
        _print_eval("Stage 2", eval_out)
    else:
        print(f"⚠️  {S2_CKPT.name} not found - skip Stage 2")

    # ---- Ablation table ----
    print("\n" + "=" * 78)
    print("ABLATION TABLE — Test set, threshold-tuned on validation")
    print("=" * 78)
    _print_table(results)

    # ---- Save ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_PATH}")


def _print_eval(stage_name: str, ev: dict) -> None:
    d_test = ev["default_threshold_metrics"]["test"]
    t_test = ev["tuned_threshold_metrics"]["test"]
    print(
        f"\n{stage_name} — DEFAULT threshold (0.5):\n"
        f"  Test  F1={d_test['f1']:.4f}  P={d_test['precision']:.4f}  "
        f"R={d_test['recall']:.4f}  AUROC={d_test['auroc']:.4f}"
    )
    print(
        f"\n{stage_name} — TUNED threshold ({ev['best_threshold']:.4f}):\n"
        f"  Val   F1={ev['val_f1_at_tuned_threshold']:.4f}\n"
        f"  Test  F1={t_test['f1']:.4f}  P={t_test['precision']:.4f}  "
        f"R={t_test['recall']:.4f}  AUROC={t_test['auroc']:.4f}\n"
        f"  ⇒ Test F1 improvement: {t_test['f1'] - d_test['f1']:+.4f}"
    )


def _print_table(results: dict) -> None:
    """Pretty-print the ablation table."""
    print(
        f"\n  {'Stage':<32} {'Threshold':>10} {'F1':>8} {'Precision':>10} "
        f"{'Recall':>8} {'AUROC':>8}"
    )
    print("  " + "-" * 76)
    for key, label in [
        ("stage_1", "Stage 1 — Baseline"),
        ("stage_2", "Stage 2 — + Edge-MLP (Novelty 1)"),
    ]:
        if key not in results:
            continue
        ev = results[key]
        for thresh_key, thresh_val_label in [
            ("default_threshold_metrics", f"0.5"),
            ("tuned_threshold_metrics", f"{ev['best_threshold']:.4f}"),
        ]:
            t = ev[thresh_key]["test"]
            mode = "default" if "default" in thresh_key else "tuned  "
            print(
                f"  {label} ({mode}){'':<3} {thresh_val_label:>10} "
                f"{t['f1']:>8.4f} {t['precision']:>10.4f} "
                f"{t['recall']:>8.4f} {t['auroc']:>8.4f}"
            )
        print()


if __name__ == "__main__":
    main()
