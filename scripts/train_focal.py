"""Train Stage 3a — Edge-Enhanced GraphSAGE + Focal Loss.

Same Stage 2 architecture (EdgeEnhancedGraphSAGE), but the loss function
swaps BCEWithLogitsLoss(pos_weight) for FocalLoss(gamma, alpha). This is
the first half of Novelty 2 — it isolates the contribution of the loss
function from the Imbalance Sampler.

Stage 3a: model = EdgeEnhancedGraphSAGE, loss = FocalLoss
Stage 3b: same + k-hop fraud subgraph sampler with hard negatives (TBD)

Reads:  data/graph/paysim_graph.pt
Writes: checkpoints/stage3a_focal.pt
        reports/stage3a_metrics.json

Usage:
    python scripts/train_focal.py
    python scripts/train_focal.py --gamma 2.0 --alpha 0.75
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.training.losses import FocalLoss
from graphsage.training.trainer import save_checkpoint, train_node_classifier

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "model_config.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "stage3a_focal.pt"
METRICS_PATH = REPO_ROOT / "reports" / "stage3a_metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 3a — Edge-MLP + Focal Loss")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--edge-mlp-hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None, help="Focal loss focusing parameter (default 2.0)")
    p.add_argument("--alpha", type=float, default=None, help="Focal loss class balance (default 0.75)")
    return p.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config()

    hidden_dim = args.hidden_dim or cfg["model"]["hidden_dim"]
    edge_mlp_hidden = args.edge_mlp_hidden or cfg["model"].get("edge_mlp_hidden", 32)
    dropout = args.dropout if args.dropout is not None else cfg["model"]["dropout"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    weight_decay = cfg["training"].get("weight_decay", 1e-5)
    patience = args.patience or cfg["training"].get("early_stopping_patience", 5)
    gamma = args.gamma if args.gamma is not None else cfg["training"].get("focal_gamma", 2.0)
    alpha = args.alpha if args.alpha is not None else cfg["training"].get("focal_alpha", 0.75)

    print("=" * 60)
    print("Stage 3a — Edge-MLP + Focal Loss training (Novelty 2 part 1)")
    print("=" * 60)
    print(f"  hidden_dim:      {hidden_dim}")
    print(f"  edge_mlp_hidden: {edge_mlp_hidden}")
    print(f"  dropout:         {dropout}")
    print(f"  num_epochs:      {epochs}")
    print(f"  learning_rate:   {lr}")
    print(f"  weight_decay:    {weight_decay}")
    print(f"  patience:        {patience}")
    print(f"  focal_gamma:     {gamma}")
    print(f"  focal_alpha:     {alpha}")

    # ---- Load graph ----
    print(f"\nLoading {GRAPH_PATH.name}...")
    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  {data}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if not hasattr(data, "train_mask"):
        raise RuntimeError("Graph has no train_mask. Run scripts/make_splits.py first.")

    # ---- Build model ----
    in_dim = int(data.x.shape[1])
    edge_dim = int(data.edge_attr.shape[1])
    print(
        f"\nBuilding EdgeEnhancedGraphSAGE("
        f"in_dim={in_dim}, edge_dim={edge_dim}, hidden_dim={hidden_dim})"
    )
    model = EdgeEnhancedGraphSAGE(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        edge_mlp_hidden=edge_mlp_hidden,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ---- Build Focal Loss ----
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
    print(f"\nLoss: FocalLoss(gamma={gamma}, alpha={alpha})")

    # ---- Train ----
    print("\nTraining...")
    result = train_node_classifier(
        model=model,
        data=data,
        num_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
        use_edge_attr=True,
        loss_fn=loss_fn,
    )

    # ---- Save ----
    save_checkpoint(
        result,
        CHECKPOINT_PATH,
        extra={
            "stage": "3a",
            "model_class": "EdgeEnhancedGraphSAGE",
            "loss_class": "FocalLoss",
            "novelty": "Edge-MLP attention + Focal Loss (Novelty 2 part 1)",
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "edge_mlp_hidden": edge_mlp_hidden,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "num_epochs": epochs,
                "patience": patience,
                "focal_gamma": gamma,
                "focal_alpha": alpha,
            },
        },
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "3a",
        "best_epoch": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "test_metrics": result.final_test_metrics,
        "epochs_run": len(result.history),
        "history": [vars(m) for m in result.history],
        "loss": "FocalLoss",
        "focal_gamma": gamma,
        "focal_alpha": alpha,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
