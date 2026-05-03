"""Train Stage 2 — Edge-Enhanced GraphSAGE on the PaySim graph.

Reads:  data/graph/paysim_graph.pt
Writes: checkpoints/stage2_edge_mlp.pt
        reports/stage2_metrics.json

Usage:
    python scripts/train_edge_mlp.py
    python scripts/train_edge_mlp.py --epochs 30 --hidden-dim 128

The Stage 2 model:
- Same 2-layer architecture as Stage 1 BUT each SAGEConv is replaced by
  EdgeEnhancedSAGEConv (Novelty 1 — MLP-based per-edge attention).
- Same loss as Stage 1 (BCEWithLogitsLoss with pos_weight). Focal Loss
  is reserved for Stage 3 to keep the ablation honest.
- Same hyperparameters as Stage 1 by default. Any improvement in F1 over
  Stage 1 isolates the contribution of the Edge-MLP attention layer.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.training.trainer import save_checkpoint, train_node_classifier

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "model_config.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "stage2_edge_mlp.pt"
METRICS_PATH = REPO_ROOT / "reports" / "stage2_metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 2 Edge-Enhanced GraphSAGE")
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    p.add_argument("--lr", type=float, default=None, help="override learning_rate")
    p.add_argument("--hidden-dim", type=int, default=None, help="override hidden_dim")
    p.add_argument("--edge-mlp-hidden", type=int, default=None, help="override edge_mlp_hidden")
    p.add_argument("--dropout", type=float, default=None, help="override dropout")
    p.add_argument("--patience", type=int, default=None, help="override early_stopping_patience")
    return p.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config()

    # Resolve hyperparameters: CLI > config > built-in defaults
    hidden_dim = args.hidden_dim or cfg["model"]["hidden_dim"]
    edge_mlp_hidden = args.edge_mlp_hidden or cfg["model"].get("edge_mlp_hidden", 32)
    dropout = args.dropout if args.dropout is not None else cfg["model"]["dropout"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    weight_decay = cfg["training"].get("weight_decay", 1e-5)
    patience = args.patience or cfg["training"].get("early_stopping_patience", 5)

    print("=" * 60)
    print("Stage 2 — Edge-Enhanced GraphSAGE training (Novelty 1)")
    print("=" * 60)
    print(f"  hidden_dim:      {hidden_dim}")
    print(f"  edge_mlp_hidden: {edge_mlp_hidden}")
    print(f"  dropout:         {dropout}")
    print(f"  num_epochs:      {epochs}")
    print(f"  learning_rate:   {lr}")
    print(f"  weight_decay:    {weight_decay}")
    print(f"  patience:        {patience}")

    # ---- Load graph ----
    print(f"\nLoading {GRAPH_PATH.name}...")
    t0 = time.time()
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  {data}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if not hasattr(data, "train_mask"):
        raise RuntimeError(
            f"Graph at {GRAPH_PATH} has no train_mask. "
            "Run scripts/make_splits.py first."
        )
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        raise RuntimeError(
            f"Graph at {GRAPH_PATH} has no edge_attr. "
            "Stage 2 requires edge features. Re-run scripts/build_graph.py."
        )

    # ---- Build model ----
    in_dim = int(data.x.shape[1])
    edge_dim = int(data.edge_attr.shape[1])
    print(
        f"\nBuilding EdgeEnhancedGraphSAGE("
        f"in_dim={in_dim}, edge_dim={edge_dim}, "
        f"hidden_dim={hidden_dim}, edge_mlp_hidden={edge_mlp_hidden})"
    )
    model = EdgeEnhancedGraphSAGE(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        edge_mlp_hidden=edge_mlp_hidden,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_edge_mlp_params = sum(
        p.numel()
        for layer in [model.conv1.edge_mlp, model.conv2.edge_mlp]
        for p in layer.parameters()
    )
    print(f"  Total parameters:    {n_params:,}")
    print(f"  Edge-MLP parameters: {n_edge_mlp_params:,}  (the Novelty 1 contribution)")

    # ---- Train ----
    print("\nTraining...")
    result = train_node_classifier(
        model=model,
        data=data,
        num_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
        use_edge_attr=True,  # ← key difference from Stage 1
    )

    # ---- Save checkpoint ----
    save_checkpoint(
        result,
        CHECKPOINT_PATH,
        extra={
            "stage": 2,
            "model_class": "EdgeEnhancedGraphSAGE",
            "novelty": "Edge-MLP attention (Novelty 1)",
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "edge_mlp_hidden": edge_mlp_hidden,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "num_epochs": epochs,
                "patience": patience,
            },
        },
    )

    # ---- Save metrics summary ----
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": 2,
        "best_epoch": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "test_metrics": result.final_test_metrics,
        "epochs_run": len(result.history),
        "history": [vars(m) for m in result.history],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
