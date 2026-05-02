"""Train Stage 1 — Baseline GraphSAGE on the PaySim graph.

Reads:  data/graph/paysim_graph.pt
Writes: checkpoints/stage1_baseline.pt
        reports/stage1_metrics.json

Usage (any of these work):
    python scripts/train_baseline.py
    python scripts/train_baseline.py --epochs 30 --hidden-dim 128

The Stage 1 baseline:
- Stock 2-layer GraphSAGE (mean aggregator)
- Node features only — edge_attr ignored
- BCEWithLogitsLoss with pos_weight (no Focal Loss yet)
- No imbalance sampler (Focal + sampler are added in Stage 3)

This establishes the "before" number. Anything beyond this F1 is what
Novelties 1 and 2 contribute.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from graphsage.models.baseline import BaselineGraphSAGE
from graphsage.training.trainer import save_checkpoint, train_node_classifier

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "model_config.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "stage1_baseline.pt"
METRICS_PATH = REPO_ROOT / "reports" / "stage1_metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 baseline GraphSAGE")
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    p.add_argument("--lr", type=float, default=None, help="override learning_rate")
    p.add_argument("--hidden-dim", type=int, default=None, help="override hidden_dim")
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
    dropout = args.dropout if args.dropout is not None else cfg["model"]["dropout"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    weight_decay = cfg["training"].get("weight_decay", 1e-5)
    patience = args.patience or cfg["training"].get("early_stopping_patience", 5)

    print("=" * 60)
    print("Stage 1 — Baseline GraphSAGE training")
    print("=" * 60)
    print(f"  hidden_dim:    {hidden_dim}")
    print(f"  dropout:       {dropout}")
    print(f"  num_epochs:    {epochs}")
    print(f"  learning_rate: {lr}")
    print(f"  weight_decay:  {weight_decay}")
    print(f"  patience:      {patience}")

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

    # ---- Build model ----
    in_dim = int(data.x.shape[1])
    print(f"\nBuilding BaselineGraphSAGE(in_dim={in_dim}, hidden_dim={hidden_dim})")
    model = BaselineGraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ---- Train ----
    print("\nTraining...")
    result = train_node_classifier(
        model=model,
        data=data,
        num_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
        use_edge_attr=False,
    )

    # ---- Save checkpoint ----
    save_checkpoint(
        result,
        CHECKPOINT_PATH,
        extra={
            "stage": 1,
            "model_class": "BaselineGraphSAGE",
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "num_epochs": epochs,
                "patience": patience,
            },
        },
    )

    # ---- Save metrics summary (small JSON for the report) ----
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": 1,
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
