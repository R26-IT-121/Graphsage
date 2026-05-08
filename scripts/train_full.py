"""Train Stage 3b — Full system: Edge-MLP + Focal Loss + Graph-Aware Imbalance Sampler.

This is the complete Novelty 2 implementation: balanced mini-batch training
on k-hop fraud subgraphs with Hard Negative Mining and Focal Loss. Replaces
the full-batch training of Stages 1-3a, where the aggregate weight of 3.2M
negative examples dominates the gradient regardless of per-example weighting.

Reads:  data/graph/paysim_graph.pt
Writes: checkpoints/stage3_full.pt
        reports/stage3_metrics.json

Usage:
    python scripts/train_full.py
    python scripts/train_full.py --pos-per-batch 128 --neg-per-batch 128
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.sampling.imbalance_sampler import GraphAwareImbalanceSampler
from graphsage.training.losses import FocalLoss
from graphsage.training.threshold_tuning import (
    evaluate_with_tuned_threshold,
    metrics_at_threshold,
)
from graphsage.training.trainer import select_device

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "graph" / "paysim_graph.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "model_config.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "stage3_full.pt"
METRICS_PATH = REPO_ROOT / "reports" / "stage3_metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 3b — full system (Novelty 2)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--edge-mlp-hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--k-hop", type=int, default=None)
    p.add_argument("--pos-per-batch", type=int, default=64)
    p.add_argument("--neg-per-batch", type=int, default=64)
    p.add_argument("--hard-neg-ratio", type=float, default=0.5)
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
    alpha = args.alpha if args.alpha is not None else cfg["training"].get("focal_alpha", 0.95)
    k_hop = args.k_hop or cfg["sampling"].get("k_hop", 2)

    print("=" * 60)
    print("Stage 3b — Full system (Edge-MLP + Focal + Sampler)")
    print("=" * 60)
    print(f"  hidden_dim:         {hidden_dim}")
    print(f"  edge_mlp_hidden:    {edge_mlp_hidden}")
    print(f"  dropout:            {dropout}")
    print(f"  num_epochs:         {epochs}")
    print(f"  learning_rate:      {lr}")
    print(f"  weight_decay:       {weight_decay}")
    print(f"  patience:           {patience}")
    print(f"  focal_gamma:        {gamma}")
    print(f"  focal_alpha:        {alpha}")
    print(f"  k_hop:              {k_hop}")
    print(f"  pos_per_batch:      {args.pos_per_batch}")
    print(f"  neg_per_batch:      {args.neg_per_batch}")
    print(f"  hard_neg_ratio:     {args.hard_neg_ratio}")

    # ---- Load graph ----
    print(f"\nLoading {GRAPH_PATH.name}...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  {data}")

    if not hasattr(data, "train_mask"):
        raise RuntimeError("Graph has no train_mask. Run scripts/make_splits.py first.")

    # ---- Build model + loss + sampler + optimizer ----
    in_dim = int(data.x.shape[1])
    edge_dim = int(data.edge_attr.shape[1])
    model = EdgeEnhancedGraphSAGE(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        edge_mlp_hidden=edge_mlp_hidden,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EdgeEnhancedGraphSAGE ({n_params:,} params)")

    loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
    print(f"Loss:  FocalLoss(gamma={gamma}, alpha={alpha})")

    sampler = GraphAwareImbalanceSampler(
        data=data,
        k_hop=k_hop,
        pos_per_batch=args.pos_per_batch,
        neg_per_batch=args.neg_per_batch,
        hard_negative_ratio=args.hard_neg_ratio,
    )
    print(
        f"Sampler: {sampler.num_pos_train:,} train mules, "
        f"{sampler.num_neg_train:,} train legit, "
        f"{sampler.steps_per_epoch()} steps/epoch"
    )

    device = select_device()
    print(f"Device: {device}")
    try:
        data = data.to(device)
        model = model.to(device)
    except (RuntimeError, MemoryError) as exc:
        print(f"  Could not move graph to {device}: {exc}")
        print("  Falling back to CPU.")
        device = torch.device("cpu")
        data = data.to(device)
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training loop ----
    history = []
    best_val_f1 = 0.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_train_loss = 0.0
        steps = sampler.steps_per_epoch()

        for _ in range(steps):
            batch = sampler.sample()
            bx = batch.x.to(device)
            bei = batch.edge_index.to(device)
            bea = batch.edge_attr.to(device)
            by = batch.y.to(device)
            seed_idx = batch.seed_local_idx.to(device)

            optimizer.zero_grad()
            logits_sub = model(bx, bei, bea)
            loss = loss_fn(logits_sub[seed_idx], by)
            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())

        epoch_train_loss /= steps

        # ---- Validate (full-batch on full graph) ----
        model.eval()
        with torch.no_grad():
            logits_full = model(data.x, data.edge_index, data.edge_attr)
            val_logits = logits_full[data.val_mask].cpu()
            val_y = data.y[data.val_mask].cpu()
            val_m = metrics_at_threshold(val_logits, val_y, 0.5)

        elapsed = time.time() - t0
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_f1": val_m["f1"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_auroc": val_m["auroc"],
            "elapsed_s": elapsed,
        }
        history.append(epoch_metrics)
        print(
            f"  epoch {epoch:>3} | train_loss {epoch_train_loss:.4f} | "
            f"val_F1 {val_m['f1']:.4f} | P {val_m['precision']:.4f} | "
            f"R {val_m['recall']:.4f} | AUROC {val_m['auroc']:.4f} | {elapsed:.1f}s"
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stopping: no val F1 improvement for {patience} epochs. "
                    f"Best F1={best_val_f1:.4f} at epoch {best_epoch}."
                )
                break

    # ---- Final test eval (default + tuned thresholds) ----
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_full = model(data.x, data.edge_index, data.edge_attr)
            val_logits = logits_full[data.val_mask].cpu()
            val_y = data.y[data.val_mask].cpu()
            test_logits = logits_full[data.test_mask].cpu()
            test_y = data.y[data.test_mask].cpu()

        eval_out = evaluate_with_tuned_threshold(val_logits, val_y, test_logits, test_y)
        d = eval_out["default_threshold_metrics"]["test"]
        t = eval_out["tuned_threshold_metrics"]["test"]
        thresh = eval_out["best_threshold"]
        print(
            f"\nTest @ default 0.5: F1={d['f1']:.4f}  P={d['precision']:.4f}  "
            f"R={d['recall']:.4f}  AUROC={d['auroc']:.4f}"
        )
        print(
            f"Test @ tuned {thresh:.4f}: F1={t['f1']:.4f}  P={t['precision']:.4f}  "
            f"R={t['recall']:.4f}  AUROC={t['auroc']:.4f}"
        )
    else:
        eval_out = {}

    # ---- Save ----
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "stage": 3,
            "model_class": "EdgeEnhancedGraphSAGE",
            "loss_class": "FocalLoss",
            "novelty": "Edge-MLP + Focal Loss + Graph-Aware Imbalance Sampler",
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
                "k_hop": k_hop,
                "pos_per_batch": args.pos_per_batch,
                "neg_per_batch": args.neg_per_batch,
                "hard_neg_ratio": args.hard_neg_ratio,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "stage": 3,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "history": history,
                "evaluation": eval_out,
            },
            indent=2,
        )
    )
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
