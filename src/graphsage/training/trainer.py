"""Reusable full-batch training loop for binary node classification.

Designed for the 3-stage ablation:
    Stage 1: BaselineGraphSAGE + BCEWithLogitsLoss(pos_weight)
    Stage 2: same + EdgeEnhancedSAGEConv (edge_attr passed through)
    Stage 3: same + Focal Loss + Graph-Aware Imbalance Sampler

The trainer auto-detects the device (CUDA / MPS / CPU). Works on Mac, PC, and
Kaggle without modification.

Returns a dict of per-epoch metrics for plotting and the best validation
checkpoint state_dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data


def select_device() -> torch.device:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_auroc: float
    elapsed_s: float


@dataclass
class TrainResult:
    history: list[EpochMetrics] = field(default_factory=list)
    best_state_dict: dict = field(default_factory=dict)
    best_val_f1: float = 0.0
    best_epoch: int = -1
    final_test_metrics: dict = field(default_factory=dict)


def _binary_metrics(logits: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
    """Compute precision, recall, F1, AUROC at threshold 0.5."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_np = y_true.cpu().numpy()
    out = {
        "precision": float(precision_score(y_np, preds, zero_division=0)),
        "recall": float(recall_score(y_np, preds, zero_division=0)),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
    }
    # AUROC needs both classes present
    if len(set(y_np)) > 1:
        out["auroc"] = float(roc_auc_score(y_np, probs))
    else:
        out["auroc"] = 0.0
    return out


def train_node_classifier(
    model: nn.Module,
    data: Data,
    *,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
    use_edge_attr: bool = False,
    loss_fn: Callable | None = None,
    device: torch.device | None = None,
    log_every: int = 1,
) -> TrainResult:
    """Full-batch training with early stopping.

    Parameters
    ----------
    model : nn.Module
        Forward signature must be model(x, edge_index) for Stage 1
        OR model(x, edge_index, edge_attr) when use_edge_attr=True.
    data : torch_geometric.data.Data
        Must contain x, edge_index, y, train_mask, val_mask, test_mask.
        Optional: edge_attr (required if use_edge_attr=True).
    loss_fn : callable, optional
        If None, uses BCEWithLogitsLoss with positive class weight set
        to N_negative / N_positive on the training mask.
    """
    if device is None:
        device = select_device()
    print(f"Device: {device}")

    # Move graph to device. The graph is large (3.27M nodes); MPS may have
    # issues with such big tensors. We try GPU first and fall back to CPU
    # gracefully if an OOM occurs.
    try:
        data = data.to(device)
        model = model.to(device)
    except (RuntimeError, MemoryError) as exc:
        print(f"  Could not move graph to {device}: {exc}")
        print("  Falling back to CPU.")
        device = torch.device("cpu")
        data = data.to(device)
        model = model.to(device)

    # Default loss: BCE with positive class weight to handle imbalance
    y_train = data.y[data.train_mask].float()
    if loss_fn is None:
        n_pos = float(y_train.sum().item())
        n_neg = float(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.1f})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    result = TrainResult()
    epochs_no_improve = 0
    y_full = data.y.float()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        if use_edge_attr:
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
        train_loss = loss_fn(logits[data.train_mask], y_full[data.train_mask])
        train_loss.backward()
        optimizer.step()

        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            if use_edge_attr:
                logits_eval = model(data.x, data.edge_index, data.edge_attr)
            else:
                logits_eval = model(data.x, data.edge_index)
            val_loss = loss_fn(logits_eval[data.val_mask], y_full[data.val_mask])
            val_metrics = _binary_metrics(
                logits_eval[data.val_mask], data.y[data.val_mask]
            )

        elapsed = time.time() - t0
        epoch_m = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss.item()),
            val_loss=float(val_loss.item()),
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_f1=val_metrics["f1"],
            val_auroc=val_metrics["auroc"],
            elapsed_s=elapsed,
        )
        result.history.append(epoch_m)

        if epoch % log_every == 0:
            print(
                f"  epoch {epoch:>3} | train_loss {epoch_m.train_loss:.4f} | "
                f"val_loss {epoch_m.val_loss:.4f} | val_F1 {epoch_m.val_f1:.4f} | "
                f"P {epoch_m.val_precision:.4f} | R {epoch_m.val_recall:.4f} | "
                f"AUROC {epoch_m.val_auroc:.4f} | {elapsed:.1f}s"
            )

        # ---- Early stopping on val F1 ----
        if epoch_m.val_f1 > result.best_val_f1:
            result.best_val_f1 = epoch_m.val_f1
            result.best_epoch = epoch
            result.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"  Early stopping: no val F1 improvement for "
                    f"{early_stopping_patience} epochs. Best F1={result.best_val_f1:.4f} "
                    f"at epoch {result.best_epoch}."
                )
                break

    # ---- Final test eval with best weights ----
    if result.best_state_dict:
        model.load_state_dict(result.best_state_dict)
        model.eval()
        with torch.no_grad():
            if use_edge_attr:
                logits_test = model(data.x, data.edge_index, data.edge_attr)
            else:
                logits_test = model(data.x, data.edge_index)
            test_metrics = _binary_metrics(
                logits_test[data.test_mask], data.y[data.test_mask]
            )
        result.final_test_metrics = test_metrics
        print(
            f"\nTest (best epoch {result.best_epoch}): "
            f"F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
            f"R={test_metrics['recall']:.4f}  AUROC={test_metrics['auroc']:.4f}"
        )

    return result


def save_checkpoint(
    result: TrainResult, path: str | Path, extra: dict | None = None
) -> None:
    """Save the best model weights and metadata to a single .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.best_state_dict,
            "best_val_f1": result.best_val_f1,
            "best_epoch": result.best_epoch,
            "history": [vars(m) for m in result.history],
            "final_test_metrics": result.final_test_metrics,
            **(extra or {}),
        },
        path,
    )
    size_mb = path.stat().st_size / 1024**2
    print(f"Checkpoint saved to {path}  ({size_mb:.1f} MB)")
