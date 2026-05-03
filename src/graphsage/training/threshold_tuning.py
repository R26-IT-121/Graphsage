"""Post-training threshold tuning for severe class imbalance.

Standard practice in fraud detection: the default 0.5 decision threshold is
almost never optimal under severe class imbalance. With pos_weight set so
high (~454 in our case), the model produces inflated logits and predicts
positive too aggressively, giving high recall but low precision.

This module sweeps thresholds on the VALIDATION set, picks the threshold
that maximises val F1, and applies it to the TEST set. This is NOT cheating
- the test set is never used to choose the threshold.
"""

from __future__ import annotations

import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold_for_f1(
    logits: torch.Tensor, y_true: torch.Tensor
) -> tuple[float, float]:
    """Return (best_threshold, val_f1_at_that_threshold).

    Uses sklearn's precision_recall_curve which sweeps all unique thresholds
    efficiently (no brute-force grid search).
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y_true.cpu().numpy()

    if len(set(y_np)) < 2:
        # Only one class present in val — can't tune meaningfully.
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_np, probs)
    # F1 at each threshold (broadcast). Exclude last point which has no threshold.
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    f1s = f1s[:-1]
    best_idx = int(f1s.argmax())
    return float(thresholds[best_idx]), float(f1s[best_idx])


def metrics_at_threshold(
    logits: torch.Tensor, y_true: torch.Tensor, threshold: float
) -> dict[str, float]:
    """Compute precision/recall/F1/AUROC at a specific threshold."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    y_np = y_true.cpu().numpy()
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_np, preds, zero_division=0)),
        "recall": float(recall_score(y_np, preds, zero_division=0)),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
    }
    if len(set(y_np)) > 1:
        out["auroc"] = float(roc_auc_score(y_np, probs))
    else:
        out["auroc"] = 0.0
    return out


def evaluate_with_tuned_threshold(
    val_logits: torch.Tensor,
    val_y: torch.Tensor,
    test_logits: torch.Tensor,
    test_y: torch.Tensor,
) -> dict:
    """End-to-end: tune on val, evaluate on test at default + tuned thresholds."""
    # Default threshold (what trainer.py reports)
    default_test = metrics_at_threshold(test_logits, test_y, 0.5)
    default_val = metrics_at_threshold(val_logits, val_y, 0.5)

    # Find best threshold on VAL set only
    best_thresh, val_f1_tuned = find_best_threshold_for_f1(val_logits, val_y)

    # Apply to test
    tuned_test = metrics_at_threshold(test_logits, test_y, best_thresh)
    tuned_val = metrics_at_threshold(val_logits, val_y, best_thresh)

    return {
        "best_threshold": best_thresh,
        "val_f1_at_tuned_threshold": val_f1_tuned,
        "default_threshold_metrics": {
            "val": default_val,
            "test": default_test,
        },
        "tuned_threshold_metrics": {
            "val": tuned_val,
            "test": tuned_test,
        },
    }
