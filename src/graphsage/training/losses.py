"""Loss functions for severe class imbalance.

Stage 3 of the ablation introduces Focal Loss (Lin et al., ICCV 2017) as a
direct replacement for BCEWithLogitsLoss(pos_weight=...). Where pos_weight
gives a uniform multiplicative bonus to ALL positive examples, Focal Loss
uses (1 - p)^gamma to dynamically focus the loss on HARD examples (low
predicted probability) while down-weighting easy examples. This produces
better-calibrated probabilities and avoids the threshold-inflation
phenomenon seen in Stages 1-2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss with optional alpha balancing.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y == 1 else (1 - p), and alpha_t = alpha if y == 1
    else (1 - alpha).

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 reduces to BCE. Higher gamma focuses
        more on hard examples. Proposal value: 2.0.
    alpha : float | None
        Optional class balancing weight. alpha=0.25 (the Lin et al. default
        for object detection) gives 4x weight to the positive class. For
        PaySim's 773:1 imbalance, alpha is typically tuned higher (e.g.
        0.75 - 0.95) so that fraud examples carry meaningful weight.
        If None, no class balancing is applied.
    reduction : str
        "mean" (default) | "sum" | "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.75,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor [N]
            Raw model outputs (before sigmoid).
        targets : Tensor [N]
            Binary labels in {0, 1}.
        """
        targets = targets.float()
        # Per-element BCE. We use the numerically stable formulation.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: predicted probability of the TRUE class.
        # If y == 1: p_t = sigmoid(logits)
        # If y == 0: p_t = 1 - sigmoid(logits)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation factor
        focal_factor = (1.0 - p_t) ** self.gamma

        loss = focal_factor * bce

        # Alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
