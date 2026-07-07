"""Calibration study — raw focal-loss probabilities vs ECDF percentile ranks.

Motivation (found while building the demo): Focal Loss with alpha=0.95 inflates
raw sigmoid outputs so mule and non-mule score distributions overlap heavily
even though ranking is good. The API therefore serves percentile-rank (ECDF)
calibrated scores. This script quantifies that decision:

- Reliability diagrams (equal-count bins): predicted score vs observed mule
  fraction, raw vs calibrated.
- Expected Calibration Error (ECE) for both.

The ECDF is fitted on the VALIDATION snapshot scores and applied to test —
same protocol as the serving path (fit on one population, apply to another).

Consumes the *_scores.pt files from scripts/train_temporal.py (no graph/model
needed). Writes reports/temporal/calibration.json + calibration.png.

Usage:
    python scripts/calibration_study.py [--stage 3b] [--seed 0] [--bins 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = REPO_ROOT / "reports" / "temporal"

# Reference palette (dataviz skill): blue = raw, aqua = calibrated.
C_RAW, C_CAL, C_REF, C_INK = "#2a78d6", "#1baf7a", "#c3c2b7", "#52514e"


def ecdf_calibrate(fit_scores: np.ndarray, apply_scores: np.ndarray) -> np.ndarray:
    ref = np.sort(fit_scores)
    return np.searchsorted(ref, apply_scores, side="right") / len(ref)


def reliability(scores: np.ndarray, y: np.ndarray, n_bins: int):
    """Equal-count binning -> (mean predicted, observed positive rate, weight)."""
    order = np.argsort(scores)
    bins = np.array_split(order, n_bins)
    pred = np.array([scores[b].mean() for b in bins])
    obs = np.array([y[b].mean() for b in bins])
    w = np.array([len(b) / len(y) for b in bins])
    return pred, obs, w


def ece(pred: np.ndarray, obs: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(pred - obs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="3b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    path = SCORES_DIR / f"stage{args.stage}_seed{args.seed}_scores.pt"
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/train_temporal.py first.")
    blob = torch.load(path, weights_only=True, map_location="cpu")

    val_probs = torch.sigmoid(blob["val_logits"]).numpy()
    test_probs = torch.sigmoid(blob["test_logits"]).numpy()
    test_y = blob["test_y"].numpy()
    test_cal = ecdf_calibrate(val_probs, test_probs)

    # The observed mule rate is what a perfectly calibrated score predicts on
    # average, so compare each variant's ECE against the base rate context.
    base_rate = float(test_y.mean())
    pred_r, obs_r, w_r = reliability(test_probs, test_y, args.bins)
    pred_c, obs_c, w_c = reliability(test_cal, test_y, args.bins)
    ece_raw, ece_cal = ece(pred_r, obs_r, w_r), ece(pred_c, obs_c, w_c)

    result = {
        "stage": args.stage,
        "seed": args.seed,
        "bins": args.bins,
        "test_base_rate": round(base_rate, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_ecdf_calibrated": round(ece_cal, 4),
        "raw_score_range": [round(float(test_probs.min()), 4),
                            round(float(test_probs.max()), 4)],
        "note": (
            "ECDF fitted on validation scores, applied to test — matches the "
            "serving path in graphsage.inference.predictor"
        ),
    }
    (SCORES_DIR / "calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (pred, obs, e, label, color) in zip(
        axes,
        [
            (pred_r, obs_r, ece_raw, "Raw focal-loss probability", C_RAW),
            (pred_c, obs_c, ece_cal, "ECDF percentile score", C_CAL),
        ],
    ):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=C_REF, zorder=1)
        ax.plot(pred, obs, marker="o", ms=5, lw=2, color=color, zorder=2)
        ax.set_title(f"{label}\nECE = {e:.3f}", fontsize=10)
        ax.set_xlabel("mean predicted score (bin)")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_facecolor("#fcfcfb")
    axes[0].set_ylabel("observed mule fraction (bin)")
    fig.suptitle(
        f"Reliability — Stage {args.stage}, temporal test window "
        f"(base rate {base_rate:.3f})",
        fontsize=11, color=C_INK,
    )
    fig.tight_layout()
    out_png = SCORES_DIR / "calibration.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
