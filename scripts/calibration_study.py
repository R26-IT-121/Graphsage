"""Calibration study — three score transforms compared on the temporal test set.

Motivation (found while building the demo): Focal Loss with alpha=0.95 inflates
raw sigmoid outputs so mule and non-mule score distributions overlap heavily
even though ranking is good. This script quantifies three options:

1. Raw sigmoid — what the model emits.
2. ECDF percentile rank — what the API currently serves. Monotone, so ranking
   is untouched and scores become comparable across accounts, but it is a RANK
   and does NOT produce calibrated probabilities.
3. Isotonic regression fitted on validation labels — equally monotone (same
   AUROC) and genuinely probability-calibrated.

Outputs reliability diagrams (equal-count bins) and Expected Calibration Error
for each. All transforms are fitted on the VALIDATION snapshot and applied to
test, matching the serving protocol (fit on one population, apply to another).

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
C_RAW, C_CAL, C_ISO = "#2a78d6", "#1baf7a", "#eda100"
C_REF, C_INK = "#c3c2b7", "#52514e"


def ecdf_calibrate(fit_scores: np.ndarray, apply_scores: np.ndarray) -> np.ndarray:
    """Percentile rank. Monotone, so ranking/AUROC are unchanged — but the
    output is a RANK, not a probability: it spreads scores uniformly over
    [0,1] while the true positive rate stays near the base rate, so it does
    not by itself produce calibrated probabilities."""
    ref = np.sort(fit_scores)
    return np.searchsorted(ref, apply_scores, side="right") / len(ref)


def isotonic_calibrate(
    fit_scores: np.ndarray, fit_y: np.ndarray, apply_scores: np.ndarray
) -> np.ndarray:
    """Isotonic regression fitted on validation labels — the standard fix when
    a model ranks well but its scores are not probabilities. Also monotone,
    so it preserves ranking exactly like the ECDF transform."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(fit_scores, fit_y)
    return iso.predict(apply_scores)


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
    val_y = blob["val_y"].numpy()
    test_probs = torch.sigmoid(blob["test_logits"]).numpy()
    test_y = blob["test_y"].numpy()
    test_ecdf = ecdf_calibrate(val_probs, test_probs)
    test_iso = isotonic_calibrate(val_probs, val_y, test_probs)

    # The observed mule rate is what a perfectly calibrated score predicts on
    # average, so compare each variant's ECE against the base rate context.
    base_rate = float(test_y.mean())
    arms = [
        ("Raw focal-loss probability", test_probs, C_RAW),
        ("ECDF percentile score", test_ecdf, C_CAL),
        ("Isotonic-calibrated", test_iso, C_ISO),
    ]
    curves, eces = [], {}
    for label, scores, color in arms:
        pred, obs, w = reliability(scores, test_y, args.bins)
        e = ece(pred, obs, w)
        curves.append((pred, obs, e, label, color))
        eces[label] = e

    result = {
        "stage": args.stage,
        "seed": args.seed,
        "bins": args.bins,
        "test_base_rate": round(base_rate, 4),
        "ece_raw": round(eces["Raw focal-loss probability"], 4),
        "ece_ecdf_calibrated": round(eces["ECDF percentile score"], 4),
        "ece_isotonic_calibrated": round(eces["Isotonic-calibrated"], 4),
        "raw_score_range": [round(float(test_probs.min()), 4),
                            round(float(test_probs.max()), 4)],
        "note": (
            "ECDF (the current serving transform) is a RANK, not a probability: "
            "it makes scores comparable across accounts but does not fix "
            "calibration. Isotonic regression, fitted on validation labels and "
            "equally monotone (so ranking/AUROC are identical), is what yields "
            "probability-calibrated scores."
        ),
    }
    (SCORES_DIR / "calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (pred, obs, e, label, color) in zip(axes, curves):
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
