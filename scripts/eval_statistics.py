"""Statistical analysis of the temporal ablation — CIs, seeds, paired deltas.

Consumes the *_scores.pt files written by scripts/train_temporal.py and needs
no graph or model, so it runs anywhere in seconds.

Three levels of evidence, strongest first:
1. Multi-seed mean +/- std of test F1 / PR-AUC per stage (training variance).
2. Bootstrap 95% CI per stage (test-set sampling variance): resample the test
   nodes with replacement, recompute the metric, take the 2.5/97.5 percentiles.
3. PAIRED bootstrap of stage differences: the same resampled node indices are
   applied to both stages' predictions, so the delta distribution isolates the
   model difference from sampling noise. If the 95% CI of (full - baseline)
   straddles 0, the honest claim is "no significant difference on this data."

Writes: reports/temporal/statistics.json
Usage:
    python scripts/eval_statistics.py [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = REPO_ROOT / "reports" / "temporal"
STAGE_ORDER = ["1", "2", "3a", "3b"]
STAGE_LABEL = {
    "1": "Stage 1 baseline",
    "2": "Stage 2 +Edge-MLP",
    "3a": "Stage 3a +Focal",
    "3b": "Stage 3b full",
}


def load_runs() -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted(SCORES_DIR.glob("stage*_scores.pt")):
        blob = torch.load(path, weights_only=True, map_location="cpu")
        runs[str(blob["stage"])].append(blob)
    if not runs:
        raise SystemExit(
            f"No score files in {SCORES_DIR} — run scripts/train_temporal.py first."
        )
    return runs


def test_metrics(blob: dict) -> tuple[float, float]:
    probs = torch.sigmoid(blob["test_logits"]).numpy()
    y = blob["test_y"].numpy()
    preds = (probs >= blob["tuned_threshold"]).astype(int)
    return (
        float(f1_score(y, preds, zero_division=0)),
        float(average_precision_score(y, probs)),
    )


def bootstrap_ci(
    probs: np.ndarray, y: np.ndarray, threshold: float, idx: np.ndarray
) -> dict:
    """95% CI of F1 and PR-AUC over precomputed resample index matrix."""
    f1s, aps = [], []
    for sample in idx:
        ys = y[sample]
        if ys.sum() == 0:
            continue
        ps = probs[sample]
        f1s.append(f1_score(ys, (ps >= threshold).astype(int), zero_division=0))
        aps.append(average_precision_score(ys, ps))
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    alo, ahi = np.percentile(aps, [2.5, 97.5])
    return {
        "f1_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "pr_auc_ci95": [round(float(alo), 4), round(float(ahi), 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="RNG for resampling")
    args = parser.parse_args()

    runs = load_runs()
    rng = np.random.default_rng(args.seed)

    report: dict = {"protocol": "temporal_snapshots_leakage_free", "stages": {}}

    # --- level 1: multi-seed aggregates -------------------------------------
    print(f"{'stage':22} {'seeds':>5} {'test F1 (mean±std)':>22} {'PR-AUC (mean±std)':>20}")
    for stage in STAGE_ORDER:
        if stage not in runs:
            continue
        f1s, aps = zip(*(test_metrics(b) for b in runs[stage]))
        entry = {
            "seeds": len(f1s),
            "test_f1_mean": round(float(np.mean(f1s)), 4),
            "test_f1_std": round(float(np.std(f1s)), 4),
            "test_pr_auc_mean": round(float(np.mean(aps)), 4),
            "test_pr_auc_std": round(float(np.std(aps)), 4),
            "per_seed_f1": [round(v, 4) for v in f1s],
        }
        report["stages"][stage] = entry
        print(
            f"{STAGE_LABEL[stage]:22} {entry['seeds']:>5} "
            f"{entry['test_f1_mean']:>13.4f} ±{entry['test_f1_std']:.4f} "
            f"{entry['test_pr_auc_mean']:>12.4f} ±{entry['test_pr_auc_std']:.4f}"
        )

    # --- level 2 + 3: bootstrap on a shared index matrix (seed 0 runs) ------
    first = {s: runs[s][0] for s in runs}
    any_blob = next(iter(first.values()))
    n = len(any_blob["test_y"])
    idx = rng.integers(0, n, size=(args.bootstrap, n))

    print(f"\nBootstrap 95% CIs ({args.bootstrap} resamples, seed-0 runs):")
    probs_by_stage, thr_by_stage = {}, {}
    y = any_blob["test_y"].numpy()
    for stage in STAGE_ORDER:
        if stage not in first:
            continue
        b = first[stage]
        probs_by_stage[stage] = torch.sigmoid(b["test_logits"]).numpy()
        thr_by_stage[stage] = float(b["tuned_threshold"])
        ci = bootstrap_ci(probs_by_stage[stage], y, thr_by_stage[stage], idx)
        report["stages"][stage].update(ci)
        print(
            f"  {STAGE_LABEL[stage]:22} F1 {ci['f1_ci95']}  PR-AUC {ci['pr_auc_ci95']}"
        )

    print("\nPaired deltas vs Stage 1 (same resamples — positive favors the variant):")
    report["paired_deltas_vs_stage1"] = {}
    if "1" in probs_by_stage:
        for stage in STAGE_ORDER[1:]:
            if stage not in probs_by_stage:
                continue
            deltas = []
            for sample in idx:
                ys = y[sample]
                if ys.sum() == 0:
                    continue
                f1_a = f1_score(
                    ys,
                    (probs_by_stage["1"][sample] >= thr_by_stage["1"]).astype(int),
                    zero_division=0,
                )
                f1_b = f1_score(
                    ys,
                    (probs_by_stage[stage][sample] >= thr_by_stage[stage]).astype(int),
                    zero_division=0,
                )
                deltas.append(f1_b - f1_a)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            significant = bool(lo > 0 or hi < 0)
            entry = {
                "delta_f1_mean": round(float(np.mean(deltas)), 4),
                "delta_f1_ci95": [round(float(lo), 4), round(float(hi), 4)],
                "significant_at_95": significant,
            }
            report["paired_deltas_vs_stage1"][stage] = entry
            verdict = "SIGNIFICANT" if significant else "not significant"
            print(
                f"  {STAGE_LABEL[stage]:22} ΔF1 {entry['delta_f1_mean']:+.4f} "
                f"CI {entry['delta_f1_ci95']}  -> {verdict}"
            )

    out = SCORES_DIR / "statistics.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
