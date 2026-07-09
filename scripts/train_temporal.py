"""Train any ablation stage under the leakage-free temporal protocol.

Uses the snapshot graphs from scripts/build_temporal_graph.py: features and
message passing see only past edges, labels only their own window. Supports
multi-seed runs for statistical reporting (mean +/- std across seeds).

Stages:
    1   BaselineGraphSAGE, BCE + pos_weight        (full-batch)
    2   EdgeEnhancedGraphSAGE, BCE + pos_weight    (full-batch)
    3a  EdgeEnhancedGraphSAGE, Focal Loss          (full-batch)
    3b  EdgeEnhancedGraphSAGE, Focal + Sampler     (balanced mini-batches)

Reads:  data/graph/paysim_temporal.pt
Writes: reports/temporal/stage{S}_seed{N}.json
        checkpoints/temporal_stage{S}_seed{N}.pt

Usage:
    python scripts/train_temporal.py --stage 1 --seed 0
    python scripts/train_temporal.py --stage 3b --seed 2 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from graphsage.models.baseline import BaselineGraphSAGE
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE
from graphsage.sampling.imbalance_sampler import GraphAwareImbalanceSampler
from graphsage.training.losses import FocalLoss
from graphsage.training.threshold_tuning import (
    find_best_threshold_for_f1,
    metrics_at_threshold,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP_PATHS = {
    "v1": REPO_ROOT / "data" / "graph" / "paysim_temporal.pt",
    "v2": REPO_ROOT / "data" / "graph" / "paysim_temporal_v2.pt",
}

STAGES = ("1", "2", "3a", "3b")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward(model, snap, device) -> torch.Tensor:
    if isinstance(model, BaselineGraphSAGE):
        return model(snap.x.to(device), snap.edge_index.to(device))
    return model(
        snap.x.to(device), snap.edge_index.to(device), snap.edge_attr.to(device)
    )


def eval_snapshot(model, snap, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) restricted to the snapshot's eval nodes."""
    model.eval()
    with torch.no_grad():
        logits = forward(model, snap, device)
    mask = snap.eval_mask
    return logits[mask].cpu(), snap.y[mask].to(torch.float32).cpu()


def full_metrics(logits: torch.Tensor, y: torch.Tensor, threshold: float) -> dict:
    m = metrics_at_threshold(logits, y, threshold)
    probs = torch.sigmoid(logits).numpy()
    y_np = y.numpy()
    m["auroc"] = float(roc_auc_score(y_np, probs))
    m["pr_auc"] = float(average_precision_score(y_np, probs))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pos-per-batch", type=int, default=128)
    parser.add_argument("--neg-per-batch", type=int, default=128)
    parser.add_argument(
        "--features",
        choices=("v1", "v2"),
        default="v1",
        help="v1 = 5 structural aggregates; v2 = 12-dim behavioural set",
    )
    parser.add_argument(
        "--no-prior-init",
        action="store_true",
        help="Disable Focal Loss prior bias init (Lin et al. 2017) on stages 3a/3b",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device()
    print(f"stage={args.stage} seed={args.seed} device={device}")

    snap_path = SNAP_PATHS[args.features]
    snaps = torch.load(snap_path, weights_only=False, map_location="cpu")
    train, val, test = snaps["train"], snaps["val"], snaps["test"]
    in_dim = train.x.shape[1]
    edge_dim = train.edge_attr.shape[1]
    print(f"features={args.features} ({snap_path.name}, in_dim={in_dim})")

    if args.stage == "1":
        model = BaselineGraphSAGE(in_dim=in_dim, hidden_dim=args.hidden_dim)
    else:
        model = EdgeEnhancedGraphSAGE(
            in_dim=in_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim
        )

    # Focal Loss prior init (Lin et al. 2017), ported from the May-17 fix in
    # the team repo: start the classifier bias at log(pi/(1-pi)) so the
    # optimiser begins at the class prior instead of sigmoid(0)=0.5 — prevents
    # the inverted-basin convergence and reduces the raw-probability inflation
    # quantified in calibration_study.py.
    prior_init = args.stage in ("3a", "3b") and not args.no_prior_init
    if prior_init:
        pi = float(train.y[train.eval_mask].float().mean())
        pi = max(min(pi, 0.5), 1e-4)
        model.classifier.bias.data.fill_(float(np.log(pi / (1 - pi))))
        print(f"prior init: pi={pi:.5f}, bias={float(model.classifier.bias):.3f}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_y = train.y.to(torch.float32)
    train_mask = train.eval_mask
    n_pos = int(train.y.sum())
    n_neg = int(train_mask.sum()) - n_pos

    if args.stage in ("1", "2"):
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(n_neg / max(n_pos, 1)).to(device)
        )
    else:
        loss_fn = FocalLoss(gamma=2.0, alpha=0.95)

    sampler = None
    if args.stage == "3b":
        sampler = GraphAwareImbalanceSampler(
            train,
            k_hop=2,
            pos_per_batch=args.pos_per_batch,
            neg_per_batch=args.neg_per_batch,
            hard_negative_ratio=1.0,
            seed=args.seed,
        )

    best_val_f1, best_epoch, best_state = -1.0, -1, None
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if sampler is None:
            optimizer.zero_grad()
            logits = forward(model, train, device)
            loss = loss_fn(
                logits[train_mask], train_y[train_mask].to(device)
            )
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())
        else:
            losses = []
            for _ in range(sampler.steps_per_epoch):
                batch = sampler.sample()
                optimizer.zero_grad()
                logits_sub = model(
                    batch.x.to(device),
                    batch.edge_index.to(device),
                    batch.edge_attr.to(device),
                )
                loss = loss_fn(
                    logits_sub[batch.seed_local_idx.to(device)],
                    batch.y.to(device),
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            train_loss = float(np.mean(losses))

        val_logits, val_y = eval_snapshot(model, val, device)
        thr, val_f1 = find_best_threshold_for_f1(val_logits, val_y)
        probs = torch.sigmoid(val_logits).numpy()
        val_ap = float(average_precision_score(val_y.numpy(), probs))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1_tuned": val_f1,
                "val_pr_auc": val_ap,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(
            f"epoch {epoch:>3}  loss {train_loss:8.5f}  "
            f"val F1(tuned) {val_f1:.4f}  val PR-AUC {val_ap:.4f}"
            f"  [{history[-1]['elapsed_s']}s]{marker}"
        )
        if epoch - best_epoch >= args.patience:
            print(f"early stop (no val improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)

    # Threshold tuned on val, frozen for test — with PR-AUC everywhere.
    val_logits, val_y = eval_snapshot(model, val, device)
    threshold, _ = find_best_threshold_for_f1(val_logits, val_y)
    test_logits, test_y = eval_snapshot(model, test, device)
    results = {
        "protocol": "temporal_snapshots_leakage_free",
        "stage": args.stage,
        "seed": args.seed,
        "features": args.features,
        "prior_init": prior_init,
        "best_epoch": best_epoch,
        "tuned_threshold": threshold,
        "val": full_metrics(val_logits, val_y, threshold),
        "test": full_metrics(test_logits, test_y, threshold),
        "history": history,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "loss": type(loss_fn).__name__,
            "sampler": sampler is not None,
        },
    }

    out_dir = REPO_ROOT / "reports" / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"stage{args.stage}{'_v2' if args.features == 'v2' else ''}_seed{args.seed}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Raw scores let eval_statistics.py / calibration_study.py run without
    # reloading the graph or model (they bootstrap from these tensors).
    torch.save(
        {
            "val_logits": val_logits,
            "val_y": val_y,
            "test_logits": test_logits,
            "test_y": test_y,
            "tuned_threshold": threshold,
            "stage": args.stage,
            "seed": args.seed,
            "features": args.features,
        },
        out_dir / f"{tag}_scores.pt",
    )

    ckpt_path = REPO_ROOT / "checkpoints" / f"temporal_{tag}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "stage": args.stage,
            "seed": args.seed,
            "features": args.features,
            "prior_init": prior_init,
            "protocol": "temporal_snapshots_leakage_free",
            "hyperparameters": results["hyperparameters"],
        },
        ckpt_path,
    )
    t = results["test"]
    print(
        f"\nTEST  F1 {t['f1']:.4f}  P {t['precision']:.4f}  R {t['recall']:.4f}  "
        f"AUROC {t['auroc']:.4f}  PR-AUC {t['pr_auc']:.4f}"
    )
    print(f"wrote {out_path.relative_to(REPO_ROOT)} and {ckpt_path.name}")


if __name__ == "__main__":
    main()
