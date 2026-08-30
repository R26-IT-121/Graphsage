"""Build a demo/evaluation CSV from the held-out test window.

Why this exists
---------------
The demo file the team has been using (`deepsentinel_batch_test_50.csv`)
is drawn from steps 9–355. The temporal split is:

    train  step   1 – 600
    val    step 601 – 700
    test   step 701 – 743

So that file is training data. Every number measured on it — including
"GraphSAGE shows no separation between fraud and clean" — is a statement
about data the model was fitted on, not about how it generalises. It also
means a panel asking "was the model trained on this?" gets "yes".

This script builds the same kind of file from step 701 onward, so the
demo is a genuine out-of-sample result. That is a stronger story, not a
weaker one.

Usage
-----
    python scripts/make_demo_slice.py \
        --raw data/raw/PS_20174392719_1491204439457_log.csv \
        --out demo_test_window_50.csv \
        --fraud 15 --clean 35

Get the raw file with `python scripts/download_paysim.py` (needs Kaggle
credentials), or copy it from a teammate who has trained on it.

Optionally check that the graph service can actually anchor each row —
a demo transaction that 404s is worse than no demo transaction:

    ... --verify http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

# From src/graphsage/data/splits.py. Kept in sync by hand deliberately:
# this script must not import the package, so it runs anywhere.
TRAIN_END = 600
VAL_END = 700

COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
           "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
           "isFlaggedFraud", "isFraud"]


def resolves(base: str, row: dict) -> bool:
    """Does the graph service have an edge for this pair?"""
    payload = {
        "transaction_id": "slice-probe",
        "step": int(float(row["step"])), "type": row["type"],
        "amount": float(row["amount"]),
        "nameOrig": row["nameOrig"], "nameDest": row["nameDest"],
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "isFlaggedFraud": 0,
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/graph/analyze",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25):
            return True
    except Exception:                                    # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw PaySim CSV")
    ap.add_argument("--out", default="demo_test_window_50.csv")
    ap.add_argument("--fraud", type=int, default=15)
    ap.add_argument("--clean", type=int, default=35)
    ap.add_argument("--from-step", type=int, default=VAL_END + 1,
                    help=f"default {VAL_END + 1}: the test window only")
    ap.add_argument("--verify", metavar="GRAPH_URL",
                    help="drop rows the graph service cannot anchor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        sys.stderr.write(f"{raw} not found. See the docstring for how to get it.\n")
        return 1

    fraud, clean = [], []
    seen = 0
    with raw.open(newline="") as fh:
        for row in csv.DictReader(fh):
            seen += 1
            if int(float(row["step"])) < args.from_step:
                continue
            # Contract §2: no fraud exists outside these two types, so a demo
            # built from the others cannot exercise the detectors.
            if row["type"] not in ("TRANSFER", "CASH_OUT"):
                continue
            (fraud if row.get("isFraud") in ("1", "1.0") else clean).append(row)

    print(f"  read {seen:,} rows")
    print(f"  step >= {args.from_step}: {len(fraud)} fraud, {len(clean)} clean")
    if not fraud:
        sys.stderr.write(
            f"No fraud at or after step {args.from_step}. Widen with "
            f"--from-step {TRAIN_END + 1} to include validation, but say so "
            "wherever the results are reported.\n")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(fraud)
    rng.shuffle(clean)

    if args.verify:
        print(f"  checking each row anchors in the graph at {args.verify} ...")
        fraud = [r for r in fraud if resolves(args.verify, r)][: args.fraud]
        clean = [r for r in clean if resolves(args.verify, r)][: args.clean]
    else:
        fraud, clean = fraud[: args.fraud], clean[: args.clean]

    picked = fraud + clean
    rng.shuffle(picked)                                  # never label-ordered

    if not picked:
        sys.stderr.write(
            "Nothing left to write.\n"
            + ("Every candidate was dropped by --verify: the graph service "
               "could not anchor a single one. That usually means the rows "
               "are not real PaySim edges, or the serving bundle was built "
               "from a different extract.\n" if args.verify else
               "No rows matched the filters.\n"))
        return 1
    if args.verify and (len(fraud) < args.fraud or len(clean) < args.clean):
        print(f"  note: --verify kept {len(fraud)}/{args.fraud} fraud and "
              f"{len(clean)}/{args.clean} clean; the rest had no edge in the graph")

    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            r.setdefault("isFlaggedFraud", "0")
            w.writerow(r)

    steps = [int(float(r["step"])) for r in picked]
    print()
    print(f"  wrote {out}")
    print(f"    {len(picked)} rows — {len(fraud)} fraud, {len(clean)} clean")
    print(f"    steps {min(steps)}–{max(steps)}  "
          f"({'test window only' if min(steps) > VAL_END else 'includes validation'})")
    print(f"    fraud rate {len(fraud) / len(picked):.1%} — enriched far above "
          "PaySim's 0.129%, so use it for demonstration, and quote alert "
          "volumes from a true-rate sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
