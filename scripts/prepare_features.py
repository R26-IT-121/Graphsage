"""Compute engineered edge features from PaySim and save to parquet.

Bridges the EDA findings (reports/eda_findings.md §11) into a reusable artifact
that the graph builder (T4) consumes. Run once, reuse forever.

Usage:
    python scripts/prepare_features.py

Output:
    data/processed/features.parquet   (all rows, 10 columns, ~150 MB compressed)
    data/processed/feature_metadata.json  (sanity stats)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"
OUT_PARQUET = OUT_DIR / "features.parquet"
OUT_META = OUT_DIR / "feature_metadata.json"

DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "category",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "category",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found at {RAW_PATH}")

    print(f"Loading {RAW_PATH.name}...")
    t0 = time.time()
    df = pd.read_csv(RAW_PATH, dtype=DTYPES)
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # === Filter to TRANSFER + CASH_OUT (EDA report §3) ===
    before = len(df)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"  Filtered TRANSFER+CASH_OUT: {len(df):,} rows ({before - len(df):,} dropped)")

    # Sanity: 100% of fraud should survive
    fraud_count = int(df["isFraud"].sum())
    assert fraud_count == 8213, f"Lost fraud during filter: got {fraud_count}, expected 8213"
    print(f"  Fraud preserved: {fraud_count:,} (100%)")

    # === Compute the 6 edge features (EDA report §11) ===
    print("\nComputing edge features...")

    # 1. amount_log  — log1p tames the heavy tail (max ~92M)
    df["amount_log"] = np.log1p(df["amount"]).astype("float32")

    # 2. drain_ratio — fraction of sender balance moved (clamped to [0, 1])
    df["drain_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        np.clip(df["amount"] / df["oldbalanceOrg"], 0.0, 1.0),
        0.0,
    ).astype("float32")

    # 3. src_drained — sender's balance hits 0 after the transfer
    df["src_drained"] = (df["newbalanceOrig"] == 0).astype("int8")

    # 4. dst_was_empty — receiver had 0 balance before the transfer (mule signature)
    df["dst_was_empty"] = (df["oldbalanceDest"] == 0).astype("int8")

    # 5. time_gap — hours since this destination's previous inbound transfer
    #    EDA §7 showed senders are one-shot, so per-destination gap is the right signal.
    #    -1 sentinel means "first inbound for this destination" (no prior gap).
    df = df.sort_values(["nameDest", "step"], kind="stable")
    df["time_gap"] = (
        df.groupby("nameDest", observed=True)["step"]
        .diff()
        .fillna(-1)
        .astype("float32")
    )

    # 6. type_is_transfer — 1 for TRANSFER, 0 for CASH_OUT
    df["type_is_transfer"] = (df["type"] == "TRANSFER").astype("int8")

    print("  Done (6 features added)")

    # === Restore chronological order and select output columns ===
    df = df.sort_values("step", kind="stable").reset_index(drop=True)

    output_cols = [
        "step",
        "nameOrig",
        "nameDest",
        "isFraud",
        "amount_log",
        "drain_ratio",
        "src_drained",
        "dst_was_empty",
        "time_gap",
        "type_is_transfer",
    ]
    out = df[output_cols].copy()

    # Account IDs as plain strings (more portable across Parquet readers)
    out["nameOrig"] = out["nameOrig"].astype(str)
    out["nameDest"] = out["nameDest"].astype(str)

    # === Save ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {OUT_PARQUET}...")
    t0 = time.time()
    out.to_parquet(OUT_PARQUET, compression="snappy", index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1024**2
    print(f"  Wrote {size_mb:.1f} MB in {time.time() - t0:.1f}s")

    # === Sanity & metadata ===
    senders = out["nameOrig"].nunique()
    receivers = out["nameDest"].nunique()
    all_accounts = set(out["nameOrig"]).union(out["nameDest"])

    metadata = {
        "rows": int(len(out)),
        "fraud_count": int(out["isFraud"].sum()),
        "fraud_rate_pct": float(out["isFraud"].mean() * 100),
        "unique_senders": int(senders),
        "unique_receivers": int(receivers),
        "unique_accounts_total": int(len(all_accounts)),
        "step_min": int(out["step"].min()),
        "step_max": int(out["step"].max()),
        "feature_columns": [
            "amount_log",
            "drain_ratio",
            "src_drained",
            "dst_was_empty",
            "time_gap",
            "type_is_transfer",
        ],
        "filter": "type in {TRANSFER, CASH_OUT}",
        "sentinel_time_gap_first_inbound": -1.0,
    }
    OUT_META.write_text(json.dumps(metadata, indent=2))
    print(f"  Wrote metadata to {OUT_META.name}")

    # === Console summary ===
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Rows:               {metadata['rows']:>12,}")
    print(f"  Fraud rate:         {metadata['fraud_rate_pct']:>12.4f}%")
    print(f"  Unique senders:     {metadata['unique_senders']:>12,}")
    print(f"  Unique receivers:   {metadata['unique_receivers']:>12,}")
    print(f"  Unique nodes:       {metadata['unique_accounts_total']:>12,}")
    print(f"  Edges (= rows):     {metadata['rows']:>12,}")
    print(f"  Step range:         {metadata['step_min']} → {metadata['step_max']}")
    print()
    print("First 3 rows:")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
