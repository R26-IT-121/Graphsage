"""Download the PaySim dataset from Kaggle and put it where the pipeline reads it.

Usage:
    python scripts/download_paysim.py

Requires Kaggle API credentials. Either:
  - Set KAGGLE_USERNAME and KAGGLE_KEY env vars (see .env.example), or
  - Place ~/.kaggle/kaggle.json from https://www.kaggle.com/settings/account

The dataset is `ealaxi/paysim1` (~470 MB CSV).

kagglehub downloads into its own cache and returns that path — on Colab that is
/kaggle/input/paysim1, nowhere near this repo. This script used to stop there and
print "symlink or copy it if needed", so the very next step, prepare_features.py,
failed with FileNotFoundError every time. It now links the CSV into data/raw/
under the exact name the pipeline expects, and says what it did.
"""

import os
import shutil
import sys
from pathlib import Path

# The filename scripts/prepare_features.py opens. Kaggle ships exactly this name;
# it is pinned here so a renamed or re-uploaded dataset fails loudly rather than
# half-working.
EXPECTED = "PS_20174392719_1491204439457_log.csv"


def main() -> int:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / EXPECTED

    if target.exists():
        print(f"Already present: {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return 0

    try:
        import kagglehub
    except ImportError:
        sys.stderr.write(
            "kagglehub is not installed. Run `pip install -e .` from the repo root first.\n"
        )
        return 1

    print(f"Downloading ealaxi/paysim1 ...")
    cached = Path(kagglehub.dataset_download("ealaxi/paysim1"))
    print(f"Cached at: {cached}")

    csvs = sorted(cached.rglob("*.csv"))
    if not csvs:
        sys.stderr.write(f"No CSV found under {cached}\n")
        return 1

    src = next((c for c in csvs if c.name == EXPECTED), None)
    if src is None:
        # The dataset shipped under a different name. Take the largest CSV, but
        # say so — a silent rename is how the wrong file ends up being trained on.
        src = max(csvs, key=lambda c: c.stat().st_size)
        print(f"warning: expected {EXPECTED}, found {src.name} — using it")

    # A symlink keeps 470 MB off the Colab disk. Fall back to a copy where
    # symlinks are unavailable (some Windows setups, some mounted volumes).
    try:
        target.symlink_to(src)
        how = "linked"
    except (OSError, NotImplementedError):
        shutil.copy2(src, target)
        how = "copied"

    size = target.stat().st_size / 1e6
    print(f"{how} -> {target} ({size:.0f} MB)")
    if size < 400:
        sys.stderr.write(f"warning: expected ~470 MB, got {size:.0f} MB\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
