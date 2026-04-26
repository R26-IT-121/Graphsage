"""Download the PaySim dataset from Kaggle into data/raw/.

Usage:
    python scripts/download_paysim.py

Requires Kaggle API credentials. Either:
  - Set KAGGLE_USERNAME and KAGGLE_KEY env vars (see .env.example), or
  - Place ~/.kaggle/kaggle.json from https://www.kaggle.com/settings/account

The dataset is `ealaxi/paysim1` (~470 MB CSV).
"""

import os
import sys
from pathlib import Path


def main() -> int:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        sys.stderr.write(
            "kagglehub is not installed. Run `pip install -e .` from the repo root first.\n"
        )
        return 1

    print(f"Downloading ealaxi/paysim1 into {raw_dir} ...")
    cached_path = kagglehub.dataset_download("ealaxi/paysim1")
    print(f"Dataset cached at: {cached_path}")
    print("Note: kagglehub stores the dataset in its own cache directory.")
    print(f"Symlink or copy the CSV from {cached_path} into {raw_dir} if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
