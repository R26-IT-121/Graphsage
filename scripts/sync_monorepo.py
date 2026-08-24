"""Keep this component repo and the team monorepo's GraphSage/ byte-identical.

The two copies drifted once already: work committed here (live inference) never
reached the monorepo, while work committed there (the progress report, the
presentation script, the Streamlit dashboard) never reached here. Neither side
was wrong — nothing was watching. This is that watcher.

This repo is the source of truth. Run --check before every push to the monorepo;
run --apply to mirror.

Usage:
    python scripts/sync_monorepo.py --check
    python scripts/sync_monorepo.py --apply
    python scripts/sync_monorepo.py --check --monorepo /path/to/R26-IT-121
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONOREPO = Path.home() / "Downloads" / "Fusion Engine" / "R26-IT-121"


def tracked_files(root: Path) -> set[Path]:
    """Git's file list, not the filesystem's — build artefacts and the 204 MB
    graph live beside the source and must never be compared or copied."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {Path(line) for line in out.splitlines() if line}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--monorepo", type=Path, default=DEFAULT_MONOREPO)
    ap.add_argument("--apply", action="store_true", help="copy; default is dry-run")
    ap.add_argument("--check", action="store_true", help="exit 1 if drifted")
    args = ap.parse_args()

    dst_root = args.monorepo / "GraphSage"
    if not dst_root.is_dir():
        print(f"error: {dst_root} not found — pass --monorepo", file=sys.stderr)
        return 2

    ours = tracked_files(REPO_ROOT)
    theirs = {
        p.relative_to("GraphSage")
        for p in tracked_files(args.monorepo)
        if p.parts and p.parts[0] == "GraphSage"
    }

    missing = sorted(ours - theirs)                       # here, not there
    extra = sorted(theirs - ours)                         # there, not here
    changed = sorted(
        f for f in ours & theirs
        if (REPO_ROOT / f).exists()
        and not filecmp.cmp(REPO_ROOT / f, dst_root / f, shallow=False)
    )

    for label, files in (("only here", missing), ("only in monorepo", extra),
                         ("content differs", changed)):
        for f in files:
            print(f"  {label:18} {f}")

    if not (missing or extra or changed):
        print(f"in sync — {len(ours)} tracked files identical")
        return 0

    if args.apply:
        for f in missing + changed:
            (dst_root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / f, dst_root / f)
        print(f"\ncopied {len(missing) + len(changed)} file(s) to {dst_root}")
        if extra:
            # Never deleted automatically: the monorepo is where teammates
            # commit, so an unknown file there is more likely their work than
            # our stale leftover.
            print(f"left {len(extra)} monorepo-only file(s) alone — review by hand")
        return 0

    print(f"\n{len(missing) + len(extra) + len(changed)} file(s) out of sync "
          "— run with --apply")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
