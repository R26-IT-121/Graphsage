"""Keep this component repo and the team monorepo's GraphSage/ byte-identical.

The two copies drifted once already: work committed here (live inference) never
reached the monorepo, while work committed there (the progress report, the
presentation script, the Streamlit dashboard) never reached here. Neither side
was wrong — nothing was watching. This is that watcher.

Either side can be the source of truth; say which. As of 2026-09-01 the user
works mainly in the monorepo, so --from-monorepo is the usual direction and
mirrors GraphSage/ back into this repo before pushing to the org remote.

Usage:
    python scripts/sync_monorepo.py --check                  # this repo -> monorepo
    python scripts/sync_monorepo.py --apply
    python scripts/sync_monorepo.py --from-monorepo --check  # monorepo -> this repo
    python scripts/sync_monorepo.py --from-monorepo --apply
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
    ap.add_argument("--from-monorepo", action="store_true",
                    help="treat the monorepo's GraphSage/ as the source of truth "
                         "and mirror it into this repo (the usual direction now)")
    args = ap.parse_args()

    dst_root = args.monorepo / "GraphSage"
    if not dst_root.is_dir():
        print(f"error: {dst_root} not found — pass --monorepo", file=sys.stderr)
        return 2

    # Repo-scoped config that is correctly different on each side: the
    # monorepo's ignore paths are prefixed with GraphSage/ and it carries rules
    # for other members' folders. Mirroring it would break both.
    REPO_LOCAL = {Path(".gitignore")}

    ours = tracked_files(REPO_ROOT) - REPO_LOCAL
    theirs = {
        p.relative_to("GraphSage")
        for p in tracked_files(args.monorepo)
        if p.parts and p.parts[0] == "GraphSage"
    } - REPO_LOCAL

    missing = sorted(ours - theirs)                       # here, not there
    extra = sorted(theirs - ours)                         # there, not here
    changed = sorted(
        f for f in ours & theirs
        if (REPO_ROOT / f).exists()
        and not filecmp.cmp(REPO_ROOT / f, dst_root / f, shallow=False)
    )

    # Naming follows the direction, so the output never implies the wrong one.
    if args.from_monorepo:
        src_root, dest_root = dst_root, REPO_ROOT
        to_copy_new, to_leave = extra, missing
        new_label, leave_label = "only in monorepo", "only here"
    else:
        src_root, dest_root = REPO_ROOT, dst_root
        to_copy_new, to_leave = missing, extra
        new_label, leave_label = "only here", "only in monorepo"

    for label, files in ((new_label, to_copy_new), (leave_label, to_leave),
                         ("content differs", changed)):
        for f in files:
            print(f"  {label:18} {f}")

    if not (missing or extra or changed):
        print(f"in sync — {len(ours)} tracked files identical")
        return 0

    if args.apply:
        for f in to_copy_new + changed:
            (dest_root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_root / f, dest_root / f)
        print(f"\ncopied {len(to_copy_new) + len(changed)} file(s) to {dest_root}")
        if to_leave:
            # Never deleted automatically. A file that exists only on the other
            # side is more likely someone's uncommitted work than our leftover.
            print(f"left {len(to_leave)} file(s) on the other side alone "
                  "— review by hand")
        return 0

    print(f"\n{len(missing) + len(extra) + len(changed)} file(s) out of sync "
          "— run with --apply")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
