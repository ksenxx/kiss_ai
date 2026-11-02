# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S7 F2: a retried rescue compares bytes, not ``filecmp``'s cache.

``_land_rescued_file`` decided "already safe" with ``filecmp.cmp(...,
shallow=False)``, whose process-wide cache is keyed by path, size and
mtime.  A first rescue that failed closed on another file cached the
equality of ``a.bin``; the task then replaced ``a.bin`` with different
bytes of the same length and the same mtime (an archive extraction, a
``copy2``), and the retried rescue trusted the cached verdict: the only
copy of the new bytes was dropped with the worktree.

Real repository, worktree and files; nothing is mocked.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import init_repo, run_git_checked

pytestmark = posix_only("a symlinked ancestor blocks the first rescue")


def test_retried_rescue_lands_a_same_size_same_mtime_replacement(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitignore").write_text("*.bin\n")
    run_git_checked(repo, "add", ".gitignore")
    run_git_checked(repo, "commit", "-q", "-m", "ignore bin")
    GitWorktreeOps.prepare_repo(repo)
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-cache"
    assert GitWorktreeOps.create(repo, "kiss/wt-cache", wt_dir)
    src = wt_dir / "a.bin"
    dst = repo / "a.bin"
    src.write_bytes(b"OLD DATA")
    shutil.copy2(src, dst)
    (wt_dir / "blocked").mkdir()
    (wt_dir / "blocked" / "b.bin").write_bytes(b"blocked on the first rescue")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "blocked").symlink_to(outside, target_is_directory=True)

    # The escaping destination fails the rescue closed; a.bin was compared equal.
    assert GitWorktreeOps.rescue_ignored_files(wt_dir, repo) == (0, False)

    stat = src.stat()
    replacement = wt_dir / "replacement.bin"
    replacement.write_bytes(b"NEW DATA")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(src)
    assert src.stat().st_mtime_ns == dst.stat().st_mtime_ns
    (repo / "blocked").unlink()

    assert GitWorktreeOps.rescue_ignored_files(wt_dir, repo) == (2, True)

    assert dst.read_bytes() == b"OLD DATA"
    siblings = list(repo.glob("a.kiss-rescued-*.bin"))
    assert [p.read_bytes() for p in siblings] == [b"NEW DATA"]
    assert (repo / "blocked" / "b.bin").read_bytes() == b"blocked on the first rescue"
