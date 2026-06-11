# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUG-6B-1: ``ensure_excluded`` crashed on a non-UTF-8 info/exclude file.

Git treats exclude files as raw bytes — non-UTF-8 patterns and comments
(e.g. a Latin-1 filename pattern or comment) are perfectly legal and
git honors them.  ``GitWorktreeOps.ensure_excluded`` read the file with
a STRICT UTF-8 ``Path.read_text()``, so any such file raised
``UnicodeDecodeError``.  The production call site
(``_try_setup_worktree``) swallows the exception, so the
``.kiss-worktrees/`` entry was silently never added: the user's
``git status`` was polluted with ``?? .kiss-worktrees/...`` forever,
``has_uncommitted_changes(repo)`` misreported a clean repo as dirty,
and every merge created a junk "kiss: auto-stash before merge" stash
cycle.  Inconsistent with BUG-5B-2's fix (``_git`` decodes all git
output with ``errors="surrogateescape"``).

All tests use real git repos — no mocks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps

_LATIN1_EXCLUDE = b"# caf\xe9 latin-1 comment\nb\xfcild/\n"


def _make_repo(path: Path) -> Path:
    """Create a real git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def test_ensure_excluded_survives_non_utf8_exclude_file(tmp_path: Path) -> None:
    """ensure_excluded must not crash on legal non-UTF-8 exclude bytes."""
    repo = _make_repo(tmp_path / "repo")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_bytes(_LATIN1_EXCLUDE)

    GitWorktreeOps.ensure_excluded(repo)  # raised UnicodeDecodeError pre-fix

    data = exclude.read_bytes()
    assert b".kiss-worktrees/" in data, "entry was not appended"
    assert _LATIN1_EXCLUDE.rstrip(b"\n") in data, "existing bytes were destroyed"


def test_worktree_stays_invisible_with_non_utf8_exclude(tmp_path: Path) -> None:
    """End-to-end: a clean repo must STAY clean after worktree creation.

    Mirrors the production flow in ``_try_setup_worktree``: the
    ``ensure_excluded`` call is wrapped in a broad try/except, so a
    crash silently skips the exclusion and ``?? .kiss-worktrees/``
    pollutes the user's git status (and triggers junk auto-stash
    cycles on every merge).
    """
    repo = _make_repo(tmp_path / "repo")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_bytes(_LATIN1_EXCLUDE)

    try:
        GitWorktreeOps.ensure_excluded(repo)
    except Exception:
        pass  # production swallows this (see _try_setup_worktree)

    wt_dir = repo / ".kiss-worktrees" / "wt1"
    assert GitWorktreeOps.create(repo, "kiss/wt-bug6b1", wt_dir)
    (wt_dir / "agent.txt").write_text("agent work\n")

    assert not GitWorktreeOps.has_uncommitted_changes(repo), (
        "clean main repo misreported as dirty: .kiss-worktrees/ was not "
        "excluded because ensure_excluded crashed on non-UTF-8 exclude bytes"
    )


def test_ensure_excluded_idempotent_on_non_utf8_file(tmp_path: Path) -> None:
    """Entry already present alongside non-UTF-8 bytes: no dup, no crash."""
    repo = _make_repo(tmp_path / "repo")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_bytes(_LATIN1_EXCLUDE + b".kiss-worktrees/\n")

    GitWorktreeOps.ensure_excluded(repo)
    GitWorktreeOps.ensure_excluded(repo)

    data = exclude.read_bytes()
    assert data.count(b".kiss-worktrees/") == 1, "duplicate entry appended"
