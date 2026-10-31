# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Explorer git providers after the 2026-10-30 sweep.

1. ``git_status`` (the Source Control view's background poll) runs
   ``git status --no-optional-locks``: a plain ``git status`` refreshes
   stale index entries and writes the index back under ``index.lock``,
   so a task's ``git add`` / ``git commit`` running at that moment in
   the same worktree failed with "index.lock: File exists".
2. ``_blob_text`` tells a path missing at a revision (an added or
   deleted file: one side of the diff is empty) from a git failure by
   git's message, so it runs git in the C locale.  Under a translated
   locale the German "existiert nicht" never matched and every added or
   deleted file came back as an error instead of a one-sided diff.  The
   test needs a German locale and git's German catalog; a machine
   without them skips it (``LOCPATH`` may point at a locale compiled
   with ``localedef``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiss.server.explorer import git_file_at, git_file_diff, git_status

_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@x",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@x",
    "HOME": "/nonexistent",
    "PATH": os.environ.get("PATH", ""),
}

_GERMAN = {"LC_ALL": "de_DE.UTF-8", "LANGUAGE": "de"}


def _git(
    repo: Path, *args: str, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run git in *repo* with a fixed identity.

    Line endings are stored as they are: with ``HOME`` pointing nowhere,
    Git for Windows' system config (``core.autocrlf=true``) would commit
    LF for the files below while the daemon's git, running under the
    user's config, compares them as they are on disk and reports every
    one modified.
    """
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args],
        capture_output=True, text=True, env=env or _ENV,
    )


def _repo(tmp_path: Path) -> tuple[Path, str]:
    """A repository whose second commit adds ``b.txt``; returns it and that commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main").check_returncode()
    (repo / "a.txt").write_bytes(b"a\n")  # LF on every platform (write_text gives CRLF on Windows)
    _git(repo, "add", "a.txt").check_returncode()
    _git(repo, "commit", "-q", "-m", "first").check_returncode()
    (repo / "b.txt").write_bytes(b"b\n")
    _git(repo, "add", "b.txt").check_returncode()
    _git(repo, "commit", "-q", "-m", "second").check_returncode()
    return repo, _git(repo, "rev-parse", "HEAD").stdout.strip()


def _index_identity(index: Path) -> tuple[int, int]:
    """Inode and mtime of an index file: both change when git writes it anew."""
    st = index.stat()
    return st.st_ino, st.st_mtime_ns


def test_git_status_leaves_the_index_of_every_worktree_untouched(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    linked = tmp_path / "linked"
    assert _git(repo, "worktree", "add", "-q", "-b", "side", str(linked)).returncode == 0
    # Same content, older mtime: the entries are stat-dirty, which a
    # plain ``git status`` repairs by writing the index under its lock.
    for checkout in (repo, linked):
        os.utime(checkout / "a.txt", (1_000_000_000, 1_000_000_000))
    indexes = [repo / ".git" / "index", repo / ".git" / "worktrees" / "linked" / "index"]
    before = [_index_identity(index) for index in indexes]

    reply = git_status(str(repo))

    assert reply["changes"] == []
    assert [wt.get("changes") for wt in reply["worktrees"]] == [[], []]
    assert [_index_identity(index) for index in indexes] == before, "no index was rewritten"


def test_a_missing_side_is_found_under_a_translated_locale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, second = _repo(tmp_path)
    probe = _git(repo, "show", "HEAD:nowhere.txt", env={**_ENV, **os.environ, **_GERMAN})
    if "does not exist" in probe.stderr:
        pytest.skip("git prints no German messages here (no de_DE locale or catalog)")
    for name, value in _GERMAN.items():
        monkeypatch.setenv(name, value)

    added = git_file_diff(str(repo), second, "b.txt")
    assert "error" not in added, added
    assert added["originalMissing"] is True and added["modified"] == "b\n"
    (repo / "b.txt").unlink()
    deleted = git_file_diff(str(repo), "", "b.txt")
    assert deleted["modifiedMissing"] is True and deleted["original"] == "b\n"
    first = _git(repo, "rev-parse", "HEAD~1").stdout.strip()
    assert git_file_at(str(repo), first, "b.txt") == {"error": f"No b.txt in commit {first[:8]}"}
