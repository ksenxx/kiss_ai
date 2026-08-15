# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing Unicode git-path handling bugs.

These tests use real temporary git repositories with ``core.quotePath``
enabled.  In that configuration git emits non-ASCII filenames as
C-style quoted paths such as ``"caf\\303\\251.txt"``.  Several VS Code /
Sorcar helpers previously parsed those command outputs as if they were
plain UTF-8 paths, so autocommit, worktree dirty state copying, and
changed-file lists either missed the files entirely or surfaced escaped
pseudo-paths to the UI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.server.server import VSCodeServer


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *repo* and assert that it succeeds."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _make_repo(tmp_path: Path) -> Path:
    """Create a git repo configured to quote non-ASCII pathnames."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "core.quotePath", "true")
    (repo / "README.md").write_text("# repo\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_autocommit_dirty_files_are_real_unicode_paths(tmp_path: Path) -> None:
    """Autocommit prompts must expose usable workspace-relative paths.

    Reproduction: ``VSCodeServer._main_dirty_files`` parses
    ``git status --porcelain`` by stripping quote characters only.  For
    non-ASCII names it returns escaped strings such as
    ``caf\\303\\251.txt`` instead of the real ``café.txt`` path shown in the
    workspace and accepted by follow-up file operations.
    """
    repo = _make_repo(tmp_path)
    tracked = "café.txt"
    untracked = "日.txt"
    (repo / tracked).write_text("before\n")
    _git(repo, "add", tracked)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / tracked).write_text("after\n")
    (repo / untracked).write_text("new\n")
    server = VSCodeServer()
    server.work_dir = str(repo)

    changed = server._main_dirty_files(str(repo))

    assert tracked in changed
    assert untracked in changed
    assert not any("\\303" in path or "\\346" in path for path in changed)


def test_worktree_changed_file_lists_are_real_unicode_paths(tmp_path: Path) -> None:
    """Worktree changed-file helpers must return real non-ASCII paths.

    Reproduction: ``GitWorktreeOps._diff_name_only`` (historically
    surfaced via the since-removed ``unstaged_files``/``staged_files``
    wrappers) directly splits ``git diff --name-only`` output, which is
    C-quoted when ``core.quotePath`` is true.  Callers then compare
    escaped pseudo-paths against real filenames.
    """
    repo = _make_repo(tmp_path)
    filename = "café.txt"
    (repo / filename).write_text("before\n")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / filename).write_text("unstaged change\n")
    assert GitWorktreeOps._diff_name_only(repo) == [filename]

    _git(repo, "add", filename)
    assert GitWorktreeOps._diff_name_only(repo, "--cached") == [filename]


def test_copy_dirty_state_copies_staged_unicode_rename(tmp_path: Path) -> None:
    """Worktree baseline copying must handle staged non-ASCII renames.

    Reproduction: ``git status --porcelain -uall`` prints a staged rename
    as ``R  "caf\\303\\251.txt" -> "r\\303\\251sum\\303\\251.txt"``.  The current
    parser tries to use the still-quoted source and destination strings as
    paths, returns ``False``, and leaves the new filename absent from the
    worktree baseline.
    """
    repo = _make_repo(tmp_path)
    old_name = "café.txt"
    new_name = "résumé.txt"
    (repo / old_name).write_text("old content\n")
    _git(repo, "add", old_name)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / old_name).rename(repo / new_name)
    _git(repo, "add", "-A")
    wt_dir = repo / ".kiss-worktrees" / "unicode-rename"
    assert GitWorktreeOps.create(repo, "unicode-rename", wt_dir)

    copied = GitWorktreeOps.copy_dirty_state(repo, wt_dir)

    assert copied is True
    assert (wt_dir / new_name).read_text() == "old content\n"
    assert not (wt_dir / old_name).exists()
