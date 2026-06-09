# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing Unicode git-path handling bugs.

These tests use real temporary git repositories with ``core.quotePath``
enabled.  In that configuration git emits non-ASCII filenames as
C-style quoted paths such as ``"caf\\303\\251.txt"``.  Several VS Code /
Sorcar helpers currently parse those command outputs as if they were
plain UTF-8 paths, so merge review, autocommit prompts, worktree dirty
state copying, and changed-file lists either miss the files entirely or
surface escaped pseudo-paths to the UI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.vscode.diff_merge import _prepare_merge_view
from kiss.agents.vscode.server import VSCodeServer


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


def _read_pending_merge(data_dir: Path) -> dict[str, Any]:
    """Load the pending merge manifest written by ``_prepare_merge_view``."""
    manifest: dict[str, Any] = json.loads((data_dir / "pending-merge.json").read_text())
    return manifest


def test_merge_view_includes_modified_tracked_unicode_file(tmp_path: Path) -> None:
    """A modified tracked non-ASCII file must open a merge review.

    Reproduction: ``git diff -U0`` outputs ``diff --git
    "a/caf\\303\\251.txt" "b/caf\\303\\251.txt"``.  The VS Code diff parser
    does not unquote that form, so ``_prepare_merge_view`` reports
    ``No changes`` even though ``café.txt`` is modified.
    """
    repo = _make_repo(tmp_path)
    filename = "café.txt"
    (repo / filename).write_text("before\n")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / filename).write_text("before\nafter\n")
    data_dir = tmp_path / "merge-data"

    result = _prepare_merge_view(str(repo), str(data_dir), {}, set())

    assert result.get("status") == "opened"
    manifest = _read_pending_merge(data_dir)
    names = [f["name"] for f in manifest["files"]]
    assert filename in names


def test_merge_view_includes_new_untracked_unicode_file(tmp_path: Path) -> None:
    """A new untracked non-ASCII file must open a merge review.

    Reproduction: ``git ls-files --others`` emits the quoted string
    ``"\\346\\227\\245.txt"`` for ``日.txt`` when ``core.quotePath`` is true.
    ``_capture_untracked`` returns that literal escaped pseudo-path, so
    ``_prepare_merge_view`` looks for a file that does not exist and drops
    the real untracked file from the merge review.
    """
    repo = _make_repo(tmp_path)
    filename = "日.txt"
    (repo / filename).write_text("new file\n")
    data_dir = tmp_path / "merge-data"

    result = _prepare_merge_view(str(repo), str(data_dir), {}, set())

    assert result.get("status") == "opened"
    manifest = _read_pending_merge(data_dir)
    names = [f["name"] for f in manifest["files"]]
    assert filename in names


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

    Reproduction: ``GitWorktreeOps.unstaged_files`` and ``staged_files``
    directly split ``git diff --name-only`` output, which is C-quoted when
    ``core.quotePath`` is true.  Callers such as worktree conflict checks
    then compare escaped pseudo-paths against real filenames.
    """
    repo = _make_repo(tmp_path)
    filename = "café.txt"
    (repo / filename).write_text("before\n")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / filename).write_text("unstaged change\n")
    assert GitWorktreeOps.unstaged_files(repo) == [filename]

    _git(repo, "add", filename)
    assert GitWorktreeOps.staged_files(repo) == [filename]


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
