# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test reproducing a Unicode git-path autocommit bug.

This test uses a real temporary git repository with ``core.quotePath``
enabled.  In that configuration git emits non-ASCII filenames as
C-style quoted paths such as ``"caf\\303\\251.txt"``.
``VSCodeServer._main_dirty_files`` previously parsed that command
output as if it were plain UTF-8 paths, so autocommit either missed the
files entirely or surfaced escaped pseudo-paths to the UI.

The worktree-helper variants of this bug
(``GitWorktreeOps._diff_name_only`` and
``GitWorktreeOps.copy_dirty_state``), which depend only on
``kiss.core`` and ``kiss.agents.sorcar``, live in
``kiss.tests.agents.sorcar.test_unicode_git_path_bugs`` together with
the git helpers imported below.
"""

from __future__ import annotations

from pathlib import Path

from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_unicode_git_path_bugs import (
    _git,
    _make_repo,
)


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
