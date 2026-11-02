# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S7 F1/F4: worktree paths with CR/LF and repo roots with a trailing space.

* ``registered_worktrees`` read ``git worktree list --porcelain`` line by
  line, so a worktree whose name held a carriage return or a newline
  was paired with its branch under a TRUNCATED path.  ``reclaim_
  orphaned_worktrees`` then handed that path (an unrelated directory
  when one existed there) to ``remove``, whose ``rmtree`` fallback
  deleted it.  The list is now NUL-terminated (``-z``).
* ``discover_repo`` stripped whitespace off ``rev-parse --show-toplevel``,
  so a repository named ``project `` was discovered as its neighbour
  ``project``.  Only git's record terminator is removed now.

Real repositories and git; nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import init_repo, run_git_checked

pytestmark = posix_only("carriage returns, newlines and trailing spaces in path names")

NAMES = ("kiss_wt-cr\rname", "kiss_wt-lf\nname", "kiss_wt-tab\tname", 'kiss_wt-quote"name')


def test_registered_worktrees_keeps_control_characters_in_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    wanted: dict[str, Path] = {}
    for i, name in enumerate(NAMES):
        wt_dir = repo / ".kiss-worktrees" / name
        run_git_checked(repo, "worktree", "add", "-q", "-b", f"kiss/wt-path-{i}", str(wt_dir))
        wanted[f"kiss/wt-path-{i}"] = wt_dir

    listed = {branch: wt_dir for wt_dir, branch in GitWorktreeOps.registered_worktrees(repo)}

    assert listed["main"] == repo
    for branch, wt_dir in wanted.items():
        assert listed[branch] == wt_dir, (branch, listed[branch].name)
        assert listed[branch].is_dir()


def test_reclaim_does_not_delete_the_directory_named_by_a_truncated_path(tmp_path: Path) -> None:
    repo = tmp_path / "main-repo"
    init_repo(repo)
    GitWorktreeOps.prepare_repo(repo)
    branch = "kiss/wt-cr-path"
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-victim\ractual"
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    assert GitWorktreeOps.save_original_branch(repo, branch, "main")
    # An unrelated repository sits exactly where the truncated path points.
    bystander = repo / ".kiss-worktrees" / "kiss_wt-victim"
    init_repo(bystander)
    (bystander / "valuable.txt").write_text("the only copy\n")
    run_git_checked(bystander, "add", "valuable.txt")
    run_git_checked(bystander, "commit", "-q", "-m", "unique data")

    assert GitWorktreeOps.reclaim_orphaned_worktrees(repo) == 1

    assert (bystander / "valuable.txt").read_text() == "the only copy\n"
    assert (bystander / ".git").is_dir()
    assert not wt_dir.exists()
    assert (repo / "seed.txt").is_file()


def _init_empty_repo(path: Path) -> Path:
    """``git init`` *path* (``init_repo`` refuses a root whose name ends in a space)."""
    path.mkdir()
    run_git_checked(path, "init", "-q", "-b", "main")
    return path


def test_discover_repo_keeps_a_trailing_space_of_the_root_name(tmp_path: Path) -> None:
    neighbour = _init_empty_repo(tmp_path / "project")
    spaced = _init_empty_repo(tmp_path / "project ")
    alone = _init_empty_repo(tmp_path / "solo ")

    assert GitWorktreeOps.discover_repo(spaced) == spaced.resolve()
    assert GitWorktreeOps.discover_repo(alone) == alone.resolve()
    assert GitWorktreeOps.discover_repo(neighbour) == neighbour.resolve()
