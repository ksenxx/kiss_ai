# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming bugs found in worktree audit round 6.

Each test CONFIRMS the bug exists (assertions pass when buggy behaviour
is present).

BUG-25: merge() with an unknown original_branch (crash during setup)
        must not orphan the branch silently: the branch is kept and the
        user is told the target branch is unknown.

BUG-26: merge() calls delete_branch and sets self._wt = None OUTSIDE
        repo_lock.  Creates a window for concurrent operations on the
        same repo.

BUG-28 (obsolete): covered _start_merge_session, removed together
        with the interactive diff/merge review workflow.

BUG-29: merge() keeps the user's dirty state stashed after a merge
        conflict; the instructions must tell the user to run
        `git stash pop` after resolving, since `git merge --squash`
        refuses a dirty working tree.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
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


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestBug25MergeBranchKeptOnNoneOriginal:
    """BUG-25: When original_branch is None, merge() must keep the
    branch and tell the user the target branch is unknown instead of
    silently orphaning it.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_branch_kept_and_user_told(self) -> None:
        """BUG-25: Branch is kept and the message names it."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = ChatSorcarAgent("test")
        agent._chat_id = "test25"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None
        branch_name = wt.branch

        (wt.wt_dir / "work.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        agent._wt = GitWorktree(
            repo_root=wt.repo_root,
            branch=wt.branch,
            original_branch=None,
            wt_dir=wt.wt_dir,
            baseline_commit=wt.baseline_commit,
        )

        result = agent.merge()

        assert GitWorktreeOps.branch_exists(repo, branch_name), (
            "Branch must be preserved for manual resolution"
        )
        assert branch_name in result
        assert "original branch is unknown" in result
        assert agent._merge_conflict_warning is None
        assert agent._wt is not None, "the branch stays pending"





class TestBug29ConflictInstructionsIgnoreDirtyState:
    """BUG-29: When merge() encounters a merge conflict AND the user
    had dirty state that was stashed, the instructions must account
    for the stash: `git merge --squash` requires a clean working
    tree, so the user's files stay stashed and the steps end with
    `git stash pop`.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_conflict_instructions_fail_with_dirty_tree(self) -> None:
        """BUG-29: Demonstrate that the conflict instructions fail."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = ChatSorcarAgent("test")
        agent._chat_id = "test29"

        (repo / "user_file.txt").write_text("user dirty state\n")

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "README.md").write_text("# Agent\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent change")

        (repo / "README.md").write_text("# User conflict\n")
        _git("add", "README.md", cwd=repo)
        _git("commit", "-m", "user conflict", cwd=repo)

        (repo / "user_file.txt").write_text("user dirty state again\n")

        warning = agent.merge()
        assert "Merge conflict detected" in warning
        assert "git stash pop" in warning, (
            "the user's dirty state is stashed; the steps must restore it"
        )
        assert not (repo / "user_file.txt").exists(), (
            "dirty state must stay stashed so the squash merge can run"
        )

        if wt.baseline_commit:
            assert "cherry-pick" in warning, (
                "Warning should contain cherry-pick instructions when "
                "baseline exists"
            )
        else:
            assert "merge --squash" in warning, (
                "Warning should contain merge --squash instructions "
                "when no baseline"
            )
