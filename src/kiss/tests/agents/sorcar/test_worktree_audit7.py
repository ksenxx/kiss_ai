# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests verifying fixes for bugs found in worktree audit round 7.

BUG-30 FIX: _try_setup_worktree reads current_branch(repo) inside
        repo_lock, preventing races with concurrent tab checkouts.

BUG-31 FIX: Both merge() and the commit_and_merge tool distinguish
        MERGE_FAILED from CONFLICT, giving correct diagnostic messages
        for commit failures (e.g. pre-commit hook rejection).

BUG-32 (obsolete): covered the per-tab merge review data directories
        (``_merge_data_dir``/``_finish_merge``), removed together with
        the interactive diff/merge review workflow.

BUG-33 FIX: copy_dirty_state reads ``git status --porcelain -z``, whose
        names git never quotes, correctly handling files with
        non-ASCII characters, control chars, or special characters.
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
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
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


class TestBug30Fix:
    """BUG-30 FIX: current_branch is read inside repo_lock, preventing
    races with concurrent checkouts.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


    def test_correct_original_branch_recorded(self) -> None:
        """BUG-30 FIX: original_branch is correctly recorded for new worktree."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = ChatSorcarAgent("agent_a")
        agent._chat_id = "tab_a"

        assert GitWorktreeOps.current_branch(repo) == "main"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None
        assert wt.original_branch == "main"

        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        GitWorktreeOps.delete_branch(repo, wt.branch)

    def test_retire_previous_worktree_is_a_no_op_without_a_prior(self) -> None:
        """BUG-30 FIX: _retire_previous_worktree does nothing without a worktree."""
        agent = ChatSorcarAgent("test")
        agent._chat_id = "no_prior"
        agent._retire_previous_worktree()
        assert agent._wt is None
        assert agent._merge_conflict_warning is None


class TestBug31Fix:
    """BUG-31 FIX: MERGE_FAILED is now handled distinctly from CONFLICT
    in both merge() and the commit_and_merge tool, with correct messages.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)



    def test_merge_commit_failure_not_reported_as_conflict(self) -> None:
        """BUG-31 FIX: A commit failure is NOT called a 'conflict'."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = ChatSorcarAgent("test")
        agent._chat_id = "test31"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "agent_file.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = agent.merge()

        assert "conflict" not in result.lower(), (
            f"BUG-31 NOT fixed: merge result says 'conflict': {result}"
        )
        assert "commit failed" in result.lower(), (
            f"BUG-31 NOT fixed: merge result doesn't mention commit failure: {result}"
        )
        assert "--no-verify" in result, (
            f"BUG-31 NOT fixed: no --no-verify suggestion: {result}"
        )

        hook.unlink()
        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, wt.branch):
            GitWorktreeOps.delete_branch(repo, wt.branch)

    def test_commit_and_merge_commit_failure_not_conflict_message(self) -> None:
        """BUG-31 FIX: commit_and_merge gives correct message for MERGE_FAILED."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = ChatSorcarAgent("test")
        agent._chat_id = "test31r"
        agent.work_dir = str(repo)

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "agent_file.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        warning = agent.commit_and_merge()
        assert agent._wt is not None, "the branch must be kept after MERGE_FAILED"
        assert GitWorktreeOps.branch_exists(repo, wt.branch)

        assert "conflict" not in warning.lower(), (
            f"BUG-31 NOT fixed: message reports a conflict: {warning}"
        )
        assert "commit failed" in warning.lower(), (
            f"BUG-31 NOT fixed: message doesn't mention commit failure: {warning}"
        )
        assert "--no-verify" in warning, (
            f"BUG-31 NOT fixed: no --no-verify suggestion: {warning}"
        )

        hook.unlink()
        if GitWorktreeOps.branch_exists(repo, wt.branch):
            GitWorktreeOps.delete_branch(repo, wt.branch)


class TestBug33Fix:
    """BUG-33 FIX: copy_dirty_state mirrors a file whose name git would
    C-quote in plain ``git status --porcelain`` output.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_non_ascii_filename_copied(self) -> None:
        """BUG-33 FIX: File with non-ASCII name IS copied to worktree."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        _git("config", "core.quotePath", "true", cwd=repo)

        non_ascii_name = "café.txt"
        non_ascii_file = repo / non_ascii_name
        non_ascii_file.write_text("content of café\n")

        status = _git("status", "--porcelain", "-uall", cwd=repo)
        assert status.stdout.strip(), "sanity: git status shows the new file"

        wt_dir = Path(self._tmpdir) / "wt"
        wt_dir.mkdir()
        _git("worktree", "add", "-b", "test-wt", str(wt_dir), cwd=repo)

        result = GitWorktreeOps.copy_dirty_state(repo, wt_dir)
        assert result is True, "Should return True (dirty state was copied)"

        wt_file = wt_dir / non_ascii_name
        assert wt_file.exists(), (
            "BUG-33 NOT fixed: non-ASCII file was NOT copied to worktree"
        )
        assert wt_file.read_text() == "content of café\n"

        _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
        _git("worktree", "prune", cwd=repo)
        _git("branch", "-D", "test-wt", cwd=repo)
