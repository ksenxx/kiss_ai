# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for WorktreeSorcarAgent ↔ VSCode extension workflow.

Validates the full commit-and-merge / discard workflow as exercised by
the extension: task execution → worktree_done broadcast → user action
(merge or discard) → worktree_result broadcast → git state cleanup.

Every test uses real git repos (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
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
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
    raise_exc: BaseException | None = None,
) -> Any:
    """Patch SorcarAgent's parent ``run()`` to return a canned value.

    Args:
        return_value: String to return from the fake run.
        raise_exc: If set, the fake run raises this exception instead
            of returning *return_value*.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        if raise_exc is not None:
            raise raise_exc
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


def _current_branch(repo: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _branch_exists(repo: Path, branch: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify",
         f"refs/heads/{branch}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _file_in_repo(repo: Path, filename: str) -> bool:
    return (repo / filename).exists()


class TestWorktreeWorkflow:
    """Agent-level tests for commit-and-merge / discard workflow."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent(self) -> WorktreeSorcarAgent:
        return WorktreeSorcarAgent("test")


    def test_merge_propagates_file_changes_to_original_branch(self) -> None:
        """After merge, files created in the worktree appear on original."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "new_file.txt").write_text("hello from worktree")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "add new_file")

        agent.merge()
        assert _file_in_repo(self.repo, "new_file.txt")
        assert (self.repo / "new_file.txt").read_text() == "hello from worktree"

    @pytest.mark.slow
    def test_merge_commits_changes(self) -> None:
        """After merge, changes are committed on the original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "new_file.txt").write_text("hello from worktree")
        (wt_dir / "README.md").write_text("modified\n")

        agent.merge()

        assert _file_in_repo(self.repo, "new_file.txt")
        assert (self.repo / "README.md").read_text() == "modified\n"

        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        porcelain = status.stdout.strip()
        assert not porcelain, "Working tree should be clean after merge"

        merge_head = self.repo / ".git" / "MERGE_HEAD"
        assert not merge_head.exists()

    def test_merge_restores_original_branch_as_head(self) -> None:
        """After merge, HEAD is the original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        original = agent._original_branch
        assert original is not None

        agent.merge()
        assert _current_branch(self.repo) == original

    def test_merge_deletes_task_branch(self) -> None:
        """After merge, the task branch no longer exists."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.merge()
        assert not _branch_exists(self.repo, branch)

    def test_merge_removes_worktree_dir(self) -> None:
        """After merge, the worktree directory is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        agent.merge()
        assert not wt_dir.exists()

    def test_merge_cleans_git_config(self) -> None:
        """After merge, branch.<name>.kiss-original config is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.merge()
        r = _git("config", f"branch.{branch}.kiss-original", cwd=self.repo)
        assert r.returncode != 0


    def test_discard_does_not_propagate_file_changes(self) -> None:
        """After discard, files from the worktree do not appear on original."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "new_file.txt").write_text("should not appear")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "add new_file")

        agent.discard()
        assert not _file_in_repo(self.repo, "new_file.txt")

    def test_discard_restores_original_branch_as_head(self) -> None:
        """After discard, HEAD is back on original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        original = agent._original_branch
        assert original is not None

        agent.discard()
        assert _current_branch(self.repo) == original

    def test_discard_deletes_task_branch(self) -> None:
        """After discard, the task branch no longer exists."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.discard()
        assert not _branch_exists(self.repo, branch)

    def test_discard_removes_worktree_dir(self) -> None:
        """After discard, the worktree directory is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        agent.discard()
        assert not wt_dir.exists()


    def test_do_nothing_method_does_not_exist(self) -> None:
        """WorktreeSorcarAgent no longer has a do_nothing() method."""
        agent = self._agent()
        assert not hasattr(agent, "do_nothing")


    def test_merge_auto_commits_uncommitted_worktree_changes(self) -> None:
        """Uncommitted changes in the worktree are auto-committed on merge."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommitted.txt").write_text("not staged or committed")

        agent.merge()
        assert _file_in_repo(self.repo, "uncommitted.txt")


    def test_merge_conflict_preserves_pending_state(self) -> None:
        """On merge conflict, agent stays pending so discard still works."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "wt conflict")

        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main conflict", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict" in msg
        assert agent._wt_pending
        assert agent._wt_branch is not None

        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        assert not status.stdout.strip(), "Working tree should be clean after conflict"

        agent.discard()
        assert not agent._wt_pending


    def test_run_result_is_plain_task_output(self) -> None:
        """run() returns only the task result — no merge-instructions suffix."""
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "agent.merge()" not in result
        assert "agent.discard()" not in result
        assert "do_nothing" not in result
        assert agent._wt_pending
        agent.discard()
