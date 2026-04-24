"""Test: merge fails when main working tree has dirty/staged changes.

Reproduces the bug where ``squash_merge_branch`` fails with a misleading
"Merge conflict" error when the real issue is uncommitted changes in the
main repo's working tree or index that overlap with the branch being
merged.

The fix: ``merge()`` should stash local changes before the squash merge,
then restore them afterward, so user edits in the main repo don't block
the merge.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
)
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
    (path / "fileA.txt").write_text("original A\n")
    (path / "fileB.txt").write_text("original B\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
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


class TestDirtyWorkingTreeMerge:
    """Merge must handle dirty main working tree gracefully."""

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

    def test_squash_merge_fails_with_dirty_overlap(self) -> None:
        """BUG REPRODUCTION: squash_merge_branch fails when the main working
        tree has uncommitted changes that overlap with the merge.

        This simulates what happened when the user's editor had modified
        a file in the main repo while the agent was working in a worktree.
        """
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileB.txt").write_text("modified B by agent\n")

        (self.repo / "fileB.txt").write_text("user edit to B\n")

        msg = agent.merge()
        assert "Successfully merged" in msg

        content = (self.repo / "fileB.txt").read_text()
        assert content

    def test_squash_merge_fails_with_staged_overlap(self) -> None:
        """squash_merge_branch fails when main index has staged changes
        that overlap with the merge (e.g., after a previous commit failure).
        """
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileA.txt").write_text("modified A by agent\n")

        (self.repo / "fileA.txt").write_text("staged edit to A\n")
        _git("add", "fileA.txt", cwd=self.repo)

        msg = agent.merge()
        assert "Successfully merged" in msg

    def test_squash_merge_succeeds_with_non_overlapping_dirty(self) -> None:
        """Dirty files that don't overlap with merge should not block it."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileA.txt").write_text("modified A by agent\n")

        (self.repo / "fileB.txt").write_text("user edit to B\n")

        msg = agent.merge()
        assert "Successfully merged" in msg

        assert (self.repo / "fileB.txt").read_text() == "user edit to B\n"

    def test_squash_merge_branch_dirty_index_returns_conflict(self) -> None:
        """Verify the raw squash_merge_branch fails with dirty index."""
        _git("checkout", "-b", "feature", cwd=self.repo)
        (self.repo / "fileA.txt").write_text("feature change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "feature work", cwd=self.repo)
        _git("checkout", "main", cwd=self.repo)

        (self.repo / "fileA.txt").write_text("dirty staged\n")
        _git("add", "fileA.txt", cwd=self.repo)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "feature")
        assert result == MergeResult.CONFLICT

        _git("reset", "--hard", "HEAD", cwd=self.repo)
        _git("branch", "-D", "feature", cwd=self.repo)
