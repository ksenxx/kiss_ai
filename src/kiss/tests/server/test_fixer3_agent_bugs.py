# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for Fixer-3 findings (tmp/findings-2.md F7, F9, F14).

Each test drives the real agent classes against a fresh git repo with an
isolated persistence DB.  No mocks/patches libraries are used; where a
finding is only reachable through an LLM call, the *parent* ``run`` of
``SorcarAgent`` is temporarily replaced with a deterministic raising
function (the same convention as ``test_autocommit_off_on_failure.py``)
so the real ``WorktreeSorcarAgent.run`` / ``ChatSorcarAgent.run`` /
``SorcarAgent.run`` code paths under test all execute for real.

Covered findings:

* F7 — ``WorktreeSorcarAgent.run``'s direct-execution fallback propagated
  non-``KISSError`` exceptions while the worktree path converts them to a
  YAML ``success: false`` result.
* F9 — ``merge()`` / ``_release_worktree()`` blamed a pre-commit hook when
  ``auto_commit_enabled=False`` was the real reason finalize returned False.
* F14 — ``_preserve_pending_worktree_for_review`` force-committed
  uncommitted changes via ``commit_all`` even under ``--no-auto-commit``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

pytestmark = pytest.mark.usefixtures("stubbed_agent_model")

_PARENT_CLASS = cast(Any, SorcarAgent.__mro__[1])


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


def _raising_run(self: Any, *args: Any, **kwargs: Any) -> str:
    """Deterministic parent-run replacement: always fails fast."""
    raise RuntimeError("fixer3-deterministic-failure")


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer3-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None

        self._original_parent_run = _PARENT_CLASS.run

    def tearDown(self) -> None:
        _PARENT_CLASS.run = self._original_parent_run

        from kiss.server import agent_state

        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup_worktree_agent(self) -> WorktreeSorcarAgent:
        """Real worktree on a fresh branch with one uncommitted file."""
        agent = WorktreeSorcarAgent("fixer3-wt")
        agent.auto_commit_enabled = False
        wt_work = agent._try_setup_worktree(Path(self.repo), self.repo)
        assert wt_work is not None, "worktree setup failed"
        assert agent._wt is not None
        Path(agent._wt.wt_dir, "uncommitted.txt").write_text("pending work\n")
        return agent

    @staticmethod
    def _porcelain(cwd: Path) -> str:
        return _run_git(str(cwd), "status", "--porcelain").stdout.strip()


class TestWorktreeFallbackExceptionContract(_Base):
    """F7: fallback (non-git) path must return YAML failure, not raise."""

    def test_non_git_fallback_returns_yaml_failure(self) -> None:
        _PARENT_CLASS.run = _raising_run
        agent = WorktreeSorcarAgent("fixer3-f7-fallback")
        non_git = str(Path(self.tmpdir) / "not-a-repo")
        Path(non_git).mkdir(parents=True, exist_ok=True)

        result = agent.run(
            prompt_template="do something",
            work_dir=non_git,
            use_worktree=True,
        )

        parsed = yaml.safe_load(result)
        self.assertIs(parsed["success"], False)
        self.assertIn("fixer3-deterministic-failure", parsed["summary"])

    def test_worktree_path_returns_yaml_failure(self) -> None:
        """Companion guard: the worktree path keeps the same contract."""
        _PARENT_CLASS.run = _raising_run
        agent = WorktreeSorcarAgent("fixer3-f7-worktree")

        result = agent.run(
            prompt_template="do something",
            work_dir=self.repo,
            use_worktree=True,
        )

        parsed = yaml.safe_load(result)
        self.assertIs(parsed["success"], False)
        self.assertIn("fixer3-deterministic-failure", parsed["summary"])
        if agent._wt is not None:
            agent.discard()


class TestMergeNoAutoCommitMessage(_Base):
    """F9: don't blame a pre-commit hook when Auto-commit is off."""

    def test_explicit_merge_commits_even_when_auto_commit_is_off(
        self,
    ) -> None:
        """Clicking merge IS consent to commit, so it must not refuse.

        Auto-commit off governs the AUTOMATIC paths only (see
        ``test_release_worktree_warning_reports_auto_commit_disabled``
        below); refusing the user's own merge click would strand the
        work with no way to publish it from the UI.
        """
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None

        msg = agent.merge()

        self.assertIn("Successfully merged", msg)
        self.assertNotIn("pre-commit hook", msg)
        self.assertFalse(wt.wt_dir.exists())
        self.assertTrue(Path(self.repo, "uncommitted.txt").exists())

    def test_release_worktree_warning_reports_auto_commit_disabled(
        self,
    ) -> None:
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None

        released = agent._release_worktree()

        self.assertIsNone(released)
        warning = agent._merge_conflict_warning or ""
        self.assertIn("Auto-commit is turned off", warning)
        self.assertNotIn("pre-commit hook", warning)
        self.assertTrue(wt.wt_dir.exists())
        self.assertIn("uncommitted.txt", self._porcelain(wt.wt_dir))


class TestPreserveForReviewNoAutoCommit(_Base):
    """F14: preserve path must not force-commit under --no-auto-commit."""

    def test_preserve_keeps_changes_uncommitted_and_dir_intact(self) -> None:
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None
        agent._pending_review = True

        preserved = agent._preserve_pending_worktree_for_review()

        self.assertTrue(preserved)
        self.assertTrue(wt.wt_dir.exists())
        self.assertIn("uncommitted.txt", self._porcelain(wt.wt_dir))
        log = _run_git(
            str(wt.wt_dir), "log", "--format=%s",
        ).stdout
        self.assertNotIn("late-arriving", log)
        self.assertIsNone(agent._wt)
        self.assertFalse(agent._pending_review)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
