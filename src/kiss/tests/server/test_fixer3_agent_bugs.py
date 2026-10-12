# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for Fixer-3 findings (tmp/findings-2.md F7, F9).

Each test drives the real agent classes against a fresh git repo with an
isolated persistence DB.  No mocks/patches libraries are used; where a
finding is only reachable through an LLM call, the *parent* ``run`` of
``SorcarAgent`` is temporarily replaced with a deterministic raising
function (the same convention as ``test_autocommit_off_on_failure.py``)
so the real ``ChatSorcarAgent.run`` / ``ChatSorcarAgent.run`` /
``SorcarAgent.run`` code paths under test all execute for real.

Covered findings:

* F7 — ``ChatSorcarAgent.run``'s direct-execution fallback propagated
  non-``KISSError`` exceptions while the worktree path converts them to a
  YAML ``success: false`` result.
* F9 — ``merge()`` must commit the worktree's pending work itself and
  never blame a pre-commit hook for work it did not try to commit.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.conftest import init_repo, run_git

_PARENT_CLASS = cast(Any, SorcarAgent.__mro__[1])


def _raising_run(self: Any, *args: Any, **kwargs: Any) -> str:
    """Deterministic parent-run replacement: always fails fast."""
    raise RuntimeError("fixer3-deterministic-failure")


def _raising_run_in_worktree(self: Any, *args: Any, **kwargs: Any) -> str:
    """Parent-run replacement that enters a worktree, then fails fast."""
    self.work_dir = str(Path(str(kwargs.get("work_dir") or ".")).resolve())
    self.create_worktree()
    raise RuntimeError("fixer3-deterministic-failure")


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer3-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

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

    def _setup_worktree_agent(self) -> ChatSorcarAgent:
        """Real worktree on a fresh branch with one uncommitted file."""
        agent = ChatSorcarAgent("fixer3-wt")
        wt_work = agent._try_setup_worktree(Path(self.repo), self.repo)
        assert wt_work is not None, "worktree setup failed"
        assert agent._wt is not None
        Path(agent._wt.wt_dir, "uncommitted.txt").write_text("pending work\n")
        return agent

    @staticmethod
    def _porcelain(cwd: Path) -> str:
        return run_git(str(cwd), "status", "--porcelain").stdout.strip()


class TestWorktreeFallbackExceptionContract(_Base):
    """F7: fallback (non-git) path must return YAML failure, not raise."""

    def test_non_git_fallback_returns_yaml_failure(self) -> None:
        _PARENT_CLASS.run = _raising_run
        agent = ChatSorcarAgent("fixer3-f7-fallback")
        non_git = str(Path(self.tmpdir) / "not-a-repo")
        Path(non_git).mkdir(parents=True, exist_ok=True)

        result = agent.run(
            prompt_template="do something",
            work_dir=non_git,
        )

        parsed = yaml.safe_load(result)
        self.assertIs(parsed["success"], False)
        self.assertIn("fixer3-deterministic-failure", parsed["summary"])

    def test_worktree_path_returns_yaml_failure(self) -> None:
        """Companion guard: the worktree path keeps the same contract."""
        _PARENT_CLASS.run = _raising_run_in_worktree
        agent = ChatSorcarAgent("fixer3-f7-worktree")

        result = agent.run(
            prompt_template="do something",
            work_dir=self.repo,
        )

        parsed = yaml.safe_load(result)
        self.assertIs(parsed["success"], False)
        self.assertIn("fixer3-deterministic-failure", parsed["summary"])
        self.assertIsNotNone(agent._wt)
        agent.discard()


class TestMergeCommitsPendingWork(_Base):
    """F9: merge commits the pending work; no false pre-commit blame."""

    def test_explicit_merge_commits_pending_work(self) -> None:
        """Clicking merge IS consent to commit, so it must not refuse."""
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None

        msg = agent.merge()

        self.assertIn("Successfully merged", msg)
        self.assertNotIn("pre-commit hook", msg)
        self.assertFalse(wt.wt_dir.exists())
        self.assertTrue(Path(self.repo, "uncommitted.txt").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
