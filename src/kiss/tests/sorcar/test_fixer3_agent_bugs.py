# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for Fixer-3 findings (tmp/findings-2.md F1, F2, F7, F9, F14).

Each test drives the real agent classes against a fresh git repo with an
isolated persistence DB.  No mocks/patches libraries are used; where a
finding is only reachable through an LLM call, the *parent* ``run`` of
``SorcarAgent`` is temporarily replaced with a deterministic raising
function (the same convention as ``test_autocommit_off_on_failure.py``)
so the real ``WorktreeSorcarAgent.run`` / ``ChatSorcarAgent.run`` /
``SorcarAgent.run`` code paths under test all execute for real.

Covered findings:

* F1 — base ``run_tasks_parallel`` used ``parent_task_id=None`` while the
  chat path uses the ``""`` sentinel (regression guard: persisted column
  and ``new_tab`` payload use the empty-string convention).
* F2 — base ``run_tasks_parallel`` broadcast ``subagentDone`` with tab id
  ``{parent}__sub_{idx}`` while the chat executor registers tabs as
  ``task-{parent}__sub_{idx}``.
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

import yaml

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.json_printer import JsonPrinter

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
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self._original_parent_run = _PARENT_CLASS.run

    def tearDown(self) -> None:
        _PARENT_CLASS.run = self._original_parent_run

        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        with ChatSorcarAgent._running_agents_lock:
            ChatSorcarAgent.running_agents.clear()

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

    # -- worktree helpers -------------------------------------------------

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


class TestSubagentDoneTabIdFormat(_Base):
    """F2: base executor must use the chat-style ``task-…__sub_N`` id."""

    def test_subagent_done_uses_task_prefixed_tab_id(self) -> None:
        _PARENT_CLASS.run = _raising_run
        printer = JsonPrinter()
        events: list[dict[str, Any]] = []
        printer.broadcast = events.append  # type: ignore[assignment]
        parent_task_id = "f" * 32
        printer._thread_local.task_id = parent_task_id

        results = run_tasks_parallel(
            ["dummy task"],
            max_workers=1,
            work_dir=self.repo,
            printer=printer,
        )

        self.assertEqual(len(results), 1)
        parsed = yaml.safe_load(results[0])
        self.assertIs(parsed["success"], False)
        done = [e for e in events if e.get("type") == "subagentDone"]
        self.assertTrue(done, f"no subagentDone in {events!r}")
        self.assertEqual(
            done[0].get("tab_id"),
            f"task-{parent_task_id}__sub_0",
        )


class TestSubagentSentinelConvention(_Base):
    """F1 regression guard: base path uses the ``""`` sentinel shape."""

    def test_base_parallel_subagent_info_matches_chat_convention(self) -> None:
        _PARENT_CLASS.run = _raising_run
        printer = JsonPrinter()
        events: list[dict[str, Any]] = []
        printer.broadcast = events.append  # type: ignore[assignment]

        run_tasks_parallel(
            ["dummy task"],
            max_workers=1,
            work_dir=self.repo,
            printer=printer,
        )

        # The sub-agent's ``new_tab`` broadcast reads
        # ``_subagent_info["parent_tab_id"]``; with the harmonized
        # sentinel dict it must be the empty string (never None).
        new_tabs = [e for e in events if e.get("type") == "new_tab"]
        self.assertTrue(new_tabs, f"no new_tab in {events!r}")
        self.assertEqual(new_tabs[0].get("parent_tab_id"), "")

        # The persisted sub-agent row must carry the empty-string
        # "no persisted parent" sentinel in the parent_task_id column.
        conn = _persistence._get_db()
        rows = conn.execute(
            "SELECT parent_task_id FROM task_history"
        ).fetchall()
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row[0], "")


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
    """F9: don't blame a pre-commit hook under ``--no-auto-commit``."""

    def test_merge_reports_auto_commit_disabled(self) -> None:
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None

        msg = agent.merge()

        self.assertIn("auto-commit is disabled", msg)
        self.assertNotIn("pre-commit hook", msg)
        # Worktree must be preserved with the change still uncommitted.
        self.assertTrue(wt.wt_dir.exists())
        self.assertIn("uncommitted.txt", self._porcelain(wt.wt_dir))
        agent.discard()

    def test_release_worktree_warning_reports_auto_commit_disabled(
        self,
    ) -> None:
        agent = self._setup_worktree_agent()
        wt = agent._wt
        assert wt is not None

        released = agent._release_worktree()

        self.assertIsNone(released)
        warning = agent._merge_conflict_warning or ""
        self.assertIn("Auto-commit is disabled", warning)
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
        # The worktree directory is preserved for manual review …
        self.assertTrue(wt.wt_dir.exists())
        # … the user's changes are still UNCOMMITTED …
        self.assertIn("uncommitted.txt", self._porcelain(wt.wt_dir))
        # … and no forced "late-arriving" commit landed on the branch.
        log = _run_git(
            str(wt.wt_dir), "log", "--format=%s",
        ).stdout
        self.assertNotIn("late-arriving", log)
        # Agent state is reset.
        self.assertIsNone(agent._wt)
        self.assertFalse(agent._pending_review)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
