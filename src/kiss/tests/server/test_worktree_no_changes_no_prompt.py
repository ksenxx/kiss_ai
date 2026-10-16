# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Regression tests: no merge/discard prompt when the worktree is empty.

Bug report: "in worktree mode, why are you asking for Auto-commit and
merge when nothing has been changed in the worktree?"

When the agent isolated itself with ``create_worktree``, finished
without ``commit_and_merge`` and made no file changes in the worktree,
the previous code path broadcast a ``worktree_done`` event whose
frontend handler unconditionally renders the "Merge or Discard?" bar.
There is nothing to merge in that scenario, so the prompt is
meaningless noise.

These tests run the real :class:`AgentServer` against a fresh git
repo with an isolated persistence DB; only the LLM-backed parent
``run`` is replaced by a stub that creates the worktree like the real
agent's tool call would.  They verify that:

1. ``worktree_done`` is NOT broadcast when the worktree has no changes
   (the branch is preserved).
2. ``worktree_done`` IS broadcast (with the merge/discard prompt) when
   the worktree has real changes — the existing behavior we don't want
   to regress.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import init_repo, run_git


def _list_kiss_wt_branches(repo: str) -> list[str]:
    """List all ``kiss/wt-*`` branches currently present in *repo*."""
    result = run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


class TestNoWorktreePromptWhenEmpty(unittest.TestCase):
    """Worktree post-task UX when the agent never merged."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-test-")
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

        self.server = AgentServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        agent_state.agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _stub_agent(self, filename: str | None) -> None:
        """Replace the parent ``run`` with a stub that isolates itself
        in a worktree, writes *filename* there (nothing when ``None``)
        and finishes without ``commit_and_merge``."""

        def stub_run(self_agent: object, **kwargs: object) -> str:
            agent = cast(ChatSorcarAgent, self_agent)
            agent._enter_work_dir(str(kwargs.get("work_dir") or ""))
            message = agent.create_worktree()
            assert agent._wt_pending, message
            if filename is not None:
                (Path(agent.work_dir) / filename).write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _run(self, prompt: str) -> list[str]:
        self.server._run_task_inner({
            "prompt": prompt,
            "workDir": self.repo,
            "tabId": "0",
            "model": "",
        })
        return [e["type"] for e in self.events]

    def test_no_worktree_done_when_no_changes(self) -> None:
        """Bug repro: the merge/discard prompt must not appear when the
        worktree has no changes."""
        self._stub_agent(None)
        types = self._run("task with no changes")
        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"Expected the empty worktree branch to be discarded; got {branches}"
        )
        assert "worktree_result" in types, types
        assert "worktree_done" not in types, (
            f"BUG: worktree_done was broadcast even though the worktree "
            f"is empty. Events: {types}"
        )

    def test_worktree_done_when_changes_exist(self) -> None:
        """Sanity / non-regression: when the agent actually modified
        files in the worktree, ``worktree_done`` is broadcast so the
        user gets the Merge / Discard buttons."""
        self._stub_agent("agent_out.txt")
        types = self._run("task with changes")
        assert "worktree_done" in types, (
            f"Expected worktree_done event when changes exist; got {types}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
