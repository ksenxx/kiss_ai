# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: a sub-agent task reopened from the history panel
must render with the **same indicator color and icon** as the tab the
backend originally created when the sub-task was launched.

Contract
--------
``_replay_session`` consults the task-id-keyed registry
:mod:`kiss.server.agent_state` (via ``_subagent_is_done``) to decide
whether the reopened tab is still running or already done.  The
result is broadcast as ``isDone`` on ``openSubagentTab``.

Tests
-----
1. Frontend handler (static check on ``main.js``) reads ``ev.isDone``
   and sets ``subTab.isDone`` / ``subTab.isRunning`` accordingly.
2. Frontend handler default (no ``isDone`` field) is still "running"
   — preserves the existing fresh-launch path
   (``_run_tasks_parallel``) which doesn't send ``isDone``.

The backend ``isDone`` broadcast tests (pure kiss.agents.sorcar +
kiss.server closure) moved to
``kiss.tests.server.test_subagent_history_tab_icon``.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.js"
)


class TestFrontendHandlerHonorsIsDone:
    """Static checks on ``media/main.js`` ``case 'openSubagentTab'``."""

    def _handler_source(self) -> str:
        src = _MAIN_JS.read_text(encoding="utf-8")
        idx = src.index("case 'openSubagentTab':")
        end = src.index("case 'subagentDone':", idx)
        return src[idx:end]

    def test_handler_reads_ev_is_done(self) -> None:
        body = self._handler_source()
        assert "ev.isDone" in body, body

    def test_handler_sets_is_done_and_is_running_consistently(
        self,
    ) -> None:
        body = self._handler_source()
        m_done = re.search(
            r"subTab\.isDone\s*=\s*([^;]+);", body,
        )
        # The handler sets the running state through setTabRunning(),
        # which also drops any pending-stop state along with it
        # (reports/stop_button_delay_2026-08-05.html).
        m_running = re.search(
            r"setTabRunning\(subTab,\s*([^)]+)\)", body,
        )
        assert m_done is not None, body
        assert m_running is not None, body
        done_expr = m_done.group(1).strip()
        running_expr = m_running.group(1).strip()
        assert "subDone" in done_expr or "ev.isDone" in done_expr
        assert running_expr.startswith("!"), running_expr
        assert (
            "subDone" in running_expr or "ev.isDone" in running_expr
        )

    def test_handler_default_is_running_when_is_done_missing(self) -> None:
        body = self._handler_source()
        coerce = (
            "!!ev.isDone" in body
            or "Boolean(ev.isDone)" in body
            or "ev.isDone === true" in body
        )
        assert coerce, body
