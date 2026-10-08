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
   — preserves the existing fresh-launch path (a ``run_agent`` /
   ``run_parallel`` sub-task's tab) which doesn't send ``isDone``.

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
        # Besides the general assignment from ``ev.isDone``, the handler
        # has an earlier ``subTab.isDone = true`` for a tab that missed
        # its ``subagentDone`` (the daemon's terminal state wins), so
        # look at every assignment, not only the first one.
        done_exprs = [
            m.strip() for m in re.findall(r"subTab\.isDone\s*=\s*([^;]+);", body)
        ]
        # The handler sets the running state through setTabRunning(),
        # which also drops any pending-stop state along with it
        # (reports/stop_button_delay_2026-08-05.html).
        running_exprs = [
            m.strip() for m in re.findall(r"setTabRunning\(subTab,\s*([^)]+)\)", body)
        ]
        assert any(
            "subDone" in e or "ev.isDone" in e for e in done_exprs
        ), done_exprs
        assert any(
            e.startswith("!") and ("subDone" in e or "ev.isDone" in e)
            for e in running_exprs
        ), running_exprs

    def test_handler_default_is_running_when_is_done_missing(self) -> None:
        body = self._handler_source()
        coerce = (
            "!!ev.isDone" in body
            or "Boolean(ev.isDone)" in body
            or "ev.isDone === true" in body
        )
        assert coerce, body
