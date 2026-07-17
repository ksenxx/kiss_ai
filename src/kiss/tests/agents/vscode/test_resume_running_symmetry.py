# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: a tab that loads a still-running task from history
must reach the same frontend state as a tab that runs a task fresh —
in both the VS Code extension and the web server.

Two divergences were patched to enforce this invariant:

1. Backend (``_replay_session`` in ``server.py``):
   When the resumed chat is a still-running regular (non-sub-agent)
   task, the ``status running=true`` event must be broadcast BEFORE
   the ``task_events`` replay.  This matches the fresh-run order
   (``_run_task`` emits ``status:true`` first, then events) so the
   webview's module-global ``isRunning`` flag is ``true`` while
   ``replayTaskEvents`` runs — which in turn means
   ``applyChevronState`` (called at the end of replay) hits its
   ``inRunning`` branch and leaves replayed panels visible.

2. Frontend (``case 'openSubagentTab':`` / ``case 'subagentDone':``
   in ``media/main.js``):
   Sub-agent resumes do NOT emit a separate ``status`` event (the
   sub-agent's lifecycle is signalled via ``openSubagentTab`` /
   ``subagentDone``), so the global ``isRunning`` must be synchronised
   from the handler itself when the sub-agent tab is the active one.
   Otherwise a history-loaded still-running sub-agent tab shows
   "Done" with no timer, while the originally-launched tab shows
   "Running" with a spinning timer.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_case_body(src: str, case_name: str) -> str:
    """Return the body of a ``case '<case_name>':`` block."""
    m = re.search(rf"case\s+'{re.escape(case_name)}'\s*:\s*\{{", src)
    assert m, f"Could not find case '{case_name}' block in main.js"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


# ---------- Backend order test ----------


class _StubPrinter:
    """Capture broadcast events in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def subscribe_tab(self, source_tab_id: str, viewer_tab_id: str) -> None:
        pass

    def rebind_tab(self, old: str, new: str) -> None:
        pass


@pytest.fixture
def temp_history_db() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "task_history.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """
            CREATE TABLE task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                task TEXT,
                events TEXT,
                extra TEXT,
                created_at REAL,
                has_events INTEGER DEFAULT 1
            );
            """,
        )
        conn.commit()
        conn.close()
        yield db


class TestReplaySessionStatusBeforeTaskEvents:
    """Backend invariant: ``_replay_session`` broadcasts
    ``status running=true`` strictly before ``task_events`` when the
    resumed regular chat is still running."""

    def _make_server_with_running_chat(
        self, history_db: Path, chat_id: str, tab_id: str,
    ) -> tuple[VSCodeServer, _StubPrinter]:
        events_blob = json.dumps([
            {"type": "text_delta", "text": "hello"},
        ])
        conn = sqlite3.connect(str(history_db))
        conn.execute(
            "INSERT INTO task_history (chat_id, task, events, extra, "
            "created_at, has_events) VALUES (?,?,?,?,?,?)",
            (chat_id, "test task", events_blob, "{}", 1.0, 1),
        )
        conn.commit()
        conn.close()

        printer = _StubPrinter()
        # ``VSCodeServer.__init__`` requires several args; build a
        # minimal instance via object.__new__ and inject what we need.
        srv = object.__new__(VSCodeServer)
        srv.printer = printer  # type: ignore[assignment]
        _RunningAgentState.running_agent_states.clear()
        srv._state_lock = __import__("threading").RLock()  # type: ignore
        srv._default_model = "test"  # type: ignore[attr-defined]
        srv._tab_chat_views = {}
        return srv, printer

    def test_status_true_broadcast_before_task_events(
        self, temp_history_db: Path,
    ) -> None:
        chat_id = "chat-running-abc"
        tab_id = "tab-resume-1"
        srv, printer = self._make_server_with_running_chat(
            temp_history_db, chat_id, tab_id,
        )

        # Stub the loaders to return a fixed running session.
        fake_result = {
            "events": [{"type": "text_delta", "text": "hi"}],
            "task": "test task",
            "task_id": 1,
            "chat_id": chat_id,
            "extra": "{}",
        }

        # Stub ``_reattach_running_chat`` to report "still running"
        # so the status broadcast path fires.
        with patch.object(
            VSCodeServer, "_reattach_running_chat", return_value=True,
        ), patch.object(
            VSCodeServer, "_get_tab",
        ) as mock_get_tab, patch.object(
            VSCodeServer, "_emit_pending_worktree",
        ), patch(
            "kiss.server.server._load_latest_chat_events_by_chat_id",
            return_value=fake_result,
        ):
            class _StubTab:
                agent = type("A", (), {
                    "resume_chat_by_id": lambda self, _c: None,
                })()
                use_worktree = False
                frontend_closed = False

            mock_get_tab.return_value = _StubTab()
            srv._replay_session(chat_id=chat_id, tab_id=tab_id)

        types = [ev["type"] for ev in printer.events]
        assert "status" in types and "task_events" in types, (
            f"Expected both 'status' and 'task_events' in {types}"
        )
        status_idx = types.index("status")
        events_idx = types.index("task_events")
        assert status_idx < events_idx, (
            "status running=true must be broadcast BEFORE task_events "
            "so the frontend's isRunning flag is true while "
            "replayTaskEvents executes (matching the fresh-run order "
            "and avoiding the chevron-hides-panels bug at replay time). "
            f"Got order: {types}"
        )
        status_ev = printer.events[status_idx]
        assert status_ev.get("running") is True
        assert status_ev.get("tabId") == tab_id


# ---------- Frontend: openSubagentTab + subagentDone ----------


class TestOpenSubagentTabSyncsRunningState:
    """When ``openSubagentTab`` lands on the currently active tab
    (history-load path), the handler must call ``setRunningState``
    and ``applyChevronState`` so the global running state matches the
    sub-agent's actual state."""

    def test_handler_calls_set_running_state_when_active(self) -> None:
        body = _extract_case_body(_read_main_js(), "openSubagentTab")
        # The block guarded by ``subTab.id === activeTabId`` must
        # contain a setRunningState call so the global running state
        # tracks the sub-agent's per-tab isRunning.
        m = re.search(
            r"if\s*\(\s*subTab\.id\s*===\s*activeTabId\b", body,
        )
        assert m, (
            "openSubagentTab must guard its global-state sync with "
            "`if (subTab.id === activeTabId ...)` so it only fires "
            "when the sub-agent tab is the one being viewed"
        )
        # Slice from the guard to the end of the case (the if block
        # extends to roughly the end; using the full tail is safe
        # because nothing useful follows the guard inside this case).
        region = body[m.end():]
        assert "setRunningState(" in region, (
            "openSubagentTab's `subTab.id === activeTabId` branch "
            "must call setRunningState so a history-resumed running "
            "sub-agent tab shows the running spinner/timer (matching "
            "the freshly-launched sub-agent tab's restoreTab path)"
        )
        assert "applyChevronState(" in region, (
            "openSubagentTab's active-tab branch must also call "
            "applyChevronState so the replayed panels of a still-"
            "running sub-agent are unhidden — matching the regular "
            "running-chat resume path which gets the same effect "
            "from its `case 'status':` handler"
        )


class TestSubagentDoneSyncsRunningState:
    """When ``subagentDone`` lands on the currently active tab, the
    handler must flip the global running state to false so the timer
    stops and "Running …" becomes "Done"."""

    def test_handler_calls_set_running_state_false_when_active(
        self,
    ) -> None:
        body = _extract_case_body(_read_main_js(), "subagentDone")
        m = re.search(
            r"if\s*\(\s*doneTab\.id\s*===\s*activeTabId\b", body,
        )
        assert m, (
            "subagentDone must guard its global-state sync with "
            "`if (doneTab.id === activeTabId)` so a background "
            "sub-agent's completion does not clobber the active "
            "tab's running state"
        )
        region = body[m.end():m.end() + 200]
        assert "setRunningState(false)" in region, (
            "subagentDone's active-tab branch must call "
            "setRunningState(false) so the status header flips from "
            "'Running …' to 'Done' and the timer stops — matching "
            "the regular task's `status running=false` handling"
        )
