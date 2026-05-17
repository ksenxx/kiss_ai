"""Regression: loading a still-running task into a tab whose id equals
the chat_id (the common case under the ``tab_id == chat_id``
invariant — e.g. after a VS Code reload, or when a user clicks the
history row for a chat whose live ``_TabState`` is still alive under
``_tab_states[chat_id]``) must broadcast ``status running=true``
BEFORE ``task_events``.

Without this:

  * ``_reattach_running_chat`` previously bailed out early when
    ``chat_id == new_tab_id`` (returned ``False`` without emitting
    any signal), so ``_replay_session`` did NOT broadcast
    ``status:true``.
  * The frontend's ``isRunning`` flag stayed ``false`` while
    ``replayTaskEvents`` ran, and ``applyChevronState`` at the end of
    replay collapsed every panel (``.chv-hidden``).
  * Subsequent live events from the still-running agent (which IS
    broadcasting under the same tab id, since it IS the source tab)
    were rendered into already-hidden ``.collapsible`` panels — the
    user saw no events and a collapsed chevron in the task fixed
    panel.

The fix: ``_reattach_running_chat`` no longer treats
``chat_id == new_tab_id`` as a bail-out.  When the source tab IS the
new tab, it skips ``subscribe_tab`` (the agent is already broadcasting
under that tab id — no fan-out needed) but still returns ``True`` so
``_replay_session`` emits the ``status:true`` event before the
``task_events`` replay.  This matches the fresh-run order
(``_run_task`` emits ``status:true`` first, then events) so a tab
that re-loads a still-running task reaches the same frontend state
as a tab that runs a task fresh.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from kiss.agents.vscode.server import VSCodeServer


class _StubPrinter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.subscribe_calls: list[tuple[str, str]] = []
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def subscribe_tab(self, source_tab_id: str, viewer_tab_id: str) -> None:
        self.subscribe_calls.append((source_tab_id, viewer_tab_id))

    def rebind_tab(self, old: str, new: str) -> None:
        pass


class _StubTabState:
    """Stand-in for a still-running ``_TabState``."""

    def __init__(self) -> None:
        self.task_thread = type(
            "T", (), {"is_alive": lambda self: True},
        )()
        self.is_task_active = True
        self.is_merging = False
        self.frontend_closed = False
        self.use_worktree = False
        self.agent = type(
            "A", (), {"resume_chat_by_id": lambda self, _c: None},
        )()


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


def _make_server() -> tuple[VSCodeServer, _StubPrinter]:
    srv = object.__new__(VSCodeServer)
    printer = _StubPrinter()
    srv.printer = printer  # type: ignore[assignment]
    srv._tab_states = {}  # type: ignore[attr-defined]
    srv._state_lock = threading.RLock()  # type: ignore[attr-defined,assignment]
    srv._default_model = "test"  # type: ignore[attr-defined]
    return srv, printer


class TestReattachRunningChatSameTabId:
    """``_reattach_running_chat`` must report a running re-attach
    even when ``chat_id == new_tab_id`` (the source tab IS the new
    tab — tab_id==chat_id invariant)."""

    def test_returns_true_when_running_state_exists_same_id(self) -> None:
        srv, printer = _make_server()
        chat_id = "chat-and-tab-same-id"
        srv._tab_states[chat_id] = _StubTabState()  # type: ignore[assignment]

        result = srv._reattach_running_chat(chat_id, chat_id)

        assert result is True, (
            "When the running source tab is the same as the new tab, "
            "_reattach_running_chat must still return True so the "
            "caller emits status:true before task_events."
        )
        # Self-subscription must NOT be registered: the agent is
        # already broadcasting under that tab id, fan-out is a no-op
        # and ``subscribe_tab`` correctly refuses self-subscription.
        assert printer.subscribe_calls == [], (
            "Source tab IS the new tab — no fan-out subscription "
            "should be registered."
        )

    def test_returns_false_when_no_running_state(self) -> None:
        srv, _ = _make_server()
        # No _tab_states entry → nothing to reattach to.
        assert srv._reattach_running_chat("chat-x", "chat-x") is False

    def test_subscribe_called_when_ids_differ(self) -> None:
        srv, printer = _make_server()
        chat_id = "chat-1"
        new_tab_id = "tab-2"
        srv._tab_states[chat_id] = _StubTabState()  # type: ignore[assignment]

        result = srv._reattach_running_chat(chat_id, new_tab_id)

        assert result is True
        assert printer.subscribe_calls == [(chat_id, new_tab_id)], (
            "Cross-tab resume: viewer must be subscribed to source "
            "so live broadcasts fan out to both tab ids."
        )


class TestReplaySessionSameTabIdEmitsStatusFirst:
    """End-to-end through ``_replay_session``: when the resumed chat
    is still running and ``tab_id == chat_id``, ``status running=true``
    must precede ``task_events``."""

    def test_status_before_task_events_same_tab_id(
        self, temp_history_db: Path,
    ) -> None:
        srv, printer = _make_server()
        chat_id = "chat-same-as-tab"
        # The running _TabState lives under chat_id, which IS the
        # frontend's new tab id (tab_id==chat_id invariant).
        srv._tab_states[chat_id] = _StubTabState()  # type: ignore[assignment]

        events_blob = json.dumps([
            {"type": "text_delta", "text": "hi"},
        ])
        conn = sqlite3.connect(str(temp_history_db))
        conn.execute(
            "INSERT INTO task_history (chat_id, task, events, extra, "
            "created_at, has_events) VALUES (?,?,?,?,?,?)",
            (chat_id, "running task", events_blob, "{}", 1.0, 1),
        )
        conn.commit()
        conn.close()

        fake_result = {
            "events": [{"type": "text_delta", "text": "hi"}],
            "task": "running task",
            "task_id": 1,
            "chat_id": chat_id,
            "extra": "{}",
        }

        with patch.object(
            VSCodeServer, "_get_tab",
            return_value=srv._tab_states[chat_id],
        ), patch.object(
            VSCodeServer, "_emit_pending_worktree",
        ), patch(
            "kiss.agents.vscode.server._load_latest_chat_events_by_chat_id",
            return_value=fake_result,
        ):
            # tab_id == chat_id — the critical case.
            srv._replay_session(chat_id=chat_id, tab_id=chat_id)

        types = [ev["type"] for ev in printer.events]
        assert "status" in types, (
            "_replay_session must emit a status event for a still-"
            "running same-tab-id resume — without it the frontend's "
            "isRunning flag stays false and replayed panels are "
            "hidden by applyChevronState."
        )
        assert "task_events" in types
        assert types.index("status") < types.index("task_events"), (
            f"status must precede task_events; got {types}"
        )
        status_ev = printer.events[types.index("status")]
        assert status_ev.get("running") is True
        assert status_ev.get("tabId") == chat_id

        # Self-subscription must not happen.
        assert printer.subscribe_calls == []
