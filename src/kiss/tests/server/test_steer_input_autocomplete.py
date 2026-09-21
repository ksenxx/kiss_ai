# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Text typed into a running task's composer is saved for autocomplete.

A steer-mode message (``appendUserMessage``, or a ``run`` sent to a
tab whose task is still running) never creates a ``task_history``
row, so :meth:`_CommandsMixin._echo_injected_prompt` records it in the
``steer_inputs`` table.  These tests drive the real daemon handlers
against a real sqlite file and check the two consumers:

* the fast-complete picker / ghost text (``_complete`` ->
  ``completions`` event, backed by ``_prefix_match_tasks``);
* the ArrowUp history (``_get_input_history`` -> ``inputHistory``
  event, backed by ``_load_input_history``).

The DB-failure branch of the recorder guard is exercised by pointing
the DB path at a directory (sqlite cannot open it): no mocks.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """A real :class:`VSCodeServer` whose broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _steer_texts() -> list[str]:
    rows = th._get_db().execute(
        "SELECT text FROM steer_inputs ORDER BY timestamp",
    ).fetchall()
    return [r["text"] for r in rows]


def _completion_texts(events: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for e in events:
        if e.get("type") == "completions":
            out.extend(c["text"] for c in e["completions"])
    return out


def _input_history(events: list[dict[str, Any]]) -> list[str]:
    hist = [e for e in events if e.get("type") == "inputHistory"]
    assert len(hist) == 1
    return list(hist[0]["tasks"])


class TestSteerInputSavedForAutocomplete:
    """Steer-mode text reaches both autocomplete consumers."""

    def setup_method(self) -> None:
        agent_state.agent_states.clear()
        th._get_db().execute("DELETE FROM steer_inputs")

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def test_append_user_message_is_saved_and_completed(self) -> None:
        server, events = _make_server()
        st = AgentState("task-1", tab_id="tab-1", server_owned=True)
        st.is_task_active = True
        agent_state.register(st)

        server._cmd_append_user_message(
            {"tabId": "tab-1", "prompt": "also update the changelog"},
        )

        assert st.pending_user_messages == ["also update the changelog"]
        assert _steer_texts() == ["also update the changelog"]

        server._complete("also up")
        assert "also update the changelog" in _completion_texts(events)
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert ghost and ghost[-1]["suggestion"] == "date the changelog"

        server._get_input_history()
        assert "also update the changelog" in _input_history(events)

    def test_run_on_busy_tab_is_saved(self) -> None:
        """A ``run`` while the tab's task thread is alive is steer text."""
        server, events = _make_server()
        st = AgentState("task-2", tab_id="tab-2", server_owned=True)
        st.is_task_active = True
        st.task_thread = threading.Thread(target=lambda: None)
        agent_state.register(st)

        server._cmd_run({"tabId": "tab-2", "prompt": "prefer pytest -x"})

        assert st.pending_user_messages == ["prefer pytest -x"]
        assert _steer_texts() == ["prefer pytest -x"]
        assert [e["text"] for e in events if e.get("type") == "prompt"] == [
            "prefer pytest -x",
        ]

    def test_dropped_message_is_not_saved(self) -> None:
        """A message no live task accepts is neither queued nor remembered."""
        server, events = _make_server()
        agent_state.register(
            AgentState("task-idle", tab_id="tab-idle", server_owned=True),
        )

        server._cmd_append_user_message(
            {"tabId": "tab-idle", "prompt": "nobody is listening"},
        )
        server._cmd_append_user_message(
            {"tabId": "no-such-tab", "prompt": "nobody is listening"},
        )

        assert _steer_texts() == []
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_answer_to_pending_question_is_echoed_but_not_saved(self) -> None:
        """A plain message while ``ask_user_question`` blocks IS the answer.

        It is delivered and echoed like a ``userAnswer``, and like one it
        stays out of the autocomplete history (it may be a secret).
        """
        server, events = _make_server()
        st = AgentState("task-ask", tab_id="tab-ask", server_owned=True)
        st.is_task_active = True
        st.pending_ask_question = "API key?"
        st.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(st)

        server._cmd_append_user_message(
            {"tabId": "tab-ask", "prompt": "sk-secret-value"},
        )

        assert st.user_answer_queue.get_nowait() == "sk-secret-value"
        assert st.pending_user_messages == []
        assert [e["text"] for e in events if e.get("type") == "prompt"] == [
            "sk-secret-value",
        ]
        assert _steer_texts() == []

    def test_answer_via_run_on_asking_tab_is_not_saved(self) -> None:
        """Same rule for the ``run``-to-a-tab-awaiting-an-answer path."""
        server, events = _make_server()
        st = AgentState("task-ask2", tab_id="tab-ask2", server_owned=True)
        st.is_task_active = True
        st.task_thread = threading.Thread(target=lambda: None)
        st.pending_ask_question = "password?"
        st.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(st)

        server._cmd_run({"tabId": "tab-ask2", "prompt": "hunter2"})

        assert st.user_answer_queue.get_nowait() == "hunter2"
        assert [e["text"] for e in events if e.get("type") == "prompt"] == [
            "hunter2",
        ]
        assert _steer_texts() == []

    def test_followup_task_list_is_saved(self) -> None:
        """A ``<task>`` list typed mid-run is prompt history too."""
        server, _events = _make_server()
        st = AgentState("task-fu", tab_id="tab-fu", server_owned=True)
        st.is_task_active = True
        agent_state.register(st)

        text = "<task>run the linter</task><task>run the tests</task>"
        server._cmd_append_user_message({"tabId": "tab-fu", "prompt": text})

        assert st.queued_followup_tasks == ["run the linter", "run the tests"]
        assert _steer_texts() == [text]

    def test_repeat_message_saved_once(self) -> None:
        server, _events = _make_server()
        st = AgentState("task-3", tab_id="tab-3", server_owned=True)
        st.is_task_active = True
        agent_state.register(st)

        for _ in range(3):
            server._cmd_append_user_message(
                {"tabId": "tab-3", "prompt": "keep going"},
            )

        assert _steer_texts() == ["keep going"]

    def test_echo_survives_unwritable_db(self, tmp_path: Path) -> None:
        """A failing sqlite write must not swallow the user's prompt echo."""
        server, events = _make_server()
        st = AgentState("task-4", tab_id="tab-4", server_owned=True)
        st.is_task_active = True
        agent_state.register(st)

        saved = (th._DB_PATH, th._KISS_DIR)
        th._close_db()
        # A directory is not a database file: every open fails.
        th._DB_PATH = tmp_path
        th._KISS_DIR = tmp_path
        try:
            server._cmd_append_user_message(
                {"tabId": "tab-4", "prompt": "still echoed"},
            )
        finally:
            th._close_db()
            th._DB_PATH, th._KISS_DIR = saved

        assert st.pending_user_messages == ["still echoed"]
        assert [e["text"] for e in events if e.get("type") == "prompt"] == [
            "still echoed",
        ]
        assert _steer_texts() == []
