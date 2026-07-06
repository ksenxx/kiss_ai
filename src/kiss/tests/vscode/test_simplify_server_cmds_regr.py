# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking in behavior of code paths that are
being simplified in ``server.py`` / ``task_runner.py`` / ``commands.py``.

No mocks/patches/fakes: a real :class:`JsonPrinter` subclass captures
broadcasts, and persistence goes to the real SQLite database under the
test ``KISS_HOME`` (set by the session conftest).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.vscode.commands import _parse_int
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer, _extra_for_replay


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records broadcasts and tab cleanups."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.cleaned_tabs: list[str] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it to stdout."""
        self.events.append(event)

    def cleanup_tab(self, tab_id: str) -> None:
        """Record per-tab cleanup calls."""
        self.cleaned_tabs.append(tab_id)


def _make_server() -> tuple[VSCodeServer, _CapturePrinter]:
    printer = _CapturePrinter()
    return VSCodeServer(printer=printer), printer


# ---------------------------------------------------------------------------
# _extra_for_replay (server.py S1)
# ---------------------------------------------------------------------------


def test_extra_for_replay_non_str_and_empty() -> None:
    assert _extra_for_replay(None) == ""
    assert _extra_for_replay(123) == ""
    assert _extra_for_replay({"model": "x"}) == ""
    assert _extra_for_replay("") == ""


def test_extra_for_replay_unparseable_string_returned_as_is() -> None:
    assert _extra_for_replay("{not json") == "{not json"


def test_extra_for_replay_non_dict_json_is_dropped() -> None:
    assert _extra_for_replay("[1, 2]") == ""
    assert _extra_for_replay("5") == ""
    assert _extra_for_replay('"s"') == ""


def test_extra_for_replay_without_stripped_keys_unchanged() -> None:
    original = json.dumps({"startTs": 12, "work_dir": "/x", "tokens": 7})
    assert _extra_for_replay(original) == original


def test_extra_for_replay_strips_global_setting_keys() -> None:
    payload = {
        "model": "m1",
        "is_worktree": True,
        "is_parallel": False,
        "auto_commit_mode": True,
        "startTs": 1,
        "endTs": 2,
        "tokens": 3,
    }
    result = _extra_for_replay(json.dumps(payload))
    assert json.loads(result) == {"startTs": 1, "endTs": 2, "tokens": 3}


# ---------------------------------------------------------------------------
# commands._parse_int
# ---------------------------------------------------------------------------


def test_parse_int() -> None:
    assert _parse_int("5") == 5
    assert _parse_int(7) == 7
    assert _parse_int(3.7) == 3
    assert _parse_int(True) == 1
    assert _parse_int(None) is None
    assert _parse_int("abc") is None
    assert _parse_int([1]) is None


# ---------------------------------------------------------------------------
# Unknown-command routing (server.py S3 site in _handle_command)
# ---------------------------------------------------------------------------


def test_unknown_command_reply_stamped_with_conn_id() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._handle_command(
        {"type": "definitelyNotACommand", "tabId": "t1", "connId": "c1"}
    )
    assert len(printer.events) == 1
    event = printer.events[0]
    assert event["type"] == "error"
    assert "Unknown command" in event["text"]
    assert event["tabId"] == "t1"
    assert event["connId"] == "c1"


def test_unknown_command_without_conn_id_has_no_conn_id_key() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._handle_command({"type": "definitelyNotACommand"})
    assert len(printer.events) == 1
    assert printer.events[0]["type"] == "error"
    assert "connId" not in printer.events[0]


def test_unknown_command_non_string_type_routed_to_error() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._handle_command({"type": [1, 2], "connId": "c2"})
    assert len(printer.events) == 1
    assert printer.events[0]["type"] == "error"
    assert printer.events[0]["connId"] == "c2"


# ---------------------------------------------------------------------------
# taskId guards in command handlers (commands.py C1)
# ---------------------------------------------------------------------------


def test_bad_task_ids_are_ignored_without_broadcast() -> None:
    server, printer = _make_server()
    printer.events.clear()
    # Non-string taskId: deleteTask must be a no-op.
    server._handle_command({"type": "deleteTask", "taskId": 123})
    # Empty taskId: setFavorite must be a no-op.
    server._handle_command(
        {"type": "setFavorite", "taskId": "", "isFavorite": True}
    )
    # No chatId and no (string) taskId: resumeSession must be a no-op.
    server._handle_command(
        {"type": "resumeSession", "chatId": "", "taskId": None, "tabId": "t1"}
    )
    assert printer.events == []
    assert printer.cleaned_tabs == []


# ---------------------------------------------------------------------------
# connId stamping on reply events (server.py S3)
# ---------------------------------------------------------------------------


def test_get_frequent_tasks_conn_id_stamping() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._get_frequent_tasks(conn_id="c9")
    server._get_frequent_tasks()
    assert len(printer.events) == 2
    assert printer.events[0]["type"] == "frequentTasks"
    assert printer.events[0]["connId"] == "c9"
    assert printer.events[1]["type"] == "frequentTasks"
    assert "connId" not in printer.events[1]


def test_get_input_history_conn_id_stamping() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._get_input_history(conn_id="c3")
    server._get_input_history()
    assert len(printer.events) == 2
    assert printer.events[0]["type"] == "inputHistory"
    assert printer.events[0]["connId"] == "c3"
    assert printer.events[1]["type"] == "inputHistory"
    assert "connId" not in printer.events[1]


def test_get_models_conn_id_stamping() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._get_models(conn_id="c4")
    server._get_models()
    models_events = [e for e in printer.events if e["type"] == "models"]
    assert len(models_events) == 2
    assert models_events[0]["connId"] == "c4"
    assert "connId" not in models_events[1]
    assert isinstance(models_events[0]["models"], list)


# ---------------------------------------------------------------------------
# cleanup_tab guard (server.py S4) via _replay_session no-result branch
# ---------------------------------------------------------------------------


def test_replay_session_missing_chat_cleans_tab() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._replay_session("no-such-chat-" + uuid.uuid4().hex, tab_id="tabX")
    assert printer.cleaned_tabs == ["tabX"]
    # No running agent to rebind → no status/task_events broadcast.
    assert printer.events == []


def test_replay_session_empty_tab_id_is_noop() -> None:
    server, printer = _make_server()
    printer.events.clear()
    server._replay_session("whatever-chat", tab_id="")
    assert printer.events == []
    assert printer.cleaned_tabs == []


def test_replay_session_with_plain_printer_lacking_cleanup_tab() -> None:
    """The getattr guard must tolerate printers without cleanup_tab."""

    class _MinimalPrinter(JsonPrinter):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[dict[str, Any]] = []

        def broadcast(self, event: dict[str, Any]) -> None:
            """Record *event*."""
            self.events.append(event)

    printer = _MinimalPrinter()
    if hasattr(JsonPrinter, "cleanup_tab"):
        # Base class provides it; the guard is exercised elsewhere.
        # Still run the call to assert no crash.
        pass
    server = VSCodeServer(printer=printer)
    printer.events.clear()
    server._replay_session("no-such-chat-" + uuid.uuid4().hex, tab_id="tabZ")
    assert printer.events == []


# ---------------------------------------------------------------------------
# int/str task-id coercion (server.py S2) via _get_history end-to-end
# ---------------------------------------------------------------------------


def test_get_history_coerces_legacy_int_row_ids() -> None:
    tag = uuid.uuid4().hex
    tid_str, _chat1 = persistence._add_task(f"task-str-{tag}", "")
    tid_int, _chat2 = persistence._add_task(f"task-int-{tag}", "")
    int_id = int(str(uuid.uuid4().int)[:12])
    # Rewrite the row id through the persistence layer's own connection.
    # A raw ``sqlite3.connect(_current_db_path())`` can silently create a
    # brand-new empty database file (→ "no such table: task_history")
    # when an earlier test swapped or deleted the DB file while the
    # thread-local connection stayed cached, so it is order-dependent.
    db = persistence._get_db()
    with persistence._rw_lock.write_lock():
        db.execute(
            "UPDATE task_history SET id = ? WHERE id = ?", (int_id, tid_int)
        )

    server, printer = _make_server()
    printer.events.clear()
    server._get_history(None, conn_id="h1")
    history_events = [e for e in printer.events if e["type"] == "history"]
    assert len(history_events) == 1
    event = history_events[0]
    assert event["connId"] == "h1"
    by_title = {s["title"]: s for s in event["sessions"]}
    str_session = by_title[f"task-str-{tag}"]
    int_session = by_title[f"task-int-{tag}"]
    # String ids pass through unchanged.
    assert str_session["task_id"] == tid_str
    # Legacy int row ids are coerced to strings.
    assert int_session["task_id"] == str(int_id)
