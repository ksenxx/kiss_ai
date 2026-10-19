# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the recent / frequent ranking of the ``/`` popup.

The composer's command popup lists the three most recently run
commands first, then the ones run often, then the rest.  The ranking
comes from the task history (a slash run keeps its raw ``/xxx text``
as the row's task text), is computed by
:func:`kiss.agents.sorcar.persistence._load_slash_command_usage` and
shipped as the ``recent`` / ``frequent`` lists of the ``seaCommands``
event (:func:`kiss.server.commands.sea_commands_event`), which the
``getSeaCommands`` handler and the registry-rescan broadcast share.

A real temporary ``history.db`` is used; nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.server.commands import sea_commands_event
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer


@pytest.fixture
def history_db(tmp_path: Path) -> Iterator[None]:
    """Point the persistence module at an empty temporary ``history.db``."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir()
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    try:
        yield
    finally:
        th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _run(task: str, ts: float, parent_task_id: str = "") -> None:
    """Record a finished run of *task* at time *ts* in the task history."""
    extra: dict[str, object] | None = (
        {"parent_task_id": parent_task_id} if parent_task_id else None
    )
    task_id, _chat = th._add_task(task, "", extra)
    db = th._get_db()
    with th._rw_lock.write_lock(), th._immediate_txn(db):
        db.execute("UPDATE task_history SET timestamp = ? WHERE id = ?", (ts, task_id))


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records broadcasts."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def test_usage_counts_runs_per_command_and_keeps_the_latest_time(history_db: None) -> None:
    """Rows are grouped by command name; sub-agent rows and non-commands are skipped."""
    _run("/review the diff", 100.0)
    _run("/review again", 300.0)
    _run("/slack send hi", 200.0)
    _run("/gmail", 50.0)
    # A sub-agent's row is not a run the user made.
    _run("/gcal tomorrow", 400.0, parent_task_id="a" * 32)
    # Not commands: plain text, a lone slash, a name glued to
    # punctuation, a path.
    _run("plain prompt", 500.0)
    _run("/", 500.0)
    _run("/review.py is broken", 500.0)
    _run("/tmp/x.py has a bug", 500.0)

    assert th._load_slash_command_usage() == {
        "review": (2, 300.0),
        "slack": (1, 200.0),
        "gmail": (1, 50.0),
    }


def test_usage_counts_messages_typed_into_a_running_chat(history_db: None) -> None:
    """A ``/ask`` typed while a task runs is a steer input, not a task row."""
    th._record_steer_input("/ask what is this?")
    th._record_steer_input("/ask and that?")
    th._record_steer_input("/ask what is this?")  # same text: one row
    th._record_steer_input("just steering text")
    _run("/ask older", 1.0)
    usage = th._load_slash_command_usage()
    assert set(usage) == {"ask"}
    count, last = usage["ask"]
    assert count == 3
    assert last > 1.0


def test_event_lists_three_recent_then_frequent_then_nothing_else(history_db: None) -> None:
    """``recent`` is the newest three, ``frequent`` the rest run twice or more, by count."""
    commands = ["alpha", "beta", "delta", "eps", "gamma", "zeta"]
    # zeta: 1 run, newest.  beta: 2 runs.  gamma: 1 run.  Those three are
    # the most recent.  alpha: 3 runs, delta: 4 runs, eps: 1 run -> the
    # frequent group is delta then alpha; eps (one run) is "the rest".
    _run("/zeta", 900.0)
    _run("/beta", 800.0)
    _run("/beta", 100.0)
    _run("/gamma", 700.0)
    _run("/alpha", 600.0)
    _run("/alpha", 50.0)
    _run("/alpha", 40.0)
    _run("/delta", 500.0)
    for ts in (30.0, 20.0, 10.0):
        _run("/delta", ts)
    _run("/eps", 400.0)
    # Run often, but no longer registered: never listed.
    for ts in (1.0, 2.0, 3.0):
        _run("/retired", ts)

    event = sea_commands_event(commands)
    assert event == {
        "type": "seaCommands",
        "commands": commands,
        "recent": ["zeta", "beta", "gamma"],
        "frequent": ["delta", "alpha"],
    }


def test_frequent_ties_keep_the_newer_command_first(history_db: None) -> None:
    """Equal run counts are ordered by the most recent run."""
    _run("/a", 900.0)
    _run("/b", 800.0)
    _run("/c", 700.0)
    _run("/old", 100.0)
    _run("/old", 90.0)
    _run("/new", 200.0)
    _run("/new", 80.0)
    event = sea_commands_event(["a", "b", "c", "new", "old"])
    assert event["recent"] == ["a", "b", "c"]
    assert event["frequent"] == ["new", "old"]


def test_event_without_history_has_empty_rankings(history_db: None) -> None:
    """A fresh install lists every command in name order and nothing ranked."""
    event = sea_commands_event(["gmail", "slack"])
    assert event["recent"] == []
    assert event["frequent"] == []
    assert event["commands"] == ["gmail", "slack"]


def test_event_survives_an_unreadable_history(history_db: None) -> None:
    """A corrupt ``history.db`` costs the ranking, not the command list."""
    th._get_db().close()
    th._db_conn = None
    th._DB_PATH.write_bytes(b"not a sqlite file" * 64)
    event = sea_commands_event(["gmail", "slack"])
    assert event["commands"] == ["gmail", "slack"]
    assert event["recent"] == []
    assert event["frequent"] == []


def test_get_sea_commands_handler_ships_the_ranking(history_db: None) -> None:
    """The ``getSeaCommands`` reply carries ``recent`` and ``frequent``."""
    sea_commands._reset_for_tests()
    try:
        names = sea_commands.refresh_registry()
        assert "slack" in names and "gmail" in names
        _run("/slack hello", 900.0)
        _run("/gmail inbox", 800.0)
        _run("/gmail archive", 100.0)
        printer = _CapturePrinter()
        server = AgentServer(printer=printer)
        server._handle_command({"type": "getSeaCommands", "connId": "c1"})
        events = [e for e in printer.events if e["type"] == "seaCommands"]
        assert len(events) == 1
        assert events[0]["connId"] == "c1"
        assert events[0]["commands"] == names
        assert events[0]["recent"] == ["slack", "gmail"]
        assert events[0]["frequent"] == []
    finally:
        sea_commands._reset_for_tests()
