"""Side-channel spend (``/ask`` answers, task updates) reaches the task's cost.

A side channel can end while its task still runs (the spend is banked on
the task's live agent) or after the task finished (the spend is added to
the task's row and every finished ancestor's row; a running ancestor
banks it on its live agent).  The ``/ask`` answerer is dispatched to the
daemon, so its spend arrives in the ``TaskResult`` and is charged the
same way.
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.persistence import (
    _add_late_task_usage,
    _add_task,
    _get_db,
    _load_chat_events_by_task_id,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _agent_usage
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state, commands
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.server.task_update import charge_side_channel_usage
from kiss.tests.conftest import requires_unix_sockets


def _row(task_id: str) -> tuple[int, float, int]:
    """The persisted ``(tokens, cost, steps)`` of *task_id*."""
    row = _get_db().execute(
        "SELECT tokens, cost, steps FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row[0]), float(row[1]), int(row[2])


def _task(finished: bool, parent: str = "", cost: float = 1.0) -> str:
    """Add a task row (finished when *finished*); return its id."""
    extra: dict[str, Any] = {"cost": cost, "tokens": 100, "steps": 2}
    if finished:
        extra["endTs"] = int(time.time() * 1000)
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = _add_task("side-channel spend task", "", extra)
    return task_id


def _server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """A real server whose printer's broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = _capture  # type: ignore[assignment]
    return server, events


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Leave no agent states behind."""
    yield
    for state in agent_state.snapshot():
        agent_state.unregister(state.task_id, state)


def test_late_usage_updates_the_finished_row_and_finished_ancestors() -> None:
    """Every finished row on the chain gets the spend; the UI is told."""
    root = _task(finished=True, cost=5.0)
    child = _task(finished=True, parent=root, cost=2.0)
    server, events = _server()
    agent = SorcarAgent("late-usage-child")

    charge_side_channel_usage(server.printer, agent, child, 0.25, 50, 3)

    assert _row(child) == (150, 2.25, 5)
    assert _row(root) == (150, 5.25, 5)
    assert _agent_usage(agent) == (0.0, 0, 0)
    usage = [e for e in events if e.get("type") == "usage_info"]
    assert {e["cost"] for e in usage} == {"$2.2500", "$5.2500"}
    assert any(e.get("type") == "tasks_updated" for e in events)


def test_running_task_banks_the_spend_on_its_live_agent() -> None:
    """While the task runs its row is untouched and its agent pays."""
    task = _task(finished=False)
    agent = SorcarAgent("late-usage-running")

    charge_side_channel_usage(None, agent, task, 0.5, 70, 1)

    assert _row(task) == (100, 1.0, 2)
    assert _agent_usage(agent) == (pytest.approx(0.5), 70, 1)


def test_finished_subagent_of_a_running_parent_charges_the_parent() -> None:
    """The first unfinished ancestor's live agent receives the spend."""
    root = _task(finished=False, cost=3.0)
    child = _task(finished=True, parent=root)
    parent_agent = WorktreeSorcarAgent("late-usage-parent")
    state = AgentState(root, agent=parent_agent, tab_id="tab-late", server_owned=True)
    agent_state.register(state)

    charge_side_channel_usage(None, SorcarAgent("child"), child, 0.75, 10, 0)

    assert _row(child) == (110, 1.75, 2)
    assert _row(root) == (100, 3.0, 2)
    assert _agent_usage(parent_agent) == (pytest.approx(0.75), 10, 0)


def test_running_ancestor_without_a_live_agent_is_skipped() -> None:
    """The finished rows are still updated; nothing else is charged."""
    root = _task(finished=False)
    child = _task(finished=True, parent=root)
    updated, running = _add_late_task_usage(child, 1, 0.1, 0)
    assert running == root
    assert [u[0] for u in updated] == [child]
    charge_side_channel_usage(None, SorcarAgent("orphan"), child, 0.1, 1, 0)
    assert _row(child)[1] == pytest.approx(1.2)
    assert _row(root) == (100, 1.0, 2)


def test_missing_rows_and_zero_spend_charge_nothing() -> None:
    """An unknown task id or a zero spend is a no-op."""
    assert _add_late_task_usage("no-such-task", 1, 1.0, 1) == ([], "")
    task = _task(finished=True)
    agent = SorcarAgent("zero")
    charge_side_channel_usage(None, agent, task, 0.0, 0, 0)
    charge_side_channel_usage(None, agent, "", 1.0, 1, 1)
    charge_side_channel_usage(None, agent, "no-such-task", 1.0, 1, 1)
    assert _row(task) == (100, 1.0, 2)
    assert _agent_usage(agent) == (0.0, 0, 0)


class _AnswerDaemon:
    """A real UDS daemon stand-in that answers one ``/ask`` run.

    Sends ``status running=true``, a successful ``result`` carrying
    *cost* / tokens / steps, then ``status running=false``.
    """

    def __init__(self, cost: str, wait_for_stop: bool = False) -> None:
        """Bind a UNIX-domain listener in a fresh temp dir.

        With *wait_for_stop* the task never finishes on its own: the
        result (a failure carrying the spend) and the terminal status
        are sent only once the client's ``stop`` arrives.
        """
        self.cost = cost
        self.wait_for_stop = wait_for_stop
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_ask_spend_"))
        self.sock_path = self._dir / "daemon.sock"
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            reader = conn.makefile("rb")
            tab_id = json.loads(reader.readline().decode("utf-8")).get("tabId", "")
            if self.wait_for_stop:
                conn.sendall(json.dumps(
                    {"type": "status", "running": True, "tabId": tab_id},
                ).encode() + b"\n")
                while json.loads(reader.readline() or b"{}").get("type") != "stop":
                    pass
            for event in (
                {"type": "status", "running": True},
                {
                    "type": "result", "taskId": "ask-child",
                    "success": not self.wait_for_stop,
                    "text": "<p>because</p>", "cost": self.cost,
                    "total_tokens": 1234, "step_count": 4,
                },
                {"type": "status", "running": False},
            ):
                conn.sendall(json.dumps({**event, "tabId": tab_id}).encode() + b"\n")
            try:
                reader.readline()
            except OSError:
                pass

    def close(self) -> None:
        """Shut down the listener and remove the temp socket dir."""
        self._srv.close()
        self._thread.join(timeout=10)
        shutil.rmtree(self._dir, ignore_errors=True)


@requires_unix_sockets
def test_ask_answer_spend_is_charged_to_the_running_owner_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end over a real socket: the answer's cost reaches the owner."""
    owner_task = _task(finished=False)
    owner_agent = SorcarAgent("ask-owner")
    daemon = _AnswerDaemon("$0.3515")
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    server, events = _server()
    try:
        server._dispatch_ask_side_channel(
            tab_id="tab-ask", owner_task_id=owner_task, owner_agent=owner_agent,
            chat_id="", question="why?",
        )
        deadline = time.monotonic() + 10
        while not any(e.get("type") == "ask_answer" for e in events):
            assert time.monotonic() < deadline, "no ask_answer broadcast"
            time.sleep(0.02)
    finally:
        daemon.close()
    assert _agent_usage(owner_agent) == (pytest.approx(0.3515), 1234, 4)
    assert _row(owner_task) == (100, 1.0, 2)


@requires_unix_sockets
def test_ask_answer_stopped_on_timeout_still_charges_its_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out answer's spend (from the failure result) is charged."""
    owner_task = _task(finished=True)
    daemon = _AnswerDaemon("$0.2000", wait_for_stop=True)
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(daemon.sock_path))
    monkeypatch.setattr(commands, "_ASK_TIMEOUT_SECONDS", 0.3)
    server, events = _server()
    try:
        server._dispatch_ask_side_channel(
            tab_id="tab-ask", owner_task_id=owner_task,
            owner_agent=SorcarAgent("ask-owner-done"), chat_id="",
            question="why?",
        )
        deadline = time.monotonic() + 10
        while not any(e.get("type") == "ask_answer" for e in events):
            assert time.monotonic() < deadline, "no ask_answer broadcast"
            time.sleep(0.02)
    finally:
        daemon.close()
    assert _row(owner_task) == (1334, pytest.approx(1.2), 6)


def test_tree_token_shares_add_up_to_the_top_level_total() -> None:
    """Fractional per-model token shares are rounded without drift."""
    from kiss.agents.sorcar.persistence import _round_shares

    assert _round_shares({"a": 0.5, "b": 0.5}, 1) == {"a": 1, "b": 0}
    assert _round_shares({"a": 2.4, "b": 0.6}, 3) == {"a": 2, "b": 1}
    assert _round_shares({}, 0) == {}


def test_concurrent_late_charges_publish_the_final_total_last() -> None:
    """The last persisted usage_info always shows the row's final total."""
    task = _task(finished=True)
    threads = [
        threading.Thread(
            target=charge_side_channel_usage,
            args=(None, None, task, 0.01, 1, 0),
        )
        for _ in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    tokens, cost, _ = _row(task)
    assert (tokens, cost) == (116, pytest.approx(1.16))
    session = _load_chat_events_by_task_id(task)
    assert session is not None
    usage = [
        e for e in session["events"]  # type: ignore[attr-defined]
        if e.get("type") == "usage_info"
    ]
    assert len(usage) == 16
    assert usage[-1]["cost"] == "$1.1600"
    assert usage[-1]["total_tokens"] == 116
