# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A daemon connection lost after the ``run`` was sent is not a task that "could not run".

Audit 2026-10-30 (S6), BUG: ``dispatch_result``'s generic branch turned
every ``ConnectionError`` into "the task could not run", including the
ones ``daemon_client.run`` raises after the ``run`` frame went out (the
daemon closing the socket mid-run, a frame over the client limit, a
kill whose ``stop`` could not be written).  The task may still be
running on the daemon then.  ``run`` now raises those as
``ConnectionLostError`` and ``dispatch_result`` reports the sub-task
as possibly still running; a failure before the ``run`` went out stays
a plain ``ConnectionError``, "could not run".

Real dispatches against local-WSS daemon stand-ins; nothing is mocked.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

from kiss.agents.sorcar import cron_agent, daemon_client
from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH, make_run_agent_tool
from kiss.tests.agents.sorcar.test_dispatch_stop_cascade import _RecordingDaemon
from kiss.tests.agents.sorcar.test_dispatch_timeout import _LocalDaemon


def test_a_daemon_dropping_a_running_sub_task_reports_that_it_may_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _RecordingDaemon(mode="drop")
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    try:
        out = make_run_agent_tool(str(tmp_path))("dropped child", DEFAULT_AGENT_PATH, timeout="30")
    finally:
        daemon.close()
    assert daemon.run_cmd is not None, "the run never reached the daemon"
    assert out.startswith(
        "Error: the connection to the daemon failed while the sorcar agent task ran",
    ), out
    assert "closed the connection" in out
    assert out.endswith("it MAY STILL BE RUNNING (and spending) on the daemon."), out


def test_only_a_failure_after_the_run_was_sent_is_a_lost_connection(tmp_path: Path) -> None:
    with pytest.raises(ConnectionError, match="Cannot connect") as unreachable:
        daemon_client.run(
            "never sent", endpoint_file=str(tmp_path / "no-daemon.json"), timeout=5.0,
        )
    assert not isinstance(unreachable.value, daemon_client.ConnectionLostError)
    daemon = _RecordingDaemon(mode="drop")
    try:
        with pytest.raises(daemon_client.ConnectionLostError, match="closed the connection"):
            daemon_client.run("dropped child", endpoint_file=daemon.endpoint_file, timeout=10.0)
    finally:
        daemon.close()


class _ClosingDaemon(_LocalDaemon):
    """A daemon stand-in that closes every connection right after the ``auth`` handshake."""

    def __init__(self) -> None:
        super().__init__("kiss_connection_lost_")

    async def _handle(self, ws: ServerConnection) -> None:
        return  # returning closes the connection


class _HoldTheRunUntilTheSocketIsClosed:
    """Trace hook on ``daemon_client.run``: hold the ``run`` frame until the connection is dead.

    Fires at the first line of ``run``'s frame once the command is
    built, i.e. just before it is sent.
    """

    def __init__(self) -> None:
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not daemon_client.run.__code__:
            return None
        if event == "line" and frame.f_locals.get("cmd") is not None and not self.fired:
            self.fired = True
            _wait_until_closed(frame.f_locals["ws"])
        return self


def _wait_until_closed(ws: Any) -> None:
    deadline = time.monotonic() + 10
    while ws.state is not State.CLOSED and time.monotonic() < deadline:
        time.sleep(0.01)


def _run_traced(trace: Any, **kwargs: Any) -> BaseException | None:
    """Call ``daemon_client.run`` on a worker thread traced by *trace*; return what it raised."""
    outcome: list[BaseException | None] = []

    def call() -> None:
        sys.settrace(trace)
        try:
            daemon_client.run("dropped child", timeout=30.0, **kwargs)
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001 - captured for the assertions
            outcome.append(exc)
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=call, name="traced-caller")
    worker.start()
    worker.join(60)
    assert not worker.is_alive()
    return outcome[0]


def test_a_run_the_daemon_closed_on_before_it_was_sent_could_not_run() -> None:
    daemon = _ClosingDaemon()
    trace = _HoldTheRunUntilTheSocketIsClosed()
    try:
        error = _run_traced(trace, endpoint_file=daemon.endpoint_file)
    finally:
        daemon.close()
    assert trace.fired, "the run was never built: the race was not exercised"
    assert type(error) is ConnectionError, repr(error)
    assert "before the task was sent" in str(error)


class _CancelOnceTheSocketIsClosed:
    """Trace hook on ``daemon_client.run``: cancel once the started task's connection is dead.

    At the first line of ``run``'s frame after the task's ``running=true``
    arrived, waits (bounded) until the daemon's close has been processed
    on the client's connection, then sets the cancel: the loop's next
    turn tries to send the ``stop`` on the closed connection.
    """

    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not daemon_client.run.__code__:
            return None
        if event == "line" and frame.f_locals.get("started") and not self.fired:
            self.fired = True
            _wait_until_closed(frame.f_locals["ws"])
            self.cancel.set()
        return self


def test_a_cancel_whose_stop_cannot_be_sent_is_a_lost_connection() -> None:
    daemon = _RecordingDaemon(mode="drop")
    cancel = threading.Event()
    trace = _CancelOnceTheSocketIsClosed(cancel)
    try:
        error = _run_traced(trace, endpoint_file=daemon.endpoint_file, cancel=cancel)
    finally:
        daemon.close()
    assert trace.fired, "the task never started: the race was not exercised"
    assert isinstance(error, daemon_client.ConnectionLostError), repr(error)
    assert "failed while stopping the cancelled task" in str(error)
