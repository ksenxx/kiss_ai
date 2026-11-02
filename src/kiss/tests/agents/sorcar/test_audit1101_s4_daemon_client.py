# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S4): ``daemon_client.run`` under a caller's timeout, a Stop and a handoff.

* F3 — ``run`` checked its *timeout* only in its receive loop.  A
  connection drop sent it into ``_Inbox.reconnect``, whose retries and
  backoff knew only the reconnect window: a 1 s timeout stayed blocked
  for the whole window and ended with ``ConnectionLostError`` instead
  of the documented ``TimeoutError``.  The reconnect now stops at the
  caller's deadline and hands the wait back to ``run``.
* F4 — ``_connect_and_resume`` cleaned the replacement connection up
  only when the daemon closed it.  A Stop (``KeyboardInterrupt``)
  landing in the ``resumeSession`` send left the opened, authenticated
  socket and its reader thread alive for good.  It is closed now.
* F5 — ``_run_ended`` took any ``result`` as proof the task was over,
  so a timeout after a session handoff (``finish(success=False,
  is_continue=True)``) closed the tab of a task still running.  A
  continuation result no longer counts.
* F1 — a reconnect before the run's first stamped event sent a
  ``resumeSession`` with an empty ``taskId``, which the daemon answered
  with the chat's latest row (another task's transcript).  The resume
  now names the run by its run token, the one id the daemon knows it
  by in its setup window (``test_audit1101_s4_resume_by_run_token``
  covers the daemon's side).

The daemon is a real local-WSS stand-in (:func:`kiss.tests.local_ws.fake_daemon`)
on its own event-loop thread; nothing of the client is mocked.  The
Stop of F4 is injected the way the daemon injects one — a
``KeyboardInterrupt`` raised between two bytecodes of the client's
thread, here by a trace function at the send's line.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from kiss.agents.sorcar import daemon_client
from kiss.tests.agents.sorcar.test_daemon_client_tab_close_wave3 import (
    RUNNING,
    SPEND,
    _ScriptedDaemon,
)
from kiss.tests.local_ws import fake_daemon

CLEAR = {"type": "clear", "chat_id": "chat-audit1101"}
CONTINUATION = {
    "type": "result", "success": False, "is_continue": True,
    "summary": "<p>Continue the unfinished task in the next session.</p>",
    "text": "success: false\nis_continue: true\nsummary: keep going\n",
    "cost": "$0.0200", "total_tokens": 7, "step_count": 2,
}
FAILED = {
    "type": "result", "success": False, "is_continue": False,
    "summary": "<p>Task failed after 2 sub-sessions</p>", "cost": "$0.0300",
    "total_tokens": 9, "step_count": 3,
}
CONTINUED_AND_DONE = {**CONTINUATION, "success": True}


class _Daemon:
    """A daemon stand-in on its own loop thread that drops the run and shapes every reconnect.

    The first connection is the run: its ``run`` is answered with a
    ``clear`` naming a chat and a ``status running=true`` (the client
    then holds a resumable run), after which the connection is closed —
    the drop.  Every later connection is a reconnect attempt, handled
    per *later*: ``"refuse"`` closes it before its ``auth`` is answered
    (a daemon that is down), ``"hang"`` never answers the ``auth`` (a
    daemon that accepts but is wedged), ``"accept"`` authenticates it,
    answers its ``resumeSession`` with the ``replayDone`` the daemon
    sends, confirms a ``stop`` with the terminal status and keeps the
    connection until the client closes it.  ``refuse_first`` refuses
    that many reconnects before *later* applies.
    """

    def __init__(self, tmp: Path, later: str, refuse_first: int = 0) -> None:
        self.endpoint_file = tmp / "sorcar-local.json"
        self.later = later
        self.refuse_first = refuse_first
        self.connections = 0
        self.run_cmd: dict[str, Any] = {}
        self.resumes: list[dict[str, Any]] = []
        self.stops: list[dict[str, Any]] = []
        #: Set once a reconnect connection the daemon kept has ended.
        self.reconnect_closed = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(tmp),), daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise AssertionError("the stand-in daemon never came up")

    async def _serve(self, tmp: Path) -> None:
        async with fake_daemon(
            tmp, self._handle, answer_auth=False, endpoint_file=self.endpoint_file,
        ):
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        self.connections += 1
        number = self.connections
        try:
            if number > 1 and (number - 1 <= self.refuse_first or self.later == "refuse"):
                return  # closed before the auth is answered, like a daemon that is down
            auth = json.loads(await ws.recv())
            if number > 1 and self.later == "hang":
                await ws.wait_closed()
                return
            local = auth.get("token") == "test-token"
            await ws.send(json.dumps({"type": "auth_ok", "local": local}))
            if number == 1:
                self.run_cmd = json.loads(await ws.recv())
                tab_id = self.run_cmd["tabId"]
                for event in (CLEAR, RUNNING):
                    await ws.send(json.dumps({**event, "tabId": tab_id}))
                return  # the drop
            resume = json.loads(await ws.recv())
            self.resumes.append(resume)
            await ws.send(json.dumps({"type": "replayDone", "replayId": resume.get("replayId")}))
            async for message in ws:
                cmd = json.loads(message)
                if cmd.get("type") == "stop":
                    self.stops.append(cmd)
                    await ws.send(json.dumps(
                        {"type": "status", "running": False, "tabId": cmd.get("tabId")},
                    ))
        except ConnectionClosed:
            pass
        finally:
            if number > 1:
                self.reconnect_closed.set()

    def close(self) -> None:
        """Stop serving, wait for the loop thread to end and close its loop."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(10)
        assert not self._thread.is_alive(), "the stand-in daemon never stopped"
        self._loop.close()


def _run(
    daemon: _Daemon, trace: Any = None, **kwargs: Any,
) -> tuple[BaseException | None, float]:
    """Run a task against *daemon* on a worker thread; return its error and the seconds it took."""
    out: dict[str, Any] = {}

    def body() -> None:
        if trace is not None:
            sys.settrace(trace)
        try:
            out["result"] = daemon_client.run(
                "audit1101 task", endpoint_file=daemon.endpoint_file, **kwargs,
            )
        except BaseException as exc:  # reported by the test
            out["error"] = exc
        finally:
            sys.settrace(None)

    started = time.monotonic()
    worker = threading.Thread(target=body, name="audit1101-run", daemon=True)
    worker.start()
    worker.join(60)
    assert not worker.is_alive(), "daemon_client.run did not return"
    return out.get("error"), time.monotonic() - started


class _StopAtResumeSend:
    """Trace function: a Stop at the ``resumeSession`` send of ``_connect_and_resume``.

    Raises ``KeyboardInterrupt`` at the line event of the send, i.e.
    after the replacement connection was opened and authenticated and
    before the resume went out, and keeps that connection in :attr:`ws`
    for the test to inspect (and close, should the client not).
    """

    def __init__(self) -> None:
        source, start = inspect.getsourcelines(daemon_client._connect_and_resume)
        self.line = start + next(
            i for i, text in enumerate(source) if "local_endpoint.send(ws," in text
        )
        self.ws: Any = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not daemon_client._connect_and_resume.__code__:
            return None
        if event == "line" and frame.f_lineno == self.line:
            self.ws = frame.f_locals["ws"]
            sys.settrace(None)
            raise KeyboardInterrupt("Stop injected after the replacement connection opened")
        return self


@pytest.mark.parametrize("later", ["refuse", "hang"])
def test_a_timeout_expiring_during_the_reconnect_ends_the_wait_on_time(
    tmp_path: Path, later: str,
) -> None:
    """F3: the daemon drops the run and stays down; a 1 s timeout raises ``TimeoutError`` at 1 s.

    Before: the reconnect's backoff (``refuse``) or its opening
    handshake (``hang``) ran on for the whole 6 s window and raised
    ``ConnectionLostError``.
    """
    daemon = _Daemon(tmp_path, later)
    try:
        error, elapsed = _run(daemon, timeout=1.0, reconnect_seconds=6.0)
    finally:
        daemon.close()
    assert type(error) is TimeoutError, repr(error)
    assert str(error) == "Task did not finish within 1.0 seconds"
    assert 1.0 <= elapsed < 4.0, elapsed
    assert daemon.connections >= 2, "no reconnect was attempted"


def test_stop_on_timeout_after_the_reconnect_was_cut_short_still_stops_the_task(
    tmp_path: Path,
) -> None:
    """The timeout wins over a failing reconnect, and the stop path reconnects once to send it.

    The first reconnect is refused; the deadline passes during its
    backoff.  ``stop_on_timeout`` then reconnects (the daemon is back),
    sends the stop and gets its confirmation: ``StoppedOnTimeoutError``.
    """
    daemon = _Daemon(tmp_path, "accept", refuse_first=1)
    try:
        error, elapsed = _run(
            daemon, timeout=1.0, reconnect_seconds=30.0, stop_on_timeout=True,
        )
    finally:
        daemon.close()
    assert isinstance(error, daemon_client.StoppedOnTimeoutError), repr(error)
    assert elapsed < 10.0, elapsed
    assert len(daemon.resumes) == 1 and len(daemon.stops) == 1, (daemon.resumes, daemon.stops)
    assert daemon.stops[0]["tabId"] == daemon.resumes[0]["tabId"]
    assert error.result.reconnects == 1


def test_a_reconnect_before_the_first_stamped_event_names_the_run_by_its_token(
    tmp_path: Path,
) -> None:
    """F1: the resume of a run with no persisted id yet carries the run token as ``taskId``.

    The stand-in daemon drops the run after ``clear`` and ``status``
    (neither stamps a task id) and answers the reconnect's resume with a
    bare ``replayDone``, so the run ends as lost.  Before: ``taskId``
    was ``""``.
    """
    daemon = _Daemon(tmp_path, "accept")
    try:
        error, _elapsed = _run(daemon, timeout=10.0, reconnect_seconds=30.0)
    finally:
        daemon.close()
    assert type(error) is daemon_client.ConnectionLostError, repr(error)
    token = daemon.run_cmd["taskId"]
    assert token and [r["taskId"] for r in daemon.resumes] == [token], (daemon.resumes, token)


def test_a_stop_during_the_resume_send_closes_the_replacement_connection(tmp_path: Path) -> None:
    """F4: a Stop between opening the replacement connection and sending the resume closes it.

    Before: ``run`` raised the ``KeyboardInterrupt`` but the opened
    socket (and its reader thread) stayed alive; the daemon never saw
    the connection end.
    """
    daemon = _Daemon(tmp_path, "accept")
    stop = _StopAtResumeSend()
    try:
        error, _ = _run(daemon, stop, timeout=30.0, reconnect_seconds=30.0)
        assert isinstance(error, KeyboardInterrupt), repr(error)
        assert stop.ws is not None, "the Stop did not land at the resume send"
        assert daemon.resumes == []
        assert daemon.reconnect_closed.wait(5), "the replacement connection was left open"
        assert stop.ws.state is State.CLOSED
        assert not stop.ws.recv_events_thread.is_alive()
    finally:
        if stop.ws is not None:
            stop.ws.close()
        daemon.close()


@pytest.mark.parametrize(
    ("script", "closed"),
    [
        pytest.param([RUNNING, SPEND, CONTINUATION], False, id="continuation"),
        pytest.param([RUNNING, SPEND, CONTINUATION, FAILED], True, id="then-failed"),
        pytest.param([RUNNING, SPEND, CONTINUED_AND_DONE], True, id="continue-and-success"),
    ],
)
def test_a_timeout_after_a_continuation_result_keeps_the_running_tasks_tab(
    script: list[dict[str, Any]], closed: bool,
) -> None:
    """F5: a session handoff's ``result`` is no proof the task is over; a terminal one is.

    Before: the ``closeTab`` went out after the continuation too, hiding
    a task that was still running on every surface.
    """
    daemon = _ScriptedDaemon(script)
    try:
        with pytest.raises(TimeoutError):
            daemon_client.run("continuing task", endpoint_file=daemon.endpoint_file, timeout=0.3)
        assert daemon.sent() == (["closeTab"] if closed else [])
    finally:
        daemon.close()
