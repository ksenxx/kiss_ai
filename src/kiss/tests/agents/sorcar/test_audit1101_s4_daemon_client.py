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
  continuation result no longer counts (nor does any ``result`` since
  audit 2026-11-10, S3-F2: only the task's terminal status does).
* F1 — a reconnect before the run's first stamped event sent a
  ``resumeSession`` with an empty ``taskId``, which the daemon answered
  with the chat's latest row (another task's transcript).  The resume
  now names the run by its run token, the one id the daemon knows it
  by in its setup window (``test_audit1101_s4_resume_by_run_token``
  covers the daemon's side).
* Audit 2026-11-12 (B4) — every send starts a writer thread, and the
  ``RuntimeError`` of one that cannot start (out of threads) was none
  of the errors the stop paths expect: a timed-out run ended with it,
  its task never stopped and reported as plainly failed, and the stop
  of the abort cascade replaced the caller's own Stop with it.

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
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.test_daemon_client_tab_close_wave3 import (
    LOST,
    RUN_TOKEN,
    RUNNING,
    SPEND,
    _ScriptedDaemon,
)
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
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
    daemon: _Daemon | _ScriptedDaemon, trace: Any = None, **kwargs: Any,
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


class _RaiseAt:
    """Trace function: raise *error* at the line of *function* holding *text*.

    With *command*, only in a frame whose ``cmd`` is of that type.  The
    frame's connection is kept in :attr:`ws` for the test to inspect
    (and close, should the client not).
    """

    def __init__(self, function: Any, text: str, error: BaseException, command: str = "") -> None:
        source, start = inspect.getsourcelines(function)
        self.code = function.__code__
        self.line = start + next(i for i, line in enumerate(source) if text in line)
        self.error, self.command = error, command
        self.ws: Any = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not self.code:
            return None
        if (
            event == "line" and frame.f_lineno == self.line
            and frame.f_locals.get("cmd", {}).get("type", "") == self.command
        ):
            self.ws = frame.f_locals["ws"]
            sys.settrace(None)
            raise self.error
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


@pytest.mark.parametrize(
    ("script", "task_id"),
    [
        pytest.param([FAILED | {"taskId": RUN_TOKEN}], "", id="never-started"),
        pytest.param([RUNNING, FAILED | {"taskId": "row-7"}], "row-7", id="started"),
    ],
)
def test_the_run_token_echoed_on_a_result_is_no_task_id(
    script: list[dict[str, Any]], task_id: str,
) -> None:
    """A run that never started reports no history row id; one that did reports its row's.

    A run whose worker thread cannot start has no row: the daemon
    (``_report_run_start_failure``) stamps the client's own run token as
    ``taskId`` on the failure ``result`` and on the terminal status.
    Before: ``TaskResult.task_id`` was that token instead of ``""``.
    """
    daemon = _ScriptedDaemon([*script, {"type": "status", "running": False, "taskId": RUN_TOKEN}])
    try:
        result = daemon_client.run(
            "audit1101 task", endpoint_file=daemon.endpoint_file, timeout=30.0,
        )
    finally:
        daemon.close()
    assert not result.success and result.task_id == task_id, result


def test_a_stop_during_the_resume_send_closes_the_replacement_connection(tmp_path: Path) -> None:
    """F4: a Stop between opening the replacement connection and sending the resume closes it.

    Before: ``run`` raised the ``KeyboardInterrupt`` but the opened
    socket (and its reader thread) stayed alive; the daemon never saw
    the connection end.
    """
    daemon = _Daemon(tmp_path, "accept")
    # after the replacement connection was opened and authenticated, before the resume went out
    stop = _RaiseAt(
        daemon_client._connect_and_resume, "local_endpoint.send(ws,", KeyboardInterrupt("Stop"))
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
    ("function", "text", "command", "error", "script"),
    [
        pytest.param(daemon_client._Inbox.__init__, "self._joiner =", "",
                     KeyboardInterrupt("Stop"), [], id="inbox-construction"),
        pytest.param(daemon_client._read_frames, "threading.Thread(", "",
                     RuntimeError("can't start new thread"), [], id="reader-thread-start"),
        pytest.param(daemon_client._send, "local_endpoint.send(", "closeTab",
                     KeyboardInterrupt("Stop"),
                     [RUNNING, FAILED, {"type": "status", "running": False}], id="closeTab-send"),
        pytest.param(daemon_client._send, "local_endpoint.send(", "stop",
                     KeyboardInterrupt("Stop"), [RUNNING, LOST], id="stop-send"),
    ],
)
def test_an_abort_while_run_owns_its_connection_closes_it(
    function: Any, text: str, command: str, error: BaseException, script: list[dict[str, Any]],
) -> None:
    """B/S3-F2: a Stop, or a reader thread that cannot start, never leaves ``run``'s socket open.

    Before: one landing while the inbox was built, or at the last
    ``stop`` / ``closeTab`` send, skipped the close: the socket and its
    reader threads outlived the call and the daemon never saw it end.
    """
    daemon = _ScriptedDaemon(script)
    stop = _RaiseAt(function, text, error, command)
    before = set(threading.enumerate())
    try:
        raised, _ = _run(daemon, stop, timeout=30.0)
        assert raised is error, repr(raised)
        assert stop.ws is not None, "the abort did not land"
        assert daemon.disconnected.wait(5), "the connection was left open"
        assert stop.ws.state is State.CLOSED
        stop.ws.recv_events_thread.join(5)
        assert not stop.ws.recv_events_thread.is_alive()
        for helper in set(threading.enumerate()) - before:
            if helper.name == "sorcar-local-recv":
                helper.join(5)
                assert not helper.is_alive()
    finally:
        if stop.ws is not None:
            stop.ws.close()
        daemon.close()


@pytest.mark.parametrize(
    "script",
    [
        pytest.param([RUNNING, SPEND, CONTINUATION], id="continuation"),
        pytest.param([RUNNING, SPEND, CONTINUATION, FAILED], id="then-failed"),
        pytest.param([RUNNING, SPEND, CONTINUED_AND_DONE], id="continue-and-success"),
    ],
)
def test_a_timeout_after_a_continuation_result_keeps_the_running_tasks_tab(
    script: list[dict[str, Any]],
) -> None:
    """F5: a session handoff's ``result`` is no proof the task is over.

    Before: the ``closeTab`` went out after the continuation too, hiding
    a task that was still running on every surface.  Nor is a terminal
    ``result`` before the task's terminal status (S3-F2): the daemon
    still persists and cleans up, so the tab stays in all three.
    """
    daemon = _ScriptedDaemon(script)
    try:
        with pytest.raises(TimeoutError):
            daemon_client.run("continuing task", endpoint_file=daemon.endpoint_file, timeout=0.3)
        assert daemon.sent() == []
    finally:
        daemon.close()


def _run_into(out: dict[str, Any], daemon: _ScriptedDaemon, kwargs: dict[str, Any]) -> None:
    """Thread body: run a task against *daemon*; its result or its error goes to *out*."""
    try:
        out["result"] = daemon_client.run(
            "audit1112 task", endpoint_file=daemon.endpoint_file, **kwargs,
        )
    except BaseException as exc:  # reported by the test
        out["error"] = exc


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
@pytest.mark.parametrize(
    ("script", "timeout", "expected"),
    [
        pytest.param([RUNNING], 1.5, daemon_client.ConnectionLostError, id="timeout"),
        pytest.param(
            [CLEAR, RUNNING], 1.5, daemon_client.ConnectionLostError, id="timeout-resumable",
        ),
        pytest.param([RUNNING], None, KeyboardInterrupt, id="caller-stop"),
    ],
)
def test_a_stop_no_thread_is_left_to_send_is_a_lost_connection(
    script: list[dict[str, Any]], timeout: float | None, expected: type[BaseException],
) -> None:
    """Out of threads, the stop of a running task cannot be written: the task may still run.

    A timed-out run ends with ``ConnectionLostError`` (before: the
    writer thread's ``RuntimeError``), also when its run can be resumed:
    a reconnect attempt that finds no thread is a failed attempt of its
    window, as one that finds no daemon is.  A caller's Stop stays the
    error of the run (before: the ``RuntimeError`` of the stop the
    abort cascade sends replaced it).
    """
    daemon = _ScriptedDaemon(script)
    started = threading.Event()
    out: dict[str, Any] = {}
    kwargs = {
        "timeout": timeout, "stop_on_timeout": True, "reconnect_seconds": 1.0,
        "running": started, "cancel": threading.Event(),  # a cancel: the wait wakes every 0.5 s
    }
    worker = threading.Thread(target=_run_into, args=(out, daemon, kwargs), daemon=True)
    worker.start()
    try:
        assert started.wait(10), "the task never started"
        with nproc_limit_lowered_to_one():
            if timeout is None:
                assert inject_async_exception(worker.ident or 0, KeyboardInterrupt) == 1
            worker.join(30)
        assert not worker.is_alive(), "daemon_client.run did not return"
        assert isinstance(out.get("error"), expected), repr(out)
        if timeout is not None:
            assert "can't start new thread" in str(out["error"])
        assert daemon.sent() == []  # no stop reached the daemon: the task runs on
    finally:
        daemon.close()
