# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Only a refused token resets the Matrix cursor, and a Stop before the first step wins.

Two gaps were left in ``matrix_sea`` after wave 4:

* ``_refuses_since_token`` read every non-429 4xx as a refused ``since`` token,
  so a proxy's 408 timeout (or a 401/403) made ``poll_messages`` re-sync with
  ``since=None``, adopt the fresh cursor and skip every message in between.
  Homeservers refuse a token with HTTP 400 only (Synapse: ``M_UNKNOWN``
  "Invalid stream token"; Dendrite: ``M_INVALID_PARAM``).
* ``_start_on_loop`` checked the caller's future, then created the task.  A Stop
  landing after that check, or after the task was created but before its first
  step ran, queued its cancel *behind* that first step: the coroutine started,
  and a send went out, after the caller already had its Stop.

Every request goes through the real nio client against a local homeserver
stand-in, on the backend's real background loop.  The Stop is a real
``KeyboardInterrupt`` raised by a caller-thread tracer; a loop-thread tracer
holds ``_start_on_loop`` at the racy instant until the Stop has returned.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import linecache
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.matrix.matrix_sea import _start_on_loop
from kiss.tests.agents.third_party_agents.recording_http import serve_recording
from kiss.tests.agents.third_party_agents.test_matrix_send_retry_wave2 import (
    _Sleeper,
    _started_backend,
)
from kiss.tests.agents.third_party_agents.test_matrix_wave4 import (
    _ROOM,
    _backend_for,
    _Homeserver,
    _HomeserverHandler,
)


class _KeepAliveHandler(_HomeserverHandler):
    """The homeserver stand-in over HTTP/1.1: a warmed connection stays open for the next send."""

    protocol_version = "HTTP/1.1"


@pytest.fixture
def homeserver(isolated_kiss_home: Path) -> Iterator[_Homeserver]:
    """A local homeserver stand-in; ``KISS_HOME`` is isolated so no real config is read."""
    yield from serve_recording(_HomeserverHandler, _Homeserver)


@pytest.fixture
def keepalive_homeserver(isolated_kiss_home: Path) -> Iterator[_Homeserver]:
    """The homeserver stand-in keeping connections alive, like a real homeserver."""
    yield from serve_recording(_KeepAliveHandler, _Homeserver)


@pytest.mark.parametrize(
    "failure",
    [
        (408, "text/html", "<html><body><h1>408 Request Time-out</h1></body></html>"),
        (
            401,
            "application/json",
            json.dumps({"errcode": "M_UNKNOWN_TOKEN", "error": "Unrecognised access token"}),
        ),
        (403, "application/json", json.dumps({"errcode": "M_FORBIDDEN", "error": "Forbidden"})),
    ],
    ids=["proxy-408", "401", "403"],
)
def test_failure_that_is_not_about_the_token_keeps_the_cursor(
    homeserver: _Homeserver, failure: tuple[int, str, str]
) -> None:
    """A 4xx other than 400 is no verdict on ``since``: the tick is a no-op, not a full sync.

    The old classifier answered every such failure with ``since=None`` and
    adopted the fresh cursor, skipping the backlog after ``cursor-1``.
    """
    homeserver.failure = failure
    homeserver.fail_since_only = True
    backend = _backend_for(homeserver)
    try:
        assert backend.poll_messages(_ROOM, "cursor-1") == ([], "cursor-1")
        assert [r["query"].get("since") for r in homeserver.requests] == [["cursor-1"]]
        assert backend._next_batch == ""
        # Once the failure clears, the kept cursor picks up the backlog.
        homeserver.failure = None
        messages, cursor = backend.poll_messages(_ROOM, "cursor-1")
    finally:
        backend.disconnect()
    assert [m["text"] for m in messages] == ["newest message"]
    assert cursor == "cursor-full"


@pytest.mark.parametrize(
    "refusal",
    [
        {"errcode": "M_UNKNOWN", "error": "Invalid stream token"},
        {"errcode": "M_INVALID_PARAM", "error": "malformed sync token"},
    ],
    ids=["synapse", "dendrite"],
)
def test_refused_token_still_falls_back_to_a_full_sync(
    homeserver: _Homeserver, refusal: dict[str, str]
) -> None:
    """A 400 refusing the token is retried once without it, and the fresh cursor is adopted."""
    homeserver.failure = (400, "application/json", json.dumps(refusal))
    homeserver.fail_since_only = True
    backend = _backend_for(homeserver)
    try:
        messages, cursor = backend.poll_messages(_ROOM, "stale-cursor")
    finally:
        backend.disconnect()
    assert [r["query"].get("since") for r in homeserver.requests] == [["stale-cursor"], None]
    assert [m["text"] for m in messages] == ["newest message"]
    assert cursor == "cursor-full"


class _HoldStartOnLoop:
    """Loop-thread tracer holding ``_start_on_loop`` once, until :attr:`release` is set.

    With *line* set it stops before the line whose source contains it
    (``"ensure_future"``: the old cancellation check already passed, no task
    exists yet).  With *line* ``None`` it stops as the function returns (the
    task exists, its first step is still queued).
    """

    def __init__(self, line: str | None) -> None:
        self.line = line
        self.held = threading.Event()
        self.release = threading.Event()
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Global tracer: trace ``_start_on_loop`` frames only, until fired."""
        if self.fired or frame.f_code is not _start_on_loop.__code__:
            return None
        return self.local

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        """Local tracer: hold the loop thread at the chosen instant."""
        if self.fired:
            return None
        if self.line is None:
            at_target = event == "return"
        else:
            source = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            at_target = event == "line" and self.line in source
        if at_target:
            self.fired = True
            self.held.set()
            self.release.wait(10)
        return self.local


class _StopWhileHeld:
    """Caller-thread tracer: as ``_run`` waits on its future, raise Stop once the loop is held."""

    def __init__(self, hold: _HoldStartOnLoop) -> None:
        self.hold = hold
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Raise ``KeyboardInterrupt`` on the first call of ``Future.result``."""
        if (
            event == "call"
            and not self.fired
            and frame.f_code is concurrent.futures.Future.result.__code__
        ):
            self.fired = True
            assert self.hold.held.wait(5)
            raise KeyboardInterrupt("injected stop")
        return None


def _swap_tracer(tracer: Any) -> Any:
    """Install *tracer* on the calling thread; return the one it replaces (e.g. coverage's)."""
    previous = sys.gettrace()
    sys.settrace(tracer)
    return previous


def _call_and_report(
    func: Callable[..., Any], args: tuple[Any, ...], result: list[Any], done: threading.Event
) -> None:
    """Loop callback: append ``func(*args)`` to *result*, then set *done*."""
    result.append(func(*args))
    done.set()


def _on_loop_thread(loop: asyncio.AbstractEventLoop, func: Callable[..., Any], *args: Any) -> Any:
    """Run ``func(*args)`` on *loop*'s thread, wait for it and return its result."""
    result: list[Any] = []
    done = threading.Event()
    loop.call_soon_threadsafe(_call_and_report, func, args, result, done)
    assert done.wait(5)
    return result[0]


def _run_stopped_while_held(backend: Any, coro: Any, line: str | None) -> None:
    """Call ``backend._run(coro)``; a Stop hits it while ``_start_on_loop`` is held at *line*.

    The loop is released only after the Stop has propagated out of ``_run``,
    then drained, so whatever the stopped coroutine would do has been done.
    """
    loop = backend._loop
    assert loop is not None
    hold = _HoldStartOnLoop(line)
    loop_tracer = _on_loop_thread(loop, _swap_tracer, hold)
    caller_tracer = _swap_tracer(_StopWhileHeld(hold))
    try:
        with pytest.raises(KeyboardInterrupt):
            backend._run(coro)
    finally:
        sys.settrace(caller_tracer)
        hold.release.set()
        _on_loop_thread(loop, sys.settrace, loop_tracer)
    assert hold.fired
    backend._run(asyncio.sleep(0))


_WINDOWS = pytest.mark.parametrize(
    "line", ["ensure_future", None], ids=["after-the-check", "task-not-yet-stepped"]
)


@_WINDOWS
def test_stop_before_the_first_step_never_starts_the_coroutine(
    isolated_kiss_home: Path, line: str | None
) -> None:
    """A Stop that returned before the task's first step keeps the coroutine from starting."""
    backend = _started_backend()
    sleeper = _Sleeper()
    coro = sleeper.sleep()
    try:
        _run_stopped_while_held(backend, coro, line)
        assert not sleeper.started.is_set()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
    finally:
        backend.disconnect()


@_WINDOWS
def test_stop_before_the_first_step_sends_nothing(
    keepalive_homeserver: _Homeserver, line: str | None
) -> None:
    """The review's reproduction: a send stopped before its first step never reaches the server.

    The keep-alive connection is warmed first, so the stopped send's first
    step would write the whole request at once (the old code delivered it).
    """
    homeserver = keepalive_homeserver
    backend = _backend_for(homeserver)
    try:
        backend.send_message(_ROOM, "warm-up")
        sent_before = len(homeserver.requests)
        coro = backend._client.room_send(
            _ROOM, "m.room.message", {"msgtype": "m.text", "body": "message after Stop"}
        )
        _run_stopped_while_held(backend, coro, line)
        # A later send on the same loop is answered only after the loop ran the
        # stopped coroutine's step, if it ever had one.
        backend.send_message(_ROOM, "after the Stop")
    finally:
        backend.disconnect()
    bodies = [r["body"] for r in homeserver.requests[sent_before:] if "/send/" in r["path"]]
    assert bodies == [{"msgtype": "m.text", "body": "after the Stop"}]
