# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Failed Matrix requests are reported as failures, and a Stop never orphans one.

``_client_from_config`` turns nio's internal retry off, so a 429 comes back at
once as nio's error response.  Three paths of ``matrix_sea`` mishandled that:

* ``poll_messages`` read every failed sync as a rejected ``since`` token and
  re-synced with ``since=None``, adopting the fresh cursor: a rate limit (or a
  proxy's 5xx) skipped every message between the persisted cursor and now.
* the LLM-facing tools (``send_text_message``, ``join_room``, ...) never looked
  at nio's response, so a refused request was reported as ``"ok": true``.
* ``_run`` handed the coroutine to the loop before entering the ``try`` that
  cancels it, so a Stop landing right then left it running with no owner.

Every request goes through the real nio ``AsyncClient`` the backend builds,
against a local homeserver stand-in, on the backend's real background loop.  The
Stop is a real ``KeyboardInterrupt`` raised by a tracer, the delivery model of
``PyThreadState_SetAsyncExc``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest

from kiss.agents.third_party_agents.matrix.matrix_sea import (
    MatrixChannelBackend,
    _client_from_config,
)
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    serve_recording,
)
from kiss.tests.agents.third_party_agents.test_matrix_send_retry_wave2 import (
    _Sleeper,
    _started_backend,
)

_ROOM = "!room:example.org"
_ANN = "@ann:example.org"
_RATE_LIMITED = (
    429,
    "application/json",
    json.dumps({"errcode": "M_LIMIT_EXCEEDED", "error": "Too Many Requests", "retry_after_ms": 50}),
)


def _sync_body() -> dict[str, Any]:
    """Return a sync answer with one text message in :data:`_ROOM` and cursor ``cursor-full``."""
    message = {
        "type": "m.room.message",
        "event_id": "$newest",
        "sender": _ANN,
        "origin_server_ts": 1700000000000,
        "content": {"msgtype": "m.text", "body": "newest message"},
    }
    room = {
        "timeline": {"events": [message], "limited": False, "prev_batch": "p0"},
        "state": {"events": []},
        "ephemeral": {"events": []},
        "account_data": {"events": []},
        "summary": {},
        "unread_notifications": {},
    }
    return {
        "next_batch": "cursor-full",
        "rooms": {"join": {_ROOM: room}, "invite": {}, "leave": {}},
    }


def _success_body(path: str) -> dict[str, Any] | None:
    """Return the success answer of the client-server endpoint at *path* (``None``: unknown)."""
    if path.endswith("/sync"):
        return _sync_body()
    if path.endswith("/joined_rooms"):
        return {"joined_rooms": [_ROOM]}
    if "/join/" in path:
        return {"room_id": _ROOM}
    if path.endswith(("/leave", "/invite", "/kick")):
        return {}
    if "/send/m.room.message/" in path:
        return {"event_id": "$event1"}
    if path.endswith("/joined_members"):
        return {"joined": {_ANN: {"display_name": "Ann", "avatar_url": None}}}
    if path.endswith("/createRoom"):
        return {"room_id": "!new:example.org"}
    if "/profile/" in path:
        return {"displayname": "Ann", "avatar_url": "mxc://example.org/ann"}
    return None


class _Homeserver(RecordingServer):
    """Homeserver stand-in whose ``failure`` (status, content type, body) answers requests.

    ``failure`` answers every request, or only a sync carrying ``since`` when
    ``fail_since_only`` is set; ``None`` lets every endpoint succeed.
    """

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.failure: tuple[int, str, str] | None = None
        self.fail_since_only = False


class _HomeserverHandler(RecordingHandler):
    """Records each request (JSON body included) and answers per :class:`_Homeserver`."""

    server: _Homeserver  # type: ignore[assignment]

    def _serve(self) -> None:
        """Record the request; answer the configured failure or the endpoint's success."""
        entry = self.record_request(self._read_body())
        failure = self.server.failure
        if failure is not None and (not self.server.fail_since_only or "since" in entry["query"]):
            status, content_type, body = failure
            self._reply(status, body, content_type)
            return
        answer = _success_body(unquote(entry["path"]))
        if answer is None:
            self._json(404, {"errcode": "M_UNRECOGNIZED", "error": "Unrecognized request"})
        else:
            self._json(200, answer)

    do_GET = do_POST = do_PUT = _serve  # noqa: N815


@pytest.fixture
def homeserver(isolated_kiss_home: Path) -> Iterator[_Homeserver]:
    """A local homeserver stand-in; ``KISS_HOME`` is isolated so no real config is read."""
    yield from serve_recording(_HomeserverHandler, _Homeserver)


def _backend_for(server: _Homeserver) -> MatrixChannelBackend:
    """Build a backend whose real nio client talks to *server*."""
    backend = MatrixChannelBackend()
    backend._client = _client_from_config(
        {"homeserver_url": server.base_url, "access_token": "tok", "user_id": "@bot:example.org"}
    )
    return backend


@pytest.mark.parametrize(
    "failure",
    [
        _RATE_LIMITED,
        (400, "application/json", json.dumps({"errcode": "M_LIMIT_EXCEEDED", "error": "Slow"})),
        (502, "text/html", "<html>Bad Gateway</html>"),
    ],
    ids=["429", "limit-errcode", "proxy-502"],
)
def test_transient_sync_failure_keeps_the_cursor(
    homeserver: _Homeserver, failure: tuple[int, str, str]
) -> None:
    """A rate limit or a server error says nothing about the cursor: the tick is a no-op.

    Re-syncing with ``since=None`` instead adopted the fresh cursor and skipped
    every message between the persisted cursor and now.
    """
    homeserver.failure = failure
    homeserver.fail_since_only = True
    backend = _backend_for(homeserver)
    try:
        assert backend.poll_messages(_ROOM, "cursor-1") == ([], "cursor-1")
        assert [r["query"].get("since") for r in homeserver.requests] == [["cursor-1"]]
        assert backend._next_batch == ""
        # Once the homeserver recovers, the kept cursor picks up the backlog.
        homeserver.failure = None
        messages, cursor = backend.poll_messages(_ROOM, "cursor-1")
    finally:
        backend.disconnect()
    assert [m["text"] for m in messages] == ["newest message"]
    assert cursor == "cursor-full"
    assert homeserver.requests[-1]["query"].get("since") == ["cursor-1"]


def test_failed_first_sync_is_a_no_op(homeserver: _Homeserver) -> None:
    """Without any cursor a failed sync is not retried and returns the cursor it got."""
    homeserver.failure = _RATE_LIMITED
    backend = _backend_for(homeserver)
    try:
        assert backend.poll_messages(_ROOM, "0") == ([], "0")
    finally:
        backend.disconnect()
    assert len(homeserver.requests) == 1


_TOOL_CALLS: list[tuple[str, tuple[Any, ...]]] = [
    ("list_rooms", ()),
    ("join_room", ("#room:example.org",)),
    ("leave_room", (_ROOM,)),
    ("send_text_message", (_ROOM, "hi")),
    ("send_notice", (_ROOM, "hi")),
    ("get_room_members", (_ROOM,)),
    ("invite_user", (_ROOM, _ANN)),
    ("kick_user", (_ROOM, _ANN, "spam")),
    ("create_room", ("Room", "Topic")),
    ("get_profile", (_ANN,)),
]


@pytest.mark.parametrize(("tool", "args"), _TOOL_CALLS, ids=[name for name, _ in _TOOL_CALLS])
def test_rate_limited_tool_reports_failure(
    homeserver: _Homeserver, tool: str, args: tuple[Any, ...]
) -> None:
    """A 429 nio returns as an error response makes the tool answer ``"ok": false``."""
    homeserver.failure = _RATE_LIMITED
    backend = _backend_for(homeserver)
    try:
        result = json.loads(getattr(backend, tool)(*args))
    finally:
        backend.disconnect()
    assert result["ok"] is False
    assert "M_LIMIT_EXCEEDED Too Many Requests" in result["error"]
    assert len(homeserver.requests) == 1


def test_forbidden_invite_reports_failure(homeserver: _Homeserver) -> None:
    """A refusal other than a rate limit is reported too, with nio's errcode and text."""
    homeserver.failure = (
        403,
        "application/json",
        json.dumps({"errcode": "M_FORBIDDEN", "error": "You are not in this room."}),
    )
    backend = _backend_for(homeserver)
    try:
        result = json.loads(backend.invite_user(_ROOM, _ANN))
    finally:
        backend.disconnect()
    assert result == {
        "ok": False,
        "error": f"Matrix invite of {_ANN} to {_ROOM} failed: "
        "M_FORBIDDEN You are not in this room.",
    }


def test_tools_report_what_the_homeserver_answered(homeserver: _Homeserver) -> None:
    """Every tool answers the homeserver's success, with the request nio sent on the wire."""
    backend = _backend_for(homeserver)
    try:
        assert json.loads(backend.list_rooms()) == {"ok": True, "rooms": [{"id": _ROOM}]}
        assert json.loads(backend.join_room("#room:example.org")) == {
            "ok": True,
            "room_id": _ROOM,
        }
        assert json.loads(backend.leave_room(_ROOM)) == {"ok": True}
        sent = json.loads(backend.send_text_message(_ROOM, "hi"))
        assert sent == {"ok": True, "event_id": "$event1"}
        assert homeserver.requests[-1]["body"] == {"msgtype": "m.text", "body": "hi"}
        noticed = json.loads(backend.send_notice(_ROOM, "note"))
        assert noticed == {"ok": True, "event_id": "$event1"}
        assert homeserver.requests[-1]["body"] == {"msgtype": "m.notice", "body": "note"}
        assert json.loads(backend.get_room_members(_ROOM)) == {
            "ok": True,
            "members": [{"user_id": _ANN, "display_name": "Ann"}],
        }
        assert json.loads(backend.invite_user(_ROOM, _ANN)) == {"ok": True}
        assert homeserver.requests[-1]["body"] == {"user_id": _ANN}
        assert json.loads(backend.kick_user(_ROOM, _ANN, "spam")) == {"ok": True}
        assert homeserver.requests[-1]["body"] == {"user_id": _ANN, "reason": "spam"}
        created = json.loads(backend.create_room("Room", "Topic", is_public=True, alias="room"))
        assert created == {"ok": True, "room_id": "!new:example.org"}
        body = homeserver.requests[-1]["body"]
        assert [body[k] for k in ("name", "topic", "visibility", "room_alias_name")] == [
            "Room",
            "Topic",
            "public",
            "room",
        ]
        assert json.loads(backend.get_profile(_ANN)) == {
            "ok": True,
            "display_name": "Ann",
            "avatar_url": "mxc://example.org/ann",
        }
        backend.send_message(_ROOM, "reply")
        assert homeserver.requests[-1]["body"] == {"msgtype": "m.text", "body": "reply"}
    finally:
        backend.disconnect()


class _StopAfterScheduling:
    """Tracer raising ``KeyboardInterrupt`` as ``loop.call_soon_threadsafe`` returns.

    That is the instant ``_run`` has handed the coroutine to the loop, the Stop
    window the review found.  With *wait_for* set the tracer first waits for
    that event, so the Stop lands while the coroutine already runs.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, wait_for: threading.Event | None) -> None:
        self.code = type(loop).call_soon_threadsafe.__code__
        self.wait_for = wait_for
        self.fired = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Trace only ``call_soon_threadsafe`` frames; raise once, on the first return."""
        if frame.f_code is not self.code:
            return None
        if event == "return" and not self.fired:
            self.fired = True
            if self.wait_for is not None:
                assert self.wait_for.wait(5)
            raise KeyboardInterrupt("injected stop")
        return self


def _hold_loop(busy: threading.Event, release: threading.Event) -> None:
    """Loop callback: report the loop busy, then hold it until *release* is set."""
    busy.set()
    release.wait(10)


def test_stop_while_scheduling_closes_the_coroutine_unstarted() -> None:
    """A Stop raised before the loop picked the coroutine up means it never starts."""
    backend = _started_backend()
    loop = backend._loop
    assert loop is not None
    sleeper = _Sleeper()
    busy, release = threading.Event(), threading.Event()
    loop.call_soon_threadsafe(_hold_loop, busy, release)
    assert busy.wait(5)
    coro = sleeper.sleep()
    try:
        sys.settrace(_StopAfterScheduling(loop, None))
        try:
            with pytest.raises(KeyboardInterrupt, match="injected stop"):
                backend._run(coro, timeout=60)
        finally:
            sys.settrace(None)
            release.set()
        # The loop runs callbacks in order: this one runs after the stopped one.
        backend._run(asyncio.sleep(0))
        assert not sleeper.started.is_set()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
    finally:
        backend.disconnect()


class _Stubborn:
    """A coroutine that swallows its cancellation and finishes normally anyway."""

    def __init__(self) -> None:
        self.cancelled = threading.Event()

    async def run(self) -> str:
        """Sleep for an hour; on cancellation set :attr:`cancelled` and return."""
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
        return "finished anyway"


def _record_loop_error(
    errors: list[dict[str, Any]], loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """Loop exception handler appending each error *context* to *errors*."""
    errors.append(context)


def test_coroutine_finishing_after_its_caller_gave_up_breaks_nothing() -> None:
    """A coroutine that outlives a timed-out ``_run`` and finishes anyway is dropped quietly.

    Its result has nowhere to go (the caller's future is cancelled); copying it
    there would raise inside the loop's callback.
    """
    backend = _started_backend()
    loop = backend._loop
    assert loop is not None
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(functools.partial(_record_loop_error, errors))
    stubborn = _Stubborn()
    try:
        with pytest.raises(TimeoutError):
            backend._run(stubborn.run(), timeout=0.3)
        assert stubborn.cancelled.wait(5)
        assert backend._run(asyncio.sleep(0, result="still usable")) == "still usable"
        assert errors == []
    finally:
        backend.disconnect()


def test_stop_after_the_coroutine_started_cancels_it() -> None:
    """A Stop raised once the coroutine runs, before ``_run`` waits on it, cancels it."""
    backend = _started_backend()
    loop = backend._loop
    assert loop is not None
    sleeper = _Sleeper()
    try:
        sys.settrace(_StopAfterScheduling(loop, sleeper.started))
        try:
            with pytest.raises(KeyboardInterrupt, match="injected stop"):
                backend._run(sleeper.sleep(), timeout=60)
        finally:
            sys.settrace(None)
        assert sleeper.cancelled.wait(5)
    finally:
        backend.disconnect()
