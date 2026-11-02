# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C4): a keep-alive ping has an absolute deadline and dies with the task.

The ping's ``timeout`` is the SDK's inactivity timeout: a provider or
proxy that keeps sending response bytes (leading JSON whitespace, a
chunk every few milliseconds) before each read timeout can hold the
non-streamed ``messages.create`` for good.  ``PromptCacheKeepAlive.stop``
then waited for that request without bound, so ending the tool call or
pressing Stop hung the task.  Now the ping is aborted (its socket shut
down) as soon as the task's stop event is set or
``KEEP_ALIVE_TIMEOUT_SECONDS`` have passed in total, while a ping in
flight when a tool simply ends is still awaited and billed.  Round 2:
the abort stays armed until the request ends, so a Stop landing while
the ping's connection is still being set up (no socket in the pool yet)
cuts the request as soon as its connection is up.  Real SDK against a
local chunked HTTP server; nothing is mocked.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

import kiss.core.models.anthropic_model as am
from kiss.core import stop_signal
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.prompt_cache_keepalive import PromptCacheKeepAlive

_REPLY = {
    "id": "msg_ping",
    "type": "message",
    "role": "assistant",
    "model": "claude-under-test",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class _DrippingServer(ThreadingHTTPServer):
    """Answers the first request with a chunk of whitespace every 20 ms until released.

    Later requests (and the first one once ``release`` is set) get the
    complete reply at once.  ``entered`` is set when the first request's
    body starts dripping.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = 0
        self.lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        server: _DrippingServer = self.server  # type: ignore[assignment]
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with server.lock:
            server.requests += 1
            drip = server.requests == 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if drip:
                server.entered.set()
                while not server.release.wait(0.02):
                    self._chunk(b" ")
            self._chunk(json.dumps(_REPLY).encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass  # the client shut the socket down: the ping was aborted

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def server() -> Iterator[_DrippingServer]:
    srv = _DrippingServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.release.set()
        srv.shutdown()
        srv.server_close()
        thread.join(5)


def _model(server: _DrippingServer, http_client: httpx2.Client | None = None) -> AnthropicModel:
    model = AnthropicModel("claude-under-test", api_key="local-only")
    model.client = Anthropic(
        api_key="local-only",
        base_url=f"http://127.0.0.1:{server.server_port}",
        max_retries=0,
        http_client=http_client,
    )
    model.conversation = [{"role": "user", "content": "Keep the local test cache warm"}]
    return model


def _keeper(model: AnthropicModel, owner_stop: threading.Event | None) -> PromptCacheKeepAlive:
    """A keep-alive whose first ping is due at once, created on a thread bound to *owner_stop*."""
    stop_signal.set_thread_stop_event(owner_stop)
    try:
        return PromptCacheKeepAlive(model, {}, [], time.time() - 300, max_pings=1)
    finally:
        stop_signal.set_thread_stop_event(None)


def test_a_task_stop_aborts_a_ping_the_server_keeps_dripping(
    monkeypatch: pytest.MonkeyPatch, server: _DrippingServer
) -> None:
    # The deadline is far away: only the Stop can end the ping in time.
    monkeypatch.setattr(am, "KEEP_ALIVE_TIMEOUT_SECONDS", 10.0)
    model = _model(server)
    owner_stop = threading.Event()
    keeper = _keeper(model, owner_stop)
    # Safety net so a stop() that waits for the server fails instead of hanging.
    safety = threading.Timer(6.0, server.release.set)
    safety.daemon = True
    try:
        with keeper:
            assert server.entered.wait(5), "the ping never reached the server"
            owner_stop.set()
            safety.start()
            started = time.monotonic()
        waited = time.monotonic() - started
        assert waited < 3, f"stop() waited {waited:.1f}s on a dripping ping"
        assert not keeper._thread.is_alive()
        assert keeper.responses == []
        assert keeper.last_ping_at is None
        # The model's client is still good for the next request.
        reply = model.client.with_options(max_retries=0).messages.create(
            model="claude-under-test",
            max_tokens=5,
            messages=[{"role": "user", "content": "again"}],
            timeout=5,
        )
        assert reply.content[0].text == "ok"
    finally:
        safety.cancel()
        server.release.set()
        keeper.stop()
        model.client.close()


def test_a_task_stop_during_connection_setup_still_aborts_the_ping(
    monkeypatch: pytest.MonkeyPatch, server: _DrippingServer
) -> None:
    """The Stop lands while the ping's connection is being set up, when the pool holds
    no socket to shut down yet.  The watchdog must stay armed until the request ends:
    the connection is then shut down as soon as it is up, the ping fails instead of
    reading the server's drip, and stop() returns without waiting for the server.  The
    connect is held with httpcore's documented ``trace`` request extension."""
    monkeypatch.setattr(am, "KEEP_ALIVE_TIMEOUT_SECONDS", 10.0)
    connecting, connect = threading.Event(), threading.Event()

    def trace(event: str, info: dict[str, Any]) -> None:
        if event == "connection.connect_tcp.started":
            connecting.set()
            connect.wait(5)

    def hold_connect(request: httpx2.Request) -> None:
        request.extensions["trace"] = trace

    model = _model(server, httpx2.Client(event_hooks={"request": [hold_connect]}, trust_env=False))
    owner_stop = threading.Event()
    keeper = _keeper(model, owner_stop)
    safety = threading.Timer(6.0, server.release.set)
    safety.daemon = True
    try:
        keeper.__enter__()
        assert connecting.wait(5), "the ping never started connecting"
        owner_stop.set()
        watchdog = next(
            t for t in threading.enumerate() if t.name == "anthropic-keepalive-abort-watchdog"
        )
        # It can only end once the request has (the block is left), and
        # the request cannot end while its connect is held.
        watchdog.join(1.0)
        assert watchdog.is_alive(), "the watchdog retired while the connection was being set up"
        connect.set()
        safety.start()
        started = time.monotonic()
        keeper.stop()
        waited = time.monotonic() - started
        assert waited < 3, f"stop() waited {waited:.1f}s on a ping stopped during its connect"
        assert not keeper._thread.is_alive()
        assert keeper.responses == [], "a ping stopped before it was sent was billed"
        reply = model.client.with_options(max_retries=0).messages.create(
            model="claude-under-test",
            max_tokens=5,
            messages=[{"role": "user", "content": "again"}],
            timeout=5,
        )
        assert reply.content[0].text == "ok"
    finally:
        connect.set()
        safety.cancel()
        server.release.set()
        keeper.stop()
        model.client.close()


def test_a_dripping_ping_ends_at_its_deadline_without_any_stop(
    monkeypatch: pytest.MonkeyPatch, server: _DrippingServer
) -> None:
    monkeypatch.setattr(am, "KEEP_ALIVE_TIMEOUT_SECONDS", 1.0)
    model = _model(server)
    keeper = _keeper(model, None)
    try:
        keeper.__enter__()
        assert server.entered.wait(5), "the ping never reached the server"
        # Nobody stops anything: the ping's own deadline must end it.
        keeper._thread.join(4)
        assert not keeper._thread.is_alive(), "the ping outlived its deadline"
        assert keeper.responses == []
        started = time.monotonic()
        keeper.stop()
        assert time.monotonic() - started < 1
    finally:
        server.release.set()
        keeper.stop()
        model.client.close()


def test_stop_without_a_task_stop_waits_for_the_ping_and_bills_it(
    monkeypatch: pytest.MonkeyPatch, server: _DrippingServer
) -> None:
    """A tool that ends while a ping is in flight: the ping is awaited, not cut.

    ``stop()`` (what leaving the block does) runs on a thread of its own
    while the server is still dripping: it must still be inside it until
    the server releases the reply, and return once it did.  Ordering is
    established by events, not by elapsed time."""
    monkeypatch.setattr(am, "KEEP_ALIVE_TIMEOUT_SECONDS", 10.0)
    model = _model(server)
    owner_stop = threading.Event()  # bound, but never set
    keeper = _keeper(model, owner_stop)
    stopping = threading.Thread(target=keeper.stop, daemon=True)
    try:
        keeper.__enter__()
        assert server.entered.wait(5), "the ping never reached the server"
        stopping.start()
        # Nothing released the reply: stop() can only have returned by
        # giving up on the ping.
        stopping.join(0.5)
        assert stopping.is_alive(), "stop() returned while the ping was still in flight"
        assert keeper._thread.is_alive(), "the ping was cut without any Stop"
        assert keeper.responses == []
        server.release.set()
        stopping.join(5)
        assert not stopping.is_alive(), "stop() did not return once the ping ended"
        assert len(keeper.responses) == 1
        assert keeper.responses[0].content[0].text == "ok"
        assert keeper.last_ping_at is not None
    finally:
        server.release.set()
        keeper.stop()
        model.client.close()
