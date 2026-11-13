# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: Stop pressed while an Anthropic stream is live is honoured promptly.

``AnthropicModel._create_message`` runs its loop through the shared
:meth:`~kiss.core.models.model.Model._watched_events` wrapper instead
of hand-running a ``StreamAbortWatchdog``.  The scenario the watchdog
exists for is pinned here: the provider has sent part of a thinking
block and then keeps the connection alive with ``ping`` events (bytes
that reset the httpx read clock, which the SDK filters out before
yielding), so the agent thread is parked in ``recv()``.  A Stop pressed
*after* the first token must unwind as ``KeyboardInterrupt`` well
before the stall timeout, with the thinking bracket closed and the
partial usage kept for billing.

The other end of a request is pinned too: a Stop, or the deadline on the
headers, while the endpoint's address is still being looked up, when the
request has no socket yet.

A real ``ThreadingHTTPServer`` speaks genuine Anthropic SSE to the real
``anthropic`` SDK; no mocks or patches.
"""

from __future__ import annotations

import functools
import socket
import threading
import time
from collections.abc import Generator
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

from kiss.core import stop_signal
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.stream_abort import is_stall_error
from kiss.tests.core.models.anthropic_sse_harness import (
    ScriptedAnthropicServer,
    sse,
    text_message_stream,
)
from kiss.tests.server.pause import Pause

_MODEL = "claude-stop-mid-stream-under-test"
# Long enough that a stop reported only by the stall clock fails the test.
_STALL = 30.0
_DEADLINE = 20.0
_FIRST_TOKEN = "Let me think…"

_PREFIX = b"".join(
    [
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stop",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": _MODEL,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": _FIRST_TOKEN},
            },
        ),
    ]
)


class _PingingHandler(BaseHTTPRequestHandler):
    """Send the thinking prefix, then only ``ping`` events until released."""

    protocol_version = "HTTP/1.1"
    release: threading.Event

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_PREFIX) + 1_000_000))
        self.end_headers()
        self.wfile.write(_PREFIX)
        self.wfile.flush()
        while not self.release.wait(timeout=0.1):
            try:
                self.wfile.write(sse("ping", {"type": "ping"}))
                self.wfile.flush()
            except OSError:
                break
        self.close_connection = True
        self.connection.close()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def endpoint() -> Generator[str]:
    """A real endpoint that never finishes its reply on its own."""
    release = threading.Event()
    handler = type("_Handler", (_PingingHandler,), {"release": release})
    server = _DaemonServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_stop_after_first_token_aborts_the_parked_stream(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """A Stop during a live stream unwinds as KeyboardInterrupt long before the stall clock."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", endpoint)
    thinking: list[bool] = []
    first_token = threading.Event()

    def on_token(token: str) -> None:
        if token:
            first_token.set()

    model = AnthropicModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL},
        token_callback=on_token,
        thinking_callback=thinking.append,
    )
    model.initialize("Think about it.")
    stop = threading.Event()
    outcome: list[BaseException | None] = []

    def target() -> None:
        stop_signal.set_thread_stop_event(stop)
        try:
            model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    assert first_token.wait(_DEADLINE), "the stream never delivered its first token"
    pressed_at = time.monotonic()
    stop.set()
    worker.join(_DEADLINE)
    assert not worker.is_alive(), f"generate() still running after {_DEADLINE}s"
    elapsed = time.monotonic() - pressed_at

    assert isinstance(outcome[0], KeyboardInterrupt), repr(outcome[0])
    assert elapsed < _STALL / 2, f"stop took {elapsed:.1f}s; only the stall clock fired"
    assert thinking == [True, False], thinking
    assert model._thinking_open is False
    # The usage carried by message_start was billed by the provider.
    partial = model.take_partial_usage_response()
    assert partial is not None
    assert model.extract_input_output_token_counts_from_response(partial)[0] == 3
    assert model.take_partial_usage_response() is None


def _looking_up(port: int | None, frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher for ``socket.getaddrinfo``: the lookup of a connect to *port*."""
    return frame_locals.get("port") == port


def ended_during_its_lookup(
    model: AnthropicModel, base_url: str, stopped: bool
) -> BaseException | None:
    """Return what ends ``model.generate()`` while the lookup of *base_url*'s host is held.

    The lookup is held at its entry, as a resolver that does not answer
    holds it.  With *stopped* the task's Stop is pressed then; without,
    the request is left to the deadline on its headers.
    """
    stop = threading.Event()
    looking_up = functools.partial(_looking_up, urlsplit(base_url).port)
    with (
        futures.ThreadPoolExecutor(
            1, initializer=stop_signal.set_thread_stop_event, initargs=(stop,)
        ) as task,  # bound to the Stop the way the runner binds a task thread
        Pause(socket.getaddrinfo.__code__, looking_up) as held,
    ):
        turn = task.submit(model.generate)
        assert held.paused.wait(10), "the request never reached its address lookup"
        if stopped:
            stop.set()
        futures.wait([turn], _DEADLINE)
        assert turn.done(), "generate() still waits for the address lookup"
        return turn.exception()


@pytest.mark.parametrize("stopped", [True, False], ids=["stop", "deadline"])
def test_a_request_ends_while_its_endpoint_is_still_being_looked_up(
    stopped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop and the header deadline end a request whose endpoint's address is not known yet.

    This client's pool was the one never made abortable: until the lookup
    returned its header watchdog had no socket to shut down, so
    ``generate`` waited for as long as the resolver did and then sent the
    request it had been told to stop.  The deadline is twice the stall
    timeout plus the SDK's longest backoff, 8 s.
    """
    config = {} if stopped else {"stream_stall_timeout": 0.05}
    with ScriptedAnthropicServer(text_message_stream("hello")) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        model = AnthropicModel(_MODEL, api_key="test-key", model_config=config)
        model.initialize("hi")
        try:
            error = ended_during_its_lookup(model, server.base_url, stopped)
            if stopped:
                assert isinstance(error, KeyboardInterrupt), repr(error)
            else:
                assert error is not None and is_stall_error(error), repr(error)
            assert server.requests == [], "the request was sent after it was cut"
            if stopped:  # the cut pool connects again: the same client serves the next turn
                assert model.generate()[0] == "hello"
                assert len(server.requests) == 1
        finally:
            model.client.close()
