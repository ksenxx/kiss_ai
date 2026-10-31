"""A rate-limited or unreachable Matrix send fails at once and leaves nothing running.

matrix-nio's ``AsyncClient._send`` retries a 429 and a failed connection forever
unless ``AsyncClientConfig.max_limit_exceeded`` / ``max_timeouts`` are set (both
default to ``None``).  ``MatrixChannelBackend.send_message`` therefore blocked for
``_run``'s 120 s timeout on a 429 instead of reaching ``_raise_on_error``,
and the coroutine ``_run`` abandoned kept retrying on the background loop, so it
could deliver the reply after ``ChannelRunner._send_with_retry`` had resent it.

The sends go through the real nio client the backend builds
(``_client_from_config``) on the backend's real background loop, against a local
HTTP homeserver stand-in or a port nothing listens on.  Each blocking send runs
on a helper thread joined with a bound, so the pre-fix behaviour fails the test
instead of hanging it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import socket
import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.third_party_agents.matrix.matrix_sea import (
    MatrixChannelBackend,
    _client_from_config,
)
from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookServer,
    serve_json_webhook,
)

SEND_BOUND_SECONDS = 15.0


@pytest.fixture
def homeserver() -> Iterator[JsonWebhookServer]:
    """A local HTTP server answering every PUT with the test's canned JSON reply."""
    yield from serve_json_webhook()


@pytest.fixture
def refusing_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _backend_for(homeserver_url: str) -> MatrixChannelBackend:
    """Build a backend whose real nio client talks to *homeserver_url*."""
    backend = MatrixChannelBackend()
    backend._client = _client_from_config(
        {"homeserver_url": homeserver_url, "access_token": "tok", "user_id": "@bot:example.org"}
    )
    return backend


def _send_and_record(backend: MatrixChannelBackend, outcome: list[BaseException]) -> None:
    """Send one message; record the exception it raised, if any."""
    try:
        backend.send_message("!room:example.org", "hello")
    except BaseException as exc:  # noqa: BLE001 — the test inspects whatever escaped
        outcome.append(exc)


def _bounded_send(backend: MatrixChannelBackend) -> BaseException | None:
    """Run ``send_message`` on a helper thread; return what it raised.

    Fails the test when the send is still blocked after
    :data:`SEND_BOUND_SECONDS` (the pre-fix client retried forever).
    """
    outcome: list[BaseException] = []
    thread = threading.Thread(target=_send_and_record, args=(backend, outcome), daemon=True)
    thread.start()
    thread.join(SEND_BOUND_SECONDS)
    assert not thread.is_alive(), "send_message is still retrying inside nio"
    return outcome[0] if outcome else None


def test_rate_limited_send_raises_at_once_and_is_not_retried(
    homeserver: JsonWebhookServer,
) -> None:
    """A 429 reaches ``_raise_on_error``; nio sends the request exactly once."""
    homeserver.response_status = 429
    homeserver.response_body = {
        "errcode": "M_LIMIT_EXCEEDED",
        "error": "Too Many Requests",
        "retry_after_ms": 50,
    }
    backend = _backend_for(homeserver.base_url)
    try:
        error = _bounded_send(backend)
        assert isinstance(error, RuntimeError)
        assert "M_LIMIT_EXCEEDED" in str(error)
        # nio's retry loop runs inside the send's coroutine, which has returned.
        assert len(homeserver.requests) == 1
    finally:
        backend.disconnect()


def test_refused_connection_raises_at_once(refusing_port: int) -> None:
    """A homeserver that refuses connections makes the send raise, not time out."""
    backend = _backend_for(f"http://127.0.0.1:{refusing_port}")
    try:
        error = _bounded_send(backend)
        assert error is not None
        assert not isinstance(error, TimeoutError)
    finally:
        backend.disconnect()


class _Sleeper:
    """An hour-long coroutine that reports when it started and when it was cancelled."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def sleep(self) -> None:
        """Sleep for an hour; set :attr:`cancelled` when cancelled."""
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def stop_once_started(self, frame: Any, event: str, arg: Any) -> Any:
        """Tracer raising ``KeyboardInterrupt`` as ``Future.result`` is called.

        It waits for the coroutine to be running first, so the Stop lands
        while ``_run`` waits on a live coroutine.
        """
        if event == "call" and frame.f_code is concurrent.futures.Future.result.__code__:
            assert self.started.wait(5)
            raise KeyboardInterrupt("injected stop")
        return None


def _started_backend() -> MatrixChannelBackend:
    """Return a backend whose background loop is already running a coroutine."""
    backend = MatrixChannelBackend()
    backend._run(asyncio.sleep(0))
    return backend


def test_run_timeout_cancels_the_coroutine() -> None:
    """A coroutine that outlives ``_run``'s timeout is cancelled, not left running."""
    backend = _started_backend()
    sleeper = _Sleeper()
    try:
        with pytest.raises(TimeoutError):
            backend._run(sleeper.sleep(), timeout=0.5)
        assert sleeper.started.is_set()
        assert sleeper.cancelled.wait(5)
    finally:
        backend.disconnect()


def test_stopped_run_cancels_the_coroutine() -> None:
    """A Stop landing while ``_run`` waits cancels the coroutine too.

    The Stop is a real ``KeyboardInterrupt`` raised by a tracer as ``_run``
    calls ``Future.result``, the delivery model of ``PyThreadState_SetAsyncExc``.
    """
    backend = _started_backend()
    sleeper = _Sleeper()
    try:
        sys.settrace(sleeper.stop_once_started)
        try:
            with pytest.raises(KeyboardInterrupt, match="injected stop"):
                backend._run(sleeper.sleep(), timeout=60)
        finally:
            sys.settrace(None)
        assert sleeper.cancelled.wait(5)
    finally:
        backend.disconnect()
