# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop pressed before Gemini's response headers must land as fast as one after.

Finding A2-F1 (``tmp/findings-A2.md``): Gemini arms the stream watchdog
before its lazy SDK generator issues the request.  A Stop already set in
that window was claimed on the first poll, the abort found no response
to shut down, and the watchdog thread exited — so the task then waited
for the whole byte-level read timeout (``stream_stall_timeout``, 180 s
by default) before the stop was reported.

Real ``google-genai`` SDK, real httpx, a real loopback SSE server that
holds the headers back, then writes one function-call-only chunk (no
text, so no token callback ever checks the stop flag) and goes silent.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core import stop_signal
from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import chunk, function_call_part

HEADER_DELAY = 1.0
STALL = 4.0
# The watchdog polls every 0.1 s; anything near STALL means it died.
ABORT_BUDGET = 1.0


class _Clock:
    """What the endpoint and the worker observed, in ``time.monotonic()``."""

    def __init__(self) -> None:
        self.headers_sent = threading.Event()
        self.headers_at = 0.0
        self.release = threading.Event()


class _SlowHeadersHandler(BaseHTTPRequestHandler):
    """Holds the headers for ``HEADER_DELAY``, sends one chunk, then nothing."""

    clock: _Clock

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer ``streamGenerateContent`` with delayed headers and silence."""
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(HEADER_DELAY)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.clock.headers_at = time.monotonic()
        self.clock.headers_sent.set()
        body = chunk([function_call_part("noop", {})])
        self.wfile.write(f"data: {json.dumps(body)}\r\n\r\n".encode())
        self.wfile.flush()
        self.clock.release.wait(timeout=60.0)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Generator[_Clock]:
    """A loopback Gemini endpoint with slow headers, wired into the SDK."""
    clock = _Clock()
    handler = type("_ClockedHandler", (_SlowHeadersHandler,), {"clock": clock})
    server = _DaemonServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield clock
    finally:
        clock.release.set()
        server.shutdown()
        server.server_close()


def _generate_under_stop(stop: threading.Event) -> dict[str, object]:
    """Run one real Gemini turn on a worker thread bound to *stop*.

    Returns the outcome once the worker has unwound: the error it raised
    (or its result) and the time it finished.
    """
    outcome: dict[str, object] = {}

    def target() -> None:
        stop_signal.set_thread_stop_event(stop)
        model = GeminiModel(
            "gemini-stop-under-test",
            api_key="test-key",
            model_config={"stream_stall_timeout": STALL},
            token_callback=lambda _text: None,
        )
        model.initialize("Explain the stop button.")
        try:
            outcome["result"] = model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc
        outcome["finished_at"] = time.monotonic()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(HEADER_DELAY + STALL + 10.0)
    assert not worker.is_alive(), "generate() never unwound"
    return outcome


class TestStopBeforeHeaders:
    """A Stop set before the request is issued must not wait out the stall."""

    def test_pre_header_stop_lands_right_after_the_headers(self, endpoint: _Clock) -> None:
        """The watchdog must keep polling until the socket exists, then abort it."""
        stop = threading.Event()
        stop.set()
        outcome = _generate_under_stop(stop)

        assert isinstance(outcome.get("error"), KeyboardInterrupt), f"got {outcome!r}"
        assert endpoint.headers_sent.is_set()
        after_headers = float(outcome["finished_at"]) - endpoint.headers_at  # type: ignore[arg-type]
        assert after_headers < ABORT_BUDGET, (
            f"stop landed {after_headers:.2f}s after the headers; the watchdog died "
            f"and the task waited for the {STALL}s read timeout"
        )

    def test_post_header_stop_is_the_yardstick(self, endpoint: _Clock) -> None:
        """Control: a Stop after the headers lands within the same budget."""
        stop = threading.Event()
        stop_at: list[float] = []

        def press_stop_after_headers() -> None:
            endpoint.headers_sent.wait(timeout=HEADER_DELAY + 10.0)
            time.sleep(0.2)
            stop_at.append(time.monotonic())
            stop.set()

        threading.Thread(target=press_stop_after_headers, daemon=True).start()
        outcome = _generate_under_stop(stop)

        assert isinstance(outcome.get("error"), KeyboardInterrupt), f"got {outcome!r}"
        after_stop = float(outcome["finished_at"]) - stop_at[0]  # type: ignore[arg-type]
        assert after_stop < ABORT_BUDGET, f"post-header stop took {after_stop:.2f}s"
