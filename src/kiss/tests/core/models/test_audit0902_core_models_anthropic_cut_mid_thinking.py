# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An Anthropic stream that dies mid-thinking must close the thinking bracket.

Sibling error handling that had drifted apart: the Chat Completions,
Responses and Gemini transports all close an open thinking block in a
``finally`` — for a stop, a stall AND any other transport failure —
because ``KISSAgent`` retries a non-``KISSError`` in the same run without
resetting the printer, so a bracket left open renders the retry's whole
answer as reasoning.  ``AnthropicModel._create_message`` closed the
bracket only on its stop and stall paths; a connection dropped by the
provider in the middle of a thinking block left ``thinking_callback``
at ``True`` forever.

Every test runs a REAL ``AnthropicModel`` against a real local
``/v1/messages`` endpoint.  The endpoint announces a long
``Content-Length`` so that every premature end of the body — the
server hanging up, or the watchdog shutting the socket down — surfaces
to the SDK as an ``httpx.RemoteProtocolError`` rather than a clean EOF,
which is the "any other transport failure" route under test.
"""

import json
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.tests.core.models.anthropic_sse_harness import sse

_MODEL = "claude-cut-under-test"
_STALL = 1.0
_DEADLINE = 20.0


def _thinking_prefix() -> bytes:
    """The SSE prefix of a turn that opens a thinking block and says one thing.

    The first delta is empty: the adapter must not open the bracket for
    a delta that carries no text.
    """
    return b"".join(
        [
            sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_cut",
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
                    "delta": {"type": "thinking_delta", "thinking": ""},
                },
            ),
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Let me think…"},
                },
            ),
        ]
    )


class _Script:
    """What the endpoint does after the thinking prefix.

    ``"cut"`` hangs up at once; ``"ping"`` keeps writing SSE ``ping``
    events (bytes that reset httpx's read clock while the SDK filters
    them out, so only the event-level watchdog can notice the stall);
    ``"headers_never"`` never even answers.
    """

    def __init__(self) -> None:
        self.mode = "cut"
        self.release = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    script: _Script

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        if self.script.mode == "headers_never":
            self.script.release.wait(timeout=60.0)
            self.close_connection = True
            self.connection.close()
            return
        body = _thinking_prefix()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body) + 1_000_000))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        if self.script.mode == "ping":
            while not self.script.release.wait(timeout=0.1):
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
def endpoint() -> Generator[tuple[str, _Script]]:
    script = _Script()
    handler = type("_ScriptedHandler", (_Handler,), {"script": script})
    server = _DaemonServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", script
    finally:
        script.release.set()
        server.shutdown()
        server.server_close()


class _Run:
    """One ``generate()`` on a worker thread, with its callbacks recorded."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str, stop: bool
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
        self.thinking_events: list[bool] = []
        self.tokens: list[str] = []
        self.model = AnthropicModel(
            _MODEL,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL},
            token_callback=self.tokens.append,
            thinking_callback=self.thinking_events.append,
        )
        self.model.initialize("Think about it.")
        self.stop = stop
        self.error: BaseException | None = None
        self.result: Any = None

    def _target(self) -> None:
        if self.stop:
            event = threading.Event()
            event.set()
            stop_signal.set_thread_stop_event(event)
        try:
            self.result = self.model.generate()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            self.error = exc

    def __call__(self) -> BaseException | None:
        worker = threading.Thread(target=self._target, daemon=True)
        started = time.monotonic()
        worker.start()
        worker.join(_DEADLINE)
        assert not worker.is_alive(), f"generate() still running after {_DEADLINE}s"
        self.elapsed = time.monotonic() - started
        return self.error


def _assert_bracket_closed(run: _Run) -> None:
    assert "".join(run.tokens) == "Let me think…"
    assert run.thinking_events == [True, False], json.dumps(run.thinking_events)
    assert run.model._thinking_open is False


class TestEveryExitClosesTheThinkingBracket:
    """Stop, stall and provider failure must all end with ``thinking_callback(False)``."""

    def test_connection_cut_mid_thinking(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: tuple[str, _Script]
    ) -> None:
        base_url, script = endpoint
        script.mode = "cut"
        run = _Run(monkeypatch, base_url, stop=False)
        error = run()
        # The failure is the provider's, not a stall and not a stop.
        assert error is not None
        assert not isinstance(error, (TimeoutError, KeyboardInterrupt)), repr(error)
        _assert_bracket_closed(run)

    def test_connection_cut_while_stop_was_requested(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: tuple[str, _Script]
    ) -> None:
        base_url, script = endpoint
        script.mode = "cut"
        run = _Run(monkeypatch, base_url, stop=True)
        assert isinstance(run(), KeyboardInterrupt)
        _assert_bracket_closed(run)

    def test_watchdog_abort_surfacing_as_protocol_error_is_a_stall(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: tuple[str, _Script]
    ) -> None:
        base_url, script = endpoint
        script.mode = "ping"
        run = _Run(monkeypatch, base_url, stop=False)
        error = run()
        assert isinstance(error, TimeoutError), repr(error)
        assert "stalled" in str(error) and _MODEL in str(error)
        _assert_bracket_closed(run)

    def test_headers_never_arrive_while_stop_was_requested(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: tuple[str, _Script]
    ) -> None:
        base_url, script = endpoint
        script.mode = "headers_never"
        run = _Run(monkeypatch, base_url, stop=True)
        assert isinstance(run(), KeyboardInterrupt)
        assert run.thinking_events == []
        assert run.model._thinking_open is False
