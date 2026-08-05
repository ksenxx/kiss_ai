# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Stop must land while a model stream is silent.

Bug reproduction (post-mortem ``reports/stop_button_delay_2026-08-05.html``,
task ``709ebce3`` in ``~/.kiss/sorcar.db``, 5 Aug 2026):

* A sub-agent issued an Anthropic request at 04:58:34 that returned
  response headers and then delivered NOTHING for 178 seconds.
* The user pressed Stop repeatedly.  ``VSCodeServer._stop_task`` set the
  task's stop event every time, but the flag is only read when the agent
  emits something (``JsonPrinter.print`` / ``token_callback``), and the
  forced ``KeyboardInterrupt`` cannot be delivered to a thread parked in
  a C-level socket read.  The task therefore ignored Stop until the
  stream finally produced its first event, at 05:01:32 — three minutes
  of a button that looked broken.

Fix under test: the requesting thread publishes its stop event via
:mod:`kiss.core.stop_signal` (``JsonPrinter._thread_local.stop_event`` is
a property over it), and ``_StreamStallWatchdog`` watches that event
alongside the stall clock.  A stop now closes the wedged response within
one poll interval, and the aborted call raises ``KeyboardInterrupt`` —
the same signal the cooperative path raises — instead of the retryable
``TimeoutError`` a stall produces, which the agentic loop would answer by
re-asking the model.

Test strategy (no mocks, patches, or fakes): a local
``ThreadingHTTPServer`` speaks the real Anthropic SSE wire format to the
real ``anthropic`` SDK client, routed through the SDK's own
``ANTHROPIC_BASE_URL``, and stalls exactly like the production request
did.  The stop event is bound the way production binds it — by assigning
``JsonPrinter._thread_local.stop_event`` on the thread making the call.
The stall timeout is set far above the test deadline so a passing test
can only be explained by the stop, never by the stall watchdog.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.server.json_printer import JsonPrinter

_MODEL = "claude-stop-under-test"
# Far above the test deadline: the stall watchdog must never be what
# ends these calls.
_STALL_TIMEOUT = 120.0
_DEADLINE = 15.0

_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Finish the task",
        "parameters": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    },
}


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _thinking_prefix(model_name: str) -> list[bytes]:
    """A turn that starts thinking and then goes quiet forever."""
    return [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "Let me think…",
                },
            },
        ),
    ]


class _SilentState:
    """Per-test policy for the local Anthropic-compatible server."""

    def __init__(self) -> None:
        self.mode = "silent"
        self.stop = threading.Event()
        self.serving = threading.Event()

    def reset(self) -> None:
        """Restore the default policy (headers, then total silence)."""
        self.mode = "silent"
        self.stop = threading.Event()
        self.serving = threading.Event()


_STATE = _SilentState()


class _SilentHandler(BaseHTTPRequestHandler):
    """Answers /v1/messages with headers and then no events at all."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if _STATE.mode == "think_then_silent":
            for chunk in _thinking_prefix(body.get("model", _MODEL)):
                self.wfile.write(chunk)
            self.wfile.flush()
        _STATE.serving.set()
        _STATE.stop.wait(timeout=120.0)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def silent_server() -> Generator[str]:
    """A local Anthropic endpoint that accepts a request and goes quiet."""
    _STATE.reset()
    server = _DaemonThreadingHTTPServer(("127.0.0.1", 0), _SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    _STATE.stop.set()
    server.shutdown()


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    server_url: str,
    token_callback: Callable[[str], None] | None = None,
    thinking_callback: Callable[[bool], None] | None = None,
) -> AnthropicModel:
    """Build a model through the real ``initialize()`` code path."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server_url)
    model = AnthropicModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )
    model.initialize("Use one browser profile in web_use_tool.py.")
    return model


def _call_with_stop(
    model: AnthropicModel,
    stop_event: threading.Event,
    stop_after: float = 0.5,
) -> tuple[BaseException | None, float]:
    """Run one model turn, requesting a stop while the stream is silent.

    Binds *stop_event* the way ``task_runner`` binds it — by assigning
    ``JsonPrinter._thread_local.stop_event`` on the calling thread — then
    presses Stop from another thread once the server has the request.

    Args:
        model: The model whose turn to run.
        stop_event: The event the simulated Stop click sets.
        stop_after: Seconds to wait after the server is serving before
            setting the event.

    Returns:
        The exception the call raised (``None`` when it returned) and the
        seconds elapsed between the stop request and the call unwinding.
    """
    printer = JsonPrinter()
    outcome: dict[str, Any] = {}
    stopped_at: dict[str, float] = {}

    def target() -> None:
        printer._thread_local.stop_event = stop_event
        try:
            outcome["result"] = model.generate_and_process_with_tools(
                {}, tools_schema=[_FINISH_TOOL],
            )
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc
        finally:
            printer._thread_local.stop_event = None

    def presser() -> None:
        _STATE.serving.wait(timeout=_DEADLINE)
        time.sleep(stop_after)
        stopped_at["t"] = time.monotonic()
        stop_event.set()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    threading.Thread(target=presser, daemon=True).start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(
            f"model call still running {_DEADLINE}s after Stop — the stop "
            f"event is not reaching the stream (pre-fix behaviour)"
        )
    elapsed = time.monotonic() - stopped_at.get("t", time.monotonic())
    return outcome.get("error"), elapsed


class TestStopAbortsSilentStream:
    """A Stop during a quiet request must unblock the agent at once."""

    def test_stop_unblocks_silent_stream_quickly(
        self, monkeypatch: pytest.MonkeyPatch, silent_server: str,
    ) -> None:
        """The production wedge: headers arrive, then nothing.

        Before the fix this call sat for the full 120s stall window
        ignoring the stop event, which is what made the button look
        dead for three minutes.
        """
        model = _make_model(monkeypatch, silent_server)
        error, elapsed = _call_with_stop(model, threading.Event())
        assert isinstance(error, KeyboardInterrupt), f"got {error!r}"
        assert elapsed < 5.0, f"stop took {elapsed:.1f}s to land"

    def test_stop_is_not_reported_as_a_retryable_stall(
        self, monkeypatch: pytest.MonkeyPatch, silent_server: str,
    ) -> None:
        """A user stop must not look like a transient provider stall.

        ``KISSAgent._run_agentic_loop`` retries ``TimeoutError``, so
        reusing the stall error here would re-ask the model and keep the
        stopped task alive.
        """
        model = _make_model(monkeypatch, silent_server)
        error, _elapsed = _call_with_stop(model, threading.Event())
        assert not isinstance(error, TimeoutError)
        assert "stop" in str(error).lower()

    def test_stop_mid_thinking_closes_the_thinking_bracket(
        self, monkeypatch: pytest.MonkeyPatch, silent_server: str,
    ) -> None:
        """A stop after thinking started must close the bracket.

        Otherwise the UI keeps rendering everything as "thinking" after
        the task has already ended.
        """
        _STATE.mode = "think_then_silent"
        thinking: list[bool] = []
        tokens: list[str] = []
        model = _make_model(
            monkeypatch,
            silent_server,
            token_callback=tokens.append,
            thinking_callback=thinking.append,
        )
        error, _elapsed = _call_with_stop(model, threading.Event())
        assert isinstance(error, KeyboardInterrupt), f"got {error!r}"
        assert "Let me think…" in "".join(tokens)
        assert thinking == [True, False]

    def test_unstopped_silent_stream_still_stalls_out(
        self, monkeypatch: pytest.MonkeyPatch, silent_server: str,
    ) -> None:
        """Watching the stop event must not disarm the stall watchdog."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", silent_server)
        model = AnthropicModel(
            _MODEL,
            api_key="test-key",
            model_config={"stream_stall_timeout": 1.5},
        )
        model.initialize("Use one browser profile in web_use_tool.py.")
        printer = JsonPrinter()
        printer._thread_local.stop_event = threading.Event()
        try:
            with pytest.raises(TimeoutError, match="stalled"):
                model.generate_and_process_with_tools(
                    {}, tools_schema=[_FINISH_TOOL],
                )
        finally:
            printer._thread_local.stop_event = None


def _openai_reasoning_chunk() -> bytes:
    """One Chat Completions chunk carrying a reasoning delta."""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-stop-under-test",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me think…",
                },
                "finish_reason": None,
            },
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _OpenAISilentHandler(BaseHTTPRequestHandler):
    """Sends one reasoning chunk, then never speaks again."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_openai_reasoning_chunk())
        self.wfile.flush()
        _STATE.serving.set()
        _STATE.stop.wait(timeout=120.0)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def openai_silent_server() -> Generator[str]:
    """A local OpenAI endpoint that starts thinking and then goes quiet."""
    _STATE.reset()
    server = _DaemonThreadingHTTPServer(("127.0.0.1", 0), _OpenAISilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    _STATE.stop.set()
    server.shutdown()


class TestStopAbortsSilentOpenAIStream:
    """The OpenAI adapters must honour Stop too, not only Anthropic.

    ``generate_and_process_with_tools`` is the call Sorcar's agentic loop
    makes, and its client is built with a 1800-second timeout — half an
    hour of a dead-looking Stop button for anyone on an OpenAI-compatible
    provider.
    """

    def _run(self, server_url: str) -> tuple[BaseException | None, list[bool]]:
        """Stop a wedged tool-calling turn, returning its error/callbacks."""
        from kiss.core.models.openai_compatible_model import (
            OpenAICompatibleModel,
        )

        thinking: list[bool] = []
        model = OpenAICompatibleModel(
            "gpt-stop-under-test",
            base_url=server_url,
            api_key="test-key",
            token_callback=lambda _t: None,
            thinking_callback=thinking.append,
        )
        model.initialize("Use one browser profile in web_use_tool.py.")

        def dummy() -> str:
            """Dummy tool."""
            return "ok"

        printer = JsonPrinter()
        stop_event = threading.Event()
        outcome: dict[str, Any] = {}

        def target() -> None:
            printer._thread_local.stop_event = stop_event
            try:
                model.generate_and_process_with_tools({"dummy": dummy})
            except BaseException as exc:  # noqa: BLE001 — reported below
                outcome["error"] = exc
            finally:
                printer._thread_local.stop_event = None

        def presser() -> None:
            _STATE.serving.wait(timeout=_DEADLINE)
            time.sleep(0.5)
            stop_event.set()

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        threading.Thread(target=presser, daemon=True).start()
        worker.join(_DEADLINE)
        if worker.is_alive():
            pytest.fail(
                f"tool-calling turn still running {_DEADLINE}s after Stop — "
                f"the 1800s client timeout is holding the agent"
            )
        return outcome.get("error"), thinking

    def test_stop_unblocks_the_tool_calling_stream(
        self, openai_silent_server: str,
    ) -> None:
        """The agentic path must abort, not wait out the client timeout."""
        error, _thinking = self._run(openai_silent_server)
        assert isinstance(error, KeyboardInterrupt), f"got {error!r}"

    def test_stop_closes_the_thinking_bracket(
        self, openai_silent_server: str,
    ) -> None:
        """A stop mid-thinking must not leave the UI stuck in "thinking"."""
        _error, thinking = self._run(openai_silent_server)
        assert thinking == [True, False], thinking
