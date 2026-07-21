# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a stalled Anthropic stream must not hang the agent.

Bug reproduction ("task stuck in thinking", production failure at
2026-07-21 10:08 in ``~/.kiss/sorcar.db``, task
``f554c68446fa42af89c2fd3c7cc14f63``, model = ``claude-fable-5``):

* Step 1 of the task completed normally (Read ./SORCAR.md).  Step 2's
  provider request was issued at 10:08:16.710 ("Step 2/100 start" in
  ``~/.kiss/kiss-web-stderr.log``) and then NOTHING happened: no stream
  event, no thinking delta, no log line, no error — for 5.5 minutes,
  until the user stopped the task by hand.  Concurrent tasks in the same
  process kept streaming normally, so this was a per-request hang.
* Root cause: ``AnthropicModel.initialize`` built ``Anthropic(api_key=…)``
  with the SDK defaults — ``httpx.Timeout(connect=5, read=600)`` and 2
  silent retries — and ``_create_message`` iterated the stream with no
  stall detection.  A request that the API accepts but never answers (or
  a stream that dies mid-turn) therefore blocks the agent's step loop
  for 10–30 minutes with zero output.  ``KISSAgent``'s retry/fallback
  machinery only reacts to raised exceptions, so it never fired.

Fix under test (including the cross-model-review findings):

1. ``AnthropicModel.initialize`` builds the client with a bounded
   no-bytes-flowing timeout (``httpx.Timeout(stream_stall_timeout,
   connect=10)``; default 180s, overridable via
   ``model_config["stream_stall_timeout"]``) and ``max_retries=1`` so
   the SDK's silent pre-header retries are bounded too.
2. ``_create_message`` converts both ``httpx.TimeoutException``
   (mid-stream byte stall) and ``anthropic.APITimeoutError`` (headers
   never arrive; SDK raises it after its own bounded retries) into a
   clear, retryable ``TimeoutError``.
3. An event-level ``_StreamStallWatchdog`` closes the response when no
   SSE *event* is yielded within the stall window — catching wedged
   requests that keep the connection alive with ``ping`` events, which
   the SDK filters out before yielding (so they reset the byte-level
   timeout while the agent still sees nothing).
4. A stall that strikes after a thinking block started closes the
   thinking bracket (``thinking_callback(False)``) so the UI does not
   stay in "thinking" mode across the retry.
5. ``KISSAgent._run_agentic_loop`` treats the ``TimeoutError`` like any
   retryable model error: it retries, and after ``MAX_CONSECUTIVE_ERRORS``
   raises a visible ``KISSError`` instead of hanging forever.

Test strategy (no mocks, patches of code under test, or fakes of the
SDK): a local ``ThreadingHTTPServer`` speaks the real Anthropic SSE wire
format to the real ``anthropic`` SDK client.  Stall modes reproduce the
distinct production-relevant hangs: ``silent`` (200 + SSE headers, then
zero bytes), ``no_headers`` (request accepted, response never starts),
``ping_only`` (only keep-alive pings forever), and ``think_then_ping``
(a thinking block starts, then only pings).  The client is routed to the
local server via the SDK's own ``ANTHROPIC_BASE_URL`` environment
variable so the fixed ``initialize()`` code path (client construction
incl. timeout/retries) is exercised verbatim.  Every potentially-hanging
call runs on a daemon worker thread with a hard deadline, so on pre-fix
code the tests FAIL fast instead of hanging CI.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import MODEL_INFO, ModelInfo

_MODEL = "claude-stall-under-test"
#: Tight stall timeout for the tests (seconds).
_STALL_TIMEOUT = 1.5
#: Hard deadline for calls that hung (or took 600s+) on the pre-fix code.
#: Far below the SDK-default 600s read timeout, yet generous enough for
#: slow CI machines (the worst fixed path is the no-headers one:
#: 2 SDK attempts x stall + backoff).
_FAST_FAIL_BUDGET = 30.0

_OPENAI_FINISH_TOOL = {
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


def _message_start(model_name: str) -> bytes:
    return _sse(
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
    )


def _thinking_block_prefix(model_name: str) -> list[bytes]:
    """A turn that STARTS thinking (start + one delta) and never finishes."""
    return [
        _message_start(model_name),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think…"},
            },
        ),
    ]


def _finish_tool_stream(model_name: str) -> list[bytes]:
    """A normal agentic turn: one ``finish`` tool_use block."""
    return [
        _message_start(model_name),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_finish",
                    "name": "finish",
                    "input": {},
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"result": "recovered"}',
                },
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


class _StallState:
    """Shared, thread-safe request log + per-test stall policy.

    Attributes:
        mode: The stall behavior for stalled requests — ``"silent"``
            (200 + SSE headers, then zero bytes: the production
            accepted-but-dead request), ``"no_headers"`` (request read,
            response never starts), ``"ping_only"`` (SSE keep-alive
            ``ping`` events forever, no message events), or
            ``"think_then_ping"`` (a thinking block starts, then only
            pings).
        stall_first_n: Requests 1..N stall; later requests answer
            normally.  Use a huge value to stall every request.
        request_count: Number of POSTs received so far.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Restore the default policy (silent-stall everything)."""
        self.mode = "silent"
        self.stall_first_n = 10**9
        self.request_count = 0
        self.stop = threading.Event()

    def next_request_stalls(self) -> bool:
        with self.lock:
            self.request_count += 1
            return self.request_count <= self.stall_first_n


_STATE = _StallState()


class _StallHandler(BaseHTTPRequestHandler):
    """Accepts /v1/messages and stalls per the shared ``_StallState``."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model_name = body.get("model", _MODEL)
        if not _STATE.next_request_stalls():
            self._send_sse_headers()
            self._write_chunks(_finish_tool_stream(model_name))
            return
        if _STATE.mode == "no_headers":
            # Request accepted (and fully read), response never starts.
            _STATE.stop.wait(timeout=120.0)
            return
        self._send_sse_headers()
        if _STATE.mode == "silent":
            # Headers sent, then not a single byte: the production hang.
            _STATE.stop.wait(timeout=120.0)
            return
        if _STATE.mode == "think_then_ping":
            self._write_chunks(_thinking_block_prefix(model_name))
        # ping_only / think_then_ping: keep-alive pings keep BYTES flowing
        # (defeating a pure read-timeout) while the SDK filters the events
        # out, so the agent still sees nothing.
        while not _STATE.stop.wait(timeout=0.2):
            try:
                self._write_chunks([_sse("ping", {"type": "ping"})])
            except OSError:
                return  # client aborted (watchdog closed the response)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

    def _write_chunks(self, chunks: list[bytes]) -> None:
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def stall_server() -> Generator[str]:
    _STATE.reset()
    server = _DaemonThreadingHTTPServer(("127.0.0.1", 0), _StallHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    _STATE.stop.set()
    server.shutdown()


def _run_bounded(fn: Callable[[], Any], deadline: float = _FAST_FAIL_BUDGET) -> Any:
    """Run *fn* on a daemon thread with a hard deadline.

    Bounds the damage when the code under test regresses to the pre-fix
    hang: the test FAILS after *deadline* seconds instead of blocking the
    suite for the 600s SDK default (or forever).

    Args:
        fn: The zero-argument callable to execute.
        deadline: Seconds to wait before declaring a hang.

    Returns:
        The exception *fn* raised, or ``("ok", result)`` when it returned.
    """
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    start = time.monotonic()
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        pytest.fail(
            f"call still running after {deadline}s — the stall timeout is "
            f"not being enforced (pre-fix hang behavior)"
        )
    elapsed = time.monotonic() - start
    assert elapsed < deadline
    if "error" in outcome:
        return outcome["error"]
    return ("ok", outcome["result"])


def _assert_stall_timeout_error(outcome: Any) -> None:
    assert isinstance(outcome, TimeoutError), f"expected TimeoutError, got {outcome!r}"
    msg = str(outcome)
    assert "stalled" in msg
    assert _MODEL in msg
    assert "stream_stall_timeout" in msg


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    server_url: str,
    token_callback: Callable[[str], None] | None = None,
    thinking_callback: Callable[[bool], None] | None = None,
) -> AnthropicModel:
    """Build an AnthropicModel through its REAL initialize() code path.

    ``ANTHROPIC_BASE_URL`` is the SDK's own routing knob, so the fixed
    client construction (timeout and retry bounds included) is exercised
    verbatim.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server_url)
    m = AnthropicModel(
        _MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )
    m.initialize("Update ./README.md based on the latest code in the project.")
    return m


def _register_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the synthetic ``claude-*`` model so the model() factory
    routes it through the real ``AnthropicModel`` (no fallback, so the
    retry path — not the fallback path — is what recovers)."""
    monkeypatch.setitem(
        MODEL_INFO,
        _MODEL,
        ModelInfo(
            context_length=128_000,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            is_function_calling_supported=True,
            is_embedding_supported=False,
            is_generation_supported=True,
            fallback=None,
            extended_thinking=False,
        ),
    )


def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the model() factory a non-empty Anthropic key on CI machines."""
    from kiss.core import config as config_module

    if not getattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", ""):
        monkeypatch.setattr(
            config_module.DEFAULT_CONFIG,
            "ANTHROPIC_API_KEY",
            "test-key",
            raising=False,
        )


class TestAdapterAbortsStalledStream:
    """``AnthropicModel`` must abort a stalled stream, not hang."""

    def test_tools_turn_times_out_fast_with_clear_error(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """The agentic (tools) path — the one that hung in production —
        must raise an actionable ``TimeoutError`` within seconds."""
        m = _make_model(monkeypatch, stall_server)
        outcome = _run_bounded(
            lambda: m.generate_and_process_with_tools(
                {}, tools_schema=[_OPENAI_FINISH_TOOL]
            )
        )
        _assert_stall_timeout_error(outcome)

    def test_plain_generate_times_out_fast(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """The no-tools path goes through the same stream and must abort too."""
        m = _make_model(monkeypatch, stall_server)
        _assert_stall_timeout_error(_run_bounded(m.generate))

    def test_headers_never_arrive_times_out_fast(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """A request the server accepts but never answers (no response
        headers at all) surfaces as ``anthropic.APITimeoutError`` after the
        SDK's now-bounded retries; it must also become the clear
        ``TimeoutError`` — review finding: the first fix only caught
        ``httpx.TimeoutException`` and left the SDK's 2 silent retries."""
        _STATE.mode = "no_headers"
        m = _make_model(monkeypatch, stall_server)
        outcome = _run_bounded(
            lambda: m.generate_and_process_with_tools(
                {}, tools_schema=[_OPENAI_FINISH_TOOL]
            )
        )
        _assert_stall_timeout_error(outcome)
        # max_retries=1 → exactly 2 attempts, not the SDK-default 3.
        assert _STATE.request_count == 2

    def test_ping_only_stream_times_out_fast(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """Keep-alive pings keep bytes flowing (so the httpx read timeout
        never fires) while the SDK filters the events out — the agent sees
        nothing.  The event-level watchdog must abort — review finding:
        byte-level timeout alone cannot catch this wedge."""
        _STATE.mode = "ping_only"
        m = _make_model(monkeypatch, stall_server)
        outcome = _run_bounded(
            lambda: m.generate_and_process_with_tools(
                {}, tools_schema=[_OPENAI_FINISH_TOOL]
            )
        )
        _assert_stall_timeout_error(outcome)

    def test_stall_mid_thinking_closes_thinking_bracket(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """A stall after a thinking block started must emit the closing
        ``thinking_callback(False)`` — review finding: otherwise the
        printer/UI renders everything after the retry as "thinking"
        forever (the visible symptom of the original bug)."""
        _STATE.mode = "think_then_ping"
        thinking_events: list[bool] = []
        tokens: list[str] = []
        m = _make_model(
            monkeypatch,
            stall_server,
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        outcome = _run_bounded(
            lambda: m.generate_and_process_with_tools(
                {}, tools_schema=[_OPENAI_FINISH_TOOL]
            )
        )
        _assert_stall_timeout_error(outcome)
        assert "Let me think…" in "".join(tokens)
        assert thinking_events == [True, False]

    def test_stalled_turn_is_not_appended_to_conversation(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """An aborted turn must leave the conversation unchanged so the
        retry replays identical history."""
        m = _make_model(monkeypatch, stall_server)
        before = [dict(msg) for msg in m.conversation]
        outcome = _run_bounded(
            lambda: m.generate_and_process_with_tools(
                {}, tools_schema=[_OPENAI_FINISH_TOOL]
            )
        )
        _assert_stall_timeout_error(outcome)
        assert m.conversation == before


class TestAgentSurvivesStalledStream:
    """KISSAgent must retry after a stall and never get stuck thinking."""

    def test_agent_recovers_when_stream_stalls_once(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """One dead request (the production scenario), then a healthy API:
        the agent must retry and finish instead of hanging forever."""
        _register_model(monkeypatch)
        _ensure_api_key(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", stall_server)
        _STATE.stall_first_n = 1
        agent = KISSAgent("test-stall-recovery")
        outcome = _run_bounded(
            lambda: agent.run(
                model_name=_MODEL,
                prompt_template="Update ./README.md based on the latest code.",
                max_steps=5,
                max_budget=1.0,
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
                verbose=False,
            ),
            deadline=2 * _FAST_FAIL_BUDGET,
        )
        assert outcome == ("ok", "recovered")
        assert _STATE.request_count == 2

    def test_agent_fails_visibly_when_api_stays_dead(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """Every request stalls: the agent must surface a KISSError after
        its bounded retries — a visible failure the orchestrator can
        report — rather than the silent infinite "thinking" of the bug."""
        _register_model(monkeypatch)
        _ensure_api_key(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", stall_server)
        agent = KISSAgent("test-stall-hard-failure")
        outcome = _run_bounded(
            lambda: agent.run(
                model_name=_MODEL,
                prompt_template="Update ./README.md based on the latest code.",
                max_steps=5,
                max_budget=1.0,
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
                verbose=False,
            ),
            deadline=4 * _FAST_FAIL_BUDGET,
        )
        assert isinstance(outcome, KISSError)
        msg = str(outcome)
        assert "consecutive errors" in msg
        assert "stalled" in msg
