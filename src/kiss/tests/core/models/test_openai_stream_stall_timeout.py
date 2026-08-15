# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: OpenAI-compatible streams must time out when they stall.

Audit findings 01/F1 and 02/R1.  ``model_config["stream_stall_timeout"]``
was accepted, documented and popped off the request kwargs by both
OpenAI-compatible transports — and read by neither.  Both called
``stop_aware_events`` without ``stall_timeout``, so the watchdog's stall
branch was disabled, and ``stop_aware_events`` never read
``watchdog.stalled`` either: even if a stall *had* been detected, the
aborted socket ends the iterator at EOF and the adapter would have
returned its partial text as a successful completion.

A wedged gateway that keeps the connection alive without emitting events
therefore held the agent for the client's full 1800-second timeout, while
the identical wedge on ``anthropic_model`` is aborted after
``stream_stall_timeout`` seconds with a retryable ``TimeoutError``.

No mocks: a real ``ThreadingHTTPServer`` streams genuine SSE bytes to the
real OpenAI SDK and then goes quiet, exactly like the wedged provider.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_STALL_TIMEOUT = 2.0
_DEADLINE = 20.0
_MODEL = "gpt-stall-under-test"


def _tool_schema() -> list[dict[str, Any]]:
    """Return a one-tool Chat-Completions schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the task",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
            },
        }
    ]


class _StallPolicy:
    """Serves a couple of deltas and then holds the connection open."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.release = threading.Event()

    def __call__(self, request: Request) -> Reply:
        """Return the scripted partial stream, then wedge."""
        return Reply(sse_chunks=self.chunks, hold=self.release)


_CHAT_CHUNKS = [
    chat_chunk(
        {
            "id": "chatcmpl-stall",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "partial "},
                    "finish_reason": None,
                }
            ],
        }
    ),
    chat_chunk(
        {
            "id": "chatcmpl-stall",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [
                {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
            ],
        }
    ),
]

_RESPONSES_CHUNKS = [
    responses_event(
        "response.created",
        {"response": {"id": "resp_stall", "status": "in_progress", "output": []}},
    ),
    responses_event(
        "response.output_text.delta",
        {"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "partial"},
    ),
]


@pytest.fixture
def chat_stall_server() -> Generator[tuple[ScriptedOpenAIServer, _StallPolicy]]:
    """A Chat Completions endpoint that streams two deltas and then wedges."""
    policy = _StallPolicy(_CHAT_CHUNKS)
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy
        policy.release.set()


@pytest.fixture
def responses_stall_server() -> Generator[tuple[ScriptedOpenAIServer, _StallPolicy]]:
    """A Responses endpoint that streams one delta and then wedges."""
    policy = _StallPolicy(_RESPONSES_CHUNKS)
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy
        policy.release.set()


def _run_with_deadline(call: Any) -> tuple[BaseException | None, float]:
    """Run *call* on a worker thread bounded by the test deadline.

    Args:
        call: A zero-argument callable performing the model turn.

    Returns:
        ``(exception_or_None, elapsed_seconds)``.  Fails the test when
        the call is still running at the deadline, which is the pre-fix
        behaviour (it would sit there for the client's 1800s timeout).
    """
    outcome: dict[str, BaseException] = {}
    started = time.monotonic()

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(
            f"the wedged stream was still being read {_DEADLINE}s later — "
            f"stream_stall_timeout={_STALL_TIMEOUT}s was ignored"
        )
    return outcome.get("error"), time.monotonic() - started


def _assert_stalled(error: BaseException | None, elapsed: float) -> None:
    """Assert the turn failed as a retryable stall, promptly.

    Args:
        error: The exception the turn raised, if any.
        elapsed: Seconds the turn took.
    """
    assert isinstance(error, TimeoutError), f"got {error!r}"
    assert "stream_stall_timeout" in str(error)
    assert elapsed < _DEADLINE / 2, f"stall took {elapsed:.1f}s to surface"


class TestChatCompletionsStallTimeout:
    """v1 (Chat Completions) must honour ``stream_stall_timeout``."""

    def test_toolless_stream_stalls_out(
        self, chat_stall_server: tuple[ScriptedOpenAIServer, _StallPolicy]
    ) -> None:
        """``generate()`` must raise instead of returning partial text."""
        server, _policy = chat_stall_server
        tokens: list[str] = []
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=tokens.append,
        )
        model.initialize("Say something.")
        error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)
        assert "".join(tokens) == "partial answer"

    def test_tool_calling_stream_stalls_out(
        self, chat_stall_server: tuple[ScriptedOpenAIServer, _StallPolicy]
    ) -> None:
        """The agentic path (tools attached) must stall out too."""
        server, _policy = chat_stall_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Use the finish tool.")
        error, elapsed = _run_with_deadline(
            lambda: model.generate_and_process_with_tools(
                {}, tools_schema=_tool_schema()
            )
        )
        _assert_stalled(error, elapsed)


class TestResponsesStallTimeout:
    """v2 (Responses API) must honour ``stream_stall_timeout`` as well."""

    def test_responses_stream_stalls_out(
        self, responses_stall_server: tuple[ScriptedOpenAIServer, _StallPolicy]
    ) -> None:
        """A wedged ``/v1/responses`` stream must raise, not hang."""
        server, _policy = responses_stall_server
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)


class _SlowConsumer:
    """A token callback that blocks longer than the stall timeout.

    The whole SSE body is already buffered by the transport, so the
    watchdog stalls (no event reaches it while the consumer is busy) and
    aborts a stream that then still finishes cleanly from the buffer.
    That is the path where an aborted stall reaches the caller as EOF
    rather than as a transport error — the case that used to be reported
    as a silently truncated success.
    """

    def __init__(self, delay: float) -> None:
        """Remember how long the first token blocks for.

        Args:
            delay: Seconds the first token callback sleeps.
        """
        self.delay = delay
        self.tokens: list[str] = []

    def __call__(self, token: str) -> None:
        """Record the token, blocking once on the first one."""
        if not self.tokens:
            time.sleep(self.delay)
        self.tokens.append(token)


def _buffered_stream(request: Request) -> Reply:
    """Answer with the whole stream in one complete, buffered body."""
    return Reply(
        sse_chunks=[
            *_CHAT_CHUNKS,
            chat_chunk(
                {
                    "id": "chatcmpl-stall",
                    "object": "chat.completion.chunk",
                    "model": _MODEL,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


def _crashing_stream(request: Request) -> Reply:
    """Answer with a body that stops arriving halfway through."""
    return Reply(sse_chunks=[*_CHAT_CHUNKS, b"data: [DONE]\n\n"], truncate_after=1)


class TestStallIsReportedEvenWithoutATransportError:
    """A stalled stream that ends at EOF must still raise."""

    def test_slow_consumer_stall_is_not_a_silent_truncation(self) -> None:
        """The partial text must not be returned as a completion."""
        consumer = _SlowConsumer(_STALL_TIMEOUT * 2)
        with ScriptedOpenAIServer(_buffered_stream) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
                token_callback=consumer,
            )
            model.initialize("Say something.")
            error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)


class TestUnrelatedStreamErrorsPropagate:
    """Only stops and stalls are translated; everything else is raised."""

    def test_provider_crash_is_not_disguised(self) -> None:
        """A truncated body must surface as the transport's own error."""
        with ScriptedOpenAIServer(_crashing_stream) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": 120.0},
                token_callback=lambda _t: None,
            )
            model.initialize("Say something.")
            error, _elapsed = _run_with_deadline(model.generate)
        assert error is not None
        assert not isinstance(error, TimeoutError | KeyboardInterrupt), (
            f"a provider crash was disguised as a stop or a stall: {error!r}"
        )


class TestStallTimeoutIsConfigurable:
    """The default must stay the shared 180s, and be overridable."""

    def test_default_and_override(self) -> None:
        """An unset key uses the module default; a set key wins."""
        from kiss.core.models.stream_abort import DEFAULT_STREAM_STALL_TIMEOUT

        default_model = OpenAICompatibleModel(
            _MODEL, base_url="http://127.0.0.1:1/v1", api_key="k",
        )
        assert default_model._stream_stall_timeout == DEFAULT_STREAM_STALL_TIMEOUT
        tuned = OpenAICompatibleModel2(
            _MODEL,
            base_url="http://127.0.0.1:1/v1",
            api_key="k",
            model_config={"stream_stall_timeout": 5},
        )
        assert tuned._stream_stall_timeout == 5.0
