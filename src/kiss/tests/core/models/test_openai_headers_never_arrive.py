# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a streaming request whose headers never arrive must fail.

Audit B1 (2026-10-01).  ``stop_aware_events`` arms its watchdog only once
``create(stream=True)`` has returned the response headers.  Before that,
the only clock was the SDK client's scalar 1800 s timeout, so a gateway
that accepted the TCP connection and then queued the request forever
parked the agent in ``recv()`` for 30 minutes per attempt, deaf to Stop.

The streaming creates now carry a per-request
``httpx.Timeout(stream_stall_timeout, connect=10)``, and a timeout raised
by either clock is reported as the same retryable stall error.  Each test
below talks to a real loopback HTTP server that reads the request and
writes nothing at all until the test releases it.  Non-streaming calls
keep the long client default: the control test proves a silent-but-slow
non-streaming reply still succeeds past the stall timeout.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_STALL_TIMEOUT = 1.0
# Generous bound: the SDK retries a pre-headers timeout once (``_MAX_RETRIES
# = 1``) with a sub-second backoff, so two silent attempts plus overhead
# must still finish well inside this; the pre-fix behaviour was 1800 s.
_DEADLINE = 12.0
_MODEL = "gpt-silent-gateway-under-test"


class _SilentGateway:
    """Reads every request and answers nothing until ``release`` is set.

    Args:
        release_after: When set, the gate opens this many seconds after a
            request arrives, so the silence is measured from the server's
            side rather than from before the client started.
    """

    def __init__(self, release_after: float | None = None) -> None:
        self.release = threading.Event()
        self.release_after = release_after

    def __call__(self, request: Request) -> Reply:
        """Hold the status line back; reply normally once released."""
        if self.release_after is not None:
            threading.Timer(self.release_after, self.release.set).start()
        if request.path == "/v1/responses":
            return Reply(
                headers_gate=self.release,
                sse_chunks=[
                    responses_event(
                        "response.completed",
                        {
                            "response": {
                                "id": "resp_late",
                                "status": "completed",
                                "output": [],
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 0,
                                    "total_tokens": 1,
                                },
                            }
                        },
                    )
                ],
            )
        if request.body.get("stream"):
            return Reply(
                headers_gate=self.release,
                sse_chunks=[
                    chat_chunk(
                        {
                            "id": "chatcmpl-late",
                            "object": "chat.completion.chunk",
                            "model": _MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "late"},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )
                ],
            )
        return Reply(
            headers_gate=self.release,
            json_body={
                "id": "chatcmpl-late",
                "object": "chat.completion",
                "created": 0,
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "late but complete"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            },
        )


@pytest.fixture
def silent_server() -> Generator[tuple[ScriptedOpenAIServer, _SilentGateway]]:
    """An endpoint that accepts connections but sends no headers."""
    gateway = _SilentGateway()
    with ScriptedOpenAIServer(gateway) as server:
        yield server, gateway
        gateway.release.set()


def _run_with_deadline(call: Any) -> tuple[BaseException | None, float]:
    """Run *call* on a worker thread bounded by the test deadline.

    Args:
        call: A zero-argument callable performing the model turn.

    Returns:
        ``(exception_or_None, elapsed_seconds)``.  Fails the test when
        the call is still running at the deadline, which is the pre-fix
        behaviour (waiting for the client's 1800 s timeout).
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
            f"still waiting for response headers {_DEADLINE}s later — "
            f"stream_stall_timeout={_STALL_TIMEOUT}s did not bound the request"
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
    assert elapsed >= _STALL_TIMEOUT, f"gave up after only {elapsed:.2f}s"
    assert elapsed < _DEADLINE / 2, f"stall took {elapsed:.1f}s to surface"


def _generate_then_stop(model: Any) -> None:
    """Run ``model.generate()`` on a thread whose Stop fires a quarter-stall in.

    Args:
        model: The initialised model under test.
    """
    stop = threading.Event()
    stop_signal.set_thread_stop_event(stop)
    try:
        threading.Timer(_STALL_TIMEOUT / 4, stop.set).start()
        model.generate()
    finally:
        stop_signal.set_thread_stop_event(None)


def _assert_stopped(error: BaseException | None, elapsed: float) -> None:
    """Assert the turn unwound as a user stop, not a retryable stall.

    Args:
        error: The exception the turn raised, if any.
        elapsed: Seconds the turn took.
    """
    assert isinstance(error, KeyboardInterrupt), f"got {error!r}"
    assert elapsed < _DEADLINE / 2, f"stop took {elapsed:.1f}s to surface"


def _tool_schema() -> list[dict[str, Any]]:
    """Return one trivial tool so the adaptive (tools) path is exercised."""
    return [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


class TestChatCompletionsHeadersNeverArrive:
    """v1 (Chat Completions) streaming must not wait 1800 s for headers."""

    def test_toolless_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """``generate()`` with a token callback raises the stall error."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        error, elapsed = _run_with_deadline(model.generate)
        _assert_stalled(error, elapsed)
        assert server.requests, "the request never reached the server"

    def test_tool_calling_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """The agentic path (tools attached) is bounded the same way."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Use the finish tool.")
        error, elapsed = _run_with_deadline(
            lambda: model.generate_and_process_with_tools({}, tools_schema=_tool_schema())
        )
        _assert_stalled(error, elapsed)

    def test_stop_before_headers_is_a_stop_not_a_stall(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """A Stop pressed while headers are pending must not be retried.

        No watchdog exists before the headers arrive, so the per-request
        timeout is what ends the wait; it must surface as the user's
        ``KeyboardInterrupt`` rather than the retryable stall error.
        """
        server, _gateway = silent_server
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_stopped(*_run_with_deadline(lambda: _generate_then_stop(model)))
        assert server.requests, "the request never reached the server"

    def test_non_streaming_call_keeps_the_long_client_timeout(self) -> None:
        """Without a token callback the stall timeout must not apply.

        A long reasoning turn with no streaming legitimately sends nothing
        for minutes; the server here stays silent for 2.5x the stall
        timeout and the call must still complete with the full reply.
        """
        with ScriptedOpenAIServer(_SilentGateway(2.5 * _STALL_TIMEOUT)) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            )
            model.initialize("Think hard.")
            error, elapsed = _run_with_deadline(model.generate)
            assert error is None, f"non-streaming call failed: {error!r}"
            assert elapsed >= 2.5 * _STALL_TIMEOUT
            assert model.conversation[-1]["content"] == "late but complete"
            assert len(server.requests) == 1


class TestResponsesHeadersNeverArrive:
    """v2 (Responses API) streaming must not wait 1800 s for headers."""

    def test_responses_stream_times_out(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """A silent ``/v1/responses`` endpoint raises the stall error."""
        server, _gateway = silent_server
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
        assert server.requests, "the request never reached the server"

    def test_stop_before_headers_is_a_stop_not_a_stall(
        self, silent_server: tuple[ScriptedOpenAIServer, _SilentGateway]
    ) -> None:
        """v2 unwinds a pre-headers Stop as ``KeyboardInterrupt`` too."""
        server, _gateway = silent_server
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"stream_stall_timeout": _STALL_TIMEOUT},
            token_callback=lambda _t: None,
        )
        model.initialize("Say something.")
        _assert_stopped(*_run_with_deadline(lambda: _generate_then_stop(model)))
        assert server.requests, "the request never reached the server"
