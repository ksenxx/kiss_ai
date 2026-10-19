# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a stall after the usage chunk must not discard the answer.

Finding B2/F2.  In ``OpenAICompatibleModel._stream_chat_completion`` the
``for chunk in events`` loop only ended when the SSE body ended.  With
``stream_options.include_usage`` the server sends the finish chunk, then
the usage chunk, then ``data: [DONE]``.  A connection that went silent
between the usage chunk and ``[DONE]`` (the answer and its billing were
already complete) hit the stall watchdog, which raised ``TimeoutError``
and made the agent re-send the whole request: the finished turn was
billed twice.  ``OpenAICompatibleModel2`` never had this problem because
it breaks on ``response.completed``.

The scripted server below streams a complete turn and then holds the
connection open with SSE keep-alive comments (no events) for longer than
the stall timeout.  Pre-fix: ``TimeoutError`` after ``stall`` seconds.
Post-fix: the answer and its usage come back immediately.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-usage-tail-under-test"
_STALL = 1.5


def _chunk(choices: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> bytes:
    """Build one Chat Completions SSE chunk."""
    payload: dict[str, Any] = {
        "id": "chatcmpl-tail",
        "object": "chat.completion.chunk",
        "model": _MODEL,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return chat_chunk(payload)


class _HoldAfterUsage:
    """Stream a complete turn, then keep the connection silent (no ``[DONE]``)."""

    def __init__(self) -> None:
        """Create the release gate the test opens on exit."""
        self.release = threading.Event()

    def __call__(self, request: Request) -> Reply:
        """Answer with content, finish, usage, then hold the socket."""
        return Reply(
            sse_chunks=[
                _chunk([{"index": 0, "delta": {"role": "assistant", "content": "The "}}]),
                _chunk([{"index": 0, "delta": {"content": "answer."}}]),
                _chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
                _chunk(
                    [],
                    usage={
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                ),
            ],
            hold=self.release,
        )


class TestStallAfterUsageChunk:
    """The turn is complete once ``finish_reason`` and usage have arrived."""

    def test_answer_and_usage_survive_a_silent_tail(self) -> None:
        """No ``TimeoutError``, one request, usage billed from the usage chunk."""
        responder = _HoldAfterUsage()
        tokens: list[str] = []
        with ScriptedOpenAIServer(responder) as server:
            try:
                model = OpenAICompatibleModel(
                    _MODEL,
                    base_url=server.base_url,
                    api_key="test-key",
                    model_config={"stream_stall_timeout": _STALL},
                    token_callback=tokens.append,
                )
                model.initialize("question")
                started = time.monotonic()
                text, response = model.generate()
                elapsed = time.monotonic() - started
            finally:
                responder.release.set()
            request_count = len(server.requests)
        assert text == "The answer."
        assert tokens == ["The ", "answer."]
        assert elapsed < _STALL, f"waited {elapsed:.2f}s for a finished turn"
        assert request_count == 1
        counts = model.extract_input_output_token_counts_from_response(response)
        assert counts == (7, 7, 4, 0)

    def test_usage_on_the_finish_chunk_also_ends_the_turn(self) -> None:
        """Providers that put usage on the finish chunk itself get the same break."""
        release = threading.Event()

        def responder(_request: Request) -> Reply:
            """Content, then one chunk carrying both ``finish_reason`` and usage."""
            return Reply(
                sse_chunks=[
                    _chunk([{"index": 0, "delta": {"role": "assistant", "content": "x"}}]),
                    _chunk(
                        [{"index": 0, "delta": {"content": "y"}, "finish_reason": "stop"}],
                        usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                    ),
                ],
                hold=release,
            )

        with ScriptedOpenAIServer(responder) as server:
            try:
                model = OpenAICompatibleModel(
                    _MODEL,
                    base_url=server.base_url,
                    api_key="test-key",
                    model_config={"stream_stall_timeout": _STALL},
                    token_callback=lambda _t: None,
                )
                model.initialize("q")
                started = time.monotonic()
                text, response = model.generate()
                elapsed = time.monotonic() - started
            finally:
                release.set()
        assert text == "xy"
        assert elapsed < _STALL
        assert model.extract_input_output_token_counts_from_response(response)[:2] == (2, 2)
