# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: silence after ``finish_reason`` but before usage is a stall.

``OpenAICompatibleModel._stream_chat_completion`` keeps a complete answer
when the connection breaks after ``finish_reason`` (only the usage tail
was lost), but a *timeout* after ``finish_reason`` is still reported as
the retryable stall it is, never swallowed as a kept answer.  Two clocks
bound that silence and both must agree: the stream watchdog (event-level
stall) and the per-request ``httpx`` read timeout, which the SDK raises
through the event iterator as ``httpx.ReadTimeout`` — a subclass of the
``httpx.HTTPError`` that the connection-lost handler catches.  Which clock
fires first depends on the watchdog's poll tick, so this test pins the
outcome both paths must produce rather than one path.

No mocks: a real ``ThreadingHTTPServer`` streams the ``finish_reason``
chunk to the real OpenAI SDK and then goes silent (no keep-alives).
"""

from __future__ import annotations

import threading
import time

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-tail-stall-under-test"
_STALL = 1.0


def _finish_then_silence(gate: threading.Event) -> list[bytes]:
    """Return the chunks: the complete answer with ``finish_reason``, then usage behind *gate*."""
    return [
        chat_chunk(
            {
                "id": "chatcmpl-tail",
                "object": "chat.completion.chunk",
                "model": _MODEL,
                "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}],
            }
        ),
        chat_chunk(
            {
                "id": "chatcmpl-tail",
                "object": "chat.completion.chunk",
                "model": _MODEL,
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ),
    ]


class TestTailStallAfterFinishReason:
    """A silent provider after ``finish_reason`` is a stall, not a kept answer."""

    def test_silence_before_usage_is_reported_as_a_stall(self) -> None:
        """``generate()`` raises the retryable stall error after one request."""
        gate = threading.Event()
        chunks = _finish_then_silence(gate)

        def respond(_request: Request) -> Reply:
            return Reply(sse_chunks=chunks, chunk_gate=gate)

        tokens: list[str] = []
        try:
            with ScriptedOpenAIServer(respond) as server:
                model = OpenAICompatibleModel(
                    _MODEL,
                    base_url=server.base_url,
                    api_key="test-key",
                    model_config={"stream_stall_timeout": _STALL},
                    token_callback=tokens.append,
                )
                model.initialize("Say hello.")
                started = time.monotonic()
                with pytest.raises(TimeoutError, match="stalled"):
                    model.generate()
                assert time.monotonic() - started < 10 * _STALL
                assert tokens == ["hello"]
                assert len(server.requests) == 1
        finally:
            gate.set()
