# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a Stop that lands on the terminal Responses event is honoured.

``OpenAICompatibleModel2._consume_stream_events`` leaves its loop with
``break`` on ``response.completed``, which skips the post-loop stop check
inside ``stop_aware_events`` (closing the generator runs only its
cleanup).  A Stop requested after the last event was handed out — here,
from the token callback of the ``output_text.delta`` that precedes the
terminal event, with both events already in the client's buffer — was
therefore never read: the transport returned a completed response, the
tool calls went to the agent, and the stopped task kept running.  The
Chat Completions transport closes the same window after its own
``break``; this test pins the Responses transport to the same contract.

Both SSE events are written in ONE chunk so the terminal event is
parsed from the SDK's buffer with no socket read in between: the
watchdog (which polls) cannot claim the stop first and abort the
socket, which would mask the missing check by surfacing the stop
through the transport-error path instead.

No mocks: a real ``ThreadingHTTPServer`` answers the real OpenAI SDK
over real sockets, and the stop is the thread's real stop event.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "gpt-stop-on-terminal-under-test"
_USAGE = {
    "input_tokens": 7,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 3,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 10,
}


def _completed_payload() -> dict[str, Any]:
    """Return the ``response.completed`` payload carrying the billed usage."""
    return {
        "id": "resp_stop",
        "object": "response",
        "created_at": 0,
        "model": _MODEL,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": _USAGE,
    }


def _one_chunk_stream(_request: Request) -> Reply:
    """Answer with the text delta and the terminal event in a single write."""
    delta = responses_event(
        "response.output_text.delta",
        {
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "logprobs": [],
        },
    )
    completed = responses_event("response.completed", {"response": _completed_payload()})
    return Reply(sse_chunks=[delta + completed])


class TestStopOnTerminalEvent:
    """A Stop set while the terminal event is already buffered still stops the turn."""

    def test_stop_during_last_delta_unwinds_as_keyboard_interrupt(self) -> None:
        """The turn raises ``KeyboardInterrupt`` and keeps the billed usage."""
        stop = threading.Event()
        tokens: list[str] = []

        def on_token(token: str) -> None:
            tokens.append(token)
            stop.set()

        with ScriptedOpenAIServer(_one_chunk_stream) as server:
            model = OpenAICompatibleModel2(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                token_callback=on_token,
            )
            model.initialize("Say hello.")
            stop_signal.set_thread_stop_event(stop)
            try:
                with pytest.raises(KeyboardInterrupt):
                    model.generate()
            finally:
                stop_signal.set_thread_stop_event(None)
            assert tokens == ["Hello"]
            billed = model.take_partial_usage_response()
            assert billed is not None, "the completed response's usage was dropped"
            assert model.extract_input_output_token_counts_from_response(billed) == (7, 3, 0, 0, 0)
            assert len(server.requests) == 1
