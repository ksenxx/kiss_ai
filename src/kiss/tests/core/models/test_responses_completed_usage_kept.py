# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A completed Responses stream keeps its usage if post-processing raises.

October 2026 cost audit: after ``response.completed`` the adapter held
the (already billed) response only in a local while it closed the
thinking bracket, parsed the payload and emitted the trailing text
through the token callback.  A Stop (``KeyboardInterrupt``) raised by
that callback unwound the call with nothing in the partial-usage slot,
so the agent recorded zero tokens and zero cost for a paid call.
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.test_openai_compatible_model2 import (
    _CapturingHandler,
    _echo,
    _stream_sse_event,
    _tool_call_response_json,
    _tool_call_stream_sse_body,
    capture_server,  # noqa: F401 — pytest fixture
)


def _body_with_trailing_text(reasoning_only: bool = False) -> bytes:
    """Stream ``hel`` as a delta; the terminal payload says ``hello``.

    The missing ``lo`` is delivered through the token callback AFTER the
    terminal event — the window the audit found.  With *reasoning_only*
    the only streamed delta is a reasoning summary, so the thinking
    bracket is still open at the terminal event and closes after it.
    """
    completed = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_test", "object": "response", "created_at": 0, "model": "gpt-5.5",
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }],
            "usage": {
                "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 1100,
            },
        },
    }
    if reasoning_only:
        delta = {
            "type": "response.reasoning_summary_text.delta", "sequence_number": 1,
            "item_id": "rs_1", "output_index": 0, "summary_index": 0, "delta": "thinking",
        }
        return _stream_sse_event("response.reasoning_summary_text.delta", delta) + (
            _stream_sse_event("response.completed", completed)
        )
    delta = {
        "type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_1",
        "output_index": 0, "content_index": 0, "delta": "hel", "logprobs": [],
    }
    return _stream_sse_event("response.output_text.delta", delta) + _stream_sse_event(
        "response.completed", completed
    )


def test_stop_in_trailing_token_callback_keeps_the_billed_usage(
    capture_server: str,  # noqa: F811 — fixture
) -> None:
    """The response is handed to the partial-usage drain; a clean call leaves none."""
    seen: list[str] = []

    def _stop_on_tail(token: str) -> None:
        seen.append(token)
        if token == "lo":
            raise KeyboardInterrupt("injected stop")

    m = OpenAICompatibleModel2(
        "gpt-5.5", base_url=capture_server, api_key="k", token_callback=_stop_on_tail,
    )
    m.initialize("hi")
    _CapturingHandler.next_response_body = _body_with_trailing_text()
    with pytest.raises(KeyboardInterrupt):
        m.generate()
    assert seen == ["hel", "lo"]
    partial = m.take_partial_usage_response()
    assert partial is not None
    assert m.extract_input_output_token_counts_from_response(partial) == (1000, 100, 0, 0)
    assert m.take_partial_usage_response() is None

    # The same stream completing normally leaves nothing to double count.
    m.token_callback = seen.append
    m.initialize("hi")
    _CapturingHandler.next_response_body = _body_with_trailing_text()
    text, _resp = m.generate()
    assert text == "hello"
    assert m.take_partial_usage_response() is None


def test_stop_in_closing_thinking_callback_keeps_the_billed_usage(
    capture_server: str,  # noqa: F811 — fixture
) -> None:
    """The thinking bracket closes after ``response.completed``; a Stop there keeps usage."""

    def _stop_on_close(is_start: bool) -> None:
        if not is_start:
            raise KeyboardInterrupt("injected stop")

    m = OpenAICompatibleModel2(
        "gpt-5.5", base_url=capture_server, api_key="k",
        token_callback=lambda _t: None, thinking_callback=_stop_on_close,
    )
    m.initialize("hi")
    _CapturingHandler.next_response_body = _body_with_trailing_text(reasoning_only=True)
    with pytest.raises(KeyboardInterrupt):
        m.generate()
    partial = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(partial) == (1000, 100, 0, 0)
    assert m.take_partial_usage_response() is None


def test_tool_call_validation_failure_keeps_the_billed_usage(
    capture_server: str,  # noqa: F811 — fixture
) -> None:
    """A paid reply the tool path rejects (function_call without a name) stays billable."""
    m = OpenAICompatibleModel2(
        "gpt-5.5", base_url=capture_server, api_key="k", token_callback=lambda _t: None,
    )
    m.initialize("hi")
    _CapturingHandler.next_response_body = _tool_call_stream_sse_body(name="")
    with pytest.raises(KISSError, match="function_call without name"):
        m.generate_and_process_with_tools({"echo": _echo})
    partial = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(partial) == (3, 5, 0, 0)
    assert m.take_partial_usage_response() is None

    # Non-streaming: the same validation failure, then a clean call leaves nothing.
    m.token_callback = None
    m.initialize("hi")
    _CapturingHandler.next_response_body = _tool_call_response_json(name="").encode()
    with pytest.raises(KISSError, match="function_call without name"):
        m.generate_and_process_with_tools({"echo": _echo})
    partial = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(partial) == (5, 4, 0, 0)
    m.initialize("hi")
    _CapturingHandler.next_response_body = _tool_call_response_json().encode()
    calls, _content, _resp = m.generate_and_process_with_tools({"echo": _echo})
    assert [c["name"] for c in calls] == ["echo"]
    assert m.take_partial_usage_response() is None
