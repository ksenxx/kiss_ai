# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Gemini turn is finished only when the provider says how it ended.

``_stream_turn`` accumulated parts and usage without ever reading the
candidate's ``finishReason`` or the ``promptFeedback`` of a blocked
prompt, so a stream a gateway closed before its terminal chunk, a
``SAFETY`` stop, a blocked prompt and an output cap that landed before
any tool call all came back as finished turns and were appended to the
conversation.  Now, as on the Anthropic and OpenAI transports: a stream
without a finish reason is refused, a block or a filter stop is a
:class:`ContentFilterError` (the agentic loop fails over instead of
retrying the same family), and ``MAX_TOKENS`` on a tool-bearing turn
with no completed call is an :class:`OutputCapError`.  A completed call
followed by the cap, and a tool-less turn the cap cut, are still used.

Every case drives the real ``google-genai`` SDK against a real local
endpoint; the usage seen before a refused end stays billable through
``take_partial_usage_response``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.kiss_error import ContentFilterError, KISSError, OutputCapError
from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    function_call_part,
    serve,
    text_part,
)

_USAGE = {"promptTokenCount": 123, "candidatesTokenCount": 7}
_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write text.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
_CALL = function_call_part("write", {"path": "target.txt", "content": "FULL"})


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _turn(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[str, GeminiScript],
    chunks: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[GeminiModel, list[dict[str, Any]] | KISSError, str]:
    """Run one tool turn against the endpoint streaming *chunks*; return model, calls, text."""
    base_url, script = endpoint
    script.play(chunks)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel("gemini-under-test", api_key="test-key")
    model.initialize("Write the complete content.")
    try:
        calls, text, _response = model.generate_and_process_with_tools({}, tools)
    except KISSError as error:
        return model, error, ""
    return model, calls, text


def test_a_stream_cut_before_its_finish_reason_is_refused(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    """A clean EOF before the terminal chunk is not a turn: nothing appended, usage kept."""
    chunks = [chunk([text_part("Partial answer. "), _CALL], usage=_USAGE)]
    model, outcome, _text = _turn(monkeypatch, gemini_endpoint, chunks, [_WRITE_TOOL])
    assert isinstance(outcome, KISSError), f"the cut stream was accepted: {outcome}"
    assert "finish reason" in str(outcome)
    assert [m["role"] for m in model.conversation] == ["user"]
    partial = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(partial)[:2] == (123, 7)

    # Control: the same stream ended by a finish-reason chunk is a finished turn.
    model, outcome, text = _turn(
        monkeypatch, gemini_endpoint, [*chunks, chunk([], finish_reason="STOP")], [_WRITE_TOOL]
    )
    assert isinstance(outcome, list) and [c["name"] for c in outcome] == ["write"]
    assert text == "Partial answer. "
    assert [m["role"] for m in model.conversation] == ["user", "assistant"]
    assert model.take_partial_usage_response() is None


@pytest.mark.parametrize(
    "chunks",
    [
        [{"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": _USAGE}],
        [chunk([text_part("Partial answer. ")], usage=_USAGE), chunk([], finish_reason="SAFETY")],
    ],
    ids=["prompt-blocked", "safety-stop"],
)
def test_a_block_or_a_filter_stop_is_a_content_filter_error(
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
    chunks: list[dict[str, Any]],
) -> None:
    model, outcome, _text = _turn(monkeypatch, gemini_endpoint, chunks, [_WRITE_TOOL])
    assert isinstance(outcome, ContentFilterError), f"the refused turn was accepted: {outcome}"
    assert outcome.stop_reason == "SAFETY"
    assert "SAFETY" in str(outcome)
    assert [m["role"] for m in model.conversation] == ["user"]
    partial = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(partial)[:2] == (123, 7)


def test_an_output_cap_before_any_tool_call_is_an_output_cap_error(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    capped = [chunk([text_part("Incomplete answer.")], usage=_USAGE, finish_reason="MAX_TOKENS")]
    model, outcome, _text = _turn(monkeypatch, gemini_endpoint, capped, [_WRITE_TOOL])
    assert isinstance(outcome, OutputCapError), f"the capped turn was accepted: {outcome}"
    assert outcome.stop_reason == "MAX_TOKENS"
    assert [m["role"] for m in model.conversation] == ["user"]
    partial = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(partial)[:2] == (123, 7)

    # A tool-less caller still reads the cut-off text ...
    model, outcome, text = _turn(monkeypatch, gemini_endpoint, capped, [])
    assert outcome == [] and text == "Incomplete answer."
    assert [m["role"] for m in model.conversation] == ["user", "assistant"]
    # ... and a call completed before the cap is still used.
    completed = [chunk([_CALL], usage=_USAGE, finish_reason="MAX_TOKENS")]
    _model, outcome, _text = _turn(monkeypatch, gemini_endpoint, completed, [_WRITE_TOOL])
    assert isinstance(outcome, list) and [c["name"] for c in outcome] == ["write"]
