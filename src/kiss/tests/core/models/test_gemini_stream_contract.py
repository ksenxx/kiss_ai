# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end contract tests for :class:`GeminiModel`'s streaming path.

Three defects from audit ``tmp/audit/03-core-models-c.md`` are pinned
here, all against a real local Gemini endpoint driving the real
``google-genai`` SDK (see :mod:`gemini_sse_harness`) — no mocks, no
patches, no test doubles:

* **I1** — ``_parse_parts`` concatenated ``part.text`` without checking
  ``part.thought``, so summarized reasoning leaked into the assistant
  message on the tool-calling path (and was re-uploaded as prompt
  context on every later step), while the non-tool path filtered it.
* **I3** — the adapter kept the *last* streamed chunk as the response
  handed to ``extract_input_output_token_counts_from_response``, so a
  terminal ``finishReason``-only chunk zeroed the step's cost.
* **R1** — the thinking bracket was tracked on a private duplicate of
  the base ``_thinking_open`` flag which ``reset_conversation`` never
  cleared and no ``finally`` ever closed, so a stream that failed
  mid-thought left the next task's printer stuck in thinking mode.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    function_call_part,
    serve,
    text_part,
)

_MODEL = "gemini-contract-under-test"
_SECRET = "SECRET-REASONING-DO-NOT-ECHO"
_ANSWER = "The visible answer."

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


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    tokens: list[str] | None = None,
    thinking: list[bool] | None = None,
) -> GeminiModel:
    """Build a real ``GeminiModel`` pointed at the local endpoint."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        _MODEL,
        api_key="test-key",
        token_callback=None if tokens is None else tokens.append,
        thinking_callback=None if thinking is None else thinking.append,
    )
    model.initialize("Summarize the repository.")
    return model


def _thought_then_answer_then_call() -> list[dict[str, Any]]:
    """A turn that thinks, answers and calls a tool, then finishes."""
    return [
        chunk([text_part(_SECRET, thought=True)]),
        chunk(
            [text_part(_ANSWER), function_call_part("finish", {"result": "ok"})],
            usage={
                "promptTokenCount": 120,
                "candidatesTokenCount": 40,
                "thoughtsTokenCount": 17,
                "cachedContentTokenCount": 20,
            },
        ),
        chunk([], finish_reason="STOP"),
    ]


class TestThoughtTextNeverReachesContent:
    """I1: summarized reasoning must not become assistant content."""

    def test_tool_path_excludes_thought_text(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """The tool-calling path must return and store only real text."""
        base_url, script = gemini_endpoint
        script.play(_thought_then_answer_then_call())
        tokens: list[str] = []
        model = _make_model(monkeypatch, base_url, tokens=tokens)

        function_calls, content, _ = model.generate_and_process_with_tools(
            {}, tools_schema=[_FINISH_TOOL],
        )

        assert content == _ANSWER
        assert _SECRET not in content
        assert _SECRET not in json.dumps(model.conversation, default=str)
        assert [fc["name"] for fc in function_calls] == ["finish"]
        # The reasoning is still shown live, just not stored.
        assert _SECRET in "".join(tokens)

    def test_thought_signature_is_kept_for_the_tool_call(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Dropping thought text must not drop the call's thought signature."""
        base_url, script = gemini_endpoint
        script.play([
            chunk([text_part(_SECRET, thought=True)]),
            chunk([
                text_part(_ANSWER),
                function_call_part(
                    "finish", {"result": "ok"}, thought_signature="c2ln",
                ),
            ]),
        ])
        model = _make_model(monkeypatch, base_url, tokens=[])

        function_calls, content, _ = model.generate_and_process_with_tools(
            {}, tools_schema=[_FINISH_TOOL],
        )

        assert content == _ANSWER
        assert model._thought_signatures[function_calls[0]["id"]] == b"sig"

    def test_both_paths_agree_on_content(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """``generate`` and the tool path must produce the same text."""
        base_url, script = gemini_endpoint
        script.play(_thought_then_answer_then_call())
        plain = _make_model(monkeypatch, base_url, tokens=[])
        plain_content, _ = plain.generate()

        script.play(_thought_then_answer_then_call())
        tooled = _make_model(monkeypatch, base_url, tokens=[])
        _, tooled_content, _ = tooled.generate_and_process_with_tools(
            {}, tools_schema=[_FINISH_TOOL],
        )

        assert plain_content == tooled_content == _ANSWER

    def test_unstreamed_path_excludes_thought_text(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Without a token callback the SDK's unary call is used."""
        base_url, script = gemini_endpoint
        script.play(_thought_then_answer_then_call())
        model = _make_model(monkeypatch, base_url)

        _, content, _ = model.generate_and_process_with_tools(
            {}, tools_schema=[_FINISH_TOOL],
        )

        assert content == _ANSWER


class TestUsageComesFromTheChunkThatCarriesIt:
    """I3: a trailing usage-free chunk must not zero the step's cost."""

    def test_usage_from_second_to_last_chunk(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Usage on chunk N-1 and a bare ``finishReason`` on chunk N."""
        base_url, script = gemini_endpoint
        script.play(_thought_then_answer_then_call())
        model = _make_model(monkeypatch, base_url, tokens=[])

        _, _, response = model.generate_and_process_with_tools(
            {}, tools_schema=[_FINISH_TOOL],
        )
        counts = model.extract_input_output_token_counts_from_response(response)

        # prompt 120 - cached 20 = 100 input; 40 + 17 thoughts = 57 output.
        assert counts == (100, 57, 20, 0)

    def test_usage_survives_on_the_plain_generate_path(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """``generate()`` must report the same non-zero usage."""
        base_url, script = gemini_endpoint
        script.play(_thought_then_answer_then_call())
        model = _make_model(monkeypatch, base_url, tokens=[])

        _, response = model.generate()

        assert model.extract_input_output_token_counts_from_response(response) == (
            100, 57, 20, 0,
        )

    def test_empty_stream_falls_back_to_a_unary_call(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A stream yielding nothing must still produce a usable turn."""
        base_url, script = gemini_endpoint
        script.play([])
        tokens: list[str] = []
        model = _make_model(monkeypatch, base_url, tokens=tokens)

        content, response = model.generate()

        assert content == ""
        assert model.extract_input_output_token_counts_from_response(response) == (
            0, 0, 0, 0,
        )


class TestThinkingBracketIsBalanced:
    """R1: the thinking bracket is one flag, always closed, always reset."""

    def test_bracket_closes_when_the_stream_dies_mid_thought(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A connection cut mid-thought must still emit the closing False."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part(_SECRET, thought=True)])], after="cut")
        thinking: list[bool] = []
        model = _make_model(monkeypatch, base_url, tokens=[], thinking=thinking)

        with pytest.raises(Exception):  # noqa: B017 — any transport failure
            model.generate()

        assert thinking == [True, False]
        assert model._thinking_open is False

    def test_reset_conversation_clears_the_bracket(
        self, monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """A reused instance must open a fresh bracket for the next task."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part(_SECRET, thought=True)])], after="cut")
        model = _make_model(monkeypatch, base_url, tokens=[], thinking=[])
        with pytest.raises(Exception):  # noqa: B017 — any transport failure
            model.generate()

        # KISSAgent._reset reuses the adapter and rebinds a new printer.
        model.reset_conversation()
        assert model._thinking_open is False
        second: list[bool] = []
        second_tokens: list[str] = []
        model.thinking_callback = second.append
        model.token_callback = second_tokens.append
        script.play(_thought_then_answer_then_call())
        model.initialize("Second task.")
        model.generate()

        assert second == [True, False]
