# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: both streamed entry points must behave identically.

Audit findings 02/R2 and 02/R4.

* **R2** — the ~45-line Chat Completions streaming loop existed verbatim
  in two places (``_stream_text`` and the streaming branch of
  ``generate_and_process_with_tools``), and the copies had already
  drifted apart in which request call they made.  Any fix applied to one
  silently missed the other.  The loop now lives once, and this suite is
  the guard that keeps the two entry points observationally identical.
* **R4** — the usage extractor computed ``text_output_tokens`` and then
  returned ``completion_tokens`` on the non-audio branch, leaving a
  reader unable to tell whether the asymmetry was deliberate.  The two
  branches now agree on what "output tokens" means; these tests pin that
  meaning for both.

No mocks: one real ``ThreadingHTTPServer`` streams the same SSE bytes to
both entry points through the real OpenAI SDK.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-parity-under-test"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text back",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    }
]


def _delta_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
    """Render one Chat Completions chunk carrying *delta*.

    Args:
        delta: The choice delta object.
        finish_reason: The terminal finish reason, when any.

    Returns:
        The SSE bytes.
    """
    return chat_chunk(
        {
            "id": "chatcmpl-parity",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
    )


def _reasoning_and_text_stream(request: Request) -> Reply:
    """Stream reasoning deltas, text deltas and a terminal usage chunk."""
    return Reply(
        sse_chunks=[
            _delta_chunk({"role": "assistant", "reasoning_content": "think"}),
            _delta_chunk({"reasoning_content": "ing"}),
            _delta_chunk({"content": "hello "}),
            _delta_chunk({"content": "world"}, finish_reason="stop"),
            chat_chunk(
                {
                    "id": "chatcmpl-parity",
                    "object": "chat.completion.chunk",
                    "model": _MODEL,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 40,
                        "total_tokens": 140,
                        "prompt_tokens_details": {"cached_tokens": 10},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


class _Printer:
    """Collects what a printer would have rendered for one turn."""

    def __init__(self) -> None:
        """Start with empty transcripts."""
        self.tokens: list[str] = []
        self.thinking: list[bool] = []

    def token(self, text: str) -> None:
        """Record a streamed token."""
        self.tokens.append(text)

    def bracket(self, is_start: bool) -> None:
        """Record a thinking-bracket transition."""
        self.thinking.append(is_start)


def _make_model(base_url: str, printer: _Printer) -> OpenAICompatibleModel:
    """Build a streaming model wired to *printer*.

    Args:
        base_url: The scripted server's ``/v1`` root.
        printer: The transcript collector.

    Returns:
        An initialized model.
    """
    model = OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        token_callback=printer.token,
        thinking_callback=printer.bracket,
    )
    model.initialize("Say hello.")
    return model


class TestStreamingEntryPointsAgree:
    """One loop, so both entry points must produce one behaviour."""

    def test_transcripts_and_usage_match(self) -> None:
        """Same bytes in, same callbacks and same usage out."""
        with ScriptedOpenAIServer(_reasoning_and_text_stream) as server:
            toolless = _Printer()
            model_a = _make_model(server.base_url, toolless)
            content_a, response_a = model_a.generate()

            with_tools = _Printer()
            model_b = _make_model(server.base_url, with_tools)
            _calls, content_b, response_b = (
                model_b.generate_and_process_with_tools(
                    {}, tools_schema=_TOOLS
                )
            )

        assert content_a == content_b == "hello world"
        assert toolless.tokens == with_tools.tokens
        assert toolless.tokens == ["think", "ing", "hello ", "world"]
        assert toolless.thinking == with_tools.thinking == [True, False]
        assert model_a.extract_input_output_token_counts_from_response(
            response_a
        ) == model_b.extract_input_output_token_counts_from_response(response_b)

    def test_both_entry_points_see_the_terminal_usage_chunk(self) -> None:
        """The usage chunk must win over the last content chunk."""
        with ScriptedOpenAIServer(_reasoning_and_text_stream) as server:
            printer = _Printer()
            model = _make_model(server.base_url, printer)
            _content, response = model.generate()
            counts = model.extract_input_output_token_counts_from_response(
                response
            )
        assert counts == (90, 40, 10, 0)


def _split_tool_call_stream(request: Request) -> Reply:
    """Stream a tool call the way providers really send one.

    The first delta carries the id and the name, later deltas carry only
    argument fragments, and one delta carries no function payload at all.
    """
    return Reply(
        sse_chunks=[
            _delta_chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_split",
                            "type": "function",
                            "function": {"name": "echo", "arguments": ""},
                        }
                    ],
                }
            ),
            _delta_chunk(
                {"tool_calls": [{"index": 0, "type": "function"}]}
            ),
            _delta_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"text": '},
                        }
                    ]
                }
            ),
            _delta_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"hi"}'},
                        }
                    ]
                },
                finish_reason="tool_calls",
            ),
            b"data: [DONE]\n\n",
        ]
    )


def test_tool_call_deltas_are_accumulated_across_chunks() -> None:
    """Id, name and argument fragments arrive in separate chunks."""
    with ScriptedOpenAIServer(_split_tool_call_stream) as server:
        printer = _Printer()
        model = _make_model(server.base_url, printer)
        function_calls, _content, _response = (
            model.generate_and_process_with_tools({}, tools_schema=_TOOLS)
        )
    assert function_calls == [
        {"id": "call_split", "name": "echo", "arguments": {"text": "hi"}}
    ]


def _usage_reply(usage: dict[str, Any]) -> Any:
    """Build a responder answering with a completion carrying *usage*.

    Args:
        usage: The usage object to return.

    Returns:
        A responder function for the scripted server.
    """

    def responder(request: Request) -> Reply:
        """Answer with a fixed completion and the requested usage."""
        return Reply(
            json_body={
                "id": "chatcmpl-usage",
                "object": "chat.completion",
                "created": 0,
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    return responder


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
            (90, 40, 10, 0),
        ),
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_tokens_details": {
                    "cached_tokens": 10,
                    "audio_tokens": 20,
                },
                "completion_tokens_details": {"audio_tokens": 15},
            },
            (70, 25, 10, 0, 0, 20, 15),
        ),
    ],
    ids=["text-only", "with-audio"],
)
def test_output_tokens_mean_the_same_thing_on_both_branches(
    usage: dict[str, Any], expected: tuple[int, ...]
) -> None:
    """Output tokens are always the non-audio completion tokens."""
    with ScriptedOpenAIServer(_usage_reply(usage)) as server:
        model = OpenAICompatibleModel(
            _MODEL, base_url=server.base_url, api_key="test-key"
        )
        model.initialize("Say hi.")
        _content, response = model.generate()
        counts = model.extract_input_output_token_counts_from_response(response)
    assert counts == expected
