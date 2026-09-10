# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: connection errors must not multiply model requests.

User report: with ``openrouter/moonshotai/kimi-k3`` "the request and
response gets repeated", and the trigger is ``APIConnectionError:
Connection error.`` — a transport failure between the client and the
provider.  Two code paths turned one logical turn into several identical
upstream generations:

* ``OpenAICompatibleBase._ensure_client`` built the ``OpenAI`` client
  without ``max_retries``, inheriting the SDK default of 2 *silent*
  re-sends on any transport failure.  When the provider had already
  received (and billed) the request but the response was lost in
  transit, every silent retry generated the same answer again — on top
  of the up-to-3 visible retries ``KISSAgent._run_agentic_loop`` adds,
  one turn could reach 9 identical upstream request/response pairs.

* ``OpenAICompatibleModel._stream_chat_completion`` discarded a stream
  that failed at *any* point, including after ``finish_reason`` had
  already been received — a response that is complete except for the
  usage/[DONE] tail.  The agent then retried, and the user watched the
  same answer stream twice.

No mocks: a real ``ThreadingHTTPServer`` (``ScriptedOpenAIServer``)
answers the real OpenAI SDK over real sockets, dropping the connection
exactly where each scenario requires.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from openai import APIConnectionError

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_MODEL = "gpt-conn-error-under-test"


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


def _delta_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
    """Render one Chat Completions delta chunk.

    Args:
        delta: The ``choices[0].delta`` object.
        finish_reason: The ``choices[0].finish_reason`` value.

    Returns:
        The SSE bytes for the chunk.
    """
    return chat_chunk(
        {
            "id": "chatcmpl-conn",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
    )


_USAGE_CHUNK = chat_chunk(
    {
        "id": "chatcmpl-conn",
        "object": "chat.completion.chunk",
        "model": _MODEL,
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
)

_DONE = b"data: [DONE]\n\n"

_COMPLETION_BODY = {
    "id": "chatcmpl-conn",
    "object": "chat.completion",
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "recovered"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class _DropPolicy:
    """Drops the connection (no response bytes) for the first N requests."""

    def __init__(self, drops: int) -> None:
        self.drops = drops

    def __call__(self, request: Request) -> Reply:
        """Drop while drops remain, then answer with a full completion."""
        if self.drops > 0:
            self.drops -= 1
            return Reply(drop=True)
        return Reply(json_body=_COMPLETION_BODY)


class _TruncatedStreamPolicy:
    """Streams the scripted chunks but drops the connection early."""

    def __init__(self, chunks: list[bytes], truncate_after: int) -> None:
        self.chunks = chunks
        self.truncate_after = truncate_after

    def __call__(self, request: Request) -> Reply:
        """Return the scripted stream, truncated mid-body."""
        return Reply(sse_chunks=self.chunks, truncate_after=self.truncate_after)


@pytest.fixture
def always_drop_server() -> Generator[ScriptedOpenAIServer]:
    """An endpoint that resets every connection before responding."""
    with ScriptedOpenAIServer(_DropPolicy(drops=10**9)) as server:
        yield server


@pytest.fixture
def drop_once_server() -> Generator[ScriptedOpenAIServer]:
    """An endpoint that resets the first connection, then recovers."""
    with ScriptedOpenAIServer(_DropPolicy(drops=1)) as server:
        yield server


def _model(server: ScriptedOpenAIServer, streaming: bool) -> OpenAICompatibleModel:
    """Build a real v1 model pointed at the scripted server.

    Args:
        server: The scripted endpoint.
        streaming: Attach a token callback (streaming transport) or not.

    Returns:
        The model under test.
    """
    return OpenAICompatibleModel(
        _MODEL,
        base_url=server.base_url,
        api_key="test-key",
        token_callback=(lambda _t: None) if streaming else None,
    )


class TestSilentSdkRetriesAreBounded:
    """The SDK must not silently repeat a request more than once."""

    def test_connection_error_repeats_the_request_exactly_once(
        self, always_drop_server: ScriptedOpenAIServer
    ) -> None:
        """One failing call sends exactly 2 requests (1 + 1 silent retry).

        With the SDK's default ``max_retries=2`` a single ``generate()``
        sent 3 identical requests — each one a separate (billable)
        generation when the provider had actually received it.
        """
        model = _model(always_drop_server, streaming=False)
        model.initialize("Say something.")
        with pytest.raises(APIConnectionError):
            model.generate()
        assert len(always_drop_server.requests) == 2

    def test_dropped_connection_recovers_on_the_single_retry(
        self, drop_once_server: ScriptedOpenAIServer
    ) -> None:
        """A one-off connection reset is retried once and succeeds."""
        model = _model(drop_once_server, streaming=False)
        model.initialize("Say something.")
        content, _response = model.generate()
        assert content == "recovered"
        assert len(drop_once_server.requests) == 2


class TestCompleteStreamSurvivesTailDrop:
    """A response whose stream died after ``finish_reason`` is kept."""

    def test_text_response_is_kept_not_regenerated(self) -> None:
        """The full text is returned from the single request."""
        chunks = [
            _delta_chunk({"role": "assistant", "content": "Hello "}),
            _delta_chunk({"content": "world"}),
            _delta_chunk({}, finish_reason="stop"),
            _USAGE_CHUNK,
            _DONE,
        ]
        policy = _TruncatedStreamPolicy(chunks, truncate_after=3)
        with ScriptedOpenAIServer(policy) as server:
            model = _model(server, streaming=True)
            model.initialize("Say something.")
            content, _response = model.generate()
            assert content == "Hello world"
            assert len(server.requests) == 1
            assert model.conversation[-1] == {
                "role": "assistant",
                "content": "Hello world",
            }

    def test_tool_call_response_is_kept_not_regenerated(self) -> None:
        """Complete tool calls are parsed from the single request."""
        chunks = [
            _delta_chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "finish", "arguments": ""},
                        }
                    ],
                }
            ),
            _delta_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"result": "ok"}'},
                        }
                    ]
                }
            ),
            _delta_chunk({}, finish_reason="tool_calls"),
            _USAGE_CHUNK,
            _DONE,
        ]
        policy = _TruncatedStreamPolicy(chunks, truncate_after=3)
        with ScriptedOpenAIServer(policy) as server:
            model = _model(server, streaming=True)
            model.initialize("Use the finish tool.")
            function_calls, _content, _response = (
                model.generate_and_process_with_tools(
                    {}, tools_schema=_tool_schema()
                )
            )
            assert function_calls == [
                {"id": "call_1", "name": "finish", "arguments": {"result": "ok"}}
            ]
            assert len(server.requests) == 1

    def test_v2_responses_stream_is_kept_after_terminal_event(self) -> None:
        """The Responses transport (kimi-k3's) keeps a completed stream.

        ``openrouter/moonshotai/kimi-k3`` is built as
        ``OpenAICompatibleModel2`` (``use_responses_api`` in MODEL_INFO),
        so the tail-drop scenario must also be covered there: a
        connection lost after the terminal ``response.completed`` event
        must not discard the complete response.
        """
        response_payload = {
            "id": "resp_conn",
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
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello world",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        }
        chunks = [
            responses_event(
                "response.output_text.delta",
                {
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Hello world",
                    "logprobs": [],
                },
            ),
            responses_event("response.completed", {"response": response_payload}),
            # Trailing bytes announced in Content-Length but never sent:
            # the connection drops while delivering the stream's tail.
            b": tail padding that is never delivered\n\n" * 8,
        ]
        policy = _TruncatedStreamPolicy(chunks, truncate_after=2)
        with ScriptedOpenAIServer(policy) as server:
            model = OpenAICompatibleModel2(
                _MODEL,
                base_url=server.base_url,
                api_key="test-key",
                token_callback=lambda _t: None,
            )
            model.initialize("Say something.")
            content, _response = model.generate()
            assert content == "Hello world"
            assert len(server.requests) == 1

    def test_drop_before_finish_reason_still_raises(self) -> None:
        """A genuinely incomplete stream must fail, not pass as success."""
        chunks = [
            _delta_chunk({"role": "assistant", "content": "Hello "}),
            _delta_chunk({"content": "world"}),
            _delta_chunk({}, finish_reason="stop"),
            _USAGE_CHUNK,
            _DONE,
        ]
        policy = _TruncatedStreamPolicy(chunks, truncate_after=2)
        with ScriptedOpenAIServer(policy) as server:
            model = _model(server, streaming=True)
            model.initialize("Say something.")
            with pytest.raises(Exception):
                model.generate()
            # The partial text must not be recorded as a completed turn.
            assert model.conversation[-1]["role"] == "user"
