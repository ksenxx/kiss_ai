# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Responses delegate's lifecycle.

Audit findings 02/I1, 02/I5, 02/R3 and 02/I4.  ``OpenAICompatibleModel``
(Chat Completions) routes tool-bearing reasoning turns through a cached
``OpenAICompatibleModel2`` delegate, and that cache leaked state in four
distinct ways:

* **I1** — the delegate captured ``token_callback`` / ``thinking_callback``
  by value when it was constructed and nothing ever refreshed them.
  ``KISSAgent._reset`` reuses a model across sub-sessions and rebinds the
  *model's* callbacks to the new run's printer, so run 2's delegated
  turns streamed every token and thinking bracket into run 1's dead
  printer — while run 2's own non-delegated turns streamed correctly.
* **I5** — ``delegate.initialize("_")`` ran before every delegated step,
  and ``initialize`` builds a brand-new ``OpenAI`` client.  A 100-step
  agentic run built 100 clients and 100 httpx connection pools, so every
  step paid a fresh handshake instead of reusing a keep-alive connection.
* **R3** — ``_delegate_raw_items`` grew by one whole-turn payload per tool
  call and was never evicted, not even by ``reset_conversation()``.
* **I4** — ``OpenAICompatibleModel2`` cleared its per-turn state only in
  ``initialize()``, so ``reset_conversation()`` left
  ``_pending_function_calls`` behind and the next generation was rejected
  on a fresh, empty conversation — a permanently unusable model object.

No mocks: a real ``ThreadingHTTPServer`` speaks the Responses API to the
real OpenAI SDK, and the client's connection reuse is measured from the
server side by counting distinct TCP peers.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "gpt-delegate-under-test"
_REASONING_TEXT = "weighing the options"
_TEXT = "calling the tool"

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


def echo(text: str) -> str:
    """Echo back ``text`` (test-only tool stub).

    Args:
        text: The string to echo.

    Returns:
        The input string unchanged.
    """
    return text


def _output_items(call_id: str) -> list[dict[str, Any]]:
    """Return the ``response.output`` items of one tool-bearing turn.

    Args:
        call_id: The ``call_id`` of the emitted function call.

    Returns:
        A reasoning item, an assistant message and a function call.
    """
    return [
        {
            "type": "reasoning",
            "id": f"rs_{call_id}",
            "summary": [{"type": "summary_text", "text": _REASONING_TEXT}],
        },
        {
            "type": "message",
            "id": f"msg_{call_id}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": _TEXT, "annotations": []}],
        },
        {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": "echo",
            "arguments": '{"text": "hi"}',
        },
    ]


def _completed_response(call_id: str) -> dict[str, Any]:
    """Return a completed Responses payload carrying one tool call.

    Args:
        call_id: The ``call_id`` of the emitted function call.

    Returns:
        The response object.
    """
    return {
        "id": f"resp_{call_id}",
        "object": "response",
        "created_at": 0,
        "model": _MODEL,
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": _output_items(call_id),
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 9,
        },
    }


def _stream_chunks(call_id: str) -> list[bytes]:
    """Render the SSE events of one streamed tool-bearing turn.

    Args:
        call_id: The ``call_id`` of the emitted function call.

    Returns:
        The SSE chunks, ending with ``response.completed``.
    """
    return [
        responses_event(
            "response.created",
            {"response": {"id": f"resp_{call_id}", "status": "in_progress",
                          "output": []}},
        ),
        responses_event(
            "response.reasoning_summary_text.delta",
            {
                "item_id": f"rs_{call_id}",
                "output_index": 0,
                "summary_index": 0,
                "delta": _REASONING_TEXT,
            },
        ),
        responses_event(
            "response.output_text.delta",
            {
                "item_id": f"msg_{call_id}",
                "output_index": 1,
                "content_index": 0,
                "delta": _TEXT,
            },
        ),
        responses_event(
            "response.output_item.added",
            {
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": "echo",
                    "arguments": "",
                },
            },
        ),
        responses_event(
            "response.function_call_arguments.delta",
            {
                "item_id": f"fc_{call_id}",
                "output_index": 2,
                "delta": '{"text": "hi"}',
            },
        ),
        responses_event(
            "response.completed", {"response": _completed_response(call_id)}
        ),
    ]


class _ToolTurnPolicy:
    """Answers every ``/v1/responses`` call with one fresh tool call."""

    def __init__(self) -> None:
        """Start the call-id counter."""
        self.turns = 0
        self.lock = threading.Lock()

    def __call__(self, request: Request) -> Reply:
        """Return a streamed or JSON tool-call turn, matching the request."""
        with self.lock:
            self.turns += 1
            call_id = f"call_{self.turns}"
        if request.body.get("stream"):
            return Reply(sse_chunks=_stream_chunks(call_id))
        return Reply(json_body=_completed_response(call_id))


def _make_chat_model(
    base_url: str,
    token_callback: Any = None,
    thinking_callback: Any = None,
) -> OpenAICompatibleModel:
    """Build a v1 model that delegates tool turns to the Responses API.

    Args:
        base_url: The scripted server's ``/v1`` root.
        token_callback: Optional streamed-token callback.
        thinking_callback: Optional thinking-bracket callback.

    Returns:
        An initialized model.
    """
    model = OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        model_config={"reasoning_effort": "high", "use_responses_api": True},
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )
    model.initialize("Do the thing.")
    return model


def _run_tool_turn(model: OpenAICompatibleModel) -> None:
    """Run one delegated tool turn and answer its tool call.

    Args:
        model: The model under test.
    """
    model.generate_and_process_with_tools({"echo": echo}, tools_schema=_TOOLS)
    model.add_function_results_to_conversation_and_return(
        [("echo", {"result": "hi"})]
    )


class _Printer:
    """Collects what one run's printer would have rendered."""

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


class TestDelegateCallbacksFollowTheModel:
    """I1: the cached delegate must stream to the *current* printer."""

    def test_second_run_streams_to_the_new_printer(self) -> None:
        """Rebinding the model's callbacks must reach the delegate."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            first = _Printer()
            model = _make_chat_model(
                server.base_url, first.token, first.bracket
            )
            _run_tool_turn(model)
            assert _REASONING_TEXT in "".join(first.tokens)
            first_transcript = list(first.tokens)

            # Exactly what KISSAgent._reset does when the same model
            # instance is reused for the next task.
            second = _Printer()
            model.token_callback = second.token
            model.thinking_callback = second.bracket
            model.reset_conversation()
            model.initialize("Do the next thing.")
            _run_tool_turn(model)

        assert _REASONING_TEXT in "".join(second.tokens), (
            "the new run's printer received nothing: the delegate is still "
            "streaming into the previous run's printer"
        )
        assert _TEXT in "".join(second.tokens)
        assert second.thinking == [True, False]
        assert first.tokens == first_transcript, (
            "a finished task's printer received the next run's tokens"
        )


class TestDelegateReusesItsConnection:
    """I5: the delegate must not rebuild its client on every step."""

    def test_five_turns_share_a_connection(self) -> None:
        """Five delegated steps must not open five connections."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = _make_chat_model(server.base_url)
            for _ in range(5):
                _run_tool_turn(model)
            connections = server.connection_keys
            requests = len(server.requests)
        assert requests == 5
        assert len(connections) <= 2, (
            f"{len(connections)} connections for {requests} delegated steps — "
            f"a new OpenAI client (and pool) is being built per step"
        )


class TestDelegateRawItemsAreBounded:
    """R3: the per-call_id item cache must not grow without bound."""

    def test_cache_tracks_only_live_tool_calls(self) -> None:
        """Entries survive only while their call_id is in the conversation."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = _make_chat_model(server.base_url)
            for _ in range(4):
                _run_tool_turn(model)
                live = {
                    tc["id"]
                    for msg in model.conversation
                    for tc in (msg.get("tool_calls") or [])
                }
                assert set(model._delegate_raw_items) == live

            model.reset_conversation()
            assert model._delegate_raw_items == {}

    def test_trimmed_tool_call_is_evicted(self) -> None:
        """Context trimming must take the cached items with it.

        Sorcar drops older messages to stay inside the context window;
        the raw items cached for a call_id that is no longer anywhere in
        the conversation can never be replayed again.
        """
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = _make_chat_model(server.base_url)
            _run_tool_turn(model)
            _run_tool_turn(model)
            assert set(model._delegate_raw_items) == {"call_1", "call_2"}

            model.conversation = [
                msg
                for msg in model.conversation
                if "call_1" not in str(msg)
            ]
            _run_tool_turn(model)
        assert set(model._delegate_raw_items) == {"call_2", "call_3"}

    def test_cache_is_emptied_for_a_discarded_conversation(self) -> None:
        """A reset conversation must not keep paying for its old turns."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = _make_chat_model(server.base_url)
            _run_tool_turn(model)
            assert model._delegate_raw_items
            model.reset_conversation()
            model.initialize("Fresh start.")
            _run_tool_turn(model)
            assert len(model._delegate_raw_items) == 1


class TestResponsesModelResetsTurnState:
    """I4: ``reset_conversation()`` must clear per-turn Responses state."""

    def _make(self, base_url: str, streaming: bool) -> OpenAICompatibleModel2:
        """Build a direct Responses model.

        Args:
            base_url: The scripted server's ``/v1`` root.
            streaming: Attach a token callback to take the stream path.

        Returns:
            An initialized model.
        """
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=base_url,
            api_key="test-key",
            token_callback=(lambda _t: None) if streaming else None,
        )
        model.initialize("Do the thing.")
        return model

    def test_reset_allows_a_fresh_generation(self) -> None:
        """A stopped tool turn must not brick the model object."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = self._make(server.base_url, streaming=False)
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
            assert model._pending_function_calls

            # The user pressed Stop before the results were delivered, and
            # the agent reused the model via reset_conversation().
            model.reset_conversation()
            model.conversation = [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ]
            content, _response = model.generate()
        assert content == _TEXT

    def test_reset_clears_stream_indexes(self) -> None:
        """Ordering learned while streaming must not outlive the reset."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = self._make(server.base_url, streaming=True)
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
            assert model._last_stream_item_indexes
            model.reset_conversation()
        assert model._last_stream_item_indexes == {}
        assert model._last_stream_message_output_index is None

    def test_non_streamed_turn_does_not_inherit_stream_order(self) -> None:
        """A non-streamed turn must not reuse the previous turn's indexes."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = self._make(server.base_url, streaming=True)
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
            assert model._last_stream_message_output_index is not None
            model.add_function_results_to_conversation_and_return(
                [("echo", {"result": "hi"})]
            )
            model.token_callback = None
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
        assert model._last_stream_message_output_index is None
        assert model._last_stream_item_indexes == {}


class TestDelegateStateSurvivesReuse:
    """The delegate must stay usable across a reset of its owner."""

    def test_owner_reset_clears_the_delegate_pending_calls(self) -> None:
        """A reset owner must hand the delegate a clean slate."""
        with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
            model = _make_chat_model(server.base_url)
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
            delegate = model._responses_delegate
            assert delegate is not None
            assert delegate._pending_function_calls

            model.reset_conversation()
            assert delegate._pending_function_calls == []

            model.initialize("Fresh start.")
            _run_tool_turn(model)
            assert model._responses_delegate is delegate


@pytest.mark.parametrize("streaming", [False, True])
def test_delegated_turn_returns_the_tool_call(streaming: bool) -> None:
    """Both transports must still parse the delegated tool call."""
    printer = _Printer()
    with ScriptedOpenAIServer(_ToolTurnPolicy()) as server:
        model = _make_chat_model(
            server.base_url,
            printer.token if streaming else None,
            printer.bracket if streaming else None,
        )
        function_calls, content, _response = (
            model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
        )
    assert content == _TEXT
    assert function_calls == [
        {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
    ]
