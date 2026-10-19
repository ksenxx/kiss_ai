# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the system prompt survives a v1 <-> v2 hand-off exactly once.

Finding B2/F1.  ``OpenAICompatibleModel`` (Chat Completions) stores the
configured ``system_instruction`` as the first conversation message in
``initialize()`` and never re-injects it; ``OpenAICompatibleModel2``
(Responses) never stores it and sends it as top-level ``instructions`` on
every request, passing any ``role="system"`` conversation item through
verbatim.  The Sorcar ``set_model`` tool and the ``KISSAgent`` fallback
switch both do ``new_model.initialize("")`` and then overwrite
``new_model.conversation`` with the old one, so:

* v2 -> v1: the v1 request carried no system message at all.
* v1 -> v2: the request carried ``instructions=<prompt>`` AND
  ``input[0] = {"role": "system", "content": <prompt>}``.

Both directions are driven against a real local OpenAI-compatible HTTP
endpoint (``openai_sse_harness``) and asserted on the captured request
bodies.  No mocks.
"""

from __future__ import annotations

import json
from typing import Any

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_MODEL = "gpt-handoff-under-test"
_PROMPT = "SYS PROMPT: answer in haiku."


def _chat_reply() -> Reply:
    """Return a complete one-word Chat Completions stream."""
    base = {"id": "chatcmpl-h", "object": "chat.completion.chunk", "model": _MODEL}
    return Reply(
        sse_chunks=[
            chat_chunk(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            chat_chunk(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            chat_chunk(
                {
                    **base,
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


def _responses_reply() -> Reply:
    """Return a complete one-word Responses stream."""
    return Reply(
        sse_chunks=[
            responses_event(
                "response.created",
                {"response": {"id": "resp_h", "status": "in_progress", "output": []}},
            ),
            responses_event(
                "response.output_text.delta",
                {"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "ok"},
            ),
            responses_event(
                "response.completed",
                {
                    "response": {
                        "id": "resp_h",
                        "object": "response",
                        "status": "completed",
                        "model": _MODEL,
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {"type": "output_text", "text": "ok", "annotations": []}
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 3,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 4,
                        },
                    }
                },
            ),
        ]
    )


def _responder(request: Request) -> Reply:
    """Answer either transport's endpoint with a complete stream."""
    if request.path.endswith("/responses"):
        return _responses_reply()
    return _chat_reply()


def _v1(server: ScriptedOpenAIServer) -> OpenAICompatibleModel:
    """Build a Chat Completions model configured with the shared prompt."""
    return OpenAICompatibleModel(
        _MODEL,
        base_url=server.base_url,
        api_key="test-key",
        model_config={"system_instruction": _PROMPT},
        token_callback=lambda _t: None,
    )


def _v2(server: ScriptedOpenAIServer) -> OpenAICompatibleModel2:
    """Build a Responses model configured with the shared prompt."""
    return OpenAICompatibleModel2(
        _MODEL,
        base_url=server.base_url,
        api_key="test-key",
        model_config={"system_instruction": _PROMPT},
        token_callback=lambda _t: None,
    )


def _system_items(items: list[Any]) -> list[dict[str, Any]]:
    """Return the ``role="system"`` entries of a request's messages/input."""
    return [it for it in items if isinstance(it, dict) and it.get("role") == "system"]


def _hand_off(old: Any, new: Any) -> None:
    """Switch a live conversation to *new* the way ``set_model`` does."""
    new.initialize("")
    new.conversation = old.conversation


class TestResponsesToChatCompletions:
    """v2 -> v1: the v1 request must carry the prompt as a system message."""

    def test_prompt_is_sent_once_after_handoff(self) -> None:
        """A v2 conversation (no system item) still gets the prompt on v1."""
        with ScriptedOpenAIServer(_responder) as server:
            v2 = _v2(server)
            v2.initialize("hi")
            assert v2.generate()[0] == "ok"
            v1 = _v1(server)
            _hand_off(v2, v1)
            text, _resp = v1.generate()
            assert text == "ok"
            v1.conversation.append({"role": "user", "content": "again"})
            assert v1.generate()[0] == "ok"
            chat_requests = [r for r in server.requests if r.path.endswith("/chat/completions")]
        assert len(chat_requests) == 2
        for request in chat_requests:
            systems = _system_items(request.body["messages"])
            assert systems == [{"role": "system", "content": _PROMPT}], request.body
            assert request.body["messages"][0]["role"] == "system"

    def test_tool_turn_after_handoff_carries_prompt(self) -> None:
        """The agentic (tools) path goes through the same normalisation."""
        with ScriptedOpenAIServer(_responder) as server:
            v2 = _v2(server)
            v2.initialize("hi")
            v2.generate()
            v1 = _v1(server)
            _hand_off(v2, v1)
            v1.generate_and_process_with_tools(
                {},
                tools_schema=[
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
            request = [r for r in server.requests if r.path.endswith("/chat/completions")][-1]
        assert _system_items(request.body["messages"]) == [{"role": "system", "content": _PROMPT}]

    def test_fresh_v1_conversation_is_unchanged(self) -> None:
        """Without a hand-off the system message appears exactly once."""
        with ScriptedOpenAIServer(_responder) as server:
            v1 = _v1(server)
            v1.initialize("hi")
            v1.generate()
            request = server.requests[-1]
        assert [m["role"] for m in request.body["messages"]] == ["system", "user"]


class TestChatCompletionsToResponses:
    """v1 -> v2: the prompt must reach ``/v1/responses`` exactly once."""

    @staticmethod
    def _prompt_copies(body: dict[str, Any]) -> int:
        """Count how many times the prompt text appears in a request body."""
        return json.dumps(body).count(json.dumps(_PROMPT)[1:-1])

    def test_prompt_is_sent_once_after_handoff(self) -> None:
        """A v1 conversation (system item present) is not doubled by ``instructions``."""
        with ScriptedOpenAIServer(_responder) as server:
            v1 = _v1(server)
            v1.initialize("hi")
            assert v1.generate()[0] == "ok"
            v2 = _v2(server)
            _hand_off(v1, v2)
            text, _resp = v2.generate()
            assert text == "ok"
            request = [r for r in server.requests if r.path.endswith("/responses")][-1]
        body = request.body
        assert self._prompt_copies(body) == 1, body
        assert len(_system_items(body["input"])) + bool(body.get("instructions")) == 1

    def test_fresh_v2_conversation_uses_instructions(self) -> None:
        """Without a hand-off the prompt travels as top-level ``instructions``."""
        with ScriptedOpenAIServer(_responder) as server:
            v2 = _v2(server)
            v2.initialize("hi")
            v2.generate()
            request = server.requests[-1]
        assert request.body["instructions"] == _PROMPT
        assert _system_items(request.body["input"]) == []
        assert self._prompt_copies(request.body) == 1

    def test_different_system_item_is_kept_alongside_instructions(self) -> None:
        """A system item with other text is not the configured prompt: both go."""
        with ScriptedOpenAIServer(_responder) as server:
            v2 = _v2(server)
            v2.initialize("hi")
            v2.conversation.insert(0, {"role": "system", "content": "Always be polite."})
            v2.generate()
            request = server.requests[-1]
        assert request.body["instructions"] == _PROMPT
        assert _system_items(request.body["input"]) == [
            {"role": "system", "content": "Always be polite."}
        ]


_FINISH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _delegating_v1(server: ScriptedOpenAIServer) -> OpenAICompatibleModel:
    """A v1 model whose tool turns go to ``/v1/responses`` (``reasoning_effort`` + delegation)."""
    return OpenAICompatibleModel(
        _MODEL,
        base_url=server.base_url,
        api_key="test-key",
        model_config={
            "system_instruction": _PROMPT,
            "reasoning_effort": "high",
            "use_responses_api": True,
        },
        token_callback=lambda _t: None,
    )


class TestDelegatedToolTurnAfterHandoff:
    """Review R2/I1: v2 -> v1 whose tool turns are delegated back to ``/v1/responses``.

    ``generate_and_process_with_tools`` returns through
    ``_generate_with_tools_via_responses`` before the chat path's
    normalisation, and the delegate is built without ``system_instruction``
    (so it sends no ``instructions``).  The delegate's input is derived from
    the hand-off conversation, which carries no system item, so the prompt
    was missing from the request entirely.
    """

    @staticmethod
    def _prompt_copies(body: dict[str, Any]) -> int:
        return json.dumps(body).count(json.dumps(_PROMPT)[1:-1])

    def test_delegated_tool_turn_carries_prompt_once(self) -> None:
        with ScriptedOpenAIServer(_responder) as server:
            v2 = _v2(server)
            v2.initialize("hi")
            v2.generate()
            v1 = _delegating_v1(server)
            _hand_off(v2, v1)
            v1.generate_and_process_with_tools({}, tools_schema=_FINISH_TOOL)
            v1.conversation.append({"role": "user", "content": "again"})
            v1.generate_and_process_with_tools({}, tools_schema=_FINISH_TOOL)
            requests = [r for r in server.requests if r.path.endswith("/responses")]
        # One v2 request before the hand-off, then two delegated tool turns.
        assert len(requests) == 3
        for request in requests[1:]:
            body = request.body
            assert self._prompt_copies(body) == 1, body
            assert len(_system_items(body["input"])) + bool(body.get("instructions")) == 1
            assert body["input"][0] == {"role": "system", "content": _PROMPT}

    def test_fresh_delegated_tool_turn_is_unchanged(self) -> None:
        """Without a hand-off the delegated request already carried the prompt once."""
        with ScriptedOpenAIServer(_responder) as server:
            v1 = _delegating_v1(server)
            v1.initialize("hi")
            v1.generate_and_process_with_tools({}, tools_schema=_FINISH_TOOL)
            request = server.requests[-1]
        assert request.path.endswith("/responses")
        assert self._prompt_copies(request.body) == 1
        assert request.body["input"][0] == {"role": "system", "content": _PROMPT}
        assert not request.body.get("instructions")
