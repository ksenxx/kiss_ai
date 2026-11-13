# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C3/F3): a blank tool result must still answer its tool call.

``OpenAICompatibleModel._normalize_message_for_api`` drops a message
whose string content is blank unless it declares ``tool_calls``, without
exempting ``role="tool"``.  A tool that legitimately produced nothing
(``{"result": ""}``, or whitespace) is stored in the conversation but
removed from every request built from it: the direct Chat Completions
request then lacks the tool message the assistant's ``tool_calls`` must
be followed by, the ``/v1/responses`` delegate refuses the next turn
locally as an unanswered ``function_call``, and the same Chat-format
conversation handed to a Responses model loses the output too.  Under
plain I/O (no usage footer) and for public adapter use nothing makes
such a result nonempty.

Every case drives the real OpenAI SDK against a real local endpoint; the
tool call comes from the server and the evidence is the next request's
body (or, before the fix, the local refusal that preceded any request).
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.models.model import hand_off_conversation
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-audit1101-blank"
_CALL_ID = "call_quiet"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "quiet",
            "description": "Produces no output.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_BLANK_RESULTS = ["", " \n "]


def quiet() -> str:
    """Test-only tool stub that produces nothing."""
    return ""


def _usage() -> dict[str, int]:
    return {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def _responses_reply(request: Request) -> Reply:
    """``/v1/responses``: one ``quiet`` call, then a plain answer once it is answered."""
    answered = any(
        item.get("type") == "function_call_output" for item in request.body.get("input", [])
    )
    output: list[dict[str, Any]] = (
        [
            {
                "type": "message",
                "id": "msg_done",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ]
        if answered
        else [
            {
                "type": "function_call",
                "id": "fc_quiet",
                "call_id": _CALL_ID,
                "name": "quiet",
                "arguments": "{}",
            }
        ]
    )
    return Reply(
        json_body={
            "id": "resp_blank",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": _MODEL,
            "output": output,
            "usage": _usage(),
        }
    )


def _chat_reply(request: Request) -> Reply:
    """``/v1/chat/completions``: one ``quiet`` call, then a plain answer once it is answered."""
    answered = any(msg.get("role") == "tool" for msg in request.body.get("messages", []))
    message: dict[str, Any] = (
        {"role": "assistant", "content": "done"}
        if answered
        else {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _CALL_ID,
                    "type": "function",
                    "function": {"name": "quiet", "arguments": "{}"},
                }
            ],
        }
    )
    return Reply(
        json_body={
            "id": "chatcmpl-blank",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop" if answered else "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _function_call_outputs(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (item["call_id"], item["output"])
        for item in body["input"]
        if item.get("type") == "function_call_output"
    ]


def _tool_turn(model: OpenAICompatibleModel, result: str) -> None:
    """Run the tool turn on *model* and answer its call with *result*."""
    calls, _, _ = model.generate_and_process_with_tools({"quiet": quiet}, tools_schema=_TOOLS)
    assert [call["name"] for call in calls] == ["quiet"]
    model.add_function_results_to_conversation_and_return(
        [("quiet", {"result": result, "tool_use_id": _CALL_ID})]
    )
    assert model.conversation[-1] == {"role": "tool", "tool_call_id": _CALL_ID, "content": result}


@pytest.mark.parametrize("result", _BLANK_RESULTS, ids=["empty", "whitespace"])
def test_delegated_responses_turn_carries_the_blank_output(result: str) -> None:
    """v1 delegating tools to ``/v1/responses``: the next turn answers the call."""
    with ScriptedOpenAIServer(_responses_reply) as server:
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="k",
            model_config={"use_responses_api": True, "reasoning_effort": "high"},
        )
        model.initialize("Run quiet, then continue.")
        _tool_turn(model, result)
        calls, content, _ = model.generate_and_process_with_tools(
            {"quiet": quiet}, tools_schema=_TOOLS
        )
        assert (calls, content) == ([], "done")
        assert len(server.requests) == 2
        assert _function_call_outputs(server.requests[1].body) == [(_CALL_ID, result)]


@pytest.mark.parametrize("result", _BLANK_RESULTS, ids=["empty", "whitespace"])
def test_chat_completions_request_keeps_the_blank_tool_message(result: str) -> None:
    """Direct Chat Completions: the tool message follows the assistant's ``tool_calls``."""
    with ScriptedOpenAIServer(_chat_reply) as server:
        model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Run quiet, then continue.")
        _tool_turn(model, result)
        calls, content, _ = model.generate_and_process_with_tools(
            {"quiet": quiet}, tools_schema=_TOOLS
        )
        assert (calls, content) == ([], "done")
        assert len(server.requests) == 2
        messages = server.requests[1].body["messages"]
        assert messages[-1] == {"role": "tool", "tool_call_id": _CALL_ID, "content": result}
        assert messages[-2]["tool_calls"][0]["id"] == _CALL_ID


@pytest.mark.parametrize("result", _BLANK_RESULTS, ids=["empty", "whitespace"])
def test_chat_conversation_handed_to_responses_model_keeps_the_output(result: str) -> None:
    """The Chat-format conversation handed to a Responses model answers the call too."""
    with ScriptedOpenAIServer(_chat_reply) as chat, ScriptedOpenAIServer(_responses_reply) as rsp:
        old = OpenAICompatibleModel(_MODEL, base_url=chat.base_url, api_key="k")
        old.initialize("Run quiet, then continue.")
        _tool_turn(old, result)
        new = OpenAICompatibleModel2(_MODEL, base_url=rsp.base_url, api_key="k")
        new.initialize("Run quiet, then continue.")
        hand_off_conversation(old, new)
        assert new.generate()[0] == "done"
        assert len(rsp.requests) == 1
        assert _function_call_outputs(rsp.requests[0].body) == [(_CALL_ID, result)]


@pytest.mark.parametrize("delegated", [False, True], ids=["chat", "responses"])
def test_tool_result_with_lone_surrogates_still_reaches_the_request(delegated: bool) -> None:
    """A tool result no UTF-8 body can carry is repaired instead of failing every request.

    ``json.loads`` turns a ``"\\ud83d"`` escape into a lone surrogate.  The
    SDK raised ``UnicodeEncodeError`` while building the body, before
    anything was sent, and again for every later turn, since the result
    stays in the conversation.  The body now carries well-formed text: a
    pair split into two code points joined, a lone half as U+FFFD.
    """
    config = {"use_responses_api": True, "reasoning_effort": "high"} if delegated else {}
    with ScriptedOpenAIServer(_responses_reply if delegated else _chat_reply) as server:
        model = OpenAICompatibleModel(
            _MODEL, base_url=server.base_url, api_key="k", model_config=config
        )
        model.initialize("Run quiet, then continue.")
        _tool_turn(model, "caf\ud800 \ud83d\ude00")
        calls, content, _ = model.generate_and_process_with_tools(
            {"quiet": quiet}, tools_schema=_TOOLS
        )
        assert (calls, content) == ([], "done")
        body = server.requests[1].body
        sent = _function_call_outputs(body)[0][1] if delegated else body["messages"][-1]["content"]
        assert sent == "caf\ufffd \U0001f600"
