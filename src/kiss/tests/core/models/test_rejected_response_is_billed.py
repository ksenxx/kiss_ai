# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a billed response the adapter rejects still counts as spend.

Bug: the OpenAI adapters raise on a completion truncated at the output
token limit (Chat ``finish_reason="length"``, Responses
``status="incomplete"``) so its half-written tool arguments are never
used.  The provider bills that call anyway, but ``KISSAgent._execute_step``
only accounted responses that were returned, so every truncated turn
(and its retry) vanished from the task's cost.

The agent test runs a real ``KISSAgent`` against a scripted local
endpoint that answers with a truncated response reporting OpenRouter's
``usage.cost``.  Truncation is not retryable, so the run fails, but its
``budget_used`` must still hold that billed cost (it was 0 before the fix).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)
from kiss.tests.core.models.test_openrouter_reported_cost import (
    _OPENROUTER_MODEL,
    _finish_reply,
)

_TRUNCATED_COST = 0.004
_FINISH_COST = 0.0005


def _truncated_reply(request: Request) -> Reply:
    """A billed response cut off at the output-token limit, on either transport."""
    usage = {"cost": _TRUNCATED_COST}
    if request.path.endswith("/responses"):
        return Reply(
            json_body={
                "id": "resp_trunc",
                "object": "response",
                "created_at": 0,
                "model": "m",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
                "usage": {
                    "input_tokens": 700,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 300,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1000,
                    **usage,
                },
            }
        )
    return Reply(
        json_body={
            "id": "chatcmpl-trunc",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "partial"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": 300,
                "total_tokens": 1000,
                **usage,
            },
        }
    )


def _run_failing_agent(
    responder: Callable[[Request], Reply], extra_config: dict[str, Any]
) -> tuple[KISSAgent, int]:
    """Run an agent whose every model call gets a billed but unusable reply.

    Args:
        responder: Builds the scripted reply for each request.
        extra_config: Model configuration merged into the endpoint's.

    Returns:
        The failed agent and the number of model requests it made.
    """
    calls: list[str] = []

    def counting(request: Request) -> Reply:
        """Count the request and answer it with *responder*."""
        calls.append(request.path)
        return responder(request)

    agent = KISSAgent("rejected-billing")
    with ScriptedOpenAIServer(counting) as server:
        with pytest.raises(KISSError):
            agent.run(
                model_name=_OPENROUTER_MODEL,
                prompt_template="hi",
                max_steps=4,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": server.base_url,
                    "api_key": "sk-test",
                    **extra_config,
                },
            )
    return agent, len(calls)


@pytest.mark.parametrize(
    "extra_config",
    [{}, {"use_responses_api": True}],
    ids=["chat", "responses"],
)
def test_truncated_response_is_billed_when_the_run_fails(
    extra_config: dict[str, Any],
) -> None:
    agent, calls = _run_failing_agent(_truncated_reply, extra_config)
    assert calls >= 1
    assert agent.budget_used == pytest.approx(calls * _TRUNCATED_COST)
    assert agent.total_tokens_used == calls * 1000


def _no_choices_reply(request: Request) -> Reply:
    """A gateway error delivered as HTTP 200 with no choices but billed usage."""
    return Reply(
        json_body={
            "id": "chatcmpl-err",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [],
            "error": {"message": "upstream overloaded"},
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": 300,
                "total_tokens": 1000,
                "cost": _TRUNCATED_COST,
            },
        }
    )


def test_no_choices_response_usage_is_billed() -> None:
    agent, calls = _run_failing_agent(_no_choices_reply, {})
    assert calls >= 1
    assert agent.budget_used == pytest.approx(calls * _TRUNCATED_COST)



def _streamed_truncated_reply(request: Request) -> Reply:
    """A streamed Chat Completions answer truncated with ``finish_reason="length"``."""
    base = {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    delta = {"index": 0, "delta": {"role": "assistant", "content": "part"}}
    return Reply(
        sse_chunks=[
            chat_chunk({**base, "choices": [{**delta, "finish_reason": None}]}),
            chat_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}),
            chat_chunk(
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 700,
                        "completion_tokens": 300,
                        "total_tokens": 1000,
                        "cost": _TRUNCATED_COST,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


def test_streamed_truncated_response_is_billed() -> None:
    """The streaming Chat Completions path (token callback set) bills it too."""
    tokens: list[str] = []
    calls: list[str] = []

    def responder(request: Request) -> Reply:
        """Stream a truncated answer first, then a streamed-or-plain finish."""
        calls.append(request.path)
        if len(calls) == 1:
            return _streamed_truncated_reply(request)
        return _finish_reply(request, {"cost": _FINISH_COST})

    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel(
            _OPENROUTER_MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            token_callback=tokens.append,
        )
        model.initialize("hi")
        with pytest.raises(KISSError, match="truncated"):
            model.generate()
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_TRUNCATED_COST)
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (700, 300)
    assert model.take_partial_usage_response() is None


def _incomplete_response_body() -> dict:
    """The Responses-API body of the truncated call (its usage was billed)."""
    request = Request(path="/v1/responses", body={}, connection_key="test")
    body = _truncated_reply(request).json_body
    assert isinstance(body, dict)
    return body


def test_non_streamed_incomplete_responses_call_is_kept() -> None:
    def responder(request: Request) -> Reply:
        """Answer every Responses call with a truncated, billed body."""
        return _truncated_reply(request)

    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel2(
            _OPENROUTER_MODEL, base_url=server.base_url, api_key="sk-test"
        )
        model.initialize("hi")
        with pytest.raises(KISSError, match="incomplete"):
            model.generate()
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_TRUNCATED_COST)
    assert model.take_partial_usage_response() is None


@pytest.mark.parametrize(
    ("event_type", "match"),
    [("response.incomplete", "incomplete"), ("response.failed", "response.failed")],
)
def test_streamed_terminal_failure_keeps_the_usage(event_type: str, match: str) -> None:
    body = _incomplete_response_body()
    if event_type == "response.failed":
        body = {**body, "status": "failed", "error": {"code": "x", "message": "boom"}}

    def responder(request: Request) -> Reply:
        """Stream a Responses call that ends in *event_type*."""
        return Reply(sse_chunks=[responses_event(event_type, {"response": body})])

    tokens: list[str] = []
    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel2(
            _OPENROUTER_MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            token_callback=tokens.append,
        )
        model.initialize("hi")
        with pytest.raises(KISSError, match=match):
            model.generate()
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_TRUNCATED_COST)
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (700, 300)


def _echo(text: str) -> str:
    """Echo *text* (a tool so the request carries a tools schema)."""
    return text


def test_chat_adapter_hands_over_its_delegates_rejected_response() -> None:
    """Tools + ``reasoning_effort`` on a Responses-capable endpoint make the
    Chat adapter delegate the turn to an internal Responses model; the
    agent drains the Chat adapter's hook, so the delegate's stash must be
    moved there."""
    with ScriptedOpenAIServer(_truncated_reply) as server:
        model = OpenAICompatibleModel(
            _OPENROUTER_MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"use_responses_api": True, "reasoning_effort": "high"},
        )
        model.initialize("hi")
        with pytest.raises(KISSError, match="incomplete"):
            model.generate_and_process_with_tools({"_echo": _echo})
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_TRUNCATED_COST)
    assert model.take_partial_usage_response() is None


def _bad_request(request: Request) -> Reply:
    """Reject every request with HTTP 400: nothing was generated or billed."""
    return Reply(status=400, json_body={"error": {"message": "bad request"}})


def test_delegate_http_error_leaves_nothing_to_bill() -> None:
    with ScriptedOpenAIServer(_bad_request) as server:
        model = OpenAICompatibleModel(
            _OPENROUTER_MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"use_responses_api": True, "reasoning_effort": "high"},
        )
        model.initialize("hi")
        with pytest.raises(Exception, match="bad request"):
            model.generate_and_process_with_tools({"_echo": _echo})
    assert model.take_partial_usage_response() is None
