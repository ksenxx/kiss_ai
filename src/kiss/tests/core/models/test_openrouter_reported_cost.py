# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: OpenRouter's reported ``usage.cost`` is what the agent bills.

OpenRouter attaches ``usage.cost`` (credits charged, USD) and
``usage.cost_details.upstream_inference_cost`` to every response and to
the final usage chunk of a stream
(https://openrouter.ai/docs/cookbook/administration/usage-accounting).
The same model id is billed at different rates depending on which
upstream OpenRouter routes to (a live audit measured -46% to +19%
between the catalog estimate and the actual charge on multi-provider
models), so for ``openrouter/*`` models the agent must bill the reported
figure and use ``calculate_cost`` only when the field is absent.  Other
providers never send the field and keep the catalog estimate.

No mocks: a real ``ThreadingHTTPServer`` returns the genuine JSON and SSE
bodies, to the real OpenAI SDK, and a real ``KISSAgent`` runs one turn.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_OPENROUTER_MODEL = "openrouter/deepseek/deepseek-v4-flash"
_DIRECT_MODEL = "gpt-5.4"
_INPUT_TOKENS = 1_000
_OUTPUT_TOKENS = 100
_FINISH_TOOL = [
    {
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
]


def _finish_reply(request: Request, usage_extra: dict[str, Any]) -> Reply:
    """A turn that calls ``finish`` and reports *usage_extra*, on either transport.

    Args:
        request: The recorded request; its path selects the Chat
            Completions or Responses body shape.
        usage_extra: Fields merged into the standard ``usage`` block
            (OpenRouter's ``cost`` / ``cost_details``).

    Returns:
        The scripted reply.
    """
    arguments = json.dumps({"result": "done"})
    if request.path.endswith("/responses"):
        return Reply(
            json_body={
                "id": "resp_cost",
                "object": "response",
                "created_at": 0,
                "model": "deepseek/deepseek-v4-flash",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_finish",
                        "name": "finish",
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
                "usage": {
                    "input_tokens": _INPUT_TOKENS,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": _OUTPUT_TOKENS,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                    **usage_extra,
                },
            }
        )
    return Reply(
        json_body={
            "id": "chatcmpl-cost",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_finish",
                                "type": "function",
                                "function": {"name": "finish", "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": _INPUT_TOKENS,
                "completion_tokens": _OUTPUT_TOKENS,
                "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                **usage_extra,
            },
        }
    )


def _run_agent(model_name: str, usage_extra: dict[str, Any]) -> KISSAgent:
    """Run one ``finish`` turn of a real agent against a scripted endpoint.

    Args:
        model_name: The catalog model name to run under.
        usage_extra: Extra ``usage`` fields the endpoint reports.

    Returns:
        The finished agent, whose ``budget_used`` is under test.
    """

    def responder(request: Request) -> Reply:
        """Answer every call with the scripted ``finish`` turn."""
        return _finish_reply(request, usage_extra)

    with ScriptedOpenAIServer(responder) as server:
        agent = KISSAgent("reported-cost")
        result = agent.run(
            model_name=model_name,
            prompt_template="hi",
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={"base_url": server.base_url, "api_key": "sk-test"},
        )
    assert result == "done"
    return agent


class TestAgentBillsReportedCost:
    """``KISSAgent.budget_used`` follows ``usage.cost`` for OpenRouter models."""

    def test_openrouter_reported_cost_replaces_the_catalog_estimate(self) -> None:
        estimate = calculate_cost(_OPENROUTER_MODEL, _INPUT_TOKENS, _OUTPUT_TOKENS)
        reported = 0.00123
        assert reported != pytest.approx(estimate)
        agent = _run_agent(
            _OPENROUTER_MODEL,
            {"cost": reported, "cost_details": {"upstream_inference_cost": 0}},
        )
        assert agent.budget_used == pytest.approx(reported)
        assert agent.total_tokens_used == _INPUT_TOKENS + _OUTPUT_TOKENS

    def test_byok_upstream_inference_cost_is_added(self) -> None:
        """Under BYOK the upstream bills the user directly; both parts are spend."""
        agent = _run_agent(
            _OPENROUTER_MODEL,
            {"cost": 0.0001, "cost_details": {"upstream_inference_cost": 0.002}},
        )
        assert agent.budget_used == pytest.approx(0.0021)

    def test_missing_cost_field_falls_back_to_the_catalog(self) -> None:
        agent = _run_agent(_OPENROUTER_MODEL, {})
        assert agent.budget_used == pytest.approx(
            calculate_cost(_OPENROUTER_MODEL, _INPUT_TOKENS, _OUTPUT_TOKENS)
        )

    def test_non_openrouter_model_ignores_a_cost_field(self) -> None:
        """Only OpenRouter documents ``usage.cost``; a direct provider's
        extra field of the same name is not trusted over the catalog."""
        agent = _run_agent(_DIRECT_MODEL, {"cost": 42.0})
        assert agent.budget_used == pytest.approx(
            calculate_cost(_DIRECT_MODEL, _INPUT_TOKENS, _OUTPUT_TOKENS)
        )


def _chat_stream_reply(usage_extra: dict[str, Any]) -> Reply:
    """A streamed Chat Completions answer whose final chunk carries usage.

    Args:
        usage_extra: Fields merged into the final chunk's ``usage``.

    Returns:
        The scripted SSE reply.
    """
    base = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 0, "model": "m"}
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
            chat_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            chat_chunk(
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 2,
                        "total_tokens": 32,
                        "prompt_tokens_details": {"cached_tokens": 10},
                        **usage_extra,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


class TestStreamedChatCompletionsCarryTheCost:
    """The v1 transport reads the cost off the final usage chunk of a stream."""

    def test_streamed_usage_chunk_cost_is_extracted(self) -> None:
        def responder(request: Request) -> Reply:
            """Stream a two-token answer with OpenRouter's usage tail."""
            return _chat_stream_reply(
                {"cost": 0.00042, "cost_details": {"upstream_inference_cost": None}}
            )

        with ScriptedOpenAIServer(responder) as server:
            model = OpenAICompatibleModel(
                _OPENROUTER_MODEL,
                base_url=server.base_url,
                api_key="sk-test",
                token_callback=lambda _t: None,
            )
            model.initialize("hi")
            content, response = model.generate()
        assert content == "ok"
        assert model.extract_input_output_token_counts_from_response(response) == (20, 2, 10, 0)
        assert model.extract_cost_from_response(response) == pytest.approx(0.00042)

    def test_streamed_usage_without_cost_yields_none(self) -> None:
        def responder(request: Request) -> Reply:
            """Stream the same answer from a provider that reports no cost."""
            return _chat_stream_reply({})

        with ScriptedOpenAIServer(responder) as server:
            model = OpenAICompatibleModel(
                _OPENROUTER_MODEL,
                base_url=server.base_url,
                api_key="sk-test",
                token_callback=lambda _t: None,
            )
            model.initialize("hi")
            _content, response = model.generate()
        assert model.extract_cost_from_response(response) is None


class TestCostFieldShapes:
    """Dict-form usage (Responses-delegate path) and malformed values."""

    def _model(self) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(_OPENROUTER_MODEL, base_url="http://127.0.0.1:9", api_key="k")

    def test_dict_response_is_read(self) -> None:
        response = {"usage": {"cost": 0.5, "cost_details": {"upstream_inference_cost": 0.25}}}
        assert self._model().extract_cost_from_response(response) == pytest.approx(0.75)

    def test_non_numeric_cost_is_ignored(self) -> None:
        model = self._model()
        assert model.extract_cost_from_response({"usage": {"cost": "0.5"}}) is None
        assert model.extract_cost_from_response({"usage": {"cost": True}}) is None
        assert model.extract_cost_from_response({"usage": None}) is None
        assert model.extract_cost_from_response({}) is None

    def test_non_numeric_upstream_cost_counts_as_zero(self) -> None:
        response = {"usage": {"cost": 0.5, "cost_details": {"upstream_inference_cost": "n/a"}}}
        model = self._model()
        assert model.extract_cost_from_response(response) == pytest.approx(0.5)
        assert model.extract_cost_from_response({"usage": {"cost": 0.5}}) == pytest.approx(0.5)


class TestResponsesTransportCarriesTheCost:
    """The v2 (Responses API) transport shares the same extraction."""

    def test_responses_body_cost_is_extracted(self) -> None:
        def responder(request: Request) -> Reply:
            """Answer the Responses call with OpenRouter's usage block."""
            assert request.path.endswith("/responses")
            return _finish_reply(
                request, {"cost": 0.0031, "cost_details": {"upstream_inference_cost": 0}}
            )

        with ScriptedOpenAIServer(responder) as server:
            model = OpenAICompatibleModel2(
                _OPENROUTER_MODEL, base_url=server.base_url, api_key="sk-test"
            )
            model.initialize("hi")
            function_calls, _content, response = model.generate_and_process_with_tools(
                {}, tools_schema=_FINISH_TOOL
            )
        assert function_calls[0]["name"] == "finish"
        assert model.extract_input_output_token_counts_from_response(response)[:2] == (
            _INPUT_TOKENS,
            _OUTPUT_TOKENS,
        )
        assert model.extract_cost_from_response(response) == pytest.approx(0.0031)
