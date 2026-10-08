# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the OpenAI processing tier that served a call scales its cost.

OpenAI prices every token rate by processing tier
(https://developers.openai.com/api/docs/pricing): Flex at half the
Standard rate the catalog holds, Fast (``service_tier`` ``fast``, or
``priority`` as GPT-5.6 and earlier report it) at twice, Ultrafast at six
times.  The response's ``service_tier`` names the tier that actually
served the request (a ramp-limited Fast request comes back ``default``),
so ``KISSAgent`` multiplies the catalog estimate by that tier's factor.

No mocks: a real ``ThreadingHTTPServer`` returns genuine Chat Completions
bodies to the real OpenAI SDK, and a real ``KISSAgent`` runs one turn.
"""

from __future__ import annotations

import json

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-6-astra"
_INPUT_TOKENS = 1_000
_OUTPUT_TOKENS = 100
_STANDARD = calculate_cost(_MODEL, _INPUT_TOKENS, _OUTPUT_TOKENS)


def _finish_reply(service_tier: str | None, model: str = _MODEL) -> Reply:
    """A Chat Completions ``finish`` turn served at *service_tier* (omitted when None)."""
    body = {
        "id": "chatcmpl-tier",
        "object": "chat.completion",
        "created": 0,
        "model": model,
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
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({"result": "done"}),
                            },
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
        },
    }
    if service_tier is not None:
        body["service_tier"] = service_tier
    return Reply(json_body=body)


def _run_agent(service_tier: str | None, model: str = _MODEL) -> KISSAgent:
    """Run one ``finish`` turn of a real agent against an endpoint serving *service_tier*."""

    def responder(request: Request) -> Reply:
        """Answer the Chat Completions call with the scripted tier."""
        assert request.path.endswith("/chat/completions"), request.path
        return _finish_reply(service_tier, model)

    with ScriptedOpenAIServer(responder) as server:
        agent = KISSAgent("tier-cost")
        result = agent.run(
            model_name=model,
            prompt_template="hi",
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={"base_url": server.base_url, "api_key": "sk-test"},
        )
    assert result == "done"
    return agent


class TestServedTierScalesTheCatalogEstimate:
    @pytest.mark.parametrize(
        ("service_tier", "factor"),
        [
            (None, 1.0),
            ("default", 1.0),
            ("auto", 1.0),
            ("scale", 1.0),
            ("flex", 0.5),
            ("fast", 2.0),
            ("priority", 2.0),
            ("ultrafast", 6.0),
        ],
    )
    def test_budget_follows_the_served_tier(self, service_tier: str | None, factor: float) -> None:
        agent = _run_agent(service_tier)
        assert agent.budget_used == pytest.approx(_STANDARD * factor)
        assert agent.total_tokens_used == _INPUT_TOKENS + _OUTPUT_TOKENS

    def test_flex_astra_matches_the_published_flex_rates(self) -> None:
        """gpt-6-astra Flex: $5 / $25 per 1M (half of $10 / $50)."""
        agent = _run_agent("flex")
        assert agent.budget_used == pytest.approx((1_000 * 5.0 + 100 * 25.0) / 1e6)


class TestFastTierIsPricedPerModel:
    """Fast is 2x Standard on the GPT-5.x/GPT-6 line but not on older models
    (pricing page, Fast tab): gpt-5.5 $12.50/$75 on $5/$30, gpt-4.1
    $3.50/$14 on $2/$8, gpt-4o $4.25/$17 on $2.50/$10, o3 $3.50/$14 on $2/$8,
    gpt-5-mini $0.45/$3.60 on $0.25/$2, gpt-4o-mini $0.25/$1 on $0.15/$0.60,
    o4-mini $2/$8 on $1.10/$4.40."""

    @pytest.mark.parametrize(
        ("model", "fast_input", "fast_output"),
        [
            ("gpt-5.5", 12.5, 75.0),
            ("gpt-5.4", 5.0, 30.0),
            ("gpt-5.2", 3.5, 28.0),
            ("gpt-5", 2.5, 20.0),
            ("gpt-5-mini", 0.45, 3.6),
            ("gpt-4.1", 3.5, 14.0),
            ("gpt-4.1-mini", 0.7, 2.8),
            ("gpt-4.1-nano", 0.2, 0.8),
            ("gpt-4o", 4.25, 17.0),
            ("gpt-4o-mini", 0.25, 1.0),
            ("o4-mini", 2.0, 8.0),
            ("o3", 3.5, 14.0),
        ],
    )
    def test_fast_matches_the_published_rates(
        self, model: str, fast_input: float, fast_output: float
    ) -> None:
        for tier in ("fast", "priority"):
            agent = _run_agent(tier, model)
            expected = (_INPUT_TOKENS * fast_input + _OUTPUT_TOKENS * fast_output) / 1e6
            assert agent.budget_used == pytest.approx(expected), (model, tier)

    def test_thinking_alias_and_harbor_prefix_share_the_base_ratio(self) -> None:
        model = OpenAICompatibleModel("gpt-5.5-xhigh", base_url="http://127.0.0.1:9", api_key="k")
        assert model.cost_multiplier_for_response({"service_tier": "fast"}) == 2.5
        model = OpenAICompatibleModel("openai/gpt-4.1", base_url="http://127.0.0.1:9", api_key="k")
        assert model.cost_multiplier_for_response({"service_tier": "priority"}) == 1.75


class TestMultiplierShapes:
    def test_non_string_tier_is_standard(self) -> None:
        model = OpenAICompatibleModel(_MODEL, base_url="http://127.0.0.1:9", api_key="k")
        assert model.cost_multiplier_for_response({"service_tier": None}) == 1.0
        assert model.cost_multiplier_for_response({"service_tier": 2}) == 1.0
        assert model.cost_multiplier_for_response({}) == 1.0
        assert model.cost_multiplier_for_response({"service_tier": "flex"}) == 0.5
