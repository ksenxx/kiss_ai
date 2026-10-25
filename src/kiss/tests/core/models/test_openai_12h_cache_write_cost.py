# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: OpenAI 12-hour prompt-cache writes bill at 2.5x input.

GPT-5.6 and later bill a cache write at 1.25x the input rate for the
default 30-minute lifetime and at 2.5x for the 12-hour lifetime a
Responses request selects with ``prompt_cache_options.ttl="12h"``
(https://developers.openai.com/api/docs/guides/prompt-caching).  The
usage splits the writes in
``input_tokens_details.cache_write_tokens_details`` into
``ttl_30m_tokens`` and ``ttl_12h_tokens``.  Before this fix every write
was billed at the 30-minute rate, under-reporting 12-hour writes by half.

No mocks: a real ``ThreadingHTTPServer`` returns genuine Responses bodies
to the real OpenAI SDK, and a real ``KISSAgent`` runs one turn.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO, calculate_cost
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-6-astra"


def _usage(cached: int, write_30m: int, write_12h: int, ordinary: int) -> dict[str, Any]:
    """Responses usage whose input is split into reads, both write lifetimes and the rest."""
    write = write_30m + write_12h
    return {
        "input_tokens": ordinary + cached + write,
        "input_tokens_details": {
            "cached_tokens": cached,
            "cache_write_tokens": write,
            "cache_write_tokens_details": {
                "ttl_30m_tokens": write_30m,
                "ttl_12h_tokens": write_12h,
            },
        },
        "output_tokens": 100,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": ordinary + cached + write + 100,
    }


def _finish_reply(usage: dict[str, Any]) -> Reply:
    """A Responses turn that calls ``finish`` and reports *usage*."""
    return Reply(
        json_body={
            "id": "resp_12h",
            "object": "response",
            "created_at": 0,
            "model": _MODEL,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "finish",
                    "arguments": json.dumps({"result": "done"}),
                    "status": "completed",
                }
            ],
            "usage": usage,
        }
    )


def test_agent_bills_12h_writes_at_two_and_a_half_times_input() -> None:
    """The documented example usage, priced at gpt-6-astra's published rates."""
    seen: list[Request] = []

    def responder(request: Request) -> Reply:
        """Record the request and answer with the documented usage mix."""
        seen.append(request)
        return _finish_reply(_usage(cached=2_000, write_30m=3_000, write_12h=4_000, ordinary=1_000))

    agent = KISSAgent("openai-12h-cache")
    with ScriptedOpenAIServer(responder) as server:
        result = agent.run(
            model_name=_MODEL,
            prompt_template="hi",
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={
                "base_url": server.base_url,
                "api_key": "sk-test",
                "use_responses_api": True,
                "prompt_cache_options": {"ttl": "12h"},
            },
        )
    assert result == "done"
    assert seen[0].path.endswith("/responses")
    assert seen[0].body["prompt_cache_options"] == {"ttl": "12h"}
    # $10 input, $1 read, $12.50 30m write, $25 12h write, $50 output per MTok.
    expected = (1_000 * 10 + 2_000 * 1 + 3_000 * 12.5 + 4_000 * 25 + 100 * 50) / 1e6
    assert agent.budget_used == pytest.approx(expected)
    assert agent.total_tokens_used == 10_100


def test_extractor_splits_and_caps_the_12h_share() -> None:
    """``ttl_12h_tokens`` moves to the long-lifetime slot, never above the write total."""
    model = OpenAICompatibleModel2(_MODEL, base_url="http://127.0.0.1:9", api_key="k")
    assert model.extract_input_output_token_counts_from_response(
        {"usage": _usage(cached=2_000, write_30m=3_000, write_12h=4_000, ordinary=1_000)}
    ) == (1_000, 100, 2_000, 3_000, 4_000)
    malformed = _usage(cached=0, write_30m=0, write_12h=500, ordinary=10)
    malformed["input_tokens_details"]["cache_write_tokens"] = 200
    malformed["input_tokens"] = 210
    assert model.extract_input_output_token_counts_from_response({"usage": malformed}) == (
        10,
        100,
        0,
        0,
        200,
    )


@pytest.mark.parametrize(
    ("name", "write_12h"),
    [
        ("gpt-6-astra", 25.0),
        ("gpt-6.1-sol", 5.0),
        ("gpt-6-luna", 0.25),
        ("gpt-5.6-sol", 10.0),
        ("gpt-6-astra-high", 25.0),
        ("openrouter/openai/gpt-6-astra", 25.0),
    ],
)
def test_12h_write_rates_match_the_pricing_page(name: str, write_12h: float) -> None:
    """The derived long-lifetime write rate equals OpenAI's published 12h column."""
    assert MODEL_INFO[name].cache_write_1h_price_per_1M == pytest.approx(write_12h)


def test_models_without_write_billing_have_no_12h_tier() -> None:
    """Pre-GPT-5.6 models bill no cache writes, so no long-lifetime rate is derived."""
    assert MODEL_INFO["gpt-5.5"].cache_write_price_per_1M == 0.0
    assert MODEL_INFO["gpt-5.5"].cache_write_1h_price_per_1M is None


def test_long_context_doubles_the_12h_write_rate() -> None:
    """Above 272K prompt tokens the 12h write rate doubles with input ($50 for astra)."""
    short = calculate_cost(_MODEL, 0, 0, num_cache_write_1h_tokens=100_000) / 0.1
    long = calculate_cost(_MODEL, 0, 0, num_cache_write_1h_tokens=300_000) / 0.3
    assert short == pytest.approx(25.0)
    assert long == pytest.approx(50.0)
