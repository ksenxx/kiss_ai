# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Anthropic fast mode and US-only inference scale a call's cost.

The Anthropic API reports two price modifiers in the response ``usage``:
``speed`` is ``"fast"`` when fast mode served the call, which bills every
token category at twice the standard rate
(https://platform.claude.com/docs/en/build-with-claude/fast-mode: Opus 5.5
$8/$40 on $4/$20), and ``inference_geo`` is ``"us"`` when inference was
pinned to US infrastructure, which bills Claude 4.6 and later at 1.1x
(https://platform.claude.com/docs/en/manage-claude/data-residency) and
can be selected by a workspace default the request never mentions.  The
two stack, and prompt-caching multipliers apply on top of both.

No mocks: a real local ``/v1/messages`` server streams genuine Messages
API events to the real Anthropic SDK, and a real ``KISSAgent`` runs one
``finish`` turn against it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import calculate_cost
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse

_MODEL = "claude-opus-5-5"
_USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 100,
    "cache_read_input_tokens": 20_000,
    "cache_creation_input_tokens": 3_000,
    "cache_creation": {"ephemeral_5m_input_tokens": 3_000, "ephemeral_1h_input_tokens": 0},
}
_STANDARD = calculate_cost(_MODEL, 1_000, 100, 20_000, 3_000)


def _finish_turn(usage_extra: dict[str, Any], model: str = _MODEL) -> list[bytes]:
    """SSE chunks of one ``finish`` tool-use turn whose usage carries *usage_extra*."""
    start_usage = {**_USAGE, **usage_extra, "output_tokens": 1}
    return [
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_speed",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            },
        ),
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_finish", "name": "finish"},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"result": "done"}),
                },
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": _USAGE["output_tokens"]},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


def _run_agent(
    monkeypatch: pytest.MonkeyPatch, usage_extra: dict[str, Any], model: str = _MODEL
) -> KISSAgent:
    """Run one ``finish`` turn of a real agent against a server reporting *usage_extra*."""
    with ScriptedAnthropicServer(_finish_turn(usage_extra, model)) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        agent = KISSAgent("speed-geo-cost")
        result = agent.run(
            model_name=model,
            prompt_template="hi",
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={"api_key": "sk-test"},
        )
        assert result == "done"
        assert len(server.requests) == 1
    return agent


class TestServedSpeedAndGeoScaleTheCatalogEstimate:
    @pytest.mark.parametrize(
        ("usage_extra", "factor"),
        [
            ({}, 1.0),
            ({"speed": "standard"}, 1.0),
            ({"inference_geo": "global"}, 1.0),
            ({"speed": "fast"}, 2.0),
            ({"inference_geo": "us"}, 1.1),
            ({"speed": "fast", "inference_geo": "us"}, 2.2),
        ],
    )
    def test_budget_follows_the_reported_modifiers(
        self, monkeypatch: pytest.MonkeyPatch, usage_extra: dict[str, Any], factor: float
    ) -> None:
        agent = _run_agent(monkeypatch, usage_extra)
        assert agent.budget_used == pytest.approx(_STANDARD * factor)
        usage = agent.last_call_usage
        assert usage is not None
        assert usage["input_tokens"] == 1_000
        assert usage["output_tokens"] == 100

    def test_fast_matches_the_published_opus_5_5_rates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fast Opus 5.5: $8 input, $40 output, $0.40 cache read, $10 5m write."""
        agent = _run_agent(monkeypatch, {"speed": "fast"})
        published = (1_000 * 8.0 + 100 * 40.0 + 20_000 * 0.40 + 3_000 * 10.0) / 1e6
        assert agent.budget_used == pytest.approx(published)

    def test_us_geo_on_a_pre_4_6_model_bills_standard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Claude Haiku 4.5 does not support inference_geo; standard pricing applies."""
        model = "claude-haiku-4-5"
        agent = _run_agent(monkeypatch, {"inference_geo": "us"}, model)
        assert agent.budget_used == pytest.approx(calculate_cost(model, 1_000, 100, 20_000, 3_000))


class TestMultiplierOnTheAdapter:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("claude-opus-4-6", 1.1),
            ("claude-sonnet-4-5", 1.0),
            ("claude-opus-4-5-20251101", 1.0),
            ("claude-haiku-5-5", 1.1),
            ("claude-fable-5-1", 1.1),
            ("claude-opus-5-5-high", 1.1),
            ("claude-3-5-haiku-20241022", 1.0),
        ],
    )
    def test_us_geo_applies_from_claude_4_6(self, name: str, expected: float) -> None:
        model = AnthropicModel(name, api_key="sk-test")
        response = {"usage": {"input_tokens": 1, "output_tokens": 1, "inference_geo": "us"}}
        assert model.cost_multiplier_for_response(_as_object(response)) == pytest.approx(expected)

    def test_bedrock_ignores_both_fields(self) -> None:
        model = AnthropicModel("bedrock/anthropic.claude-opus-5-5", api_key="token", bedrock=True)
        usage = {"input_tokens": 1, "output_tokens": 1, "speed": "fast", "inference_geo": "us"}
        assert model.cost_multiplier_for_response(_as_object({"usage": usage})) == 1.0

    def test_a_response_without_usage_bills_standard(self) -> None:
        model = AnthropicModel(_MODEL, api_key="sk-test")
        assert model.cost_multiplier_for_response(_as_object({"usage": None})) == 1.0
        assert model.cost_multiplier_for_response(object()) == 1.0


def _as_object(payload: dict[str, Any]) -> Any:
    """Build an SDK-shaped response (attribute access) from *payload*."""
    from types import SimpleNamespace

    usage = payload.get("usage")
    return SimpleNamespace(usage=None if usage is None else SimpleNamespace(**usage))
