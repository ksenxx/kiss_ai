# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end cost check for Claude on Amazon Bedrock: catalog -> UI event.

A real ``KISSAgent`` runs one turn through the real ``AnthropicBedrock``
client against a local endpoint that answers in Bedrock's binary event
stream, reporting input, output, cache-read and cache-write tokens.  The
tests then compare three things with the published Amazon Bedrock price
table (aws.amazon.com/bedrock/pricing, Anthropic tab, 2026-10):

* ``agent.budget_used`` (what the budget guard and the parent ledger see),
* ``agent.last_call_usage`` (the per-call tokens),
* the ``usage_info`` event's ``cost`` / ``cost_usd`` (what the chat
  header and the task list show), including the string the server
  renders through ``kiss.core.printer.cost_fields``.

No mocks, patches or test doubles: the SDK, sockets, event-stream
decoding, usage extraction, ``calculate_cost`` and the agent's ledger all
run for real.

Run with::

    uv run pytest src/kiss/tests/core/models/test_bedrock_cost_end_to_end.py -v
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import (
    MODEL_INFO,
    _lookup_model_info,
    _seed_new_custom_entry,
    bedrock_catalog_name,
    bedrock_model_id,
    bedrock_price_factor,
    calculate_cost,
)
from kiss.core.printer import Printer, cost_fields
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse
from kiss.tests.core.models.test_bedrock_claude import (
    _AWS_FIELDS,
    _BEDROCK_CONTENT_TYPE,
    bedrock_stream,
)

_INPUT = 100_000
_OUTPUT = 10_000
_CACHE_READ = 400_000
_CACHE_WRITE = 200_000


def _finish_turn(usage: dict[str, Any]) -> list[bytes]:
    """SSE chunks of one ``finish`` tool-use turn whose usage is *usage*.

    ``message_start`` carries the prompt-side counts and the 5-minute /
    1-hour cache-write split exactly as Bedrock's InvokeModel stream
    does; ``message_delta`` carries the final output count.
    """
    return [
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_bedrock_cost",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-under-test",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 1},
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
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


def _usage(
    input_tokens: int = _INPUT,
    output_tokens: int = _OUTPUT,
    cache_read: int = _CACHE_READ,
    cache_write: int = _CACHE_WRITE,
) -> dict[str, Any]:
    """Bedrock usage with all cache writes in the 5-minute bucket."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_write,
            "ephemeral_1h_input_tokens": 0,
        },
    }


class _RecordingPrinter(Printer):
    """Keeps every event the agent emits, as the daemon's printer would."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Record one event."""
        self.events.append({"content": content, "type": type, **kwargs})

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens."""

    def reset(self) -> None:
        """Nothing to reset."""


def _run_bedrock_turn(
    monkeypatch: pytest.MonkeyPatch, model: str, usage: dict[str, Any]
) -> tuple[KISSAgent, dict[str, Any]]:
    """Run one ``finish`` turn of a real agent on Bedrock reporting *usage*.

    Returns:
        The agent and the ``usage_info`` event it emitted.
    """
    agent, event, _ = _run_bedrock_turn_on_wire(monkeypatch, model, usage)
    return agent, event


def _run_bedrock_turn_on_wire(
    monkeypatch: pytest.MonkeyPatch, model: str, usage: dict[str, Any]
) -> tuple[KISSAgent, dict[str, Any], str]:
    """As :func:`_run_bedrock_turn`, also returning the URL path the SDK requested."""
    for field in _AWS_FIELDS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, field, "")
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_BEARER_TOKEN_BEDROCK", "bearer-token")
    printer = _RecordingPrinter()
    with ScriptedAnthropicServer(
        chunks=bedrock_stream(_finish_turn(usage)), content_type=_BEDROCK_CONTENT_TYPE
    ) as server:
        monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", server.base_url)
        agent = KISSAgent("bedrock-cost")
        result = agent.run(
            model_name=model,
            prompt_template="hi",
            max_steps=3,
            max_budget=10.0,
            verbose=False,
            printer=printer,
        )
        assert result == "done"
        assert len(server.requests) == 1
        path = server.request_paths[0]
    usage_events = [e for e in printer.events if e["type"] == "usage_info"]
    assert len(usage_events) == 1
    return agent, usage_events[0], path


def _published(
    input_rate: float,
    output_rate: float,
    cache_read_rate: float,
    cache_write_rate: float,
    usage: dict[str, Any] | None = None,
) -> float:
    """USD for *usage* at the per-million rates of the Bedrock price table."""
    usage = usage or _usage()
    total: float = (
        usage["input_tokens"] * input_rate
        + usage["output_tokens"] * output_rate
        + usage["cache_read_input_tokens"] * cache_read_rate
        + usage["cache_creation_input_tokens"] * cache_write_rate
    )
    return total / 1e6


def _check_event(
    event: dict[str, Any],
    agent: KISSAgent,
    model: str,
    expected: float,
    cache_read: int = _CACHE_READ,
) -> None:
    """The ``usage_info`` event carries the exact and display costs the UI shows."""
    assert event["model"] == model
    assert event["cost_usd"] == pytest.approx(expected)
    assert event["cost"] == f"${expected:.4f}"
    assert event["cache_read"] == cache_read
    display, exact = cost_fields(event["cost"], event["cost_usd"], 0.0)
    assert display == f"${expected:.4f}"
    assert exact == pytest.approx(expected)
    assert agent.budget_used == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Global cross-region profile: the direct Claude price (Opus 4.6:
        # $5 in, $25 out, $0.50 cache read, $6.25 5m cache write).
        ("bedrock/global.anthropic.claude-opus-4-6-v1", _published(5.0, 25.0, 0.50, 6.25)),
        # US geographic profile: Bedrock's 10% premium on every rate.
        ("bedrock/us.anthropic.claude-opus-4-6-v1:0", _published(5.5, 27.5, 0.55, 6.875)),
        # In-region id (no geo prefix) is priced from the same 1.1x table.
        ("bedrock/anthropic.claude-opus-4-6-v1", _published(5.5, 27.5, 0.55, 6.875)),
        # Effort suffix changes nothing about the price.
        ("bedrock/us.anthropic.claude-opus-4-6-v1:0-high", _published(5.5, 27.5, 0.55, 6.875)),
        # Sonnet 5.5 reads cache at 0.05x input ($0.10 global, $0.11 US).
        ("bedrock/global.anthropic.claude-sonnet-5-5", _published(2.0, 10.0, 0.10, 2.50)),
        ("bedrock/us.anthropic.claude-sonnet-5-5", _published(2.2, 11.0, 0.11, 2.75)),
        # Fable 5.1 reads cache at 0.025x input ($0.25 global, $0.275 US).
        ("bedrock/global.anthropic.claude-fable-5-1", _published(10.0, 50.0, 0.25, 12.50)),
        ("bedrock/us.anthropic.claude-fable-5-1", _published(11.0, 55.0, 0.275, 13.75)),
        # Opus 4.1 and older carry no geographic premium on Bedrock.
        ("bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0", _published(15.0, 75.0, 1.50, 18.75)),
    ],
)
def test_bedrock_turn_is_billed_at_the_published_rates(
    monkeypatch: pytest.MonkeyPatch, model: str, expected: float
) -> None:
    """Budget, per-call usage and the UI event all match the Bedrock price table."""
    agent, event = _run_bedrock_turn(monkeypatch, model, _usage())
    usage = agent.last_call_usage
    assert usage is not None
    assert usage["input_tokens"] == _INPUT
    assert usage["output_tokens"] == _OUTPUT
    assert usage["cache_read"] == _CACHE_READ
    assert usage["cache_write"] == _CACHE_WRITE
    assert usage["cost"] == pytest.approx(expected)
    assert agent.total_tokens_used == _INPUT + _OUTPUT + _CACHE_READ + _CACHE_WRITE
    _check_event(event, agent, model, expected)
    assert agent.budget_used == pytest.approx(
        calculate_cost(model, _INPUT, _OUTPUT, _CACHE_READ, _CACHE_WRITE)
    )


@pytest.mark.parametrize(
    ("model", "read", "write", "write_1h"),
    [
        ("bedrock/us.anthropic.claude-sonnet-5-5", 0.11, 2.75, 4.4),
        ("bedrock/us.anthropic.claude-fable-5-1", 0.275, 13.75, 22.0),
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", 0.11, 1.375, 2.2),
        ("bedrock/global.anthropic.claude-opus-5-5", 0.2, 5.0, 8.0),
        ("bedrock/us.anthropic.claude-opus-4-6-v1", 0.55, 6.875, 11.0),
    ],
)
def test_catalog_cache_prices_are_the_published_figures(
    model: str, read: float, write: float, write_1h: float
) -> None:
    """Derived cache rates are the table's figures, not float-noisy products."""
    info = _lookup_model_info(model)
    assert info is not None
    assert (info.cache_read_price_per_1M, info.cache_write_price_per_1M) == (read, write)
    assert info.cache_write_1h_price_per_1M == write_1h


def test_us_profile_costs_exactly_ten_percent_more_than_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The geographic premium is applied once, to the whole turn."""
    global_agent, _ = _run_bedrock_turn(
        monkeypatch, "bedrock/global.anthropic.claude-opus-4-6-v1", _usage()
    )
    us_agent, _ = _run_bedrock_turn(
        monkeypatch, "bedrock/us.anthropic.claude-opus-4-6-v1", _usage()
    )
    assert us_agent.budget_used == pytest.approx(1.1 * global_agent.budget_used)


def test_haiku_5_5_long_prompt_uses_the_long_context_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over 100K prompt tokens (input + cache) Haiku 5.5 bills 5x, then the US premium.

    Bedrock's table: Haiku 5.5 long context on a US profile is $0.55 in,
    $2.75 out, $0.055 cache read, $0.6875 5m cache write.
    """
    usage = _usage(input_tokens=60_000, output_tokens=5_000, cache_read=50_000, cache_write=10_000)
    agent, event = _run_bedrock_turn(monkeypatch, "bedrock/us.anthropic.claude-haiku-5-5", usage)
    expected = _published(0.55, 2.75, 0.055, 0.6875, usage)
    _check_event(event, agent, "bedrock/us.anthropic.claude-haiku-5-5", expected, 50_000)


def test_haiku_5_5_short_prompt_uses_the_standard_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At or below 100K prompt tokens Haiku 5.5 bills the base rates ($0.11 / $0.55 US)."""
    usage = _usage(input_tokens=60_000, output_tokens=5_000, cache_read=30_000, cache_write=10_000)
    agent, event = _run_bedrock_turn(monkeypatch, "bedrock/us.anthropic.claude-haiku-5-5", usage)
    expected = _published(0.11, 0.55, 0.011, 0.1375, usage)
    _check_event(event, agent, "bedrock/us.anthropic.claude-haiku-5-5", expected, 30_000)


def test_cache_write_without_a_ttl_split_bills_the_one_hour_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache write Bedrock does not attribute to a TTL is billed at 2x input, never less."""
    usage = _usage()
    del usage["cache_creation"]
    agent, event = _run_bedrock_turn(
        monkeypatch, "bedrock/global.anthropic.claude-opus-4-6-v1", usage
    )
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["cache_write"] == _CACHE_WRITE
    _check_event(
        event,
        agent,
        "bedrock/global.anthropic.claude-opus-4-6-v1",
        _published(5.0, 25.0, 0.50, 10.0, usage),
    )


_OPUS_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-6-v1"
)


def test_inference_profile_arn_is_priced_and_sent_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ARN-form id bills as its ``us.`` profile while the ARN itself goes on the wire."""
    model = "bedrock/" + _OPUS_ARN + "-high"
    assert bedrock_model_id(model) == _OPUS_ARN
    assert bedrock_catalog_name(model) == "claude-opus-4-6-high"
    agent, event, path = _run_bedrock_turn_on_wire(monkeypatch, model, _usage())
    assert unquote(path) == f"/model/{_OPUS_ARN}/invoke-with-response-stream"
    _check_event(event, agent, model, _published(5.5, 27.5, 0.55, 6.875))


@pytest.mark.parametrize(
    ("model", "catalog_key", "factor"),
    [
        # A geographic prefix the catalog has never seen still names its model.
        ("bedrock/ca.anthropic.claude-sonnet-5-5", "claude-sonnet-5-5", 1.1),
        (
            "bedrock/us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-sonnet-4-5-20250929",
            1.1,
        ),
        # A global ARN carries no premium.
        (
            "bedrock/arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-4-6-v1",
            "claude-opus-4-6",
            1.0,
        ),
    ],
)
def test_any_profile_prefix_resolves_to_its_claude_entry(
    model: str, catalog_key: str, factor: float
) -> None:
    assert bedrock_catalog_name(model) == catalog_key
    assert bedrock_price_factor(model) == factor
    direct = MODEL_INFO[catalog_key]
    info = _lookup_model_info(model)
    assert info is not None
    assert info.input_price_per_1M == pytest.approx(direct.input_price_per_1M * factor)
    assert _seed_new_custom_entry(model)["input_price_per_1M"] == info.input_price_per_1M


@pytest.mark.parametrize(
    ("model", "sibling"),
    [
        # Sonnet 4 is listed only under its global profile; Claude 3 Haiku
        # only under its US profile (both at the base rate, since Bedrock
        # charges no premium on models older than Claude 4.5).
        (
            "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
            "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            "bedrock/eu.anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
        ),
    ],
)
def test_other_profile_of_a_bedrock_only_model_is_priced_by_its_sibling(
    monkeypatch: pytest.MonkeyPatch, model: str, sibling: str
) -> None:
    """A profile the catalog lacks bills like the listed profile of the same model."""
    assert bedrock_catalog_name(model) == sibling
    usage = _usage()
    expected = calculate_cost(sibling, _INPUT, _OUTPUT, _CACHE_READ, _CACHE_WRITE)
    assert expected > 0
    assert (
        _seed_new_custom_entry(model)["input_price_per_1M"]
        == MODEL_INFO[sibling].input_price_per_1M
    )
    agent, event = _run_bedrock_turn(monkeypatch, model, usage)
    _check_event(event, agent, model, expected)


def test_a_claude_model_no_entry_describes_stays_unpriced() -> None:
    """Without any catalog entry the cost is an error, never a silent $0."""
    model = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
    assert bedrock_catalog_name(model) is None
    assert _lookup_model_info(model) is None
    with pytest.raises(KISSError, match="unknown model"):
        calculate_cost(model, 10, 10)
    assert bedrock_catalog_name("bedrock/amazon.nova-pro-v1:0") is None
