# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Gemini call is billed at the rate of the service tier that served it.

Google prices Priority at 1.8x and Flex at 0.5x of the Standard rates
the catalog holds (https://ai.google.dev/gemini-api/docs/pricing, read
2026-11-19), and reports the tier that served a streamed call only in
the body, as ``usageMetadata.serviceTier``, which the SDK drops.  Before
this, ``model_config={"service_tier": ...}`` was forwarded and every such
call was recorded at the Standard rate: Flex at twice what Google
charged, Priority at 56% of it.

Every test drives the real SDK, the real httpx transport and the real
adapter against a local endpoint (:mod:`gemini_sse_harness`); the
expected amounts are the cells of Google's page.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import httpx
import pytest
from google.genai import types

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.gemini_model import GeminiModel, _ServiceTierTap
from kiss.core.models.model_info import calculate_cost
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    function_call_part,
    serve,
    text_part,
)

_M = 1_000_000
# 1M uncached input, 1M cached input and 1M output tokens.
_USAGE = {"promptTokenCount": 2 * _M, "cachedContentTokenCount": _M, "candidatesTokenCount": _M}
# The same below gemini-2.5-pro's 200K long-prompt threshold, x 10.
_SHORT = {"promptTokenCount": _M // 5, "cachedContentTokenCount": _M // 10,
          "candidatesTokenCount": _M // 10}


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _billed(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[str, GeminiScript],
    name: str,
    usage: dict[str, Any],
    asked: str | None = None,
    parts: list[dict[str, Any]] | None = None,
    **framing: Any,
) -> float:
    """Return what the agent records for one streamed turn of model *name*.

    Args:
        usage: The turn's ``usageMetadata``, ``serviceTier`` included when
            the endpoint reports the tier that served it.
        asked: The ``service_tier`` of ``model_config``, if any.
        parts: The answer's parts (default: the text ``OK``).
        framing: How the endpoint frames the stream (``GeminiScript.play``).
    """
    base_url, script = endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        name, api_key="test-key", model_config={"service_tier": asked} if asked else None
    )
    model.initialize("Say OK.")
    last = chunk(parts or [text_part("K")], usage=usage, finish_reason="STOP")
    script.play([chunk([text_part("O")]), last], **framing)
    _, content, response = model.generate_and_process_with_tools({})
    assert parts or content == "OK"  # the stream itself is read as before
    assert script.requests[-1].get("serviceTier") == asked  # the tier is asked for on the wire
    agent = KISSAgent("gemini-service-tier")
    agent.model = model
    agent.budget_used = 0.0
    agent.total_tokens_used = 0
    agent._update_tokens_and_budget_from_response(response)
    return agent.budget_used


@pytest.mark.parametrize(
    ("name", "usage", "served", "dollars"),
    [
        # Gemini 4 Carbon: $2.00 / $0.10 / $10.00; Flex and Priority scale all three.
        ("gemini-4-carbon", _USAGE, "standard", 2.00 + 0.10 + 10.00),
        ("gemini-4-carbon", _USAGE, "flex", 1.00 + 0.05 + 5.00),
        ("gemini-4-carbon", _USAGE, "priority", 3.60 + 0.18 + 18.00),
        ("gemini-4-carbon-low", _USAGE, "priority", 3.60 + 0.18 + 18.00),
        # Gemini 2.5 Pro up to 200K: $1.25 / $0.125 / $10.00; Flex keeps the cached price.
        ("gemini-2.5-pro", _SHORT, "standard", (1.25 + 0.125 + 10.00) / 10),
        ("gemini-2.5-pro", _SHORT, "flex", (0.625 + 0.125 + 5.00) / 10),
        ("gemini-2.5-pro", _SHORT, "priority", (2.25 + 0.225 + 18.00) / 10),
        # ... and above 200K: $2.50 / $0.25 / $15.00.
        ("gemini-2.5-pro", _USAGE, "flex", 1.25 + 0.25 + 7.50),
        ("gemini-2.5-pro", _USAGE, "priority", 4.50 + 0.45 + 27.00),
        # Gemini 3.1 Pro Preview above 200K ($4.00 / $0.40 / $18.00): cached "Same as Standard".
        ("gemini-3.1-pro-preview", _USAGE, "flex", 2.00 + 0.40 + 9.00),
        # Gemini 3 Flash Preview ($0.50 / $0.05 / $3.00): the same on Flex, x 1.8 on Priority.
        ("gemini-3-flash-preview", _USAGE, "flex", 0.25 + 0.05 + 1.50),
        ("gemini-3-flash-preview", _USAGE, "priority", 0.90 + 0.09 + 5.40),
        # Gemini 3.5 Flash-Lite: $0.30 / $0.03 / $2.50; cached is $0.02 on Flex, $0.05 on Priority.
        ("gemini-3.5-flash-lite", _USAGE, "standard", 0.30 + 0.03 + 2.50),
        ("gemini-3.5-flash-lite", _USAGE, "flex", 0.15 + 0.02 + 1.25),
        ("gemini-3.5-flash-lite", _USAGE, "priority", 0.54 + 0.05 + 4.50),
        ("gemini-3.5-flash-lite-high", _USAGE, "flex", 0.15 + 0.02 + 1.25),
    ],
)
def test_the_served_tier_sets_the_price(
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
    name: str,
    usage: dict[str, Any],
    served: str,
    dollars: float,
) -> None:
    """The amount recorded is the page's price for the tier the body reports."""
    usage = dict(usage, serviceTier=served)
    cost = _billed(monkeypatch, gemini_endpoint, name, usage, asked=served)
    assert cost == pytest.approx(dollars)


def test_a_priority_request_served_as_standard_bills_standard(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    """Google downgrades Priority traffic over its limit, and says so in the response."""
    usage = dict(_USAGE, serviceTier="standard")
    cost = _billed(monkeypatch, gemini_endpoint, "gemini-4-carbon", usage, asked="priority")
    assert cost == pytest.approx(2.00 + 0.10 + 10.00)


@pytest.mark.parametrize("newline", ["\r\n", "\n", "\r"])
@pytest.mark.parametrize("pretty", [False, True])
def test_the_tier_is_read_however_the_stream_is_framed(
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
    pretty: bool,
    newline: str,
) -> None:
    """An event of several ``data:`` lines, any of the three line ends: the SDK reads them all."""
    for served, asked, dollars in (("standard", "priority", 12.10), ("flex", None, 6.05)):
        usage = dict(_USAGE, serviceTier=served)
        cost = _billed(
            monkeypatch, gemini_endpoint, "gemini-4-carbon", usage, asked=asked,
            pretty=pretty, newline=newline,
        )
        assert cost == pytest.approx(dollars)


@pytest.mark.parametrize(
    ("asked", "dollars"),
    [("flex", 6.05), ("PRIORITY", 21.78), (types.ServiceTier.FLEX, 6.05), (None, 12.10)],
)
def test_a_response_naming_no_tier_bills_the_tier_asked_for(
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
    asked: Any,
    dollars: float,
) -> None:
    """Without ``serviceTier`` in the body the request's own tier is the estimate."""
    base_url, script = gemini_endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        "gemini-4-carbon", api_key="k", model_config={"service_tier": asked} if asked else None
    )
    model.initialize("Say OK.")
    script.play([chunk([text_part("OK")], usage=_USAGE, finish_reason="STOP")])
    _, _, response = model.generate_and_process_with_tools({})
    counts = model.extract_input_output_token_counts_from_response(response)
    cost = calculate_cost("gemini-4-carbon", *counts) * model.cost_multiplier_for_response(response)
    assert cost == pytest.approx(dollars)


def test_a_tool_argument_named_service_tier_is_not_the_served_tier(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    """Only ``usageMetadata.serviceTier`` counts, not the model's own output."""
    parts = [
        function_call_part("deploy", {"serviceTier": "flex"}),
        text_part('"serviceTier": "flex"'),
    ]
    for usage, dollars in ((_USAGE, 12.10), (dict(_USAGE, serviceTier="priority"), 21.78)):
        cost = _billed(monkeypatch, gemini_endpoint, "gemini-4-carbon", usage, parts=parts)
        assert cost == pytest.approx(dollars)


def test_the_tier_of_a_stream_cut_before_its_end_is_billed(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    """The usage a failed turn leaves behind carries the tier it was served on."""
    base_url, script = gemini_endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel("gemini-4-carbon", api_key="test-key")
    model.initialize("Say OK.")
    script.play([chunk([text_part("O")], usage=dict(_USAGE, serviceTier="flex"))], after="cut")
    with pytest.raises(Exception):  # noqa: B017, PT011 - the cut itself is tested elsewhere
        model.generate_and_process_with_tools({})
    partial = model.take_partial_usage_response()
    assert partial is not None
    assert model.cost_multiplier_for_response(partial) == 0.5


def test_a_unary_response_names_its_tier_in_a_header() -> None:
    """``x-gemini-service-tier`` is where a non-streamed response reports it."""
    model = GeminiModel("gemini-4-carbon", api_key="test-key")
    for value, factor in (("priority", 1.8), ("Flex", 0.5), ("standard", 1.0), ("economy", 1.0)):
        response = types.GenerateContentResponse(
            sdk_http_response=types.HttpResponse(headers={"X-Gemini-Service-Tier": value})
        )
        assert model.cost_multiplier_for_response(response) == factor
    assert model.cost_multiplier_for_response(types.GenerateContentResponse()) == 1.0
    assert model.cost_multiplier_for_response(object()) == 1.0


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 100_000])
def test_the_tap_passes_every_byte_on_and_reads_split_events(size: int) -> None:
    """Events cut anywhere by the network, of one line or several, ended in any way or not."""
    body = (
        b'data: {"candidates": [],\r\ndata: "usageMetadata": {"serviceTier": "flex"}}\r\n\r\n'
        b': keep-alive\r\ndata: not json "serviceTier"\r\n\r\n'
        b'data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", '
        b'"args": {"serviceTier": "flex"}}}]}}]}\n\n'
        b'data: {"usageMetadata":\rdata: {"promptTokenCount": 1,\r'
        b'data: "serviceTier": "STANDARD"}}\r\r'
        b'data: {"usageMetadata": {"promptTokenCount": 1, "serviceTier": "PRIORITY"}}'
    )
    pieces = [body[i : i + size] for i in range(0, len(body), size)]
    response = httpx.Response(200, stream=httpx.ByteStream(b""))
    tap = _ServiceTierTap(httpx.ByteStream(b""), response)
    tap._stream = pieces  # type: ignore[assignment]  # the body, as the network cut it
    seen: list[str | None] = []
    passed = b""
    for piece in tap:
        passed += piece
        seen.append(response.extensions.get("kiss_gemini_service_tier"))
    assert passed == body
    # Each tier is known by the time the bytes after its event are handed on,
    # the one of the event a lone "\r" ends included; no other event sets one.
    assert seen[-1] == "standard"
    assert list(dict.fromkeys(seen))[-2:] == (["flex", "standard"] if size < 100 else ["standard"])
    # The last event, which nothing ends, is read when the body does.
    assert response.extensions["kiss_gemini_service_tier"] == "priority"


def test_cache_read_only_is_the_cached_share_at_the_request_rates() -> None:
    """The share is priced like the whole request, long-prompt tier included."""
    for prompt, rate in ((100_000, 0.125), (300_000, 0.25)):
        counts = (prompt - 50_000, 1000, 50_000, 0)
        share = calculate_cost("gemini-2.5-pro", *counts, cache_read_only=True)
        assert share == pytest.approx(50_000 * rate / _M)
        assert share < calculate_cost("gemini-2.5-pro", *counts)
    audio = (0, 0, 10_000, 0, 0, 0, 0, 20_000, 0)  # gemini-2.5-flash: $0.03 text, $0.10 audio
    assert calculate_cost("gemini-2.5-flash", *audio, cache_read_only=True) == pytest.approx(
        (10_000 * 0.03 + 20_000 * 0.10) / _M
    )
