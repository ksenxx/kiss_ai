# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cost audit 2026-10-13: catalog rules and partial billing on Stop.

Four defects, each reproduced end to end:

* ``openrouter/google/gemma-*`` entries publish no cache prices, yet the
  OpenRouter Google rule gave them Gemini's 0.1x cache read and a free
  cache write; they now bill cached tokens at the input price.
* ``_openai_fast_multiplier`` kept the ``openrouter/openai/`` prefix, so
  ``openrouter/openai/gpt-5.5`` on the Fast tier got 2.0x instead of the
  published 2.5x.
* A Stop (``KeyboardInterrupt``) raised from the thinking callback while a
  Responses stream ended in ``response.failed`` / ``response.incomplete``
  lost the billed usage, and the Chat adapter's delegation wrapper only
  handed over a delegate's usage on ``Exception``.
* A Stop raised from the token callback while Gemini's unary fallback
  streamed its parts lost that response's usage.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import PACKAGE_MODEL_INFO_PATH, calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models import gemini_sse_harness
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)


def test_gemma_on_openrouter_bills_cached_tokens_at_the_input_price() -> None:
    catalog = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    gemma = [
        name
        for name, entry in catalog.items()
        if name.startswith("openrouter/google/gemma")
        and entry.get("input_price_per_1M", 0) > 0
        and entry.get("cache_read_price_per_1M") is None
    ]
    assert gemma, "the catalog carries no priced Gemma entry without cache prices"
    for name in gemma:
        inp = calculate_cost(name, 1_000_000, 0, 0, 0)
        assert calculate_cost(name, 0, 0, 1_000_000, 0) == pytest.approx(inp)
        assert calculate_cost(name, 0, 0, 0, 1_000_000) == pytest.approx(inp)
    # Gemini on OpenRouter keeps Google's published 0.1x cache read.
    gemini = "openrouter/google/gemini-3.1-pro-preview"
    assert calculate_cost(gemini, 0, 0, 1_000_000, 0) == pytest.approx(
        0.1 * calculate_cost(gemini, 1_000_000, 0, 0, 0)
    )


@pytest.mark.parametrize(
    ("name", "factor"),
    [
        ("openrouter/openai/gpt-5.5", 2.5),
        ("openrouter/openai/gpt-4.1", 1.75),
        ("gpt-5.5-high", 2.5),
        ("gpt-6-astra", 2.0),
    ],
)
def test_fast_tier_factor_ignores_the_openrouter_prefix(name: str, factor: float) -> None:
    model = OpenAICompatibleModel(name, base_url="http://127.0.0.1:9", api_key="sk-test")
    assert model.cost_multiplier_for_response({"service_tier": "fast"}) == pytest.approx(factor)
    assert model.cost_multiplier_for_response({"service_tier": "flex"}) == pytest.approx(0.5)


_USAGE = {"input_tokens": 700, "output_tokens": 300, "total_tokens": 1000, "cost": 0.004}
_MODEL = "openrouter/openai/gpt-6-astra"


def _terminal_body(status: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "output": [],
        "usage": _USAGE,
    }
    if status == "failed":
        body["error"] = {"code": "x", "message": "boom"}
    else:
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def _thinking_then(event_type: str, status: str) -> list[bytes]:
    """A stream that opens a thinking block and then ends in *event_type*."""
    return [
        responses_event(
            "response.reasoning_summary_text.delta",
            {"output_index": 0, "summary_index": 0, "delta": "hmm"},
        ),
        responses_event(event_type, {"response": _terminal_body(status)}),
    ]


def _ignore_token(text: str) -> None:
    """A bound token callback makes the adapters stream."""


def _stop_on_close(is_start: bool) -> None:
    """The printer's thinking callback with the user's Stop pending."""
    if not is_start:
        raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("event_type", "status"),
    [("response.failed", "failed"), ("response.incomplete", "incomplete")],
)
def test_stop_during_thinking_close_keeps_terminal_failure_usage(
    event_type: str, status: str
) -> None:
    def responder(request: Request) -> Reply:
        """Stream the thinking block and the terminal event."""
        return Reply(sse_chunks=_thinking_then(event_type, status))

    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            token_callback=_ignore_token,
            thinking_callback=_stop_on_close,
        )
        model.initialize("hi")
        with pytest.raises(KeyboardInterrupt):
            model.generate()
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_USAGE["cost"])
    assert model.take_partial_usage_response() is None


def _echo(text: str) -> str:
    """Echo *text* (a tool so the request carries a tools schema)."""
    return text


def test_stop_in_a_delegated_responses_call_hands_over_its_usage() -> None:
    def responder(request: Request) -> Reply:
        """Stream a completed, billed turn after a thinking block."""
        return Reply(sse_chunks=_thinking_then("response.completed", "completed"))

    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"use_responses_api": True, "reasoning_effort": "high"},
            token_callback=_ignore_token,
            thinking_callback=_stop_on_close,
        )
        model.initialize("hi")
        with pytest.raises(KeyboardInterrupt):
            model.generate_and_process_with_tools({"_echo": _echo})
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_USAGE["cost"])
    assert model.take_partial_usage_response() is None


class _StreamsNothingFirst(gemini_sse_harness._Handler):
    """Answers the first (streaming) request with no chunk at all.

    The adapter then falls back to the unary call, which gets the
    scripted chunks.
    """

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Serve the script, emptied for the first request only."""
        scripted = self.script.chunks
        if not self.script.requests:
            self.script.chunks = []
        try:
            super().do_POST()
        finally:
            self.script.chunks = scripted


@pytest.fixture
def unary_endpoint() -> Generator[tuple[str, gemini_sse_harness.GeminiScript]]:
    script = gemini_sse_harness.GeminiScript()
    handler = type("_Handler", (_StreamsNothingFirst,), {"script": script})
    server = gemini_sse_harness._DaemonServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", script
    finally:
        script.release.set()
        server.shutdown()
        server.server_close()


def _stop_on_token(text: str) -> None:
    raise KeyboardInterrupt


def test_stop_during_gemini_unary_fallback_keeps_its_usage(
    monkeypatch: pytest.MonkeyPatch,
    unary_endpoint: tuple[str, gemini_sse_harness.GeminiScript],
) -> None:
    base_url, script = unary_endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    usage = {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}
    script.play([gemini_sse_harness.chunk([gemini_sse_harness.text_part("done")], usage=usage)])
    model = GeminiModel("gemini-3.8-flash", api_key="test-key", token_callback=_stop_on_token)
    model.initialize("hi")
    with pytest.raises(KeyboardInterrupt):
        model.generate()
    assert len(script.requests) == 2, "the empty stream must fall back to the unary call"
    rejected = model.take_partial_usage_response()
    assert rejected is not None
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (11, 5)
    assert model.take_partial_usage_response() is None


def test_kiss_error_paths_are_unchanged_for_terminal_failures() -> None:
    """Without a Stop the terminal events still raise KISSError and bill."""

    def responder(request: Request) -> Reply:
        return Reply(sse_chunks=_thinking_then("response.failed", "failed"))

    with ScriptedOpenAIServer(responder) as server:
        model = OpenAICompatibleModel2(
            _MODEL, base_url=server.base_url, api_key="sk-test", token_callback=_ignore_token
        )
        model.initialize("hi")
        with pytest.raises(KISSError, match="boom"):
            model.generate()
        rejected = model.take_partial_usage_response()
    assert model.extract_cost_from_response(rejected) == pytest.approx(_USAGE["cost"])
