# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end cost tests against real local SSE servers (no mocks).

1. Gemini image models price generated images far above text output
   (gemini-3.1-flash-image: $60 vs $3 per 1M tokens).  The IMAGE share of
   ``candidatesTokensDetails`` used to be billed at the text output rate.
2. A Chat Completions provider may put ``usage`` on a chunk that also
   carries content.  The usage used to be recorded only after the token
   callback ran, so a Stop raised by that callback lost the spend.
"""

from __future__ import annotations

import io
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import MODEL_INFO, calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_IMAGE_MODEL = "gemini-3.1-flash-image"


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _image_usage(image_tokens: int) -> dict[str, Any]:
    """Usage of a turn with 100 prompt tokens and 10 text + *image_tokens* output."""
    details = [{"modality": "TEXT", "tokenCount": 10}]
    if image_tokens:
        details.append({"modality": "IMAGE", "tokenCount": image_tokens})
    return {
        "promptTokenCount": 100,
        "candidatesTokenCount": 10 + image_tokens,
        "totalTokenCount": 110 + image_tokens,
        "candidatesTokensDetails": details,
    }


def _run_gemini_agent(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[str, GeminiScript],
    image_tokens: int,
) -> KISSAgent:
    """Run one real non-agentic turn of the image model against *endpoint*."""
    base_url, script = endpoint
    script.play([chunk([text_part("hi")], usage=_image_usage(image_tokens))], after="close")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    agent = KISSAgent("image-output-cost")
    agent.model = GeminiModel(_IMAGE_MODEL, api_key="test-key")
    agent.run(
        _IMAGE_MODEL,
        "Draw a cat.",
        is_agentic=False,
        printer=ConsolePrinter(file=io.StringIO()),
    )
    return agent


def test_gemini_image_output_billed_at_image_rate(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    agent = _run_gemini_agent(monkeypatch, gemini_endpoint, image_tokens=1290)
    info = MODEL_INFO[_IMAGE_MODEL]
    assert info.image_output_price_per_1M == 60.0
    expected = (
        100 * info.input_price_per_1M + 10 * info.output_price_per_1M + 1290 * 60.0
    ) / 1e6
    assert agent.budget_used == pytest.approx(expected)
    assert agent.total_tokens_used == 1400
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["output_tokens"] == 1300


def test_gemini_text_only_turn_of_image_model_uses_text_rate(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    agent = _run_gemini_agent(monkeypatch, gemini_endpoint, image_tokens=0)
    assert agent.budget_used == pytest.approx(calculate_cost(_IMAGE_MODEL, 100, 10))


def test_image_tokens_without_image_price_fall_back_to_output_rate() -> None:
    assert MODEL_INFO["gemini-2.5-flash"].image_output_price_per_1M is None
    assert calculate_cost(
        "gemini-2.5-flash", 0, 0, num_image_output_tokens=1000
    ) == pytest.approx(calculate_cost("gemini-2.5-flash", 0, 1000))


# ------------------------------------------------- usage on a content chunk

_CHAT_MODEL = "gpt-usage-on-content-chunk"


def _content_and_usage_chunk() -> bytes:
    return chat_chunk(
        {
            "id": "chatcmpl-usage",
            "object": "chat.completion.chunk",
            "model": _CHAT_MODEL,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )


def _reply(_request: Request) -> Reply:
    return Reply(sse_chunks=[_content_and_usage_chunk(), b"data: [DONE]\n\n"])


def _press_stop(_token: str) -> None:
    raise KeyboardInterrupt


def test_stop_in_callback_keeps_usage_of_the_same_chunk() -> None:
    with ScriptedOpenAIServer(_reply) as server:
        model = OpenAICompatibleModel(
            _CHAT_MODEL,
            base_url=server.base_url,
            api_key="test-key",
            token_callback=_press_stop,
        )
        model.initialize("Say hi.")
        with pytest.raises(KeyboardInterrupt):
            model.generate()
    partial = model.take_partial_usage_response()
    assert partial is not None, "usage on the stopped chunk was dropped"
    assert model.extract_input_output_token_counts_from_response(partial)[:2] == (11, 7)
    assert model.take_partial_usage_response() is None
