"""Gemini audio input is billed at the model's audio rate.

Several Gemini models price audio input above text (per 1M tokens,
ai.google.dev/gemini-api/docs/pricing): gemini-2.5-flash $1.00 audio vs
$0.30 text, gemini-3-flash-preview $1.00 vs $0.50.  The response's
``usage_metadata.prompt_tokens_details`` gives the per-modality split;
the uncached AUDIO share must reach ``calculate_cost`` as audio input.
"""

from __future__ import annotations

import pytest
from google.genai import types

from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import calculate_cost


def _response(
    prompt: int,
    output: int,
    audio: int = 0,
    cached: int = 0,
    cached_audio: int = 0,
) -> types.GenerateContentResponse:
    """A real SDK response carrying the given usage counts."""
    details = [types.ModalityTokenCount(modality=types.MediaModality.TEXT,
                                        token_count=prompt - audio)]
    if audio:
        details.append(types.ModalityTokenCount(
            modality=types.MediaModality.AUDIO, token_count=audio,
        ))
    cache_details = (
        [types.ModalityTokenCount(
            modality=types.MediaModality.AUDIO, token_count=cached_audio,
        )]
        if cached_audio else None
    )
    return types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=output,
            cached_content_token_count=cached or None,
            prompt_tokens_details=details,
            cache_tokens_details=cache_details,
        ),
    )


def _cost(model: str, usage: tuple[int, ...]) -> float:
    """``calculate_cost`` for an extractor's usage tuple."""
    padded = (*usage, 0, 0, 0, 0)[:8]
    return calculate_cost(
        model, padded[0], padded[1], padded[2], padded[3], padded[4],
        num_audio_input_tokens=padded[5], num_audio_output_tokens=padded[6],
        num_audio_cache_read_tokens=padded[7],
    )


def _model(name: str) -> GeminiModel:
    """A GeminiModel instance (no request is sent)."""
    return GeminiModel(name, api_key="test-key")


def test_audio_prompt_is_split_and_billed_at_the_audio_rate() -> None:
    """100k audio + 200 text prompt tokens on gemini-2.5-flash."""
    usage = _model("gemini-2.5-flash").extract_input_output_token_counts_from_response(
        _response(prompt=100_200, output=500, audio=100_000),
    )
    assert usage == (200, 500, 0, 0, 0, 100_000, 0, 0)
    cost = _cost("gemini-2.5-flash", usage)
    expected = (200 * 0.30 + 100_000 * 1.00 + 500 * 2.50) / 1_000_000
    assert cost == pytest.approx(expected)


def test_cached_audio_is_billed_at_the_cached_audio_rate() -> None:
    """Cached audio costs Google's cached-audio price, not the text one.

    gemini-3-flash-preview: text $0.50, audio $1.00, cached text $0.05,
    cached audio $0.10, output $3.00 per 1M tokens.
    """
    usage = _model("gemini-3-flash-preview").extract_input_output_token_counts_from_response(
        _response(
            prompt=10_000, output=10, audio=6_000, cached=5_000, cached_audio=4_000,
        ),
    )
    # text input = 10000 - 5000 cached - 2000 uncached audio
    assert usage == (3_000, 10, 1_000, 0, 0, 2_000, 0, 4_000)
    expected = (
        3_000 * 0.50 + 10 * 3.00 + 1_000 * 0.05 + 2_000 * 1.00 + 4_000 * 0.10
    ) / 1_000_000
    assert _cost("gemini-3-flash-preview", usage) == pytest.approx(expected)


def test_fully_cached_audio_prompt() -> None:
    """gemini-2.5-flash, 100k prompt tokens, all cached audio: $0.10/1M."""
    usage = _model("gemini-2.5-flash").extract_input_output_token_counts_from_response(
        _response(
            prompt=100_000, output=0, audio=100_000, cached=100_000,
            cached_audio=100_000,
        ),
    )
    assert usage == (0, 0, 0, 0, 0, 0, 0, 100_000)
    assert _cost("gemini-2.5-flash", usage) == pytest.approx(0.01)


def test_usage_through_kiss_agent_billing() -> None:
    """KISSAgent bills the 8-tuple exactly as ``calculate_cost`` does."""
    from kiss.core.kiss_agent import KISSAgent

    agent = KISSAgent("gemini-audio-billing")
    agent.model = _model("gemini-2.5-flash")
    agent.budget_used = 0.0
    agent.total_tokens_used = 0
    agent._update_tokens_and_budget_from_response(
        _response(
            prompt=100_200, output=500, audio=100_000, cached=50_000,
            cached_audio=50_000,
        ),
    )
    expected = (200 * 0.30 + 50_000 * 1.00 + 50_000 * 0.10 + 500 * 2.50) / 1_000_000
    assert agent.budget_used == pytest.approx(expected)
    assert agent.total_tokens_used == 100_700
    # Per-call statistics (llm_call events, autorouter) count audio too.
    assert agent.last_call_usage == {
        "input_tokens": 50_200,
        "output_tokens": 500,
        "cache_read": 50_000,
        "cache_write": 0,
        "cost": pytest.approx(expected),
    }


def test_text_only_prompt_keeps_the_four_tuple() -> None:
    """Without audio the usage shape and cost are unchanged."""
    usage = _model("gemini-2.5-flash").extract_input_output_token_counts_from_response(
        _response(prompt=1_000, output=100, cached=400),
    )
    assert usage == (600, 100, 400, 0)


def test_models_without_an_audio_premium_bill_audio_as_text() -> None:
    """gemini-2.5-pro has no audio price: audio costs the text rate."""
    cost = calculate_cost("gemini-2.5-pro", 0, 0, num_audio_input_tokens=100_000)
    assert cost == pytest.approx(100_000 * 1.25 / 1_000_000)
