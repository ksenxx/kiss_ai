# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for audio-token billing and talk-TTS cost attribution.

Reproduces two per-task cost-reporting bugs found in the July 2026 audit:

1. OpenAI GPT audio models (``gpt-audio``, ``gpt-audio-1.5``) bill AUDIO
   tokens at $32/M input and $64/M output — 12.8x/6.4x their text rates —
   but KISS billed every token at text rates because MODEL_INFO.json had
   no audio prices and the OpenAI usage extractor ignored
   ``prompt_tokens_details.audio_tokens`` /
   ``completion_tokens_details.audio_tokens``.

2. The ``talk`` tool's server-side TTS (``synthesize_talk_audio``) ran a
   throwaway ``TalkSynthesisAgent`` whose ``budget_used`` was discarded,
   so TTS spend never reached the calling Sorcar agent's reported task
   cost.

Pricing ground truth (verified 2026-07 on developers.openai.com model
pages and openrouter.ai):

* gpt-audio / gpt-audio-1.5 / gpt-audio-2025-08-28: text $2.50/$10.00,
  audio $32.00/$64.00 per 1M tokens.
* gpt-audio-mini family: text $0.60/$2.40, audio $0.60/$2.40 (equal).
"""

from types import SimpleNamespace

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO, calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


def _audio_response(
    prompt_tokens: int,
    completion_tokens: int,
    audio_input: int,
    audio_output: int,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build a Chat-Completions response shaped like the openai SDK's.

    ``audio_tokens`` are SUBSETS of ``prompt_tokens`` /
    ``completion_tokens`` (openai SDK ``CompletionUsage`` semantics).
    """
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(
            audio_tokens=audio_input,
            cached_tokens=cached_tokens,
        ),
        completion_tokens_details=SimpleNamespace(
            audio_tokens=audio_output,
            reasoning_tokens=0,
        ),
    )
    return SimpleNamespace(usage=usage)


class TestAudioPricesRegistered:
    """gpt-audio family entries must carry explicit audio prices."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-audio",
            "gpt-audio-1.5",
            "gpt-audio-2025-08-28",
            "openrouter/openai/gpt-audio",
        ],
    )
    def test_gpt_audio_family_audio_prices(self, model_name: str) -> None:
        info = MODEL_INFO[model_name]
        assert info.audio_input_price_per_1M == 32.0
        assert info.audio_output_price_per_1M == 64.0

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-audio-mini",
            "gpt-audio-mini-2025-10-06",
            "gpt-audio-mini-2025-12-15",
            "openrouter/openai/gpt-audio-mini",
        ],
    )
    def test_gpt_audio_mini_family_audio_prices(self, model_name: str) -> None:
        # openrouter.ai/openai/gpt-audio-mini lists audio rates EQUAL to
        # the text rates ($0.60 in / $2.40 out per 1M).
        info = MODEL_INFO[model_name]
        assert info.audio_input_price_per_1M == 0.6
        assert info.audio_output_price_per_1M == 2.4


class TestCalculateCostAudioTokens:
    def test_gpt_audio_15_audio_tokens_billed_at_audio_rates(self) -> None:
        cost = calculate_cost(
            "gpt-audio-1.5",
            num_input_tokens=100,
            num_output_tokens=50,
            num_audio_input_tokens=200,
            num_audio_output_tokens=1_000,
        )
        expected = (100 * 2.5 + 50 * 10.0 + 200 * 32.0 + 1_000 * 64.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_typical_talk_call_costs_6x_more_than_text_billing(self) -> None:
        # A ~30s talk() utterance: ~60 text prompt tokens, ~20 text output
        # tokens, ~600 audio output tokens.  Text-rate billing charged
        # $0.0068; the correct bill is $0.0387 (5.7x more).
        correct = calculate_cost(
            "gpt-audio-1.5", 60, 20, num_audio_output_tokens=600,
        )
        old_wrong = calculate_cost("gpt-audio-1.5", 60, 20 + 600)
        assert correct == pytest.approx((60 * 2.5 + 20 * 10.0 + 600 * 64.0) / 1e6)
        assert correct > 5 * old_wrong

    def test_gpt_audio_mini_audio_rates_equal_text_rates(self) -> None:
        cost = calculate_cost(
            "gpt-audio-mini",
            num_input_tokens=100,
            num_output_tokens=50,
            num_audio_input_tokens=300,
            num_audio_output_tokens=500,
        )
        expected = (100 * 0.6 + 50 * 2.4 + 300 * 0.6 + 500 * 2.4) / 1e6
        assert cost == pytest.approx(expected)

    def test_model_without_audio_prices_falls_back_to_text_rates(self) -> None:
        # Text-only models have no audio prices; audio counts (which
        # should never occur for them) bill at text rates rather than
        # silently costing $0.
        cost = calculate_cost(
            "gpt-5.6-sol",
            num_input_tokens=0,
            num_output_tokens=0,
            num_audio_input_tokens=1_000,
            num_audio_output_tokens=1_000,
        )
        assert cost == pytest.approx((1_000 * 5.0 + 1_000 * 30.0) / 1e6)

    def test_zero_audio_tokens_matches_legacy_arithmetic(self) -> None:
        assert calculate_cost("gpt-audio-1.5", 1_000, 2_000) == pytest.approx(
            (1_000 * 2.5 + 2_000 * 10.0) / 1e6
        )


class TestOpenAIExtractorAudioSplit:
    def test_audio_tokens_split_out_of_text_totals(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-audio-1.5", base_url="https://api.openai.com/v1", api_key="test",
        )
        response = _audio_response(
            prompt_tokens=25,
            completion_tokens=300,
            audio_input=8,
            audio_output=280,
        )
        assert m.extract_input_output_token_counts_from_response(response) == (
            17,  # 25 prompt - 8 audio
            20,  # 300 completion - 280 audio
            0,
            0,
            0,
            8,
            280,
        )

    def test_text_only_response_keeps_legacy_4_tuple(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-5.6-sol", base_url="https://api.openai.com/v1", api_key="test",
        )
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=40,
            prompt_tokens_details=SimpleNamespace(
                audio_tokens=0, cached_tokens=30,
            ),
            completion_tokens_details=SimpleNamespace(audio_tokens=0),
        )
        response = SimpleNamespace(usage=usage)
        assert m.extract_input_output_token_counts_from_response(response) == (
            70,
            40,
            30,
            0,
        )

    def test_cached_and_audio_input_both_subtracted(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-audio-1.5", base_url="https://api.openai.com/v1", api_key="test",
        )
        response = _audio_response(
            prompt_tokens=100,
            completion_tokens=10,
            audio_input=20,
            audio_output=0,
            cached_tokens=30,
        )
        counts = m.extract_input_output_token_counts_from_response(response)
        assert counts == (50, 10, 30, 0, 0, 20, 0)

    def test_missing_details_treated_as_zero_audio(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-audio-1.5", base_url="https://api.openai.com/v1", api_key="test",
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        response = SimpleNamespace(usage=usage)
        assert m.extract_input_output_token_counts_from_response(response) == (
            10,
            5,
            0,
            0,
        )


class TestKISSAgentBudgetIncludesAudio:
    def test_budget_used_reflects_audio_rates_and_tokens_counted(self) -> None:
        agent = KISSAgent("audio-budget-test")
        agent.model = OpenAICompatibleModel(
            "gpt-audio-1.5", base_url="https://api.openai.com/v1", api_key="test",
        )
        response = _audio_response(
            prompt_tokens=60,
            completion_tokens=620,
            audio_input=0,
            audio_output=600,
        )
        agent._update_tokens_and_budget_from_response(response)
        expected = (60 * 2.5 + 20 * 10.0 + 600 * 64.0) / 1e6
        assert agent.budget_used == pytest.approx(expected)
        assert agent.total_tokens_used == 680
        assert agent.context_tokens_used == 680
