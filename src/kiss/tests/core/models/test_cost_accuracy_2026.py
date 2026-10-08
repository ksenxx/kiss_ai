# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Cost-accuracy audit regression tests (July 2026).

Verifies that the pipeline from provider usage reports to
``calculate_cost`` charges exactly what the providers bill, per their
published pricing pages:

* OpenAI GPT-5.6+ families bill prompt-cache WRITES at 1.25x the input
  rate (earlier families have free cache writes).
* OpenAI GPT-5.4/5.5/5.6 have a long-context pricing tier that triggers
  on prompts with >272k INPUT tokens (per the official model pages),
  billed at 2x input / 1.5x output for the full request.
* Gemini 3 Pro / 3.1 Pro have a long-context (prompts >200k) pricing tier.
* Gemini server-side tool-use prompt tokens count as input tokens.
* Direct Moonshot (kimi-*) models carry their published per-family
  cache-hit prices (0.10x-0.20x input) and bill 5-minute cache writes at
  the uncached input rate.
* Z.AI GLM models carry their published cached-input prices.
* Anthropic Opus 5.5 reads its cache at 0.05x input ($0.20 on $4.00),
  Fable/Mythos 5.1 at 0.025x, every other Claude at 0.1x.
* OpenAI gpt-6-sol / gpt-6-luna share gpt-6-astra's long-context tier.
* Together publishes cached-input prices that the catalog carries, so a
  cache hit reported in ``prompt_tokens_details.cached_tokens`` is not
  billed at the full input rate.
"""

import json
from types import SimpleNamespace

import pytest

from kiss.core.models import model_info
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    PACKAGE_MODEL_INFO_PATH,
    ModelInfo,
    _apply_cache_pricing,
    calculate_cost,
)


class TestOpenAIGpt56CacheWritePricing:
    def test_gpt56_models_bill_cache_writes_at_1_25x_input(self):
        """gpt-5.6-* cache writes are 1.25x input ($5.00/$2.50/$0.25 per MTok).

        Promotional prices from the Sep 2026 pricing page (sol $4, terra $2,
        luna $0.20 per MTok input; promo runs through at least Nov 21, 2026).
        """
        expected = {
            "gpt-5.6-sol": 5.00,
            "gpt-5.6-terra": 2.50,
            "gpt-5.6-luna": 0.25,
        }
        for name, cw in expected.items():
            info = MODEL_INFO[name]
            assert info.cache_write_price_per_1M == pytest.approx(cw), name
            assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25), (
                name
            )

    def test_gpt56_xhigh_aliases_bill_cache_writes(self):
        for name in ("gpt-5.6-sol-xhigh", "gpt-5.6-terra-xhigh", "gpt-5.6-luna-xhigh"):
            info = MODEL_INFO[name]
            assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25), (
                name
            )

    def test_openrouter_openai_gpt56_passthrough_bills_cache_writes(self):
        info = MODEL_INFO["openrouter/openai/gpt-5.6-sol"]
        assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)

    def test_pre_gpt56_openai_models_keep_free_cache_writes(self):
        for name in ("gpt-5.5", "gpt-5.4", "gpt-5", "gpt-4.1", "gpt-4o", "o3"):
            assert MODEL_INFO[name].cache_write_price_per_1M == 0.0, name

    def test_codex_gpt56_stays_free(self):
        """Subscription CLI models are $0 — the 1.25x rule must not price them."""
        info = MODEL_INFO["codex/gpt-5.6-sol"]
        assert (info.cache_write_price_per_1M or 0.0) == 0.0

    def test_gpt56_short_context_cost_matches_published_rates(self):
        cost = calculate_cost("gpt-5.6-sol", 100_000, 10_000, 20_000, 30_000)
        expected = (100_000 * 4.0 + 10_000 * 20.0 + 20_000 * 0.40 + 30_000 * 5.0) / 1e6
        assert cost == pytest.approx(expected)


class TestOpenAIGpt56LongContextPricing:
    def test_gpt56_terra_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-terra", 300_000, 10_000, 50_000, 40_000)
        expected = (300_000 * 4.0 + 10_000 * 18.0 + 50_000 * 0.40 + 40_000 * 5.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_sol_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-sol", 300_000, 5_000, 0, 0)
        assert cost == pytest.approx((300_000 * 8.0 + 5_000 * 30.0) / 1e6)

    def test_gpt56_luna_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-luna", 300_000, 5_000, 10_000, 10_000)
        expected = (300_000 * 0.40 + 5_000 * 1.80 + 10_000 * 0.04 + 10_000 * 0.50) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_xhigh_alias_gets_long_context_pricing(self):
        assert calculate_cost("gpt-5.6-sol-xhigh", 300_000, 5_000) == pytest.approx(
            calculate_cost("gpt-5.6-sol", 300_000, 5_000)
        )

    def test_openrouter_long_context_scales_openrouter_prices(self):
        """OpenRouter passthrough scales its OWN listed prices, not OpenAI's.

        openrouter/openai/gpt-5.6-sol lists $2/$10 per MTok on
        openrouter.ai (vs $4/$20 direct), so its long-context tier is
        2x/1.5x of $2/$10 — not the direct-OpenAI $8/$30 dollar rates.
        """
        info = MODEL_INFO["openrouter/openai/gpt-5.6-sol"]
        assert info.input_price_per_1M == pytest.approx(2.0)
        cost = calculate_cost("openrouter/openai/gpt-5.6-sol", 300_000, 10_000, 50_000, 40_000)
        expected = (300_000 * 4.0 + 10_000 * 15.0 + 50_000 * 0.40 + 40_000 * 5.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt55_long_context_cache_writes_stay_free(self):
        cost = calculate_cost("gpt-5.5", 300_000, 5_000, 0, 40_000)
        assert cost == pytest.approx((300_000 * 10.0 + 5_000 * 45.0) / 1e6)

    def test_openai_threshold_is_272k_not_200k(self):
        cost = calculate_cost("gpt-5.6-sol", 250_000, 5_000)
        assert cost == pytest.approx((250_000 * 4.0 + 5_000 * 20.0) / 1e6)
        assert calculate_cost("gpt-5.5", 272_000, 0) == pytest.approx(272_000 * 5.0 / 1e6)
        assert calculate_cost("gpt-5.5", 272_001, 0) == pytest.approx(272_001 * 10.0 / 1e6)


class TestGpt6AstraPricing:
    """gpt-6-astra rates from https://developers.openai.com/api/docs/models/gpt-6-astra.

    Standard tier: $10 input / $1 cached input / $12.50 cache writes /
    $50 output per MTok.  Prompts with more than 272K input tokens are
    priced at 2x input and cache rates and 1.5x output ($20/$2/$25/$75)
    for the full request.
    """

    def test_base_prices_match_published_rates(self):
        for name in ("gpt-6-astra", "openrouter/openai/gpt-6-astra"):
            info = MODEL_INFO[name]
            assert info.input_price_per_1M == 10.0, name
            assert info.output_price_per_1M == 50.0, name

    def test_cache_read_billed_at_0_1x_input(self):
        info = MODEL_INFO["gpt-6-astra"]
        assert info.cache_read_price_per_1M == pytest.approx(1.00)

    def test_cache_writes_billed_at_1_25x_input(self):
        for name in ("gpt-6-astra", "gpt-6-astra-xhigh", "openrouter/openai/gpt-6-astra"):
            info = MODEL_INFO[name]
            assert info.cache_write_price_per_1M == pytest.approx(12.50), name

    def test_codex_subscription_astra_stays_free(self):
        assert calculate_cost("codex/gpt-6-astra", 1_000_000, 100_000, 50_000, 50_000) == 0.0

    def test_short_context_cost_matches_official_caching_formula(self):
        """Mirrors the cost formula in the OpenAI prompt-caching guide."""
        cost = calculate_cost("gpt-6-astra", 50_000, 10_000, 30_000, 20_000)
        expected = (50_000 * 10.0 + 10_000 * 50.0 + 30_000 * 1.00 + 20_000 * 12.50) / 1e6
        assert cost == pytest.approx(expected)

    def test_long_context_reprices_full_request(self):
        """280K in + 20K out is ~$7.10; trimmed to 272K it is ~$3.72."""
        assert calculate_cost("gpt-6-astra", 280_000, 20_000) == pytest.approx(7.10)
        assert calculate_cost("gpt-6-astra", 272_000, 20_000) == pytest.approx(3.72)

    def test_long_context_cache_rates_doubled(self):
        cost = calculate_cost("gpt-6-astra", 200_000, 5_000, 80_000, 10_000)
        expected = (200_000 * 20.0 + 5_000 * 75.0 + 80_000 * 2.00 + 10_000 * 25.00) / 1e6
        assert cost == pytest.approx(expected)

    def test_thinking_aliases_price_like_base(self):
        for level in ("low", "medium", "high", "xhigh"):
            assert calculate_cost(
                f"gpt-6-astra-{level}", 300_000, 5_000, 10_000, 10_000
            ) == pytest.approx(calculate_cost("gpt-6-astra", 300_000, 5_000, 10_000, 10_000)), level

    def test_context_length_stays_at_intentional_500k_cap(self):
        """The catalog caps >=1M context windows at 500K by design.

        gpt-6-astra's real window is 1,050,000 tokens, but
        ``update_models._cap_context_length`` deliberately caps every
        >=1M context at 500,000 in the on-disk catalog.
        """
        assert MODEL_INFO["gpt-6-astra"].context_length == 500_000


class TestGeminiLongContextPricing:
    def test_gemini31_pro_long_context_cost(self):
        cost = calculate_cost("gemini-3.1-pro-preview", 250_000, 10_000, 20_000, 0)
        expected = (250_000 * 4.0 + 10_000 * 18.0 + 20_000 * 0.40) / 1e6
        assert cost == pytest.approx(expected)

    def test_gemini31_pro_short_context_cost_unchanged(self):
        cost = calculate_cost("gemini-3.1-pro-preview", 100_000, 10_000)
        assert cost == pytest.approx((100_000 * 2.0 + 10_000 * 12.0) / 1e6)

    def test_gemini25_pro_long_context_cost_unchanged(self):
        cost = calculate_cost("gemini-2.5-pro", 250_000, 10_000, 20_000, 0)
        expected = (250_000 * 2.50 + 10_000 * 15.0 + 20_000 * 0.25) / 1e6
        assert cost == pytest.approx(expected)


class TestGeminiToolUsePromptTokens:
    def _model(self) -> GeminiModel:
        return GeminiModel.__new__(GeminiModel)

    def test_tool_use_prompt_tokens_counted_as_input(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1_000,
                candidates_token_count=200,
                thoughts_token_count=50,
                cached_content_token_count=300,
                tool_use_prompt_token_count=400,
            )
        )
        usage = self._model().extract_input_output_token_counts_from_response(response)
        assert usage == (1_100, 250, 300, 0)

    def test_missing_tool_use_field_defaults_to_zero(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=500,
                candidates_token_count=100,
            )
        )
        usage = self._model().extract_input_output_token_counts_from_response(response)
        assert usage == (500, 100, 0, 0)


class TestDirectMoonshotCachePricing:
    def test_direct_kimi_models_carry_published_prices(self):
        """platform.kimi.ai July 2026: hit / miss / output per MTok."""
        expected = {
            "kimi-k2.5": (0.10, 0.60, 3.00),
            "kimi-k2.6": (0.16, 0.95, 4.00),
            "kimi-k2.7-code": (0.19, 0.95, 4.00),
            "kimi-k3": (0.30, 3.00, 15.00),
        }
        for name, (hit, miss, out) in expected.items():
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(hit), name
            assert info.input_price_per_1M == pytest.approx(miss), name
            assert info.output_price_per_1M == pytest.approx(out), name
            # A 5-minute cache write is billed at the uncached input rate
            # (kimi-k3: $3.00 write on $3.00 input) and reported in
            # ``prompt_tokens_details.cache_write_tokens``, which the
            # adapter removes from the uncached remainder:
            # https://platform.kimi.ai/docs/guide/use-context-caching-feature-of-kimi-api
            assert info.cache_write_price_per_1M == pytest.approx(miss), name
            assert calculate_cost(name, 0, 0, 0, 1_000_000) == pytest.approx(miss), name

    def test_kimi_k3_aliases_bill_cache_writes_like_the_base_model(self):
        for name in ("kimi-k3-low", "kimi-k3-high", "kimi-k3-max"):
            assert MODEL_INFO[name].cache_write_price_per_1M == pytest.approx(3.0), name

    def test_moonshot_v1_fallback_cache_read_quarter(self):
        """Entries without explicit cache prices fall back to 0.25x read / 1x write."""
        info = MODEL_INFO["moonshot-v1-8k"]
        assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.25)
        assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M)

    def test_kimi_k25_cache_hit_cost(self):
        cost = calculate_cost("kimi-k2.5", 100_000, 0, 1_000_000, 0)
        assert cost == pytest.approx((100_000 * 0.60 + 1_000_000 * 0.10) / 1e6)

    def test_kimi_k3_cost(self):
        cost = calculate_cost("kimi-k3", 100_000, 10_000, 500_000, 0)
        expected = (100_000 * 3.0 + 10_000 * 15.0 + 500_000 * 0.30) / 1e6
        assert cost == pytest.approx(expected)

    def test_openrouter_kimi_k3_cache_read_not_overcharged(self):
        """OpenRouter kimi-k3 keeps its published cache price on the base entry and every alias.

        The literal prices are not pinned, and neither is the cache/input
        ratio: ``update_models.py`` copies openrouter.ai's headline
        listing, which is the cheapest endpoint's price.  On 2026-09-30 it
        was 0.1x ($3.00 / $0.30); on 2026-10-01 the headline endpoint
        (Relace) published $0.7072 input with an undiscounted $0.7072
        cache read.  What must hold is that the catalog bills the explicit
        published ``input_cache_read_price`` rather than the 0.25x
        direct-Moonshot default, and that the thinking aliases bill
        exactly like the base model.
        """
        raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())["openrouter/moonshotai/kimi-k3"]
        base = MODEL_INFO["openrouter/moonshotai/kimi-k3"]
        assert base.input_price_per_1M > 0
        assert base.cache_read_price_per_1M == pytest.approx(raw["cache_read_price_per_1M"])
        for name in (
            "openrouter/moonshotai/kimi-k3-low",
            "openrouter/moonshotai/kimi-k3-high",
            "openrouter/moonshotai/kimi-k3-max",
        ):
            info = MODEL_INFO[name]
            assert info.input_price_per_1M == pytest.approx(base.input_price_per_1M), name
            assert info.output_price_per_1M == pytest.approx(base.output_price_per_1M), name
            assert info.cache_read_price_per_1M == pytest.approx(base.cache_read_price_per_1M), name
        # ``~moonshotai/kimi-latest`` is OpenRouter's rolling alias.  Its
        # ``prompt``/``completion`` prices are a blend that openrouter.ai
        # publishes independently of kimi-k3 (about half the k3 input
        # price), while ``input_cache_read`` is the upstream cache rate.
        # The 0.1x ratio therefore does not hold for the alias, and neither
        # does any fixed ratio to the blended input price (2026-09-26: $1.03
        # input, $0.30 cache read = 0.29x).  The updater prices each
        # OpenRouter row independently, so the alias is not tied to the
        # kimi-k3 entry either.  The blended input price can even drop
        # below the cache rate (2026-09-30: $0.318 input, $0.40 cache
        # read), so no bound ties the two.  What must hold is that the
        # explicit OpenRouter cache price was kept: a positive price that
        # is not the 0.25x direct-Moonshot default.
        latest = MODEL_INFO["openrouter/~moonshotai/kimi-latest"]
        assert latest.input_price_per_1M > 0
        cache_read = latest.cache_read_price_per_1M
        assert cache_read is not None
        assert cache_read > 0
        assert cache_read != pytest.approx(latest.input_price_per_1M * 0.25)


class TestDirectCatalogPricesMatchProviderPages:
    def test_gemini_36_and_37_flash_are_standard_not_batch_price(self):
        """ai.google.dev pricing 2026-09: 3.6/3.7/3.8 Flash are $0.75 / $3.75.

        The batch tier is $0.375 / $1.875, which the catalog had picked up for 3.7.
        """
        for name in ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
            info = MODEL_INFO[name]
            assert info.input_price_per_1M == pytest.approx(0.75), name
            assert info.output_price_per_1M == pytest.approx(3.75), name
            assert info.cache_read_price_per_1M == pytest.approx(0.075), name
        assert calculate_cost("gemini-3.7-flash", 1_000_000, 100_000, 500_000, 0) == pytest.approx(
            (1_000_000 * 0.75 + 100_000 * 3.75 + 500_000 * 0.075) / 1e6
        )

    def test_gemini_tts_and_embedding_2_are_not_free(self):
        """ai.google.dev pricing 2026-10: 3.1 Flash TTS $1 / $20, Embedding 2 $0.20 input.

        Both entries had been left at $0 / $0, so any call showed $0.00.
        """
        assert calculate_cost("gemini-3.1-flash-tts-preview", 1_000, 2_000) == pytest.approx(
            (1_000 * 1.0 + 2_000 * 20.0) / 1e6
        )
        assert calculate_cost("gemini-embedding-2", 1_000_000, 0) == pytest.approx(0.20)

    def test_gpt_audio_mini_audio_tokens_priced_separately(self):
        """developers.openai.com pricing 2026-09: audio $10 / $20, text $0.60 / $2.40."""
        info = MODEL_INFO["gpt-audio-mini"]
        assert info.audio_input_price_per_1M == pytest.approx(10.0)
        assert info.audio_output_price_per_1M == pytest.approx(20.0)
        cost = calculate_cost("gpt-audio-mini", 1_000, 1_000, 0, 0, 0, 1_000, 1_000)
        assert cost == pytest.approx(
            (1_000 * 0.60 + 1_000 * 2.40 + 1_000 * 10.0 + 1_000 * 20.0) / 1e6
        )


class TestLongContextTierUsesPromptTokens:
    def test_large_output_does_not_trigger_long_context_tier(self):
        cost = calculate_cost("gpt-5.6-sol", 250_000, 100_000)
        assert cost == pytest.approx((250_000 * 4.0 + 100_000 * 20.0) / 1e6)

    def test_prompt_side_cache_tokens_count_toward_tier(self):
        cost = calculate_cost("gpt-5.6-sol", 250_000, 1_000, 60_000, 0)
        expected = (250_000 * 8.0 + 1_000 * 30.0 + 60_000 * 0.80) / 1e6
        assert cost == pytest.approx(expected)

    def test_gemini_output_excluded_from_tier_decision(self):
        cost = calculate_cost("gemini-2.5-pro", 150_000, 100_000)
        assert cost == pytest.approx((150_000 * 1.25 + 100_000 * 10.0) / 1e6)


class TestGlmCachePricing:
    def test_glm_models_carry_published_cached_input_prices(self):
        expected = {
            "glm-4.5": 0.11,
            "glm-4.5-air": 0.03,
            "glm-4.5-airx": 0.22,
            "glm-4.5-x": 0.45,
            "glm-4.6": 0.11,
            "glm-4.7": 0.11,
        }
        for name, price in expected.items():
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(price), name

    def test_glm45_cache_hit_cost(self):
        cost = calculate_cost("glm-4.5", 10_000, 5_000, 100_000, 0)
        assert cost == pytest.approx((10_000 * 0.6 + 5_000 * 2.2 + 100_000 * 0.11) / 1e6)


class TestAnthropicCacheReadFamilies:
    """https://platform.claude.com/docs/en/about-claude/pricing (2026-09)."""

    def test_opus_55_cache_read_is_five_percent_of_input(self):
        info = MODEL_INFO["claude-opus-5-5"]
        assert info.input_price_per_1M == pytest.approx(4.0)
        assert info.cache_read_price_per_1M == pytest.approx(0.20)
        assert info.cache_write_price_per_1M == pytest.approx(5.0)
        assert info.cache_write_1h_price_per_1M == pytest.approx(8.0)
        cost = calculate_cost("claude-opus-5-5", 10_000, 1_000, 1_000_000, 0)
        assert cost == pytest.approx((10_000 * 4.0 + 1_000 * 20.0 + 1_000_000 * 0.20) / 1e6)

    def test_fable_51_reads_at_two_and_a_half_percent(self):
        info = MODEL_INFO["claude-fable-5-1"]
        assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.025)

    def test_fable_5_and_older_opus_keep_ten_percent(self):
        for name in ("claude-fable-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1), (
                name
            )

    def test_openrouter_opus_55_passthrough_carries_the_same_rate(self):
        info = MODEL_INFO["openrouter/anthropic/claude-opus-5.5"]
        assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.05)

    def test_sonnet_55_cache_read_is_five_percent_of_input(self):
        """Sonnet 5.5: $2 / $10, 5m write $2.50, 1h write $4, read $0.10 (2026-10)."""
        for name in (
            "claude-sonnet-5-5",
            "claude-sonnet-5-5-xhigh",
            "openrouter/anthropic/claude-sonnet-5.5",
            "openrouter/~anthropic/claude-sonnet-latest",
        ):
            info = MODEL_INFO[name]
            assert info.input_price_per_1M == pytest.approx(2.0), name
            assert info.output_price_per_1M == pytest.approx(10.0), name
            assert info.cache_read_price_per_1M == pytest.approx(0.10), name
            assert info.cache_write_price_per_1M == pytest.approx(2.5), name
            assert info.cache_write_1h_price_per_1M == pytest.approx(4.0), name
        cost = calculate_cost("claude-sonnet-5-5", 10_000, 1_000, 1_000_000, 20_000)
        expected = (10_000 * 2.0 + 1_000 * 10.0 + 1_000_000 * 0.10 + 20_000 * 2.5) / 1e6
        assert cost == pytest.approx(expected)


class TestClaudeHaiku55LongPromptTier:
    """Haiku 5.5 bills prompts over 100K tokens at 5x every rate.

    https://platform.claude.com/docs/en/models/haiku-5-5/overview:
    $0.10/$0.50, 5m write $0.125, 1h write $0.20, read $0.01 up to 100K
    prompt tokens; $0.50/$2.50, $0.625, $1, $0.05 above.  OpenRouter
    publishes the same ``min_prompt_tokens: 100000`` override.
    """

    def test_short_prompt_bills_base_rates(self):
        info = MODEL_INFO["claude-haiku-5-5"]
        assert info.input_price_per_1M == pytest.approx(0.10)
        assert info.output_price_per_1M == pytest.approx(0.50)
        assert info.cache_read_price_per_1M == pytest.approx(0.01)
        assert info.cache_write_price_per_1M == pytest.approx(0.125)
        assert info.cache_write_1h_price_per_1M == pytest.approx(0.20)
        cost = calculate_cost("claude-haiku-5-5", 60_000, 8_000, 30_000, 10_000)
        expected = (60_000 * 0.10 + 8_000 * 0.50 + 30_000 * 0.01 + 10_000 * 0.125) / 1e6
        assert cost == pytest.approx(expected)

    def test_exactly_100k_prompt_tokens_is_still_the_short_tier(self):
        assert calculate_cost("claude-haiku-5-5", 100_000, 1_000) == pytest.approx(
            (100_000 * 0.10 + 1_000 * 0.50) / 1e6
        )

    def test_prompt_over_100k_reprices_the_whole_request_at_5x(self):
        cost = calculate_cost("claude-haiku-5-5", 100_001, 1_000)
        assert cost == pytest.approx((100_001 * 0.50 + 1_000 * 2.50) / 1e6)

    def test_cache_buckets_count_toward_the_tier_and_scale_5x(self):
        cost = calculate_cost(
            "claude-haiku-5-5",
            50_000,
            2_000,
            40_000,
            20_000,
            num_cache_write_1h_tokens=5_000,
        )
        expected = (
            50_000 * 0.50 + 2_000 * 2.50 + 40_000 * 0.05 + 20_000 * 0.625 + 5_000 * 1.0
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_large_output_alone_does_not_trigger_the_tier(self):
        assert calculate_cost("claude-haiku-5-5", 1_000, 120_000) == pytest.approx(
            (1_000 * 0.10 + 120_000 * 0.50) / 1e6
        )

    def test_aliases_and_openrouter_twins_inherit_the_tier(self):
        base = calculate_cost("claude-haiku-5-5", 200_000, 1_000)
        assert base == pytest.approx((200_000 * 0.50 + 1_000 * 2.50) / 1e6)
        for name in (
            "anthropic/claude-haiku-5-5",
            "openrouter/anthropic/claude-haiku-5.5",
            "openrouter/~anthropic/claude-haiku-latest",
        ):
            assert calculate_cost(name, 200_000, 1_000) == pytest.approx(base), name

    def test_other_claude_models_have_no_long_prompt_tier(self):
        for name in (
            "claude-haiku-4-5",
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            "claude-fable-5-1",
            "openrouter/anthropic/claude-sonnet-5.5",
        ):
            info = MODEL_INFO[name]
            expected = (300_000 * info.input_price_per_1M + 1_000 * info.output_price_per_1M) / 1e6
            assert calculate_cost(name, 300_000, 1_000) == pytest.approx(expected), name


class TestGpt6LongContextTier:
    """gpt-6-sol $2/$0.20/$2.50/$10 -> $4/$0.40/$5/$15 above 272k prompt tokens;
    gpt-6-luna $0.10/$0.01/$0.125/$0.50 -> $0.20/$0.02/$0.25/$0.75
    (https://developers.openai.com/api/docs/pricing)."""

    def test_gpt6_sol_long_context_doubles_input_and_cache_and_uplifts_output(self):
        cost = calculate_cost("gpt-6-sol", 100_000, 10_000, 150_000, 50_000)
        expected = (100_000 * 4.0 + 150_000 * 0.40 + 50_000 * 5.0 + 10_000 * 15.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt6_sol_short_context_unchanged(self):
        cost = calculate_cost("gpt-6-sol", 100_000, 10_000, 50_000, 50_000)
        expected = (100_000 * 2.0 + 50_000 * 0.20 + 50_000 * 2.5 + 10_000 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt6_luna_long_context(self):
        cost = calculate_cost("gpt-6-luna", 300_000, 10_000)
        assert cost == pytest.approx((300_000 * 0.20 + 10_000 * 0.75) / 1e6)

    def test_thinking_aliases_and_openrouter_twins_inherit_the_tier(self):
        base = calculate_cost("gpt-6-sol", 300_000, 5_000, 10_000, 10_000)
        alias = calculate_cost("gpt-6-sol-high", 300_000, 5_000, 10_000, 10_000)
        assert alias == pytest.approx(base)
        twin = MODEL_INFO["openrouter/openai/gpt-6-sol"]
        twin_cost = calculate_cost("openrouter/openai/gpt-6-sol", 300_000, 5_000)
        expected = (
            300_000 * twin.input_price_per_1M * 2.0 + 5_000 * twin.output_price_per_1M * 1.5
        ) / 1e6
        assert twin_cost == pytest.approx(expected)


class TestGpt61SolPricing:
    """gpt-6.1-sol $2/$0.10/$2.50/$10 -> $4/$0.20/$5/$15 above 272k prompt
    tokens; cached input is 5% of input
    (https://developers.openai.com/api/docs/models/gpt-6.1-sol)."""

    def test_catalog_rates(self):
        for name in ("gpt-6.1-sol", "gpt-6.1-sol-high", "openrouter/openai/gpt-6.1-sol"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(0.10), name
            assert info.cache_write_price_per_1M == pytest.approx(2.50), name

    def test_short_context(self):
        cost = calculate_cost("gpt-6.1-sol", 100_000, 10_000, 100_000, 50_000)
        expected = (100_000 * 2.0 + 100_000 * 0.10 + 50_000 * 2.5 + 10_000 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_long_context_uplift(self):
        cost = calculate_cost("gpt-6.1-sol", 100_000, 10_000, 150_000, 50_000)
        expected = (100_000 * 4.0 + 150_000 * 0.20 + 50_000 * 5.0 + 10_000 * 15.0) / 1e6
        assert cost == pytest.approx(expected)
        alias = calculate_cost("gpt-6.1-sol-medium", 100_000, 10_000, 150_000, 50_000)
        assert alias == pytest.approx(expected)


class TestGptProLongContextTier:
    """gpt-5.5-pro and gpt-5.4-pro bill $30/$180 below 272k prompt tokens
    and $60/$270 above (https://developers.openai.com/api/docs/pricing).

    ``update_models.py`` keeps ``-pro`` models out of the shipped catalog,
    so the test registers them the way a user's MY_MODELS.json entry would.
    """

    @pytest.mark.parametrize("name", ["gpt-5.5-pro", "gpt-5.4-pro"])
    def test_pro_long_context_uplift(self, name):
        assert name not in MODEL_INFO
        MODEL_INFO[name] = ModelInfo(1_050_000, 30.0, 180.0, True, False, True)
        try:
            short = calculate_cost(name, 272_000, 10_000)
            assert short == pytest.approx((272_000 * 30.0 + 10_000 * 180.0) / 1e6)
            long = calculate_cost(name, 272_001, 10_000)
            assert long == pytest.approx((272_001 * 60.0 + 10_000 * 270.0) / 1e6)
        finally:
            del MODEL_INFO[name]


class TestGeminiProLatestLongContextTier:
    """OpenRouter's rolling ``~google/gemini-pro-latest`` carries the same
    200K override as gemini-3.1-pro ($2/$0.20/$12 -> $4/$0.40/$18,
    https://openrouter.ai/api/v1/models), so the token-priced fallback must
    apply it too."""

    def test_uplift_above_200k(self):
        name = "openrouter/~google/gemini-pro-latest"
        short = calculate_cost(name, 200_000, 10_000, 0)
        assert short == pytest.approx((200_000 * 2.0 + 10_000 * 12.0) / 1e6)
        long = calculate_cost(name, 150_000, 10_000, 60_000)
        expected = (150_000 * 4.0 + 60_000 * 0.40 + 10_000 * 18.0) / 1e6
        assert long == pytest.approx(expected)


class TestDirectOverrideWithReadPriceOnly:
    """A direct-provider entry that publishes only a read price (e.g. a
    MY_MODELS.json override) keeps the provider's write rule; only
    ``openrouter/*`` entries treat a missing write price as unpublished."""

    def test_anthropic_override_gets_write_multipliers(self):
        info = ModelInfo(200_000, 3.0, 15.0, True, False, True, 0.30)
        _apply_cache_pricing("claude-sonnet-4-6", info)
        assert info.cache_read_price_per_1M == pytest.approx(0.30)
        assert info.cache_write_price_per_1M == pytest.approx(3.75)
        assert info.cache_write_1h_price_per_1M == pytest.approx(6.0)

    def test_openrouter_published_read_keeps_missing_write(self):
        info = ModelInfo(200_000, 3.0, 15.0, True, False, True, 0.30)
        _apply_cache_pricing("openrouter/anthropic/claude-sonnet-4.6", info)
        assert info.cache_write_price_per_1M is None
        assert info.cache_write_1h_price_per_1M == pytest.approx(6.0)


class TestTogetherCachedInputPrices:
    """https://www.together.ai/pricing and ``/v1/models`` ``pricing.cached_input``.

    Kimi-K3 was $3.00/$15.00 with $0.30 cached input until 2026-09; Together
    cut it to $2.70/$13.50/$0.27 (``/v1/models`` on 2026-10-01).
    """

    def test_together_models_carry_published_cache_read_prices(self):
        expected = {
            "moonshotai/Kimi-K3": 0.27,
            "zai-org/GLM-5.3": 0.26,
            "zai-org/GLM-5.3-Flash": 0.03,
            "deepseek-ai/DeepSeek-V4-Pro-0813": 0.13,
            "deepseek-ai/DeepSeek-V4.1-Flash": 0.006,
            "Qwen/Qwen3.5-397B-A17B": 0.35,
        }
        for name, price in expected.items():
            assert MODEL_INFO[name].cache_read_price_per_1M == pytest.approx(price), name

    def test_together_kimi_k3_cache_hit_is_not_billed_as_input(self):
        cost = calculate_cost("moonshotai/Kimi-K3", 90, 20, 3_012, 0)
        assert cost == pytest.approx((90 * 2.7 + 20 * 13.5 + 3_012 * 0.27) / 1e6)

    def test_together_models_without_a_published_cache_price_bill_hits_as_input(self):
        info = MODEL_INFO["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
        assert info.cache_read_price_per_1M is None
        assert calculate_cost("meta-llama/Llama-3.3-70B-Instruct-Turbo", 0, 0, 1_000_000, 0) == (
            pytest.approx(info.input_price_per_1M)
        )


class TestHotAddedCustomModelCachePricing:
    """A MY_MODELS.json entry saved while the daemon runs is priced like a
    catalog entry: ``_lookup_model_info`` applies the provider's cache
    discount instead of billing every cache hit at the full input price
    until the next restart."""

    def test_lazy_lookup_matches_import_time_normalization(self, tmp_path):
        name = "gpt-4.1-audit-custom"
        entry = {
            "context_length": 128_000,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 8.0,
            "endpoint": "https://example.invalid/v1",
        }
        saved = model_info.USER_MY_MODELS_PATH
        model_info.USER_MY_MODELS_PATH = tmp_path / "MY_MODELS.json"
        try:
            model_info.USER_MY_MODELS_PATH.write_text(json.dumps({name: entry}), encoding="utf-8")
            info = model_info._lookup_model_info(name)
            assert info is not None
            # OpenAI 4.1 family: cache reads at 0.25x, no cache-write charge.
            assert info.cache_read_price_per_1M == pytest.approx(0.5)
            assert info.cache_write_price_per_1M == pytest.approx(0.0)
            assert calculate_cost(name, 0, 0, 100_000, 0) == pytest.approx(0.05)
            restart = model_info._build_model_info_entry(entry)
            model_info._apply_cache_pricing(name, restart)
            assert (
                info.cache_read_price_per_1M,
                info.cache_write_price_per_1M,
                info.cache_write_1h_price_per_1M,
            ) == (
                restart.cache_read_price_per_1M,
                restart.cache_write_price_per_1M,
                restart.cache_write_1h_price_per_1M,
            )
            assert model_info._lookup_model_info("openai/" + name) is not None
        finally:
            model_info.USER_MY_MODELS_PATH = saved
