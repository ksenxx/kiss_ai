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
  cache-hit prices (0.10x-0.20x input) and free cache writes.
* Z.AI GLM models carry their published cached-input prices.
"""

from types import SimpleNamespace

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import MODEL_INFO, calculate_cost


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
            assert info.cache_write_price_per_1M == pytest.approx(
                info.input_price_per_1M * 1.25
            ), name

    def test_gpt56_xhigh_aliases_bill_cache_writes(self):
        for name in ("gpt-5.6-sol-xhigh", "gpt-5.6-terra-xhigh", "gpt-5.6-luna-xhigh"):
            info = MODEL_INFO[name]
            assert info.cache_write_price_per_1M == pytest.approx(
                info.input_price_per_1M * 1.25
            ), name

    def test_openrouter_openai_gpt56_passthrough_bills_cache_writes(self):
        info = MODEL_INFO["openrouter/openai/gpt-5.6-sol"]
        assert info.cache_write_price_per_1M == pytest.approx(
            info.input_price_per_1M * 1.25
        )

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
        expected = (
            300_000 * 4.0 + 10_000 * 18.0 + 50_000 * 0.40 + 40_000 * 5.0
        ) / 1e6
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
        expected = (
            300_000 * 4.0 + 10_000 * 15.0 + 50_000 * 0.40 + 40_000 * 5.0
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt55_long_context_cache_writes_stay_free(self):
        cost = calculate_cost("gpt-5.5", 300_000, 5_000, 0, 40_000)
        assert cost == pytest.approx((300_000 * 10.0 + 5_000 * 45.0) / 1e6)

    def test_openai_threshold_is_272k_not_200k(self):
        cost = calculate_cost("gpt-5.6-sol", 250_000, 5_000)
        assert cost == pytest.approx((250_000 * 4.0 + 5_000 * 20.0) / 1e6)
        assert calculate_cost("gpt-5.5", 272_000, 0) == pytest.approx(
            272_000 * 5.0 / 1e6
        )
        assert calculate_cost("gpt-5.5", 272_001, 0) == pytest.approx(
            272_001 * 10.0 / 1e6
        )


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
        expected = (
            50_000 * 10.0 + 10_000 * 50.0 + 30_000 * 1.00 + 20_000 * 12.50
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_long_context_reprices_full_request(self):
        """280K in + 20K out is ~$7.10; trimmed to 272K it is ~$3.72."""
        assert calculate_cost("gpt-6-astra", 280_000, 20_000) == pytest.approx(7.10)
        assert calculate_cost("gpt-6-astra", 272_000, 20_000) == pytest.approx(3.72)

    def test_long_context_cache_rates_doubled(self):
        cost = calculate_cost("gpt-6-astra", 200_000, 5_000, 80_000, 10_000)
        expected = (
            200_000 * 20.0 + 5_000 * 75.0 + 80_000 * 2.00 + 10_000 * 25.00
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_thinking_aliases_price_like_base(self):
        for level in ("low", "medium", "high", "xhigh"):
            assert calculate_cost(
                f"gpt-6-astra-{level}", 300_000, 5_000, 10_000, 10_000
            ) == pytest.approx(
                calculate_cost("gpt-6-astra", 300_000, 5_000, 10_000, 10_000)
            ), level

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
            assert info.cache_write_price_per_1M == 0.0, name
            assert calculate_cost(name, 0, 0, 0, 1_000_000) == 0.0, name

    def test_moonshot_v1_fallback_cache_read_quarter(self):
        """Entries without an explicit cache-read price fall back to 0.25x."""
        info = MODEL_INFO["moonshot-v1-8k"]
        assert info.cache_read_price_per_1M == pytest.approx(
            info.input_price_per_1M * 0.25
        )
        assert info.cache_write_price_per_1M == 0.0

    def test_kimi_k25_cache_hit_cost(self):
        cost = calculate_cost("kimi-k2.5", 100_000, 0, 1_000_000, 0)
        assert cost == pytest.approx((100_000 * 0.60 + 1_000_000 * 0.10) / 1e6)

    def test_kimi_k3_cost(self):
        cost = calculate_cost("kimi-k3", 100_000, 10_000, 500_000, 0)
        expected = (100_000 * 3.0 + 10_000 * 15.0 + 500_000 * 0.30) / 1e6
        assert cost == pytest.approx(expected)

    def test_openrouter_kimi_k3_cache_read_not_overcharged(self):
        info = MODEL_INFO["openrouter/moonshotai/kimi-k3"]
        assert info.cache_read_price_per_1M == pytest.approx(0.30)


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
        assert cost == pytest.approx(
            (10_000 * 0.6 + 5_000 * 2.2 + 100_000 * 0.11) / 1e6
        )
