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
* OpenAI GPT-5.6 models have a long-context (>200k tokens) pricing tier.
* Gemini 3 Pro / 3.1 Pro have a long-context (>200k) pricing tier.
* Gemini server-side tool-use prompt tokens count as input tokens.
* Direct Moonshot (kimi-*) cache hits are billed at 0.25x input.
* Z.AI GLM models carry their published cached-input prices.
"""

from types import SimpleNamespace

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import MODEL_INFO, calculate_cost


class TestOpenAIGpt56CacheWritePricing:
    def test_gpt56_models_bill_cache_writes_at_1_25x_input(self):
        """gpt-5.6-* cache writes are 1.25x input ($6.25/$3.125/$1.25 per MTok)."""
        expected = {
            "gpt-5.6-sol": 6.25,
            "gpt-5.6-terra": 3.125,
            "gpt-5.6-luna": 1.25,
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
        # gpt-5.6-sol short context: $5 in, $30 out, $0.50 cached, $6.25 cache write.
        cost = calculate_cost("gpt-5.6-sol", 100_000, 10_000, 20_000, 30_000)
        expected = (100_000 * 5.0 + 10_000 * 30.0 + 20_000 * 0.50 + 30_000 * 6.25) / 1e6
        assert cost == pytest.approx(expected)


class TestOpenAIGpt56LongContextPricing:
    def test_gpt56_terra_long_context_cost(self):
        # >200k total tokens: $5 in, $22.50 out, $0.50 cached, $6.25 cache write.
        cost = calculate_cost("gpt-5.6-terra", 300_000, 10_000, 50_000, 40_000)
        expected = (
            300_000 * 5.0 + 10_000 * 22.50 + 50_000 * 0.50 + 40_000 * 6.25
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_sol_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-sol", 250_000, 5_000, 0, 0)
        assert cost == pytest.approx((250_000 * 10.0 + 5_000 * 45.0) / 1e6)

    def test_gpt56_luna_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-luna", 250_000, 5_000, 10_000, 10_000)
        expected = (250_000 * 2.0 + 5_000 * 9.0 + 10_000 * 0.20 + 10_000 * 2.50) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_xhigh_alias_gets_long_context_pricing(self):
        assert calculate_cost("gpt-5.6-sol-xhigh", 250_000, 5_000) == pytest.approx(
            calculate_cost("gpt-5.6-sol", 250_000, 5_000)
        )

    def test_gpt55_long_context_cache_writes_stay_free(self):
        # gpt-5.5 has no cache-write fee even in the long-context tier.
        cost = calculate_cost("gpt-5.5", 250_000, 5_000, 0, 40_000)
        assert cost == pytest.approx((250_000 * 10.0 + 5_000 * 45.0) / 1e6)


class TestGeminiLongContextPricing:
    def test_gemini31_pro_long_context_cost(self):
        # >200k prompts: $4 in, $18 out, $0.40 cached.
        cost = calculate_cost("gemini-3.1-pro-preview", 250_000, 10_000, 20_000, 0)
        expected = (250_000 * 4.0 + 10_000 * 18.0 + 20_000 * 0.40) / 1e6
        assert cost == pytest.approx(expected)

    def test_gemini3_pro_long_context_cost(self):
        cost = calculate_cost("gemini-3-pro-preview", 250_000, 10_000)
        assert cost == pytest.approx((250_000 * 4.0 + 10_000 * 18.0) / 1e6)

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
        # input = (1000 - 300 cached) + 400 tool-use; output = 200 + 50 thoughts.
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
    def test_direct_kimi_models_cache_read_quarter(self):
        for name in ("kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code", "moonshot-v1-8k"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(
                info.input_price_per_1M * 0.25
            ), name
            assert info.cache_write_price_per_1M == 0.0, name

    def test_kimi_k25_cache_hit_cost(self):
        # kimi-k2.5: $0.60 miss / $0.15 hit per MTok.
        cost = calculate_cost("kimi-k2.5", 100_000, 0, 1_000_000, 0)
        assert cost == pytest.approx((100_000 * 0.60 + 1_000_000 * 0.15) / 1e6)


class TestGlmCachePricing:
    def test_glm_models_carry_published_cached_input_prices(self):
        # docs.z.ai/guides/overview/pricing (cached-input $ per MTok).
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
