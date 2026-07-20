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
        # >272k prompt tokens: $5 in, $22.50 out, $0.50 cached, $6.25 cache write.
        cost = calculate_cost("gpt-5.6-terra", 300_000, 10_000, 50_000, 40_000)
        expected = (
            300_000 * 5.0 + 10_000 * 22.50 + 50_000 * 0.50 + 40_000 * 6.25
        ) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_sol_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-sol", 300_000, 5_000, 0, 0)
        assert cost == pytest.approx((300_000 * 10.0 + 5_000 * 45.0) / 1e6)

    def test_gpt56_luna_long_context_cost(self):
        cost = calculate_cost("gpt-5.6-luna", 300_000, 5_000, 10_000, 10_000)
        expected = (300_000 * 2.0 + 5_000 * 9.0 + 10_000 * 0.20 + 10_000 * 2.50) / 1e6
        assert cost == pytest.approx(expected)

    def test_gpt56_xhigh_alias_gets_long_context_pricing(self):
        assert calculate_cost("gpt-5.6-sol-xhigh", 300_000, 5_000) == pytest.approx(
            calculate_cost("gpt-5.6-sol", 300_000, 5_000)
        )

    def test_gpt55_long_context_cache_writes_stay_free(self):
        # gpt-5.5 has no cache-write fee even in the long-context tier.
        cost = calculate_cost("gpt-5.5", 300_000, 5_000, 0, 40_000)
        assert cost == pytest.approx((300_000 * 10.0 + 5_000 * 45.0) / 1e6)

    def test_openai_threshold_is_272k_not_200k(self):
        # Official model pages: "Prompts with >272K input tokens are
        # priced at 2x input and 1.5x output".  A 250k prompt (between
        # the old 200k threshold and 272k) must stay at SHORT rates.
        cost = calculate_cost("gpt-5.6-sol", 250_000, 5_000)
        assert cost == pytest.approx((250_000 * 5.0 + 5_000 * 30.0) / 1e6)
        # 272k exactly is still short; 272,001 crosses into long.
        assert calculate_cost("gpt-5.5", 272_000, 0) == pytest.approx(
            272_000 * 5.0 / 1e6
        )
        assert calculate_cost("gpt-5.5", 272_001, 0) == pytest.approx(
            272_001 * 10.0 / 1e6
        )


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
            # Must be EXACTLY 0.0 (not None): a None cache-write price
            # falls back to the full input price in calculate_cost.
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
        # kimi-k2.5: $0.60 miss / $0.10 hit / $3.00 out per MTok.
        cost = calculate_cost("kimi-k2.5", 100_000, 0, 1_000_000, 0)
        assert cost == pytest.approx((100_000 * 0.60 + 1_000_000 * 0.10) / 1e6)

    def test_kimi_k3_cost(self):
        cost = calculate_cost("kimi-k3", 100_000, 10_000, 500_000, 0)
        expected = (100_000 * 3.0 + 10_000 * 15.0 + 500_000 * 0.30) / 1e6
        assert cost == pytest.approx(expected)

    def test_openrouter_kimi_k3_cache_read_not_overcharged(self):
        # openrouter.ai lists Moonshot's kimi-k3 cache read at $0.30 (0.1x),
        # so the generic 0.25x OpenRouter-Moonshot rule must not apply.
        info = MODEL_INFO["openrouter/moonshotai/kimi-k3"]
        assert info.cache_read_price_per_1M == pytest.approx(0.30)


class TestLongContextTierUsesPromptTokens:
    def test_large_output_does_not_trigger_long_context_tier(self):
        # 250k prompt + 100k output stays in the short-context tier even
        # though the total exceeds 272k (tiers key off prompt size).
        cost = calculate_cost("gpt-5.6-sol", 250_000, 100_000)
        assert cost == pytest.approx((250_000 * 5.0 + 100_000 * 30.0) / 1e6)

    def test_prompt_side_cache_tokens_count_toward_tier(self):
        # 250k fresh input + 60k cache reads = 310k prompt -> long tier.
        cost = calculate_cost("gpt-5.6-sol", 250_000, 1_000, 60_000, 0)
        expected = (250_000 * 10.0 + 1_000 * 45.0 + 60_000 * 1.00) / 1e6
        assert cost == pytest.approx(expected)

    def test_gemini_output_excluded_from_tier_decision(self):
        # 150k prompt + 100k output: total 250k > 200k, but the Gemini
        # tier keys off the prompt, so short rates apply.
        cost = calculate_cost("gemini-2.5-pro", 150_000, 100_000)
        assert cost == pytest.approx((150_000 * 1.25 + 100_000 * 10.0) / 1e6)


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
