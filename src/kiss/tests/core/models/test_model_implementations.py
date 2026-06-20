# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test suite for model implementation coverage.

These tests verify the actual model implementations (AnthropicModel, GeminiModel,
OpenAICompatibleModel) using real API calls. No mocks are used.
"""

from types import SimpleNamespace

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    _apply_cache_pricing,
    _mi,
    _openai_cache_read_multiplier,
    calculate_cost,
    get_flaky_reason,
    is_model_flaky,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)

MODEL_CONFIGS = [
    pytest.param("claude-haiku-4-5", "AnthropicModel", "4", marks=requires_anthropic_api_key),
    pytest.param("gemini-3-flash-preview", "GeminiModel", "6", marks=requires_gemini_api_key),
    pytest.param("gpt-4.1-mini", "OpenAICompatibleModel", "10", marks=requires_openai_api_key),
]


@requires_anthropic_api_key
class TestAnthropicModel:
    @pytest.mark.timeout(60)
    def test_get_embedding_raises_error(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        with pytest.raises(KISSError, match="(?i)embedding"):
            m.get_embedding("test text")

    @pytest.mark.timeout(60)
    def test_normalize_content_blocks(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        assert m._normalize_content_blocks(None) == []
        input_blocks = [{"type": "text", "text": "Hello"}]
        assert m._normalize_content_blocks(input_blocks) == input_blocks

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "config_key,config_value,expected_key,expected_value",
        [
            ("max_completion_tokens", 500, "max_tokens", 500),
            ("stop", "END", "stop_sequences", ["END"]),
            ("stop", ["END", "STOP"], "stop_sequences", ["END", "STOP"]),
        ],
    )
    def test_build_create_kwargs_options(
        self, config_key, config_value, expected_key, expected_value
    ):
        m = model("claude-haiku-4-5", model_config={config_key: config_value})
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        kwargs = m._build_create_kwargs()
        assert kwargs.get(expected_key) == expected_value


class TestAnthropicTokenExtraction:
    def test_split_cache_creation_tokens_are_preserved(self):
        m = AnthropicModel("claude-opus-4-8", api_key="test")
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=40,
                ephemeral_1h_input_tokens=50,
            ),
        )
        response = SimpleNamespace(usage=usage)
        assert m.extract_input_output_token_counts_from_response(response) == (
            100,
            20,
            30,
            40,
            50,
        )

    def test_aggregate_cache_creation_is_conservative_one_hour(self):
        m = AnthropicModel("claude-opus-4-8", api_key="test")
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )
        response = SimpleNamespace(usage=usage)
        assert m.extract_input_output_token_counts_from_response(response) == (
            100,
            20,
            30,
            0,
            50,
        )


@requires_gemini_api_key
class TestGeminiModel:
    @pytest.mark.timeout(60)
    def test_get_embedding(self):
        m = model("gemini-3-flash-preview")
        m.initialize("test")
        try:
            embedding = m.get_embedding("Hello world", embedding_model="models/text-embedding-005")
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert isinstance(embedding[0], float)
        except KISSError as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                pytest.skip(f"Embedding model not available: {e}")
            raise


@requires_openai_api_key
class TestOpenAIModel:
    @pytest.mark.timeout(60)
    def test_get_embedding(self):
        m = model("text-embedding-3-small")
        m.initialize("test")
        embedding = m.get_embedding("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)


class TestModelInfo:
    def test_is_model_flaky(self):
        assert isinstance(is_model_flaky("gpt-4.1-mini"), bool)

    def test_get_flaky_reason_for_non_flaky_model(self):
        reason = get_flaky_reason("gpt-4.1-mini")
        assert reason is None or reason == ""

    def test_all_models_have_valid_context_and_pricing(self):
        for name, info in MODEL_INFO.items():
            assert info.context_length > 0, f"{name}: invalid context_length"
            assert info.input_price_per_1M >= 0, f"{name}: invalid input_price"
            assert info.output_price_per_1M >= 0, f"{name}: invalid output_price"
            if info.is_embedding_supported:
                assert info.output_price_per_1M == 0.0, f"{name}: embedding should have 0 output"

    def test_minimax_m2_5_in_model_info(self):
        assert "minimax-m2.5" in MODEL_INFO
        info = MODEL_INFO["minimax-m2.5"]
        assert info.context_length == 1000000
        assert info.input_price_per_1M == 0.15
        assert info.output_price_per_1M == 1.20
        assert info.is_function_calling_supported is True
        assert info.is_generation_supported is True
        assert info.is_embedding_supported is False

    def test_minimax_m2_5_lightning_in_model_info(self):
        assert "minimax-m2.5-lightning" in MODEL_INFO
        info = MODEL_INFO["minimax-m2.5-lightning"]
        assert info.context_length == 1000000
        assert info.input_price_per_1M == 0.30
        assert info.output_price_per_1M == 2.40
        assert info.is_function_calling_supported is True

    def test_minimax_m2_5_openrouter_in_model_info(self):
        assert "openrouter/minimax/minimax-m2.5" in MODEL_INFO

    def test_minimax_api_key_routing(self):
        from kiss.tests.conftest import get_required_api_key_for_model

        assert get_required_api_key_for_model("minimax-m2.5") == "MINIMAX_API_KEY"
        assert get_required_api_key_for_model("minimax-m2.5-lightning") == "MINIMAX_API_KEY"


class TestCachePricing:
    def test_anthropic_model_has_cache_pricing(self):
        info = MODEL_INFO["claude-sonnet-4-5"]
        assert info.cache_read_price_per_1M == pytest.approx(0.30)
        assert info.cache_write_price_per_1M == pytest.approx(3.75)
        assert info.cache_write_1h_price_per_1M == pytest.approx(6.00)

    def test_anthropic_cache_pricing_formula(self):
        for name, info in MODEL_INFO.items():
            if not name.startswith("claude-"):
                continue
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
            assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)
            assert info.cache_write_1h_price_per_1M == pytest.approx(info.input_price_per_1M * 2.0)

    def test_openai_model_has_cache_read_pricing(self):
        # gpt-4.1 family caches at 0.25x input; OpenAI never charges for writes.
        info = MODEL_INFO["gpt-4.1-mini"]
        assert info.cache_read_price_per_1M == pytest.approx(0.10)  # 0.25 * 0.40
        assert info.cache_write_price_per_1M == 0.0

    def test_openai_gpt41_and_o3_cache_read_is_quarter(self):
        for name in ("gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini", "o3-deep-research"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.25)

    def test_openai_gpt4o_and_o1_cache_read_is_half(self):
        for name in ("gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "o1", "o3-mini"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.5)

    def test_openai_cache_read_multiplier_classification(self):
        assert _openai_cache_read_multiplier("gpt-5.4") == 0.10
        assert _openai_cache_read_multiplier("gpt-5.4-pro") == 1.0
        assert _openai_cache_read_multiplier("gpt-chat-latest") == 0.10
        assert _openai_cache_read_multiplier("gpt-latest") == 0.10
        assert _openai_cache_read_multiplier("gpt-mini-latest") == 0.10
        assert _openai_cache_read_multiplier("gpt-image-1-mini") == 0.10
        assert _openai_cache_read_multiplier("gpt-image-2") == 0.25
        assert _openai_cache_read_multiplier("gpt-4.1") == 0.25
        assert _openai_cache_read_multiplier("o3") == 0.25
        assert _openai_cache_read_multiplier("o4-mini") == 0.25
        assert _openai_cache_read_multiplier("o1") == 0.50
        assert _openai_cache_read_multiplier("o3-mini") == 0.50
        assert _openai_cache_read_multiplier("gpt-4o") == 0.50

    def test_gemini_cache_pricing(self):
        # Direct Gemini context cache read = 0.1x input; no separate write cost.
        for name in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
            assert info.cache_write_price_per_1M == 0.0

    def test_openrouter_provider_cache_pricing(self):
        # OpenRouter Gemini implicit caching reads at 0.25x.
        g = MODEL_INFO["openrouter/google/gemini-2.5-pro"]
        assert g.cache_read_price_per_1M == pytest.approx(g.input_price_per_1M * 0.25)
        # OpenRouter passthrough OpenAI mirrors OpenAI per-model rates.
        assert MODEL_INFO["openrouter/openai/gpt-5.5"].cache_read_price_per_1M == pytest.approx(
            MODEL_INFO["openrouter/openai/gpt-5.5"].input_price_per_1M * 0.1
        )
        assert MODEL_INFO["openrouter/openai/gpt-4o"].cache_read_price_per_1M == pytest.approx(
            MODEL_INFO["openrouter/openai/gpt-4o"].input_price_per_1M * 0.5
        )
        # DeepSeek V4: cache-hit pricing is much cheaper than the old 0.1x rule.
        d = MODEL_INFO["openrouter/deepseek/deepseek-v4-flash"]
        assert d.cache_read_price_per_1M == pytest.approx(d.input_price_per_1M * 0.02)
        assert d.cache_write_price_per_1M == pytest.approx(d.input_price_per_1M)
        # Qwen implicit cache reads at 0.2x; writes remain explicit-cache 1.25x.
        q = MODEL_INFO["openrouter/qwen/qwen3-max"]
        assert q.cache_read_price_per_1M == pytest.approx(q.input_price_per_1M * 0.2)
        assert q.cache_write_price_per_1M == pytest.approx(q.input_price_per_1M * 1.25)
        # Moonshot / Grok: read 0.25x, no write cost.
        for name in ("openrouter/moonshotai/kimi-k2.5", "openrouter/x-ai/grok-4.3"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.25)
            assert info.cache_write_price_per_1M == 0.0

    def test_openrouter_anthropic_cache_pricing(self):
        info = MODEL_INFO["openrouter/anthropic/claude-opus-4.8"]
        assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
        assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)
        assert info.cache_write_1h_price_per_1M == pytest.approx(info.input_price_per_1M * 2.0)

    def test_gpt_oss_openrouter_has_no_cache_pricing(self):
        info = MODEL_INFO["openrouter/openai/gpt-oss-120b"]
        assert info.cache_read_price_per_1M is None
        assert info.cache_write_price_per_1M is None

    def test_undocumented_providers_have_no_cache_pricing(self):
        # Providers without a documented cache discount fall back to full input price.
        for name in ("minimax-m2.5", "deepseek-ai/DeepSeek-V3-0324", "Qwen/Qwen3.6-Plus"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M is None, f"{name} should not have cache pricing"
            assert info.cache_write_price_per_1M is None

    def test_embedding_models_no_cache_pricing(self):
        for name in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M is None
            assert info.cache_write_price_per_1M is None

    def test_calculate_cost_unknown_model_with_cache_tokens(self):
        with pytest.raises(KISSError, match="unknown model"):
            calculate_cost("unknown-model-xyz", 1000, 1000, 500, 500)
        assert calculate_cost("unknown-model-xyz", 0, 0, 0, 0) == 0.0

    def test_calculate_cost_numeric_per_family(self):
        # 1M cache-read tokens -> price equals the per-1M cache-read rate in USD.
        assert calculate_cost("gpt-5.5", 0, 0, 1_000_000, 0) == pytest.approx(1.00)
        assert calculate_cost("gpt-4o", 0, 0, 1_000_000, 0) == pytest.approx(1.25)
        assert calculate_cost("o3", 0, 0, 1_000_000, 0) == pytest.approx(0.50)
        assert calculate_cost("gemini-2.5-pro", 0, 0, 1_000_000, 0) == pytest.approx(0.25)
        # Anthropic cache writes distinguish 5-minute (1.25x) and 1-hour (2.0x).
        assert calculate_cost("claude-opus-4-8", 0, 0, 0, 1_000_000) == pytest.approx(6.25)
        assert calculate_cost("claude-opus-4-8", 0, 0, 0, 0, 1_000_000) == pytest.approx(10.00)
        # Combined: gpt-5.4 applies long-context tiers above 200k tokens.
        assert calculate_cost("gpt-5.4", 1_000_000, 1_000_000, 1_000_000, 0) == pytest.approx(
            5.0 + 22.5 + 0.5
        )

    def test_calculate_cost_strips_provider_prefix(self):
        # Harbor-style provider prefix resolves to the same pricing.
        assert calculate_cost("openai/gpt-5.5", 0, 0, 1_000_000, 0) == pytest.approx(1.00)

    def test_apply_cache_pricing_respects_existing_prices(self):
        info = _mi(1000, 10.0, 20.0, cr=1.0, cw=2.0)
        _apply_cache_pricing("gpt-4o", info)
        assert info.cache_read_price_per_1M == 1.0
        assert info.cache_write_price_per_1M == 2.0

    def test_model_info_explicit_cache_prices_override_loop(self):
        info = _mi(1000, 10.0, 20.0, cr=1.0, cw=2.0)
        assert info.cache_read_price_per_1M == 1.0
        assert info.cache_write_price_per_1M == 2.0

    def test_long_context_tiers_apply_after_threshold(self):
        expected_openai = (201_000 * 10.00 + 201_000 * 45.00 + 201_000 * 1.00) / 1_000_000
        assert calculate_cost("gpt-5.5", 201_000, 201_000, 201_000, 0) == pytest.approx(
            expected_openai
        )
        expected_gemini = (201_000 * 2.50 + 201_000 * 15.00 + 201_000 * 0.25) / 1_000_000
        assert calculate_cost("gemini-2.5-pro", 201_000, 201_000, 201_000, 0) == pytest.approx(
            expected_gemini
        )

    def test_image_model_prices_updated_to_current_text_defaults(self):
        assert MODEL_INFO["gpt-image-1-mini"].input_price_per_1M == pytest.approx(2.00)
        assert MODEL_INFO["gpt-image-1-mini"].output_price_per_1M == pytest.approx(8.00)
        assert MODEL_INFO["gpt-image-1.5"].output_price_per_1M == pytest.approx(32.00)


@requires_anthropic_api_key
class TestAnthropicCacheControl:

    @pytest.mark.timeout(60)
    def test_cache_control_disabled_via_model_config(self):
        m = model("claude-haiku-4-5", model_config={"enable_cache": False})
        assert isinstance(m, AnthropicModel)
        m.initialize("test prompt")

        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        openai_schema = m._build_openai_tools_schema({"dummy_tool": dummy_tool})
        tools = m._build_anthropic_tools_schema(openai_schema)
        kwargs = m._build_create_kwargs(tools=tools)
        assert "cache_control" not in kwargs["tools"][-1]
        msg = m.conversation[0]
        assert isinstance(msg["content"], str)


class TestModelConfigBaseUrlOverride:

    @pytest.mark.timeout(60)
    @requires_openai_api_key
    def test_base_url_and_api_key_override_calls_endpoint_and_returns_response(self):
        api_key = config_module.DEFAULT_CONFIG.OPENAI_API_KEY
        m = model(
            "gpt-4.1-mini",
            model_config={
                "base_url": "https://api.openai.com/v1",
                "api_key": api_key,
            },
        )
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("Reply with exactly the word OK and nothing else.")
        text, _ = m.generate()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "ok" in text.lower().strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
