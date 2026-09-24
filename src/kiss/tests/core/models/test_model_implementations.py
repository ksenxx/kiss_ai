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

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    ModelInfo,
    _apply_cache_pricing,
    _openai_cache_read_multiplier,
    calculate_cost,
    model,
)
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)


def _mi_for_test(
    ctx: int,
    inp: float,
    out: float,
    fc: bool = True,
    emb: bool = False,
    gen: bool = True,
    cr: float | None = None,
    cw: float | None = None,
    cw1h: float | None = None,
    thinking: str | None = None,
) -> ModelInfo:
    """Local replacement for the old private ``_mi`` helper.

    The runtime model table now lives in MODEL_INFO.json so the
    short-syntax ``_mi`` helper is gone, but these tests still want to
    construct synthetic ``ModelInfo`` instances quickly.
    """
    return ModelInfo(ctx, inp, out, fc, emb, gen, cr, cw, cw1h, thinking)


MODEL_CONFIGS = [
    pytest.param("claude-haiku-4-5", "AnthropicModel", "4", marks=requires_anthropic_api_key),
    pytest.param("gemini-3-flash-preview", "GeminiModel", "6", marks=requires_gemini_api_key),
    pytest.param("gpt-4.1-mini", "OpenAICompatibleModel", "10", marks=requires_openai_api_key),
]


@requires_anthropic_api_key
class TestAnthropicModel:
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
        m = model("gemini-embedding-001")
        m.initialize("test")
        embedding = m.get_embedding("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)

    @pytest.mark.timeout(60)
    def test_get_embedding_explicit_model_overrides_instance(self):
        """``embedding_model`` wins over the instance's own model name."""
        m = model("gemini-3-flash-preview")
        m.initialize("test")
        embedding = m.get_embedding("Hello world", embedding_model="gemini-embedding-001")
        assert len(embedding) > 0

    @pytest.mark.timeout(60)
    def test_get_embedding_non_embedding_model_fails(self):
        """A chat model embeds under its own name and is rejected by the API.

        Regression: the old fallback to ``gemini-embedding-001`` made every
        Gemini model look like an embedder, so ``update_models.py`` flagged
        TTS and transcription models ``emb: true``.
        """
        m = model("gemini-3-flash-preview")
        m.initialize("test")
        with pytest.raises(KISSError, match="gemini-3-flash-preview"):
            m.get_embedding("Hello world")

    @pytest.mark.timeout(120)
    def test_update_models_embedding_probe(self):
        """The catalog probe passes only for a real Gemini embedding model."""
        from kiss.scripts.update_models import test_embedding as probe

        assert probe("gemini-embedding-001") is True
        assert probe("gemini-3-flash-preview") is False


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
    def test_all_models_have_valid_context_and_pricing(self):
        for name, info in MODEL_INFO.items():
            assert info.context_length > 0, f"{name}: invalid context_length"
            assert info.input_price_per_1M >= 0, f"{name}: invalid input_price"
            assert info.output_price_per_1M >= 0, f"{name}: invalid output_price"
            if info.is_embedding_supported:
                assert info.output_price_per_1M == 0.0, f"{name}: embedding should have 0 output"

    def test_glm_4_6_in_model_info(self):
        assert "glm-4.6" in MODEL_INFO
        info = MODEL_INFO["glm-4.6"]
        assert info.context_length == 204800
        assert info.input_price_per_1M == 0.60
        assert info.output_price_per_1M == 2.20
        assert info.is_function_calling_supported is True
        assert info.is_generation_supported is True
        assert info.is_embedding_supported is False

    def test_glm_4_5_flash_in_model_info(self):
        assert "glm-4.5-flash" in MODEL_INFO
        info = MODEL_INFO["glm-4.5-flash"]
        assert info.input_price_per_1M == 0.0
        assert info.output_price_per_1M == 0.0
        assert info.is_function_calling_supported is True

    def test_moonshot_v1_32k_in_model_info(self):
        assert "moonshot-v1-32k" in MODEL_INFO
        info = MODEL_INFO["moonshot-v1-32k"]
        assert info.context_length == 32768
        assert info.is_function_calling_supported is True

    def test_kimi_in_model_info(self):
        assert "kimi-k2.6" in MODEL_INFO

    def test_every_model_asks_for_its_own_provider_key(self):
        """Every catalog entry's key must match the provider that routes it.

        The skip decision in ``has_api_key_for_model`` is only as good as
        this agreement: a vendor added to the routing registry with no
        matching credential here would make its live tests run without
        one (hanging on a 401) or skip when the key is present.
        """
        from kiss.core.models.model_info import get_model_provider
        from kiss.tests.conftest import get_required_api_key_for_model

        expected_by_provider = {
            "OpenAI": "OPENAI_API_KEY",
            "OpenRouter": "OPENROUTER_API_KEY",
            "Together": "TOGETHER_API_KEY",
            "Z.AI": "ZAI_API_KEY",
            "Moonshot": "MOONSHOT_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Gemini": "GEMINI_API_KEY",
            "Claude Code CLI": None,
            "Codex CLI": None,
            "Unknown": None,
        }
        seen = set()
        for name in MODEL_INFO:
            provider = get_model_provider(name)
            assert provider in expected_by_provider, f"{name} → {provider}"
            assert get_required_api_key_for_model(name) == expected_by_provider[provider]
            seen.add(provider)
        assert {"OpenAI", "Anthropic", "Gemini", "Codex CLI"} <= seen


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
            read_mult = 0.025 if name.startswith(("claude-fable-5-1", "claude-mythos-5-1")) else 0.1
            assert info.cache_read_price_per_1M == pytest.approx(
                info.input_price_per_1M * read_mult
            ), name
            assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)
            assert info.cache_write_1h_price_per_1M == pytest.approx(info.input_price_per_1M * 2.0)

    def test_fable_51_cache_read_is_quarter_of_a_tenth(self):
        """platform.claude.com pricing: Fable 5.1 cache hits are $0.25/MTok on a $10 base."""
        info = MODEL_INFO["claude-fable-5-1"]
        assert info.input_price_per_1M == pytest.approx(10.0)
        assert info.cache_read_price_per_1M == pytest.approx(0.25)
        assert info.cache_write_price_per_1M == pytest.approx(12.5)
        assert info.cache_write_1h_price_per_1M == pytest.approx(20.0)
        assert MODEL_INFO["claude-fable-5"].cache_read_price_per_1M == pytest.approx(1.0)
        cost = calculate_cost("claude-fable-5-1", 1_000, 500, 400_000, 2_000)
        assert cost == pytest.approx(
            (1_000 * 10.0 + 500 * 50.0 + 400_000 * 0.25 + 2_000 * 12.5) / 1e6
        )
        for name in (
            "openrouter/anthropic/claude-fable-5.1",
            "openrouter/~anthropic/claude-fable-latest",
        ):
            assert MODEL_INFO[name].cache_read_price_per_1M == pytest.approx(0.25), name
            assert MODEL_INFO[name].cache_write_price_per_1M == pytest.approx(12.5), name

    def test_openai_model_has_cache_read_pricing(self):
        info = MODEL_INFO["gpt-4.1-mini"]
        assert info.cache_read_price_per_1M == pytest.approx(0.10)
        assert info.cache_write_price_per_1M == 0.0

    def test_openai_gpt41_and_o3_cache_read_is_quarter(self):
        # o3-deep-research left the catalog in the 2026-09-22 refresh (OpenAI
        # no longer lists it), so it is no longer part of this set.
        for name in ("gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"):
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
        for name in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
            assert info.cache_write_price_per_1M == 0.0

    def test_openrouter_provider_cache_pricing(self):
        """openrouter.ai/api/v1/models ``pricing.input_cache_read`` / ``input_cache_write``
        (2026-09) are stored verbatim in the catalog for every gateway model."""
        g = MODEL_INFO["openrouter/google/gemini-2.5-pro"]
        assert g.cache_read_price_per_1M == pytest.approx(0.125)
        assert g.cache_read_price_per_1M == pytest.approx(g.input_price_per_1M * 0.1)
        assert MODEL_INFO["openrouter/openai/gpt-5.5"].cache_read_price_per_1M == pytest.approx(
            MODEL_INFO["openrouter/openai/gpt-5.5"].input_price_per_1M * 0.1
        )
        assert MODEL_INFO["openrouter/openai/gpt-4o"].cache_read_price_per_1M == pytest.approx(
            MODEL_INFO["openrouter/openai/gpt-4o"].input_price_per_1M * 0.5
        )
        # DeepSeek's OpenRouter prices are floating provider averages that
        # every ``update_models.py`` refresh moves, so only the cache-read /
        # input ratios (fixed by the vendor) are pinned, not the absolute values.
        d = MODEL_INFO["openrouter/deepseek/deepseek-v4-flash"]
        assert 0 < d.input_price_per_1M < 1
        assert d.cache_read_price_per_1M == pytest.approx(d.input_price_per_1M * 0.2, rel=0.05)
        assert d.cache_write_price_per_1M is None
        p = MODEL_INFO["openrouter/deepseek/deepseek-v4-pro"]
        assert 0 < p.input_price_per_1M < 5
        assert p.cache_read_price_per_1M == pytest.approx(p.input_price_per_1M / 12, rel=0.05)
        q = MODEL_INFO["openrouter/qwen/qwen3.8-max-0902"]
        assert q.cache_read_price_per_1M == pytest.approx(0.25)
        assert q.cache_write_price_per_1M == pytest.approx(2.5)
        assert MODEL_INFO["openrouter/qwen/qwen3.7-plus"].cache_read_price_per_1M == pytest.approx(
            0.064
        )
        assert MODEL_INFO[
            "openrouter/moonshotai/kimi-k2.5"
        ].cache_read_price_per_1M == pytest.approx(0.07)
        assert MODEL_INFO["openrouter/x-ai/grok-4.3"].cache_read_price_per_1M == pytest.approx(0.2)
        assert MODEL_INFO["openrouter/x-ai/grok-4.5"].cache_read_price_per_1M == pytest.approx(0.3)
        assert MODEL_INFO["openrouter/x-ai/grok-4.5-high"].cache_read_price_per_1M == pytest.approx(
            0.3
        )
        # A vendor OpenRouter lists without a cache-read price bills cache reads at input price.
        m = MODEL_INFO["openrouter/cohere/command-r7b-12-2024"]
        assert m.cache_read_price_per_1M is None
        assert calculate_cost("openrouter/cohere/command-r7b-12-2024", 0, 0, 1_000_000, 0) == (
            pytest.approx(m.input_price_per_1M)
        )

    def test_openrouter_google_prefix_fallback_is_a_tenth(self):
        """A gateway Gemini entry without catalog cache prices gets Google's 0.1x rate."""
        info = _mi_for_test(1_000_000, 2.0, 12.0)
        _apply_cache_pricing("openrouter/google/gemini-9-pro", info)
        assert info.cache_read_price_per_1M == pytest.approx(0.2)
        assert info.cache_write_price_per_1M == 0.0

    def test_openrouter_unknown_vendor_has_no_cache_discount(self):
        """No prefix rule and no catalog price: the entry keeps ``None`` (full input price)."""
        for name in (
            "openrouter/deepseek/deepseek-v9",
            "openrouter/qwen/qwen9",
            "openrouter/x-ai/grok-9",
        ):
            info = _mi_for_test(100_000, 1.0, 2.0)
            _apply_cache_pricing(name, info)
            assert info.cache_read_price_per_1M is None, name
            assert info.cache_write_price_per_1M is None, name

    def test_openrouter_anthropic_cache_pricing(self):
        info = MODEL_INFO["openrouter/anthropic/claude-opus-4.8"]
        assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
        assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)
        assert info.cache_write_1h_price_per_1M == pytest.approx(info.input_price_per_1M * 2.0)

    def test_gpt_oss_openrouter_uses_openrouter_cache_read_price(self):
        """gpt-oss has no OpenAI-native cache rule; OpenRouter lists $0.075 read on $0.15 input."""
        info = MODEL_INFO["openrouter/openai/gpt-oss-120b"]
        assert info.cache_read_price_per_1M == pytest.approx(0.075)
        assert info.cache_write_price_per_1M is None

    def test_undocumented_providers_have_no_cache_pricing(self):
        for name in ("glm-4-32b-0414-128k", "deepseek-ai/DeepSeek-V3-0324", "Qwen/Qwen3.6-Plus"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M is None, f"{name} should not have cache pricing"
            assert info.cache_write_price_per_1M is None

    def test_embedding_models_no_cache_pricing(self):
        for name in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M is None
            assert info.cache_write_price_per_1M is None

    def test_calculate_cost_numeric_per_family(self):
        assert calculate_cost("gpt-5.5", 0, 0, 1_000_000, 0) == pytest.approx(1.00)
        assert calculate_cost("gpt-4o", 0, 0, 1_000_000, 0) == pytest.approx(1.25)
        assert calculate_cost("o3", 0, 0, 1_000_000, 0) == pytest.approx(0.50)
        assert calculate_cost("gemini-2.5-pro", 0, 0, 1_000_000, 0) == pytest.approx(0.25)
        assert calculate_cost("claude-opus-4-8", 0, 0, 0, 1_000_000) == pytest.approx(6.25)
        assert calculate_cost("claude-opus-4-8", 0, 0, 0, 0, 1_000_000) == pytest.approx(10.00)
        assert calculate_cost("gpt-5.4", 1_000_000, 1_000_000, 1_000_000, 0) == pytest.approx(
            5.0 + 22.5 + 0.5
        )

    def test_calculate_cost_strips_provider_prefix(self):
        assert calculate_cost("openai/gpt-5.5", 0, 0, 1_000_000, 0) == pytest.approx(1.00)

    def test_apply_cache_pricing_respects_existing_prices(self):
        info = _mi_for_test(1000, 10.0, 20.0, cr=1.0, cw=2.0)
        _apply_cache_pricing("gpt-4o", info)
        assert info.cache_read_price_per_1M == 1.0
        assert info.cache_write_price_per_1M == 2.0

    def test_model_info_explicit_cache_prices_override_loop(self):
        info = _mi_for_test(1000, 10.0, 20.0, cr=1.0, cw=2.0)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
