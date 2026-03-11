"""Test suite for model implementation coverage.

These tests verify the actual model implementations (AnthropicModel, GeminiModel,
OpenAICompatibleModel) using real API calls. No mocks are used.
"""

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    _mi,
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


class TestModelHelperFunctions:
    @pytest.mark.parametrize(
        "content,expected_reasoning,expected_answer",
        [
            (
                "<think>Reasoning process.</think>The answer is 42.",
                "Reasoning process.",
                "The answer is 42.",
            ),
            ("The answer is 42.", "", "The answer is 42."),
        ],
    )
    def test_extract_deepseek_reasoning(self, content, expected_reasoning, expected_answer):
        from kiss.core.models.openai_compatible_model import _extract_deepseek_reasoning

        reasoning, answer = _extract_deepseek_reasoning(content)
        assert reasoning == expected_reasoning
        assert answer == expected_answer

    def test_build_text_based_tools_prompt_empty(self):
        from kiss.core.models.openai_compatible_model import _build_text_based_tools_prompt

        assert _build_text_based_tools_prompt({}) == ""

    @pytest.mark.parametrize(
        "content,expected_count",
        [
            ('```json\n{"tool_calls": [{"name": "finish", "arguments": {}}]}\n```', 1),
            ("```json\n{invalid json}\n```", 0),
        ],
    )
    def test_parse_text_based_tool_calls(self, content, expected_count):
        from kiss.core.models.openai_compatible_model import _parse_text_based_tool_calls

        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == expected_count


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

    def test_text_embedding_004_is_gemini(self):

        m = model("text-embedding-004")
        assert isinstance(m, GeminiModel)
        assert m.model_name == "text-embedding-004"

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

    def test_minimax_m2_5_model_routing(self):
        m = model("minimax-m2.5")
        assert isinstance(m, OpenAICompatibleModel)
        assert m.model_name == "minimax-m2.5"

    def test_minimax_api_key_routing(self):
        from kiss.tests.conftest import get_required_api_key_for_model

        assert get_required_api_key_for_model("minimax-m2.5") == "MINIMAX_API_KEY"
        assert get_required_api_key_for_model("minimax-m2.5-lightning") == "MINIMAX_API_KEY"


class TestCachePricing:
    def test_anthropic_model_has_cache_pricing(self):
        info = MODEL_INFO["claude-sonnet-4"]
        assert info.cache_read_price_per_1M == pytest.approx(0.30)
        assert info.cache_write_price_per_1M == pytest.approx(3.75)

    def test_anthropic_cache_pricing_formula(self):
        for name, info in MODEL_INFO.items():
            if not name.startswith("claude-"):
                continue
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.1)
            assert info.cache_write_price_per_1M == pytest.approx(info.input_price_per_1M * 1.25)

    def test_openai_model_has_cache_read_pricing(self):
        info = MODEL_INFO["gpt-4.1-mini"]
        assert info.cache_read_price_per_1M == pytest.approx(0.20)
        assert info.cache_write_price_per_1M is None

    def test_openai_cache_read_pricing_formula(self):
        for name, info in MODEL_INFO.items():
            if not name.startswith(("gpt-", "o1", "o3", "o4", "codex-", "computer-use")):
                continue
            if name.startswith(("text-embedding", "openai/")):
                continue
            if not info.is_generation_supported:
                continue
            assert info.cache_read_price_per_1M == pytest.approx(info.input_price_per_1M * 0.5)

    def test_embedding_models_no_cache_pricing(self):
        for name in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"):
            info = MODEL_INFO[name]
            assert info.cache_read_price_per_1M is None
            assert info.cache_write_price_per_1M is None

    def test_openrouter_models_no_cache_pricing(self):
        for name, info in MODEL_INFO.items():
            if name.startswith("openrouter/") and not name.startswith(
                "openrouter/anthropic/"
            ):
                assert info.cache_read_price_per_1M is None, f"{name} should not have cache pricing"
                assert info.cache_write_price_per_1M is None, (
                    f"{name} should not have cache pricing"
                )

    def test_calculate_cost_unknown_model_with_cache_tokens(self):
        assert calculate_cost("unknown-model-xyz", 1000, 1000, 500, 500) == 0.0

    def test_model_info_explicit_cache_prices_override_loop(self):
        info = _mi(1000, 10.0, 20.0, cr=1.0, cw=2.0)
        assert info.cache_read_price_per_1M == 1.0
        assert info.cache_write_price_per_1M == 2.0


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

        tools = m._build_anthropic_tools_schema({"dummy_tool": dummy_tool})
        kwargs = m._build_create_kwargs(tools=tools)
        assert "cache_control" not in kwargs["tools"][-1]
        msg = m.conversation[0]
        assert isinstance(msg["content"], str)


class TestModelConfigBaseUrlOverride:

    @pytest.mark.timeout(60)
    @requires_openai_api_key
    def test_base_url_and_api_key_override_calls_endpoint_and_returns_response(self):
        api_key = config_module.DEFAULT_CONFIG.agent.api_keys.OPENAI_API_KEY
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
