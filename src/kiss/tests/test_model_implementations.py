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
    calculate_cost,
    get_flaky_reason,
    get_max_context_length,
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


class TestModelCommon:
    @pytest.mark.parametrize("model_name,_,__", MODEL_CONFIGS)
    @pytest.mark.timeout(60)
    def test_add_message_to_conversation(self, model_name, _, __):
        m = model(model_name)
        m.initialize("Hello")
        m.add_message_to_conversation("user", "Follow up")
        assert len(m.conversation) == 2


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

    @pytest.mark.timeout(60)
    def test_default_max_tokens(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        kwargs = m._build_create_kwargs()
        assert kwargs["max_tokens"] == 16384


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

    @pytest.mark.timeout(60)
    def test_add_message_with_usage_info(self):
        m = model("gpt-4.1-mini")
        m.initialize("Hello")
        m.set_usage_info_for_messages("Token usage: 50")
        m.add_message_to_conversation("user", "Test")
        assert "Token usage: 50" in m.conversation[-1]["content"]


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

    def test_build_text_based_tools_prompt_with_function(self):
        from kiss.core.models.openai_compatible_model import _build_text_based_tools_prompt

        def test_func(x: int) -> str:
            """A test function."""
            return str(x)

        prompt = _build_text_based_tools_prompt({"test_func": test_func})
        assert "test_func" in prompt
        assert "Available Tools" in prompt

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

    def test_calculate_cost_known_models(self):
        assert calculate_cost("gpt-5.2", 1_000_000, 0) == 1.75
        assert calculate_cost("gpt-5.2", 0, 1_000_000) == 14.0
        assert calculate_cost("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.0
        assert calculate_cost("gemini-2.5-pro", 1_000_000, 1_000_000) == 11.25

    def test_get_max_context_length_known_models(self):
        assert get_max_context_length("gpt-5.2") == 400000
        assert get_max_context_length("gpt-4.1-mini") == 128000
        assert get_max_context_length("claude-opus-4-6") == 200000
        assert get_max_context_length("gemini-2.5-pro") == 1048576

    def test_get_max_context_length_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_max_context_length("nonexistent-model-xyz")

    def test_all_models_have_valid_context_and_pricing(self):
        for name, info in MODEL_INFO.items():
            assert info.context_length > 0, f"{name}: invalid context_length"
            assert info.input_price_per_1M >= 0, f"{name}: invalid input_price"
            assert info.output_price_per_1M >= 0, f"{name}: invalid output_price"
            if info.is_embedding_supported:
                assert info.output_price_per_1M == 0.0, f"{name}: embedding should have 0 output"

    def test_text_embedding_004_is_gemini(self):
        from kiss.core.models.gemini_model import GeminiModel

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

    def test_minimax_m2_5_lightning_model_routing(self):
        m = model("minimax-m2.5-lightning")
        assert isinstance(m, OpenAICompatibleModel)
        assert m.model_name == "minimax-m2.5-lightning"

    def test_minimax_m2_5_calculate_cost(self):
        assert calculate_cost("minimax-m2.5", 1_000_000, 0) == 0.15
        assert calculate_cost("minimax-m2.5", 0, 1_000_000) == 1.20
        assert abs(calculate_cost("minimax-m2.5", 1_000_000, 1_000_000) - 1.35) < 1e-9

    def test_minimax_m2_5_lightning_calculate_cost(self):
        assert calculate_cost("minimax-m2.5-lightning", 1_000_000, 0) == 0.30
        assert calculate_cost("minimax-m2.5-lightning", 0, 1_000_000) == 2.40

    def test_minimax_m2_5_context_length(self):
        assert get_max_context_length("minimax-m2.5") == 1000000
        assert get_max_context_length("minimax-m2.5-lightning") == 1000000

    def test_minimax_api_key_routing(self):
        from kiss.tests.conftest import get_required_api_key_for_model

        assert get_required_api_key_for_model("minimax-m2.5") == "MINIMAX_API_KEY"
        assert get_required_api_key_for_model("minimax-m2.5-lightning") == "MINIMAX_API_KEY"


class TestModelConfigBaseUrlOverride:
    def test_base_url_only_returns_openai_compatible(self):
        m = model("any-model-name", model_config={"base_url": "https://custom.example/v1"})
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://custom.example/v1"
        assert m.api_key == ""
        assert m.model_name == "any-model-name"

    def test_base_url_and_api_key(self):
        m = model(
            "custom/foo",
            model_config={"base_url": "https://api.example/v1", "api_key": "sk-secret"},
        )
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://api.example/v1"
        assert m.api_key == "sk-secret"
        assert m.model_name == "custom/foo"

    def test_base_url_override_ignores_normal_routing(self):
        m = model("claude-haiku-4-5", model_config={"base_url": "https://proxy/v1"})
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://proxy/v1"
        assert m.model_name == "claude-haiku-4-5"

    def test_base_url_override_with_extra_config_passed_through(self):
        m = model(
            "gpt-4",
            model_config={
                "base_url": "https://local/v1",
                "temperature": 0.3,
                "max_tokens": 100,
            },
        )
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://local/v1"
        assert m.model_config.get("temperature") == 0.3
        assert m.model_config.get("max_tokens") == 100
        assert "base_url" not in m.model_config
        assert "api_key" not in m.model_config

    def test_base_url_override_api_key_not_in_model_config(self):
        m = model(
            "x",
            model_config={"base_url": "https://b/v1", "api_key": "key123"},
        )
        assert isinstance(m, OpenAICompatibleModel)
        assert m.api_key == "key123"
        assert "api_key" not in m.model_config
        assert "base_url" not in m.model_config

    def test_no_base_url_uses_normal_routing_openai(self):
        m = model("gpt-4.1-mini", model_config={"temperature": 0.5})
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://api.openai.com/v1"
        assert m.model_config.get("temperature") == 0.5

    def test_model_config_none_uses_normal_routing(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)

    def test_model_config_empty_no_override(self):
        m = model("gemini-2.0-flash", model_config={})
        assert isinstance(m, GeminiModel)

    def test_unknown_model_with_base_url_succeeds(self):
        m = model("unknown-vendor/model-xyz", model_config={"base_url": "https://custom/v1"})
        assert isinstance(m, OpenAICompatibleModel)
        assert m.model_name == "unknown-vendor/model-xyz"
        assert m.base_url == "https://custom/v1"

    def test_openrouter_with_base_url_override_uses_custom_url(self):
        m = model(
            "openrouter/foo",
            model_config={"base_url": "https://my-gateway/v1", "api_key": "mykey"},
        )
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://my-gateway/v1"
        assert m.api_key == "mykey"

    def test_token_callback_passed_through_with_base_url_override(self):
        received = []

        async def cb(t: str) -> None:
            received.append(t)

        m = model(
            "local",
            model_config={"base_url": "https://local/v1"},
            token_callback=cb,
        )
        assert m.token_callback is cb

    def test_base_url_trailing_slash_preserved(self):
        m = model("m", model_config={"base_url": "https://endpoint/v1/"})
        assert isinstance(m, OpenAICompatibleModel)
        assert m.base_url == "https://endpoint/v1/"

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
