"""Test suite for model implementation coverage.

These tests verify the actual model implementations (AnthropicModel, GeminiModel,
OpenAICompatibleModel) using real API calls. No mocks are used.
"""

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import (
    get_flaky_reason,
    is_model_flaky,
    model,
)
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

    def test_text_embedding_004_is_gemini(self):
        from kiss.core.models.gemini_model import GeminiModel

        m = model("text-embedding-004")
        assert isinstance(m, GeminiModel)
        assert m.model_name == "text-embedding-004"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
