"""Test suite for model implementation coverage.

These tests verify the actual model implementations (AnthropicModel, GeminiModel,
OpenAICompatibleModel) using real API calls. No mocks are used.
"""

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    ModelInfo,
    _emb,
    _mi,
    calculate_cost,
    get_flaky_reason,
    get_max_context_length,
    is_model_flaky,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import (
    add_numbers,
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
    requires_together_api_key,
    simple_test_tool,
)

MODEL_CONFIGS = [
    pytest.param("claude-haiku-4-5", "AnthropicModel", "4", marks=requires_anthropic_api_key),
    pytest.param("gemini-3-flash-preview", "GeminiModel", "6", marks=requires_gemini_api_key),
    pytest.param("gpt-4.1-mini", "OpenAICompatibleModel", "10", marks=requires_openai_api_key),
]


class TestModelCommon:
    @pytest.mark.parametrize("model_name,expected_class,_", MODEL_CONFIGS)
    def test_model_instantiation(self, model_name, expected_class, _):
        m = model(model_name)
        assert m.model_name == model_name
        assert expected_class in str(m)

    @pytest.mark.parametrize("model_name,_,expected_result", MODEL_CONFIGS)
    @pytest.mark.timeout(60)
    def test_generate_simple(self, model_name, _, expected_result):
        m = model(model_name)
        prompt = "What is 2 + 2? Reply with just the number."
        if "gemini" in model_name:
            prompt = "What is 3 + 3? Reply with just the number."
        elif "gpt" in model_name:
            prompt = "What is 5 + 5? Reply with just the number."
        m.initialize(prompt)
        content, response = m.generate()
        assert expected_result in content
        assert response is not None

    @pytest.mark.parametrize("model_name,_,__", MODEL_CONFIGS)
    @pytest.mark.timeout(60)
    def test_generate_with_tools(self, model_name, _, __):
        m = model(model_name)
        m.initialize("Use the add_numbers tool to add 5 and 7. Call the tool with a=5, b=7.")
        function_calls, content, response = m.generate_and_process_with_tools(
            {"add_numbers": add_numbers}
        )
        assert response is not None

    @pytest.mark.parametrize("model_name,_,__", MODEL_CONFIGS)
    @pytest.mark.timeout(60)
    def test_add_message_to_conversation(self, model_name, _, __):
        m = model(model_name)
        m.initialize("Hello")
        m.add_message_to_conversation("user", "Follow up")
        assert len(m.conversation) == 2

    @pytest.mark.parametrize("model_name,_,__", MODEL_CONFIGS)
    @pytest.mark.timeout(60)
    def test_extract_token_counts(self, model_name, _, __):
        m = model(model_name)
        m.initialize("Say hello")
        content, response = m.generate()
        input_tokens, output_tokens = m.extract_input_output_token_counts_from_response(response)
        assert input_tokens >= 0
        assert output_tokens >= 0


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
    def test_extract_text_from_blocks(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "tool_use", "name": "test", "id": "123"},
            {"type": "text", "text": "World"},
        ]
        assert m._extract_text_from_blocks(blocks) == "Hello World"


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

    @pytest.mark.timeout(60)
    def test_generate_with_system_prompt(self):
        m = model(
            "gemini-3-flash-preview",
            model_config={"system_instruction": "You are a helpful math assistant."},
        )
        m.initialize("What is 10 * 10?")
        content, response = m.generate()
        assert "100" in content


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


@requires_together_api_key
class TestTogetherAIModel:
    @pytest.mark.timeout(90)
    def test_generate_simple(self):
        m = model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        m.initialize("What is 7 + 7? Reply with just the number.")
        content, response = m.generate()
        assert "14" in content

    @pytest.mark.timeout(90)
    def test_generate_with_tools(self):
        m = model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        m.initialize("Use the simple_test_tool to echo 'world'. Call tool with message='world'.")
        function_calls, content, response = m.generate_and_process_with_tools(
            {"simple_test_tool": simple_test_tool}
        )
        assert response is not None


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
            ('{"tool_calls": [{"name": "test", "arguments": {"a": 1}}]}', 1),
            ("Regular text without tool calls.", 0),
            ("```json\n{invalid json}\n```", 0),
        ],
    )
    def test_parse_text_based_tool_calls(self, content, expected_count):
        from kiss.core.models.openai_compatible_model import _parse_text_based_tool_calls

        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == expected_count


class TestModelInfo:
    def test_all_models_have_required_fields(self):
        for name, info in MODEL_INFO.items():
            assert isinstance(info, ModelInfo), f"Model {name} should be ModelInfo"
            assert info.context_length is not None, f"Model {name} should have context_length"

    def test_calculate_cost_zero_tokens(self):
        assert calculate_cost("gpt-4.1-mini", 0, 0) == 0.0

    def test_get_max_context_length(self):
        assert get_max_context_length("gpt-4.1-mini") > 0

    def test_is_model_flaky(self):
        assert isinstance(is_model_flaky("gpt-4.1-mini"), bool)

    def test_get_flaky_reason_for_non_flaky_model(self):
        reason = get_flaky_reason("gpt-4.1-mini")
        assert reason is None or reason == ""

    def test_mi_helper(self):
        info = _mi(ctx=8000, inp=1.0, out=2.0, fc=True, emb=False, gen=True)
        assert info.context_length == 8000
        assert info.input_price_per_1M == 1.0
        assert info.output_price_per_1M == 2.0
        assert info.is_function_calling_supported
        assert info.is_generation_supported
        assert not info.is_embedding_supported

    def test_emb_helper(self):
        info = _emb(ctx=8000, inp=0.5)
        assert info.is_embedding_supported
        assert not info.is_generation_supported
        assert info.context_length == 8000

    def test_openrouter_model_instantiation(self):
        m = model("openrouter/anthropic/claude-3-haiku")
        assert isinstance(m, OpenAICompatibleModel)
        assert m.model_name == "openrouter/anthropic/claude-3-haiku"

    def test_text_embedding_004_is_gemini(self):
        from kiss.core.models.gemini_model import GeminiModel

        m = model("text-embedding-004")
        assert isinstance(m, GeminiModel)
        assert m.model_name == "text-embedding-004"


@requires_together_api_key
class TestDeepSeekR1:
    def test_deepseek_r1_is_reasoning_model(self):
        from kiss.core.models.openai_compatible_model import DEEPSEEK_REASONING_MODELS

        assert "DeepSeek-R1" in DEEPSEEK_REASONING_MODELS
        assert "deepseek/deepseek-r1" in DEEPSEEK_REASONING_MODELS

    def test_is_deepseek_reasoning_model_method(self):
        m = model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert isinstance(m, OpenAICompatibleModel)
        assert not m._is_deepseek_reasoning_model()

    @pytest.mark.timeout(300)
    def test_deepseek_r1_generate(self):
        m = model("deepseek-ai/DeepSeek-R1")
        m.initialize("What is 2 + 2? Think step by step, then give the answer.")
        content, response = m.generate()
        assert content is not None
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
