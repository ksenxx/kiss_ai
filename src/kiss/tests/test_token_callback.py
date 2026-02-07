"""Integration tests for the async token_callback streaming feature.

These tests use REAL API calls -- no mocks. Each provider is tested for:
  1. Simple (non-tool) generation with callback.
  2. Tool-calling generation with callback.
  3. Callback receives non-empty tokens that concatenate to match the response.
  4. KISSAgent-level integration (non-agentic and agentic).
  5. No callback (None) still works as before (regression guard).
"""

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import TokenCallback
from kiss.core.models.model_info import model
from kiss.tests.conftest import (
    add_numbers,
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)


def _make_collector() -> tuple[TokenCallback, list[str]]:
    tokens: list[str] = []

    async def _callback(token: str) -> None:
        tokens.append(token)

    return _callback, tokens


PROVIDER_CASES = [
    ("claude-haiku-4-5", "4", 1, requires_anthropic_api_key),
    ("gpt-4.1-mini", "4", 0, requires_openai_api_key),
    ("gemini-2.0-flash", "4", 0, requires_gemini_api_key),
]

PROVIDER_MODEL_PARAMS = [
    pytest.param(model_name, expected_answer, marks=mark)
    for model_name, expected_answer, _, mark in PROVIDER_CASES
]
PROVIDER_MODEL_ONLY_PARAMS = [
    pytest.param(model_name, marks=mark) for model_name, _, _, mark in PROVIDER_CASES
]
PROVIDER_TOKEN_PARAMS = [
    pytest.param(model_name, min_tokens, marks=mark)
    for model_name, _, min_tokens, mark in PROVIDER_CASES
]


class TestProviderTokenCallback:
    @pytest.mark.parametrize("model_name,expected_answer", PROVIDER_MODEL_PARAMS)
    @pytest.mark.timeout(60)
    def test_generate_streams_tokens(self, model_name, expected_answer):
        callback, tokens = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("What is 2 + 2? Reply with just the number.")
        content, response = m.generate()
        assert response is not None
        assert expected_answer in content
        assert len(tokens) > 0
        assert "".join(tokens) == content

    @pytest.mark.parametrize("model_name", PROVIDER_MODEL_ONLY_PARAMS)
    @pytest.mark.timeout(60)
    def test_generate_with_tools_streams_tokens(self, model_name):
        callback, tokens = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Use the add_numbers tool to add 3 and 4. Call the tool with a=3, b=4.")
        _, content, response = m.generate_and_process_with_tools({"add_numbers": add_numbers})
        assert response is not None
        if content:
            assert len(tokens) > 0

    @pytest.mark.parametrize("model_name,expected_answer", PROVIDER_MODEL_PARAMS)
    @pytest.mark.timeout(60)
    def test_no_callback_still_works(self, model_name, expected_answer):
        m = model(model_name, token_callback=None)
        m.initialize("What is 2 + 2? Reply with just the number.")
        content, response = m.generate()
        assert expected_answer in content

    @pytest.mark.parametrize("model_name,min_tokens", PROVIDER_TOKEN_PARAMS)
    @pytest.mark.timeout(60)
    def test_token_counts_with_streaming(self, model_name, min_tokens):
        callback, _ = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Say hello in one word.")
        _, response = m.generate()
        input_tokens, output_tokens = m.extract_input_output_token_counts_from_response(response)
        assert input_tokens >= min_tokens
        assert output_tokens >= min_tokens


@requires_openai_api_key
class TestKISSAgentTokenCallback:
    @pytest.mark.timeout(60)
    def test_non_agentic_with_callback(self):
        callback, tokens = _make_collector()
        agent = KISSAgent("test-non-agentic")
        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 7 + 7? Reply with just the number.",
            is_agentic=False,
            token_callback=callback,
        )
        assert "14" in result
        assert len(tokens) > 0

    @pytest.mark.timeout(120)
    def test_agentic_with_callback_streams_tool_output(self):
        callback, tokens = _make_collector()
        agent = KISSAgent("test-agentic-tool-output")

        def simple_calculator(expression: str) -> str:
            """Evaluate a simple arithmetic expression.

            Args:
                expression: The arithmetic expression to evaluate

            Returns:
                The result of the expression as a string
            """
            try:
                compiled = compile(expression, "<string>", "eval")
                return str(eval(compiled, {"__builtins__": {}}, {}))
            except Exception as e:
                return f"Error: {e}"

        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 123 * 456? Use the calculator tool.",
            tools=[simple_calculator],
            is_agentic=True,
            max_steps=5,
            token_callback=callback,
        )
        assert result is not None
        assert "56088" in "".join(tokens)

    @pytest.mark.timeout(60)
    def test_no_callback_regression(self):
        agent = KISSAgent("test-no-callback")
        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 9 + 9? Reply with just the number.",
            is_agentic=False,
        )
        assert "18" in result


ALL_PROVIDER_MODELS = PROVIDER_MODEL_ONLY_PARAMS


class TestTokenCallbackCrossProvider:
    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(60)
    def test_callback_receives_only_strings(self, model_name):
        callback, tokens = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Tell me a very short joke (one sentence).")
        m.generate()
        assert len(tokens) > 0
        for t in tokens:
            assert isinstance(t, str)
            assert len(t) > 0

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(60)
    def test_conversation_state_preserved_with_callback(self, model_name):
        callback, _ = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Say hello.")
        content, _ = m.generate()
        assert len(m.conversation) == 2
        assert m.conversation[-1]["role"] == "assistant"


class TestToolOutputStreaming:
    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_tool_output_appears_in_callback(self, model_name):
        callback, tokens = _make_collector()
        agent = KISSAgent("test-tool-output-stream")
        result = agent.run(
            model_name=model_name,
            prompt_template=(
                "What is 17 + 25? Use the add_numbers tool with a=17, b=25, "
                "then call finish with the answer."
            ),
            tools=[add_numbers],
            is_agentic=True,
            max_steps=5,
            token_callback=callback,
        )
        assert result is not None
        assert "42" in "".join(tokens)

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_tool_error_output_streamed(self, model_name):
        callback, tokens = _make_collector()
        agent = KISSAgent("test-tool-error-stream")

        def failing_tool(x: str) -> str:
            """A tool that always fails.

            Args:
                x: Any input string.

            Returns:
                Never returns successfully.
            """
            raise ValueError("intentional test failure")

        try:
            agent.run(
                model_name=model_name,
                prompt_template="Call the failing_tool with x='test'.",
                tools=[failing_tool],
                is_agentic=True,
                max_steps=3,
                token_callback=callback,
            )
        except Exception:
            pass
        assert "intentional test failure" in "".join(tokens)

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_no_callback_tool_output_not_affected(self, model_name):
        agent = KISSAgent("test-no-callback-tool")
        result = agent.run(
            model_name=model_name,
            prompt_template=(
                "What is 17 + 25? Use the add_numbers tool with a=17, b=25, "
                "then call finish with the answer."
            ),
            tools=[add_numbers],
            is_agentic=True,
            max_steps=5,
        )
        assert result is not None
        assert "42" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
