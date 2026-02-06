"""Integration tests for the async token_callback streaming feature.

These tests use REAL API calls -- no mocks.  Each provider is tested for:
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collector() -> tuple[TokenCallback, list[str]]:
    """Return an async callback and the list it appends tokens to."""
    tokens: list[str] = []

    async def _callback(token: str) -> None:
        tokens.append(token)

    return _callback, tokens


# ---------------------------------------------------------------------------
# AnthropicModel tests
# ---------------------------------------------------------------------------


@requires_anthropic_api_key
class TestAnthropicTokenCallback:
    """Token callback tests for Anthropic (Claude)."""

    @pytest.mark.timeout(60)
    def test_generate_streams_tokens(self):
        """generate() should invoke the callback for every text delta."""
        callback, tokens = _make_collector()
        m = model("claude-haiku-4-5", token_callback=callback)
        m.initialize("What is 2 + 2? Reply with just the number.")
        content, response = m.generate()

        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)
        reconstructed = "".join(tokens)
        # The concatenation of streamed tokens should equal the returned content.
        assert reconstructed == content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_generate_with_tools_streams_tokens(self):
        """generate_and_process_with_tools() should still invoke the callback."""
        callback, tokens = _make_collector()
        m = model("claude-haiku-4-5", token_callback=callback)
        m.initialize("Use the add_numbers tool to add 3 and 4. Call the tool with a=3, b=4.")
        function_map = {"add_numbers": add_numbers}
        function_calls, content, response = m.generate_and_process_with_tools(function_map)

        assert response is not None
        # The model may or may not emit text before tool calls; either is acceptable.
        # But the callback list should be populated if text was returned.
        if content:
            assert len(tokens) > 0

    @pytest.mark.timeout(60)
    def test_no_callback_still_works(self):
        """Passing token_callback=None should behave identically to the original code."""
        m = model("claude-haiku-4-5", token_callback=None)
        m.initialize("What is 2 + 2? Reply with just the number.")
        content, response = m.generate()
        assert "4" in content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_token_counts_with_streaming(self):
        """Token counts should still be extractable from the streamed response."""
        callback, _ = _make_collector()
        m = model("claude-haiku-4-5", token_callback=callback)
        m.initialize("Say hello in one word.")
        _, response = m.generate()
        input_tokens, output_tokens = m.extract_input_output_token_counts_from_response(response)
        assert input_tokens > 0
        assert output_tokens > 0


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------


@requires_openai_api_key
class TestOpenAITokenCallback:
    """Token callback tests for OpenAI (GPT)."""

    @pytest.mark.timeout(60)
    def test_generate_streams_tokens(self):
        """generate() should invoke the callback for every text delta."""
        callback, tokens = _make_collector()
        m = model("gpt-4.1-mini", token_callback=callback)
        m.initialize("What is 5 + 5? Reply with just the number.")
        content, response = m.generate()

        assert len(tokens) > 0, "Expected at least one callback invocation"
        reconstructed = "".join(tokens)
        assert reconstructed == content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_generate_with_tools_streams_tokens(self):
        """generate_and_process_with_tools() should still invoke the callback."""
        callback, tokens = _make_collector()
        m = model("gpt-4.1-mini", token_callback=callback)
        m.initialize("Use the add_numbers tool to add 10 and 20. Call the tool with a=10, b=20.")
        function_map = {"add_numbers": add_numbers}
        function_calls, content, response = m.generate_and_process_with_tools(function_map)

        assert response is not None
        # When the model calls a tool, it may not emit any text tokens.
        if content:
            assert len(tokens) > 0

    @pytest.mark.timeout(60)
    def test_no_callback_still_works(self):
        """Passing token_callback=None should behave identically to the original code."""
        m = model("gpt-4.1-mini", token_callback=None)
        m.initialize("What is 5 + 5? Reply with just the number.")
        content, response = m.generate()
        assert "10" in content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_token_counts_with_streaming(self):
        """Token counts should still be extractable from the streamed response."""
        callback, _ = _make_collector()
        m = model("gpt-4.1-mini", token_callback=callback)
        m.initialize("Say hello in one word.")
        _, response = m.generate()
        input_tokens, output_tokens = m.extract_input_output_token_counts_from_response(response)
        assert input_tokens >= 0
        assert output_tokens >= 0


# ---------------------------------------------------------------------------
# Gemini tests
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestGeminiTokenCallback:
    """Token callback tests for Gemini."""

    @pytest.mark.timeout(60)
    def test_generate_streams_tokens(self):
        """generate() should invoke the callback for every text delta."""
        callback, tokens = _make_collector()
        m = model("gemini-2.0-flash", token_callback=callback)
        m.initialize("What is 3 + 3? Reply with just the number.")
        content, response = m.generate()

        assert len(tokens) > 0, "Expected at least one callback invocation"
        reconstructed = "".join(tokens)
        assert reconstructed == content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_generate_with_tools_streams_tokens(self):
        """generate_and_process_with_tools() should still invoke the callback."""
        callback, tokens = _make_collector()
        m = model("gemini-2.0-flash", token_callback=callback)
        m.initialize("Use the add_numbers tool to add 6 and 7. Call the tool with a=6, b=7.")
        function_map = {"add_numbers": add_numbers}
        function_calls, content, response = m.generate_and_process_with_tools(function_map)

        assert response is not None
        if content:
            assert len(tokens) > 0

    @pytest.mark.timeout(60)
    def test_no_callback_still_works(self):
        """Passing token_callback=None should behave identically to the original code."""
        m = model("gemini-2.0-flash", token_callback=None)
        m.initialize("What is 3 + 3? Reply with just the number.")
        content, response = m.generate()
        assert "6" in content
        assert response is not None

    @pytest.mark.timeout(60)
    def test_token_counts_with_streaming(self):
        """Token counts should still be extractable from the streamed response."""
        callback, _ = _make_collector()
        m = model("gemini-2.0-flash", token_callback=callback)
        m.initialize("Say hello in one word.")
        _, response = m.generate()
        input_tokens, output_tokens = m.extract_input_output_token_counts_from_response(response)
        assert input_tokens >= 0
        assert output_tokens >= 0


# ---------------------------------------------------------------------------
# KISSAgent-level tests
# ---------------------------------------------------------------------------


@requires_openai_api_key
class TestKISSAgentTokenCallback:
    """Test that token_callback works when threaded through KISSAgent.run()."""

    @pytest.mark.timeout(60)
    def test_non_agentic_with_callback(self):
        """Non-agentic agent run should stream tokens via the callback."""
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
        """Agentic agent run should stream tool output via the callback."""
        callback, tokens = _make_collector()
        agent = KISSAgent("test-agentic-tool-output")

        def simple_calculator(expression: str) -> str:
            """Evaluate a simple arithmetic expression.

            Args:
                expression: The arithmetic expression to evaluate (e.g., '2+2', '10*5')

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
        # The tool produces "56088" which must appear in the streamed tokens.
        joined = "".join(tokens)
        assert "56088" in joined, (
            f"Expected tool output '56088' in streamed tokens, got: {joined!r}"
        )

    @pytest.mark.timeout(60)
    def test_no_callback_regression(self):
        """Agent run without a callback should still work normally."""
        agent = KISSAgent("test-no-callback")
        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 9 + 9? Reply with just the number.",
            is_agentic=False,
        )
        assert "18" in result


# ---------------------------------------------------------------------------
# Cross-provider parameterized tests
# ---------------------------------------------------------------------------

ALL_PROVIDER_MODELS = [
    pytest.param("claude-haiku-4-5", marks=requires_anthropic_api_key),
    pytest.param("gpt-4.1-mini", marks=requires_openai_api_key),
    pytest.param("gemini-2.0-flash", marks=requires_gemini_api_key),
]


class TestTokenCallbackCrossProvider:
    """Tests that apply across all providers."""

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(60)
    def test_callback_receives_only_strings(self, model_name):
        """Every argument passed to the callback must be a non-empty string."""
        callback, tokens = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Tell me a very short joke (one sentence).")
        m.generate()

        assert len(tokens) > 0
        for t in tokens:
            assert isinstance(t, str), f"Expected str, got {type(t)}"
            assert len(t) > 0, "Callback received an empty string"

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(60)
    def test_conversation_state_preserved_with_callback(self, model_name):
        """After generate(), the conversation should have the assistant message."""
        callback, _ = _make_collector()
        m = model(model_name, token_callback=callback)
        m.initialize("Say hello.")
        content, _ = m.generate()

        assert len(m.conversation) == 2  # user + assistant
        last_msg = m.conversation[-1]
        assert last_msg["role"] == "assistant"


# ---------------------------------------------------------------------------
# Tool output streaming tests (all providers via KISSAgent)
# ---------------------------------------------------------------------------


class TestToolOutputStreaming:
    """Verify that tool execution output is streamed to the token_callback."""

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_tool_output_appears_in_callback(self, model_name):
        """The deterministic tool result must appear in the collected tokens."""
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
        joined = "".join(tokens)
        # add_numbers(17, 25) returns "42" -- it must appear in the streamed output.
        assert "42" in joined, (
            f"Expected tool output '42' in streamed tokens, got: {joined!r}"
        )

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_tool_error_output_streamed(self, model_name):
        """When a tool raises, the error message should still be streamed."""
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
            pass  # Agent may raise on max steps -- that's fine.
        # The agent may finish or hit max steps -- either is fine.
        joined = "".join(tokens)
        # The error message should have been streamed.
        assert "intentional test failure" in joined, (
            f"Expected error message in streamed tokens, got: {joined!r}"
        )

    @pytest.mark.parametrize("model_name", ALL_PROVIDER_MODELS)
    @pytest.mark.timeout(120)
    def test_no_callback_tool_output_not_affected(self, model_name):
        """Without a callback, tool execution should still work normally."""
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
