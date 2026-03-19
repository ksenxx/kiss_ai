"""Tests for KISSAgent: agentic mode, non-agentic mode, retry behavior, error handling.

Merged from: test_kiss_agent_agentic, test_kiss_agent_non_agentic,
test_kiss_agent_coverage, test_error_handling, test_kiss_agent_retry.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from anthropic import AuthenticationError as AnthropicAuthError
from openai import AuthenticationError as OpenAIAuthError

from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator

TEST_MODEL = "gemini-3-flash-preview"

_DUMMY_REQUEST = httpx.Request("GET", "https://api.example.com/")


def _openai_auth_error(msg: str = "Incorrect API key provided") -> OpenAIAuthError:
    return OpenAIAuthError(
        message=msg,
        response=httpx.Response(401, request=_DUMMY_REQUEST),
        body=None,
    )


def _anthropic_auth_error(msg: str = "invalid x-api-key") -> AnthropicAuthError:
    return AnthropicAuthError(
        message=msg,
        response=httpx.Response(401, request=_DUMMY_REQUEST),
        body=None,
    )


class _RetryableErrorModel:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.model_name = "gpt-4o-mini"
        self.conversation: list[dict[str, Any]] = []

    def initialize(self, prompt: str, attachments: list[Any] | None = None) -> None:
        self.conversation.append({"role": "user", "content": prompt})

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Internal server error")
        return ([{"name": "finish", "arguments": {"result": "ok"}}], "done", object())

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        self.last_usage = usage_info

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        self.function_results = function_results

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)


class _NonRetryableErrorModel(_RetryableErrorModel):
    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        raise Exception("Unauthorized: invalid API key")


def _make_agent(model_obj: Any, max_steps: int = 5) -> KISSAgent:
    agent = KISSAgent("RetryTest")
    agent.model = model_obj
    agent.model_name = model_obj.model_name
    agent.verbose = False
    agent.printer = None
    agent.is_agentic = True
    agent.max_steps = max_steps
    agent.max_budget = 1.0
    agent.function_map = {"finish": agent.finish}
    agent.messages = []
    agent.step_count = 0
    agent.total_tokens_used = 0
    agent.budget_used = 0.0
    agent.session_info = ""
    agent.run_start_timestamp = 0
    return agent


@pytest.fixture(autouse=True)
def _restore_base_state() -> Iterator[None]:
    original_counter = Base.agent_counter
    original_budget = Base.global_budget_used
    yield
    Base.agent_counter = original_counter
    Base.global_budget_used = original_budget


@requires_gemini_api_key
class TestDuplicateToolRaises(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Error Test Agent")

    def test_duplicate_tool_raises_error(self) -> None:
        with self.assertRaises(KISSError) as context:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Test prompt",
                tools=[simple_calculator, simple_calculator],
            )
        self.assertIn("already registered", str(context.exception))


@requires_gemini_api_key
class TestNonAgenticWithToolsRaises(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Non-Agentic Test Agent")

    def test_non_agentic_with_tools_raises_error(self) -> None:
        try:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Test prompt",
                tools=[simple_calculator],
                is_agentic=False,
            )
            self.fail("Expected KISSError to be raised")
        except KISSError as e:
            self.assertIn("Tools cannot be provided", str(e))
        except AttributeError:
            pass


@requires_gemini_api_key
class TestNonAgenticGeneration(unittest.TestCase):
    def test_non_agentic_returns_response(self) -> None:
        agent = KISSAgent("NonAgentic")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Reply with exactly: HELLO",
            is_agentic=False,
            verbose=False,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


@requires_gemini_api_key
class TestSetupToolsWebBranch(unittest.TestCase):
    def test_custom_finish_tool_not_overridden(self) -> None:
        def finish(result: str) -> str:
            """Finish the task with the given result.

            Args:
                result: The final result.

            Returns:
                The result string.
            """
            return f"custom:{result}"

        agent = KISSAgent("CustomFinish")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Call finish with result='hello'.",
            tools=[finish],
            is_agentic=True,
            max_steps=5,
            verbose=False,
        )
        self.assertIn("custom:", result)


class TestAgenticLoopAuthError(unittest.TestCase):
    """Test that auth errors fail fast instead of retrying until max_steps."""

    INVALID_KEY_CONFIG = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-invalid-key-for-testing",
    }

    def test_auth_error_raises_kiss_error_fast(self) -> None:
        agent = KISSAgent("Auth Error Test")

        def dummy_tool() -> str:
            """A tool. Call this tool."""
            return "ok"

        with self.assertRaises(KISSError) as ctx:
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Call dummy_tool then finish.",
                tools=[dummy_tool],
                is_agentic=True,
                max_steps=10,
                max_budget=1.0,
                verbose=False,
                model_config=self.INVALID_KEY_CONFIG,
            )
        self.assertIn("non-retryable", str(ctx.exception).lower())
        self.assertLessEqual(agent.step_count, 1)

    def test_non_agentic_auth_error_propagates(self) -> None:
        agent = KISSAgent("Non-Agentic Auth Error Test")
        with self.assertRaises(Exception):
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Say hello",
                is_agentic=False,
                verbose=False,
                model_config=self.INVALID_KEY_CONFIG,
            )


def test_run_agentic_loop_raises_after_three_consecutive_retryable_errors() -> None:
    model = _RetryableErrorModel(failures=3)
    agent = _make_agent(model)

    with pytest.raises(KISSError, match="failed with 3 consecutive errors"):
        agent._run_agentic_loop()

    assert model.calls == 3
    retry_messages = [
        m for m in agent.messages if "Failed to get response from Model:" in str(m["content"])
    ]
    assert len(retry_messages) == 2


def test_run_agentic_loop_raises_immediately_for_non_retryable_error() -> None:
    agent = _make_agent(_NonRetryableErrorModel(failures=0))

    with pytest.raises(
        KISSError,
        match="Non-retryable error from model: Unauthorized: invalid API key",
    ):
        agent._run_agentic_loop()
