"""Tests for error handling in KISSAgent and RelentlessAgent."""

import unittest

import httpx
from anthropic import AuthenticationError as AnthropicAuthError
from openai import AuthenticationError as OpenAIAuthError

from kiss.core.kiss_agent import KISSAgent, _is_retryable_error
from kiss.core.kiss_error import KISSError

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


class TestIsRetryableError(unittest.TestCase):
    def test_openai_auth_error_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(_openai_auth_error()))

    def test_anthropic_auth_error_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(_anthropic_auth_error()))

    def test_generic_error_with_api_key_message_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(Exception("Missing API key")))

    def test_generic_error_with_unauthorized_message_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(Exception("401 Unauthorized")))

    def test_generic_error_with_permission_denied_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(Exception("permission denied for resource")))

    def test_generic_error_with_incorrect_api_key_not_retryable(self) -> None:
        self.assertFalse(_is_retryable_error(Exception("Incorrect API key provided")))

    def test_generic_timeout_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_error(TimeoutError("Connection timed out")))

    def test_generic_connection_error_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_error(ConnectionError("Connection reset")))

    def test_generic_runtime_error_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_error(RuntimeError("Server returned 500")))

    def test_value_error_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_error(ValueError("Unexpected response format")))

    def test_permission_denied_type_name_not_retryable(self) -> None:
        class PermissionDeniedError(Exception):
            pass

        self.assertFalse(_is_retryable_error(PermissionDeniedError("forbidden")))

    def test_permission_denied_class_name_not_retryable(self) -> None:
        class PermissionDenied(Exception):  # noqa: N818
            pass

        self.assertFalse(_is_retryable_error(PermissionDenied("no access")))


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


if __name__ == "__main__":
    unittest.main()
