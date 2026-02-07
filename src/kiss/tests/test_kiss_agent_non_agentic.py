"""Test suite for KISSAgent non-agentic mode using real API calls."""

import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator

TEST_MODEL = "gemini-3-flash-preview"


@requires_gemini_api_key
class TestKISSAgentNonAgentic(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Non-Agentic Test Agent")

    def test_non_agentic_simple_response(self) -> None:
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="What is 2 + 2? Reply with just the number.",
            is_agentic=False,
        )
        self.assertIn("4", result)

    def test_non_agentic_with_arguments(self) -> None:
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Say hello to {name}. Reply with just 'Hello, {name}!'",
            arguments={"name": "World"},
            is_agentic=False,
        )
        self.assertIn("Hello", result)

        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Add {a} and {b}. Reply with just the sum as a number.",
            arguments={"a": "15", "b": "25"},
            is_agentic=False,
        )
        self.assertIn("40", result)

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


if __name__ == "__main__":
    unittest.main()
