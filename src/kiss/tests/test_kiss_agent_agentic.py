"""Test suite for KISSAgent agentic mode using real API calls."""

import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator

TEST_MODEL = "gemini-3-flash-preview"


@requires_gemini_api_key
class TestKISSAgentErrorHandling(unittest.TestCase):
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
class TestKISSAgentBudgetAndSteps(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Budget Test Agent")

    def test_agent_budget_exceeded_raises_error(self) -> None:
        def expensive_tool() -> str:
            """A tool that triggers budget check."""
            return "Result"

        try:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Call expensive_tool, then call it again, then finish.",
                tools=[expensive_tool],
                max_steps=10,
                max_budget=0.00001,
            )
        except KISSError as e:
            self.assertIn("budget", str(e).lower())


if __name__ == "__main__":
    unittest.main()
