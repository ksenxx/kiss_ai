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


if __name__ == "__main__":
    unittest.main()
