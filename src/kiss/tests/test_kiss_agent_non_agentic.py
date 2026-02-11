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
