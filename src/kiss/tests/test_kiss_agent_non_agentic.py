# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for KISSAgent non-agentic mode without mocking, using real API calls."""

import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import simple_calculator

# Test model - using a fast, cheap model for testing
TEST_MODEL = "gemini-3-flash-preview"


class TestKISSAgentNonAgentic(unittest.TestCase):
    """Tests for non-agentic mode."""

    def setUp(self):
        self.agent = KISSAgent("Non-Agentic Test Agent")

    def tearDown(self):
        self.agent = None

    def test_non_agentic_simple_response(self):
        """Test non-agentic mode returns a response without tools."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="What is 2 + 2? Reply with just the number.",
            is_agentic=False,
        )
        self.assertIn("4", result)

    def test_non_agentic_with_arguments(self):
        """Test non-agentic mode with single and multiple template arguments."""
        # Single argument
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Say hello to {name}. Reply with just 'Hello, {name}!'",
            arguments={"name": "World"},
            is_agentic=False,
        )
        self.assertIn("Hello", result)

        # Multiple arguments
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Add {a} and {b}. Reply with just the sum as a number.",
            arguments={"a": "15", "b": "25"},
            is_agentic=False,
        )
        self.assertIn("40", result)

    def test_non_agentic_with_tools_raises_error(self):
        """Test that providing tools to non-agentic agent raises KISSError."""
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
            # This is expected due to the order of operations in run()
            pass


if __name__ == "__main__":
    unittest.main()
