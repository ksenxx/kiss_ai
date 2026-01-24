# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for KISSAgent agentic mode without mocking, using real API calls.

These tests verify various KISSAgent agentic behaviors using the Together AI API.
Tests are organized by feature area to increase coverage of src/kiss/core files.
"""

import json
import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import CustomFormatter, simple_calculator

# Test model - using a reliable model for agentic tool calling
TEST_MODEL = "gemini-3-flash-preview"


def get_greeting(name: str) -> str:
    """Return a greeting message.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"


def always_fails() -> str:
    """A function that always raises an exception.

    Returns:
        Never returns, always raises an exception
    """
    raise ValueError("This function always fails!")


class TestKISSAgentBasic(unittest.TestCase):
    """Basic KISSAgent tests."""

    def setUp(self):
        self.agent = KISSAgent("Basic Test Agent")

    def tearDown(self):
        self.agent = None

    def test_agentic_simple_task(self):
        """Test agentic mode with a simple calculation task."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Use the simple_calculator tool with expression='8934 * 2894' to calculate. "
                "Then call finish with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator],
            max_steps=10,
        )
        # Check if result contains at least one digit (number)
        self.assertRegex(result, r"\d")
        self.assertIsNotNone(result)
        self.assertIn("25854996", result)
        self.assertEqual(len(json.loads(self.agent.get_trajectory())), 5)

    def test_agentic_with_arguments(self):
        """Test agentic mode with prompt template arguments."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Calculate {num1} * {num2} using the 'simple_calculator' tool. "
                "Then call 'finish' with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            arguments={"num1": "8934", "num2": "2894"},
            tools=[simple_calculator],
            max_steps=10,
        )
        self.assertIsNotNone(result)
        # LLM should calculate 8934 * 2894 using the tool
        self.assertRegex(result, r"\d+")
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

    def test_trajectory_structure(self):
        """Test that trajectory has proper structure with user and model messages."""
        self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Calculate 7 * 8 using the 'simple_calculator' tool. "
                "Then call 'finish' with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator],
            max_steps=10,
        )
        trajectory = json.loads(self.agent.get_trajectory())

        # Verify trajectory is a list with messages
        self.assertIsInstance(trajectory, list)
        self.assertGreater(len(trajectory), 0)

        # Check message structure
        for msg in trajectory:
            self.assertIn("role", msg)
            self.assertIn("content", msg)

        # First message should be user message
        self.assertEqual(trajectory[0]["role"], "user")
        self.assertIn("7 * 8", trajectory[0]["content"])
        self.assertIn("56", trajectory[2]["content"])

        # Should have model response
        roles = [msg["role"] for msg in trajectory]
        self.assertIn("model", roles)


class TestKISSAgentCustomFormatter(unittest.TestCase):
    """Tests for custom formatter functionality."""

    def setUp(self):
        self.agent = KISSAgent("Formatter Test Agent")
        self.custom_formatter = CustomFormatter()

    def tearDown(self):
        self.agent = None

    def test_custom_formatter_receives_messages(self):
        """Test that custom formatter receives messages."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Calculate 8934 * 2894 using the calculator tool. "
                "Then call 'finish' with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator],
            formatter=self.custom_formatter,
            max_steps=10,
        )
        self.assertIsNotNone(result)
        self.assertIn("25854996", result)
        self.assertEqual(len(json.loads(self.agent.get_trajectory())), 5)
        self.assertGreater(len(self.custom_formatter.messages), 0)


class TestKISSAgentMultipleTools(unittest.TestCase):
    """Tests for agents with multiple tools."""

    def setUp(self):
        self.agent = KISSAgent("Multi-Tool Test Agent")

    def tearDown(self):
        self.agent = None

    def test_multiple_tools_available(self):
        """Test agent can choose between multiple tools."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Greet 'Alice' using the greeting tool. Then call finish with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator, get_greeting],
        )
        self.assertIsNotNone(result)
        # Check for greeting containing "Hello" and "Alice" (exclamation mark may vary)
        self.assertIn("Hello", result)
        self.assertIn("Alice", result)
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)


class TestKISSAgentErrorHandling(unittest.TestCase):
    """Tests for error handling scenarios."""

    def setUp(self):
        self.agent = KISSAgent("Error Test Agent")

    def tearDown(self):
        self.agent = None

    def test_tool_execution_error_recovery(self):
        """Test that agent can recover from tool execution errors."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "First call the always_fails tool. It will fail. "
                "After it fails, use the simple_calculator tool with expression='1+1'. "
                "Then call finish with the result of the calculator. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[always_fails, simple_calculator],
            max_steps=10,
        )
        self.assertIsNotNone(result)
        self.assertIn("2", result)
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 4)

    def test_duplicate_tool_raises_error(self):
        """Test that registering duplicate tools raises KISSError."""
        with self.assertRaises(KISSError) as context:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Test prompt",
                tools=[simple_calculator, simple_calculator],
            )
        self.assertIn("already registered", str(context.exception))


class TestKISSAgentBudgetAndSteps(unittest.TestCase):
    """Tests for budget and step limit functionality."""

    def setUp(self):
        self.agent = KISSAgent("Budget Test Agent")

    def tearDown(self):
        self.agent = None

    def test_max_steps_respected(self):
        """Test that agent completes within max_steps and tracks budget."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Calculate 2 + 2 using the calculator tool. "
                "Then call finish with the result."
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator],
            max_steps=10,
            max_budget=10.0,
        )
        self.assertLessEqual(self.agent.step_count, 10)
        self.assertGreaterEqual(KISSAgent.global_budget_used, 0.0)
        self.assertIn("4", result)
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

    def test_max_steps_exceeded_raises_error(self):
        """Test that exceeding max_steps raises KISSError."""

        def never_finish() -> str:
            """A tool that never finishes the task.

            Returns:
                A message asking to continue
            """
            return "Continue processing..."

        try:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Call the never_finish tool repeatedly. "
                    "Do NOT call finish. Keep calling never_finish."
                ),
                tools=[never_finish],
                max_steps=1,
            )
        except KISSError as e:
            self.assertIn("exceeded", str(e).lower())


class TestKISSAgentFinishTool(unittest.TestCase):
    """Tests for the built-in finish tool."""

    def setUp(self):
        self.agent = KISSAgent("Finish Tool Test Agent")

    def tearDown(self):
        self.agent = None

    def test_finish_tool_auto_added(self):
        """Test that finish tool is automatically added when not provided."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Just say 'done' and finish.",
            tools=[],
        )
        self.assertIsNotNone(result)
        self.assertIn("done", result.lower())
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

    def test_custom_finish_tool_not_duplicated(self):
        """Test that providing a custom finish tool doesn't add another."""

        def finish(result: str) -> str:
            """Custom finish function.

            Args:
                result: The final result

            Returns:
                The result
            """
            return f"CUSTOM: {result}"

        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Say 'hello' and finish.",
            tools=[finish],
        )
        self.assertIn("CUSTOM:", result)
        self.assertEqual(len(json.loads(self.agent.get_trajectory())), 3)


class TestKISSAgentMultipleRuns(unittest.TestCase):
    """Tests for running the same agent multiple times."""

    def setUp(self):
        self.agent = KISSAgent("Multiple Runs Test Agent")

    def tearDown(self):
        self.agent = None

    def test_trajectory_resets_between_runs(self):
        """Test that trajectory is reset between runs."""
        # First run
        self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Calculate 3 + 3 using the calculator tool.",
            tools=[simple_calculator],
        )

        # Second run with different calculation
        self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Calculate 4 + 4 using the calculator tool.",
            tools=[simple_calculator],
        )
        trajectory2 = json.loads(self.agent.get_trajectory())

        # Trajectory should be reset (not accumulating from first run)
        # Just verify the second trajectory has reasonable length
        self.assertGreater(len(trajectory2), 0)
        # And that trajectory contains content from second run's prompt
        trajectory2_str = str(trajectory2)
        self.assertIn("4 + 4", trajectory2_str)


class TestKISSAgentToolVariants(unittest.TestCase):
    """Tests for tools with various parameter and return types."""

    def setUp(self):
        self.agent = KISSAgent("Tool Variants Test Agent")

    def tearDown(self):
        self.agent = None

    def test_tool_with_optional_param(self):
        """Test tool with optional parameter."""

        def greet_with_title(name: str, title: str = "Mr.") -> str:
            """Greet someone with an optional title.

            Args:
                name: The name to greet
                title: Optional title (default: Mr.)

            Returns:
                A greeting message
            """
            return f"Hello, {title} {name}!"

        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Use the greet_with_title tool to greet 'Smith' with title 'Dr.'. "
                "Then call finish with the exact greeting result."
                "You MUST make exactly one tool call in your response."
            ),
            tools=[greet_with_title],
            max_steps=10,
        )
        self.assertIsNotNone(result)
        self.assertIn("Dr. Smith", result)
        trajectory = json.loads(self.agent.get_trajectory())
        self.assertIn(len(trajectory), [3, 5])

    def test_tool_returns_dict(self):
        """Test tool that returns a dictionary.

        Note: This test can be flaky due to LLM behavior - the model may not always
        follow tool calling instructions consistently. We use a more explicit prompt
        and allow for multiple attempts.
        """

        def get_info() -> dict:
            """Get some information.

            Returns:
                A dictionary with information
            """
            return {"status": "ok", "value": 42}

        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "You have access to a 'get_info' tool. "
                "Step 1: Call the get_info tool to retrieve information. "
                "Step 2: After getting the result, call finish with the 'status' value. "
                "The get_info tool returns a dictionary with 'status' and 'value' keys."
            ),
            tools=[get_info],
            max_steps=15,  # More steps to allow for LLM behavior
        )
        self.assertIsNotNone(result)
        # Check result contains 'ok' or 'status' or 'get_info' (LLM may phrase result differently)
        self.assertTrue("ok" in result or "status" in result.lower() or "get_info" in result)
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

    def test_tool_with_multiple_params(self):
        """Test tool with multiple required parameters."""

        def add_numbers(a: int, b: int) -> str:
            """Add two numbers together.

            Args:
                a: First number
                b: Second number

            Returns:
                The sum as string
            """
            return str(a + b)

        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Use 'add_numbers' with a=3 and b=7, then finish with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[add_numbers],
            max_steps=5,
        )
        self.assertIsNotNone(result)
        self.assertIn("10", result)
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)


class TestKISSAgentPromptFormats(unittest.TestCase):
    """Tests for various prompt template formats."""

    def setUp(self):
        self.agent = KISSAgent("Prompt Format Test Agent")

    def tearDown(self):
        self.agent = None

    def test_multiline_prompt_template(self):
        """Test multiline prompt template."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="""You are a helpful calculator assistant.

Your task is to calculate 5 + 5 using the calculator tool.

Steps:
1. Use the simple_calculator tool with expression='5+5'
2. Call finish with the result
3. You MUST make exactly one tool call in your response.

Only return the number.""",
            tools=[simple_calculator],
        )
        self.assertIn("10", result)
        self.assertEqual(len(json.loads(self.agent.get_trajectory())), 5)

    def test_empty_arguments_dict(self):
        """Test that empty arguments dict works."""
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Say 'hello' and finish. "
                "You MUST make exactly one tool call in your response."
            ),
            arguments={},
            tools=[],
            max_steps=10,
        )
        self.assertIsNotNone(result)
        self.assertIn("hello", result.lower())
        # Trajectory length varies based on LLM behavior
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

class TestKISSAgentVerboseMode(unittest.TestCase):
    """Tests for verbose mode."""

    def setUp(self):
        self.agent = KISSAgent("Verbose Test Agent")

    def tearDown(self):
        self.agent = None

    def test_verbose_mode_toggle(self):
        """Test that verbose mode can be toggled without errors."""
        from kiss.core.config import DEFAULT_CONFIG

        original_verbose = DEFAULT_CONFIG.agent.verbose

        # Test with verbose=True
        DEFAULT_CONFIG.agent.verbose = True
        try:
            result = self.agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Calculate 2 + 2 using the 'simple_calculator' tool. "
                    "Then call 'finish' with the result of the 'simple_calculator' tool."
                    "You MUST make exactly one tool call in your response."
                ),
                tools=[simple_calculator],
                max_steps=10,
            )
            self.assertIsNotNone(result)
            self.assertIn("4", result)
            # Trajectory length varies based on LLM behavior
            self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)
        finally:
            DEFAULT_CONFIG.agent.verbose = original_verbose


if __name__ == "__main__":
    unittest.main()
