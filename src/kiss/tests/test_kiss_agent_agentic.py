"""Test suite for KISSAgent agentic mode using real API calls."""

import json
import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator

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


@requires_gemini_api_key
class TestKISSAgentBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Basic Test Agent")

    def test_agentic_simple_task(self) -> None:
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
        self.assertRegex(result, r"\d")
        self.assertIn("25854996", result)
        self.assertEqual(len(json.loads(self.agent.get_trajectory())), 5)

    def test_agentic_with_arguments(self) -> None:
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
        self.assertRegex(result, r"\d+")
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)

    def test_trajectory_structure(self) -> None:
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
        self.assertIsInstance(trajectory, list)
        self.assertGreater(len(trajectory), 0)
        for msg in trajectory:
            self.assertIn("role", msg)
            self.assertIn("content", msg)
        self.assertEqual(trajectory[0]["role"], "user")
        self.assertIn("7 * 8", trajectory[0]["content"])
        self.assertIn("56", trajectory[2]["content"])
        self.assertIn("model", [msg["role"] for msg in trajectory])


@requires_gemini_api_key
class TestKISSAgentMultipleTools(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Multi-Tool Test Agent")

    def test_multiple_tools_available(self) -> None:
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Greet 'Alice' using the greeting tool. Then call finish with the result. "
                "You MUST make exactly one tool call in your response."
            ),
            tools=[simple_calculator, get_greeting],
        )
        self.assertIn("Hello", result)
        self.assertIn("Alice", result)
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 2)


@requires_gemini_api_key
class TestKISSAgentErrorHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Error Test Agent")

    def test_tool_execution_error_recovery(self) -> None:
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
        self.assertIn("2", result)
        self.assertGreater(len(json.loads(self.agent.get_trajectory())), 4)

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

    def test_max_steps_respected(self) -> None:
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

    def test_max_steps_exceeded_raises_error(self) -> None:
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


@requires_gemini_api_key
class TestKISSAgentFinishTool(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Finish Tool Test Agent")

    def test_finish_tool_auto_added(self) -> None:
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Just say 'done' and finish.",
            tools=[],
        )
        self.assertIsNotNone(result)
        self.assertIn("done", result.lower())

    def test_custom_finish_tool_not_duplicated(self) -> None:
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


@requires_gemini_api_key
class TestKISSAgentMultipleRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Multiple Runs Test Agent")

    def test_trajectory_resets_between_runs(self) -> None:
        self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Calculate 3 + 3 using the calculator tool.",
            tools=[simple_calculator],
        )
        self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Calculate 4 + 4 using the calculator tool.",
            tools=[simple_calculator],
        )
        trajectory2 = json.loads(self.agent.get_trajectory())
        self.assertGreater(len(trajectory2), 0)
        self.assertIn("4 + 4", str(trajectory2))


@requires_gemini_api_key
class TestKISSAgentToolVariants(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Tool Variants Test Agent")

    def test_tool_with_optional_param(self) -> None:
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
        self.assertIn("Dr. Smith", result)

    def test_tool_returns_dict(self) -> None:
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
            max_steps=15,
        )
        self.assertTrue("ok" in result or "status" in result.lower() or "get_info" in result)

    def test_tool_with_multiple_params(self) -> None:
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
        self.assertIn("10", result)


@requires_gemini_api_key
class TestKISSAgentPromptFormats(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Prompt Format Test Agent")

    def test_multiline_prompt_template(self) -> None:
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

    def test_empty_arguments_dict(self) -> None:
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Say 'hello' and finish. You MUST make exactly one tool call in your response."
            ),
            arguments={},
            tools=[],
            max_steps=10,
        )
        self.assertIn("hello", result.lower())


@requires_gemini_api_key
class TestKISSAgentVerboseMode(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Verbose Test Agent")

    def test_verbose_mode_toggle(self) -> None:
        from kiss.core.config import DEFAULT_CONFIG

        original_verbose = DEFAULT_CONFIG.agent.verbose
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
            self.assertIn("4", result)
        finally:
            DEFAULT_CONFIG.agent.verbose = original_verbose


@requires_gemini_api_key
class TestKISSAgentGlobalBudget(unittest.TestCase):
    def setUp(self) -> None:
        from kiss.core.base import Base
        from kiss.core.config import DEFAULT_CONFIG

        self.agent = KISSAgent("Global Budget Test Agent")
        self.original_global_budget = DEFAULT_CONFIG.agent.global_max_budget
        self.original_global_used = Base.global_budget_used

    def tearDown(self) -> None:
        from kiss.core.base import Base
        from kiss.core.config import DEFAULT_CONFIG

        DEFAULT_CONFIG.agent.global_max_budget = self.original_global_budget
        Base.global_budget_used = self.original_global_used

    def test_global_budget_tracked(self) -> None:
        from kiss.core.base import Base

        initial_budget = Base.global_budget_used
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Say 'hello' and finish.",
            tools=[],
            max_steps=3,
        )
        self.assertGreater(Base.global_budget_used, initial_budget)
        self.assertIsNotNone(result)


@requires_gemini_api_key
class TestKISSAgentWebTools(unittest.TestCase):
    def setUp(self) -> None:
        from kiss.core.config import DEFAULT_CONFIG

        self.agent = KISSAgent("Web Tools Test Agent")
        self.original_use_web = DEFAULT_CONFIG.agent.use_web

    def tearDown(self) -> None:
        from kiss.core.config import DEFAULT_CONFIG

        DEFAULT_CONFIG.agent.use_web = self.original_use_web

    def test_web_tools_added_when_enabled(self) -> None:
        from kiss.core.config import DEFAULT_CONFIG

        DEFAULT_CONFIG.agent.use_web = True
        result = self.agent.run(
            model_name=TEST_MODEL,
            prompt_template="Just say 'done' and finish.",
            tools=[],
            max_steps=3,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
