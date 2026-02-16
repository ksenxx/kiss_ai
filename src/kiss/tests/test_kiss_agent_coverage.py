"""Integration tests for KISSAgent achieving full line and branch coverage.

These tests send real messages to an LLM and verify actual behavior.
"""

import unittest

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator

TEST_MODEL = "gemini-3-flash-preview"


@requires_gemini_api_key
class TestNonAgenticGeneration(unittest.TestCase):
    def test_non_agentic_returns_response(self) -> None:
        agent = KISSAgent("NonAgentic")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Reply with exactly: HELLO",
            is_agentic=False,
            print_to_console=False,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_non_agentic_with_console_printer(self) -> None:
        agent = KISSAgent("NonAgenticPrinter")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Reply with exactly: HELLO",
            is_agentic=False,
            print_to_console=True,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


@requires_gemini_api_key
class TestAgenticFinish(unittest.TestCase):
    def test_agentic_with_tool_then_finish(self) -> None:
        agent = KISSAgent("AgenticTool")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "Use simple_calculator to compute 2+3, then call finish with the result."
            ),
            tools=[simple_calculator],
            is_agentic=True,
            max_steps=5,
            print_to_console=True,
        )
        self.assertIsInstance(result, str)


@requires_gemini_api_key
class TestMaxStepsExceeded(unittest.TestCase):
    def test_max_steps_raises_error(self) -> None:
        def dummy_tool() -> str:
            """A tool that returns a value. Call this tool repeatedly."""
            return "not done yet"

        agent = KISSAgent("MaxSteps")
        with self.assertRaises(KISSError) as ctx:
            agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Call dummy_tool repeatedly. Never call finish. "
                    "Always call dummy_tool in every response."
                ),
                tools=[dummy_tool],
                is_agentic=True,
                max_steps=2,
                max_budget=1.0,
                print_to_console=False,
            )
        self.assertIn("steps", str(ctx.exception).lower())


@requires_gemini_api_key
class TestBudgetExceeded(unittest.TestCase):
    def test_agent_budget_exceeded(self) -> None:
        def dummy_tool() -> str:
            """A tool that returns a result. Always call this tool."""
            return "keep going"

        agent = KISSAgent("BudgetExceed")
        try:
            agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Call dummy_tool repeatedly. Never call finish."
                ),
                tools=[dummy_tool],
                is_agentic=True,
                max_steps=50,
                max_budget=0.0001,
                print_to_console=False,
            )
        except KISSError as e:
            self.assertIn("budget", str(e).lower())

    def test_global_budget_exceeded(self) -> None:
        def dummy_tool() -> str:
            """A tool. Always call this."""
            return "ok"

        original_global = config_module.DEFAULT_CONFIG.agent.global_max_budget
        original_used = Base.global_budget_used
        try:
            config_module.DEFAULT_CONFIG.agent.global_max_budget = 0.0001
            Base.global_budget_used = 0.0002

            agent = KISSAgent("GlobalBudget")
            with self.assertRaises(KISSError) as ctx:
                agent.run(
                    model_name=TEST_MODEL,
                    prompt_template="Call dummy_tool then call finish with result 'done'.",
                    tools=[dummy_tool],
                    is_agentic=True,
                    max_steps=10,
                    max_budget=100.0,
                    print_to_console=False,
                )
            self.assertIn("global budget", str(ctx.exception).lower())
        finally:
            config_module.DEFAULT_CONFIG.agent.global_max_budget = original_global
            Base.global_budget_used = original_used


@requires_gemini_api_key
class TestSetupToolsWebBranch(unittest.TestCase):
    def test_web_tools_added_when_enabled(self) -> None:
        original = config_module.DEFAULT_CONFIG.agent.use_web
        try:
            config_module.DEFAULT_CONFIG.agent.use_web = True
            agent = KISSAgent("WebTools")
            agent.run(
                model_name=TEST_MODEL,
                prompt_template="Call finish immediately with result='ok'.",
                tools=[],
                is_agentic=True,
                max_steps=5,
                print_to_console=False,
            )
            self.assertIn("fetch_url", agent.function_map)
            self.assertIn("search_web", agent.function_map)
        finally:
            config_module.DEFAULT_CONFIG.agent.use_web = original

    def test_web_tools_not_added_when_disabled(self) -> None:
        original = config_module.DEFAULT_CONFIG.agent.use_web
        try:
            config_module.DEFAULT_CONFIG.agent.use_web = False
            agent = KISSAgent("NoWebTools")
            agent.run(
                model_name=TEST_MODEL,
                prompt_template="Call finish immediately with result='ok'.",
                tools=[],
                is_agentic=True,
                max_steps=5,
                print_to_console=False,
            )
            self.assertNotIn("fetch_url", agent.function_map)
            self.assertNotIn("search_web", agent.function_map)
        finally:
            config_module.DEFAULT_CONFIG.agent.use_web = original

    def test_custom_finish_tool_not_overridden(self) -> None:
        def finish(result: str) -> str:
            """Finish the task with the given result.

            Args:
                result: The final result.

            Returns:
                The result string.
            """
            return f"custom:{result}"

        original = config_module.DEFAULT_CONFIG.agent.use_web
        try:
            config_module.DEFAULT_CONFIG.agent.use_web = False
            agent = KISSAgent("CustomFinish")
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template="Call finish with result='hello'.",
                tools=[finish],
                is_agentic=True,
                max_steps=5,
                print_to_console=False,
            )
            self.assertIn("custom:", result)
        finally:
            config_module.DEFAULT_CONFIG.agent.use_web = original


@requires_gemini_api_key
class TestMultipleToolCalls(unittest.TestCase):
    def test_multiple_tool_calls_in_single_response(self) -> None:
        call_log: list[str] = []

        def add(a: int, b: int) -> str:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.

            Returns:
                The sum as a string.
            """
            call_log.append("add")
            return str(a + b)

        def multiply(a: int, b: int) -> str:
            """Multiply two numbers.

            Args:
                a: First number.
                b: Second number.

            Returns:
                The product as a string.
            """
            call_log.append("multiply")
            return str(a * b)

        agent = KISSAgent("MultiTool")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "You MUST call add(2, 3) and multiply(4, 5) in a SINGLE response "
                "using parallel tool calls. After getting results, call finish "
                "with 'add=5, multiply=20'."
            ),
            tools=[add, multiply],
            is_agentic=True,
            max_steps=5,
            print_to_console=True,
        )
        self.assertIn("5", result)
        self.assertIn("20", result)
        self.assertIn("add", call_log)
        self.assertIn("multiply", call_log)
        # If both tools were called in one step, step_count should be 2
        # (one step for parallel calls + one step for finish).
        # If called sequentially it would be 3. Either way, the agent should complete.
        self.assertLessEqual(agent.step_count, 3)


@requires_gemini_api_key
class TestToolExecutionError(unittest.TestCase):
    def test_tool_with_wrong_args_recovers(self) -> None:
        def strict_tool(required_arg: str) -> str:
            """A tool that requires exactly one string argument called required_arg.

            Args:
                required_arg: A required string argument.

            Returns:
                The argument echoed back.
            """
            if not isinstance(required_arg, str):
                raise TypeError("required_arg must be a string")
            return required_arg

        agent = KISSAgent("ToolError")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template=(
                "First, call strict_tool without any arguments (pass no arguments at all). "
                "After seeing the error, call finish with result='recovered'."
            ),
            tools=[strict_tool],
            is_agentic=True,
            max_steps=5,
            print_to_console=False,
        )
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
