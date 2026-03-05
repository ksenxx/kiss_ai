"""Integration tests for KISSAgent achieving full line and branch coverage.

These tests send real messages to an LLM and verify actual behavior.
"""

import unittest

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-3-flash-preview"


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

    def test_non_agentic_with_console_printer(self) -> None:
        agent = KISSAgent("NonAgenticPrinter")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Reply with exactly: HELLO",
            is_agentic=False,
            verbose=True,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


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
                verbose=False,
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
                prompt_template=("Call dummy_tool repeatedly. Never call finish."),
                tools=[dummy_tool],
                is_agentic=True,
                max_steps=50,
                max_budget=0.0001,
                verbose=False,
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
                    verbose=False,
                )
            self.assertIn("global budget", str(ctx.exception).lower())
        finally:
            config_module.DEFAULT_CONFIG.agent.global_max_budget = original_global
            Base.global_budget_used = original_used


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


if __name__ == "__main__":
    unittest.main()
