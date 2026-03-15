"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import tempfile
import unittest

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import (
    RelentlessAgent,
)
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# finish() standalone function tests (covers all isinstance branches)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _reset() tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _docker_bash() tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration tests: perform_task happy path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration tests: continuation and exhaustion
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestContinuation(unittest.TestCase):
    def test_empty_summary_no_progress(self) -> None:
        """Empty summary -> no progress_section added."""
        agent = RelentlessAgent("EmptySummary")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(KISSError):
                agent.run(
                    model_name=TEST_MODEL,
                    prompt_template=(
                        "Call finish(success=False, is_continue=True, summary='')"
                    ),
                    max_steps=5,
                    max_budget=2.0,
                    max_sub_sessions=2,
                    work_dir=td,
                    verbose=False,
                )


# ---------------------------------------------------------------------------
# Integration tests: exception handling paths
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestExceptionPaths(unittest.TestCase):
    def test_exception_summarizer_also_fails(self) -> None:
        """Both executor and summarizer fail (global budget exceeded)."""
        original_global = config_module.DEFAULT_CONFIG.agent.global_max_budget
        original_used = Base.global_budget_used
        try:
            config_module.DEFAULT_CONFIG.agent.global_max_budget = 0.0001
            Base.global_budget_used = 0.01

            agent = RelentlessAgent("ExcSum-Fail")
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(KISSError):
                    agent.run(
                        model_name=TEST_MODEL,
                        prompt_template="Do something.",
                        max_steps=5,
                        max_budget=10.0,
                        max_sub_sessions=1,
                        work_dir=td,
                        verbose=False,
                    )
        finally:
            config_module.DEFAULT_CONFIG.agent.global_max_budget = original_global
            Base.global_budget_used = original_used


# ---------------------------------------------------------------------------
# run() method branches
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestRunBranches(unittest.TestCase):
    def test_with_docker(self) -> None:
        """Test the Docker path in run()."""
        agent = RelentlessAgent("Docker-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='docker test done'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                docker_image="ubuntu:latest",
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertIsNone(agent.docker_manager)

# ---------------------------------------------------------------------------
# Docker-specific branch tests
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestDockerStreamCallback(unittest.TestCase):
    """Test that docker_stream callback is invoked (covers line 292)."""

    def test_stream_callback_invoked(self) -> None:
        from kiss.core.print_to_console import ConsolePrinter

        agent = RelentlessAgent("DockerStream")
        printer = ConsolePrinter()

        def docker_cmd(command: str) -> str:
            """Run a shell command inside the Docker container.

            Args:
                command: The shell command to execute.

            Returns:
                The command output as a string.
            """
            return agent._docker_bash(command, "docker cmd")

        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "First call docker_cmd(command='echo streamed_output'), "
                    "then IMMEDIATELY call "
                    "finish(success=True, is_continue=False, summary='streamed'). "
                    "Do NOT call any other tool."
                ),
                tools=[docker_cmd],
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                docker_image="ubuntu:latest",
                printer=printer,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])


if __name__ == "__main__":
    unittest.main()
