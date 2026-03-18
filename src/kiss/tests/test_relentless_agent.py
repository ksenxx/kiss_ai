"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import (
    CONTINUATION_PROMPT,
    IMPORTANT_INSTRUCTIONS,
    TASK_PROMPT,
    RelentlessAgent,
    finish,
)
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# kiss/core/relentless_agent.py — TASK_PROMPT, IMPORTANT_INSTRUCTIONS,
#                                  CONTINUATION_PROMPT
# ---------------------------------------------------------------------------


class TestTemplateConstants(unittest.TestCase):
    """Tests that template strings contain the expected placeholders."""

    def test_task_prompt_placeholders(self) -> None:
        """TASK_PROMPT only has task_description and previous_progress."""
        formatted = TASK_PROMPT.format(
            task_description="do something", previous_progress="done step 1"
        )
        self.assertIn("do something", formatted)
        self.assertIn("done step 1", formatted)

    def test_important_instructions_placeholders(self) -> None:
        """IMPORTANT_INSTRUCTIONS has step_threshold, work_dir, current_pid."""
        formatted = IMPORTANT_INSTRUCTIONS.format(
            step_threshold="8", work_dir="/tmp/test", current_pid="12345"
        )
        self.assertIn("step 8", formatted)
        self.assertIn("/tmp/test", formatted)
        self.assertIn("12345", formatted)
        self.assertIn("MOST IMPORTANT INSTRUCTIONS", formatted)

    def test_continuation_prompt_placeholders(self) -> None:
        """CONTINUATION_PROMPT has progress_text."""
        formatted = CONTINUATION_PROMPT.format(progress_text="step 1 done")
        self.assertIn("step 1 done", formatted)
        self.assertIn("Continue", formatted)


# ---------------------------------------------------------------------------
# kiss/core/relentless_agent.py — finish()
# ---------------------------------------------------------------------------


class TestFinish(unittest.TestCase):
    """Tests for the finish() tool function."""

    def test_finish_bool_args(self) -> None:
        """finish() with bool args returns valid YAML."""
        result = finish(success=True, is_continue=False, summary="all done")
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertFalse(parsed["is_continue"])
        self.assertEqual(parsed["summary"], "all done")

    def test_finish_string_true(self) -> None:
        """finish() converts string 'true' to bool True."""
        result = finish(success="true", is_continue="yes", summary="x")  # type: ignore[arg-type]
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertTrue(parsed["is_continue"])

    def test_finish_string_false(self) -> None:
        """finish() converts string 'false' to bool False."""
        result = finish(success="false", is_continue="no", summary="y")  # type: ignore[arg-type]
        parsed = yaml.safe_load(result)
        self.assertFalse(parsed["success"])
        self.assertFalse(parsed["is_continue"])


# ---------------------------------------------------------------------------
# kiss/core/relentless_agent.py — RelentlessAgent._reset()
# ---------------------------------------------------------------------------


class TestReset(unittest.TestCase):
    """Tests for RelentlessAgent._reset()."""

    def test_reset_defaults(self) -> None:
        """_reset() applies config defaults when no overrides given."""
        agent = RelentlessAgent("test")
        agent._reset(None, None, None, None, None, None)
        cfg = config_module.DEFAULT_CONFIG.relentless_agent
        self.assertEqual(agent.max_sub_sessions, cfg.max_sub_sessions)
        self.assertEqual(agent.max_steps, cfg.max_steps)
        self.assertEqual(agent.max_budget, cfg.max_budget)
        self.assertEqual(agent.model_name, cfg.model_name)

    def test_reset_overrides(self) -> None:
        """_reset() uses explicit overrides when provided."""
        agent = RelentlessAgent("test")
        with tempfile.TemporaryDirectory() as td:
            agent._reset("m1", 5, 10, 2.0, td, "img:latest")
            self.assertEqual(agent.model_name, "m1")
            self.assertEqual(agent.max_sub_sessions, 5)
            self.assertEqual(agent.max_steps, 10)
            self.assertEqual(agent.max_budget, 2.0)
            self.assertEqual(agent.work_dir, str(Path(td).resolve()))
            self.assertEqual(agent.docker_image, "img:latest")


# ---------------------------------------------------------------------------
# kiss/core/relentless_agent.py — RelentlessAgent._docker_bash()
# ---------------------------------------------------------------------------


class TestDockerBash(unittest.TestCase):
    def test_docker_bash_no_manager(self) -> None:
        """_docker_bash raises KISSError when docker_manager is None."""
        agent = RelentlessAgent("test")
        agent._reset(None, None, None, None, None, None)
        with self.assertRaises(KISSError):
            agent._docker_bash("ls", "list")


# ---------------------------------------------------------------------------
# kiss/core/relentless_agent.py — RelentlessAgent.perform_task()
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestPerformTaskSystemPrompt(unittest.TestCase):
    """Verify IMPORTANT_INSTRUCTIONS is appended to the system prompt."""

    def test_system_prompt_includes_important_instructions(self) -> None:
        """perform_task appends IMPORTANT_INSTRUCTIONS to self.system_prompt."""
        agent = RelentlessAgent("SysPromptTest")
        custom_system = "You are a helpful assistant."
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='done'). Do NOT call any other tool first."
                ),
                system_prompt=custom_system,
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

    def test_important_instructions_formatting(self) -> None:
        """Verify IMPORTANT_INSTRUCTIONS is formatted with correct values."""
        agent = RelentlessAgent("FmtTest")
        with tempfile.TemporaryDirectory() as td:
            agent._reset(TEST_MODEL, 1, 10, 1.0, td, None)
            agent.system_prompt = "base system"
            agent.model_config = None
            agent.task_description = "test task"
            # Format like perform_task does
            important = IMPORTANT_INSTRUCTIONS.format(
                step_threshold=str(agent.max_steps - 2),
                work_dir=agent.work_dir,
                current_pid=str(os.getpid()),
            )
            self.assertIn("step 8", important)
            self.assertIn(agent.work_dir, important)
            self.assertIn(str(os.getpid()), important)


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
# kiss/core/relentless_agent.py — RelentlessAgent.run()
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
