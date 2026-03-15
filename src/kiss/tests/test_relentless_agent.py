"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import os
import tempfile
import unittest

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import (
    RelentlessAgent,
    finish,
)
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# finish() standalone function tests (covers all isinstance branches)
# ---------------------------------------------------------------------------


class TestFinishFunction(unittest.TestCase):
    """Tests for the module-level finish() function."""

    def test_bool_inputs(self) -> None:
        """success/is_continue as actual bools skip isinstance(str) branches."""
        result = finish(True, False, "done")
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertFalse(parsed["is_continue"])
        self.assertEqual(parsed["summary"], "done")

    def test_string_true_values(self) -> None:
        """String 'true', 'yes', '1' all convert to True."""
        for val in ("true", "yes", "1"):
            parsed = yaml.safe_load(finish(val, val, "s"))  # type: ignore[arg-type]
            self.assertTrue(parsed["success"], f"failed for {val}")
            self.assertTrue(parsed["is_continue"], f"failed for {val}")

    def test_string_false_values(self) -> None:
        """Strings not in ('true', 'yes', '1') convert to False."""
        for val in ("false", "no", "0", "random"):
            parsed = yaml.safe_load(finish(val, val, "s"))  # type: ignore[arg-type]
            self.assertFalse(parsed["success"], f"failed for {val}")
            self.assertFalse(parsed["is_continue"], f"failed for {val}")


# ---------------------------------------------------------------------------
# _reset() tests
# ---------------------------------------------------------------------------


class TestReset(unittest.TestCase):
    def test_all_defaults(self) -> None:
        agent = RelentlessAgent("Reset-Default")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=None,
            docker_image=None,
        )
        cfg = config_module.DEFAULT_CONFIG.relentless_agent
        self.assertEqual(agent.model_name, cfg.model_name)
        self.assertEqual(agent.max_sub_sessions, cfg.max_sub_sessions)
        self.assertEqual(agent.max_steps, cfg.max_steps)
        self.assertEqual(agent.max_budget, cfg.max_budget)
        self.assertIsNone(agent.docker_image)
        self.assertTrue(os.path.isdir(agent.work_dir))

    def test_all_custom(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            agent = RelentlessAgent("Reset-Custom")
            agent._reset(
                model_name="custom-model",
                max_sub_sessions=3,
                max_steps=10,
                max_budget=5.0,
                work_dir=td,
                docker_image="ubuntu:latest",
                verbose=False,
            )
            self.assertEqual(agent.model_name, "custom-model")
            self.assertEqual(agent.max_sub_sessions, 3)
            self.assertEqual(agent.max_steps, 10)
            self.assertEqual(agent.max_budget, 5.0)
            self.assertEqual(agent.work_dir, str(os.path.realpath(td)))
            self.assertEqual(agent.docker_image, "ubuntu:latest")


# ---------------------------------------------------------------------------
# _docker_bash() tests
# ---------------------------------------------------------------------------


class TestDockerBash(unittest.TestCase):
    def test_no_manager_raises(self) -> None:
        agent = RelentlessAgent("DockerBash")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=None,
            docker_image=None,
        )
        with self.assertRaises(KISSError):
            agent._docker_bash("echo hello", "test")


# ---------------------------------------------------------------------------
# Integration tests: perform_task happy path
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestPerformTaskSuccess(unittest.TestCase):
    def test_simple_success(self) -> None:
        """Agent completes a trivial task in one step."""
        agent = RelentlessAgent("Success-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='task completed'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertGreater(agent.budget_used, 0)
        self.assertGreater(agent.total_tokens_used, 0)

    def test_success_false_not_continue(self) -> None:
        """success=False, is_continue=False -> returns immediately."""
        agent = RelentlessAgent("FailNoCont")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=False, is_continue=False, "
                    "summary='failed'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertFalse(parsed["success"])


# ---------------------------------------------------------------------------
# Integration tests: continuation and exhaustion
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestContinuation(unittest.TestCase):
    def test_continuation_then_success(self) -> None:
        """is_continue=True triggers next session; '# Continue' present -> succeed."""
        agent = RelentlessAgent("Cont-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "If the prompt below contains the text '# Continue', "
                    "call finish(success=True, is_continue=False, "
                    "summary='completed after continuation'). "
                    "Otherwise call finish(success=False, is_continue=True, "
                    "summary='need to continue')"
                ),
                max_steps=5,
                max_budget=2.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

    def test_max_sub_sessions_exhausted(self) -> None:
        """All sub-sessions used -> KISSError."""
        agent = RelentlessAgent("Exhaust-Test")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(KISSError) as ctx:
                agent.run(
                    model_name=TEST_MODEL,
                    prompt_template=(
                        "Always call finish(success=False, is_continue=True, "
                        "summary='still working on it')"
                    ),
                    max_steps=5,
                    max_budget=2.0,
                    max_sub_sessions=2,
                    work_dir=td,
                    verbose=False,
                )
            self.assertIn("sub-sessions", str(ctx.exception))

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
    def test_exception_summarizer_succeeds(self) -> None:
        """executor.run fails (step limit=1), summarizer runs."""
        # Reset global budget to avoid interference from other tests
        original_used = Base.global_budget_used
        Base.global_budget_used = 0.0
        try:
            agent = RelentlessAgent("ExcSum-OK")
            with tempfile.TemporaryDirectory() as td:
                # max_steps=1 causes KISSError at step-count check before generation.
                # That exception is caught by perform_task, summarizer runs.
                # With max_sub_sessions=1, result has is_continue=True -> KISSError.
                with self.assertRaises(KISSError):
                    agent.run(
                        model_name=TEST_MODEL,
                        prompt_template="Do something.",
                        max_steps=1,
                        max_budget=2.0,
                        max_sub_sessions=1,
                        work_dir=td,
                        verbose=False,
                    )
        finally:
            Base.global_budget_used = original_used

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
    def test_with_arguments(self) -> None:
        agent = RelentlessAgent("Args-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "The word is {word}. "
                    "IMMEDIATELY call finish(success=True, is_continue=False, summary='done'). "
                    "Do NOT call any other tool first."
                ),
                arguments={"word": "hello"},
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

    def test_without_arguments(self) -> None:
        agent = RelentlessAgent("NoArgs")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='no args'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

    def test_with_system_instructions(self) -> None:
        agent = RelentlessAgent("SysInstr")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                system_instructions="You are a helpful assistant.",
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='sys'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

    def test_with_custom_tools(self) -> None:
        def my_tool(x: str) -> str:
            """Return x uppercased.

            Args:
                x: input string.

            Returns:
                Uppercased string.
            """
            return x.upper()

        agent = RelentlessAgent("Tools-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "First call my_tool(x='hello'), then IMMEDIATELY "
                    "call finish(success=True, is_continue=False, summary='used tool'). "
                    "Do NOT call any other tool."
                ),
                tools=[my_tool],
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])

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

    def test_docker_with_printer(self) -> None:
        """Test Docker path with printer (covers stream_callback)."""
        from kiss.core.print_to_console import ConsolePrinter

        agent = RelentlessAgent("DockerPrinter")
        printer = ConsolePrinter()
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary='docker with printer'). Do NOT call any other tool first."
                ),
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                docker_image="ubuntu:latest",
                printer=printer,
            )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])


# ---------------------------------------------------------------------------
# Docker-specific branch tests
# ---------------------------------------------------------------------------


class TestDockerBashWithManager(unittest.TestCase):
    """Test _docker_bash when docker_manager IS set (covers line 121)."""

    def test_docker_bash_runs_command(self) -> None:
        from kiss.docker.docker_manager import DockerManager

        agent = RelentlessAgent("DockerBashOK")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=None,
            docker_image="ubuntu:latest",
        )
        with DockerManager("ubuntu:latest") as dm:
            agent.docker_manager = dm
            result = agent._docker_bash("echo hello", "test echo")
            self.assertIn("hello", result)


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
