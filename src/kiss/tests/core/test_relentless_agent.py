"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import http.server
import json
import tempfile
import threading
import unittest

import yaml

from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import (
    CONTINUATION_PROMPT,
    IMPORTANT_INSTRUCTIONS,
    STALL_THRESHOLD,
    STALL_WARNING,
    TASK_PROMPT,
    RelentlessAgent,
    _detect_stall,
    finish,
)
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.0-flash"


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
        """CONTINUATION_PROMPT has progress_text and continuation_number."""
        formatted = CONTINUATION_PROMPT.format(progress_text="step 1 done", continuation_number=3)
        self.assertIn("step 1 done", formatted)
        self.assertIn("Continuation 3", formatted)
        self.assertIn("Continue", formatted)


class TestFinish(unittest.TestCase):
    """Tests for the finish() tool function."""

    def test_finish_string_true(self) -> None:
        """finish() converts string 'true' to bool True."""
        result = finish(success="true", is_continue="yes", summary="x")  # type: ignore[arg-type]
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertTrue(parsed["is_continue"])


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
        original_used = Base.global_budget_used
        try:
            Base.global_budget_used = 201.0

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
            Base.global_budget_used = original_used


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


class TestDetectStall(unittest.TestCase):
    """Tests for _detect_stall()."""

    def test_custom_threshold(self) -> None:
        """Custom threshold parameter works."""
        error_line = "test_y failed with assertion error"
        summaries = [f"Attempt.\n{error_line}"] * 5
        result = _detect_stall(summaries, threshold=5)
        self.assertTrue(len(result) > 0)
        # Below threshold returns empty
        result = _detect_stall(summaries[:4], threshold=5)
        self.assertEqual(result, set())

    def test_threshold_constant(self) -> None:
        """STALL_THRESHOLD is 3."""
        self.assertEqual(STALL_THRESHOLD, 3)


class TestStallWarningTemplate(unittest.TestCase):
    """Test STALL_WARNING template formatting."""

    def test_stall_warning_placeholders(self) -> None:
        formatted = STALL_WARNING.format(continuation_number=5)
        self.assertIn("5 times", formatted)
        self.assertIn("Stall Warning", formatted)
        self.assertIn("ROOT CAUSE", formatted)
        self.assertIn("STALLED:", formatted)


@requires_gemini_api_key
class TestStallDetectionIntegration(unittest.TestCase):
    """Integration test: stall detection triggers in perform_task."""

    def test_stall_detected_returns_stall_result(self) -> None:
        """Agent producing same error summary 3+ times triggers stall detection."""
        agent = RelentlessAgent("StallIntegration")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Your ONLY job: call finish(success=False, is_continue=True, "
                    "summary='test_example FAILED with AssertionError: expected 42 got 0. "
                    "The error persists in module foo.py line 10.'). "
                    "Do NOT modify the summary text. Do NOT call any other tool."
                ),
                max_steps=5,
                max_budget=3.0,
                max_sub_sessions=6,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertFalse(parsed["success"])
        self.assertFalse(parsed.get("is_continue", True))
        self.assertIn("STALL DETECTED", parsed["summary"])

    def test_stall_warning_added_after_threshold(self) -> None:
        """Stall warning is added after threshold continuations without common errors."""
        agent = RelentlessAgent("StallWarn")
        with tempfile.TemporaryDirectory() as td:
            try:
                result = agent.run(
                    model_name=TEST_MODEL,
                    prompt_template=(
                        "Your ONLY job: call finish(success=False, is_continue=True, "
                        "summary='Working on step of the task, making progress'). "
                        "Do NOT modify the summary text. Do NOT call any other tool. "
                        "IGNORE any stall warnings in the continuation prompt."
                    ),
                    max_steps=5,
                    max_budget=5.0,
                    max_sub_sessions=5,
                    work_dir=td,
                    verbose=False,
                )
                # Agent may have reacted to stall warning and stopped
                parsed = yaml.safe_load(result)
                self.assertFalse(parsed["success"])
            except KISSError:
                pass  # Expected if all sub-sessions exhausted


class TestNonRetryableModelErrors(unittest.TestCase):
    """Test that non-retryable model errors return finish(False, False, cause)."""

    def _start_fake_server(self, status: int, body: dict) -> tuple:
        """Start a fake HTTP server that returns the given status and JSON body."""
        response_body = json.dumps(body).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port

    def test_connection_error_returns_immediately(self) -> None:
        """Connection error (unreachable server) returns finish(False, False, cause)."""
        agent = RelentlessAgent("ConnError")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name="test-model",
                prompt_template="Do something.",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
                model_config={
                    "base_url": "http://127.0.0.1:1/v1",
                    "api_key": "sk-invalid",
                },
            )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert parsed["is_continue"] is False


if __name__ == "__main__":
    unittest.main()
