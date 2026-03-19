"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import http.server
import json
import tempfile
import threading
import unittest

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

    def test_auth_error_returns_immediately(self) -> None:
        """Authentication error (401) returns finish(False, False, cause)."""
        server, port = self._start_fake_server(
            401,
            {"error": {"message": "Invalid API key", "type": "invalid_api_key"}},
        )
        try:
            agent = RelentlessAgent("AuthError")
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
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-invalid",
                    },
                )
            parsed = yaml.safe_load(result)
            assert parsed["success"] is False
            assert parsed["is_continue"] is False
            assert "summary" in parsed
            assert len(parsed["summary"]) > 0
        finally:
            server.shutdown()

    def test_permission_denied_returns_immediately(self) -> None:
        """Permission denied error (403) returns finish(False, False, cause)."""
        server, port = self._start_fake_server(
            403,
            {"error": {"message": "Permission denied", "type": "permission_denied"}},
        )
        try:
            agent = RelentlessAgent("PermDenied")
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
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-invalid",
                    },
                )
            parsed = yaml.safe_load(result)
            assert parsed["success"] is False
            assert parsed["is_continue"] is False
        finally:
            server.shutdown()

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
