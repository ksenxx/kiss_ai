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
    TASK_PROMPT,
    RelentlessAgent,
    finish,
)
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.0-flash"


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


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
        """Empty summary -> no progress_section added.

        When every sub-session returns ``is_continue=True`` with an
        empty summary, the relentless agent has no progress to carry
        forward and should exhaust ``max_sub_sessions`` and raise
        ``KISSError``.  If the Gemini API rate-limits mid-loop, a
        non-KISS exception is caught and converted into an error
        payload (no KISSError is raised) — skip in that case.
        """
        agent = RelentlessAgent("EmptySummary")
        with tempfile.TemporaryDirectory() as td:
            try:
                result = agent.run(
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
            except KISSError:
                return
            parsed = yaml.safe_load(result)
            summary = (parsed or {}).get("summary", "")
            if "429" in summary or "RESOURCE_EXHAUSTED" in summary:
                self.skipTest("Gemini API rate-limited (429)")
            self.fail(
                f"Expected KISSError after exhausting sub-sessions, got: "
                f"{result!r}"
            )


@requires_gemini_api_key
class TestExceptionPaths(unittest.TestCase):
    def test_exception_summarizer_also_fails(self) -> None:
        """Both executor and summarizer fail (global budget exceeded).

        When the model fails on the very first call (step_count <= 1),
        the agent returns immediately with success=False instead of
        retrying through the summarizer path.
        """
        original_used = Base.global_budget_used
        try:
            Base.global_budget_used = 201.0

            agent = RelentlessAgent("ExcSum-Fail")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name=TEST_MODEL,
                    prompt_template="Do something.",
                    max_steps=5,
                    max_budget=10.0,
                    max_sub_sessions=1,
                    work_dir=td,
                    verbose=False,
                )
                payload = yaml.safe_load(result)
                assert isinstance(payload, dict)
                assert payload["success"] is False
        finally:
            Base.global_budget_used = original_used


@requires_gemini_api_key
@unittest.skipUnless(_docker_available(), "Docker daemon not available")
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
        summary = (parsed or {}).get("summary", "")
        if "429" in summary or "RESOURCE_EXHAUSTED" in summary:
            self.skipTest("Gemini API rate-limited (429)")
        self.assertTrue(parsed["success"])
        self.assertIsNone(agent.docker_manager)


@requires_gemini_api_key
@unittest.skipUnless(_docker_available(), "Docker daemon not available")
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
        summary = (parsed or {}).get("summary", "")
        if "429" in summary or "RESOURCE_EXHAUSTED" in summary:
            self.skipTest("Gemini API rate-limited (429)")
        self.assertTrue(parsed["success"])


class TestMultiSessionSummaryMerge(unittest.TestCase):
    """Test that multi-session completions merge all session summaries."""

    def _start_openai_server(self, responses: list[dict]) -> tuple:
        """Start a fake OpenAI-compatible server returning sequential responses."""
        call_count = [0]
        response_list = responses

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(content_length)
                idx = min(call_count[0], len(response_list) - 1)
                call_count[0] += 1
                body = json.dumps(response_list[idx]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port

    @staticmethod
    def _make_tool_call_response(
        name: str, arguments: dict, call_id: str = "call_1"
    ) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }

    def test_merged_summary_on_completion(self) -> None:
        """After 2 continue sessions + final success, summary merges all sessions."""
        # Session 1: continue with summary "did A"
        resp1 = self._make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "did A"},
        )
        # Session 2: continue with summary "did B"
        resp2 = self._make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "did B"},
        )
        # Session 3: success with summary "did C"
        resp3 = self._make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary": "did C"},
        )
        server, port = self._start_openai_server([resp1, resp2, resp3])
        try:
            agent = RelentlessAgent("MergeTest")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="test-model",
                    prompt_template="Do multi-step work.",
                    max_steps=5,
                    max_budget=1.0,
                    max_sub_sessions=5,
                    work_dir=td,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            parsed = yaml.safe_load(result)
            assert parsed["success"] is True
            summary = parsed["summary"]
            assert "### Session 1" in summary
            assert "did A" in summary
            assert "### Session 2" in summary
            assert "did B" in summary
            assert "### Session 3" in summary
            assert "did C" in summary
        finally:
            server.shutdown()

    def test_single_session_summary_unchanged(self) -> None:
        """Single-session success does not add Session headers."""
        resp = self._make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary": "all done"},
        )
        server, port = self._start_openai_server([resp])
        try:
            agent = RelentlessAgent("SingleTest")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="test-model",
                    prompt_template="Do one-step work.",
                    max_steps=5,
                    max_budget=1.0,
                    max_sub_sessions=5,
                    work_dir=td,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            parsed = yaml.safe_load(result)
            assert parsed["success"] is True
            assert parsed["summary"] == "all done"
            assert "### Session" not in parsed["summary"]
        finally:
            server.shutdown()


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
