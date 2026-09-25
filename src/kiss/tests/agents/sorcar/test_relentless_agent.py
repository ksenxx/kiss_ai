# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for RelentlessAgent with actual LLM calls for 100% branch coverage."""

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar.relentless_agent import (
    CONTINUATION_PROMPT,
    IMPORTANT_INSTRUCTIONS,
    MAX_ZERO_PROGRESS_SESSIONS,
    TASK_PROMPT,
    WORK_DIR_LINE,
    RelentlessAgent,
    finish,
)
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-2.5-flash"


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
        """IMPORTANT_INSTRUCTIONS renders context risk, workdir, and PID."""
        formatted = IMPORTANT_INSTRUCTIONS.format(
            work_dir_line=WORK_DIR_LINE.format(work_dir="/tmp/test"),
            current_pid="12345",
        )
        self.assertIn("risk of running out of context", formatted)
        self.assertNotIn("**) or", formatted)
        self.assertIn("- Work dir: /tmp/test\n", formatted)
        self.assertIn("12345", formatted)
        self.assertIn("MOST IMPORTANT INSTRUCTIONS", formatted)

    def test_important_instructions_without_work_dir_line(self) -> None:
        """An empty work_dir_line leaves no blank line or dangling label."""
        formatted = IMPORTANT_INSTRUCTIONS.format(work_dir_line="", current_pid="7")
        self.assertNotIn("Work dir", formatted)
        self.assertIn("as HTML.\n- Current process PID: 7", formatted)

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
        result = finish(success="true", is_continue="yes", summary_in_html="x")  # type: ignore[arg-type]
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        self.assertTrue(parsed["is_continue"])


@requires_gemini_api_key
@unittest.skipUnless(_docker_available(), "Docker daemon not available")
class TestRunBranches(unittest.TestCase):
    @pytest.mark.slow
    def test_with_docker(self) -> None:
        """Test the Docker path in run()."""
        agent = RelentlessAgent("Docker-Test")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "IMMEDIATELY call finish(success=True, is_continue=False, "
                    "summary_in_html='docker test done'). Do NOT call any other tool first."
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

    @pytest.mark.slow
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
                    "finish(success=True, is_continue=False, summary_in_html='streamed'). "
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


def _start_openai_server(
    responses: list[dict], requests: list[dict] | None = None
) -> tuple[http.server.HTTPServer, int]:
    """Start a fake OpenAI-compatible server returning *responses* in order.

    The last response is repeated once the list is exhausted.  When
    *requests* is given, every request body is appended to it (parsed) so
    tests can inspect the prompts the agent actually sent.
    """
    call_count = [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(content_length))
            if requests is not None:
                requests.append(request)
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            body = json.dumps(responses[idx]).encode()
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


def _make_tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    """Build a fake OpenAI chat completion whose only content is one tool call."""
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


def _continue(summary: str) -> dict:
    """A ``finish`` call that hands the task to the next sub-session."""
    return _make_tool_call_response(
        "finish", {"success": False, "is_continue": True, "summary_in_html": summary}
    )


def _succeed(summary: str) -> dict:
    """A terminal, successful ``finish`` call."""
    return _make_tool_call_response(
        "finish", {"success": True, "is_continue": False, "summary_in_html": summary}
    )


def note(text: str) -> str:
    """Record *text*; a side-effect-free tool the fake model can call to show progress."""
    return f"noted: {text}"


_NOTE = _make_tool_call_response("note", {"text": "working"}, call_id="call_note")


def _run_scripted(
    responses: list[dict],
    *,
    max_sub_sessions: int = 10,
    docker_image: str | None = None,
    work_dir: str | None = None,
    requests: list[dict] | None = None,
) -> str:
    """Drive ``RelentlessAgent.run`` against a fake server replaying *responses*.

    Returns the run's YAML result; exceptions from the run propagate.  The
    request bodies the agent sent are appended to *requests* when given
    (so they can be inspected even after a raise).  *work_dir* defaults to
    a fresh temporary directory.  The ``note`` tool is available so a
    scripted session can make a real tool call before it calls ``finish``.
    """
    server, port = _start_openai_server(responses, requests)
    try:
        agent = RelentlessAgent("ScriptedTest")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Do multi-step work.",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=max_sub_sessions,
                work_dir=work_dir or td,
                docker_image=docker_image,
                verbose=False,
                tools=[note],
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
    finally:
        server.shutdown()
    return result


def _system_prompt(request: dict) -> str:
    """The system message of one captured chat-completion request."""
    return str(next(m["content"] for m in request["messages"] if m["role"] == "system"))


class TestMultiSessionSummaryMerge(unittest.TestCase):
    """Test that multi-session completions merge all session summaries."""

    def test_merged_summary_on_completion(self) -> None:
        """After 2 continue sessions + final success, summary merges all sessions.

        Each continuation session calls ``note`` first: a session that only
        calls ``finish`` counts as zero progress (see
        ``TestZeroProgressGuard``), and two of those in a row would stop the
        run before the final session.
        """
        server, port = _start_openai_server(
            [_NOTE, _continue("did A"), _NOTE, _continue("did B"), _succeed("did C")]
        )
        try:
            agent = RelentlessAgent("MergeTest")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Do multi-step work.",
                    max_steps=5,
                    max_budget=1.0,
                    max_sub_sessions=5,
                    work_dir=td,
                    verbose=False,
                    tools=[note],
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            parsed = yaml.safe_load(result)
            assert parsed["success"] is True
            summary = parsed["summary"]
            assert "<h3>Previous Session 1</h3>" in summary
            assert "did A" in summary
            assert "<h3>Previous Session 2</h3>" in summary
            assert "did B" in summary
            assert "<h3>Final Session</h3>" in summary
            assert "did C" in summary
            assert summary.index("did A") < summary.index("did B") < summary.index(
                "did C"
            )
        finally:
            server.shutdown()

    def test_single_session_summary_unchanged(self) -> None:
        """Single-session success does not add Session headers."""
        server, port = _start_openai_server([_succeed("all done")])
        try:
            agent = RelentlessAgent("SingleTest")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="gpt-4o-mini",
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
            assert parsed["summary"] == "<p>all done</p>"
            assert "<h3>Previous Session" not in parsed["summary"]
            assert "<h3>Final Session" not in parsed["summary"]
        finally:
            server.shutdown()


class TestZeroProgressGuard(unittest.TestCase):
    """The session loop stops after MAX_ZERO_PROGRESS_SESSIONS stalled continuations.

    A continuation session is stalled when it called no tool other than
    ``finish``, or when its summary is identical to the previous session's.
    Sessions are counted from the fake server's requests: every session
    ends with exactly one ``finish`` response.
    """

    def test_limit_is_two(self) -> None:
        """The task asks for two consecutive stalled sessions."""
        self.assertEqual(MAX_ZERO_PROGRESS_SESSIONS, 2)

    def test_two_sessions_without_tool_calls_stop_the_run(self) -> None:
        """Two finish-only continuations stop the run; the third response is never fetched."""
        with self.assertRaises(KISSError) as ctx:
            _run_scripted([_continue("did A"), _continue("did B"), _succeed("never")])
        self.assertIn("2 consecutive sub-sessions with no progress", str(ctx.exception))
        self.assertTrue(getattr(ctx.exception, "terminal_result_broadcast", False))

    def test_two_identical_summaries_stop_the_run(self) -> None:
        """Real tool calls do not save a session whose summary repeats the previous one.

        Session 1 sets the baseline summary (nothing to compare with), sessions
        2 and 3 repeat it verbatim: that is the second consecutive stall.
        """
        requests: list[dict] = []
        with self.assertRaises(KISSError) as ctx:
            _run_scripted(
                [
                    _NOTE, _continue("stuck"),
                    _NOTE, _continue("stuck"),
                    _NOTE, _continue("stuck"),
                    _succeed("never"),
                ],
                requests=requests,
            )
        self.assertIn("2 consecutive sub-sessions with no progress", str(ctx.exception))
        # 3 sessions x (note + finish) = 6 LLM calls; the success response was never used.
        self.assertEqual(len(requests), 6)

    def test_repeated_empty_summaries_stop_the_run(self) -> None:
        """An empty summary repeated is identical too.

        ``summaries`` keeps only non-empty text for the continuation prompt,
        so the comparison must not rely on it: sessions 2 and 3 repeating
        session 1's empty summary are two consecutive stalls.
        """
        requests: list[dict] = []
        with self.assertRaises(KISSError) as ctx:
            _run_scripted(
                [_NOTE, _continue("")] * 3 + [_succeed("never")],
                requests=requests,
            )
        self.assertIn("2 consecutive sub-sessions with no progress", str(ctx.exception))
        self.assertEqual(len(requests), 6)

    def test_progress_resets_the_streak(self) -> None:
        """A stalled session followed by a productive one does not accumulate.

        Sessions 1 and 3 are finish-only (stalled), but session 2 calls a tool
        and changes the summary, so no two stalls are consecutive and the
        run reaches its successful fourth session.
        """
        requests: list[dict] = []
        result = _run_scripted(
            [
                _continue("did A"),
                _NOTE, _continue("did B"),
                _continue("did C"),
                _NOTE, _succeed("did D"),
            ],
            requests=requests,
        )
        parsed = yaml.safe_load(result)
        self.assertTrue(parsed["success"])
        for text in ("did A", "did B", "did C", "did D"):
            self.assertIn(text, parsed["summary"])
        self.assertEqual(len(requests), 6)

    def test_max_sub_sessions_still_applies(self) -> None:
        """Productive continuations still end at max_sub_sessions with the old banner."""
        with self.assertRaises(KISSError) as ctx:
            _run_scripted(
                [_NOTE, _continue("did A"), _NOTE, _continue("did B"), _succeed("never")],
                max_sub_sessions=2,
            )
        self.assertIn("Task failed after 2 sub-sessions", str(ctx.exception))
        self.assertNotIn("no progress", str(ctx.exception))


class TestWorkDirLine(unittest.TestCase):
    """The ``Work dir`` line of IMPORTANT_INSTRUCTIONS depends on where tools run."""

    def test_host_run_names_the_work_dir(self) -> None:
        """Without a container the system prompt names the resolved work dir."""
        requests: list[dict] = []
        with tempfile.TemporaryDirectory() as td:
            _run_scripted([_succeed("done")], work_dir=td, requests=requests)
            expected = WORK_DIR_LINE.format(work_dir=Path(td).resolve())
        self.assertIn(expected, _system_prompt(requests[0]))

    @pytest.mark.slow
    @unittest.skipUnless(_docker_available(), "Docker daemon not available")
    def test_container_run_names_the_mounted_work_dir(self) -> None:
        """A container started from an image names the work dir: it is bind-mounted there.

        The container's tools see ``work_dir`` at its host path (and start
        in it), so the line points the model at files it can reach.
        """
        requests: list[dict] = []
        with tempfile.TemporaryDirectory() as td:
            _run_scripted(
                [_succeed("done")], work_dir=td, docker_image="ubuntu:latest",
                requests=requests,
            )
            expected = WORK_DIR_LINE.format(work_dir=Path(td).resolve())
        prompt = _system_prompt(requests[0])
        self.assertIn(expected, prompt)
        self.assertIn("- Current process PID:", prompt)

    @pytest.mark.slow
    @unittest.skipUnless(_docker_available(), "Docker daemon not available")
    def test_attached_container_without_the_mount_omits_the_work_dir(self) -> None:
        """An attached ``container:<id>`` that does not mount the work dir hides the line.

        The caller owns that container; naming a host path its Bash/Read
        cannot see would mislead the model.
        """
        import docker

        container = docker.from_env().containers.run(
            "ubuntu:latest", command="sleep infinity", detach=True,
        )
        try:
            requests: list[dict] = []
            with tempfile.TemporaryDirectory() as td:
                _run_scripted(
                    [_succeed("done")], work_dir=td,
                    docker_image=f"container:{container.id}", requests=requests,
                )
            prompt = _system_prompt(requests[0])
            self.assertNotIn("Work dir", prompt)
            self.assertIn("- Current process PID:", prompt)
        finally:
            container.remove(force=True)


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

    @pytest.mark.slow
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
