# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: first-step model errors must produce a visible result event.

Root cause: when a model fails on the very first call (e.g. codex CLI error,
auth failure), ``RelentlessAgent.perform_task`` returned an error YAML string
without ever calling ``printer.print(type="result")``.  In the VS Code task
runner, this meant no result event was broadcast, so the user saw nothing in
the chat webview.

Fix: ``RelentlessAgent.perform_task`` now calls ``printer.print(error_result,
type="result", ...)`` before returning early on first-step errors.
"""

from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import yaml

from kiss.core.relentless_agent import RelentlessAgent


def _auth_error_response() -> dict:
    """OpenAI-compatible 401 error response."""
    return {
        "error": {
            "message": "Incorrect API key provided: sk-test.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }


class _AuthErrorHandler(BaseHTTPRequestHandler):
    """Returns 401 for all requests to trigger a non-retryable error."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        body = json.dumps(_auth_error_response()).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class TestFirstStepErrorResult:
    """First-step model errors must print a result event via the printer."""

    def test_perform_task_prints_result_on_first_step_error(self) -> None:
        """When a model fails on step 1, the printer must receive a result event
        so that the VS Code webview (or any UI) can display the error."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthErrorHandler)
        port = server.server_address[1]
        import threading

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            from kiss.core.print_to_console import ConsolePrinter

            printer = ConsolePrinter()

            # Track whether print(type="result") is called
            result_events: list[dict[str, Any]] = []
            original_print = printer.print

            def tracking_print(
                content: Any, type: str = "text", **kwargs: Any,
            ) -> str:
                if type == "result":
                    result_events.append({
                        "content": str(content),
                        "type": type,
                        **kwargs,
                    })
                return original_print(content, type=type, **kwargs)

            printer.print = tracking_print  # type: ignore[assignment]

            agent = RelentlessAgent("auth-error-test")
            agent._reset(
                model_name=f"openai/gpt-4o-mini@http://127.0.0.1:{port}/v1",
                max_sub_sessions=5,
                max_steps=5,
                max_budget=10.0,
                work_dir=tempfile.mkdtemp(),
                docker_image=None,
                printer=printer,
            )
            agent.system_prompt = "You are a helpful assistant."
            agent.task_description = "Say hello"

            def noop() -> str:
                """A no-op tool."""
                return "ok"

            result = agent.perform_task([noop])

            # Verify YAML result indicates failure
            parsed = yaml.safe_load(result)
            assert isinstance(parsed, dict)
            assert parsed["success"] is False
            assert parsed.get("is_continue", False) is False

            # KEY ASSERTION: printer.print(type="result") must have been called
            assert len(result_events) >= 1, (
                "Expected at least one result event from the printer, "
                "but none were printed. This means the webview would show nothing."
            )
            result_content = result_events[0]["content"]
            # The content should be a YAML string containing success: false
            result_parsed = yaml.safe_load(result_content)
            assert isinstance(result_parsed, dict)
            assert result_parsed["success"] is False
            assert result_parsed.get("summary"), (
                "Result event must contain a non-empty summary/error message"
            )
        finally:
            server.shutdown()
            t.join(timeout=5)
