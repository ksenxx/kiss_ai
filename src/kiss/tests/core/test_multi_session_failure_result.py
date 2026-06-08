# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests verifying RelentlessAgent emits a final FAILED Result event.

These tests reproduce a bug where, in multi-session tasks, the Result tab
of the front-end never shows a final FAILED status when the task fails in
a session after one or more ``is_continue`` continuations.  The fix
guarantees that ``RelentlessAgent.perform_task`` always emits a final
``type="result"`` event carrying the merged summary and correct
``success``/``is_continue`` flags before returning or raising.
"""

import http.server
import json
import tempfile
import threading
import unittest
from typing import Any

import yaml

from kiss.core.kiss_error import KISSError
from kiss.core.printer import Printer, parse_result_yaml
from kiss.core.relentless_agent import RelentlessAgent


class _RecordingPrinter(Printer):
    """Printer that records every print() call for later inspection.

    ``token_callback`` is intentionally set to ``None`` at the instance
    level so the underlying model issues non-streaming chat completions
    (the fake OpenAI server in this test returns a single JSON body,
    not an SSE stream).  The class-level method still satisfies the
    :class:`Printer` ABC.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.tokens_offset: int = 0
        self.budget_offset: float = 0.0
        self.steps_offset: int = 0
        # Disable streaming: openai_compatible_model uses non-streaming
        # mode iff ``token_callback is None``.
        self.token_callback = None  # type: ignore[assignment,method-assign]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        self.events.append({"type": type, "content": content, "kwargs": dict(kwargs)})
        return ""

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        """Class-level definition to satisfy Printer ABC; shadowed by None in init."""

    def reset(self) -> None:
        pass


def _start_openai_server(responses: list[dict]) -> tuple[http.server.HTTPServer, int]:
    """Start a fake OpenAI-compatible server returning sequential responses."""
    call_count = [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
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


def _finish_response(success: bool, is_continue: bool, summary: str) -> dict:
    """Build a tool-call response that invokes finish() with given args."""
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
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({
                                    "success": success,
                                    "is_continue": is_continue,
                                    "summary": summary,
                                }),
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


def _last_result_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last event whose type is "result", or None."""
    for ev in reversed(events):
        if ev.get("type") == "result":
            return ev
    return None


class TestMultiSessionFailureResult(unittest.TestCase):
    """Failure across multiple sessions must surface a final FAILED Result event."""

    def test_failure_after_continue_emits_merged_failure_result(self) -> None:
        """Continue + explicit failure: final result event is success=False, merged."""
        responses = [
            _finish_response(False, True, "did A in session 1"),
            _finish_response(False, False, "session 2 failed B"),
        ]
        server, port = _start_openai_server(responses)
        try:
            printer = _RecordingPrinter()
            agent = RelentlessAgent("FailAfterContinue")
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Do multi-step work.",
                    max_steps=5,
                    max_budget=1.0,
                    max_sub_sessions=5,
                    work_dir=td,
                    verbose=False,
                    printer=printer,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
        finally:
            server.shutdown()

        parsed_return = yaml.safe_load(result)
        assert parsed_return["success"] is False
        assert "did A in session 1" in parsed_return["summary"]
        assert "session 2 failed B" in parsed_return["summary"]

        last = _last_result_event(printer.events)
        assert last is not None, "no type='result' event was ever emitted"
        parsed = parse_result_yaml(str(last["content"]))
        assert parsed is not None, f"final result event not parseable: {last}"
        assert parsed.get("success") is False, (
            f"final Result event must show success=False, got {parsed!r}"
        )
        assert not parsed.get("is_continue", False), (
            f"final Result event must have is_continue falsy, got {parsed!r}"
        )
        assert "did A in session 1" in parsed.get("summary", ""), (
            f"merged failure summary missing session 1 text: {parsed!r}"
        )
        assert "session 2 failed B" in parsed.get("summary", ""), (
            f"merged failure summary missing session 2 text: {parsed!r}"
        )

    def test_failure_after_session_exhaustion_emits_final_result(self) -> None:
        """Sub-session exhaustion must emit a FAILED Result event before raising."""
        # Every session reports is_continue=True with a different summary.
        responses = [
            _finish_response(False, True, "step 1 summary"),
            _finish_response(False, True, "step 2 summary"),
            _finish_response(False, True, "step 3 summary"),
        ]
        server, port = _start_openai_server(responses)
        try:
            printer = _RecordingPrinter()
            agent = RelentlessAgent("ExhaustionFail")
            raised = False
            with tempfile.TemporaryDirectory() as td:
                try:
                    agent.run(
                        model_name="gpt-4o-mini",
                        prompt_template="Do work that never finishes.",
                        max_steps=5,
                        max_budget=1.0,
                        max_sub_sessions=2,
                        work_dir=td,
                        verbose=False,
                        printer=printer,
                        model_config={
                            "base_url": f"http://127.0.0.1:{port}/v1",
                            "api_key": "sk-test",
                        },
                    )
                except KISSError:
                    raised = True
        finally:
            server.shutdown()

        assert raised, "expected KISSError after exhausting sub-sessions"
        last = _last_result_event(printer.events)
        assert last is not None, (
            "Bug: no type='result' event was emitted on sub-session exhaustion; "
            "front-end Result tab will not show FAILED."
        )
        parsed = parse_result_yaml(str(last["content"]))
        assert parsed is not None, f"final result event not parseable: {last}"
        assert parsed.get("success") is False, (
            f"exhaustion Result event must have success=False, got {parsed!r}"
        )
        assert not parsed.get("is_continue", False), (
            f"exhaustion Result event must have is_continue falsy, got {parsed!r}"
        )
        # The merged failure summary should mention each session's text.
        summary = parsed.get("summary", "")
        assert "step 1 summary" in summary, (
            f"merged failure summary missing session 1: {summary!r}"
        )
        assert "step 2 summary" in summary, (
            f"merged failure summary missing session 2: {summary!r}"
        )


if __name__ == "__main__":
    unittest.main()
