# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a fail-fast error in a later sub-session keeps the
earlier sessions' summaries in the terminal result.

Session 1 hands off with ``is_continue=True`` ("did A"); session 2's first
model call fails with a non-retryable provider error.  Before the fix the
error branch of :meth:`RelentlessAgent.perform_task` returned only
``"<Type>: <msg>"`` and printed it with the executor's own counters, so the
persisted result and the Result panel lost session 1's work.  Now the error
is the ``<h3>Final Session</h3>`` banner of the merged summary, emitted
through the same merged ``result`` event as every other terminal path.
"""

from __future__ import annotations

import http.server
import json
import tempfile
import threading
from typing import Any

import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.printer import Printer


class RecordingPrinter(Printer):
    """Printer that records every ``type="result"`` event."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[Any, dict[str, Any]]] = []
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        if type == "result":
            self.results.append((content, kwargs))

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        return None

    def reset(self) -> None:
        return None


_CONTINUE = {
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
                            "arguments": json.dumps(
                                {
                                    "success": False,
                                    "is_continue": True,
                                    "summary_in_html": "did A",
                                }
                            ),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
}
_DENIED = {"error": {"message": "permission denied for this key", "type": "forbidden"}}


def _start_server() -> tuple[http.server.HTTPServer, int]:
    """Answer the first call with a continue hand-off, every later one with 403."""
    calls = [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls[0] += 1
            status, payload = (200, _CONTINUE) if calls[0] == 1 else (403, _DENIED)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_fail_fast_error_keeps_prior_session_summaries() -> None:
    """The error result of session 2 carries session 1's summary and the error text."""
    server, port = _start_server()
    printer = RecordingPrinter()
    try:
        with tempfile.TemporaryDirectory() as td:
            result = RelentlessAgent("FailFastMerge").run(
                # A routed name has no OpenRouter twin, so the 403 is not
                # answered by a live fallback model.
                model_name="openrouter/openai/gpt-4o-mini",
                prompt_template="Do multi-step work.",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
                printer=printer,
                model_config={"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "sk-t"},
            )
    finally:
        server.shutdown()

    payload = yaml.safe_load(result)
    assert payload["success"] is False
    assert payload["is_continue"] is False
    summary = payload["summary"]
    assert "<h3>Previous Session 1</h3>" in summary
    assert "did A" in summary
    assert "<h3>Final Session</h3>" in summary
    assert "KISSError: " in summary and "Non-retryable error from model" in summary
    assert "permission denied" in summary

    # One inner Result (session 1's hand-off) and the outer merged terminal Result.
    assert len(printer.results) == 2
    final_content, kwargs = printer.results[-1]
    assert yaml.safe_load(str(final_content)) == payload
    assert {"step_count", "total_tokens", "cost", "cost_usd"} <= kwargs.keys()
