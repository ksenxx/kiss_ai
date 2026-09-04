# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (core-base): one implementation for the terminal result print.

``KISSAgent`` gained ``_print_result`` (shared by ``_run_non_agentic`` and
``_run_task_to_completion``), but ``_run_agentic_loop`` kept its own inline
copy of the same printer call — duplicated logic that had already drifted:
the helper skips an **empty** result (nothing to show), while the loop's
copy printed one, so a model that calls the built-in ``finish`` tool with
an empty ``result`` produced a green "Result" panel containing the
"(no result)" placeholder on the agentic path but no panel at all on the
non-agentic and run-to-completion paths.

The fix routes the agentic loop through ``_print_result``, so all three
run modes share exactly one implementation of "emit the terminal result
event".

Every test runs a real ``KISSAgent`` against a local OpenAI-compatible
HTTP server (streaming and non-streaming); nothing is mocked.

Branch coverage of the modified code (the ``_run_agentic_loop`` result
branch and ``_print_result``):

* result non-empty + printer set  -> panel printed (test 2)
* result empty + printer set      -> nothing printed (test 1)
* printer ``None`` (verbose off)  -> no print, result still returned (test 3)
"""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter

_MODEL = "gpt-4o-mini"
_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _finish_tool_call(result: str) -> dict[str, Any]:
    """Return an OpenAI tool-call object invoking the built-in ``finish``."""
    return {
        "id": "call_finish",
        "type": "function",
        "function": {"name": "finish", "arguments": json.dumps({"result": result})},
    }


class _FinishHandler(BaseHTTPRequestHandler):
    """Chat-completions endpoint that always calls ``finish``.

    The finish ``result`` argument is taken from the class attribute
    ``finish_result`` so each test controls it.  Answers both streaming
    (SSE) and non-streaming requests: an agent with a printer streams,
    one without does not.
    """

    finish_result = ""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        tool_call = _finish_tool_call(type(self).finish_result)
        if body.get("stream"):
            self._send_stream(tool_call)
        else:
            self._send_json(tool_call)

    def _send_json(self, tool_call: dict[str, Any]) -> None:
        payload = json.dumps({
            "id": "chatcmpl-finish",
            "object": "chat.completion",
            "model": _MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                "finish_reason": "tool_calls",
            }],
            "usage": _USAGE,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self, tool_call: dict[str, Any]) -> None:
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-finish",
                "object": "chat.completion.chunk",
                "model": _MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            for delta in (
                {"role": "assistant", "content": ""},
                {"tool_calls": [{"index": 0, **tool_call}]},
            )
        ]
        chunks.append({
            "id": "chatcmpl-finish",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })
        chunks.append({
            "id": "chatcmpl-finish",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [],
            "usage": _USAGE,
        })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def finish_server() -> Iterator[str]:
    """Serve a local chat-completions endpoint that always calls ``finish``."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def _run(base_url: str, printer: ConsolePrinter | None, verbose: bool | None) -> str:
    """Run one agent turn against the local endpoint and return its result."""
    agent = KISSAgent("audit0903-result-print")
    return agent.run(
        model_name=_MODEL,
        prompt_template="Do nothing and finish.",
        max_steps=5,
        max_budget=1.0,
        printer=printer,
        verbose=verbose,
        print_prompts=False,
        model_config={"base_url": base_url, "api_key": "sk-test"},
    )


def test_empty_finish_result_prints_no_result_panel(finish_server: str) -> None:
    """An empty finish result must not be printed on the agentic path.

    ``_print_result`` (the non-agentic and run-to-completion paths) skips
    empty results; the agentic loop's drifted inline copy printed a
    "Result" panel with the "(no result)" placeholder.  All paths must
    agree: nothing to show, nothing printed.
    """
    _FinishHandler.finish_result = ""
    out = io.StringIO()
    result = _run(finish_server, printer=ConsolePrinter(file=out), verbose=None)
    assert result == ""
    text = out.getvalue()
    assert "(no result)" not in text
    assert "Result" not in text


def test_nonempty_finish_result_still_prints_result_panel(finish_server: str) -> None:
    """A non-empty finish result must still produce the Result panel."""
    _FinishHandler.finish_result = "all done"
    out = io.StringIO()
    result = _run(finish_server, printer=ConsolePrinter(file=out), verbose=None)
    assert result == "all done"
    text = out.getvalue()
    assert "Result" in text
    assert "all done" in text


def test_no_printer_returns_result_without_printing(finish_server: str) -> None:
    """With ``verbose=False`` (no printer) the result is simply returned."""
    _FinishHandler.finish_result = "quiet done"
    result = _run(finish_server, printer=None, verbose=False)
    assert result == "quiet done"
