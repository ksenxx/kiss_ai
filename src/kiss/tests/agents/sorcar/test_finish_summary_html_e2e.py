# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: RelentlessAgent emits HTML summaries over the wire.

The pure conversion contract (``finish``/``ensure_html``, escaped-HTML
unescaping, CLI Result panel rendering) is covered by
``kiss.tests.core.test_finish_summary_html_e2e``; this file drives a real
:class:`RelentlessAgent` against a fake OpenAI-compatible server and
checks the summary contract end to end:

* the tool schema advertised to the LLM names ``summary_in_html``;
* an entity-escaped summary from the LLM reaches the result as real HTML;
* multi-session merges use HTML ``<h3>`` session markers instead of
  Markdown ``###`` headings;
* the system prompt instructs ``summary_in_html=``.
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
from kiss.tests.core.test_finish_summary_html_e2e import (
    ESCAPED_SUMMARY,
    UNESCAPED_SUMMARY,
)


class RecordingPrinter(Printer):
    """Printer that records every event as ``(type, content, kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, Any, dict[str, Any]]] = []
        # None disables token streaming so the fake (non-streaming)
        # OpenAI server can be used.
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:  # noqa: A002
        self.events.append((type, content, kwargs))
        return str(content)

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        return None

    def reset(self) -> None:
        return None

    def result_events(self) -> list[tuple[Any, dict[str, Any]]]:
        """Return ``(content, kwargs)`` for every ``type="result"`` event."""
        return [(c, kw) for (t, c, kw) in self.events if t == "result"]



def _make_tool_call_response(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> dict[str, Any]:
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



def _start_openai_server(
    responses: list[dict[str, Any]],
) -> tuple[Any, int, list[dict[str, Any]]]:
    """Start a fake OpenAI-compatible server replaying sequential responses.

    Returns the server, its port, and a list collecting every request body
    (so tests can inspect the tool schema the agent advertised).
    """
    call_count = [0]
    requests: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            try:
                requests.append(json.loads(raw))
            except Exception:
                requests.append({})
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port, requests



def _run_agent(
    responses: list[dict[str, Any]],
    *,
    max_sub_sessions: int = 5,
    printer: Printer | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    server, port, requests = _start_openai_server(responses)
    try:
        agent = RelentlessAgent("HtmlSummaryTest")
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Do the work.",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=max_sub_sessions,
                work_dir=td,
                verbose=False,
                printer=printer,
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
        return result, requests
    finally:
        server.shutdown()



class TestEndToEndAgentSummaryIsHtml:
    """A real agent run over the wire must produce an HTML summary."""

    def test_markdown_summary_from_llm_is_html_in_result(self) -> None:
        resp = _make_tool_call_response(
            "finish",
            {
                "success": True,
                "is_continue": False,
                "summary_in_html": "# Fixed\n\n- changed `a.py`",
            },
        )
        printer = RecordingPrinter()
        result, _ = _run_agent([resp], printer=printer)
        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        assert "<h1>" in parsed["summary"]
        assert "<li>" in parsed["summary"]
        (event_content, _kw) = printer.result_events()[-1]
        event_payload = yaml.safe_load(str(event_content))
        assert "<h1>" in event_payload["summary"]

    def test_multi_session_merge_uses_html_h3_markers(self) -> None:
        resp_continue = _make_tool_call_response(
            "finish",
            {
                "success": False,
                "is_continue": True,
                "summary_in_html": "<p>did A</p>",
            },
        )
        resp_done = _make_tool_call_response(
            "finish",
            {
                "success": True,
                "is_continue": False,
                "summary_in_html": "<p>finished B</p>",
            },
        )
        printer = RecordingPrinter()
        result, _ = _run_agent([resp_continue, resp_done], printer=printer)
        summary = yaml.safe_load(result)["summary"]
        assert "<h3>Previous Session 1</h3>" in summary
        assert "<h3>Final Session</h3>" in summary
        assert "### Previous Session" not in summary
        assert "### Final Session" not in summary
        assert "\n\n---\n\n" in summary

    def test_system_prompt_instructs_summary_in_html(self) -> None:
        resp = _make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary_in_html": "<p>ok</p>"},
        )
        _, requests = _run_agent([resp])
        messages = requests[0].get("messages", [])
        system_texts = [
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        ]
        joined = "\n".join(system_texts)
        assert "summary_in_html=" in joined
        assert 'summary="' not in joined


class TestFinishToolSchemaOverTheWire:
    """The finish tool schema advertised to the LLM uses the new name."""

    def test_tool_schema_advertises_summary_in_html(self) -> None:
        """The fake-server request body must show the renamed parameter."""
        resp = _make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary_in_html": "<p>ok</p>"},
        )
        _, requests = _run_agent([resp])
        assert requests, "no LLM request captured"
        tools = requests[0].get("tools", [])
        finish_tools = [
            t
            for t in tools
            if t.get("function", {}).get("name") == "finish"
        ]
        assert finish_tools, "finish tool not advertised"
        props = finish_tools[0]["function"]["parameters"]["properties"]
        assert "summary_in_html" in props
        assert "summary" not in props


class TestEscapedHtmlSummaryEndToEnd:
    """An escaped summary from the LLM must reach the result as real HTML."""

    def test_end_to_end_agent_escaped_summary_is_html_in_result(self) -> None:
        """A real agent run over the wire: the LLM calls finish with an
        entity-escaped summary; the emitted result must hold real HTML."""
        resp = _make_tool_call_response(
            "finish",
            {
                "success": True,
                "is_continue": False,
                "summary_in_html": ESCAPED_SUMMARY,
            },
        )
        printer = RecordingPrinter()
        result, _ = _run_agent([resp], printer=printer)
        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        assert parsed["summary"] == UNESCAPED_SUMMARY
        (event_content, _kw) = printer.result_events()[-1]
        event_payload = yaml.safe_load(str(event_content))
        assert event_payload["summary"] == UNESCAPED_SUMMARY
        assert "&lt;h3&gt;" not in event_payload["summary"]
