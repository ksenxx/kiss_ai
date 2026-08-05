# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the ``finish`` tool's summary is ALWAYS HTML.

Reproduces (and locks in the fix for) the issue where:

* ``finish``'s final-result parameter was named ``summary`` and carried
  Markdown, so every interface had to render Markdown;
* the parameter must be renamed to ``summary_in_html`` and the emitted
  summary must ALWAYS be HTML (Markdown/plain-text input is converted);
* the CLI Result panel must render the HTML summary as styled terminal
  text — never raw ``<h1>``/``<li>`` tags and never via Markdown;
* multi-session merges must use HTML ``<h3>`` session markers instead of
  Markdown ``###`` headings.
"""

from __future__ import annotations

import http.server
import inspect
import io
import json
import tempfile
import threading
from typing import Any

import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import Printer
from kiss.core.utils import ensure_html, finish


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


class TestFinishParameterRenamedToSummaryInHtml:
    """The finish tool's final-result parameter MUST be ``summary_in_html``.

    The tool schema advertised to the LLM is generated from the function
    signature, so the signature IS the externally observable contract.
    """

    def test_signature_has_summary_in_html(self) -> None:
        params = list(inspect.signature(finish).parameters)
        assert params == ["success", "is_continue", "summary_in_html"]

    def test_keyword_call_uses_new_name(self) -> None:
        parsed = yaml.safe_load(finish(True, summary_in_html="<p>done</p>"))
        assert parsed["summary"] == "<p>done</p>"

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


class TestFinishAlwaysEmitsHtml:
    """``finish`` must ALWAYS emit an HTML summary in the YAML payload."""

    def test_markdown_input_is_converted_to_html(self) -> None:
        result = finish(
            True,
            False,
            "# Report\n\n- item one\n- item two\n\n```python\nx = 1\n```",
        )
        parsed = yaml.safe_load(result)
        summary = parsed["summary"]
        assert "<h1>" in summary
        assert "<li>item one</li>" in summary
        assert "<pre>" in summary or "<code" in summary
        assert "# Report" not in summary

    def test_html_input_is_passed_through(self) -> None:
        html = "<h2>Done</h2>\n<p>All <b>good</b>.</p>"
        parsed = yaml.safe_load(finish(True, False, html))
        assert parsed["summary"] == html

    def test_plain_text_becomes_html_paragraph(self) -> None:
        parsed = yaml.safe_load(finish(False, False, "it broke"))
        assert parsed["summary"] == "<p>it broke</p>"

    def test_empty_summary_stays_empty(self) -> None:
        parsed = yaml.safe_load(finish(True))
        assert parsed["summary"] == ""

    def test_wire_format_key_is_still_summary(self) -> None:
        parsed = yaml.safe_load(finish(True, False, "<p>x</p>"))
        assert set(parsed) == {"success", "is_continue", "summary"}


class TestEnsureHtml:
    """Unit coverage for the ``ensure_html`` conversion helper."""

    def test_markdown_heading_converted(self) -> None:
        assert "<h2>Title</h2>" in ensure_html("## Title")

    def test_html_fragment_untouched(self) -> None:
        assert ensure_html("<ul><li>a</li></ul>") == "<ul><li>a</li></ul>"

    def test_empty_string_untouched(self) -> None:
        assert ensure_html("") == ""

    def test_text_with_angle_math_is_converted_not_passed_through(self) -> None:
        out = ensure_html("if a < b then b > a")
        assert out.startswith("<p>")
        assert "&lt;" in out

    def test_doctype_document_passed_through(self) -> None:
        doc = "<!DOCTYPE html><html><body>hi</body></html>"
        assert ensure_html(doc) == doc

    def test_non_string_input_is_coerced(self) -> None:
        """Some LLMs pass numbers/lists for summary_in_html — must not crash."""
        assert ensure_html(42) == "<p>42</p>"  # type: ignore[arg-type]
        parsed = yaml.safe_load(finish(True, False, 42))  # type: ignore[arg-type]
        assert parsed["summary"] == "<p>42</p>"


class TestEscapedHtmlSummaryIsUnescaped:
    """Reproduces the bug where an agent emitted entity-escaped HTML
    (``&lt;h3&gt;`` instead of ``<h3>``) in ``summary_in_html``, so the
    summary reached the client as inert text and rendered as tag soup.

    ``ensure_html`` must detect summaries that BEGIN with an
    entity-escaped HTML tag, unescape them (even when escaped more than
    once), and return real HTML.
    """

    ESCAPED = (
        "&lt;h3&gt;Page created: &lt;code&gt;reports/log.html&lt;/code&gt;&lt;/h3&gt;"
        "&lt;p&gt;A single &lt;strong&gt;self-contained&lt;/strong&gt; page.&lt;/p&gt;"
        "&lt;ul&gt;&lt;li&gt;byte-for-byte answers&lt;/li&gt;&lt;/ul&gt;"
    )
    UNESCAPED = (
        "<h3>Page created: <code>reports/log.html</code></h3>"
        "<p>A single <strong>self-contained</strong> page.</p>"
        "<ul><li>byte-for-byte answers</li></ul>"
    )

    def test_escaped_html_is_unescaped(self) -> None:
        assert ensure_html(self.ESCAPED) == self.UNESCAPED

    def test_escaped_html_with_leading_whitespace_is_unescaped(self) -> None:
        assert ensure_html("\n  " + self.ESCAPED) == "\n  " + self.UNESCAPED

    def test_doubly_escaped_html_is_unescaped(self) -> None:
        doubly = self.ESCAPED.replace("&", "&amp;")
        assert ensure_html(doubly) == self.UNESCAPED

    def test_escaped_doctype_document_is_unescaped(self) -> None:
        doc = "&lt;!DOCTYPE html&gt;&lt;html&gt;&lt;body&gt;hi&lt;/body&gt;&lt;/html&gt;"
        assert ensure_html(doc) == "<!DOCTYPE html><html><body>hi</body></html>"

    def test_finish_unescapes_escaped_summary(self) -> None:
        parsed = yaml.safe_load(finish(True, False, self.ESCAPED))
        assert parsed["summary"] == self.UNESCAPED
        assert "&lt;h3&gt;" not in parsed["summary"]

    def test_plain_text_with_entities_is_still_markdown_converted(self) -> None:
        """``a &lt; b`` unescapes to no tag — must stay on the Markdown path."""
        out = ensure_html("prove that a &lt; b holds")
        assert out == "<p>prove that a &lt; b holds</p>"

    def test_escaped_tag_example_mid_text_is_not_unescaped(self) -> None:
        """Prose *about* HTML (escaped example not at the start) must keep
        its escaped entities so the example still displays literally."""
        out = ensure_html("Type the six characters &lt;h3&gt; to open a heading.")
        assert "&lt;h3&gt;" in out
        assert "<h3>" not in out

    def test_deeply_escaped_non_html_falls_back_to_markdown(self) -> None:
        """Text that keeps unescaping without ever yielding a tag must not
        loop forever and must fall back to Markdown conversion."""
        out = ensure_html("&amp;amp;amp;amp;lt; not html")
        assert out.startswith("<p>")
        assert "not html" in out

    def test_real_html_containing_escaped_entities_is_untouched(self) -> None:
        html = "<p>keep &lt;h3&gt; literal</p>"
        assert ensure_html(html) == html

    def test_end_to_end_agent_escaped_summary_is_html_in_result(self) -> None:
        """A real agent run over the wire: the LLM calls finish with an
        entity-escaped summary; the emitted result must hold real HTML."""
        resp = _make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary_in_html": self.ESCAPED},
        )
        printer = RecordingPrinter()
        result, _ = _run_agent([resp], printer=printer)
        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        assert parsed["summary"] == self.UNESCAPED
        (event_content, _kw) = printer.result_events()[-1]
        event_payload = yaml.safe_load(str(event_content))
        assert event_payload["summary"] == self.UNESCAPED
        assert "&lt;h3&gt;" not in event_payload["summary"]


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


class TestConsoleResultPanelRendersHtml:
    """The CLI Result panel must render the HTML — not show raw tags and
    not treat the summary as Markdown."""

    def _render(self, summary: str, **payload_extra: Any) -> str:
        payload = {"success": True, "is_continue": False, "summary": summary}
        payload.update(payload_extra)
        buf = io.StringIO()
        printer = ConsolePrinter(file=buf)
        printer.print(
            yaml.dump(payload, sort_keys=False),
            type="result",
            step_count=3,
            total_tokens=42,
            cost="$0.0100",
        )
        return buf.getvalue()

    def test_html_tags_are_rendered_not_shown(self) -> None:
        out = self._render(
            "<h1>Report</h1><ul><li>item one</li><li>item two</li></ul>"
        )
        assert "<h1>" not in out
        assert "<li>" not in out
        assert "Report" in out
        assert "item one" in out
        assert "item two" in out

    def test_html_is_not_interpreted_as_markdown(self) -> None:
        # In Markdown, *stars* would become emphasis and the literal stars
        # would vanish; in HTML rendering the stars must survive as text.
        out = self._render("<p>keep *stars* literal</p>")
        assert "*stars*" in out

    def test_failed_status_still_shown_with_html_summary(self) -> None:
        out = self._render("<p>nope</p>", success=False)
        assert "Status: FAILED" in out
        assert "nope" in out
        assert "<p>" not in out
