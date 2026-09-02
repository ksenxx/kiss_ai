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
  text — never raw ``<h1>``/``<li>`` tags and never via Markdown.

The RelentlessAgent end-to-end half of this contract (tool schema over the
wire, multi-session ``<h3>`` merge markers, escaped summaries through a
real agent run) lives in
``kiss.tests.agents.sorcar.test_finish_summary_html_e2e``.
"""

from __future__ import annotations

import inspect
import io
from typing import Any

import yaml

from kiss.core.print_to_console import ConsolePrinter
from kiss.core.utils import ensure_html, finish

# An entity-escaped HTML summary and its unescaped form, shared with the
# RelentlessAgent end-to-end twin of this file in
# kiss.tests.agents.sorcar.test_finish_summary_html_e2e.
ESCAPED_SUMMARY = (
    "&lt;h3&gt;Page created: &lt;code&gt;reports/log.html&lt;/code&gt;&lt;/h3&gt;"
    "&lt;p&gt;A single &lt;strong&gt;self-contained&lt;/strong&gt; page.&lt;/p&gt;"
    "&lt;ul&gt;&lt;li&gt;byte-for-byte answers&lt;/li&gt;&lt;/ul&gt;"
)
UNESCAPED_SUMMARY = (
    "<h3>Page created: <code>reports/log.html</code></h3>"
    "<p>A single <strong>self-contained</strong> page.</p>"
    "<ul><li>byte-for-byte answers</li></ul>"
)


class TestFinishParameterRenamedToSummaryInHtml:
    """The finish tool's final-result parameter MUST be ``summary_in_html``.

    The tool schema advertised to the LLM is generated from the function
    signature, so the signature IS the externally observable contract.
    """

    def test_signature_has_summary_in_html(self) -> None:
        params = list(inspect.signature(finish).parameters)
        assert params == [
            "success",
            "is_continue",
            "summary_in_html",
            "suggested_next_task",
        ]

    def test_keyword_call_uses_new_name(self) -> None:
        parsed = yaml.safe_load(finish(True, summary_in_html="<p>done</p>"))
        assert parsed["summary"] == "<p>done</p>"


class TestFinishSuggestedNextTask:
    """``finish``'s 4th parameter carries the agent's own follow-up proposal.

    The suggestion replaces the separate follow-up LLM call the server
    used to make, so it must travel inside the result YAML itself.
    """

    def test_suggestion_is_emitted_as_yaml_key(self) -> None:
        parsed = yaml.safe_load(
            finish(True, summary_in_html="<p>done</p>", suggested_next_task="Add tests"),
        )
        assert parsed == {
            "success": True,
            "is_continue": False,
            "summary": "<p>done</p>",
            "suggested_next_task": "Add tests",
        }

    def test_suggestion_is_stripped(self) -> None:
        parsed = yaml.safe_load(
            finish(True, False, "<p>x</p>", suggested_next_task="  Run the suite \n"),
        )
        assert parsed["suggested_next_task"] == "Run the suite"

    def test_empty_suggestion_omits_key(self) -> None:
        for empty in ("", "   ", None):
            parsed = yaml.safe_load(
                finish(True, summary_in_html="<p>x</p>", suggested_next_task=empty),  # type: ignore[arg-type]
            )
            assert "suggested_next_task" not in parsed, repr(empty)
            assert list(parsed) == ["success", "is_continue", "summary"]

    def test_non_string_suggestion_is_coerced(self) -> None:
        parsed = yaml.safe_load(
            finish(True, summary_in_html="<p>x</p>", suggested_next_task=42),  # type: ignore[arg-type]
        )
        assert parsed["suggested_next_task"] == "42"


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

    ESCAPED = ESCAPED_SUMMARY
    UNESCAPED = UNESCAPED_SUMMARY

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

    def test_suggested_next_task_is_shown(self) -> None:
        out = self._render("<p>done</p>", suggested_next_task="Add a regression test")
        assert "Suggested next: Add a regression test" in out

    def test_no_suggestion_line_without_suggestion(self) -> None:
        out = self._render("<p>done</p>")
        assert "Suggested next" not in out
