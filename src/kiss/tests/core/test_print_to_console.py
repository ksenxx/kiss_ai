# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic
with real objects.
"""

import io
import re
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter
from kiss.core.utils import finish


def _make() -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    return ConsolePrinter(file=buf), buf


class TestPrintUnknownType(unittest.TestCase):
    def test_unknown_type_prints_nothing(self):
        p, buf = _make()
        p.print(SimpleNamespace(unknown_attr="value"), type="no_such_type")
        assert buf.getvalue() == ""


class TestThinkingCallbackFlow(unittest.TestCase):
    """thinking_callback(True) -> token_callback -> thinking_callback(False) flow."""

    def test_full_thinking_flow(self):
        p, buf = _make()
        p.thinking_callback(True)
        p.token_callback("I think")
        p.thinking_callback(False)

        out = buf.getvalue()
        assert "Thinking" in out
        assert "I think" in out


class TestBashStreamDedup(unittest.TestCase):
    """Test that tool_result doesn't repeat bash_stream output."""

    def test_tool_result_error_skips_content_after_bash_stream(self):
        p, buf = _make()
        p.print("line1\n", type="bash_stream")
        buf_before = buf.getvalue()
        assert "line1" in buf_before
        p.print("line1\n", type="tool_result", tool_name="Bash", is_error=True)
        buf_after = buf.getvalue()
        new_output = buf_after[len(buf_before):]
        assert "FAILED" in new_output
        assert "line1" not in new_output


class TestBashStreamInResultPanel(unittest.TestCase):
    """Integration test: streamed Bash tool output must render INSIDE the
    Result panel (i.e. between the opening ``RESULT`` rule and the
    closing rule), not above the opening rule.

    Reproduces the bug where ``bash_stream`` events were written
    directly before the ``tool_result`` event opened the ``RESULT``
    rule, leaving the result panel body empty and the streamed bash
    output stranded above the opening rule — visually "above the
    Result panel" in any :class:`ConsolePrinter` terminal (the
    default printer of every agent run without an explicit one, e.g.
    a programmatic Python-API run).
    """

    def test_bash_output_appears_between_result_rules(self) -> None:
        p, buf = _make()
        marker = "STREAMED_BASH_BODY_MARKER_xyz"
        p.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "sh -c 'unique_cmd'", "description": "x"},
        )
        p.print(marker + "\n", type="bash_stream")
        p.print(marker + "\n", type="tool_result", tool_name="Bash")

        out = buf.getvalue()
        assert marker in out, out
        assert "RESULT" in out, out
        result_idx = out.index("RESULT")
        marker_idx = out.index(marker)
        assert marker_idx > result_idx, (
            "bash_stream output rendered ABOVE the RESULT rule "
            f"(marker at {marker_idx}, RESULT at {result_idx}) — "
            "the Result panel body is empty.\n\nFull output:\n" + out
        )

    def test_multiple_bash_stream_chunks_all_between_rules(self) -> None:
        """Every chunk streamed during a single Bash call must end up
        inside the same Result panel, not split across the boundary."""
        p, buf = _make()
        p.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "printf 'a\\nb\\nc\\n'"},
        )
        p.print("a\n", type="bash_stream")
        p.print("b\n", type="bash_stream")
        p.print("c\n", type="bash_stream")
        p.print("", type="bash_stream")  # writes nothing: the cursor stays at the line start
        p.print("a\nb\nc\n", type="tool_result", tool_name="Bash")

        out = buf.getvalue()
        assert "c\n\n" not in out, "a blank line follows the last chunk:\n" + out
        result_idx = out.index("RESULT")
        for line in ("a", "b", "c"):
            assert line in out[result_idx:], (
                f"streamed chunk {line!r} not present inside the "
                "Result panel body.\nFull output:\n" + out
            )

    def test_non_bash_tool_result_unaffected(self) -> None:
        """Regression guard: tools that never emit ``bash_stream``
        (e.g. ``Read``) must still print their tool_result content
        inside the Result rules."""
        p, buf = _make()
        p.print("Read", type="tool_call", tool_input={"file_path": "/tmp/x"})
        p.print("file contents here\n", type="tool_result", tool_name="Read")
        out = buf.getvalue()
        assert "RESULT" in out
        result_idx = out.index("RESULT")
        assert "file contents here" in out[result_idx:], out


def _result_panel(raw: str) -> tuple[bool, set[str]]:
    """Print *raw* as a ``result`` event; return (says FAILED, SGR codes of the border lines)."""
    p, _ = _make()
    p._console.record = True
    p.print(raw, type="result")
    out = p._console.export_text(styles=True)
    return "Status: FAILED" in out, set(re.findall(r"\x1b\[([\d;]+)m[╭│╰─]", out))


class TestResultPanelBorder(unittest.TestCase):
    """The border of the Result panel is red exactly when the panel says ``Status: FAILED``."""

    def test_border_is_red_only_for_a_failed_result(self) -> None:
        failed, not_failed = (True, {"1;31"}), (False, {"1;32"})  # bold red, bold green
        assert _result_panel(finish(False, False, "<p>it broke</p>", "retry")) == failed
        assert _result_panel(finish(True, False, "<p>all good</p>")) == not_failed
        # "Status: Continue" is no failure, although its ``success`` is false.
        assert _result_panel(finish(False, True, "<p>more to do</p>")) == not_failed
        assert _result_panel("plain text carries no status") == not_failed
        assert _result_panel("") == not_failed


if __name__ == "__main__":
    unittest.main()
