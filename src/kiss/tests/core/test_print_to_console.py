# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter


def _make() -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    return ConsolePrinter(file=buf), buf


class TestPrintMessageSystem(unittest.TestCase):
    def test_other_subtype_ignored(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrintMessageUser(unittest.TestCase):
    def test_blocks_without_is_error_skipped(self):
        p, buf = _make()
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "OK" not in out
        assert "FAILED" not in out


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p, buf = _make()
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
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
    output stranded above the opening rule (visually "above the
    Result panel" in the sorcar CLI interactive terminal).
    """

    def test_bash_output_appears_between_result_rules(self) -> None:
        p, buf = _make()
        # Simulate the exact sequence the agentic loop emits for a
        # real Bash tool call: ``tool_call`` first (the blue Bash
        # panel), then a stream of ``bash_stream`` chunks as the
        # child process prints output, finally a ``tool_result``
        # carrying the full captured output for the model.  Use a
        # unique marker for the streamed output that does NOT appear
        # in the rendered command, so position checks don't get
        # confused with the command echoed back inside the tool_call
        # blue panel.
        marker = "STREAMED_BASH_BODY_MARKER_xyz"
        p.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "sh -c 'unique_cmd'", "description": "x"},
        )
        p.print(marker + "\n", type="bash_stream")
        p.print(marker + "\n", type="tool_result", tool_name="Bash")

        out = buf.getvalue()
        # The streamed content must be visible on the terminal.
        assert marker in out, out
        # The opening ``RESULT`` rule must be present.
        assert "RESULT" in out, out
        # And critically: the streamed output must appear AFTER the
        # opening ``RESULT`` rule, not before it — i.e. inside the
        # result panel body.
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
        p.print("a\nb\nc\n", type="tool_result", tool_name="Bash")

        out = buf.getvalue()
        result_idx = out.index("RESULT")
        for line in ("a", "b", "c"):
            # Each chunk must appear AT LEAST once after the opening
            # ``RESULT`` rule.
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


if __name__ == "__main__":
    unittest.main()
