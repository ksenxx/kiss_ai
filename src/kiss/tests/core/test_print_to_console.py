# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic
with real objects.
"""

import functools
import io
import re
import unittest
from types import SimpleNamespace
from typing import Any

from kiss.agents.sorcar.docker_manager import _timeout_message, _with_exit_code
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_alternative
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
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


class TestStreamedBashStatus(unittest.TestCase):
    """What the ``Bash`` tools write around a streamed command's output reaches the console.

    The stream carries the output only, and the result of a streamed call
    used to print as its closing rule alone: a failed, a timed-out and a
    left-running command read as one that passed.
    """

    def setUp(self) -> None:
        self.printer, self.buf = _make()
        self.tools = UsefulTools(
            stream_callback=functools.partial(self.printer.print, type="bash_stream"),
        )

    def _body(self, result: str, interrupted: bool = False) -> list[str]:
        """Print *result* as the result ending a ``Bash`` call; return the lines in its rules."""
        self.printer.print(result, type="tool_result", tool_name="Bash", interrupted=interrupted)
        opening, *body, closing = self.buf.getvalue()[self.start:].splitlines()
        assert "RESULT" in opening and closing and not closing.strip("─"), (opening, closing)
        return body

    def _bash(self, **bash: Any) -> list[str]:
        """Make a real ``Bash`` call as the agent does; return the lines in its result's rules."""
        self.printer.print("Bash", type="tool_call", tool_input=bash)
        self.start = len(self.buf.getvalue())
        self.returned = self.tools.Bash(description="status", **bash)
        return self._body(self.returned)

    def test_the_host_tools_status_follows_the_stream(self) -> None:
        self.assertEqual(self._bash(command="echo out; exit 3"), ["out", "Error (exit code 3):"])
        self.assertEqual(
            self._bash(command="echo out; sleep 30", timeout_seconds=1),
            ["out", "Error: Command execution timeout after 1s. Output before the timeout:"],
        )
        left = self._bash(command="echo out; sleep 30", timeout_seconds=1, on_timeout="background")
        self.tools.bash_job(self.returned.split()[9], action="kill")
        notice = r"\Aout\nStill running after 1s; left running as background job \w+ .+\Z"
        self.assertRegex("\n".join(left), notice)
        # Output is the stream's, however it reads: printed again it would show twice.
        lookalike = ["Error: disk full", "[exit code: 1]"]
        self.assertEqual(self._bash(command="; ".join(f"echo '{x}'" for x in lookalike)), lookalike)

    def test_the_container_tools_status_and_a_cut_short_call(self) -> None:
        timed_out = "Error: command timed out after 5s and was killed. Output before the timeout:"
        polled = _with_exit_code("out", 2) + "\n[until 'x': not matched after 2 runs]"
        waited = "\nWaited on job j1 for up to 40 s instead of sleeping (running).\n"
        for result, status, interrupted in (
            (_with_exit_code("out\n", 2), ["[exit code: 2]"], False),
            (_timeout_message(5, "out\n"), [timed_out], False),
            # Before an ``until`` verdict and what the tool says of a long sleep.
            (polled + waited, ["[exit code: 2]"], False),
            (_with_exit_code("out\n", 2) + sleep_alternative(40), ["[exit code: 2]"], False),
            ("out\n" + sleep_alternative(40), [], False),  # passed: a note is no status
            # A call the user stopped says so, as its panel of the web transcript does.
            (USER_INTERRUPTED_MESSAGE, [USER_INTERRUPTED_MESSAGE], True),
        ):
            self.start = len(self.buf.getvalue())
            self.printer.print("out\n", type="bash_stream")
            self.assertEqual(self._body(result, interrupted), ["out", *status], result)


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
