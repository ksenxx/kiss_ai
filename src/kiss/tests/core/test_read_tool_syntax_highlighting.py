# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the *Read*-tool syntax-highlighting invariant.

Invariant
=========
In the sorcar CLI *interactive* terminal the output of the ``Read``
tool MUST be rendered with syntax highlighting derived from the file
extension — exactly the same colouring the ``_format_tool_call`` panel
uses for the ``Write`` / ``Edit`` tools.  Concretely, when a model
issues ``Read(file_path="/tmp/x.py")`` the body that lands between the
opening and closing ``RESULT`` rules in the terminal must be produced
by :class:`rich.syntax.Syntax`, NOT by a plain ``file.write``.

These tests assert that contract end-to-end:

1. They drive :class:`kiss.core.print_to_console.ConsolePrinter`
   exactly the way the agentic loop (``KISSAgent._handle_function_call``)
   does — first a ``tool_call`` event for the ``Read`` call, then a
   ``tool_result`` event carrying the file contents AND the same
   ``tool_input`` (file_path / start_line) as the call.
2. They inspect the captured terminal output for the unambiguous
   marker that :class:`rich.syntax.Syntax(line_numbers=True)` was
   used — a leading line-number column (e.g. ``" 1 "``, ``" 2 "``)
   in front of each rendered line of the file body.  A plain
   ``file.write`` cannot produce a numbered gutter, so its absence
   in the previous implementation was the violation; its presence
   here proves the fix.
3. A negative-control test exercises ``Bash`` and the three
   non-content sentinels (``Error: ...``, ``(file is empty)``, the
   ``Read binary file ...`` header) — none of those must be
   syntax-highlighted, so the gutter must NOT appear.
4. An *agent-flow* test wires a real :class:`KISSAgent` and stub
   printer together to prove that the ``tool_input`` propagation
   from ``_handle_function_call`` to ``printer.print`` (which is
   what makes the highlighting possible at all) is in place.

The tests use a wide terminal width so rich never wraps the line
gutter onto a second visual line, which would defeat the simple
``" 1 "`` substring check.
"""

from __future__ import annotations

import io
import re
import unittest
from typing import Any

from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter

# A terminal wide enough that no source-code line we render in these
# tests gets wrapped onto a second visual line by Rich (which would
# put the line-number gutter only on the first visual line and break
# the "every Nth line has ` N ` in it" check).
_WIDE = 200


def _make_printer() -> tuple[ConsolePrinter, io.StringIO]:
    """Return a ``ConsolePrinter`` whose Rich console is wide enough.

    Rich's :class:`Console` snapshots the terminal width at
    construction time from the file object, so we have to overwrite
    ``_console`` with a fresh wide one before the printer renders
    anything.
    """
    from rich.console import Console

    buf = io.StringIO()
    p = ConsolePrinter(file=buf)
    p._console = Console(highlight=False, file=buf, width=_WIDE, force_terminal=False)
    return p, buf


def _between_result_rules(out: str) -> str:
    """Return the substring strictly between the opening ``RESULT`` /
    ``FAILED`` rule and the next horizontal rule that closes the
    result panel.

    A rule line is the all-``─`` row Rich draws (it always contains
    at least three ``─`` characters in a row and nothing else but the
    optional centred label).  We use a permissive regex so the result
    panel boundary is identified regardless of label, padding, or
    terminal width.
    """
    rule_re = re.compile(r"^[\s─]*(?:RESULT|FAILED)?[\s─]*$")
    lines = out.splitlines()
    open_idx = None
    for i, line in enumerate(lines):
        if "RESULT" in line or "FAILED" in line:
            open_idx = i
            break
    assert open_idx is not None, f"no RESULT/FAILED rule in output:\n{out}"
    close_idx = None
    for j in range(open_idx + 1, len(lines)):
        if lines[j].strip() and rule_re.fullmatch(lines[j]):
            close_idx = j
            break
    assert close_idx is not None, f"no closing rule in output:\n{out}"
    return "\n".join(lines[open_idx + 1 : close_idx])


def _has_line_gutter(body: str) -> bool:
    """True iff at least two consecutive numbered lines appear.

    Rich's ``Syntax(line_numbers=True)`` renders the gutter as
    right-aligned integers followed by whitespace at the start of
    each rendered line.  We look for two consecutive integers
    (``1`` and ``2``) on different lines at the start (ignoring
    leading whitespace) — a plain ``file.write`` of arbitrary file
    content cannot accidentally produce that pattern for the
    source snippets used in these tests.
    """
    lines = body.splitlines()
    seen_1 = any(re.match(r"^\s*1(?:\s|$)", line) for line in lines)
    seen_2 = any(re.match(r"^\s*2(?:\s|$)", line) for line in lines)
    return seen_1 and seen_2


class TestReadSyntaxHighlightingInvariant(unittest.TestCase):
    """The core invariant: Read output gets a syntax-highlighting
    gutter, everything else (Bash, sentinels) does not."""

    # ---- positive cases ---------------------------------------------

    def test_read_python_file_has_line_gutter(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.py"},
        )
        p.print(
            "def foo():\n    return 42\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.py"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), (
            "Read tool_result for a .py file rendered WITHOUT a "
            "syntax-highlighting line-number gutter (invariant "
            f"violated).\n\nResult-panel body:\n{body}"
        )
        # And the file content itself must be visible.
        assert "def foo" in body, body
        assert "return 42" in body, body

    def test_read_javascript_file_has_line_gutter(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.js"},
        )
        p.print(
            "function foo() {\n  return 42;\n}\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.js"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), body
        assert "function foo" in body, body

    def test_read_markdown_file_has_line_gutter(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.md"},
        )
        p.print(
            "# title\n\nsome body text\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.md"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), body

    def test_read_json_file_has_line_gutter(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.json"},
        )
        p.print(
            '{\n  "a": 1,\n  "b": 2\n}\n',
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.json"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), body

    def test_read_yaml_file_has_line_gutter(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.yaml"},
        )
        p.print(
            "a: 1\nb: 2\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.yaml"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), body

    def test_read_unknown_extension_still_has_gutter(self) -> None:
        """``lang_for_path`` returns ``"text"`` for unknown extensions
        (or no extension at all); the line-number gutter must still
        appear so the user gets a consistent visual treatment for
        every Read result."""
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.txt"},
        )
        p.print(
            "first line\nsecond line\nthird line\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.txt"},
        )
        body = _between_result_rules(buf.getvalue())
        assert _has_line_gutter(body), body
        assert "first line" in body, body

    def test_read_start_line_offset_carries_into_gutter(self) -> None:
        """When the model passes ``start_line=10`` the gutter must
        start at 10, not 1, so the line numbers shown to the user
        match the line numbers in the file."""
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.py", "start_line": 10},
        )
        p.print(
            "line ten\nline eleven\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.py", "start_line": 10},
        )
        body = _between_result_rules(buf.getvalue())
        # ``10`` must appear at the start of a body line (the
        # gutter), and the body content is still present.
        assert any(
            re.match(r"^\s*10(?:\s|$)", line) for line in body.splitlines()
        ), body
        assert "line ten" in body, body

    # ---- negative cases ---------------------------------------------

    def test_bash_tool_result_has_no_line_gutter(self) -> None:
        """``Bash`` output is plain command-line text — it must NOT
        be syntax-highlighted with a line gutter."""
        p, buf = _make_printer()
        p.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "echo hi"},
        )
        # NOTE: not going through the ``bash_stream`` path here so
        # the captured-content branch of ``_print_tool_result`` is
        # exercised — that's the path that used to plain-write *all*
        # tool results regardless of tool name.
        p.print(
            "first line\nsecond line\n",
            type="tool_result",
            tool_name="Bash",
            tool_input={"command": "echo hi"},
        )
        body = _between_result_rules(buf.getvalue())
        assert "first line" in body, body
        assert not _has_line_gutter(body), (
            "Bash tool_result rendered WITH a line gutter — only "
            "Read output should be syntax-highlighted.\n\nBody:\n"
            + body
        )

    def test_read_error_message_is_not_highlighted(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/missing.py"},
        )
        p.print(
            "Error: File not found: /tmp/missing.py",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/missing.py"},
        )
        body = _between_result_rules(buf.getvalue())
        assert "Error: File not found" in body, body
        assert not _has_line_gutter(body), body

    def test_read_empty_file_sentinel_is_not_highlighted(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/empty.py"},
        )
        p.print(
            "(file is empty)",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/empty.py"},
        )
        body = _between_result_rules(buf.getvalue())
        assert "(file is empty)" in body, body
        assert not _has_line_gutter(body), body

    def test_read_binary_header_is_not_highlighted(self) -> None:
        """A binary-attachment header is metadata, not source code."""
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.png"},
        )
        # After ``truncate_result`` strips the base64 attachment the
        # printer sees only the leading header line.
        header = (
            "Read binary file /tmp/x.png as image/png "
            "(123 bytes); content attached below.\n"
        )
        p.print(
            header,
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.png"},
        )
        body = _between_result_rules(buf.getvalue())
        assert "Read binary file" in body, body
        assert not _has_line_gutter(body), body

    def test_read_failed_tool_result_is_not_highlighted(self) -> None:
        """An ``is_error=True`` tool_result is an error envelope — it
        must use the ``FAILED`` rule and skip highlighting."""
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.py"},
        )
        p.print(
            "def foo():\n    return 42\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x.py"},
            is_error=True,
        )
        out = buf.getvalue()
        assert "FAILED" in out, out
        body = _between_result_rules(out)
        assert not _has_line_gutter(body), body

    def test_read_without_tool_input_falls_back_to_plain(self) -> None:
        """Backward-compat: legacy callers that don't forward
        ``tool_input`` on the tool_result event must still get a
        readable (plain) panel rather than a crash."""
        p, buf = _make_printer()
        p.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": "/tmp/x.py"},
        )
        p.print(
            "def foo():\n    return 42\n",
            type="tool_result",
            tool_name="Read",
        )
        body = _between_result_rules(buf.getvalue())
        assert "def foo" in body, body
        assert not _has_line_gutter(body), body


# ---- Agent-flow end-to-end tests ----------------------------------------


class _RecordingPrinter:
    """Minimal stub printer that records every ``print`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def print(self, content: Any, **kwargs: Any) -> str:
        self.calls.append((content, kwargs))
        return ""


class TestAgentForwardsToolInputOnReadResult(unittest.TestCase):
    """End-to-end check that the agentic loop forwards ``tool_input``
    on the ``tool_result`` event — the very wire that makes the
    syntax-highlighting branch reachable in the printer.
    """

    def test_execute_tool_emits_tool_input_on_read_result(self) -> None:
        printer = _RecordingPrinter()

        # Bypass ``KISSAgent.__init__`` (which expects a full
        # ModelConfig / API key chain) and inject only the
        # collaborators ``_execute_tool`` actually uses.
        agent = KISSAgent.__new__(KISSAgent)
        agent.name = "test"
        agent.printer = printer  # type: ignore[assignment]
        agent.function_map = {
            "Read": lambda file_path, **kw: f"contents of {file_path}"
        }

        function_args = {"file_path": "/tmp/hello.py", "start_line": 1}
        function_name, response = agent._execute_tool(
            {"name": "Read", "arguments": function_args}
        )

        assert function_name == "Read", function_name
        assert "contents of /tmp/hello.py" in response, response

        assert printer.calls, "agent did not emit any printer.print() call"
        result_calls = [
            (c, kw) for c, kw in printer.calls if kw.get("type") == "tool_result"
        ]
        assert result_calls, printer.calls
        _, kwargs = result_calls[-1]
        assert kwargs.get("tool_name") == "Read", kwargs
        assert kwargs.get("tool_input") == function_args, (
            "agent did not forward tool_input on the Read tool_result "
            "event; printer cannot syntax-highlight without it. "
            f"Got kwargs={kwargs}"
        )


class TestAgentExceptionPathDoesNotHighlight(unittest.TestCase):
    """When the Read tool *raises*, the agent builds a ``"Failed to call
    Read with {...}: …"`` diagnostic and forwards it as a
    ``tool_result``.  That string is NOT real source code, so it must
    NOT be rendered with the syntax-highlighting line gutter — it
    must surface the red ``FAILED`` rule instead so the user
    immediately sees the failure."""

    def test_read_raise_renders_failed_rule_not_gutter(self) -> None:
        printer, buf = _make_printer()

        def _raises(**_: Any) -> str:
            raise ValueError("simulated read failure")

        agent = KISSAgent.__new__(KISSAgent)
        agent.name = "test"
        agent.printer = printer  # type: ignore[assignment]
        agent.function_map = {"Read": _raises}

        function_args = {"file_path": "/tmp/x.py"}
        agent._execute_tool({"name": "Read", "arguments": function_args})

        out = buf.getvalue()
        assert "FAILED" in out, (
            "Agent did not surface FAILED rule for a raising Read "
            f"tool call.\n\nOutput:\n{out}"
        )
        assert "Failed to call Read" in out, out
        # The exception diagnostic must NOT be mis-rendered as
        # syntax-highlighted Python source.
        body = _between_result_rules(out)
        assert not _has_line_gutter(body), (
            "Agent exception path mis-rendered the 'Failed to call …' "
            "diagnostic as syntax-highlighted Python source.\n\nBody:\n"
            + body
        )


if __name__ == "__main__":
    unittest.main()
