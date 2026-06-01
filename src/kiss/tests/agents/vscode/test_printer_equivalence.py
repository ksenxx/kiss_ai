"""Integration test: ``ConsolePrinter`` and ``JsonPrinter`` must
produce equivalent user-visible content for the same scripted print
calls.

The two printers render to very different media — Rich panels in the
terminal vs JSON events delivered to the browser — but they are
designed to mirror each other.  This test fans an identical sequence
of ``print()`` / ``token_callback`` / ``thinking_callback`` calls
through both printers and asserts that every text fragment that
should be user-visible appears in both sinks.

The test is intentionally type-by-type so that any future regression
points directly at the ``print()`` ``type`` that diverged.

These tests use real ``ConsolePrinter`` and ``JsonPrinter``
instances with no mocks/patches — per project policy ``broadcast()``
is overridden via subclassing so events are captured without going to
stdout or the DB.
"""

from __future__ import annotations

import io
import re
import unittest
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.core.print_to_console import ConsolePrinter

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Box-drawing characters that Rich uses for Panel borders; we strip
# these so substring checks for short strings like "Done" do not get
# tripped up by adjacent corner characters.
_BOX_CHARS = (
    "─━│┃┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻"
    "┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
    "═║╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬"
)


def _normalise_console(raw: str) -> str:
    """Strip ANSI escapes + Rich box characters and collapse whitespace.

    This produces a flat plain-text view of what the terminal user
    actually reads, regardless of which box style Rich picks or how
    wide the panel borders end up.

    Args:
        raw: The raw console output captured from a ``StringIO``.

    Returns:
        A whitespace-collapsed plain-text representation.
    """
    stripped = _ANSI_RE.sub("", raw)
    for ch in _BOX_CHARS:
        stripped = stripped.replace(ch, " ")
    return re.sub(r"\s+", " ", stripped).strip()


class _RecordingBrowserPrinter(JsonPrinter):
    """``JsonPrinter`` subclass that captures broadcast events
    in a list and skips transport / DB persistence.

    Used in tests so we can inspect the exact sequence of events the
    browser frontend would receive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Capture the event in ``self.events`` and record it for the tab."""
        with self._lock:
            self.events.append(dict(event))
            self._record_event(event)


@dataclass
class _Pair:
    """Helper: a paired ConsolePrinter + recording JsonPrinter,
    both driven by the same ``print()`` / ``token_callback`` calls.
    """

    console: ConsolePrinter
    console_buf: io.StringIO
    browser: _RecordingBrowserPrinter

    def both_print(self, *args: Any, **kwargs: Any) -> None:
        """Forward ``print()`` to both printers."""
        self.console.print(*args, **kwargs)
        self.browser.print(*args, **kwargs)

    def both_token(self, token: str) -> None:
        """Forward ``token_callback`` to both printers."""
        self.console.token_callback(token)
        self.browser.token_callback(token)

    def both_thinking(self, is_start: bool) -> None:
        """Forward ``thinking_callback`` to both printers."""
        self.console.thinking_callback(is_start)
        self.browser.thinking_callback(is_start)

    def console_text(self) -> str:
        """Return the captured console output, normalised for substring checks."""
        return _normalise_console(self.console_buf.getvalue())

    def browser_payload(self) -> str:
        """Concatenate every user-visible text field across recorded events.

        Walks the event list in order and joins the string values of the
        keys that carry user-visible text (``text``, ``content``,
        ``command``, ``description``, ``old_string``, ``new_string``,
        ``name``, ``path``).  Used for end-to-end fragment-existence checks.
        """
        keys = (
            "text", "content", "command", "description",
            "old_string", "new_string", "name", "path",
        )
        parts: list[str] = []
        for ev in self.browser.events:
            for key in keys:
                value = ev.get(key)
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)


def _make_pair() -> _Pair:
    """Construct a fresh ConsolePrinter + recording JsonPrinter pair.

    The console printer is forced to a wide, plain (no-colour) width so
    Rich panels do not wrap fragments across lines in a way that would
    defeat substring checks.
    """
    buf = io.StringIO()
    console_printer = ConsolePrinter(file=buf)
    # Force a wide, non-colour console so Rich Panels don't wrap our
    # test fragments and ANSI codes don't appear in the buffer.
    console_printer._console = Console(
        file=buf, highlight=False, width=240, force_terminal=False, no_color=True,
    )
    return _Pair(console_printer, buf, _RecordingBrowserPrinter())


class PrinterEquivalenceTest(unittest.TestCase):
    """One method per ``print()`` ``type`` plus a full-script integration."""

    def _assert_in_both(self, fragment: str, pair: _Pair) -> None:
        cons = pair.console_text()
        browser = pair.browser_payload()
        self.assertIn(
            fragment, cons,
            f"console output missing fragment {fragment!r}; got:\n{cons}",
        )
        self.assertIn(
            fragment, browser,
            f"browser payload missing fragment {fragment!r}; got:\n{browser}",
        )

    def test_system_prompt_text_appears_in_both(self) -> None:
        pair = _make_pair()
        text = "You are a coding agent."
        pair.both_print(text, type="system_prompt")
        self._assert_in_both(text, pair)
        self.assertIn({"type": "system_prompt", "text": text}, pair.browser.events)

    def test_prompt_text_appears_in_both(self) -> None:
        pair = _make_pair()
        text = "Refactor the cache module"
        pair.both_print(text, type="prompt")
        self._assert_in_both(text, pair)
        self.assertIn({"type": "prompt", "text": text}, pair.browser.events)

    def test_plain_text_appears_in_both(self) -> None:
        pair = _make_pair()
        text = "Hello from the agent"
        pair.both_print(text, type="text")
        self.assertIn(text, pair.console_text())
        text_deltas = "".join(
            ev["text"] for ev in pair.browser.events if ev.get("type") == "text_delta"
        )
        self.assertIn(text, text_deltas)

    def test_bash_stream_text_appears_in_both(self) -> None:
        pair = _make_pair()
        chunk1 = "Building...\n"
        chunk2 = "Done compiling\n"
        pair.both_print(chunk1, type="bash_stream")
        pair.both_print(chunk2, type="bash_stream")
        # Browser may have buffered the second chunk behind a timer;
        # force a synchronous flush so we can inspect the payload.
        pair.browser._flush_bash()
        cons = pair.console_buf.getvalue()
        self.assertIn(chunk1, cons)
        self.assertIn(chunk2, cons)
        browser_payload = "".join(
            ev["text"] for ev in pair.browser.events if ev.get("type") == "system_output"
        )
        self.assertIn(chunk1, browser_payload)
        self.assertIn(chunk2, browser_payload)

    def test_tool_call_bash_command_appears_in_both(self) -> None:
        pair = _make_pair()
        pair.both_print(
            "Bash",
            type="tool_call",
            tool_input={"command": "ls -la", "description": "List files"},
        )
        self._assert_in_both("Bash", pair)
        self._assert_in_both("ls -la", pair)
        self._assert_in_both("List files", pair)
        tool_calls = [ev for ev in pair.browser.events if ev.get("type") == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "Bash")
        self.assertEqual(tool_calls[0]["command"], "ls -la")
        self.assertEqual(tool_calls[0]["description"], "List files")

    def test_tool_call_edit_with_diff_appears_in_both(self) -> None:
        pair = _make_pair()
        pair.both_print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": "src/foo.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        )
        self._assert_in_both("Edit", pair)
        self._assert_in_both("src/foo.py", pair)
        self._assert_in_both("x = 1", pair)
        self._assert_in_both("x = 2", pair)
        tool_calls = [ev for ev in pair.browser.events if ev.get("type") == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "Edit")
        self.assertEqual(tool_calls[0]["path"], "src/foo.py")
        self.assertEqual(tool_calls[0]["old_string"], "x = 1")
        self.assertEqual(tool_calls[0]["new_string"], "x = 2")

    def test_tool_result_error_appears_in_both(self) -> None:
        """``tool_result`` with ``is_error=True`` must surface to both sinks.

        (The non-error path is intentionally not asserted here because
        ``ConsolePrinter`` and ``JsonPrinter`` diverge on it —
        see ``test_known_divergences_non_error_tool_result`` for the
        pinned current behaviour.)
        """
        pair = _make_pair()
        err = "command not found: foozle"
        pair.both_print(err, type="tool_result", is_error=True, tool_name="Bash")
        self._assert_in_both(err, pair)
        tool_results = [
            ev for ev in pair.browser.events if ev.get("type") == "tool_result"
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertIn(err, tool_results[0]["content"])
        self.assertIs(tool_results[0]["is_error"], True)

    def test_result_panel_content_and_metadata_in_both(self) -> None:
        pair = _make_pair()
        content = "All tests passing"
        pair.both_print(content, type="result", cost="$0.0042", total_tokens=12345)
        cons = pair.console_text()
        self.assertIn(content, cons)
        # Console formats tokens with a thousands separator.
        self.assertTrue("12,345" in cons or "12345" in cons,
                        f"expected tokens in console; got: {cons}")
        self.assertIn("0.0042", cons)
        results = [ev for ev in pair.browser.events if ev.get("type") == "result"]
        self.assertEqual(len(results), 1)
        self.assertIn(content, results[0]["text"])
        self.assertEqual(results[0]["total_tokens"], 12345)
        self.assertEqual(results[0]["cost"], "$0.0042")

    def test_thinking_callback_brackets_match_in_both(self) -> None:
        pair = _make_pair()
        pair.both_thinking(True)
        pair.both_token("hmm")
        pair.both_thinking(False)
        pair.both_token("answer")
        cons = pair.console_text()
        self.assertIn("Thinking", cons)
        self.assertIn("hmm", cons)
        self.assertIn("answer", cons)
        types = [ev["type"] for ev in pair.browser.events]
        self.assertIn("thinking_start", types)
        i_start = types.index("thinking_start")
        self.assertEqual(types[i_start + 1], "thinking_delta")
        self.assertEqual(types[i_start + 2], "thinking_end")
        # After thinking_end the next token must route to text_delta:
        tail = pair.browser.events[i_start + 3:]
        self.assertTrue(
            any(ev.get("type") == "text_delta" and ev.get("text") == "answer"
                for ev in tail),
            f"expected text_delta 'answer' after thinking_end; got: {tail}",
        )

    def test_token_callback_outside_thinking_emits_text_in_both(self) -> None:
        pair = _make_pair()
        pair.both_token("hello")
        pair.both_token(" world")
        cons = pair.console_text()
        self.assertIn("hello", cons)
        self.assertIn("world", cons)
        text_deltas = "".join(
            ev["text"] for ev in pair.browser.events if ev.get("type") == "text_delta"
        )
        self.assertEqual(text_deltas, "hello world")

    def test_full_script_equivalent_fragments_in_both(self) -> None:
        """Drive a representative end-to-end script through both printers
        and assert every user-visible text fragment appears in both."""
        pair = _make_pair()

        pair.both_print("You are helpful.", type="system_prompt")
        pair.both_print("Fix the bug.", type="prompt")
        pair.both_thinking(True)
        pair.both_token("planning ")
        pair.both_token("approach")
        pair.both_thinking(False)
        pair.both_token("Here is ")
        pair.both_token("the fix.")
        pair.both_print(
            "Bash",
            type="tool_call",
            tool_input={"command": "pytest", "description": "Run tests"},
        )
        pair.both_print("bash output line 1\n", type="bash_stream")
        pair.both_print("bash output line 2\n", type="bash_stream")
        pair.browser._flush_bash()
        pair.both_print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": "src/bug.py",
                "old_string": "bad value",
                "new_string": "good value",
            },
        )
        pair.both_print(
            "error: not found", type="tool_result", is_error=True, tool_name="Read",
        )
        pair.both_print("All done", type="result", cost="$0.10", total_tokens=999)

        fragments = [
            "You are helpful.",
            "Fix the bug.",
            "planning",
            "approach",
            "Here is",
            "the fix.",
            "Bash",
            "pytest",
            "Run tests",
            "bash output line 1",
            "bash output line 2",
            "Edit",
            "src/bug.py",
            "bad value",
            "good value",
            "error: not found",
            "All done",
        ]
        for frag in fragments:
            self._assert_in_both(frag, pair)

    def test_empty_text_is_dropped_in_both(self) -> None:
        """Whitespace-only ``type="text"`` must not produce output in
        either sink.  Previously the ConsolePrinter wrote a blank line
        while the JsonPrinter dropped the event silently.
        """
        pair = _make_pair()
        pair.both_print("   ", type="text")
        pair.both_print("\n", type="text")
        pair.both_print("", type="text")
        self.assertEqual(pair.console_buf.getvalue(), "")
        text_deltas = [
            ev for ev in pair.browser.events if ev.get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, [])

    def test_message_with_non_error_content_block_appears_in_both(self) -> None:
        """A ``message`` carrying ``content`` blocks with ``is_error=False``
        must reach both sinks.  Previously the ConsolePrinter only
        rendered the block when ``is_error`` was truthy, hiding
        successful tool blocks that the browser surfaces.
        """
        pair = _make_pair()

        class _Block:
            def __init__(self, content: str, is_error: bool) -> None:
                self.content = content
                self.is_error = is_error

        class _Msg:
            def __init__(self, blocks: list[_Block]) -> None:
                self.content = blocks

        block_text = "tool finished successfully"
        msg = _Msg([_Block(block_text, is_error=False)])
        pair.both_print(msg, type="message")
        self._assert_in_both(block_text, pair)
        tool_results = [
            ev for ev in pair.browser.events if ev.get("type") == "tool_result"
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertIs(tool_results[0]["is_error"], False)
        self.assertIn(block_text, tool_results[0]["content"])

    # ------------------------------------------------------------------
    # Previously-known divergences — now fixed.  These tests assert
    # that the content reaches both sinks.  They retain their previous
    # names so blame/history is preserved.
    # ------------------------------------------------------------------

    def test_known_divergence_non_error_tool_result(self) -> None:
        """``tool_result`` with ``is_error=False`` for a core tool
        (Bash/Read/Edit/Write) must reach both sinks.

        Previously the ConsolePrinter hid the success path; that
        divergence has been fixed so the terminal user now sees the
        same Read/Edit/Write success body that the browser shows.
        """
        pair = _make_pair()
        result_content = "file contents go here"
        pair.both_print(
            result_content, type="tool_result", is_error=False, tool_name="Read",
        )
        self._assert_in_both(result_content, pair)
        tool_results = [
            ev for ev in pair.browser.events if ev.get("type") == "tool_result"
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertIn(result_content, tool_results[0]["content"])
        self.assertIs(tool_results[0]["is_error"], False)

    def test_known_divergence_usage_info(self) -> None:
        """``usage_info`` content must reach both sinks.

        Previously the ConsolePrinter had no branch for ``usage_info``
        and silently dropped the call; that divergence has been fixed
        so the terminal user now sees the same per-step token / cost /
        step summary that the browser status bar shows.
        """
        pair = _make_pair()
        pair.both_print(
            "step 3", type="usage_info", total_tokens=100, cost="$0.0010", total_steps=3,
        )
        self._assert_in_both("step 3", pair)
        usage = [ev for ev in pair.browser.events if ev.get("type") == "usage_info"]
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0]["total_tokens"], 100)
        self.assertEqual(usage[0]["cost"], "$0.0010")
        self.assertEqual(usage[0]["total_steps"], 3)


if __name__ == "__main__":
    unittest.main()
