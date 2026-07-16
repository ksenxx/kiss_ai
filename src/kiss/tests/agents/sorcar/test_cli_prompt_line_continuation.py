# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: backslash line-continuation in the initial task prompt.

Drives a real :class:`prompt_toolkit.PromptSession` (through the
:class:`~kiss.ui.cli.cli_prompt.PtkLineReader`) with a pipe
input and asserts that ending a line with an unescaped ``\\`` before
Enter inserts a newline into the buffer instead of submitting.  These
tests exercise the same code path a user hits when typing their first
task at the sorcar prompt.
"""

from __future__ import annotations

import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter


def _drive(
    tmp_path: Path,
    keystrokes: str,
    *,
    delay: float = 0.5,
) -> str:
    """Run ``PtkLineReader.read`` and feed *keystrokes* through a pipe.

    Bytes before the first ``\\x00`` marker are sent before the reader
    starts; the rest is delivered after a short timer once the event
    loop is running.  Mirrors the helper in
    :mod:`test_cli_multiline_input` so the two suites share
    identical driver behaviour.
    """
    (tmp_path / "alpha.py").write_text("alpha\n")
    completer = CliCompleter(str(tmp_path))
    hist = tmp_path / "hist"
    first, _, rest = keystrokes.partition("\x00")
    with create_pipe_input() as pipe:
        def _send_rest() -> None:
            if rest:
                pipe.send_text(rest)

        timer = threading.Timer(delay, _send_rest)
        pipe.send_text(first)
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                return reader.read("> ")
        finally:
            timer.cancel()


class TestPromptLineContinuation:
    def test_single_backslash_plus_enter_continues_to_next_line(
        self, tmp_path: Path,
    ) -> None:
        """``line one \\`` + Enter + ``line two`` + Enter → ``"line one \\nline two"``.

        This is the universal (Terminal.app-safe) way to enter
        multi-line input.  Before the fix this returned only
        ``"line one \\"`` because plain Enter submitted the buffer.
        """
        line = _drive(tmp_path, "line one \\\x00\rline two\r")
        assert line == "line one \nline two"

    def test_three_lines_joined_by_backslash_continuation(
        self, tmp_path: Path,
    ) -> None:
        line = _drive(tmp_path, "one\\\x00\rtwo\\\rthree\r")
        assert line == "one\ntwo\nthree"

    def test_backslash_with_trailing_whitespace_still_continues(
        self, tmp_path: Path,
    ) -> None:
        line = _drive(tmp_path, "hello \\   \x00\rworld\r")
        assert line == "hello \nworld"

    def test_escaped_double_backslash_submits_literal(
        self, tmp_path: Path,
    ) -> None:
        # ``literal \\`` typed (i.e. two backslashes at the end of the
        # user input) is an escaped literal — Enter submits.
        line = _drive(tmp_path, "literal \\\\\x00\r")
        assert line == "literal \\\\"

    def test_odd_backslash_count_continues_retaining_even(
        self, tmp_path: Path,
    ) -> None:
        line = _drive(tmp_path, "foo \\\\\\\x00\rbar\r")
        assert line == "foo \\\\\nbar"

    def test_no_continuation_plain_enter_still_submits(
        self, tmp_path: Path,
    ) -> None:
        line = _drive(tmp_path, "just one line\r")
        assert line == "just one line"

    def test_continuation_and_shift_enter_can_be_mixed(
        self, tmp_path: Path,
    ) -> None:
        # First newline via ``\`` continuation (universal),
        # second via CSI-u Shift+Enter (modern terminals).
        line = _drive(
            tmp_path,
            "one\\\x00\rtwo\x1b[13;2uthree\r",
        )
        assert line == "one\ntwo\nthree"
