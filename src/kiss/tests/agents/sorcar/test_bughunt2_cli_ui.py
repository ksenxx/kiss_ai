# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: escaped trailing backslashes survive the ptk idle prompt.

Bug: :func:`kiss.ui.cli.cli_repl._read_line_ptk` re-applied a
naive ``line.endswith("\\")`` continuation loop on the text that
:class:`~kiss.ui.cli.cli_prompt.PtkLineReader` returned.  The
prompt_toolkit Enter binding (``_submit_enter``) already implements the
shared POSIX-shell continuation rule from
:func:`kiss.ui.cli.cli_line_continuation.ends_with_line_continuation`:
an *odd* number of trailing backslashes continues in-buffer (the line
is never submitted), while an *even* number is an escaped literal
``\\`` and Enter submits.  Consequently the only submitted lines that
can end in a backslash carry an even (escaped-literal) count — and the
outer naive loop then wrongly treated them as continuations: it ate one
of the user's literal backslashes, re-opened the prompt, and glued the
next unrelated input line onto the mangled text (or, on EOF, silently
dropped the final backslash).

The fix routes the outer loop through the same
:func:`ends_with_line_continuation` helper, so an escaped ``\\\\``
submission is returned untouched while genuine continuation semantics
(exercised end to end by the regression tests below) stay intact.

No mocks: these tests drive the real :class:`PromptSession` through a
prompt_toolkit pipe input, exactly like a user typing at the idle
``sorcar`` prompt.
"""

from __future__ import annotations

import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter, _read_line_ptk


def _drive_read_line(
    tmp_path: Path,
    keystrokes: str,
    *,
    delay: float = 0.5,
) -> str | None:
    """Run :func:`_read_line_ptk` feeding *keystrokes* through a pipe.

    Bytes before the first ``\\x00`` marker are sent before the reader
    starts; the rest is delivered after a short timer once the event
    loop is running.  Mirrors the driver helper in
    :mod:`test_cli_prompt_line_continuation` but exercises the full
    ``_read_line_ptk`` wrapper (the code path the idle REPL actually
    calls), not just ``PtkLineReader.read``.
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
                return _read_line_ptk(reader, "> ")
        finally:
            timer.cancel()


class TestReadLinePtkEscapedBackslash:
    def test_escaped_double_backslash_submits_literal_unmangled(
        self, tmp_path: Path,
    ) -> None:
        """``literal \\\\`` + Enter must return both backslashes verbatim.

        Before the fix the outer ``endswith("\\")`` loop consumed one
        of the two literal backslashes and swallowed the next input
        line as a bogus continuation, returning
        ``"literal \\\nEXTRA"`` instead of ``"literal \\\\"``.
        """
        line = _drive_read_line(tmp_path, "literal \\\\\x00\rEXTRA\r")
        assert line == "literal \\\\", f"Got {line!r}"

    def test_escaped_backslash_not_dropped_on_eof(
        self, tmp_path: Path,
    ) -> None:
        """EOF right after an escaped ``\\\\`` submit keeps both chars.

        Before the fix the bogus continuation loop hit EOF (Ctrl+D)
        on its re-opened prompt and stripped the final backslash,
        returning ``"keep \\"`` instead of ``"keep \\\\"``.
        """
        line = _drive_read_line(tmp_path, "keep \\\\\x00\r\x04")
        assert line == "keep \\\\", f"Got {line!r}"

    def test_single_backslash_continuation_still_works(
        self, tmp_path: Path,
    ) -> None:
        """Regression guard: odd-count continuation still joins lines."""
        line = _drive_read_line(tmp_path, "one \\\x00\rtwo\r")
        assert line == "one \ntwo", f"Got {line!r}"

    def test_triple_backslash_keeps_escaped_pair_and_continues(
        self, tmp_path: Path,
    ) -> None:
        """Regression guard: ``\\\\\\`` = literal ``\\\\`` + continuation."""
        line = _drive_read_line(tmp_path, "foo \\\\\\\x00\rbar\r")
        assert line == "foo \\\\\nbar", f"Got {line!r}"

    def test_plain_line_still_submits(self, tmp_path: Path) -> None:
        """Regression guard: a line without backslashes is untouched."""
        line = _drive_read_line(tmp_path, "just one line\r")
        assert line == "just one line", f"Got {line!r}"
