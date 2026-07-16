# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Universal backslash line-continuation helper for the sorcar CLI.

macOS Terminal.app supports neither xterm's ``modifyOtherKeys`` mode
nor the Kitty keyboard protocol, so Shift+Enter is delivered as a
bare ``\\r`` — physically indistinguishable from plain Enter.  No
byte-parsing trick can rescue multi-line input on that terminal.

To provide a universal multi-line entry mechanism that works on every
terminal (including Terminal.app), the interactive input areas
(:mod:`~kiss.agents.sorcar.cli_prompt` and
:mod:`~kiss.agents.sorcar.cli_steering`) treat a trailing backslash as
a line-continuation marker — the same convention shells (bash, zsh,
fish) use.  When the buffer to be submitted ends with an unescaped
``\\``, pressing Enter inserts a newline into the buffer instead of
submitting.

The rules mirror POSIX shell continuation semantics:

* A single trailing ``\\`` (optionally followed by ASCII whitespace) is
  a continuation marker: it is stripped, a real ``\\n`` is appended,
  and the buffer is NOT submitted.
* A pair of trailing backslashes (``\\\\``) is the escape for a
  literal ``\\`` — both are kept in the buffer and Enter submits.
* An odd number ``2k+1`` of trailing backslashes is ``k`` literal
  backslashes plus one continuation marker; the marker is consumed
  and ``2k`` backslashes remain in the buffer.
* Trailing ASCII spaces / tabs after the final backslash are allowed
  and removed together with the marker (so ``foo \\   `` + Enter is
  treated the same as ``foo \\`` + Enter).
* An empty buffer submits (there is no marker to detect).
* A buffer that is only whitespace + ``\\`` continues (matching the
  shell convention where indented continuation is valid).

This module exposes the rule itself, :func:`ends_with_line_continuation`
— shared by the mid-task steering box's raw-mode line editor and the
initial-prompt :class:`~prompt_toolkit.PromptSession` so every input
surface applies one source of truth — plus :func:`read_continuations`,
the read-until-complete loop built on it that the REPL's line readers
(prompt_toolkit, plain :func:`input`, and interactive readline) share.
"""

from __future__ import annotations

from collections.abc import Callable


def ends_with_line_continuation(buf: str) -> tuple[bool, int]:
    """Detect whether *buf* ends with an unescaped trailing backslash.

    The detection allows optional ASCII spaces / tabs (but never
    newlines) between the final backslash and the end of the buffer,
    matching the shell convention that ``foo \\    `` is still a
    continuation.  Escaped backslashes (``\\\\``) are NOT continuations:
    only an odd number of consecutive trailing backslashes triggers
    the continuation, and only the outermost one is consumed.

    Args:
        buf: Current input buffer text (as accumulated by the line
            editor or prompt_toolkit buffer).

    Returns:
        A ``(is_continuation, keep_len)`` tuple.  When
        ``is_continuation`` is ``True``, *keep_len* is the number of
        leading characters of *buf* to preserve; the caller then
        appends ``"\\n"`` in place of the stripped continuation
        marker (the trailing whitespace + the odd-last backslash).
        When ``is_continuation`` is ``False``, *keep_len* equals
        ``len(buf)`` — the buffer is returned untouched and Enter
        should submit it.
    """
    if not buf:
        return (False, 0)
    # Strip trailing ASCII spaces / tabs (but never newlines — an
    # embedded ``\n`` inside the buffer must not be silently
    # collapsed).  This is what lets ``"foo \\   " + Enter`` still
    # count as a continuation.
    end = len(buf)
    while end > 0 and buf[end - 1] in " \t":
        end -= 1
    if end == 0 or buf[end - 1] != "\\":
        return (False, len(buf))
    # Count how many consecutive backslashes end the buffer (before
    # the stripped whitespace tail).  An odd count is a continuation;
    # an even count is a fully-escaped literal ``\\`` sequence.
    count = 0
    j = end
    while j > 0 and buf[j - 1] == "\\":
        count += 1
        j -= 1
    if count % 2 == 0:
        return (False, len(buf))
    # Continuation: consume the trailing whitespace and the outermost
    # backslash.  Any preceding escaped backslashes (``count - 1``
    # of them, an even number) are retained in the buffer so the
    # user can still literally include ``\\`` characters at the end
    # of a continued line.
    return (True, end - 1)


def read_continuations(
    line: str,
    read_more: Callable[[], str],
    on_continue: Callable[[], None] | None = None,
    on_eof: Callable[[], None] | None = None,
    on_interrupt: Callable[[], None] | None = None,
) -> str:
    """Extend *line* with continuation rows until it stops continuing.

    Shared read loop for the REPL's line readers: while *line* ends
    with an unescaped trailing backslash (see
    :func:`ends_with_line_continuation`), the marker is stripped and
    one more row is read via *read_more*, joined with a real ``"\\n"``.

    Args:
        line: The initially-read input line.
        read_more: Zero-argument callable reading the next row (e.g.
            ``input(prompt)`` or the prompt_toolkit session's read).
        on_continue: Optional hook invoked before each *read_more*
            (the interactive readline path redraws its panel frame
            here).
        on_eof: Optional hook invoked when *read_more* raises
            ``EOFError`` (cursor cleanup), before returning the input
            accumulated so far (with the dangling marker stripped).
        on_interrupt: Optional hook invoked when *read_more* raises
            ``KeyboardInterrupt`` (panel cleanup); the interrupt is
            then re-raised for the caller's prompt-level handling.

    Returns:
        The (possibly multi-line) input.

    Raises:
        KeyboardInterrupt: Propagated from *read_more* after the
            *on_interrupt* hook runs.
    """
    while True:
        cont, keep = ends_with_line_continuation(line)
        if not cont:
            return line
        if on_continue is not None:
            on_continue()
        try:
            more = read_more()
        except EOFError:
            if on_eof is not None:
                on_eof()
            return line[:keep]
        except KeyboardInterrupt:
            if on_interrupt is not None:
                on_interrupt()
            raise
        line = line[:keep] + "\n" + more
