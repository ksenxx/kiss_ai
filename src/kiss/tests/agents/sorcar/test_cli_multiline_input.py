# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for multi-line input + word-wrap in the sorcar CLI.

The interactive sorcar prompt (:mod:`kiss.agents.sorcar.cli_prompt`)
must let the user enter multi-line tasks directly in the input box and
must word-wrap the visible text instead of letting it scroll
horizontally off the panel.

These tests drive a real :class:`PromptSession` through a pipe input
(the same pattern as :mod:`test_at_mention_picker`) and assert that:

* Alt+Enter (Esc+Enter byte pair *or* the modifyOtherKeys / CSI-u
  combined sequences), Ctrl+J, Ctrl+Enter, Ctrl+Shift+Enter, and
  Shift+Enter (CSI-u *and* xterm modifyOtherKeys) all insert a real
  ``\\n`` into the buffer instead of submitting the line.
* A bare ``Enter`` still submits.
* The :class:`PromptSession` is configured with ``multiline=True`` and
  ``wrap_lines=True`` so long lines wrap inside the framed panel.
* A ``prompt_continuation`` is wired so wrapped / continuation visual
  rows still carry the cyan ``│`` left border of the input panel.
"""

from __future__ import annotations

import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.agents.sorcar.cli_prompt import PtkLineReader, _prompt_continuation
from kiss.agents.sorcar.cli_repl import CliCompleter


def _make_project_tree(tmp_path: Path) -> Path:
    """Create a tiny project tree so the ``@``-mention picker has files.

    The completion-menu regression tests need at least one file the
    picker can highlight when the user types ``@a`` so that pressing
    Down moves the buffer into a "completion selected" state.
    """
    (tmp_path / "alpha.py").write_text("alpha\n")
    (tmp_path / "another.md").write_text("another\n")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "a_module.py").write_text("module\n")
    return tmp_path


def _drive(
    tmp_path: Path, keystrokes: str, *, delay: float = 0.5,
) -> str:
    """Run ``PtkLineReader.read`` and feed *keystrokes* through a pipe.

    The early text is sent before the reader starts so the buffer
    already contains it; the rest is delivered by a background timer
    once prompt_toolkit's event loop is running, mirroring the helper
    used in :mod:`test_at_mention_picker`.
    """
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


def test_alt_enter_inserts_newline_then_enter_submits(tmp_path: Path) -> None:
    """Alt+Enter (Esc+Enter) inserts ``\\n``; Enter submits a multi-line line.

    Reproduces the missing multi-line input feature: typing
    ``hello`` + Alt+Enter + ``world`` + Enter must return
    ``"hello\\nworld"``.  Before the fix the very first ``\\r`` after
    Esc was interpreted as a plain Enter (the session was not
    multi-line) and the call returned just ``"hello"``.
    """
    # ``\x1b\r`` is the Esc+Enter (a.k.a. Alt+Enter / Meta+Enter)
    # sequence delivered by every terminal that does not also route
    # Shift+Enter through a CSI-u / modifyOtherKeys binding.
    line = _drive(tmp_path, "hello\x00\x1b\rworld\r")
    assert "\n" in line, f"expected a newline in {line!r}"
    assert line == "hello\nworld"


def test_ctrl_j_inserts_newline(tmp_path: Path) -> None:
    """Ctrl+J (Linefeed, ``\\n``) inserts a real newline.

    Some terminals (notably macOS Terminal.app with the default
    keyboard layout) deliver Ctrl+J as a way to send a literal LF — it
    must therefore insert a newline rather than submit.
    """
    line = _drive(tmp_path, "alpha\x00\nbeta\r")
    assert line == "alpha\nbeta"


def test_shift_enter_csi_u_inserts_newline(tmp_path: Path) -> None:
    """Kitty / CSI-u Shift+Enter (``ESC[13;2u``) inserts a newline."""
    line = _drive(tmp_path, "first\x00\x1b[13;2usecond\r")
    assert line == "first\nsecond"


def test_shift_enter_modify_other_keys_inserts_newline(
    tmp_path: Path,
) -> None:
    """xterm modifyOtherKeys Shift+Enter (``ESC[27;2;13~``) inserts ``\\n``.

    Reproduces the user-reported bug: on iTerm2 / macOS Terminal.app /
    the VS Code integrated terminal, Shift+Enter is delivered as the
    modifyOtherKeys escape sequence ``ESC[27;2;13~``.  prompt_toolkit
    pre-maps that sequence to :data:`Keys.ControlM` (plain Enter) inside
    :data:`prompt_toolkit.input.ansi_escape_sequences.ANSI_SEQUENCES`
    *before* any key bindings run, so without the unmap-on-import fix
    in :mod:`cli_prompt` Shift+Enter would submit the line.  The fix
    removes the entry from ANSI_SEQUENCES so the raw sequence reaches
    our tuple key-binding, which inserts a real newline.
    """
    line = _drive(tmp_path, "top\x00\x1b[27;2;13~bottom\r")
    assert line == "top\nbottom"


def test_alt_enter_modify_other_keys_inserts_newline(
    tmp_path: Path,
) -> None:
    """xterm modifyOtherKeys Alt+Enter (``ESC[27;3;13~``) inserts ``\\n``.

    Some terminals (notably iTerm2 with "Report modifiers" turned on,
    or VS Code's integrated terminal in some configurations) deliver
    Option/Alt+Enter as a single combined modifyOtherKeys sequence
    instead of the portable ``ESC \\r`` byte pair, so the binding for
    that sequence must also insert a newline.
    """
    line = _drive(tmp_path, "one\x00\x1b[27;3;13~two\r")
    assert line == "one\ntwo"


def test_alt_enter_csi_u_inserts_newline(tmp_path: Path) -> None:
    """kitty/foot/WezTerm CSI-u Alt+Enter (``ESC[13;3u``) inserts a newline."""
    line = _drive(tmp_path, "a\x00\x1b[13;3ub\r")
    assert line == "a\nb"


def test_ctrl_enter_modify_other_keys_inserts_newline(
    tmp_path: Path,
) -> None:
    """xterm modifyOtherKeys Ctrl+Enter (``ESC[27;5;13~``) inserts ``\\n``.

    Ctrl+Enter is delivered as ``ESC[27;5;13~`` under modifyOtherKeys;
    prompt_toolkit also pre-maps that to :data:`Keys.ControlM`, so the
    same unmap-on-import fix and tuple key-binding apply.
    """
    line = _drive(tmp_path, "x\x00\x1b[27;5;13~y\r")
    assert line == "x\ny"


def test_ctrl_shift_enter_modify_other_keys_inserts_newline(
    tmp_path: Path,
) -> None:
    """modifyOtherKeys Ctrl+Shift+Enter (``ESC[27;6;13~``) inserts ``\\n``."""
    line = _drive(tmp_path, "p\x00\x1b[27;6;13~q\r")
    assert line == "p\nq"


def test_ctrl_enter_csi_u_inserts_newline(tmp_path: Path) -> None:
    """CSI-u Ctrl+Enter (``ESC[13;5u``) inserts a newline."""
    line = _drive(tmp_path, "u\x00\x1b[13;5uv\r")
    assert line == "u\nv"


def test_plain_enter_still_submits_single_line_input(tmp_path: Path) -> None:
    """A plain ``Enter`` on a single line still submits."""
    line = _drive(tmp_path, "just one line\r")
    assert line == "just one line"


def test_three_line_input_via_alt_enter(tmp_path: Path) -> None:
    """Three lines joined by Alt+Enter are returned as one ``\\n``-joined string."""
    line = _drive(
        tmp_path,
        "one\x00\x1b\rtwo\x1b\rthree\r",
    )
    assert line == "one\ntwo\nthree"


def test_plain_enter_submits_multiline_buffer_with_embedded_newlines(
    tmp_path: Path,
) -> None:
    """Plain Enter on a buffer that already contains ``\\n`` still submits.

    After Alt+Enter / Shift+Enter / Ctrl+Enter inserted a real newline
    into the buffer, the next plain ``Enter`` must submit the *whole*
    multi-line buffer (with the embedded ``\\n`` preserved) rather than
    re-inserting another newline.  This guards against an accidental
    "Enter inserts newline" override (e.g. a future
    ``multiline=True`` default change in prompt_toolkit) that would
    leave the user with no way to ever submit a multi-line task.
    """
    line = _drive(tmp_path, "foo\x00\x1b[27;2;13~bar\x1b[27;5;13~baz\r")
    assert line == "foo\nbar\nbaz"


def test_modify_other_keys_enter_at_end_of_buffer_then_submit(
    tmp_path: Path,
) -> None:
    """Modifier+Enter immediately followed by plain ``\\r`` returns ``foo\\n``.

    Catches a subtle regression where the tuple binding for the
    modifyOtherKeys sequence might consume the trailing ``\\r`` as part
    of the same key event (or fail to fire because no follow-up byte
    arrives).  The buffer ends in a trailing newline and is then
    submitted by the bare Enter.
    """
    line = _drive(tmp_path, "foo\x00\x1b[27;5;13~\r")
    assert line == "foo\n"


def test_unmap_enter_aliases_is_idempotent(tmp_path: Path) -> None:
    """Re-running ``_unmap_enter_aliases`` does not crash or change behaviour.

    The unmap fires once at import time; a second call (e.g. via
    :func:`importlib.reload` or an explicit invocation) must be a
    no-op even though the keys are already gone.  The follow-up
    multi-line drive proves the bindings still work after the second
    unmap.
    """
    from prompt_toolkit.input.ansi_escape_sequences import (  # noqa: PLC0415
        ANSI_SEQUENCES,
    )

    from kiss.agents.sorcar.cli_prompt import (  # noqa: PLC0415
        _MODIFY_OTHER_KEYS_ENTER,
        _unmap_enter_aliases,
    )

    _unmap_enter_aliases()
    _unmap_enter_aliases()
    for seq in _MODIFY_OTHER_KEYS_ENTER:
        assert seq not in ANSI_SEQUENCES, seq

    line = _drive(tmp_path, "alpha\x00\x1b[27;2;13~omega\r")
    assert line == "alpha\nomega"


def test_prompt_session_is_multiline(tmp_path: Path) -> None:
    """The PromptSession is constructed with ``multiline=True``.

    Without ``multiline=True`` prompt_toolkit treats Enter as an
    immediate submit and never feeds an inserted ``\\n`` back to the
    buffer, so this flag is the structural pre-condition for the
    multi-line user-flow validated by the pipe-driven tests above.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    assert bool(reader.session.multiline) is True


def test_prompt_session_wraps_long_lines(tmp_path: Path) -> None:
    """The PromptSession is configured with ``wrap_lines=True``.

    Long input must wrap inside the framed panel rather than scroll
    horizontally off-screen — ``wrap_lines=True`` is what makes
    prompt_toolkit render the overflow on the next visual row.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    assert bool(reader.session.wrap_lines) is True


def test_cmd_enter_modify_other_keys_inserts_newline(
    tmp_path: Path,
) -> None:
    """xterm modifyOtherKeys Cmd+Enter (``ESC[27;9;13~``) inserts ``\\n``.

    macOS terminals that report the Meta modifier (iTerm2 with the
    "Report modifiers using CSI u" option, kitty/foot/WezTerm under
    the CSI-u keyboard protocol, the VS Code integrated terminal with
    macOptionAsMeta enabled) deliver Cmd+Enter as the modifyOtherKeys
    sequence ``ESC[27;9;13~`` (mod value 1 + Meta-bit 8 = 9).  The
    binding must insert a real newline.
    """
    line = _drive(tmp_path, "left\x00\x1b[27;9;13~right\r")
    assert line == "left\nright"


def test_cmd_enter_csi_u_inserts_newline(tmp_path: Path) -> None:
    """kitty/CSI-u Cmd+Enter (``ESC[13;9u``) inserts a newline."""
    line = _drive(tmp_path, "head\x00\x1b[13;9utail\r")
    assert line == "head\ntail"


def test_cmd_enter_inserts_newline_when_completion_selected(
    tmp_path: Path,
) -> None:
    """Cmd+Enter with a selected completion inserts ``\\n``, not autocomplete.

    Same regression contract as the Shift+Enter / Alt+Enter / Ctrl+Enter
    counterparts: the highlighted candidate must be discarded
    (originally-typed text restored) and a real newline added.
    """
    project = _make_project_tree(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\x1b[27;9;13~")  # Cmd+Enter (modifyOtherKeys)
            pipe.send_text("end\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("@a")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line == "@a\nend", (
        f"expected '@a\\nend' (no autocomplete), got {line!r}"
    )


def test_shift_enter_inserts_newline_when_completion_selected(
    tmp_path: Path,
) -> None:
    """Shift+Enter (modifyOtherKeys) with a selected completion inserts ``\\n``.

    Reproduces the user-reported bug: when the predictive / file-picker
    dropdown is open *and* a candidate has been highlighted (Down),
    pressing Shift+Enter must **not** accept the highlighted completion
    — it must restore the originally-typed text and insert a real
    newline so the user can continue on the next visual row.

    Without the ``cancel_completion()`` call in the modifier+Enter
    bindings, prompt_toolkit's :class:`Buffer` keeps the highlighted
    completion's text in the buffer; ``insert_text("\\n")`` then
    appends the newline *after* the completion text, so the line
    submitted by the trailing ``Enter`` is the completion + ``\\n`` +
    follow-up — i.e. the autocomplete was applied even though the user
    pressed Shift+Enter.
    """
    project = _make_project_tree(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\x1b[27;2;13~")  # Shift+Enter (modifyOtherKeys)
            pipe.send_text("rest\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        # ``@a`` filters the picker so a candidate is selectable.
        pipe.send_text("@a")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    # The highlighted completion must NOT be applied; the originally
    # typed ``@a`` is preserved and a real newline separates it from
    # the follow-up text.
    assert line == "@a\nrest", (
        f"expected '@a\\nrest' (no autocomplete), got {line!r}"
    )


def test_alt_enter_inserts_newline_when_completion_selected(
    tmp_path: Path,
) -> None:
    """Alt/Option+Enter with a selected completion inserts ``\\n``, not autocomplete.

    Same regression as
    :func:`test_shift_enter_inserts_newline_when_completion_selected`
    but driven by the portable Esc+Enter byte pair that every
    macOS / Linux terminal delivers for Option/Alt+Enter.
    """
    project = _make_project_tree(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\x1b\r")  # Alt+Enter (Esc + CR)
            pipe.send_text("more\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("@a")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line == "@a\nmore", (
        f"expected '@a\\nmore' (no autocomplete), got {line!r}"
    )


def test_ctrl_enter_inserts_newline_when_completion_selected(
    tmp_path: Path,
) -> None:
    """Ctrl+Enter (Command+Enter equivalent) with a selected completion inserts ``\\n``.

    ``Command+Enter`` on macOS is not transmitted by every terminal,
    but iTerm2 / WezTerm / kitty / the VS Code integrated terminal can
    be configured to deliver it (and Ctrl+Enter) as the
    modifyOtherKeys sequence ``ESC[27;5;13~``.  Same regression
    contract: the highlighted completion must not be applied.
    """
    project = _make_project_tree(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\x1b[27;5;13~")  # Ctrl+Enter (modifyOtherKeys)
            pipe.send_text("done\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("@a")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line == "@a\ndone", (
        f"expected '@a\\ndone' (no autocomplete), got {line!r}"
    )


def test_ctrl_j_inserts_newline_when_completion_selected(
    tmp_path: Path,
) -> None:
    """Ctrl+J (LF) with a selected completion inserts ``\\n``, not autocomplete.

    Some terminals (notably macOS Terminal.app) deliver Shift+Enter as
    a literal Ctrl+J (``\\n``); the same "do not accept the highlighted
    completion" guarantee must hold for that path.
    """
    project = _make_project_tree(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\n")  # Ctrl+J / LF
            pipe.send_text("tail\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("@a")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line == "@a\ntail", (
        f"expected '@a\\ntail' (no autocomplete), got {line!r}"
    )


def test_prompt_continuation_keeps_panel_left_border(tmp_path: Path) -> None:
    """Wrapped / continuation visual rows still carry the framed ``│``.

    When the input wraps onto a new visual row (or the user enters a
    multi-line task), prompt_toolkit calls the session's
    ``prompt_continuation`` to render the left margin.  The framed
    input panel paints a cyan ``│`` on the first row, so the
    continuation must paint the same glyph — otherwise the box appears
    broken on every line after the first.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    # The PromptSession stores the continuation callable directly; it
    # is the very same module-level function the production code wires
    # in, so calling it here exercises the same code path prompt_toolkit
    # uses for every wrapped / multi-line visual row.
    assert reader.session.prompt_continuation is _prompt_continuation
    rendered = _prompt_continuation(80, 1, 0)
    assert isinstance(rendered, ANSI)
    text = "".join(t[1] for t in to_formatted_text(rendered))
    assert "│" in text, (
        f"expected the framed │ glyph in the continuation, got {text!r}"
    )
