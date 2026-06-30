# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-terminal (PTY) tests for modifier+Enter newline behaviour.

The pipe-based tests in :mod:`test_cli_multiline_input` validate the
key-binding plumbing, but they cannot detect the actual user-visible
bug: on iTerm2, macOS Terminal.app, and the VS Code integrated
terminal, Shift+Enter (and every other modifier+Enter combination) is
delivered as a bare ``\\r`` — *identical* to plain Enter — unless the
application explicitly opts in to xterm's ``modifyOtherKeys`` protocol
by writing ``\\x1b[>4;2m`` to the terminal at session start.  Without
that opt-in the terminal never emits the
:data:`prompt_toolkit.input.ansi_escape_sequences.ANSI_SEQUENCES`
sequence ``\\x1b[27;2;13~`` for which our key bindings are registered,
so every Shift/Alt/Ctrl+Enter submits the buffer.  The pipe tests miss
this because they feed the modifyOtherKeys bytes directly into the
parser; the terminal layer is never exercised.

These tests drive a real :class:`PromptSession` through a fresh PTY
pair (master fd in the test process, slave fd as the child process's
stdin/stdout/stderr) and emulate the byte-level behaviour of a stock
modern terminal emulator:

* Plain Enter is delivered as ``\\r``.
* Shift+Enter is delivered as ``\\r`` *unless* the application has
  written ``\\x1b[>4;1m`` or ``\\x1b[>4;2m`` to the terminal; in that
  case the emulator instead delivers ``\\x1b[27;2;13~``.

This is the same disambiguation rule iTerm2 / xterm.js / xterm itself
use, so a Sorcar build that fails these tests is the same build that
breaks Shift+Enter for real users.
"""

from __future__ import annotations

import os
import re
import select
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.tests.agents.sorcar._pty_helper import pty_spawn

_CHILD_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    from pathlib import Path

    from kiss.agents.sorcar.cli_prompt import PtkLineReader
    from kiss.agents.sorcar.cli_repl import CliCompleter

    out_path = Path(os.environ["KISS_TEST_OUT"])
    hist_path = Path(os.environ["KISS_TEST_HIST"])
    completer = CliCompleter(os.environ["KISS_TEST_WORKDIR"])
    reader = PtkLineReader(completer, hist_path)
    try:
        line = reader.read("> ")
    except BaseException as exc:  # pragma: no cover - debugging aid
        out_path.write_text("ERROR:" + repr(exc))
        raise
    out_path.write_text(line)
    sys.exit(0)
    """,
)

# xterm modifyOtherKeys "enable" sequences.  Level 1 (``\\x1b[>4;1m``)
# turns modifyOtherKeys on for keys that have no representation in the
# usual ASCII table; level 2 (``\\x1b[>4;2m``) is the more aggressive
# variant that also rewrites keys that *do* have a representation, so
# Shift+Enter goes from a bare ``\\r`` to ``\\x1b[27;2;13~``.  Either
# value is acceptable to the test emulator; level 2 is what we want
# the production code to send.
_MOK_ENABLE_RE = re.compile(rb"\x1b\[>4;[12]m")
_MOK_DISABLE_RE = re.compile(rb"\x1b\[>4;0m")


class _TerminalEmulator(threading.Thread):
    """Background thread that drains the master PTY fd and tracks state.

    Behaves like the input side of a stock VT100-compatible terminal:

    * Drains everything the child writes to the slave, so the slave's
      output buffer never blocks the child.
    * Watches the byte stream for the xterm ``modifyOtherKeys`` enable
      sequence; :attr:`enabled` flips ``True`` as soon as either level
      1 or level 2 of the sequence is observed and stays ``True`` until
      the matching disable sequence (``\\x1b[>4;0m``) is observed.
    """

    def __init__(self, master_fd: int) -> None:
        super().__init__(daemon=True)
        self.master_fd = master_fd
        self._lock = threading.Lock()
        self._enabled = False
        self._stop = threading.Event()
        self._buffer = b""

    def enabled(self) -> bool:
        """Return ``True`` iff modifyOtherKeys is currently enabled."""
        with self._lock:
            return self._enabled

    def stop(self) -> None:
        """Ask the thread to exit at the next poll boundary."""
        self._stop.set()

    def run(self) -> None:
        """Read-and-discard loop with modifyOtherKeys state tracking."""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self.master_fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                chunk = os.read(self.master_fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            self._scan(chunk)

    def _scan(self, chunk: bytes) -> None:
        """Update the modifyOtherKeys flag from *chunk*."""
        # Keep the trailing 16 bytes from the previous chunk to catch
        # an enable sequence that was split across two reads.
        with self._lock:
            data = self._buffer + chunk
            self._buffer = data[-16:]
            if _MOK_ENABLE_RE.search(data):
                self._enabled = True
            if _MOK_DISABLE_RE.search(data):
                self._enabled = False


def _make_env(tmp_path: Path) -> dict[str, str]:
    """Build a minimal child environment for the PTY-spawned reader.

    ``PROMPT_TOOLKIT_NO_CPR=1`` disables prompt_toolkit's "ask for
    cursor position" probe so the test does not need to fabricate a
    cursor-position-report response on the master fd.  ``TERM`` is set
    to a vt100-compatible value so prompt_toolkit picks the
    :class:`Vt100_Output` path (which is the only one that calls
    :func:`enable_extended_keys` if the production code wires it up).
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ["HOME"],
        "TERM": "xterm-256color",
        "PROMPT_TOOLKIT_NO_CPR": "1",
        "KISS_TEST_OUT": str(tmp_path / "result"),
        "KISS_TEST_HIST": str(tmp_path / "h"),
        "KISS_TEST_WORKDIR": str(tmp_path),
        "PYTHONUNBUFFERED": "1",
    }
    # Inherit the venv-relative entries that ``uv run`` injects so the
    # child can ``import prompt_toolkit`` / ``import kiss`` from the
    # same interpreter as the parent.
    for key in ("VIRTUAL_ENV", "PYTHONPATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _drive_real_pty(
    tmp_path: Path,
    *,
    pre_enter_bytes: bytes,
    shift_enter_unmodified: bytes = b"\r",
    shift_enter_modify_other_keys: bytes = b"\x1b[27;2;13~",
    post_enter_bytes: bytes = b"world\r",
    init_delay: float = 1.5,
    exit_timeout: float = 15.0,
) -> tuple[str, bool]:
    """Spawn the child reader on a fresh PTY and send realistic keystrokes.

    The terminal emulator runs in a background thread so the child's
    stdout never blocks; after *init_delay* seconds (long enough for
    prompt_toolkit to draw the prompt and finish all setup writes) the
    parent writes *pre_enter_bytes*, then a Shift+Enter encoded
    according to the *current* modifyOtherKeys state (i.e. exactly
    what a real terminal would do), then *post_enter_bytes* to submit.

    Returns:
        A ``(captured_line, modify_other_keys_seen)`` tuple where the
        line is what the child wrote to ``KISS_TEST_OUT`` and the bool
        records whether the emulator observed the application enable
        modifyOtherKeys at any point during the session.
    """
    env = _make_env(tmp_path)
    out_path = Path(env["KISS_TEST_OUT"])
    pid, fd = pty_spawn(
        [sys.executable, "-c", _CHILD_SCRIPT],
        env=env,
    )
    emulator = _TerminalEmulator(fd)
    emulator.start()
    try:
        time.sleep(init_delay)
        _safe_write(fd, pre_enter_bytes)
        time.sleep(0.05)
        shift_enter = (
            shift_enter_modify_other_keys
            if emulator.enabled()
            else shift_enter_unmodified
        )
        # Snapshot the modifyOtherKeys flag *before* sending the
        # potentially-submitting byte: the child may exit between the
        # write and the post-Enter write, closing the master fd and
        # losing the flag in subsequent OSError handling.
        final_enabled = emulator.enabled()
        _safe_write(fd, shift_enter)
        time.sleep(0.05)
        _safe_write(fd, post_enter_bytes)
        _wait_for_exit(pid, exit_timeout)
    finally:
        emulator.stop()
        emulator.join(timeout=2.0)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    if not out_path.exists():
        raise AssertionError(
            "child did not write KISS_TEST_OUT; the reader hung or crashed",
        )
    return out_path.read_text(), final_enabled


def _safe_write(fd: int, data: bytes) -> None:
    """Write *data* to *fd*, ignoring ``EIO`` from a child that already exited.

    When the bug under test is present, the child submits the buffer
    on the first bare ``\\r`` and exits immediately; subsequent writes
    to the master fd then raise ``OSError(EIO)``.  Swallowing that
    OSError lets the test continue to the assertions on the captured
    line so the bug is reported as a content mismatch (much more
    informative than a generic ``OSError``).
    """
    try:
        os.write(fd, data)
    except OSError:
        pass


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Poll-wait for *pid*; kill on timeout to keep the test bounded."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return True
    os.waitpid(pid, 0)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_real_terminal_enables_modify_other_keys_at_session_start(
    tmp_path: Path,
) -> None:
    """The reader writes ``\\x1b[>4;1m`` or ``\\x1b[>4;2m`` to the terminal.

    Without that opt-in real terminals (iTerm2, macOS Terminal.app,
    VS Code integrated terminal) deliver Shift/Alt/Ctrl+Enter as a
    bare ``\\r`` and the line submits.  This test asserts the
    application sends the enable sequence so the terminal switches
    into modifyOtherKeys mode and starts disambiguating Shift+Enter
    from plain Enter.
    """
    _, mok_seen = _drive_real_pty(
        tmp_path,
        pre_enter_bytes=b"hello",
        post_enter_bytes=b"world\r",
    )
    assert mok_seen, (
        "PtkLineReader did not enable xterm modifyOtherKeys at session "
        "start; Shift+Enter is indistinguishable from plain Enter on "
        "real terminals without it.  See the comment block at the top "
        "of cli_prompt.py for the required `\\x1b[>4;2m` write."
    )


@pytest.mark.timeout(60)
def test_real_terminal_shift_enter_inserts_newline(tmp_path: Path) -> None:
    """Shift+Enter on a real PTY inserts ``\\n`` and Enter submits.

    Reproduces the user-reported bug end-to-end: with the
    ``modifyOtherKeys`` opt-in in place the (simulated) terminal
    delivers ``\\x1b[27;2;13~`` for Shift+Enter, which our key
    binding turns into a real newline; without it the terminal would
    fall back to ``\\r`` and the line would submit as ``"hello"``
    only.
    """
    line, mok_seen = _drive_real_pty(
        tmp_path,
        pre_enter_bytes=b"hello",
        post_enter_bytes=b"world\r",
    )
    assert mok_seen, "modifyOtherKeys must be enabled for Shift+Enter to work"
    assert line == "hello\nworld", (
        f"expected 'hello\\nworld', got {line!r} (mok_seen={mok_seen})"
    )


@pytest.mark.timeout(60)
def test_real_terminal_alt_enter_inserts_newline(tmp_path: Path) -> None:
    """Alt+Enter on a real PTY inserts ``\\n`` and Enter submits.

    Alt+Enter is delivered as ``\\x1b\\r`` (Esc+CR) on every modern
    terminal that has "Option as Meta" turned on (the default in
    iTerm2 and VS Code's integrated terminal); this test uses that
    encoding so it does *not* depend on modifyOtherKeys for the
    Alt+Enter path, isolating the regression from the Shift+Enter one.
    """
    line, _ = _drive_real_pty(
        tmp_path,
        pre_enter_bytes=b"hello",
        # Force the Esc+CR encoding for Alt+Enter regardless of
        # whether modifyOtherKeys ended up enabled.
        shift_enter_unmodified=b"\x1b\r",
        shift_enter_modify_other_keys=b"\x1b\r",
        post_enter_bytes=b"world\r",
    )
    assert line == "hello\nworld", line


@pytest.mark.timeout(60)
def test_real_terminal_plain_enter_submits(tmp_path: Path) -> None:
    """A plain ``\\r`` on a real PTY submits the typed line as-is.

    Guards against an over-eager fix that turned *all* incoming ``\\r``
    into newlines: plain Enter must still terminate input.
    """
    env = _make_env(tmp_path)
    out_path = Path(env["KISS_TEST_OUT"])
    pid, fd = pty_spawn(
        [sys.executable, "-c", _CHILD_SCRIPT],
        env=env,
    )
    emulator = _TerminalEmulator(fd)
    emulator.start()
    try:
        time.sleep(1.5)
        os.write(fd, b"just one line\r")
        _wait_for_exit(pid, 15.0)
    finally:
        emulator.stop()
        emulator.join(timeout=2.0)
        try:
            os.close(fd)
        except OSError:
            pass
    assert out_path.exists(), "child did not write KISS_TEST_OUT"
    assert out_path.read_text() == "just one line"
