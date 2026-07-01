# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Realistic reproduction: bare-CR multi-line on macOS Terminal.app.

macOS Terminal.app supports neither xterm's ``modifyOtherKeys`` mode
(``ESC[27;<mod>;13~``) nor the Kitty keyboard protocol
(``ESC[13;<mod>u``), so Shift+Enter is delivered as a bare ``\\r`` —
byte-identical to plain Enter.  Before the universal backslash
line-continuation fix, this meant a user typing a multi-line task on
Terminal.app saw every line submitted as its own task the moment
Shift+Enter was pressed.  The pipe-based tests in
:mod:`test_cli_multiline_input` cannot reproduce this: they feed
modifyOtherKeys / CSI-u bytes directly into the parser, which
Terminal.app never emits.

These tests drive the REAL production code paths — both
:class:`~kiss.agents.sorcar.cli_prompt.PtkLineReader` and
:class:`~kiss.agents.sorcar.cli_steering._InputBox` — through a fresh
forkpty (via :func:`pty_spawn` which internally uses
:func:`pty.openpty` + :class:`subprocess.Popen`), so
``sys.stdin.isatty()`` / ``sys.stdout.isatty()`` return ``True`` in
the child.  The parent sends BARE ``\\r`` bytes between the lines —
exactly what Terminal.app delivers — and asserts that the universal
backslash line-continuation fix combines the lines into a single
submitted task.
"""

from __future__ import annotations

import os
import select
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kiss.tests.agents.sorcar._pty_helper import pty_spawn

_READER_CHILD = textwrap.dedent(
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
    except BaseException as exc:
        out_path.write_text("ERROR:" + repr(exc))
        raise
    out_path.write_text(line)
    sys.exit(0)
    """,
)

_STEERING_CHILD = textwrap.dedent(
    """
    import io
    import os
    import sys
    import threading
    from pathlib import Path

    from kiss.agents.sorcar.cli_steering import _InputBox

    out_path = Path(os.environ["KISS_TEST_OUT"])
    box = _InputBox(threading.RLock(), sys.stdout)
    submitted = []
    aborted = threading.Event()

    def on_submit(line):
        submitted.append(line)
        # Persist after every submit so the parent can observe how
        # many separate tasks the box produced (buggy behaviour =
        # multiple entries; fixed behaviour = one entry).
        out_path.write_text("\\x1e".join(submitted))
        if len(submitted) >= 1:
            # A single submit ends this test child so the parent can
            # move on to assertions; the fixed code will produce
            # exactly one submit for the whole multi-line input.
            aborted.set()

    def on_abort():
        aborted.set()

    box.start()
    try:
        while not aborted.is_set():
            data = os.read(0, 4096)
            if not data:
                break
            box.feed(data, on_submit, on_abort)
    finally:
        box.stop()
    sys.exit(0)
    """,
)


def _make_env(tmp_path: Path) -> dict[str, str]:
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
    for key in ("VIRTUAL_ENV", "PYTHONPATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _drain_master(fd: int, seconds: float) -> None:
    """Drain output on the master fd for *seconds* to prevent PTY blocking."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return


def _safe_write(fd: int, data: bytes) -> None:
    try:
        os.write(fd, data)
    except OSError:
        pass


def _wait_for_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return
    os.waitpid(pid, 0)


def _run_reader_child(
    tmp_path: Path,
    keystrokes_before_first_cr: bytes,
    inter_cr_bytes: bytes = b"line two\r",
    *,
    init_delay: float = 1.5,
    exit_timeout: float = 15.0,
) -> str:
    """Spawn the reader on a real PTY and send *bytes* + bare CR + more."""
    env = _make_env(tmp_path)
    out_path = Path(env["KISS_TEST_OUT"])
    pid, fd = pty_spawn(
        [sys.executable, "-c", _READER_CHILD],
        env=env,
    )
    try:
        _drain_master(fd, init_delay)
        _safe_write(fd, keystrokes_before_first_cr)
        _drain_master(fd, 0.1)
        _safe_write(fd, b"\r")
        _drain_master(fd, 0.1)
        _safe_write(fd, inter_cr_bytes)
        _drain_master(fd, 0.1)
        _wait_for_exit(pid, exit_timeout)
    finally:
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
    return out_path.read_text()


def _run_steering_child(
    tmp_path: Path,
    keystrokes: bytes,
    *,
    init_delay: float = 1.0,
    exit_timeout: float = 15.0,
) -> list[str]:
    """Spawn ``_InputBox`` on a real PTY and feed *keystrokes*.

    Returns the ordered list of on_submit lines the box produced.
    """
    env = _make_env(tmp_path)
    out_path = Path(env["KISS_TEST_OUT"])
    pid, fd = pty_spawn(
        [sys.executable, "-c", _STEERING_CHILD],
        env=env,
    )
    try:
        _drain_master(fd, init_delay)
        _safe_write(fd, keystrokes)
        _drain_master(fd, 0.5)
        _wait_for_exit(pid, exit_timeout)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    if not out_path.exists():
        return []
    text = out_path.read_text()
    return text.split("\x1e")


# ---------------------------------------------------------------------------
# Reproduction: current (pre-fix) behaviour on Terminal.app
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_reader_backslash_continuation_combines_lines_bare_cr(
    tmp_path: Path,
) -> None:
    """PtkLineReader: ``line one \\`` + CR + ``line two`` + CR → one submit.

    On macOS Terminal.app Shift+Enter and Enter both emit bare ``\\r``.
    The universal backslash line-continuation fix means ending the
    first line with ``\\`` makes the first CR insert a newline instead
    of submitting, so both lines land in the same returned string.
    Before the fix the first CR submitted ``line one \\`` immediately
    and the reader exited with only ``"line one \\"`` in the file.
    """
    result = _run_reader_child(
        tmp_path,
        keystrokes_before_first_cr=b"line one \\",
        inter_cr_bytes=b"line two\r",
    )
    assert result == "line one \nline two", (
        f"expected 'line one \\nline two', got {result!r}"
    )


@pytest.mark.timeout(60)
def test_steering_backslash_continuation_combines_lines_bare_cr(
    tmp_path: Path,
) -> None:
    """_InputBox: bare-CR multi-line via ``\\`` continuation → one submit.

    Reproduces the mid-task steering path.  On macOS Terminal.app the
    inter-line ``\\r`` is byte-identical to the terminating ``\\r``;
    the fix relies on the backslash continuation to distinguish them.
    """
    lines = _run_steering_child(
        tmp_path,
        b"line one \\\rline two\r",
    )
    assert lines == ["line one \nline two"], (
        f"expected one submit with combined lines, got {lines!r}"
    )


@pytest.mark.timeout(60)
def test_reader_plain_enter_without_continuation_still_submits_first_line(
    tmp_path: Path,
) -> None:
    """Regression: without ``\\``, a bare CR still submits immediately.

    This mirrors the current (Terminal.app) behaviour we CANNOT fix:
    without an opt-in continuation marker, the CR has to submit.
    Documents the intended fallback so a future refactor can't
    accidentally change it.
    """
    result = _run_reader_child(
        tmp_path,
        keystrokes_before_first_cr=b"line one",
        # ``line two\r`` after the first CR is ignored by the exited
        # child; we only care that the reader submitted the first
        # line and exited.
        inter_cr_bytes=b"",
    )
    assert result == "line one", (
        f"expected 'line one' (plain Enter submits), got {result!r}"
    )
