# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8: fallback ``_read_line`` paths disagree on backslash rules.

``cli_repl._read_line_ptk`` and the steering box both follow the shared
POSIX-shell continuation rule in
:func:`kiss.ui.cli.cli_line_continuation.ends_with_line_continuation`:
an *even* number of trailing backslashes is an escaped literal ``\\``
and must submit, and trailing spaces/tabs after a lone ``\\`` still
count as a continuation.

The two fallback paths inside ``cli_repl._read_line`` (the non-TTY
``input()`` loop and the interactive readline loop) instead used a
naive ``line.endswith("\\")`` check.  Consequences:

* a line ending in ``\\\\`` (escaped literal backslash) wrongly lost
  one backslash and swallowed the *next* input line as a bogus
  continuation;
* a line ending in ``\\ `` (backslash + trailing space) wrongly
  submitted instead of continuing.

These end-to-end tests drive the real ``_read_line`` in a child
process — via a plain pipe for the non-TTY path and via a real PTY for
the interactive readline path — and assert both fallback paths agree
with the shared rule.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)

_CHILD_CODE = """
import os, sys
os.environ["KISS_HOME"] = {kiss_home!r}
os.environ["LC_ALL"] = "en_US.UTF-8"
from kiss.ui.cli.cli_repl import _read_line

a = _read_line("> ")
b = _read_line("> ")
sys.stdout.write("\\nA<<" + repr(a) + ">>\\n")
sys.stdout.write("B<<" + repr(b) + ">>\\n")
sys.stdout.flush()
"""


def _run_non_tty(tmp_path: Path, stdin_text: str) -> str:
    """Run two ``_read_line`` calls in a child with piped (non-TTY) stdio."""
    code = _CHILD_CODE.format(kiss_home=str(tmp_path / ".kisshome"))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"child failed: {proc.stderr!r}"
    return proc.stdout


def test_non_tty_escaped_trailing_backslash_submits(tmp_path: Path) -> None:
    """``foo\\\\`` (even count) submits verbatim; next line stays queued."""
    out = _run_non_tty(tmp_path, "foo\\\\\nNEXT\n")
    assert f"A<<{'foo' + chr(92) * 2!r}>>" in out, (
        "a line ending in an escaped literal backslash pair must submit "
        f"unchanged, matching _read_line_ptk; child stdout: {out!r}"
    )
    assert f"B<<{'NEXT'!r}>>" in out, (
        "the following line must NOT be swallowed as a bogus "
        f"continuation; child stdout: {out!r}"
    )


def test_non_tty_backslash_then_space_continues(tmp_path: Path) -> None:
    """``foo \\ `` (trailing space after the marker) still continues."""
    out = _run_non_tty(tmp_path, "foo \\ \nsecond\nTHIRD\n")
    assert f"A<<{'foo ' + chr(10) + 'second'!r}>>" in out, (
        "trailing spaces after a lone backslash must still count as a "
        f"continuation, matching the shared rule; child stdout: {out!r}"
    )
    assert f"B<<{'THIRD'!r}>>" in out, f"child stdout: {out!r}"


def _drain(fd: int, seconds: float) -> bytes:
    out = b""
    deadline = time.time() + seconds
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    return out


def test_interactive_escaped_trailing_backslash_submits(tmp_path: Path) -> None:
    """Readline path: ``foo\\\\`` + Enter submits, next line is separate."""
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

    code = _CHILD_CODE.format(kiss_home=str(tmp_path / ".kisshome"))
    pid, fd = pty_spawn([sys.executable, "-c", code])
    try:
        deadline = time.time() + 30.0
        out = b""
        while time.time() < deadline and "> " not in out.decode("utf-8", "ignore"):
            out += _drain(fd, 0.2)
        assert "> " in out.decode("utf-8", "ignore"), (
            f"prompt never appeared; output: {out!r}"
        )
        _drain(fd, 0.5)
        os.write(fd, b"foo\\\\\r")
        _drain(fd, 0.5)
        os.write(fd, b"NEXT\r")
        tail = _drain(fd, 0.5)
        # Feed one spare line so the buggy code (which swallows NEXT as
        # a continuation and then blocks in the second _read_line) still
        # terminates and reveals its wrong answers.  With correct code
        # the child has already exited, so the write may fail with EIO —
        # that is fine, the answers are already in the drained output.
        try:
            os.write(fd, b"SPARE\r")
        except OSError:
            pass
        deadline = time.time() + 10.0
        while time.time() < deadline and b"B<<" not in tail:
            tail += _drain(fd, 0.2)
        text = tail.decode("utf-8", "ignore")
        assert f"A<<{'foo' + chr(92) * 2!r}>>" in text, (
            "interactive readline path must treat an escaped literal "
            "backslash pair as a submission, matching _read_line_ptk; "
            f"child output: {text!r}"
        )
        assert f"B<<{'NEXT'!r}>>" in text, (
            f"NEXT was swallowed as a bogus continuation; output: {text!r}"
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
