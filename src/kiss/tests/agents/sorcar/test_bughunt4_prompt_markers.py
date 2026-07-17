# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ANSI colour codes in the readline prompt break editing.

Bug: ``cli_repl._read_line`` passed a prompt containing raw ANSI colour
sequences (``ESC[36m…ESC[0m``) to ``input()`` while GNU readline was
active.  Readline counts every prompt byte as a printed column unless
non-printing runs are bracketed with ``\\x01``/``\\x02``
(``RL_PROMPT_START_IGNORE`` / ``END_IGNORE``), so it believed the
4-column prompt was ~26 columns wide.  On a narrow terminal the line
then redraws/horizontal-scrolls after the *second* typed character,
garbling the input panel (measured empirically: first spurious redraw
at char 2 without markers vs char 20 with markers at 30 columns).

The test types ten characters one by one into the real ``_read_line``
on a 30-column PTY and asserts the echo stays clean (no redraws).
"""

from __future__ import annotations

import fcntl
import os
import select
import struct
import sys
import termios
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)


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


def test_typing_into_idle_prompt_does_not_redraw_on_narrow_terminal(
    tmp_path: Path,
) -> None:
    """Ten chars at 30 columns must echo cleanly (prompt width ~4)."""
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

    kiss_home = tmp_path / ".kisshome"
    child_code = f"""
import os
os.environ["KISS_HOME"] = {str(kiss_home)!r}
os.environ.pop("COLUMNS", None)
os.environ.pop("LINES", None)
os.environ["LC_ALL"] = "en_US.UTF-8"
import sys
from kiss.ui.cli.cli_repl import _PROMPT, _read_line

line = _read_line(_PROMPT)
sys.stdout.write("\\nGOT[" + repr(line) + "]\\n")
sys.stdout.flush()
"""

    pid, fd = pty_spawn([sys.executable, "-c", child_code])

    try:
        # 24 rows x 30 cols, so a miscounted ~26-column prompt wraps
        # almost immediately while a correct 4-column prompt never does.
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 30, 0, 0))
        deadline = time.time() + 30.0
        out = b""
        while time.time() < deadline and "›" not in out.decode("utf-8", "ignore"):
            out += _drain(fd, 0.2)
        assert "›" in out.decode("utf-8", "ignore"), (
            f"prompt never appeared; output: {out!r}"
        )
        _drain(fd, 0.5)  # settle: readline finishes prompt bookkeeping

        typed = b""
        for ch in b"abcdefghij":
            os.write(fd, bytes([ch]))
            typed += _drain(fd, 0.12)
        echo = typed.decode("utf-8", "ignore")
        assert echo == "abcdefghij", (
            "readline redrew/scrolled the line while typing 10 chars on a "
            "30-column terminal — the prompt's ANSI codes are counted as "
            f"printed columns (echo bytes: {echo!r})"
        )

        os.write(fd, b"\r")
        tail = _drain(fd, 2.0).decode("utf-8", "ignore")
        assert "GOT['abcdefghij']" in tail, (
            f"submitted line was corrupted; child output: {tail!r}"
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
