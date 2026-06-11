# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: Ctrl+C at the idle prompt garbles the panel's bottom rule.

Bug: ``cli_repl._read_line`` pre-draws the closed input panel (top
border, framed body row, bottom rule) and parks the cursor back on the
body row for ``input()``.  When the user presses Ctrl+C the
``KeyboardInterrupt`` escapes ``input()`` with the cursor still on the
body row, and ``run_repl``'s handler does ``print("\\n(Press Ctrl+C
again or type /exit to quit)")`` — the newline moves the cursor onto
the *bottom-rule row* and the message is printed over the border
without erasing it, leaving a garbled
``(Press Ctrl+C again or type /exit to quit)────────────╯`` row on
screen.  The ``Goodbye.`` message of a second Ctrl+C (and of ``exit``)
overprints the rule of the *next* panel the same way.

The test drives the real ``run_repl`` end to end on a PTY, renders the
child's byte stream through a small VT100 screen model, and asserts the
rows carrying the interrupt/goodbye messages contain no border-rule
remnants.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)

_CSI_RE = re.compile(r"^\x1b\[([0-9;?]*)([ -/]*)([@-~])")


class _Screen:
    """A tiny VT100 model: enough to replay the idle REPL's output."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = [[]]
        self.row = 0
        self.col = 0

    def _cell(self) -> list[str]:
        while self.row >= len(self.rows):
            self.rows.append([])
        line = self.rows[self.row]
        while self.col >= len(line):
            line.append(" ")
        return line

    def feed(self, text: str) -> None:
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                m = _CSI_RE.match(text[i:])
                if m:
                    self._csi(m.group(1), m.group(3))
                    i += m.end()
                    continue
                i += 2  # ESC + one char (ESC7/ESC8/…): ignore
                continue
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self.row += 1
                self.col = 0
            elif ch == "\b":
                self.col = max(self.col - 1, 0)
            elif ch == "\x07":
                pass
            else:
                line = self._cell()
                line[self.col] = ch
                self.col += 1
            i += 1

    def _csi(self, params: str, final: str) -> None:
        nums = [int(p) for p in params.split(";") if p.isdigit()]
        n = nums[0] if nums else 0
        if final == "K":
            line = self._cell()
            if n == 2:
                self.rows[self.row] = []
            else:
                del line[self.col :]
        elif final == "A":
            self.row = max(self.row - max(n, 1), 0)
        elif final == "B":
            self.row += max(n, 1)
        elif final == "C":
            self.col += max(n, 1)
        elif final == "D":
            self.col = max(self.col - max(n, 1), 0)
        elif final == "G":
            self.col = max(n, 1) - 1

    def lines(self) -> list[str]:
        return ["".join(line) for line in self.rows]


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


def test_ctrl_c_message_does_not_overprint_bottom_border(
    tmp_path: Path,
) -> None:
    """The interrupt/goodbye rows must not carry border remnants."""
    import pty

    kiss_home = tmp_path / ".kisshome"
    work_dir = tmp_path / "wd"
    work_dir.mkdir()
    child_code = f"""
import os
os.environ["KISS_HOME"] = {str(kiss_home)!r}
os.environ["COLUMNS"] = "80"
os.environ["LINES"] = "24"
os.environ["LC_ALL"] = "en_US.UTF-8"
import sys


class StubAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        return "summary: ok\\n"


from kiss.agents.sorcar.cli_repl import run_repl

run_repl(StubAgent(), {{"work_dir": {str(work_dir)!r},
                        "model_name": "demo-model", "verbose": True}})
sys.stdout.write("REPL_DONE\\n")
sys.stdout.flush()
"""

    pid, fd = pty.fork()
    if pid == 0:  # child: fresh interpreter attached to the PTY slave
        os.execvp(sys.executable, [sys.executable, "-c", child_code])
        os._exit(0)  # pragma: no cover - exec never returns

    try:
        deadline = time.time() + 30.0
        out = b""
        while time.time() < deadline and "›" not in out.decode("utf-8", "ignore"):
            out += _drain(fd, 0.2)
        assert "›" in out.decode("utf-8", "ignore"), (
            f"idle prompt never appeared; output: {out!r}"
        )
        out += _drain(fd, 0.5)

        os.write(fd, b"\x03")  # first Ctrl+C: arm the exit confirmation
        deadline = time.time() + 10.0
        while time.time() < deadline and b"(Press Ctrl+C" not in out:
            out += _drain(fd, 0.2)
        assert b"(Press Ctrl+C" in out, (
            f"interrupt message never appeared; output: {out!r}"
        )
        out += _drain(fd, 0.5)

        os.write(fd, b"\x03")  # second Ctrl+C: exit
        deadline = time.time() + 10.0
        while time.time() < deadline and b"REPL_DONE" not in out:
            out += _drain(fd, 0.2)

        screen = _Screen()
        screen.feed(out.decode("utf-8", "ignore"))
        lines = screen.lines()

        press_rows = [ln for ln in lines if "(Press Ctrl+C" in ln]
        assert press_rows, f"interrupt message not on screen: {lines!r}"
        for row in press_rows:
            assert "╯" not in row and "─" not in row, (
                "the Ctrl+C message was printed over the panel's bottom "
                f"rule, leaving border remnants on screen: {row!r}"
            )

        bye_rows = [ln for ln in lines if "Goodbye." in ln]
        assert bye_rows, f"goodbye message not on screen: {lines!r}"
        for row in bye_rows:
            assert "╯" not in row and "─" not in row, (
                "the Goodbye. message was printed over the panel's bottom "
                f"rule, leaving border remnants on screen: {row!r}"
            )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
