# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (F1.1): a shell that closed its pipe at the deadline edge is not a timeout.

After pipe EOF, ``_bash_streaming`` waited for the shell's exit for
exactly the time left before the caller's deadline: for a shell whose
output ended right at that edge the wait was ``process.wait(timeout≈0)``,
one ``waitpid(WNOHANG)`` while the shell was still in its last
microseconds of exiting, so a command that completed was reported as
"Command execution timeout" and its process group SIGKILLed, which
takes down a ``nohup job &`` the command deliberately left behind.
The wait now has a short floor past the deadline.

The window cannot be hit on purpose, so the test uses its deterministic
stand-in: a shell that closes its stdout at once (EOF long before the
deadline) and exits 0.2 s after the deadline but inside the floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")


def test_shell_exiting_just_past_the_deadline_after_eof_is_not_a_timeout(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    code, out = tools._bash_streaming(
        "echo started; exec >/dev/null 2>&1; sleep 0.3; touch done",
        timeout_seconds=0.1,
        cwd=str(tmp_path),
    )
    assert out == "started\n"
    assert code == 0
    assert (tmp_path / "done").exists()


def test_shell_running_long_past_the_deadline_after_eof_is_a_timeout(tmp_path: Path) -> None:
    """The floor is short: a shell still running well after the deadline times out."""
    tools = UsefulTools(work_dir=str(tmp_path))
    code, out = tools._bash_streaming(
        "echo started; exec >/dev/null 2>&1; sleep 30; touch done",
        timeout_seconds=0.1,
        cwd=str(tmp_path),
    )
    assert out == "started\n"
    assert code is None
    assert not (tmp_path / "done").exists()
