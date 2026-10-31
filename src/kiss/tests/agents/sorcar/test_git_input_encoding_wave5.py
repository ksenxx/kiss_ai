# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 3, P2_sorcar item 2: ``_git`` prepares its stdin before git starts.

``git_worktree._git`` runs git in bytes mode and encodes the ``input``
text itself (UTF-8, surrogate escapes).  The encoding ran after the child
was spawned and before the block that stops the child on any exception,
so a string that cannot be encoded (a lone surrogate from a JSON-decoded
model reply, say) or a Stop landing there left git running with its
stdin pipe open: ``git commit -F -`` (``GitWorktreeOps._git_commit``)
waited for its message forever after the caller had already rolled back.
Real git; the Stop is injected into the calling thread the way the task
runner does, from a ``sys.settrace`` hook on the real ``_git`` frame.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.git_worktree import _git
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.tests.server.conftest import init_repo

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="reads the process table from /proc"
)

UNENCODABLE = "bad\ud800message"  # a lone high surrogate: no UTF-8 form, not an escape


def _live_gits(marker: Path) -> list[int]:
    """Pids of running (not zombie) processes whose command line names *marker*."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = Path(f"/proc/{entry}/cmdline").read_bytes()
            state = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue  # exited while being read
        if os.fsencode(str(marker)) in cmdline and state != "Z":
            found.append(int(entry))
    return found


def _kill_all(pids: list[int]) -> None:
    """Kill what a broken ``_git`` left behind, so a failing run leaves no stray git."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class _StopOncePrepared:
    """``sys.settrace`` hook: Stop the thread at ``_git``'s first line after ``given`` is bound.

    ``given`` holds the encoded stdin, so the Stop lands right after the
    input was prepared, wherever that happens relative to the spawn.
    """

    def __init__(self) -> None:
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> _StopOncePrepared | None:
        if frame.f_code is not _git.__code__:
            return None
        if event == "line" and not self.fired and "given" in frame.f_locals:
            self.fired = True
            inject_async_exception(threading.get_ident(), ToolCallInterrupted)
        return self


def _repo(tmp_path: Path) -> Path:
    """A fresh repository; its path is in the command line of every git ``_git`` runs there."""
    repo = tmp_path / "repo"
    init_repo(repo)
    return repo


def test_valid_input_still_reaches_git(tmp_path: Path) -> None:
    """The positive control: the same command with encodable text hashes it."""
    repo = _repo(tmp_path)
    result = _git("hash-object", "--stdin", cwd=repo, input="hello\n")
    assert result.returncode == 0
    assert result.stdout.strip() == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert _live_gits(repo) == []


def test_unencodable_input_starts_no_git(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    try:
        with pytest.raises(UnicodeEncodeError):
            _git("hash-object", "--stdin", cwd=repo, input=UNENCODABLE)
        assert _live_gits(repo) == [], "git was left running, waiting for its stdin"
    finally:
        _kill_all(_live_gits(repo))


def test_stop_after_preparing_input_leaves_no_git(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    hook = _StopOncePrepared()
    previous = sys.gettrace()  # a coverage tracer, say: given back afterwards
    try:
        sys.settrace(hook)
        try:
            with pytest.raises(ToolCallInterrupted):
                _git("hash-object", "--stdin", cwd=repo, input="hello\n")
        finally:
            sys.settrace(previous)
        assert hook.fired
        assert _live_gits(repo) == [], "git was left running, waiting for its stdin"
    finally:
        _kill_all(_live_gits(repo))
