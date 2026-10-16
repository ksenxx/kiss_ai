# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-16: rsi7d's git calls are bounded and never wait on a terminal.

``rsi7d_sea._git`` runs ``git clone`` / ``log`` / ``status`` / ``rev-parse``
for the replay tools.  It used to run them with no timeout and the
task's stdin, so a git that blocks (a hook or filter waiting for input,
a credential prompt, a wedged object store) hung the rsi7d task and
the daemon thread running it for good.  The command now gets a closed
stdin and :data:`rsi7d_sea.GIT_TIMEOUT_SECONDS`; a stuck one is
reported as a failed process (exit 124), which the callers already
handle as "not a checkout" / "clone failed".
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.tests.conftest import posix_only


def _fake_git(folder: Path, body: str) -> None:
    """Put an executable ``git`` running *body* (a shell snippet) first on PATH."""
    script = folder / "git"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


@posix_only("the fake git is a shell script")
def test_a_stuck_git_is_cut_off_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git that never returns fails within the cap instead of hanging the task."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_git(bin_dir, "exec sleep 30")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(sea, "GIT_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    proc = sea._git("rev-parse", "--show-toplevel", cwd=tmp_path)
    elapsed = time.monotonic() - started
    assert elapsed < 10, f"the git call took {elapsed:.1f}s: no deadline"
    assert proc.returncode == 124 and proc.stdout == ""
    assert proc.stderr == "git rev-parse --show-toplevel did not finish within 0.5s"
    # The callers see the failure as "not a checkout", the way they see any failed git.
    assert sea._toplevel(tmp_path) is None


@posix_only("the fake git is a shell script")
def test_git_reads_no_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A git waiting for input gets end-of-file at once: nothing of the task's stdin reaches it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_git(bin_dir, "printf 'stdin:[%s]' \"$(cat)\"")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    # Under pytest fd 0 is already at end-of-file, so an inherited stdin would
    # look closed too: give fd 0 real content for the call, which the fake
    # git's ``cat`` would echo back if ``_git`` did not use DEVNULL.
    inherited = tmp_path / "inherited-stdin"
    inherited.write_text("INHERITED-INPUT", encoding="utf-8")
    saved_stdin = os.dup(0)
    try:
        with inherited.open("rb") as source:
            os.dup2(source.fileno(), 0)
        proc = sea._git("status", cwd=tmp_path)
    finally:
        os.dup2(saved_stdin, 0)
        os.close(saved_stdin)
    assert proc.returncode == 0 and proc.stdout == "stdin:[]"
