# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the gpt-5.6-sol Sorcar review fixes."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import git_worktree, web_use_tool


def test_profile_permission_denied_lock_is_conservatively_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consolidating the lock helpers must preserve EPERM semantics.

    The old inline implementation caught ``PermissionError`` from both
    ``readlink`` and ``kill(0)`` and treated the profile as occupied.  A
    composition that lets ``_read_lock_pid`` swallow readlink EPERM as
    ``None`` incorrectly reports the inaccessible profile as free.
    """
    profile = tmp_path / "profile"
    profile.mkdir()
    lock = profile / "SingletonLock"
    lock.symlink_to("foreign-host-123")

    def denied(_path: str) -> str:
        raise PermissionError("foreign-owned lock")

    monkeypatch.setattr(web_use_tool.os, "readlink", denied)
    assert web_use_tool._is_profile_in_use(str(profile)) is True


def test_git_timeout_kills_hook_descendants_holding_capture_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out git must kill descendants, not only its top process.

    ``subprocess.run(timeout=...)`` kills only git itself.  A hung hook
    that inherited the capture pipes then keeps ``communicate`` blocked
    until the hook exits.  The wrapper starts git in a process group and
    terminates that whole group on timeout.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(4)'])\n"
        "time.sleep(4)\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(git_worktree, "_GIT_TIMEOUT_SECONDS", 0.2)

    start = time.monotonic()
    result = git_worktree._git("status", cwd=tmp_path)
    elapsed = time.monotonic() - start

    assert result.returncode == 124
    assert "timed out" in result.stderr
    assert elapsed < 1.5, f"descendant kept capture pipes open for {elapsed:.2f}s"
