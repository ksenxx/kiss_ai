# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression test: ``git_worktree._git`` must time out.

Finding #1 (findings-6): ``git_worktree._git`` had NO timeout — a git
process wedged in a hook or local repository operation blocked the
agent thread forever.  ``vscode/diff_merge._git``
already documents and carries this protection; this test pins the same
guarantee for the sorcar-side wrapper.

The hang is reproduced with a REAL stub ``git`` executable placed
first on ``PATH`` (a shell script that sleeps far longer than the
timeout), so the actual ``subprocess.run(..., timeout=...)`` code path
is exercised — no kiss code is mocked.  Before the fix this test hung
until the outer pytest timeout instead of returning.
"""

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import _git


def _install_hanging_git(tmp_path: Path) -> Path:
    """Create a stub ``git`` that hangs, in a fresh bin dir on disk."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text("#!/bin/sh\nsleep 300\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


class TestGitTimeout:
    """``_git`` returns a synthesized failure instead of hanging."""

    def test_hanging_git_returns_timeout_completedprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hung git yields returncode 124 with a timeout message."""
        bin_dir = _install_hanging_git(tmp_path)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        # Tune ONLY the timeout duration so the test does not need to
        # wait the production timeout; the real subprocess timeout path
        # (subprocess.TimeoutExpired -> synthesized CompletedProcess)
        # is exercised unmodified.
        saved_timeout = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        try:
            start = time.monotonic()
            result = _git("status", cwd=tmp_path)
            elapsed = time.monotonic() - start
        finally:
            git_worktree._GIT_TIMEOUT_SECONDS = saved_timeout
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 124, (
            "expected the synthesized timeout returncode (124), got "
            f"{result.returncode}; stderr={result.stderr!r}"
        )
        assert "timed out" in result.stderr
        assert elapsed < 30, (
            f"_git took {elapsed:.1f}s — the 1s timeout did not fire"
        )

    def test_real_git_still_works_within_timeout(self, tmp_path: Path) -> None:
        """The normal (non-hanging) path is unchanged by the timeout."""
        repo = tmp_path / "repo"
        repo.mkdir()
        assert _git("init", cwd=repo).returncode == 0
        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout == ""
