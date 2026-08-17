# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``diff_merge._git`` must be the same hardened runner as git_worktree's.

R09-2.  ``diff_merge._git``'s docstring claimed parity with
``git_worktree._git`` while carrying its own copy of the runner, and
the copy had drifted in two observable ways:

* **Timeout budget.**  30 seconds against the very repositories the
  300-second runner works on.  ``merge_flow`` runs ``status
  --porcelain -uall`` through this helper and ``_main_dirty_files``
  maps a non-zero return to "nothing is dirty", so on a big working
  tree the synthesized timeout silently skips the whole post-task
  auto-commit and the agent's work is left uncommitted with no error.

* **What the timeout kills.**  ``subprocess.run`` kills the git
  process alone.  git's grandchildren — a credential helper,
  ``core.askPass``, a smudge/clean filter, ``ssh`` — are not in that
  set: they survive, still holding the inherited stdout/stderr pipes,
  and keep running against the repo the agent is about to touch.
  ``git_worktree._git`` runs git in its own session and kills the
  whole process **group**.  (On Windows the surviving grandchild is
  worse than a leak: ``subprocess.run`` there calls an unbounded
  ``communicate()`` after the kill, which the open pipe blocks
  forever.  That path cannot be exercised here, so these tests pin the
  POSIX-observable half: nothing outlives the timeout.)

Everything is real: a stub ``git`` first on ``PATH`` that forks a real
grandchild inheriting the pipes, real subprocesses, real timeouts.
Nothing in kiss is mocked; the timeout budget is dialled down through
the module's own knob exactly as the sibling
``test_git_worktree_timeout.py`` does, so the tests finish in seconds.
"""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.server.diff_merge import _git

#: How long the stub git sleeps.  Long enough that a run which ignores
#: the dialled-down budget is unambiguous, short enough that a failing
#: run ends by itself instead of wedging the suite.
_STUB_LIFETIME_SECONDS = 30

#: How long the grandchild waits before recording that it survived.
_GRANDCHILD_DELAY_SECONDS = 3


def _install_forking_git(tmp_path: Path, marker: Path) -> Path:
    """Install a stub ``git`` that forks a grandchild outliving a kill.

    The stub backgrounds a second shell before sleeping itself.  That
    grandchild inherits the caller's stdout/stderr pipes and is not a
    child of the spawned process, so killing the stub alone leaves it
    running — the situation a real credential helper or clean filter
    creates.  It touches *marker* once it has outlived the timeout, so
    a test can tell whether the process group really died.

    Args:
        tmp_path: Directory to create the stub ``bin`` folder in.
        marker: Path the surviving grandchild creates.

    Returns:
        The directory to prepend to ``PATH``.
    """
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text(
        "#!/bin/sh\n"
        f"sh -c 'sleep {_GRANDCHILD_DELAY_SECONDS}; touch \"{marker}\"' &\n"
        f"sleep {_STUB_LIFETIME_SECONDS}\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


class TestDiffMergeGitIsHardened:
    """The server-side git helper must not keep a private runner."""

    def test_timeout_kills_the_whole_process_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing git spawned may outlive the timeout."""
        marker = tmp_path / "grandchild-survived"
        bin_dir = _install_forking_git(tmp_path, marker)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        saved = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        try:
            start = time.monotonic()
            result = _git(str(tmp_path), "add", "-A")
            elapsed = time.monotonic() - start
            time.sleep(_GRANDCHILD_DELAY_SECONDS + 1)
        finally:
            git_worktree._GIT_TIMEOUT_SECONDS = saved

        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 124, (
            "expected the synthesized timeout returncode (124), got "
            f"{result.returncode}; stderr={result.stderr!r}"
        )
        assert elapsed < _STUB_LIFETIME_SECONDS / 2, (
            f"_git ran for {elapsed:.1f}s against a 1s budget"
        )
        assert not marker.exists(), (
            "a process git spawned outlived the timeout: only the git "
            "process itself was killed, so a credential helper or clean "
            "filter keeps running against the repo — and on Windows its "
            "hold on the inherited pipes blocks the caller forever"
        )

    def test_timeout_budget_is_shared_with_git_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One budget governs both helpers, as the docs always claimed.

        Only ``git_worktree``'s knob is dialled down here.  A private
        copy of the budget in ``diff_merge`` would ignore it and let
        the stub run to completion.
        """
        marker = tmp_path / "grandchild-survived"
        bin_dir = _install_forking_git(tmp_path, marker)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        saved = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        try:
            start = time.monotonic()
            result = _git(str(tmp_path), "status", "--porcelain")
            elapsed = time.monotonic() - start
            worktree_result = git_worktree._git("status", cwd=tmp_path)
        finally:
            git_worktree._GIT_TIMEOUT_SECONDS = saved

        assert result.returncode == worktree_result.returncode == 124, (
            "diff_merge._git kept its own timeout budget: "
            f"{result.returncode} vs {worktree_result.returncode}"
        )
        assert elapsed < _STUB_LIFETIME_SECONDS / 2, (
            f"diff_merge._git ran for {elapsed:.1f}s while git_worktree's "
            "budget was 1s"
        )

    def test_real_git_still_works(self, tmp_path: Path) -> None:
        """The normal path is unchanged: same repo, same captured output."""
        repo = tmp_path / "repo"
        repo.mkdir()
        assert _git(str(repo), "init", "-q").returncode == 0
        Path(repo, "a.txt").write_text("a\n")
        result = _git(str(repo), "status", "--porcelain")
        assert result.returncode == 0
        assert result.stdout == "?? a.txt\n"
