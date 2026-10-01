# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The git runner the server uses must kill git's whole process group.

``kiss.server.merge_flow`` and ``kiss.server.explorer`` run every git
command through :func:`kiss.agents.sorcar.git_worktree._git` (the
former ``kiss.server.diff_merge`` shim is gone).  Git's grandchildren —
a credential helper, ``core.askPass``, a smudge/clean filter, ``ssh`` —
are not children of the spawned process: a runner that killed git
alone would leave them running against the repository the agent is
about to touch, still holding the inherited stdout/stderr pipes.

Everything is real: a stub ``git`` first on ``PATH`` that forks a real
grandchild inheriting the pipes, real subprocesses, real timeouts.  The
timeout budget is dialled down through the module's own knob exactly as
``test_git_worktree_timeout.py`` does, so the test finishes in seconds.
"""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import _git
from kiss.tests.conftest import posix_only

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


class TestGitRunnerKillsGrandchildren:
    """A timed-out git takes every process it spawned down with it."""

    @posix_only("the forking git stub is a #!/bin/sh script")
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
            result = _git("add", "-A", cwd=str(tmp_path))
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

    def test_real_git_still_works(self, tmp_path: Path) -> None:
        """The normal path is unchanged: same repo, same captured output."""
        repo = tmp_path / "repo"
        repo.mkdir()
        assert _git("init", "-q", cwd=str(repo)).returncode == 0
        Path(repo, "a.txt").write_text("a\n")
        result = _git("status", "--porcelain", cwd=str(repo))
        assert result.returncode == 0
        assert result.stdout == "?? a.txt\n"
