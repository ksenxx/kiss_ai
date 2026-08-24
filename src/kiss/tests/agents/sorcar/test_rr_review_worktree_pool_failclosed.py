# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix — pool callers fail closed on ignored-file enumeration failure.

``GitWorktreeOps.list_ignored_files`` returns ``None`` when git itself
fails (as opposed to ``[]`` for "no ignored files").  The pool's two
content probes used the call in boolean context, where ``None`` is
falsy — so an enumeration FAILURE read as "no ignored content":

* ``take_spare`` could hand a spare containing foreign ignored data to
  a task, whose setup then runs ``reset --hard`` + ``git clean -fdq``
  over it;
* ``discard_all`` could destroy a worktree whose ignored content could
  never be enumerated.

Both callers must treat ``None`` exactly like "has content": preserve
the worktree and do not admit / destroy it (the same fail-closed
semantics ``rescue_ignored_files`` and the reclaim spare probe apply).

The git failure is REAL, not mocked: a ``git`` shim executable is
prepended to ``PATH`` that fails only the ``ls-files`` sub-command
(exit 128 on stderr, like a corrupt index would) and ``exec``s the
real git for everything else — so ``git status`` still reports the
spare clean and the ``ignored is None`` decision is the one under
test.  Everything else (repos, worktrees, branches) is real git.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.agents.sorcar.test_worktree_pool import (
    _enable_pool,
    _make_repo,
    _restore_pool_env,
)


def _install_ls_files_failing_git(shim_dir: Path) -> str:
    """Prepend a git shim to ``PATH`` that fails only ``ls-files``.

    Args:
        shim_dir: Directory to create the shim executable in.

    Returns:
        The previous ``PATH`` value, for restore.
    """
    real_git = shutil.which("git")
    assert real_git is not None
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    ls-files)\n"
        '      echo "fatal: injected ls-files failure" >&2\n'
        "      exit 128\n"
        "      ;;\n"
        "  esac\n"
        "done\n"
        f'exec "{real_git}" "$@"\n'
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    prev_path = os.environ["PATH"]
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{prev_path}"
    return prev_path


class TestPoolFailsClosedOnEnumerationFailure:
    """``ignored is None`` must preserve, in BOTH pool callers."""

    def setup_method(self) -> None:
        self._prev_env = _enable_pool()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmpdir.name) / "repo")
        self._shim_dir = Path(self._tmpdir.name) / "shim"
        self._shim_dir.mkdir()
        self._prev_path: str | None = None

    def teardown_method(self) -> None:
        self._restore_path()
        worktree_pool.discard_all()
        self._tmpdir.cleanup()
        _restore_pool_env(self._prev_env)

    def _restore_path(self) -> None:
        if self._prev_path is not None:
            os.environ["PATH"] = self._prev_path
            self._prev_path = None

    def _prewarm_spare(self) -> tuple[str, Path]:
        assert worktree_pool.prewarm(self.repo) is True
        with worktree_pool._pool_lock:
            (spare,) = worktree_pool._spares.values()
        branch, wt_dir = spare
        assert wt_dir.is_dir()
        return branch, wt_dir

    def _break_ls_files(self, wt_dir: Path) -> None:
        self._prev_path = _install_ls_files_failing_git(self._shim_dir)
        # Prove the injected failure hits exactly the probe under
        # test: enumeration fails, while the status probe still works
        # and reports the spare clean.
        assert GitWorktreeOps.list_ignored_files(wt_dir) is None
        assert GitWorktreeOps.has_uncommitted_changes(wt_dir) is False

    def test_take_spare_refuses_spare_it_cannot_verify(self) -> None:
        branch, wt_dir = self._prewarm_spare()
        # A file the broken enumeration can no longer report — exactly
        # what consuming the spare would destroy.
        (wt_dir / ".gitignore").write_text("*.secret\n")
        (wt_dir / "keys.secret").write_text("foreign data\n")
        subprocess.run(
            ["git", "-C", str(wt_dir), "add", ".gitignore"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt_dir), "commit", "-m", "ignore secrets"],
            capture_output=True, check=True,
        )
        self._break_ls_files(wt_dir)

        assert worktree_pool.take_spare(self.repo) is None, (
            "take_spare handed out a spare whose ignored content "
            "could not be enumerated"
        )

        self._restore_path()
        # Preserved for the reclaim pass: directory, branch, and the
        # unenumerable file are all still there.
        assert (wt_dir / "keys.secret").read_text() == "foreign data\n"
        assert GitWorktreeOps.branch_exists(self.repo, branch)
        # The entry was popped: the pool no longer offers it.
        assert worktree_pool.take_spare(self.repo) is None

    def test_discard_all_preserves_spare_it_cannot_verify(self) -> None:
        branch, wt_dir = self._prewarm_spare()
        (wt_dir / "unverified.dat").write_text("cannot be enumerated\n")
        # The file is untracked-but-not-ignored, so status would call
        # the spare dirty on its own; commit a .gitignore covering it
        # so ONLY the failed enumeration stands between discard_all
        # and the data.
        (wt_dir / ".gitignore").write_text("*.dat\n")
        subprocess.run(
            ["git", "-C", str(wt_dir), "add", ".gitignore"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt_dir), "commit", "-m", "ignore dat"],
            capture_output=True, check=True,
        )
        self._break_ls_files(wt_dir)

        worktree_pool.discard_all()

        self._restore_path()
        assert wt_dir.is_dir(), (
            "discard_all destroyed a worktree whose ignored content "
            "could not be enumerated"
        )
        assert (wt_dir / "unverified.dat").read_text() == (
            "cannot be enumerated\n"
        )
        assert GitWorktreeOps.branch_exists(self.repo, branch)
        # Pool state itself is swept regardless.
        assert worktree_pool.spare_branches() == set()

    def test_discard_all_still_discards_verifiable_clean_spare(self) -> None:
        # Control: with git healthy the same clean spare IS discarded,
        # so the preservation above is attributable to the failed
        # enumeration, not to a behavior change in discard_all.
        branch, wt_dir = self._prewarm_spare()
        worktree_pool.discard_all()
        assert not wt_dir.is_dir()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)
