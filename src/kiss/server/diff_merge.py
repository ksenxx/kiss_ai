# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Git subprocess utilities.

Historically this module also prepared the interactive diff/merge
review view and scanned the workspace for the ``@``-mention picker;
the former was removed and the latter lives in
:mod:`kiss.server.file_index`, leaving a positional-``cwd`` adapter
over the single hardened git runner in
:mod:`kiss.agents.sorcar.git_worktree`.
"""

from __future__ import annotations

import subprocess

from kiss.agents.sorcar.git_worktree import _git as _git_run
from kiss.agents.sorcar.git_worktree import _unquote_git_path


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd* with captured text output.

    A thin positional-``cwd`` adapter over
    :func:`kiss.agents.sorcar.git_worktree._git`, which is the single
    hardened git runner: one timeout budget, repo-scoped ``GIT_*``
    variables scrubbed, ``errors="surrogateescape"`` decoding, and a
    timeout path that kills the whole process **group** and then waits
    only briefly.  This module used to carry its own copy, which had
    drifted to a 10× shorter timeout and to a kill that could still
    hang forever: ``subprocess.run`` kills the git process alone and
    then waits without a bound for its output pipes to close, so a
    surviving grandchild (credential helper, ``core.askPass``, a
    smudge/clean filter, ``ssh``) that inherited them blocked the
    caller indefinitely — wedging ``repo_lock`` for every tab.

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as strings; ``returncode``
        124 when the command timed out.
    """
    return _git_run(*args, cwd=cwd)


def _capture_untracked(work_dir: str) -> set[str]:
    """Return the set of untracked files in the repo.

    Args:
        work_dir: Repository root directory.

    Returns:
        Set of untracked file paths relative to work_dir.
    """
    result = _git(work_dir, "ls-files", "--others", "--exclude-standard")
    return {
        _unquote_git_path(line)
        for line in result.stdout.split("\n")
        if line
    }
