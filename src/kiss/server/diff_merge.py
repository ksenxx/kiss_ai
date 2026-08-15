# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File scanning and git subprocess utilities.

Historically this module also prepared the interactive diff/merge
review view; that workflow was removed, leaving the file scanner used
by autocomplete and a positional-``cwd`` adapter over the single
hardened git runner in :mod:`kiss.agents.sorcar.git_worktree`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import _git as _git_run
from kiss.agents.sorcar.git_worktree import _unquote_git_path

logger = logging.getLogger(__name__)


def _load_gitignore_dirs(work_dir: str) -> tuple[set[str], set[str]]:
    """Load directory names and paths to skip from .gitignore.

    Parses .gitignore for entries without glob characters.  Following
    gitignore semantics, an entry containing a slash anywhere other
    than at its end is anchored to the repository root (``/build``,
    ``src/generated``), while a bare name (``node_modules``,
    ``build/``) matches at any depth.

    Args:
        work_dir: Repository root containing .gitignore.

    Returns:
        ``(skip_names, skip_paths)`` — *skip_names* are bare directory
        names to skip at any depth (always includes ``.git``);
        *skip_paths* are root-relative directory paths to skip at
        their exact location only.
    """
    skip_names = {".git"}
    skip_paths: set[str] = set()
    try:
        gitignore = Path(work_dir) / ".gitignore"
        for raw_line in gitignore.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            if "*" in line or "?" in line:
                continue
            entry = line.rstrip("/")
            if "/" in entry:
                skip_paths.add(entry.lstrip("/"))
            else:
                skip_names.add(entry)
    except OSError:
        logger.debug("Exception caught", exc_info=True)
    return skip_names, skip_paths


def _scan_files(work_dir: str) -> list[str]:
    """Scan workspace files, respecting .gitignore patterns.

    Args:
        work_dir: Repository root to scan.

    Returns:
        List of relative file and directory paths.
    """
    paths: list[str] = []
    skip_names, skip_paths = _load_gitignore_dirs(work_dir)
    wd = Path(work_dir)
    try:
        for root, dirs, files in wd.walk():
            rel_root = root.relative_to(wd)
            if len(rel_root.parts) > 10:
                dirs.clear()
                continue
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in skip_names
                and not d.startswith(".")
                and str(rel_root / d) not in skip_paths
            )
            for name in sorted(files):
                paths.append(str(rel_root / name).replace(os.sep, "/"))
                if len(paths) >= 5000:
                    return paths
            for d in dirs:
                paths.append(str(rel_root / d).replace(os.sep, "/") + "/")
                if len(paths) >= 5000:
                    return paths
    except OSError:  # pragma: no cover — Path.walk swallows OSErrors internally
        logger.debug("Exception caught", exc_info=True)
    return paths


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
