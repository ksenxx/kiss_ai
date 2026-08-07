# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File scanning and git subprocess utilities.

Historically this module also prepared the interactive diff/merge
review view; that workflow was removed, leaving the file scanner used
by autocomplete and the shared ``git`` subprocess helpers.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import _REPO_SCOPED_GIT_ENV, _unquote_git_path

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


_GIT_TIMEOUT_SECONDS: float = 30.0


def _scrubbed_git_env() -> dict[str, str]:
    """Return a copy of the environment without repo-scoped GIT_* vars.

    Strips ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE`` etc.
    (see :data:`kiss.agents.sorcar.git_worktree._REPO_SCOPED_GIT_ENV`)
    so an inherited variable — e.g. from a git hook that launched this
    process — cannot redirect the command away from the ``cwd`` passed
    to :func:`_git`.  This is the same scrub
    ``git_worktree._git`` applies.

    Returns:
        Environment mapping safe to pass to a git subprocess.
    """
    return {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_ENV}


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured text output.

    Always passes a 30-second timeout so a hung git process (e.g. waiting
    on a credential-helper prompt or a network remote) cannot block the
    agent thread forever (M1).  On timeout returns a non-zero
    ``CompletedProcess`` so callers don't crash.

    Repo-scoped ``GIT_*`` environment variables are scrubbed (see
    :func:`_scrubbed_git_env`) and output is decoded with
    ``errors="surrogateescape"`` because git paths are byte strings
    that may be invalid UTF-8 — a strict decode would raise
    ``UnicodeDecodeError`` out of every git call touching such a
    filename.  Both behaviors match ``git_worktree._git``.

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as strings.
    """
    try:
        return subprocess.run(
            ["git", "-c", "core.quotepath=false", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            cwd=cwd,
            env=_scrubbed_git_env(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("git %s timed out after %ss", args, _GIT_TIMEOUT_SECONDS)
        stdout = (
            exc.stdout.decode("utf-8", "surrogateescape")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", "surrogateescape")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=124,
            stdout=stdout or "",
            stderr=stderr or f"git {args[0] if args else ''} timed out",
        )


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


