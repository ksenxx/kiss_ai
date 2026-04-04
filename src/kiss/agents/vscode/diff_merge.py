"""File scanning and git utilities."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)



def _load_gitignore_dirs(work_dir: str) -> set[str]:
    """Load directory names and paths to skip from .gitignore.

    Parses .gitignore for entries without glob characters and returns
    them as a set.  Entries may be simple names (e.g. ``node_modules``)
    or paths (e.g. ``src/generated``).  Always includes ``.git``.

    Args:
        work_dir: Repository root containing .gitignore.

    Returns:
        Set of directory names/paths to skip during file scanning.
    """
    skip = {".git"}
    try:
        gitignore = Path(work_dir) / ".gitignore"
        for raw_line in gitignore.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            # Strip trailing slash (directory marker)
            name = line.rstrip("/")
            # Only use names/paths — skip glob patterns
            if "*" in name or "?" in name:
                continue
            skip.add(name)
    except OSError:
        logger.debug("Exception caught", exc_info=True)
    return skip


def _scan_files(work_dir: str) -> list[str]:
    """Scan workspace files, respecting .gitignore patterns.

    Args:
        work_dir: Repository root to scan.

    Returns:
        List of relative file and directory paths.
    """
    paths: list[str] = []
    skip = _load_gitignore_dirs(work_dir)
    wd = Path(work_dir)
    try:
        for root, dirs, files in wd.walk():
            rel_root = root.relative_to(wd)
            if len(rel_root.parts) - 1 > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in skip
                and not d.startswith(".")
                and str(rel_root / d) not in skip
            )
            for name in sorted(files):
                paths.append(str(rel_root / name).replace(os.sep, "/"))
                if len(paths) >= 5000:
                    return paths
            for d in dirs:
                paths.append(str(rel_root / d).replace(os.sep, "/") + "/")
    except OSError:  # pragma: no cover — Path.walk swallows OSErrors internally
        logger.debug("Exception caught", exc_info=True)
    return paths


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured text output.

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as strings.
    """
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
