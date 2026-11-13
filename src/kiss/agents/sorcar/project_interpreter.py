# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Python a project's commands run on, named in the Task Settings of every agent.

On 2026-11-11 twelve of fifty reviewers and 31 of 44 helpers reproduced
on the Python of a stray environment (3.14.7) what the fixers had tested
on the project's (3.13.5), and reported hangs the fixers could not see.
A fixer's result and its reviewer's are comparable only when both name
the interpreter they ran on: the harness names it to both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from kiss.agents.sorcar.useful_tools import _worktree_index

_VERSION_PROBE = "import platform; print(platform.python_version())"

#: The version each interpreter answered with, by its path, beside the
#: modification time its binary then had.
_versions: dict[str, tuple[int, str]] = {}


def project_interpreter(work_dir: str) -> str:
    """Return the Python of the project at *work_dir* as ``"<path> (Python X.Y.Z)"``.

    The project's Python is its environment's, else the ``python3`` (or
    ``python``) commands find on ``PATH``.  The environment of a work
    dir in a task worktree is the main checkout's ``.venv``, which the
    worktree shares; the worktree is the innermost one of the path, as
    :func:`~kiss.agents.sorcar.useful_tools.uv_in_worktree` reads it.
    Elsewhere, and where the main checkout has none (uv then builds
    the worktree its own), it is the nearest ``.venv`` at or above the
    work dir, no further up than the project's root (the nearest
    ``pyproject.toml``): the one a ``uv run`` typed there uses, unless
    the project is a member of a uv workspace, whose root is not
    looked for.

    Args:
        work_dir: The work directory of the task.

    Returns:
        The interpreter's path and version, or ``""`` when no Python
        that runs is found.
    """
    here = Path(work_dir)
    real = here.resolve()
    i = _worktree_index(real.parts)
    projects = [] if i is None else [Path(*real.parts[:i])]
    for directory in (here, *real.parents):
        projects.append(directory)
        if (directory / "pyproject.toml").exists():
            break
    in_environment = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    pythons = [str(project / ".venv" / in_environment) for project in projects]
    on_path = shutil.which("python3") or shutil.which("python")
    for python in dict.fromkeys((*pythons, on_path or "")):  # the main checkout's once
        version = _python_version(python)
        if version:
            return f"{python} (Python {version})"
    return ""


def _python_version(python: str) -> str:
    """Ask the interpreter at *python* for its version (``""`` when it does not run).

    An answer is kept for as long as the binary keeps its modification
    time (an environment rebuilt on another Python is asked again); a
    failure is not kept, since the same binary may run the next time (a
    probe that timed out under load, a mode fixed since).
    """
    try:
        stamp = os.stat(python).st_mtime_ns
    except OSError:  # no such binary (``""`` when PATH has no Python)
        return ""
    kept = _versions.get(python)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    try:
        probe = subprocess.run(
            [python, "-c", _VERSION_PROBE], capture_output=True, text=True, timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    version = probe.stdout.strip() if probe.returncode == 0 else ""
    if version:
        _versions[python] = (stamp, version)
    return version
