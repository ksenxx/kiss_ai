# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The per-file test durations store: written by the test suite, read by ``impacted-tests``.

``src/kiss/tests/conftest.py`` records, at the end of every pytest
session, the seconds each test file took (:func:`record_durations`);
``uv run impacted-tests --splits N`` balances its splits by those
figures (:func:`load_durations`) instead of by file size alone.  The
file is one per machine (:func:`durations_path`), shared by the main
checkout and its worktrees, so every run feeds the next balance.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from kiss.core.file_lock import exclusive_file_lock

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def durations_path(root: Path = PROJECT_ROOT) -> Path:
    """Return the file of per-test-file durations the splits are balanced by.

    ``$KISS_TEST_DURATIONS`` when set, else ``tmp/test-durations.json``
    of the main checkout: a worktree (``git rev-parse --git-common-dir``
    names the main repository's ``.git``) shares the checkout's file, so
    a task's splits are balanced by every earlier run on the machine.
    Outside git, the file is under *root* itself.
    """
    env = os.environ.get("KISS_TEST_DURATIONS")
    if env:
        return Path(env)
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=root, capture_output=True,
            text=True, timeout=30, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        common = ""
    main = (root / common).resolve().parent if common else root
    return main / "tmp" / "test-durations.json"


def load_durations(path: Path) -> dict[str, float]:
    """Return the recorded seconds by test path (relative, POSIX) in *path*; ``{}`` without.

    The file is shared by every checkout on the machine, so what is not a
    duration there is left out rather than raised: a file that is no JSON
    object, and a value that is no finite, non-negative number (every JSON
    number is read as a float).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_int=float)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, float) and 0 <= value < math.inf
    }


def record_durations(path: Path, durations: dict[str, float]) -> None:
    """Merge *durations* (seconds by test path relative to the repo root) into *path*.

    Read, merged and rewritten under an exclusive lock on its ``.lock`` sibling
    (:func:`~kiss.core.file_lock.exclusive_file_lock`, which works on
    every platform), since the splits of one ``run_commands_parallel``
    call finish together; the latest run of a file replaces its earlier
    figure.

    Args:
        path: The JSON file (:func:`durations_path`); created with its directory.
        durations: What a session measured, by ``src/kiss/tests/...`` path.
    """
    if not durations:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(path.with_suffix(".lock")):
        known = load_durations(path)
        known.update({key: round(seconds, 3) for key, seconds in durations.items()})
        staged = path.with_suffix(".tmp")
        staged.write_text(json.dumps(known, sort_keys=True, indent=0), encoding="utf-8")
        os.replace(staged, path)
