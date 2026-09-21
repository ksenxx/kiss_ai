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
import posixpath
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


# Hard cap on the number of entries ``_scan_files`` returns.  The picker
# filters this cache by substring, so a too-small cap silently hides whole
# subtrees (e.g. ``src/`` in a repo with thousands of tracked data files).
_SCAN_FILES_CAP = 1_000_000

# Suffixes of machine-generated data: run results, logs, traces, dumps,
# serialized tensors.  Only bulk matters here (see ``_bulk_data_dirs``), so
# the list errs towards data formats: a suffix missing from it merely keeps a
# data dump in the picker, whereas listing a source suffix would hide code.
_DATA_SUFFIXES = frozenset({
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".log", ".txt", ".out",
    ".err", ".csv", ".tsv", ".parquet", ".arrow", ".npy", ".npz", ".pkl",
    ".pickle", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".db",
    ".sqlite", ".sqlite3", ".patch", ".diff",
})
# A directory is a bulk data dump when its subtree (minus bulk children)
# holds at least this many files and at least this fraction of them carry a
# data suffix.
_BULK_DATA_MIN_FILES = 200
_BULK_DATA_MIN_FRACTION = 0.9


def _is_under(path: str, dirs: set[str]) -> bool:
    """Return whether *path* lies strictly inside one of *dirs*.

    Args:
        path: Slash-separated relative path; directories end with ``/``.
        dirs: Slash-separated relative directory paths without trailing ``/``.

    Returns:
        True when some entry of *dirs* is a proper ancestor of *path* (the
        directory's own ``dir/`` entry is not "under" itself).
    """
    parts = path.rstrip("/").split("/")
    return any("/".join(parts[:i]) in dirs for i in range(1, len(parts)))


def _bulk_data_dirs(own_counts: dict[str, list[int]]) -> set[str]:
    """Pick the directories whose subtree is almost entirely data files.

    Directories are folded bottom-up: a bulk subtree is recorded and NOT
    added to its parent's totals, so ``bench/`` with five scripts and a
    250-file ``bench/results/`` dump yields ``{"bench/results"}`` — the
    scripts stay visible.  Small dumps spread over many subdirectories
    still add up at the first ancestor that crosses the threshold.

    Args:
        own_counts: Map from relative directory path (``"."`` for the
            root) to ``[files, data_files]`` directly inside it; every
            walked directory has an entry, and the parent of every
            non-root entry is itself an entry.

    Returns:
        Relative paths of directories with at least ``_BULK_DATA_MIN_FILES``
        files of which at least ``_BULK_DATA_MIN_FRACTION`` are data files.
        The root is never returned: a repository that is itself a data set
        should still be browsable.
    """
    bulk: set[str] = set()
    totals = {d: list(c) for d, c in own_counts.items()}
    non_root = [d for d in totals if d != "."]
    for d in sorted(non_root, key=lambda p: p.count("/"), reverse=True):  # deepest first
        files, data_files = totals[d]
        if files >= _BULK_DATA_MIN_FILES and data_files >= _BULK_DATA_MIN_FRACTION * files:
            bulk.add(d)
            continue
        parent = totals[posixpath.dirname(d) or "."]
        parent[0] += files
        parent[1] += data_files
    return bulk


def _scan_files(work_dir: str) -> list[str]:
    """Scan workspace files, respecting .gitignore patterns.

    Directories that are bulk data dumps (see ``_bulk_data_dirs``: hundreds
    of tracked JSON/log/text results) are listed as a single ``dir/`` entry
    and their contents are left out, so the ``@``-mention picker's results
    are source files rather than run artifacts.  Directories holding code —
    even thousands of files of it — are listed in full.

    Args:
        work_dir: Repository root to scan.

    Returns:
        List of relative file and directory paths.
    """
    paths: list[str] = []
    # Per directory, [files, data_files] directly inside it.
    own_counts: dict[str, list[int]] = {}
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
                # .gitignore paths are slash-separated on every platform.
                and (rel_root / d).as_posix() not in skip_paths
            )
            n_data = sum(
                1 for f in files if os.path.splitext(f)[1].lower() in _DATA_SUFFIXES
            )
            own_counts[rel_root.as_posix()] = [len(files), n_data]
            entries = [str(rel_root / n).replace(os.sep, "/") for n in sorted(files)]
            entries += [str(rel_root / d).replace(os.sep, "/") + "/" for d in dirs]
            paths.extend(entries[: _SCAN_FILES_CAP - len(paths)])
            if len(paths) >= _SCAN_FILES_CAP:
                break
    except OSError:  # pragma: no cover — Path.walk swallows OSErrors internally
        logger.debug("Exception caught", exc_info=True)
    # The cap bounds the walk itself, so entries dropped here are not
    # refilled; that only matters for a tree of over a million entries.
    bulk = _bulk_data_dirs(own_counts)
    if not bulk:
        return paths
    return [p for p in paths if not _is_under(p, bulk)]


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
