# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""List the test files whose import graph reaches the files changed in the working tree.

``uv run impacted-tests`` prints one test path per line: every test
module under ``src/kiss/tests`` that imports, directly or through other
``kiss`` modules, a changed source file (``git diff --name-only HEAD``
plus untracked files, or the paths given on the command line).  A
changed test file is impacted by itself, and a changed ``conftest.py``
impacts every test below its directory, which pytest loads it for
without an import.  ``--splits N`` prints instead
the JSON list of ``N`` ``uv run pytest`` commands for
``run_commands_parallel``, each with a per-test timeout
(:data:`DEFAULT_TEST_TIMEOUT_SECONDS`) and balanced by the durations
earlier runs recorded (:mod:`kiss.scripts.durations`; the test suite's
conftest writes them at every session's end), by file size where none
is known.
``--owned <paths>`` lists instead the tests that are the given files or
are named for them (``test_<module>*.py``), no import hops: what a
sub-agent fixing a few files in a tree shared with siblings runs.

The 2026-10-24 trajectory audit found mid-task test runs selected by
directory globs (one of them 58 CPU-minutes for a one-module change)
and by guesswork; the import graph names the tests that can actually
see a change.  The 2026-11-08 sweep then ran the suite about five times
over (every fixer re-running the closure of its siblings' files too) and
lost a round to one 74-minute split, which ``--owned``, the timeout and
the duration balance address.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

from kiss.scripts.durations import durations_path, load_durations

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
_SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".git", "tmp"})


def _module_name(path: Path, src: Path = SRC) -> str:
    """Return the dotted module name of *path* under *src* (``a.b`` for ``src/a/b/__init__.py``)."""
    parts = path.relative_to(src).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def source_files(src: Path = SRC) -> list[Path]:
    """Return every ``.py`` file under *src*, caches skipped."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        files.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(".py"))
    return files


def imported_modules(path: Path, src: Path = SRC) -> set[str]:
    """Return the ``kiss`` module names *path* imports.

    ``from kiss.a import b`` yields both ``kiss.a`` and ``kiss.a.b``: the
    graph keeps whichever of the two is a module.  A relative import
    (``from ._helper import x``, ``from .. import y``) is resolved
    against the file's own package.

    Args:
        path: A Python source file.
        src: The source root *path* lies under (default: the repository's ``src/``).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    package = _module_name(path, src).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("kiss"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(package[: len(package) - (node.level - 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module.startswith("kiss"):
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


DEFAULT_DEPTH = 2
"""How many import hops from a changed module a test may sit: 1 is the
tests importing the module itself, 2 adds the tests of its direct
importers.  The full closure of a core module (``kiss_agent``,
``sorcar_agent``) is the whole suite, which is what the full run is for."""


def impacted_tests(changed: list[Path], src: Path = SRC, depth: int = DEFAULT_DEPTH) -> list[Path]:
    """Return the test files within *depth* import hops of *changed* in the reverse import graph.

    A changed ``conftest.py`` adds every test below its directory, whatever
    the depth: pytest loads it for them without an import.

    Args:
        changed: Changed files (absolute, or relative to the repo root);
            other files outside *src* or not ``.py`` are ignored.
        src: The source root to index (default: the repository's ``src/``).
        depth: Maximum import hops from a changed module to a test
            (:data:`DEFAULT_DEPTH`); ``0`` means no bound.
    """
    files = source_files(src)
    by_module = {_module_name(f, src): f for f in files}
    # Every imported name is kept, module or not: a deleted module has
    # no file any more but its importers still name it, and they are the
    # tests that break.
    importers: dict[str, set[str]] = defaultdict(set)
    for path in files:
        importer = _module_name(path, src)
        for name in imported_modules(path, src):
            # Importing a.b.c runs a/__init__.py and a/b/__init__.py first:
            # the importer depends on every package above the name too.
            while name:
                importers[name].add(importer)
                name = name.rpartition(".")[0]
    roots = []
    conftest_dirs = []
    for path in changed:
        resolved = path.resolve() if path.is_absolute() else (src.parent / path).resolve()
        if resolved.name == "conftest.py":
            conftest_dirs.append(resolved.parent)
        if resolved.suffix == ".py" and resolved.is_relative_to(src):
            roots.append(_module_name(resolved, src))
    seen = set(roots)
    queue = deque((root, 0) for root in roots)
    while queue:
        module, hops = queue.popleft()
        if depth and hops >= depth:
            continue
        for importer in importers.get(module, ()):
            if importer not in seen:
                seen.add(importer)
                queue.append((importer, hops + 1))
    tests_dir = src / "kiss" / "tests"
    return sorted(
        test
        for module, test in by_module.items()
        if test.is_relative_to(tests_dir)
        and test.name.startswith("test_")
        and (module in seen or any(test.is_relative_to(d) for d in conftest_dirs))
    )


def owned_tests(changed: list[Path], src: Path = SRC) -> list[Path]:
    """Return the test files among *changed* and those named for the modules in *changed*.

    No import hops: a changed ``src/kiss/a/b.py`` selects every
    ``test_b.py`` and ``test_b_*.py`` under ``src/kiss/tests``, a changed
    test file selects itself, and a module with no test named for it
    selects none (named on stderr, so the change can get one).  The
    tests a sub-agent runs on the files it changed in a tree it shares
    with siblings: the depth-1 closure of a core module is already
    hundreds of files, most of them the siblings' concern.

    Args:
        changed: The files the caller changed (absolute, or relative to
            the repo root); files outside *src* or not ``.py`` are ignored.
        src: The source root (default: the repository's ``src/``).
    """
    tests_dir = src / "kiss" / "tests"
    all_tests = [f for f in source_files(tests_dir) if f.name.startswith("test_")]
    by_resolved = {f.resolve(): f for f in all_tests}
    owned: set[Path] = set()
    for path in changed:
        resolved = path.resolve() if path.is_absolute() else (src.parent / path).resolve()
        if resolved.suffix != ".py" or not resolved.is_relative_to(src):
            continue
        if resolved.is_relative_to(tests_dir):
            if resolved in by_resolved:
                owned.add(by_resolved[resolved])
            continue
        stem = resolved.parent.name if resolved.name == "__init__.py" else resolved.stem
        named = [
            t for t in all_tests if t.stem == f"test_{stem}" or t.stem.startswith(f"test_{stem}_")
        ]
        if not named:
            print(f"no test module named for {path} (test_{stem}*.py)", file=sys.stderr)
        owned.update(named)
    return sorted(owned)


def changed_files() -> list[Path]:
    """Return the files changed against ``HEAD`` plus the untracked ones.

    ``[]`` outside git: no ``.git`` at the project root or above it.  Both
    listings name their paths from the project root (``git diff`` would
    name them from the top of the repository, which may be above it), as
    they are on disk (``-z``: unquoted, so a name that is not ASCII is a
    file too).  The listings are read as bytes and each name decoded as
    Python reads file names: one that is not UTF-8 stays that file, and
    so does one with a carriage return, which text mode would turn into
    a line feed.  A rename is listed under both names: the tests that
    still import the old one are the ones that break.

    Raises:
        OSError: When git cannot be started.
        subprocess.SubprocessError: When git times out or fails in the
            repository: no answer must not read as "nothing changed".
    """
    if not any((d / ".git").exists() for d in (PROJECT_ROOT, *PROJECT_ROOT.parents)):
        return []
    names: list[bytes] = []
    for command in (
        ["git", "diff", "--name-only", "-z", "--relative", "--no-renames", "HEAD"],
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
    ):
        done = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, timeout=30, check=False,
        )
        if done.returncode:
            raise subprocess.SubprocessError(
                f"{' '.join(command)} failed: {os.fsdecode(done.stderr).strip()}"
            )
        names += done.stdout.split(b"\0")
    return [PROJECT_ROOT / os.fsdecode(name) for name in names if name]


DEFAULT_TEST_TIMEOUT_SECONDS = 600
"""The ``--timeout`` (pytest-timeout) every split command carries: a test
still running after it fails there instead of holding its split, and
the round waiting on the split, open.  ``pyproject.toml`` keeps
``timeout = 0`` for a plain ``uv run pytest``."""


def split_commands(
    tests: list[Path], splits: int, root: Path = PROJECT_ROOT,
    durations: dict[str, float] | None = None, timeout: int = DEFAULT_TEST_TIMEOUT_SECONDS,
) -> list[str]:
    """Return at most *splits* ``uv run pytest`` commands over *tests*, balanced by duration.

    A test file weighs the seconds *durations* records for it; one with
    no record weighs its size times the median seconds per byte of the
    recorded files (its size alone when nothing is recorded, as the
    balance was before durations were kept).  The longest-weighing files
    are placed first, each into the lightest split so far.  Paths are in
    POSIX form and quoted as shell words: the commands run through the
    Bash tool, which is bash on Windows too, where a backslash would be
    eaten, and a space or a ``;`` in a path would end the argument.

    Args:
        tests: The test files to run.
        splits: The most commands to return (fewer when there are fewer tests).
        root: The repo root the paths are printed relative to.
        durations: Seconds by path relative to *root*
            (:func:`~kiss.scripts.durations.load_durations`).
        timeout: The per-test ``--timeout`` of every command; ``0`` leaves it out.
    """
    known = durations or {}
    seconds = {t: known.get(t.relative_to(root).as_posix()) for t in tests}
    rates = [s / t.stat().st_size for t, s in seconds.items() if s is not None and t.stat().st_size]
    rate = statistics.median(rates) if rates else 1.0
    weights = {t: s if s is not None else t.stat().st_size * rate for t, s in seconds.items()}
    buckets: list[list[Path]] = [[] for _ in range(max(1, min(splits, len(tests))))]
    loads = [0.0] * len(buckets)
    for test in sorted(tests, key=lambda t: (-weights[t], t)):
        index = loads.index(min(loads))
        buckets[index].append(test)
        loads[index] += weights[test]
    flag = f"--timeout={timeout} " if timeout > 0 else ""
    return [
        f"uv run pytest -q -p no:cacheprovider {flag}"
        + shlex.join(t.relative_to(root).as_posix() for t in sorted(bucket))
        for bucket in buckets
        if bucket
    ]


def main(argv: list[str] | None = None) -> int:
    """Print the impacted test files, or the split pytest commands with ``--splits``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "paths", nargs="*", help="changed files (default: git diff HEAD + untracked)"
    )
    parser.add_argument("--splits", type=int, default=0, help="print N pytest commands as JSON")
    parser.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"import hops from a change to a test (default {DEFAULT_DEPTH}; 0 = unbounded)",
    )
    parser.add_argument(
        "--owned", action="store_true",
        help="only the given test files and the tests named for the given modules "
             "(test_<module>*.py), no import hops; needs paths",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TEST_TIMEOUT_SECONDS,
        help=f"per-test timeout of each split (default {DEFAULT_TEST_TIMEOUT_SECONDS} s; 0 = none)",
    )
    args = parser.parse_args(argv)
    if args.owned and not args.paths:
        parser.error("--owned needs the files you changed: in a tree shared with siblings, "
                     "git diff holds their edits too")
    try:
        changed = [Path(p) for p in args.paths] or changed_files()
    except (OSError, subprocess.SubprocessError) as error:
        print(f"impacted-tests: cannot list the changed files: {error}", file=sys.stderr)
        return 1
    tests = owned_tests(changed) if args.owned else impacted_tests(changed, depth=args.depth)
    if args.splits:
        commands = split_commands(
            tests, args.splits, durations=load_durations(durations_path()), timeout=args.timeout,
        )
        print(json.dumps(commands))
    else:
        # Written as bytes, each name as the file system has it: a name
        # that is not UTF-8 is no text a strict stdout would encode.
        listing = "".join(f"{test.relative_to(PROJECT_ROOT).as_posix()}\n" for test in tests)
        sys.stdout.buffer.write(os.fsencode(listing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
