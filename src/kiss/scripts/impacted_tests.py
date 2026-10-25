# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""List the test files whose import graph reaches the files changed in the working tree.

``uv run impacted-tests`` prints one test path per line: every test
module under ``src/kiss/tests`` that imports, directly or through other
``kiss`` modules, a changed source file (``git diff --name-only HEAD``
plus untracked files, or the paths given on the command line).  A
changed test file is impacted by itself.  ``--splits N`` prints instead
the JSON list of ``N`` balanced ``uv run pytest`` commands for
``run_commands_parallel``.

The 2026-10-24 trajectory audit found mid-task test runs selected by
directory globs (one of them 58 CPU-minutes for a one-module change)
and by guesswork; the import graph names the tests that can actually
see a change.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
TESTS = SRC / "kiss" / "tests"
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

    Args:
        changed: Changed files (absolute, or relative to the repo root);
            files outside *src* or not ``.py`` are ignored.
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
        for name in imported_modules(path, src):
            importers[name].add(_module_name(path, src))
    roots = []
    for path in changed:
        resolved = path.resolve() if path.is_absolute() else (src.parent / path).resolve()
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
        for m in seen
        if (test := by_module.get(m)) is not None
        and test.is_relative_to(tests_dir)
        and test.name.startswith("test_")
    )


def changed_files() -> list[Path]:
    """Return the files changed against ``HEAD`` plus the untracked ones (``[]`` outside git)."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=30, check=False,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [PROJECT_ROOT / line for line in (diff + untracked).splitlines() if line.strip()]


def split_commands(tests: list[Path], splits: int, root: Path = PROJECT_ROOT) -> list[str]:
    """Return at most *splits* ``uv run pytest`` commands over *tests*, balanced by file size.

    Paths are in POSIX form: the commands run through the Bash tool,
    which is bash on Windows too, where a backslash would be eaten.
    """
    buckets: list[list[Path]] = [[] for _ in range(max(1, min(splits, len(tests))))]
    sizes = [0] * len(buckets)
    for test in sorted(tests, key=lambda t: (-t.stat().st_size, t)):
        index = sizes.index(min(sizes))
        buckets[index].append(test)
        sizes[index] += test.stat().st_size
    return [
        "uv run pytest -q -p no:cacheprovider "
        + " ".join(t.relative_to(root).as_posix() for t in sorted(bucket))
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
    args = parser.parse_args(argv)
    changed = [Path(p) for p in args.paths] or changed_files()
    tests = impacted_tests(changed, depth=args.depth)
    if args.splits:
        print(json.dumps(split_commands(tests, args.splits)))
    else:
        for test in tests:
            print(test.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
