# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Script to run all code quality checks: syntax check, lint, and type check."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
"""Repository root (``src/kiss/scripts/check.py`` is four levels deep)."""


_ERROR_LINE = re.compile(r"error|-->|^\S+:\d+(:\d+)?: |^\s*[A-Z]{1,3}\d{3,4}\b", re.IGNORECASE)
"""Lines worth repeating in the final digest: mypy/pyright ``error:``
lines, ruff's ``--> file:line:col`` locators and ``E501 …`` headers,
eslint/stylelint ``file:line:col`` locations."""

MAX_DIGEST_LINES = 60
"""Cap per failed stage so the digest stays readable when a stage
reports hundreds of errors."""


def error_lines(lines: Iterable[str]) -> list[str]:
    """Return the lines among *lines* that locate or describe an error.

    Consumes *lines* lazily and keeps at most :data:`MAX_DIGEST_LINES`
    of them, so a stage that prints megabytes of output costs no more
    memory than the digest itself.

    Args:
        lines: The combined stdout/stderr lines of one check stage.

    Returns:
        The matching lines in order, capped at :data:`MAX_DIGEST_LINES`
        with a trailing ``... N more`` line when truncated.
    """
    kept: list[str] = []
    extra = 0
    for line in lines:
        if not _ERROR_LINE.search(line):
            continue
        if len(kept) < MAX_DIGEST_LINES:
            kept.append(line.rstrip())
        else:
            extra += 1
    if extra:
        kept.append(f"... {extra} more")
    return kept


def _echo_lines(stream: Iterable[str], quiet: bool) -> Iterator[str]:
    """Pass every line of *stream* on, printing it as it arrives unless *quiet*."""
    for line in stream:
        if not quiet:
            print(line, end="", flush=True)
        yield line


def run_command(cmd: list[str], description: str, quiet: bool = False) -> list[str] | None:
    """Run a command and return its error lines when it fails.

    Args:
        cmd: List of command arguments to execute.
        description: Human-readable description of the command for logging.
        quiet: Print one status line per stage instead of streaming
            the stage's output; the error lines still reach the digest.

    Returns:
        ``None`` when the command exits with code 0, otherwise the
        :func:`error_lines` of its combined stdout/stderr for the
        final digest.
    """
    if not quiet:
        print(f"\n{'=' * 60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 60}\n", flush=True)

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as proc:
        assert proc.stdout is not None
        digest = error_lines(_echo_lines(proc.stdout, quiet))
        returncode = proc.wait()
    if returncode != 0:
        print(f"{'' if quiet else chr(10)}❌ {description} failed with exit code {returncode}")
        return digest
    print(f"{'' if quiet else chr(10)}✅ {description} passed")
    return None


FIXERS: list[tuple[list[str], str]] = [
    (["uv", "run", "ruff", "check", "--fix", "src/"], "Auto-fix lint (ruff check --fix)"),
]
"""The auto-fixers ``--fix`` runs before the checks: what they fix never
reaches the digest, so the digest holds only what a hand must fix.
Only lint fixes: the tree is not ``ruff format``-ed, and running the
formatter would rewrite every file in it."""

CACHE_FILE = PROJECT_ROOT / "tmp" / "check-cache.json"
"""Where the last run's verdict is kept, keyed by a fingerprint of the
sources: a repeat run on an unchanged tree prints the kept digest
instead of spending two minutes to recompute it (the 2026-10-24 audit
counted 16 check runs where six were needed)."""
_SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".git", "tmp", "build", "dist"})
CHECK_CONFIG_FILES = (
    "pyproject.toml",
    "uv.lock",
    "pyrightconfig.json",
    "mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "setup.cfg",
)
"""Root files whose change must invalidate the cached verdict: the
checks read their rules from them."""


def tree_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Return a hash of the path, size and mtime of every source file under *root*.

    Covers ``src/`` and the root files that steer the checks
    (:data:`CHECK_CONFIG_FILES`); caches, ``node_modules`` and ``tmp/``
    are skipped.

    Args:
        root: The repository root.
    """
    digest = hashlib.sha256()
    for name in CHECK_CONFIG_FILES:
        top = root / name
        if top.is_file():
            stat = top.stat()
            digest.update(f"{top.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    src = root / "src"
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def cached_verdict(fingerprint: str, full: bool) -> dict[str, Any] | None:
    """Return the cached ``{"exit_code", "log"}`` of the last run on this tree, or ``None``."""
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("fingerprint") != fingerprint or cached.get("full") != full:
        return None
    return cached


def save_verdict(fingerprint: str, full: bool, exit_code: int, log: str) -> None:
    """Record the verdict of a run on the tree *fingerprint* for :func:`cached_verdict`."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "full": full, "exit_code": exit_code, "log": log}
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def run_checks(
    checks: list[tuple[list[str], str]], quiet: bool = False,
) -> list[tuple[str, list[str]]]:
    """Run every check and return the failed ones with their error lines.

    Every stage runs even after an earlier one fails, so a single
    invocation reports ruff, mypy, pyright and extension errors together
    instead of revealing them one stage per run (each re-run used to cost
    an agent a full step and about a minute).  The only prerequisite is
    ``uv sync``: without the dependencies every later stage would fail
    spuriously, so a failed sync stops the run.

    Args:
        checks: ``(command, description)`` pairs, run in order.
        quiet: Passed to :func:`run_command`.

    Returns:
        ``(description, error_lines)`` for each failed check, in run
        order; empty when everything passed.  The error lines (see
        :func:`error_lines`) let ``main`` repeat every error at the very
        end of the output, where a ``| tail`` still shows them: an agent
        that piped a 700-line run through ``tail -25`` saw only the
        failed stage names and re-ran the whole two-minute check twice.
    """
    failed: list[tuple[str, list[str]]] = []
    for cmd, description in checks:
        digest = run_command(cmd, description, quiet)
        if digest is not None:
            failed.append((description, digest))
            if cmd[:2] == ["uv", "sync"]:
                break
    return failed


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped by consulting ``.gitignore`` rules.

    Uses ``git check-ignore`` to determine whether *path* (or any ancestor)
    is covered by ``.gitignore``.  The ``.git`` directory itself is always
    skipped because it is not listed in ``.gitignore`` but must never be
    cleaned.

    Args:
        path: Path object to check.

    Returns:
        True if the path is inside ``.git`` or is git-ignored, False otherwise.
    """
    if ".git" in path.parts:
        return True
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        capture_output=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


def clean_build_artifacts() -> None:
    """Remove build artifacts and caches.

    Removes directories like build/, dist/, .mypy_cache/, .pytest_cache/,
    .ruff_cache/, *.egg-info/, all __pycache__ directories, and all .pyc files.
    Skips paths inside .venv, venv, .git, and node_modules directories.

    Equivalent to:
        rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache && \
        find . -type d -name __pycache__ -exec rm -r {} + && \
        find . -type f -name "*.pyc" -delete

    Returns:
        None.
    """
    dirs_to_remove = [
        "dist",
        "build",
        "*.egg-info",
        ".claude",
        "artifacts",
        "uv_cache",
    ]

    print(f"\n{'=' * 60}")
    print("Running: Clean build artifacts")
    print(f"{'=' * 60}\n")

    removed_count = 0

    for pattern in dirs_to_remove:
        if "*" in pattern:
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_dir():
                    print(f"  Removing: {path}")
                    shutil.rmtree(path)
                    removed_count += 1
        else:
            path = PROJECT_ROOT / pattern
            if path.exists() and path.is_dir():
                print(f"  Removing: {path}")
                shutil.rmtree(path)
                removed_count += 1

    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir() and not _should_skip_path(pycache):
            print(f"  Removing: {pycache}")
            shutil.rmtree(pycache)
            removed_count += 1

    for pyc_file in PROJECT_ROOT.rglob("*.pyc"):
        if not _should_skip_path(pyc_file):
            print(f"  Removing: {pyc_file}")
            pyc_file.unlink()
            removed_count += 1

    if removed_count == 0:
        print("  No artifacts to clean.")
    else:
        print(f"\n✅ Cleaned {removed_count} items")


def main() -> int:
    """Run all code quality checks.

    Parses command-line arguments and runs code quality checks including
    dependency installation, syntax checking with py_compile, linting with ruff,
    and type checking with mypy.

    Returns:
        Exit code 0 if all checks pass, 1 if any check fails.
    """
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning build artifacts before running checks",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts, do not run checks",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run additional checks including pyright type checking",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Run the auto-fixer (ruff check --fix) first, print one "
            "status line per stage and only the residual errors at the end; a repeat "
            "run on an unchanged tree prints the previous verdict"
        ),
    )
    args = parser.parse_args()

    if not args.no_clean:
        clean_build_artifacts()
        if args.clean_only:
            return 0

    fingerprint = ""
    if args.fix:
        for cmd, description in FIXERS:
            run_command(cmd, description, quiet=True)
        fingerprint = tree_fingerprint()
        cached = cached_verdict(fingerprint, args.full)
        if cached is not None:
            print("(unchanged tree: the verdict of the previous run follows)")
            print(cached["log"], end="")
            return int(cached["exit_code"])

    checks = [
        (["uv", "sync"], "Install dependencies (uv sync)"),
        (["uv", "run", "generate-api-docs"], "Generate API docs"),
        (["uv", "run", "sea", "lint"], "Lint SEAs (sea lint)"),
        (["uv", "run", "sea", "docs"], "Generate SEA vocabulary tables (sea docs)"),
        (["uv", "run", "python", "-m", "compileall", "-q", "src/"], "Syntax check (compileall)"),
        (["uv", "run", "ruff", "check", "src/"], "Lint code (ruff)"),
        (["uv", "run", "mypy", "src/"], "Type check (mypy)"),
    ]

    if args.full:
        checks.append((["uv", "run", "pyright", "src/"], "Type check (pyright)"))
        webapp_dir = PROJECT_ROOT / "src" / "kiss" / "webapp"
        if (webapp_dir / "node_modules").is_dir() and shutil.which("npm"):
            checks.append(
                (["npm", "--prefix", str(webapp_dir), "run", "lint"], "Webapp lint")
            )

    print("\n🔍 Running all code quality checks...\n")

    failed = run_checks(checks, quiet=args.fix)
    verdict = verdict_text(failed)
    print(verdict, end="")
    if args.fix:
        save_verdict(fingerprint, args.full, 1 if failed else 0, verdict)
    return 1 if failed else 0


def verdict_text(failed: list[tuple[str, list[str]]]) -> str:
    """Return the closing report: the failed stages and their error lines, or the all-clear.

    Args:
        failed: What :func:`run_checks` returned.
    """
    if not failed:
        return "\n" + "=" * 60 + "\n✅ All checks passed!\n" + "=" * 60 + "\n\n"
    lines = ["", "=" * 60, "❌ Some checks failed. Please fix the errors above."]
    lines += [f"   ❌ {description}" for description, _lines in failed]
    lines += ["=" * 60, "Errors, repeated from each failed stage:"]
    for description, error_list in failed:
        lines += ["", f"--- {description} ---", *error_list]
    lines += ["=" * 60, "", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
