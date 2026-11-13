# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Script to run all code quality checks: syntax check, lint, and type check."""

import argparse
import contextlib
import errno
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
"""Repository root (``src/kiss/scripts/check.py`` is four levels deep)."""
#: The directory under a repository that holds the tasks' git worktrees
#: (:data:`kiss.agents.sorcar.git_worktree._WORKTREE_SUBDIR`; spelled here so
#: the check script imports nothing of the agent).
WORKTREES_DIR = ".kiss-worktrees"


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


STAGE_TIMEOUT = 1800.0
"""Seconds a stage may run before it is killed and counted as failed: far
longer than the slowest healthy stage (a cold mypy or pyright pass takes
minutes), so that only a stage that hangs meets it."""

_PIPE_GRACE = 5.0
"""Seconds of waiting for more output of a stage that was stopped: what
it had written arrives at once, and only a process that left its group (a
daemon in a session of its own) can hold the pipe open any longer."""

_POLL = 0.1
"""Seconds between two looks at a running stage: has it exited, is it late."""

_LEASH = (
    "/bin/sh", "-c",
    'exec 3<&0 </dev/null; ( (read _ <&3; kill -9 0) >/dev/null 2>&1 & ); exec "$@" 3<&-', "sh",
)
"""What a stage's command is prefixed with on POSIX: a watcher in the
stage's process group kills that group once the stage's stdin pipe (the
leash, moved to fd 3) closes.  :func:`_stop_stage` closes it, and the
kernel does when ``check`` itself is killed, so no stage outlives it.
A subshell that exits at once starts the watcher, so the stage has no
child it did not start and may wait until it has none left."""


def _pump(stream: IO[str], lines: queue.Queue[str | None]) -> None:
    """Queue every line of *stream*, then ``None``: the body of a stage's reader thread.

    The pipe ends only once every process holding its other end lets
    go, so the thread may outlive the stage; whoever takes the lines
    waits for the queue, never for the pipe, and shuts the queue down
    when it stops listening: the thread then ends at its next line
    instead of waiting for room.  The stream is closed here, by its only
    reader.
    """
    with contextlib.suppress(queue.ShutDown), stream:
        try:
            for line in stream:
                lines.put(line)
        finally:
            lines.put(None)


def _stop_stage(proc: subprocess.Popen[str]) -> None:
    """Kill the stage *proc* and, on POSIX, all it started; a no-op once they are gone."""
    if proc.stdin is not None:
        proc.stdin.close()  # the leash: its watcher kills the stage's whole group
    proc.kill()  # the stage itself: all there is on Windows, or should the watcher be gone


def _stage_lines(
    proc: subprocess.Popen[str], lines: queue.Queue[str | None], timeout: float,
    timed_out: threading.Event,
) -> Iterator[str]:
    """Yield the output lines of the stage *proc* as they arrive, then stop the stage.

    *lines* is the queue :func:`_pump` fills from the stage's pipe.
    The stage is over when its process exits, or after *timeout* seconds
    (*timed_out* is then set), not when its pipe closes: a child it left
    behind may hold the pipe.  :func:`_stop_stage` then closes the pipe,
    unless a process outside the stage's group holds it; that one is
    given up after :data:`_PIPE_GRACE` seconds.  The grace counts only
    the time spent waiting for a line, so a caller that is slow to take
    the lines loses none of them.
    """
    deadline = time.monotonic() + timeout
    look = 0.0  # when to look at the stage next
    line: str | None = ""  # None once the pipe has closed
    while line is not None:
        now = time.monotonic()
        if now >= look:
            if proc.poll() is not None or now >= deadline:
                break
            look = now + _POLL
        try:
            line = lines.get(timeout=_POLL)
        except queue.Empty:
            continue
        if line is not None:
            yield line
    else:  # the pipe closed first: only the exit is left to wait for
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(max(0.0, deadline - time.monotonic()))
    if proc.poll() is None:  # still running now that the deadline has passed
        timed_out.set()
    _stop_stage(proc)
    grace = _PIPE_GRACE
    while line is not None and grace > 0:
        waiting_since = time.monotonic()
        try:
            line = lines.get(timeout=grace)
        except queue.Empty:
            break
        grace -= time.monotonic() - waiting_since
        if line is not None:
            yield line


def run_command(
    cmd: list[str], description: str, quiet: bool = False, timeout: float = STAGE_TIMEOUT,
) -> list[str] | None:
    """Run a command and return its error lines when it fails.

    The command runs at the project root (the stages name ``src/``
    relatively, and the cached verdict is of that tree, wherever
    ``check`` was started), in a session of its own with an empty stdin,
    and is over when its process exits: whatever it left running is
    killed then.  A command still running after *timeout* seconds is
    killed the same way and fails.

    Args:
        cmd: List of command arguments to execute.
        description: Human-readable description of the command for logging.
        quiet: Print one status line per stage instead of streaming
            the stage's output; the error lines still reach the digest.
        timeout: Seconds the command may run.

    Returns:
        ``None`` when the command exits with code 0, otherwise the
        :func:`error_lines` of its combined stdout/stderr for the
        final digest, ending with a ``timed out`` line when *timeout*
        ended the command.
    """
    if not quiet:
        print(f"\n{'=' * 60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 60}\n", flush=True)

    leashed = os.name == "posix"
    if leashed and shutil.which(cmd[0]) is None:  # behind the leash it would be exit code 127
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cmd[0])
    proc = subprocess.Popen(
        [*_LEASH, *cmd] if leashed else cmd,
        stdin=subprocess.PIPE if leashed else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        cwd=PROJECT_ROOT, env=stage_env(), start_new_session=leashed,
    )
    lines: queue.Queue[str | None] = queue.Queue(maxsize=1024)  # full: the stage waits
    timed_out = threading.Event()
    try:
        threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
        digest = error_lines(_echo_lines(_stage_lines(proc, lines, timeout, timed_out), quiet))
    finally:
        _stop_stage(proc)
        lines.shutdown(immediate=True)  # nobody listens any more: see _pump
        returncode = proc.wait()
    if returncode != 0:
        if timed_out.is_set():
            digest.append(f"timed out: still running after {timeout:g} s, killed")
        print(f"{'' if quiet else chr(10)}❌ {description} failed with exit code {returncode}")
        return digest
    print(f"{'' if quiet else chr(10)}✅ {description} passed")
    return None


def in_worktree(root: Path = PROJECT_ROOT) -> bool:
    """Whether the tree at *root* is a task's git worktree (a checkout under ``.kiss-worktrees``).

    Args:
        root: The project root the check runs on.
    """
    return WORKTREES_DIR in root.resolve().parts


def main_checkout(root: Path = PROJECT_ROOT) -> Path:
    """The repository a worktree at *root* was made from; *root* itself outside a worktree.

    Args:
        root: The project root the check runs on.
    """
    parts = root.resolve().parts
    return Path(*parts[:parts.index(WORKTREES_DIR)]) if WORKTREES_DIR in parts else root


def uv_run(*command: str, root: Path = PROJECT_ROOT) -> list[str]:
    """The ``uv run`` that runs *command* in the environment of the tree at *root*.

    ``uv run`` syncs the project environment before the command, which
    inside a worktree would install the worktree as the environment's
    editable ``kiss`` package (see :func:`sync_command`): there the
    command runs with ``--no-sync`` on the environment as it is.

    Args:
        command: The command and its arguments.
        root: The project root the check runs on.
    """
    return ["uv", "run", *(["--no-sync"] if in_worktree(root) else []), *command]


def no_sync(command: list[str]) -> list[str]:
    """Return *command* with ``--no-sync`` when it is a ``uv run`` that would sync first.

    Args:
        command: The command of a stage.
    """
    if command[:2] == ["uv", "run"] and "--no-sync" not in command:
        return ["uv", "run", "--no-sync", *command[2:]]
    return command


def project_script(script: str, module: str, *args: str, root: Path = PROJECT_ROOT) -> list[str]:
    """The command running the project's console *script*, or its *module* in a worktree.

    A worktree's environment has no install of the worktree's project
    (:func:`sync_command`), so there the script runs as ``python -m
    <module>`` and imports the worktree's ``src/`` (:func:`stage_env`).

    Args:
        script: The console script's name (``sea``).
        module: The module its entry point lives in
            (``kiss.agents.sorcar.sea_cli``).
        args: The script's arguments.
        root: The project root the check runs on.
    """
    if in_worktree(root):
        return uv_run("python", "-m", module, *args, root=root)
    return uv_run(script, *args, root=root)


def stage_env(root: Path = PROJECT_ROOT) -> dict[str, str] | None:
    """The environment the stages run with inside a worktree; ``None`` (the inherited one) outside.

    A worktree has no environment of its own, and uv would build it one
    (every dependency again, on whatever Python it finds newest, 3.14
    next to a 3.13 daemon) rather than use the main checkout's: the
    stages run in that checkout's ``.venv`` (``UV_PROJECT_ENVIRONMENT``,
    unless the caller set one) and import the worktree's sources, its
    ``src/`` and ``projects/`` first on ``PYTHONPATH``, ahead of the
    editable install the environment holds of the main checkout.

    Args:
        root: The project root the check runs on.
    """
    if not in_worktree(root):
        return None
    inherited = os.environ.get("PYTHONPATH", "")
    sources = [str(root / "src"), str(root / "projects")] + ([inherited] if inherited else [])
    environment = os.environ.get("UV_PROJECT_ENVIRONMENT") or str(main_checkout(root) / ".venv")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sources),
            "UV_PROJECT_ENVIRONMENT": environment}


FIXERS: list[tuple[list[str], str]] = [
    (uv_run("ruff", "check", "--fix", "src/"), "Auto-fix lint (ruff check --fix)"),
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
    "website/kisssorcar.github.io/docs/sea-commands.md",
    "website/kisssorcar.github.io/docs/cli.md",
)
"""Files outside the source trees whose change must invalidate the cached
verdict: the root files the checks read their rules from, and the pages
``sea lint`` and ``sea docs`` read (``sea_docs.GENERATED_FILES``, spelled
here like :data:`WORKTREES_DIR`; a test holds the two lists together)."""


def tree_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Return a hash of the path, size and mtime of every source file under *root*.

    Covers the source trees ``src/`` and ``projects/`` (the type checkers
    follow the imports of ``src/`` into the packages of ``projects/``) and
    the files outside them that the checks read
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
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    for tree in ("src", "projects"):
        for dirpath, dirnames, filenames in os.walk(root / tree):
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
    """Record the verdict of a run on the tree *fingerprint* for :func:`cached_verdict`.

    The verdict is written to a file of this call's own and renamed over
    the cache, so checks that end together in one tree each publish a
    whole document.  Written into the cache itself, a short verdict kept
    the tail of a longer one saved between its open and its write: no
    JSON, and the next run checked the unchanged tree again.  (Staged
    here with the standard library alone, like the rest of the script:
    it must run on a tree whose modules do not import.)
    """
    verdict = {"fingerprint": fingerprint, "full": full, "exit_code": exit_code, "log": log}
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=CACHE_FILE.parent, prefix=f"{CACHE_FILE.name}.",
            delete_on_close=False,  # removed when the block ends, if the rename did not take it
        ) as staged:
            json.dump(verdict, staged)
            staged.close()
            os.replace(staged.name, CACHE_FILE)
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


def _removed_meanwhile(function: object, path: object, error: BaseException) -> None:
    """``shutil.rmtree`` error handler: pass over what is gone, raise anything else.

    Checks started together in one tree remove the same directories; a
    file another of them removed first is removed all the same.
    """
    if not isinstance(error, FileNotFoundError):
        raise error


def clean_build_artifacts() -> None:
    """Remove the build artifacts at the repository root.

    Equivalent to ``rm -rf dist build *.egg-info artifacts uv_cache``.
    """
    print(f"\n{'=' * 60}")
    print("Running: Clean build artifacts")
    print(f"{'=' * 60}\n")

    removed_count = 0
    for pattern in ("dist", "build", "*.egg-info", "artifacts", "uv_cache"):
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                print(f"  Removing: {path}")
                shutil.rmtree(path, onexc=_removed_meanwhile)
                removed_count += 1

    if removed_count == 0:
        print("  No artifacts to clean.")
    else:
        print(f"\n✅ Cleaned {removed_count} items")


def check_stages(
    full: bool, read_only: bool, root: Path = PROJECT_ROOT
) -> list[tuple[list[str], str]]:
    """Return the stages of a run as ``(command, description)`` pairs, in order.

    Args:
        full: Whether pyright and the webapp lint run as well.
        read_only: Whether to leave out the stages that change the tree or
            the environment (``uv sync``, the API docs, the SEA tables) and
            to run the others on the environment as it is (``uv run
            --no-sync``), so that the run can go beside the tests.
        root: The tree the stages check.
    """
    stages: list[tuple[list[str], str]] = []
    if not read_only:
        stages += [
            (sync_command(root), "Install dependencies (uv sync)"),
            (project_script("generate-api-docs", "kiss.scripts.generate_api_docs", root=root),
             "Generate API docs"),
        ]
    stages.append(
        (project_script("sea", "kiss.agents.sorcar.sea_cli", "lint", root=root),
         "Lint SEAs (sea lint)")
    )
    if not read_only:
        stages.append(
            (project_script("sea", "kiss.agents.sorcar.sea_cli", "docs", root=root),
             "Generate SEA vocabulary tables (sea docs)")
        )
    stages += [
        (uv_run("python", "-m", "compileall", "-q", "src/", root=root),
         "Syntax check (compileall)"),
        (uv_run("ruff", "check", "src/", root=root), "Lint code (ruff)"),
        (uv_run("mypy", "src/", root=root), "Type check (mypy)"),
    ]
    if full:
        stages.append((uv_run("pyright", "src/", root=root), "Type check (pyright)"))
        webapp_dir = root / "src" / "kiss" / "webapp"
        if (webapp_dir / "node_modules").is_dir() and shutil.which("npm"):
            stages.append(
                (["npm", "--prefix", str(webapp_dir), "run", "lint"], "Webapp lint")
            )
    if read_only:
        stages = [(no_sync(command), description) for command, description in stages]
    return stages


def main() -> int:
    """Run all code quality checks on the tree this script is part of.

    Cleans the build artifacts, then runs every stage (dependency sync,
    the generated docs, ``sea lint``, ``compileall``, ruff, mypy; with
    ``--full`` pyright and the webapp lint as well) and prints the
    verdict with the error lines of each failed stage.  ``--read-only``
    neither cleans nor syncs nor generates docs: it only reads the
    sources, for a run beside the tests.

    Returns:
        Exit code 0 if all checks pass, 1 if any check fails.
    """
    parser = argparse.ArgumentParser(description="Run code quality checks")
    cleaning = parser.add_mutually_exclusive_group()
    cleaning.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning build artifacts before running checks",
    )
    cleaning.add_argument(
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
    parser.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Change neither the tree nor the environment: no cleaning, no uv sync, "
            "no generated docs, no --fix; for a run beside the tests"
        ),
    )
    args = parser.parse_args()
    if args.read_only and (args.fix or args.clean_only):
        parser.error("--read-only changes nothing: it goes with neither --fix nor --clean-only")

    if not args.no_clean and not args.read_only:
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

    print("\n🔍 Running all code quality checks...\n")

    failed = run_checks(check_stages(args.full, args.read_only), quiet=args.fix)
    verdict = verdict_text(failed)
    print(verdict, end="")
    if args.fix:
        save_verdict(fingerprint, args.full, 1 if failed else 0, verdict)
    return 1 if failed else 0


def sync_command(root: Path = PROJECT_ROOT) -> list[str]:
    """The ``uv sync`` that installs the dependencies of the tree at *root*.

    Inside a task's git worktree (a checkout under ``.kiss-worktrees``)
    the project itself is not installed: the environment is the main
    checkout's (:func:`stage_env`), and an editable install of the
    worktree would re-point its ``kiss`` package at a directory that is
    removed when the worktree is merged (the 2026-11-01 audit counted 19
    such runs).  The dependencies alone are synced, ``--inexact`` so
    that the main checkout's install, extraneous to a sync that skips
    the project, is not removed either (``uv sync --no-install-project
    --dry-run`` on the shared environment says "Would uninstall
    kiss-agent-framework" without it); the later stages read the
    worktree's files by path.

    Args:
        root: The project root the check runs on.
    """
    if in_worktree(root):
        return ["uv", "sync", "--inexact", "--no-install-project"]
    return ["uv", "sync"]


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
