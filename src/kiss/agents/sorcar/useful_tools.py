# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Useful tools for agents: file editing and bash execution."""

import difflib
import functools
import hashlib
import logging
import mimetypes
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # POSIX has no msvcrt
    msvcrt = None  # type: ignore[assignment]

from kiss.agents.sorcar._concurrency import _fcntl as fcntl
from kiss.agents.sorcar.fanout_guard import parse_tasks_json
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SLUG_PREFIX,
    _WORKTREE_SUBDIR,
)
from kiss.core import tool_interrupt
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model import (
    READ_TOOL_BINARY_MIME_TYPES,
    encode_binary_attachment,
)

logger = logging.getLogger(__name__)

_MAX_BINARY_READ_BYTES = 20 * 1024 * 1024
_OUTLINE_SYMBOL_RE = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+\w+|class\s+\w+"  # Python
    r"|(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+\w+|class\s+\w+)"  # JS/TS
    r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)\s*=>|function\b)"
    r"|(?:pub\s+)?(?:fn|struct|enum|impl|trait)\s+\w+"  # Rust
    r"|func\s+(?:\([^)]*\)\s*)?\w+"  # Go
    r")"
)
_OUTLINE_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})
_OUTLINE_MAX_ENTRIES = 400
_OUTLINE_MIN_ENTRIES = 5


@contextmanager
def _file_lock(lock_path: Path, blocking: bool = True) -> Any:
    """Hold an exclusive advisory inter-process lock on *lock_path*.

    Serializes check-then-use sequences on resources shared by every
    kiss process on the machine — MCP configs and OAuth token stores,
    the cron job store, and the Chromium profile directory — across
    daemons, CLI runs, channel-agent processes, and event loops.  A
    ``threading`` lock cannot do this: the resources live on disk, not
    in one process.  The lock file itself is created mode ``0600``.

    Cross-platform: ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
    Windows (where ``fcntl`` does not exist — an unconditional import
    would make every dependent feature unavailable there).  When
    neither primitive exists the lock degrades to a best-effort no-op
    rather than breaking the caller entirely.

    Args:
        lock_path: The lock file to hold; parent directories are created.
        blocking: Whether to wait for the lock.  ``False`` gives up
            immediately when another process holds it (the cron
            scheduler's overlapping-tick skip) instead of waiting.

    Yields:
        ``True`` while the lock is held (including the degraded no-op
        case), or ``None`` when *blocking* is ``False`` and another
        process holds the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if fcntl is not None:
            flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, flags)
                locked = True
            except BlockingIOError:
                pass
        elif msvcrt is not None:  # pragma: no cover — Windows-only branch
            os.lseek(descriptor, 0, os.SEEK_SET)
            if blocking:
                while True:
                    try:
                        msvcrt.locking(  # pyright: ignore[reportAttributeAccessIssue]
                            descriptor,
                            msvcrt.LK_LOCK,  # pyright: ignore[reportAttributeAccessIssue]
                            1,
                        )
                        locked = True
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                try:
                    msvcrt.locking(  # pyright: ignore[reportAttributeAccessIssue]
                        descriptor,
                        msvcrt.LK_NBLCK,  # pyright: ignore[reportAttributeAccessIssue]
                        1,
                    )
                    locked = True
                except OSError:
                    pass
        else:  # pragma: no cover — platform without either primitive
            locked = True
        yield True if locked else None
    finally:
        try:
            if locked and fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif locked and msvcrt is not None:  # pragma: no cover — Windows-only branch
                os.lseek(descriptor, 0, os.SEEK_SET)
                with suppress(OSError):
                    msvcrt.locking(  # pyright: ignore[reportAttributeAccessIssue]
                        descriptor,
                        msvcrt.LK_UNLCK,  # pyright: ignore[reportAttributeAccessIssue]
                        1,
                    )
        finally:
            os.close(descriptor)


def _worktree_index(parts: tuple[str, ...]) -> int | None:
    """Return the index of the worktree marker segment in *parts*.

    The kiss worktree layout is ``<repo>/<subdir>/<prefix><slug>/...``
    (see ``git_worktree``, which owns both constants and creates the
    directories).  Every path guard in this module needs the same scan,
    so it lives here once: duplicating the literals would let a rename
    in ``git_worktree`` silently turn all of them into dead code, and
    the agent would write straight into the user's main checkout.

    When *parts* runs through several nested worktrees (a repo that
    itself lives inside another repo's ``.kiss-worktrees/kiss_wt-*``
    directory, e.g. a project checked out under a kiss worktree, or
    the test suite running from inside one), the *innermost* marker
    is returned: the worktree closest to the path is the one whose
    parent repo the agent is actually working on.  Matching the
    outermost marker instead would make every guard in this module
    treat the outer checkout as "the" parent repo and silently skip
    remap/fallback for the nested one.

    Args:
        parts: ``Path.parts`` of the path (or work dir) to inspect.

    Returns:
        The largest index ``i`` such that ``parts[i]`` is the worktree
        subdir and ``parts[i + 1]`` is a worktree slug, or ``None``
        when *parts* does not run through a worktree.
    """
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == _WORKTREE_SUBDIR and parts[i + 1].startswith(
            _WORKTREE_SLUG_PREFIX
        ):
            return i
    return None


def _stale_worktree_fallback(resolved: Path) -> Path | None:
    """If *resolved* lives under a now-deleted ``.kiss-worktrees/kiss_wt-*``
    directory, return the equivalent path with that worktree segment
    stripped (i.e. relative to the parent repo).

    Worktrees are torn down on autocommit / success, so a model that
    remembers a worktree path from earlier in the task ends up with a
    dangling path.  Returning the equivalent in-repo path lets the
    subsequent read succeed transparently.

    A worktree whose root directory still exists on disk is *live*,
    not stale: its working tree is authoritative for paths under it
    (it may have deleted or diverged from the parent repo's copy), so
    no fallback applies and ``None`` is returned.
    """
    parts = resolved.parts
    i = _worktree_index(parts)
    if i is None:
        return None
    if Path(*parts[: i + 2]).is_dir():
        return None
    return Path(*parts[:i], *parts[i + 2 :])


def _active_worktree_remap(resolved: Path, work_dir: str | None) -> Path | None:
    """If *work_dir* lives inside a live ``.kiss-worktrees/kiss_wt-*`` worktree
    and *resolved* points to a file in the parent repo (outside any
    worktree), return the equivalent path *inside* the active worktree.
    Otherwise return ``None``.

    This is the symmetric inverse of :func:`_stale_worktree_fallback`:
    while a worktree is *live*, the agent's tool calls must operate on
    the worktree's working tree even when the model emits an absolute
    path that points at the parent repo.  Without this remap, an LLM
    that ignores the ``Work dir:`` hint and edits e.g.
    ``/abs/repo/README.md`` would mutate the user's main checkout,
    leave the worktree clean (so the framework's auto-commit finds
    nothing), and skip the squash-merge entirely — i.e. the
    "why didn't you run the last task in worktree and why didn't you
    commit the changes?" failure mode.

    The remap is structural — it only inspects path strings
    (``.kiss-worktrees/kiss_wt-*`` segments), so it works regardless
    of whether the worktree is currently registered with git.  The
    one exception is a worktree directory that no longer exists on
    disk (torn down by a concurrent cleanup/discard/merge): remapping
    into it would dead-end every Read/Edit ("File not found" for
    files that DO exist in the parent repo) and make Write resurrect
    a zombie worktree directory whose contents are never merged.  In
    that case no remap applies — mirroring the vanished-worktree
    fallback in ``UsefulTools._spawn``, which runs Bash commands from
    the parent repo root in the same situation.

    Args:
        resolved: An already-resolved absolute path the caller is about
            to read/write/edit.
        work_dir: The agent's working directory (usually inside the
            worktree).  May be ``None`` (no worktree → no remap).

    Returns:
        The remapped worktree-internal path, or ``None`` when no remap
        applies (no active worktree in *work_dir*, *resolved* is not
        under the parent repo, or *resolved* is already inside a
        worktree).
    """
    if not work_dir:
        return None
    work_parts = Path(work_dir).resolve().parts
    i = _worktree_index(work_parts)
    if i is None:
        return None
    main_repo_parts = work_parts[:i]
    wt_root_parts = work_parts[: i + 2]
    if not Path(*wt_root_parts).is_dir():
        return None
    res_parts = resolved.parts
    if (
        len(res_parts) <= len(main_repo_parts)
        or res_parts[: len(main_repo_parts)] != main_repo_parts
    ):
        return None
    tail = res_parts[len(main_repo_parts):]
    if tail and tail[0] == _WORKTREE_SUBDIR:
        return None
    return Path(*wt_root_parts, *tail)


def _absolutize(file_path: str, work_dir: str | None) -> str:
    """Anchor a bare relative path under *work_dir* before resolving.

    ``Path("README.md").resolve()`` uses the *host process's*
    ``os.getcwd()`` — not the agent's ``work_dir``.  When the agent
    is running inside a ``.kiss-worktrees/kiss_wt-*`` worktree but
    the host process was launched somewhere unrelated (e.g. the VS
    Code extension's own directory), a relative path emitted by the
    LLM would resolve to ``<unrelated>/README.md`` and bypass the
    worktree entirely — re-creating the original "task didn't run
    in worktree / no commit" failure mode.

    Joining the relative path under ``work_dir`` first makes the
    subsequent ``_active_worktree_remap`` (and the plain resolution)
    behave consistently regardless of host cwd.

    ``~``-prefixed paths are expanded to the user's home directory
    first (matching the shell semantics the Bash tool already has);
    without this, ``Write("~/notes.txt", ...)`` would silently create
    a directory literally named ``~`` under *work_dir*.
    """
    try:
        p = Path(file_path).expanduser()
    except RuntimeError:
        p = Path(file_path)
    if p.is_absolute():
        return str(p)
    if not work_dir:
        return file_path
    return str(Path(work_dir) / p)


def _bash_parent_repo_guard(command: str, work_dir: str | None) -> str | None:
    """Refuse a Bash command that targets the parent repo's working tree.

    When the agent is running inside a ``.kiss-worktrees/kiss_wt-*``
    worktree, shell commands that hard-code an absolute path under
    the *parent* repo's working tree (e.g.
    ``echo X > /abs/repo/README.md``, ``sed -i ... /abs/repo/X``,
    ``rm /abs/repo/X``) silently mutate the user's main checkout,
    leave the worktree clean, and skip the framework's auto-commit
    — exactly the bug that prompted the worktree-path remap for
    Read/Write/Edit.  The Bash tool can't do a clean rewrite (shell
    strings are unstructured), so we refuse the command with an
    actionable error pointing the model at the worktree path.

    The guard only kicks in when:

    * ``work_dir`` is inside a live ``.kiss-worktrees/kiss_wt-*``
      worktree, and
    * the command literally contains the *parent-repo* absolute path
      prefix (not the worktree's prefix — that's a legitimate write
      inside the worktree).

    Args:
        command: The Bash command line the model wants to run.
        work_dir: The agent's working directory.

    Returns:
        An actionable error string to return to the model in place
        of running the command, or ``None`` when the command is
        allowed to proceed.
    """
    if not work_dir:
        return None
    checked: set[tuple[str, ...]] = set()
    for wd_parts in (Path(work_dir).parts, Path(work_dir).resolve().parts):
        if wd_parts in checked:
            continue
        checked.add(wd_parts)
        err = _parent_repo_guard_for_parts(command, wd_parts)
        if err is not None:
            return err
    return None


def _worktree_roots(work_dir: str | None) -> list[tuple[str, str]]:
    """Return ``(main_repo, wt_root)`` for each spelling of an active worktree *work_dir*.

    Empty when *work_dir* is not inside a live ``.kiss-worktrees/kiss_wt-*``
    worktree.  Both the given and the resolved spelling are returned when
    they differ (symlinked checkouts).
    """
    if not work_dir:
        return []
    roots: list[tuple[str, str]] = []
    for wd_parts in (Path(work_dir).parts, Path(work_dir).resolve().parts):
        i = _worktree_index(wd_parts)
        if i is None:
            continue
        pair = (str(Path(*wd_parts[:i])), str(Path(*wd_parts[: i + 2])))
        if pair not in roots and os.path.isdir(pair[1]):
            roots.append(pair)
    return roots


def rewrite_parent_repo_paths(text: str, work_dir: str | None) -> str:
    """Rewrite absolute parent-repo paths in *text* to the active worktree's.

    Sub-agent tasks written by a parent running in a worktree kept
    naming files under the parent repository (134 refused Bash commands
    in the 7-day audit of 2026-09-19).  Every ``<main_repo>/...`` path
    that is not already inside a ``.kiss-worktrees`` directory becomes
    ``<wt_root>/...``; the bare repo path itself is rewritten too.

    The rewrite is textual: prose that deliberately names the parent
    repository ("do not touch <main_repo>") is rewritten too.  A parent
    that needs the child to act on the main checkout must say so
    without an absolute path (the worktree is the only tree a
    sub-agent's tools may modify anyway).

    Args:
        text: A task description or shell command.
        work_dir: The dispatching agent's working directory.

    Returns:
        *text* with the paths rewritten, or unchanged when *work_dir*
        is not inside a live worktree.
    """
    for main_repo, wt_root in _worktree_roots(work_dir):
        pattern = re.escape(main_repo) + r"(?=[/\\]|[\s'\";|&<>()`,:]|$)"
        text = re.sub(pattern, functools.partial(_swap_root, wt_root=wt_root), text)
    return text


def _swap_root(match: re.Match[str], wt_root: str) -> str:
    """Replace one matched repo root unless it already names a worktree."""
    rest = match.string[match.end():]
    if rest[:1] in ("/", "\\") and rest[1:].startswith(".kiss-worktrees"):
        return match.group(0)
    return wt_root


def _parent_repo_guard_for_parts(
    command: str, wd_parts: tuple[str, ...]
) -> str | None:
    """Apply the parent-repo guard for one spelling of the work_dir parts.

    Args:
        command: The Bash command line the model wants to run.
        wd_parts: ``Path.parts`` of one spelling (given or resolved) of
            the agent's working directory.

    Returns:
        The refusal message, or ``None`` when the command is allowed.
    """
    i = _worktree_index(wd_parts)
    if i is None:
        return None
    main_repo = str(Path(*wd_parts[:i]))
    wt_root = str(Path(*wd_parts[: i + 2]))
    if not os.path.isdir(wt_root):
        return None
    pattern = re.escape(main_repo) + r"(?=/|[\s'\";|&<>()`]|$)"
    for m in re.finditer(pattern, command):
        tail_start = m.start()
        end = tail_start + len(main_repo)
        while end < len(command) and command[end] not in " \t\n'\";|&<>()`":
            end += 1
        hit = command[tail_start:end]
        if hit == wt_root or hit.startswith(wt_root + os.sep):
            continue
        suggested = rewrite_parent_repo_paths(command, wt_root)
        return (
            f"Error: command references the parent-repo path "
            f"{hit!r}, which is outside the active worktree "
            f"{wt_root!r}.  Rewrite the command to use the "
            f"worktree path (or a path relative to it) so the "
            f"change is captured by the framework's auto-commit "
            f"and does not mutate the user's main checkout. "
            f"Suggested command: {suggested[:2000]}"
        )
    return None


def _suggest_close_path(resolved: Path) -> str:
    """Return a ``Did you mean: …`` suffix for a missing file, or ``""``.

    Looks in *resolved*'s parent directory for the closest filename
    (case-insensitive) via :func:`difflib.get_close_matches`.  If the
    parent itself does not exist, walks upward to the nearest existing
    ancestor and suggests an entry from it.
    """
    parent = resolved.parent
    while parent != parent.parent and not parent.is_dir():
        parent = parent.parent
    if not parent.is_dir():
        return ""
    try:
        names = [p.name for p in parent.iterdir()]
    except OSError:  # pragma: no cover — permission edge case
        return ""
    matches = difflib.get_close_matches(resolved.name, names, n=1, cutoff=0.6)
    if not matches:
        lowered = {n.lower(): n for n in names}
        ci = difflib.get_close_matches(
            resolved.name.lower(), list(lowered), n=1, cutoff=0.6
        )
        if ci:
            matches = [lowered[ci[0]]]
    if matches:
        return f" Did you mean: {parent / matches[0]} ?"
    return ""


def _outline(file_path: str, lines: list[str], size: int) -> str | None:
    """Build the outline a whole-file ``Read`` of a long file returns.

    Args:
        file_path: The path as the model passed it (for the header).
        lines: The file's lines.
        size: The file size in characters.

    Returns:
        The line count plus a ``line: symbol`` list of definitions and
        headings and how to read a range, or ``None`` when the file has
        too few recognisable symbols for an outline to be useful (the
        caller then returns the first window as before).
    """
    pattern = (
        _OUTLINE_HEADING_RE
        if Path(file_path).suffix.lower() in _MARKDOWN_SUFFIXES
        else _OUTLINE_SYMBOL_RE
    )
    entries = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(lines, 1)
        if pattern.match(line)
    ]
    if len(entries) < _OUTLINE_MIN_ENTRIES:
        return None
    shown = entries[:_OUTLINE_MAX_ENTRIES]
    more = len(entries) - len(shown)
    return (
        f"{file_path}: {len(lines):,} lines, {size:,} chars. Too long to send "
        f"whole; outline of its {len(entries)} definitions/headings (line: text):\n"
        + "\n".join(shown)
        + (f"\n... {more} more entries" if more else "")
        + "\n\nRead a range with Read(file_path, start_line=N, max_lines=M), or "
        "locate text with Bash(\"grep -n PATTERN FILE\")."
    )


def _find_windows_bash() -> str | None:  # pragma: no cover — Windows only
    """Find bash.exe on Windows (Git for Windows, WSL, etc.)."""
    found = shutil.which("bash")
    if found:
        return found
    for candidate in [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


_WINDOWS_BASH: str | None = _find_windows_bash() if sys.platform == "win32" else None


def _popen_kwargs(command: str) -> dict[str, Any]:
    """Return Popen kwargs appropriate for the current platform.

    On Unix, uses ``shell=True`` with ``start_new_session=True``.
    On Windows with bash available, invokes bash directly.
    On Windows without bash, falls back to PowerShell.

    Args:
        command: The command string to execute.

    Returns:
        Dict of keyword arguments for ``subprocess.Popen``.
    """
    if sys.platform != "win32":
        return {
            "args": command,
            "shell": True,
            "start_new_session": True,
        }
    else:  # pragma: no cover — Windows only
        if _WINDOWS_BASH:
            return {
                "args": [_WINDOWS_BASH, "-c", command],
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        ps = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return {
            "args": [ps, "-NoProfile", "-Command", command],
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    worst_msg = f"\n\n... [truncated {len(output)} chars] ...\n\n"
    if max_chars < len(worst_msg):
        return output[:max_chars]
    remaining = max_chars - len(worst_msg)
    head = remaining // 2
    tail = remaining - head
    dropped = len(output) - head - tail
    msg = f"\n\n... [truncated {dropped} chars] ...\n\n"
    if tail:
        return output[:head] + msg + output[-tail:]
    return output[:head] + msg


def _clean_env(work_dir: str | None = None) -> dict[str, str]:
    """Return a fresh copy of ``os.environ`` without ``VIRTUAL_ENV``.

    When the agent process runs inside a virtual-env (e.g. the VS Code
    extension's own ``.venv``), the ``VIRTUAL_ENV`` variable leaks into
    child processes and causes ``uv run`` to emit a spurious warning about
    a mismatched environment. Stripping it lets ``uv`` (and other tools)
    discover the correct project ``.venv`` on their own.

    When ``work_dir`` is provided, ``KISS_WORKDIR`` is overridden so that
    child processes (e.g. project scripts that compute "project root" via
    that env var) target the agent's working directory — critical when
    the agent is running inside a git worktree, where the inherited
    ``KISS_WORKDIR`` would otherwise still point at the original repo
    checkout and writes would leak out of the worktree.

    Args:
        work_dir: Agent working directory to expose to child processes
            via ``KISS_WORKDIR``.  ``None`` or ``""`` (the unresolved
            default on a freshly constructed agent) leaves the inherited
            value untouched.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    if work_dir:
        env["KISS_WORKDIR"] = work_dir
    return env


def _format_bash_result(returncode: int, output: str, max_output_chars: int) -> str:
    if returncode != 0:
        msg = f"Error (exit code {returncode}):"
        if output:
            msg += f"\n{output}"
        return _truncate_output(msg, max_output_chars)
    return _truncate_output(output, max_output_chars)


def _run_one_command(
    command: str, cancel: threading.Event, work_dir: str | None, timeout_seconds: float,
) -> tuple[int | None, str, float]:
    """Run one host ``run_commands_parallel`` command on a worker thread.

    Args:
        command: The shell command.
        cancel: The fan-out's stop event; setting it kills the shell.
        work_dir: The agent's working directory (worktree guard + cwd).
        timeout_seconds: Deadline for the shell.

    Returns:
        ``(returncode, output, seconds)``: the exit code (``None`` on
        timeout, ``-1`` when the command was refused or failed to
        launch), the combined output, and the wall time spent.
    """
    guard = _bash_parent_repo_guard(command, work_dir)
    if guard is not None:
        return -1, guard, 0.0
    started = time.monotonic()
    try:
        # Stream-less: printers attribute output by thread-local task
        # id, so a callback from this worker thread would detach the
        # output from the task.
        runner = UsefulTools(stop_event=cancel, work_dir=work_dir)
        returncode, output = runner._bash_streaming(command, timeout_seconds)
    except Exception as e:
        logger.debug("Exception caught", exc_info=True)
        returncode, output = -1, f"Error: {e}"
    return returncode, output, time.monotonic() - started


CommandRunner = Callable[[str, threading.Event], tuple[int | None, str, float]]

NOT_STARTED = "Not started: the task was stopped."


def run_commands_pool(
    commands: list[str],
    run_one: CommandRunner,
    max_workers: int,
    stop_event: threading.Event | None,
    max_output_chars: int,
) -> str:
    """Run *commands* concurrently through *run_one* and render the report.

    The one engine behind both ``run_commands_parallel`` tools (host
    shell and Docker container).  The workers watch ONE cancel event,
    set here when the task's *stop_event* or the tool panel's Stop
    (:func:`kiss.core.tool_interrupt.current_tool_interrupt_event`,
    thread-local and therefore invisible to workers) fires; queued
    commands are cancelled at the same time so a stop never starts
    work.  An exception injected into the waiting thread (a task Stop's
    ``KeyboardInterrupt``, a forced ``ToolCallInterrupted``) also
    cancels everything before propagating.

    Args:
        commands: The shell commands, already validated.
        run_one: ``run_one(command, cancel) -> (exit_code, output,
            seconds)``; must not raise.
        max_workers: Concurrency bound; ``0`` runs all commands at once.
        stop_event: The task's stop event, or ``None``.
        max_output_chars: Per-command output cap in the report.

    Returns:
        The report (see :func:`_format_parallel_report`).

    Raises:
        ToolCallInterrupted: When the tool panel's Stop was pressed.
    """
    cancel = threading.Event()
    interrupt = tool_interrupt.current_tool_interrupt_event()
    watched = [e for e in (stop_event, interrupt) if e is not None]
    pool = ThreadPoolExecutor(max_workers=max_workers or len(commands))
    try:
        futures = [pool.submit(run_one, command, cancel) for command in commands]
        while not all(f.done() for f in futures):
            if any(e.is_set() for e in watched):
                cancel.set()
                for f in futures:
                    f.cancel()
            # Short slices keep this loop in Python, where an injected
            # KeyboardInterrupt / ToolCallInterrupted lands.
            wait(futures, timeout=0.2)
    except BaseException:
        cancel.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    tool_interrupt.raise_if_interrupted()
    results = [(-1, NOT_STARTED, 0.0) if f.cancelled() else f.result() for f in futures]
    return _format_parallel_report(commands, results, max_output_chars)


def _format_parallel_report(
    commands: list[str],
    results: list[tuple[int | None, str, float]],
    max_output_chars: int,
) -> str:
    """Render the ``run_commands_parallel`` report: a tally, then one section per command."""
    codes = [code for code, _, _ in results]
    succeeded = sum(1 for code in codes if code == 0)
    timed_out = sum(1 for code in codes if code is None)
    failed = len(codes) - succeeded - timed_out
    lines = [
        f"{len(commands)} commands: {succeeded} succeeded, {failed} failed, "
        f"{timed_out} timed out."
    ]
    for index, (command, (code, output, seconds)) in enumerate(
        zip(commands, results, strict=True), start=1
    ):
        if code is None:
            status = f"TIMED OUT after {seconds:.1f}s"
        elif code < -1:
            status = f"killed by signal {-code} in {seconds:.1f}s"
        else:
            status = f"exit {code} in {seconds:.1f}s"
        lines.append(f"\n### [{index}/{len(commands)}] {status}\n$ {command}")
        lines.append(_truncate_output(output.rstrip("\n"), max_output_chars) or "(no output)")
    return "\n".join(lines)


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill a subprocess and all its children.

    On Windows, uses ``taskkill /T /F`` to kill the entire process tree.
    On Unix, sends ``SIGKILL`` to the process group created by
    ``start_new_session=True``, falling back to ``process.kill()``.

    Args:
        process: The subprocess to terminate.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows only
        # Bounded like ``git_worktree._git``'s taskkill: a hung taskkill
        # would otherwise wedge the stop monitor / timeout path forever.
        with suppress(subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
            )
        if process.poll() is None:
            # taskkill hung or failed: at least the shell itself must
            # not survive, or the caller reports ``exit code None``
            # for a command that is still running.
            with suppress(OSError):
                process.kill()
    else:
        if process.poll() is not None:
            # Already reaped: the shell's PID — and therefore the PGID
            # named after it — may have been recycled by an unrelated
            # process, so ``killpg`` here could SIGKILL a stranger's
            # process group.  Nothing is left to kill anyway (the group
            # leader is gone and its exit already broke the pipes this
            # module reads).  Same hazard ``web_use_tool`` guards with
            # ``_process_identity``.
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:  # pragma: no cover — Popen.send_signal polls first in Python 3.13+
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover — unreachable after SIGKILL
        pass


def _stop_monitor(
    stop_events: list[threading.Event],
    process: subprocess.Popen,
    done: threading.Event,
) -> None:
    """Wait for any of *stop_events* to fire, then kill *process* group.

    Exits when *done* is set (process finished normally) or one of the
    events fires: the task's stop event (the agent was stopped) or the
    running tool call's interrupt event (the user pressed the tool
    panel's Stop button).
    """
    while not done.wait(timeout=0.2):
        for event in stop_events:
            if event.is_set():
                _kill_process_group(process)
                return


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        work_dir: str | None = None,
    ) -> None:
        """Initialise the tools.

        Args:
            stream_callback: Optional sink that receives Bash output line
                by line for live streaming to the UI.
            stop_event: Optional event signalled by the host agent when
                the user requests a stop; the Bash tool kills the
                running command's process group.
            work_dir: Agent working directory.  When set, ``Bash``
                subprocesses are launched with ``cwd=work_dir`` and the
                ``KISS_WORKDIR`` env var is forced to ``work_dir`` so
                project scripts that derive a "project root" from it
                stay inside the worktree the agent is operating on.
        """
        self.stream_callback = stream_callback
        self.stop_event = stop_event
        self.work_dir = work_dir
        # Read dedupe: (path, start_line, max_lines) -> sha of the file text
        # the model was last shown for that window (see :meth:`Read`).
        self._reads_shown: dict[tuple[str, int, int], str] = {}
        # Files this instance has shown the model (Read) or written itself
        # (Write, Edit).  Edit and Write refuse an existing file that is not
        # in this set, so the read-before-modify rule of SYSTEM.md is
        # enforced by the tools and not only requested by the prompt.
        self.read_files: set[Path] = set()

    def forget_reads(self) -> None:
        """Forget which file windows the model has already been shown.

        Called by the agent when earlier tool outputs left the model's
        context (compaction, a new session), so the next ``Read`` of an
        unchanged file returns its content instead of the dedupe stub.
        ``read_files`` is kept: the model has still seen the file once in
        this task, so Edit and Write stay permitted on it.
        """
        self._reads_shown.clear()

    def _unread_error(self, file_path: str, verb: str) -> str:
        """Return the error text for modifying a file the model has not read."""
        return (
            f"Error: {file_path} has not been read in this session. "
            f"Call Read on it before {verb} it."
        )

    def _spawn(self, command: str) -> subprocess.Popen:
        """Launch *command* with the shared Popen configuration.

        Output (stdout + stderr combined) is captured as UTF-8 text and
        the child runs in ``self.work_dir`` with a cleaned environment.
        Invalid UTF-8 bytes in the output (e.g. ``cat`` of a binary,
        ``grep`` on a binary file) are replaced with U+FFFD instead of
        raising ``UnicodeDecodeError`` — strict decoding would lose the
        ENTIRE output (and, on the streaming path, leak the exception
        out of the tool) even when the command itself succeeded.

        Vanished-worktree fallback: if ``self.work_dir`` points at a
        directory that no longer exists on disk (e.g. the per-task
        ``.kiss-worktrees/kiss_wt-*`` worktree was torn down by a
        concurrent cleanup, discard, or crashed merge between
        ``RelentlessAgent`` sub-sessions), launching the subprocess
        with that missing ``cwd`` would crash *every* Bash call with
        ``FileNotFoundError: [Errno 2] No such file or directory``
        before any output is produced.  In that case we transparently
        fall back to the parent repository root (with the
        ``.kiss-worktrees/kiss_wt-<slug>`` segment stripped) so the
        agent can keep working from the user's main checkout instead
        of dying on every command.  If even the fallback path does
        not exist we drop ``cwd`` entirely and let the child inherit
        the agent process's cwd.

        Args:
            command: The shell command to execute.

        Returns:
            The started subprocess.
        """
        cwd: str | None = self.work_dir or None
        env_work_dir: str | None = self.work_dir
        if cwd is not None and not os.path.isdir(cwd):
            fallback = _stale_worktree_fallback(Path(cwd))
            if fallback is not None and fallback.is_dir():
                logger.warning(
                    "Bash work_dir %r vanished mid-task; "
                    "falling back to parent repo root %r",
                    cwd,
                    str(fallback),
                )
                cwd = str(fallback)
                env_work_dir = cwd
            else:
                logger.warning(
                    "Bash work_dir %r vanished mid-task and no "
                    "fallback parent exists; running without cwd",
                    cwd,
                )
                cwd = None
                env_work_dir = None
        return subprocess.Popen(
            **_popen_kwargs(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_env(env_work_dir),
            cwd=cwd,
        )

    def _start_stop_monitor(
        self, process: subprocess.Popen, done: threading.Event,
    ) -> None:
        """Start a daemon thread that kills *process* on a stop or interrupt.

        Watches this instance's ``stop_event`` and the interrupt event of
        the tool call running on the calling thread
        (:func:`kiss.core.tool_interrupt.current_tool_interrupt_event`),
        so both the task's Stop button and the tool panel's own Stop
        button kill the shell.  No-op when neither exists.  The monitor
        exits when *done* is set (process finished normally).

        Args:
            process: The running subprocess to watch.
            done: Event the caller sets once the process has finished.
        """
        events = [
            event
            for event in (self.stop_event, tool_interrupt.current_tool_interrupt_event())
            if event is not None
        ]
        if events:
            threading.Thread(
                target=_stop_monitor,
                args=(events, process, done),
                daemon=True,
            ).start()

    def Read(  # noqa: N802
        self,
        file_path: str,
        max_lines: int = 2000,
        start_line: int = 1,
        force: bool = False,
    ) -> str:
        """Read file contents; a long file with no line range returns its outline first.

        Args:
            file_path: Absolute path to file.
            max_lines: Maximum number of lines to return.
            start_line: 1-indexed line at which to begin the returned
                window.  ``start_line=1`` (the default) reads from the
                top of the file and is backward-compatible.  Values
                less than 1 are rejected; values beyond EOF return an
                explicit sentinel rather than empty content so the
                model is not misled into thinking the file is empty.
            force: Re-send a window that is unchanged since an earlier
                Read in this task (by default such a Read returns a
                one-line "unchanged" note instead of the content).
        """
        if start_line < 1:
            return (
                f"Error: start_line must be >= 1 (got {start_line}); the "
                f"parameter is 1-indexed."
            )
        if max_lines < 1:
            return f"Error: max_lines must be >= 1 (got {max_lines})."
        try:
            expanded = _absolutize(file_path, self.work_dir)
            resolved = Path(expanded).resolve()

            remapped = _active_worktree_remap(resolved, self.work_dir)
            if remapped is not None:
                resolved = remapped
            elif not resolved.exists():
                fallback = _stale_worktree_fallback(resolved)
                if fallback is not None and fallback.exists():
                    resolved = fallback.resolve()

            if resolved.is_dir():
                return self._read_directory_listing(file_path, resolved)

            if resolved.exists() and not resolved.is_file():
                return (
                    f"Error: {file_path} is not a regular file "
                    f"(FIFO/device/socket); reading it could block forever. "
                    f"Use Bash (which has a timeout) if you really need its "
                    f"contents."
                )

            try:
                text = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.debug("Binary file detected", exc_info=True)
                shown = self._read_binary(file_path, resolved)
                if not shown.startswith("Error:"):
                    self.read_files.add(resolved)
                return shown
            except FileNotFoundError:
                suggestion = _suggest_close_path(resolved)
                return f"Error: File not found: {file_path}.{suggestion}"

            self.read_files.add(resolved)
            if text == "":
                return "(file is empty)"

            lines = text.splitlines(keepends=True)
            total = len(lines)
            if start_line > total:
                return (
                    f"Error: start_line={start_line} is past EOF "
                    f"(file has {total} line{'s' if total != 1 else ''})."
                )
            outline_limit = DEFAULT_CONFIG.read_outline_lines
            whole_file_requested = start_line == 1 and max_lines >= 2000
            outline = (
                _outline(file_path, lines, len(text))
                if whole_file_requested and 0 < outline_limit < total else None
            )
            window = lines[start_line - 1 : start_line - 1 + max_lines]
            last = start_line - 1 + len(window)
            remaining = total - last
            if outline is not None:
                content, shown = outline, "outline"
                key = (str(resolved), 0, 0)
            else:
                content = "".join(window)
                if remaining > 0:
                    content += f"\n[truncated: {remaining} more lines]"
                shown = f"lines {start_line}-{last} of {total}"
                key = (str(resolved), start_line, max_lines)
            digest = hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
            if DEFAULT_CONFIG.read_dedupe and not force and self._reads_shown.get(key) == digest:
                return (
                    f"Unchanged since your earlier Read of {file_path} ({shown}): the "
                    f"content is already in your context above. Pass force=True to "
                    f"re-read it."
                )
            self._reads_shown[key] = digest
            return content
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _read_directory_listing(self, file_path: str, resolved: Path) -> str:
        """Return a helpful directory listing when Read is called on a dir.

        Models occasionally call ``Read`` on a directory path (e.g. when
        searching for the right module).  Instead of returning the bare
        ``[Errno 21] Is a directory`` error we surface a one-per-line
        listing so the model can self-correct on the next turn.
        """
        try:
            entries = sorted(
                p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir()
            )
        except OSError as e:  # pragma: no cover — permission edge case
            return f"Error: Cannot list directory {file_path}: {e}"
        listing = "\n".join(entries) if entries else "(empty directory)"
        return (
            f"Error: {file_path} is a directory, not a file. "
            f"Pass a file path inside it, or use Bash('ls -la ...').\n"
            f"Directory contents:\n{listing}"
        )

    def _read_binary(self, file_path: str, resolved: Path) -> str:
        """Encode a binary file as a sentinel attachment or return error."""
        mime_type, _ = mimetypes.guess_type(str(resolved))
        if mime_type in READ_TOOL_BINARY_MIME_TYPES:
            size = resolved.stat().st_size
            if size > _MAX_BINARY_READ_BYTES:
                return (
                    f"Error: Binary file {file_path} is too large to embed "
                    f"inline ({size} bytes > {_MAX_BINARY_READ_BYTES} byte "
                    f"limit, mime={mime_type}). Use Bash tools (e.g. "
                    f"ffmpeg/ImageMagick to downscale, or split the file) "
                    f"to produce a smaller artifact first."
                )
            data = resolved.read_bytes()
            header = (
                f"Read binary file {file_path} as {mime_type} "
                f"({len(data)} bytes); content attached below.\n"
            )
            return header + encode_binary_attachment(mime_type, data)
        size = resolved.stat().st_size
        return (
            f"Error: Cannot read binary file: {file_path} "
            f"(size: {size} bytes, mime={mime_type or 'unknown'}). "
            f"The Read tool only embeds binaries with a supported "
            f"MIME type (images, PDFs, audio, video); use a "
            f"different tool to handle this binary file."
        )

    def Write(  # noqa: N802
        self,
        file_path: str,
        content: str,
    ) -> str:
        """Write content to a file, creating it if it doesn't exist or overwriting if it does.

        Args:
            file_path: Path to the file to write.
            content: The full content to write to the file.
        """
        try:
            expanded = _absolutize(file_path, self.work_dir)
            resolved = Path(expanded).resolve()
            remapped = _active_worktree_remap(resolved, self.work_dir)
            if remapped is not None:
                resolved = remapped
            else:
                fallback = _stale_worktree_fallback(resolved)
                if fallback is not None:
                    resolved = fallback
            if resolved.exists() and not resolved.is_file():
                return (
                    f"Error: {file_path} exists and is not a regular file "
                    f"(directory/FIFO/device/socket); refusing to write to it."
                )
            if resolved.is_file() and resolved not in self.read_files:
                return self._unread_error(file_path, "overwriting")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8", newline="")
            self.read_files.add(resolved)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.

        Returns:
            The output of the edit operation.
        """
        try:
            expanded = _absolutize(file_path, self.work_dir)
            resolved = Path(expanded).resolve()
            remapped = _active_worktree_remap(resolved, self.work_dir)
            if remapped is not None:
                resolved = remapped
            elif not resolved.is_file():
                fallback = _stale_worktree_fallback(resolved)
                if fallback is not None and fallback.is_file():
                    resolved = fallback.resolve()
            if not resolved.is_file():
                return f"Error: File not found: {file_path}"
            if resolved not in self.read_files:
                return self._unread_error(file_path, "editing")
            if old_string == new_string:
                return "Error: new_string must be different from old_string"
            if old_string == "":
                return (
                    "Error: old_string must not be empty. "
                    "Use the Write tool to create or overwrite a file."
                )
            content = resolved.read_text(encoding="utf-8", newline="")
            count = content.count(old_string)
            if count == 0 and "\r\n" in content and "\r\n" not in old_string:
                old_string = old_string.replace("\n", "\r\n")
                new_string = new_string.replace("\r\n", "\n").replace("\n", "\r\n")
                count = content.count(old_string)
            if count == 0:
                return "Error: String not found in file"
            if not replace_all and count > 1:
                return (
                    f"Error: String appears {count} times (not unique). "
                    f"Use replace_all=True to replace all occurrences."
                )
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
            resolved.write_text(new_content, encoding="utf-8", newline="")
            replaced = count if replace_all else 1
            return f"Successfully replaced {replaced} occurrence(s) in {file_path}"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: float = 300,
        max_output_chars: int = 50000,
    ) -> str:
        """Runs a bash command and returns its output.

        Args:
            command: The bash command to run.
            description: A brief description of the command.
            timeout_seconds: Timeout in seconds for the command.
            max_output_chars: Maximum characters in output before truncation.

        Returns:
            The output of the command.
        """
        del description

        guard = _bash_parent_repo_guard(command, self.work_dir)
        if guard is not None:
            return guard

        if self.stream_callback:
            # Contract (see test_bash_background_pipe_hang): an exception
            # raised by the caller-supplied stream callback must PROPAGATE
            # out of Bash (after the process group is killed), not be
            # swallowed into an "Error:" string like internal failures are.
            returncode, output = self._bash_streaming(command, timeout_seconds)
        else:
            try:
                returncode, output = self._bash_streaming(command, timeout_seconds)
            except Exception as e:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)
                return f"Error: {e}"
        if returncode is None:
            return "Error: Command execution timeout"
        return _format_bash_result(returncode, output, max_output_chars)

    def run_commands_parallel(
        self,
        commands: str,
        max_workers: int = 0,
        timeout_seconds: float = 1800,
        max_output_chars: int = 5000,
    ) -> str:
        """Run several independent shell commands concurrently, with no LLM sub-agents.

        Use this instead of ``run_parallel`` whenever the parallel work is
        just shell commands whose output you read once they finish: test
        splits (one ``pytest`` invocation per split), builds, linters,
        benchmarks, batch conversions.  Each command runs in its own
        thread under the same rules as ``Bash`` (working directory,
        environment, worktree guard, Stop button); output is not streamed
        live, the combined report is returned when the last command ends.

        Args:
            commands: A JSON-encoded list of shell command strings, e.g.
                ``'["uv run pytest tests/a.py", "uv run pytest tests/b.py"]'``.
                Shell substitutions such as ``"$(cat cmds.json)"`` are not
                expanded and are rejected.
            max_workers: Maximum number of commands running at once.  ``0``
                (default) runs all of them at once.
            timeout_seconds: Timeout in seconds applied to EACH command.
            max_output_chars: Maximum characters of each command's output
                kept in the report (head and tail are kept when truncating).

        Returns:
            A report starting with a tally line
            (``N commands: K succeeded, F failed, T timed out``) followed by
            one section per command, in input order, giving its exit code,
            wall time and (truncated) output.  A string starting with
            ``Error:`` when the arguments were invalid.
        """
        try:
            command_list = parse_tasks_json(commands, name="commands")
        except ValueError as e:
            return f"Error: {e.args[0]}"
        if max_workers < 0:
            return f"Error: max_workers must be 0 or a positive integer, got {max_workers}."
        run_one = functools.partial(
            _run_one_command, work_dir=self.work_dir, timeout_seconds=timeout_seconds,
        )
        return run_commands_pool(
            command_list, run_one, max_workers, self.stop_event, max_output_chars,
        )

    def _consume_stream(
        self,
        out_queue: "queue.Queue[str | None]",
        chunks: list[str],
        deadline: float,
        stop: threading.Event | None = None,
        interrupt: threading.Event | None = None,
    ) -> bool:
        """Consume streamed lines from *out_queue* until EOF, *deadline*, *stop* or *interrupt*.

        Runs on the thread that called :meth:`Bash` so that
        ``stream_callback`` executes with the caller's thread-local
        context intact — printers route every event (task attribution,
        per-task bash buffers, recordings, stop events) by thread-local
        ``task_id``, so invoking the callback from an internal reader
        thread would silently detach the output from the task (bash
        output vanishing from the tool panel and unattributed
        "garbage" events reaching every webview client).

        Args:
            out_queue: Queue fed by the reader thread; ``None`` marks EOF.
            chunks: Accumulator that received lines are appended to.
            deadline: ``time.monotonic()`` timestamp to stop waiting at.
            stop: When given, the wait also ends once it is set.
                ``_stop_monitor`` alone cannot end the wait when the
                shell has already exited but a background child still
                holds the stdout pipe (no group to kill), so without
                this a stop would block for the whole *deadline*.
            interrupt: The running tool call's interrupt event (the
                tool panel's Stop button), ending the wait the same way.

        The queue is polled in 0.2 s slices rather than one long C-level
        wait, so the loop keeps returning to Python: that is where an
        asynchronously injected exception — the task Stop's
        ``KeyboardInterrupt`` — can land and reach :meth:`Bash`'s
        process-group kill.

        Returns:
            True when the EOF sentinel was received, False on deadline,
            stop or interrupt.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if stop is not None and stop.is_set():
                return False
            if interrupt is not None and interrupt.is_set():
                return False
            remaining = min(remaining, 0.2)
            try:
                line = out_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                return True
            chunks.append(line)
            if self.stream_callback is not None:
                self.stream_callback(line)

    def _bash_streaming(
        self, command: str, timeout_seconds: float,
    ) -> tuple[int | None, str]:
        """Run *command* through a reader thread, bounded by *timeout_seconds*.

        Single engine behind :meth:`Bash` and :meth:`run_commands_parallel`
        for both the streaming and the non-streaming case
        (``stream_callback`` may be ``None``).  Returns the shell's exit
        code — ``None`` for a genuine timeout — and its combined
        stdout/stderr, untruncated.  The timeout is a deadline on the
        SHELL, not on its descendants:

        * Shell still running at the deadline — genuine timeout.  The
          whole process group is killed and the timeout error returned.
        * Shell already exited but the stdout pipe is still open (a
          background child inherited it — e.g. ``cd d && nohup job &``
          wraps ``job`` in a pipe-holding subshell) — the command itself
          SUCCEEDED, so its output is returned and the lingering
          children are LEFT RUNNING.  Killing the group here used to
          silently SIGKILL deliberately detached background jobs minutes
          after this method returned success-looking output.  The reader
          thread stays behind in discard mode so a child that writes to
          the inherited pipe can never block on a full pipe buffer.

        A ``stop_event`` ends the wait early in both states: a running
        shell is killed and its exit code reported; an exited shell's
        output is returned after the same short EOF grace period.
        """
        process = self._spawn(command)
        done = threading.Event()
        abandoned = threading.Event()
        out_queue: queue.Queue[str | None] = queue.Queue()

        def _drain_stdout() -> None:
            try:
                assert process.stdout is not None
                for line in iter(process.stdout.readline, ""):
                    if not abandoned.is_set():
                        out_queue.put(line)
            finally:
                if abandoned.is_set():
                    # The main thread returned long ago and never closes
                    # the pipe in this mode; release the fd here.
                    try:
                        process.stdout.close()  # type: ignore[union-attr]
                    except Exception:
                        logger.debug("stdout close failed", exc_info=True)
                out_queue.put(None)

        timed_out = False
        eof = False
        chunks: list[str] = []
        try:
            # The helper threads start INSIDE the kill/cleanup region:
            # a ``Thread.start()`` that raises (thread exhaustion) or a
            # stop injected here must still kill the shell just spawned
            # and set ``done`` — otherwise the command runs on unowned
            # and an already-started monitor polls forever.
            reader = threading.Thread(target=_drain_stdout, daemon=True)
            self._start_stop_monitor(process, done)
            reader.start()
            interrupt = tool_interrupt.current_tool_interrupt_event()
            eof = self._consume_stream(
                out_queue, chunks, time.monotonic() + timeout_seconds,
                stop=self.stop_event, interrupt=interrupt,
            )
            if not eof:
                # A stop is not a timeout: the shell is killed (the
                # monitor may already have done so) and its exit code
                # reported, exactly as when the stop landed on a running
                # shell before this loop learned to observe it.
                stopped = self.stop_event is not None and self.stop_event.is_set()
                interrupted = interrupt is not None and interrupt.is_set()
                still_running = process.poll() is None
                timed_out = still_running and not stopped and not interrupted
                if still_running:
                    _kill_process_group(process)
                # The tool panel's Stop keeps this whole tail inside the
                # interrupt's 1 s cooperative grace: a longer EOF wait
                # would let the forced injection land in this loop
                # instead of at the raise below.
                eof = self._consume_stream(
                    out_queue, chunks, time.monotonic() + (0.5 if interrupted else 5),
                )
                if not eof:
                    abandoned.set()
                    if timed_out:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:  # pragma: no cover
                            pass
            else:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    _kill_process_group(process)
        except BaseException:
            # The reader thread outlives this frame: in discard mode it
            # closes the pipe itself once the child lets go of it.
            abandoned.set()
            _kill_process_group(process)
            raise
        finally:
            done.set()
            if eof:
                process.stdout.close()  # type: ignore[union-attr]

        # The tool panel's own Stop: whether the monitor killed the shell
        # (EOF) or the loop above noticed the event first, the shell is
        # dead and its pipe closed or abandoned, so this is the safe
        # point to raise the interrupt the agent loop turns into "User
        # interrupted the tool call.".
        tool_interrupt.raise_if_interrupted()

        if timed_out:
            return None, ""
        return process.returncode, "".join(chunks)
