# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Useful tools for agents: file editing and bash execution."""

import difflib
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
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # POSIX has no msvcrt
    msvcrt = None  # type: ignore[assignment]

from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SLUG_PREFIX,
    _WORKTREE_SUBDIR,
)
from kiss.core.models.model import (
    READ_TOOL_BINARY_MIME_TYPES,
    encode_binary_attachment,
)

logger = logging.getLogger(__name__)

_MAX_BINARY_READ_BYTES = 20 * 1024 * 1024


@contextmanager
def _file_lock(lock_path: Path) -> Any:
    """Hold an exclusive advisory inter-process lock on *lock_path*.

    Serializes check-then-use sequences on resources shared by every
    kiss process on the machine — MCP configs and OAuth token stores,
    and the Chromium profile directory — across daemons, CLI runs,
    channel-agent processes, and event loops.  A ``threading`` lock
    cannot do this: the resources live on disk, not in one process.
    The lock file itself is created mode ``0600``.

    Cross-platform: ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
    Windows (where ``fcntl`` does not exist — an unconditional import
    would make every dependent feature unavailable there).  When
    neither primitive exists the lock degrades to a best-effort no-op
    rather than breaking the caller entirely.

    Args:
        lock_path: The lock file to hold; parent directories are created.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover — Windows-only branch
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(  # pyright: ignore[reportAttributeAccessIssue]
                        descriptor,
                        msvcrt.LK_LOCK,  # pyright: ignore[reportAttributeAccessIssue]
                        1,
                    )
                    break
                except OSError:
                    time.sleep(0.05)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover — Windows-only branch
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

    Args:
        parts: ``Path.parts`` of the path (or work dir) to inspect.

    Returns:
        The index ``i`` such that ``parts[i]`` is the worktree subdir
        and ``parts[i + 1]`` is a worktree slug, or ``None`` when
        *parts* does not run through a worktree.
    """
    for i, part in enumerate(parts):
        if (
            part == _WORKTREE_SUBDIR
            and i + 1 < len(parts)
            and parts[i + 1].startswith(_WORKTREE_SLUG_PREFIX)
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
        return (
            f"Error: command references the parent-repo path "
            f"{hit!r}, which is outside the active worktree "
            f"{wt_root!r}.  Rewrite the command to use the "
            f"worktree path (or a path relative to it) so the "
            f"change is captured by the framework's auto-commit "
            f"and does not mutate the user's main checkout."
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


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill a subprocess and all its children.

    On Windows, uses ``taskkill /T /F`` to kill the entire process tree.
    On Unix, sends ``SIGKILL`` to the process group created by
    ``start_new_session=True``, falling back to ``process.kill()``.

    Args:
        process: The subprocess to terminate.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows only
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
        )
    else:
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
    stop_event: threading.Event,
    process: subprocess.Popen,
    done: threading.Event,
) -> None:
    """Wait for *stop_event* to fire, then kill *process* group.

    Exits when *done* is set (process finished normally) or *stop_event*
    fires (agent was stopped).
    """
    while not done.wait(timeout=0.2):
        if stop_event.is_set():
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
        """Start a daemon thread that kills *process* if ``stop_event`` fires.

        No-op when this instance has no ``stop_event``.  The monitor
        exits when *done* is set (process finished normally).

        Args:
            process: The running subprocess to watch.
            done: Event the caller sets once the process has finished.
        """
        if self.stop_event:
            threading.Thread(
                target=_stop_monitor,
                args=(self.stop_event, process, done),
                daemon=True,
            ).start()

    def Read(  # noqa: N802
        self,
        file_path: str,
        max_lines: int = 2000,
        start_line: int = 1,
    ) -> str:
        """Read file contents.

        Args:
            file_path: Absolute path to file.
            max_lines: Maximum number of lines to return.
            start_line: 1-indexed line at which to begin the returned
                window.  ``start_line=1`` (the default) reads from the
                top of the file and is backward-compatible.  Values
                less than 1 are rejected; values beyond EOF return an
                explicit sentinel rather than empty content so the
                model is not misled into thinking the file is empty.
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
                return self._read_binary(file_path, resolved)
            except FileNotFoundError:
                suggestion = _suggest_close_path(resolved)
                return f"Error: File not found: {file_path}.{suggestion}"

            if text == "":
                return "(file is empty)"

            lines = text.splitlines(keepends=True)
            total = len(lines)
            if start_line > total:
                return (
                    f"Error: start_line={start_line} is past EOF "
                    f"(file has {total} line{'s' if total != 1 else ''})."
                )
            window = lines[start_line - 1 : start_line - 1 + max_lines]
            remaining = total - (start_line - 1) - len(window)
            if remaining > 0:
                return "".join(window) + f"\n[truncated: {remaining} more lines]"
            return "".join(window)
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
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8", newline="")
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
            return self._bash_streaming(command, timeout_seconds, max_output_chars)

        try:
            return self._bash_streaming(command, timeout_seconds, max_output_chars)
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _consume_stream(
        self,
        out_queue: "queue.Queue[str | None]",
        chunks: list[str],
        deadline: float,
    ) -> bool:
        """Consume streamed lines from *out_queue* until EOF or *deadline*.

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

        Returns:
            True when the EOF sentinel was received, False on deadline.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                line = out_queue.get(timeout=remaining)
            except queue.Empty:
                return False
            if line is None:
                return True
            chunks.append(line)
            if self.stream_callback is not None:
                self.stream_callback(line)

    def _bash_streaming(self, command: str, timeout_seconds: float, max_output_chars: int) -> str:
        """Run *command* through a reader thread, bounded by *timeout_seconds*.

        Single engine behind :meth:`Bash` for both the streaming and the
        non-streaming case (``stream_callback`` may be ``None``).  The
        timeout is a deadline on the SHELL, not on its descendants:

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

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        self._start_stop_monitor(process, done)
        reader.start()
        timed_out = False
        eof = False
        chunks: list[str] = []
        try:
            eof = self._consume_stream(
                out_queue, chunks, time.monotonic() + timeout_seconds,
            )
            if not eof:
                timed_out = process.poll() is None
                if timed_out:
                    _kill_process_group(process)
                eof = self._consume_stream(
                    out_queue, chunks, time.monotonic() + 5,
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
            _kill_process_group(process)
            raise
        finally:
            done.set()
            if eof:
                process.stdout.close()  # type: ignore[union-attr]

        if timed_out:
            return "Error: Command execution timeout"

        output = "".join(chunks)
        return _format_bash_result(process.returncode, output, max_output_chars)
