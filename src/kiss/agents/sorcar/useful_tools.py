# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Useful tools for agents: file editing and bash execution."""

import difflib
import logging
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.core.models.model import (
    READ_TOOL_BINARY_MIME_TYPES,
    encode_binary_attachment,
)

logger = logging.getLogger(__name__)


def _expand_pwd_prefix(file_path: str, work_dir: str | None) -> str:
    """Expand a literal ``PWD/`` prefix to the agent's working directory.

    The system prompt instructs the model to interpret ``PWD`` as the
    current working directory, but models routinely pass it through as a
    literal path component (e.g. ``PWD/SORCAR.md``).  This helper
    rewrites such paths so the subsequent Read still works.
    """
    if file_path == "PWD":
        return work_dir or os.getcwd()
    if file_path.startswith("PWD/"):
        base = work_dir or os.getcwd()
        # Strip extra leading slashes (e.g. "PWD//etc/passwd"):
        # os.path.join discards *base* when the second component is
        # absolute, which would silently escape the working directory.
        suffix = file_path[len("PWD/") :].lstrip("/")
        return os.path.join(base, suffix) if suffix else base
    return file_path


def _stale_worktree_fallback(resolved: Path) -> Path | None:
    """If *resolved* lives under a now-deleted ``.kiss-worktrees/kiss_wt-*``
    directory, return the equivalent path with that worktree segment
    stripped (i.e. relative to the parent repo).

    Worktrees are torn down on autocommit / success, so a model that
    remembers a worktree path from earlier in the task ends up with a
    dangling path.  Returning the equivalent in-repo path lets the
    subsequent read succeed transparently.
    """
    parts = resolved.parts
    for i, part in enumerate(parts):
        if (
            part == ".kiss-worktrees"
            and i + 1 < len(parts)
            and parts[i + 1].startswith("kiss_wt-")
        ):
            return Path(*parts[:i], *parts[i + 2 :])
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

        Args:
            command: The shell command to execute.

        Returns:
            The started subprocess.
        """
        return subprocess.Popen(
            **_popen_kwargs(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_env(self.work_dir),
            cwd=self.work_dir or None,
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
    ) -> str:
        """Read file contents.

        Args:
            file_path: Absolute path to file.
            max_lines: Maximum number of lines to return.
        """
        try:
            expanded = _expand_pwd_prefix(file_path, self.work_dir)
            resolved = Path(expanded).resolve()

            # Stale-worktree fallback: if the caller passed a path inside
            # a .kiss-worktrees/kiss_wt-* directory that has since been
            # torn down (autocommit / task finish), try the equivalent
            # path under the parent repo before giving up.
            if not resolved.exists():
                fallback = _stale_worktree_fallback(resolved)
                if fallback is not None and fallback.exists():
                    resolved = fallback.resolve()

            if resolved.is_dir():
                return self._read_directory_listing(file_path, resolved)

            # Non-regular files (FIFOs, devices, sockets): opening a FIFO
            # with no writer blocks FOREVER, and /dev/zero-style devices
            # stream endlessly — either would hang the agent with no
            # timeout.  Refuse them up front.
            if resolved.exists() and not resolved.is_file():
                return (
                    f"Error: {file_path} is not a regular file "
                    f"(FIFO/device/socket); reading it could block forever. "
                    f"Use Bash (which has a timeout) if you really need its "
                    f"contents."
                )

            try:
                text = resolved.read_text()
            except UnicodeDecodeError:
                logger.debug("Binary file detected", exc_info=True)
                return self._read_binary(file_path, resolved)
            except FileNotFoundError:
                suggestion = _suggest_close_path(resolved)
                return f"Error: File not found: {file_path}.{suggestion}"

            if text == "":
                return "(file is empty)"

            lines = text.splitlines(keepends=True)
            if len(lines) > max_lines:
                return (
                    "".join(lines[:max_lines])
                    + f"\n[truncated: {len(lines) - max_lines} more lines]"
                )
            return text
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
            expanded = _expand_pwd_prefix(file_path, self.work_dir)
            resolved = Path(expanded).resolve()
            # Refuse existing non-regular targets: opening a FIFO for
            # writing blocks forever when it has no reader, hanging the
            # agent with no timeout (directories/devices are also wrong).
            if resolved.exists() and not resolved.is_file():
                return (
                    f"Error: {file_path} exists and is not a regular file "
                    f"(directory/FIFO/device/socket); refusing to write to it."
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
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
            expanded = _expand_pwd_prefix(file_path, self.work_dir)
            resolved = Path(expanded).resolve()
            if not resolved.is_file():
                return f"Error: File not found: {file_path}"
            if old_string == new_string:
                return "Error: new_string must be different from old_string"
            if old_string == "":
                # str.count("") == len(content) + 1, so an empty
                # old_string would interleave new_string between every
                # character (replace_all) or silently overwrite an empty
                # file.  Reject it explicitly instead.
                return (
                    "Error: old_string must not be empty. "
                    "Use the Write tool to create or overwrite a file."
                )
            # Read WITHOUT universal-newline translation: reading the
            # translated text and writing it back would silently rewrite
            # EVERY line ending in a CRLF file as LF (huge spurious
            # diffs) even for a one-character edit.
            content = resolved.read_text(newline="")
            count = content.count(old_string)
            if count == 0 and "\r\n" in content and "\r\n" not in old_string:
                # Models see files through Read(), which translates CRLF
                # to LF — so old_string/new_string carry LF even when the
                # file on disk uses CRLF.  Retry with CRLF-normalised
                # strings so the edit applies and the replacement keeps
                # the file's CRLF convention.
                old_string = old_string.replace("\n", "\r\n")
                # Collapse-then-expand so a new_string that already
                # carries some CRLFs is not corrupted into "\r\r\n".
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
            # newline="" prevents os.linesep translation on write so the
            # preserved (untranslated) line endings round-trip verbatim.
            resolved.write_text(new_content, newline="")
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

        if self.stream_callback:
            return self._bash_streaming(command, timeout_seconds, max_output_chars)

        try:
            process = self._spawn(command)
            done = threading.Event()
            self._start_stop_monitor(process, done)
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:  # pragma: no cover — unreachable after SIGKILL
                    pass
                return "Error: Command execution timeout"
            except BaseException:  # pragma: no cover — KeyboardInterrupt timing-dependent
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:
                    pass
                raise
            finally:
                done.set()
            return _format_bash_result(process.returncode, stdout, max_output_chars)
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _bash_streaming(self, command: str, timeout_seconds: float, max_output_chars: int) -> str:
        assert self.stream_callback is not None
        process = self._spawn(command)
        timed_out = False
        done = threading.Event()

        def _kill() -> None:
            nonlocal timed_out
            timed_out = True
            _kill_process_group(process)

        timer = threading.Timer(timeout_seconds, _kill)
        timer.start()
        self._start_stop_monitor(process, done)
        try:
            chunks: list[str] = []
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                chunks.append(line)
                self.stream_callback(line)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                _kill_process_group(process)
        except BaseException:  # pragma: no cover — KeyboardInterrupt timing-dependent
            _kill_process_group(process)
            raise
        finally:
            done.set()
            timer.cancel()
            process.stdout.close()  # type: ignore[union-attr]

        if timed_out:
            return "Error: Command execution timeout"

        output = "".join(chunks)
        return _format_bash_result(process.returncode, output, max_output_chars)
