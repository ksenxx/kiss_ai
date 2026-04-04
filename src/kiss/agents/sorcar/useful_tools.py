"""Useful tools for agents: file editing and bash execution."""

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _log_exc() -> None:
    logger.debug("Exception caught", exc_info=True)


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



def _clean_env() -> dict[str, str]:
    """Return a fresh copy of ``os.environ`` without ``VIRTUAL_ENV``.

    When the agent process runs inside a virtual-env (e.g. the VS Code
    extension's own ``.venv``), the ``VIRTUAL_ENV`` variable leaks into
    child processes and causes ``uv run`` to emit a spurious warning about
    a mismatched environment. Stripping it lets ``uv`` (and other tools)
    discover the correct project ``.venv`` on their own.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
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
    ) -> None:
        self.stream_callback = stream_callback
        self.stop_event = stop_event

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
            resolved = Path(file_path).resolve()
            text = resolved.read_text()
            lines = text.splitlines(keepends=True)
            if len(lines) > max_lines:
                return (
                    "".join(lines[:max_lines])
                    + f"\n[truncated: {len(lines) - max_lines} more lines]"
                )
            return text
        except Exception as e:
            _log_exc()
            return f"Error: {e}"

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
            resolved = Path(file_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            _log_exc()
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
            resolved = Path(file_path).resolve()
            if not resolved.is_file():
                return f"Error: File not found: {file_path}"
            if old_string == new_string:
                return "Error: new_string must be different from old_string"
            content = resolved.read_text()
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
            resolved.write_text(new_content)
            replaced = count if replace_all else 1
            return f"Successfully replaced {replaced} occurrence(s) in {file_path}"
        except Exception as e:
            _log_exc()
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

        env = _clean_env()

        try:
            process = subprocess.Popen(
                **_popen_kwargs(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            done = threading.Event()
            monitor = None
            if self.stop_event:
                monitor = threading.Thread(
                    target=_stop_monitor,
                    args=(self.stop_event, process, done),
                    daemon=True,
                )
                monitor.start()
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
            _log_exc()
            return f"Error: {e}"

    def _bash_streaming(self, command: str, timeout_seconds: float, max_output_chars: int) -> str:
        assert self.stream_callback is not None
        process = subprocess.Popen(
            **_popen_kwargs(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_env(),
        )
        timed_out = False
        done = threading.Event()

        def _kill() -> None:
            nonlocal timed_out
            timed_out = True
            _kill_process_group(process)

        timer = threading.Timer(timeout_seconds, _kill)
        timer.start()
        monitor = None
        if self.stop_event:
            monitor = threading.Thread(
                target=_stop_monitor,
                args=(self.stop_event, process, done),
                daemon=True,
            )
            monitor.start()
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
