"""Useful tools for agents: file editing and bash execution."""

import logging
import os
import re
import shlex
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


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


DISALLOWED_BASH_COMMANDS = {
    ".",
    "env",
    "eval",
    "exec",
    "source",
}


_SHELL_PREFIX_TOKENS = frozenset(("!", "{", "}", "(", ")", "&"))
_REDIRECT_RE = re.compile(r"^[0-9]*[<>][<>&]*")


def _extract_leading_command_name(part: str) -> str | None:
    try:
        tokens = shlex.split(part)
    except ValueError:
        logger.debug("Exception caught", exc_info=True)
        return None
    if not tokens:
        return None

    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", tokens[i]):
        i += 1

    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_PREFIX_TOKENS:
            i += 1
            continue
        m = _REDIRECT_RE.match(token)
        if m:
            if m.end() < len(token):
                i += 1
            else:
                i += 2
            continue
        break

    if i >= len(tokens):
        return None
    name = tokens[i].lstrip("({")
    if not name:
        return None
    return name.split("/")[-1]


def _split_respecting_quotes(command: str, pattern: re.Pattern[str]) -> list[str]:
    """Split *command* on *pattern* while skipping quoted and escaped regions."""
    segments: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "\\":
            current.append(command[i : i + 2])
            i += 2
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < len(command):
                if command[j] == "\\" and quote == '"':
                    j += 2
                    continue
                if command[j] == quote:
                    j += 1
                    break
                j += 1
            current.append(command[i:j])
            i = j
            continue
        m = pattern.match(command, i)
        if m:
            segments.append("".join(current))
            current = []
            i = m.end()
            continue
        current.append(ch)
        i += 1
    segments.append("".join(current))
    return segments


_CONTROL_RE = re.compile(r"&&|\|\||;|\n|(?<![<>|&])&(?![&>])")
_PIPE_RE = re.compile(r"(?<!>)\|(?!\|)")


def _extract_command_names(command: str) -> list[str]:
    names: list[str] = []
    stripped_command = _strip_heredocs(command)
    segments = _split_respecting_quotes(stripped_command, _CONTROL_RE)
    for segment in segments:
        for part in _split_respecting_quotes(segment, _PIPE_RE):
            name = _extract_leading_command_name(part.strip())
            if name:
                names.append(name)
    return names


def _strip_heredocs(command: str) -> str:
    """Strip heredoc content from a bash command.

    Removes everything between << DELIM and DELIM (or <<- DELIM and DELIM),
    so that heredoc body text is not parsed as command arguments.
    """
    return re.sub(
        r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(?:.*?\n)*?[ \t]*\1[ \t]*(?=\n|$)",
        "",
        command,
        flags=re.DOTALL,
    )


def _format_bash_result(returncode: int, output: str, max_output_chars: int) -> str:
    if returncode != 0:
        msg = f"Error (exit code {returncode}):"
        if output:
            msg += f"\n{output}"
        return _truncate_output(msg, max_output_chars)
    return _truncate_output(output, max_output_chars)


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.stream_callback = stream_callback

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
            logger.debug("Exception caught", exc_info=True)
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
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: float = 30,
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

        for command_name in _extract_command_names(command):
            if command_name in DISALLOWED_BASH_COMMANDS:
                return f"Error: Command '{command_name}' is not allowed in Bash tool"

        if self.stream_callback:
            return self._bash_streaming(command, timeout_seconds, max_output_chars)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:
                    pass
                return "Error: Command execution timeout"
            except BaseException:
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:
                    pass
                raise
            return _format_bash_result(process.returncode, stdout, max_output_chars)
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _bash_streaming(self, command: str, timeout_seconds: float, max_output_chars: int) -> str:
        assert self.stream_callback is not None
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False

        def _kill() -> None:
            nonlocal timed_out
            timed_out = True
            _kill_process_group(process)

        timer = threading.Timer(timeout_seconds, _kill)
        timer.start()
        try:
            chunks: list[str] = []
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                chunks.append(line)
                self.stream_callback(line)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
        except BaseException:
            _kill_process_group(process)
            raise
        finally:
            timer.cancel()

        if timed_out:
            return "Error: Command execution timeout"

        output = "".join(chunks)
        return _format_bash_result(process.returncode, output, max_output_chars)
