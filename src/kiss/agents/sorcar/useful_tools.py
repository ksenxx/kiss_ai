"""Useful tools for agents: file editing and bash execution."""

import difflib
import hashlib
import locale
import logging
import os
import re
import shlex
import signal
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
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


def _read_text_with_fallback(path: Path) -> str:
    """Read a text file with UTF-8 first and local-encoding fallback."""
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError("Binary file cannot be displayed as text")

    encodings: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", locale.getpreferredencoding(False) or "utf-8"):
        if encoding not in encodings:
            encodings.append(encoding)

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            logger.debug("Exception caught", exc_info=True)

    return raw.decode(encodings[-1], errors="replace")


@dataclass(frozen=True)
class _ReadSnapshot:
    content_hash: str
    size_bytes: int
    line_count: int


@dataclass(frozen=True)
class _EditOutcome:
    updated_text: str
    replaced_count: int


def _hash_bytes(raw: bytes) -> str:
    return hashlib.md5(raw).hexdigest()


def _count_lines(text: str) -> int:
    return len(text.splitlines())


def _read_text_and_snapshot(path: Path) -> tuple[str, _ReadSnapshot]:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError("Binary file cannot be displayed as text")

    encodings: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", locale.getpreferredencoding(False) or "utf-8"):
        if encoding not in encodings:
            encodings.append(encoding)

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            logger.debug("Exception caught", exc_info=True)
    else:
        text = raw.decode(encodings[-1], errors="replace")

    return text, _ReadSnapshot(
        content_hash=_hash_bytes(raw),
        size_bytes=len(raw),
        line_count=_count_lines(text),
    )


def _write_text_preserving_newlines(path: Path, content: str) -> None:
    with path.open("w", newline="") as f:
        f.write(content)


def _compute_edit_outcome(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> _EditOutcome:
    occurrence_count = content.count(old_string)
    if replace_all:
        if occurrence_count == 0:
            raise ValueError("String not found in file")
        updated = content.replace(old_string, new_string)
        return _EditOutcome(updated_text=updated, replaced_count=occurrence_count)

    if occurrence_count == 0:
        raise ValueError("String not found in file")
    if occurrence_count > 1:
        raise ValueError(
            f"String appears {occurrence_count} times (not unique)\n"
            "Hint: Use replace_all=true to replace all occurrences"
        )
    updated = content.replace(old_string, new_string, 1)
    return _EditOutcome(updated_text=updated, replaced_count=1)


def _run_with_timeout(
    timeout_seconds: float,
    operation: Callable[[], _EditOutcome],
) -> _EditOutcome | None:
    result: dict[str, _EditOutcome] = {}
    error: dict[str, Exception] = {}

    def _worker() -> None:
        try:
            result["value"] = operation()
        except Exception as exc:  # pragma: no cover - exercised via caller
            error["value"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return None
    if "value" in error:
        raise error["value"]
    return result["value"]


def _format_changed_section(updated_text: str, needle: str) -> str:
    lines = updated_text.splitlines()
    matched = [f"{idx}:{line}" for idx, line in enumerate(lines, start=1) if needle in line]
    if not matched:
        matched = ["(No context available)"]
    if len(matched) > 8:
        matched = matched[:8] + [f"... [{len(matched) - 8} more matching lines]"]
    return "\n".join(
        [
            "Changed section:",
            "----------------------------------------",
            *matched,
            "----------------------------------------",
        ]
    )


def _count_changed_lines(old_text: str, new_text: str) -> int:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _detect_shell_prefix(
    *,
    os_name: str | None = None,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return the shell invocation prefix for command execution.

    On Windows, prefer Git Bash when present because many agent commands use
    bash syntax. Fallback is cmd.exe for baseline compatibility.
    """
    resolved_os = os_name or os.name
    env = environ if environ is not None else os.environ

    if resolved_os == "nt":
        # Prefer Git Bash, not the WSL launcher at System32\bash.exe.
        program_roots = [
            env.get("ProgramW6432"),
            env.get("ProgramFiles"),
            env.get("ProgramFiles(x86)"),
        ]
        for root in program_roots:
            if not root:
                continue
            for rel in ("Git\\bin\\bash.exe", "Git\\usr\\bin\\bash.exe"):
                candidate = os.path.join(root, rel)
                if os.path.isfile(candidate):
                    return [candidate, "-lc"]

        bash_on_path = which("bash")
        if bash_on_path and "windows\\system32\\bash.exe" not in bash_on_path.lower():
            return [bash_on_path, "-lc"]
        return [env.get("COMSPEC", "cmd.exe"), "/c"]

    shell = env.get("SHELL") or which("sh") or "sh"
    return [shell, "-c"]


def _normalize_windows_drive_path(path: str) -> str:
    """Normalize Git-Bash/WSL-style Windows drive paths to native Windows format."""
    if os.name != "nt":
        return path
    normalized = path.replace("\\", "/")
    match = re.match(r"^/(?:mnt/)?([a-zA-Z])(?:/(.*))?$", normalized)
    if not match:
        return path
    drive = match.group(1).upper()
    rest = match.group(2) or ""
    if not rest:
        return f"{drive}:{os.sep}"
    return f"{drive}:{os.sep}{rest.replace('/', os.sep)}"


def _resolve_tool_path(file_path: str) -> Path:
    normalized = _normalize_windows_drive_path(file_path)
    if os.name == "nt" and file_path.startswith("/") and normalized == file_path:
        raise ValueError(f"Invalid path on Windows: {file_path}")
    return Path(normalized).resolve()


def _to_git_bash_drive_path(path: str) -> str:
    """Convert a native Windows drive path to a Git-Bash-compatible path."""
    normalized = _normalize_windows_drive_path(path)
    match = re.match(r"^([a-zA-Z]):(?:[\\/](.*))?$", normalized)
    if not match:
        return normalized.replace("\\", "/")
    drive = match.group(1).lower()
    tail = (match.group(2) or "").replace("\\", "/")
    return f"/{drive}/{tail}" if tail else f"/{drive}"


def _rewrite_windows_bash_command(command: str, shell_prefix: list[str]) -> str:
    """Rewrite common cmd.exe-only path patterns before running under Windows bash."""
    if os.name != "nt" or not shell_prefix or shell_prefix[-1] != "-lc":
        return command

    match = re.match(
        r"^(?P<prefix>\s*)cd\s+/d\s+(?P<path>(?:\"[^\"]+\"|'[^']+'|[^&;|]+?))(?P<suffix>\s*(?:&&|\|\||;).*)?$",
        command,
        flags=re.DOTALL,
    )
    if not match:
        return command

    raw_path = match.group("path").strip()
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in {"'", '"'}:
        raw_path = raw_path[1:-1]
    rewritten_path = shlex.quote(_to_git_bash_drive_path(raw_path))
    suffix = match.group("suffix") or ""
    return f"{match.group('prefix')}cd {rewritten_path}{suffix}"


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
        match = _REDIRECT_RE.match(token)
        if match:
            if match.end() < len(token):
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
        match = pattern.match(command, i)
        if match:
            segments.append("".join(current))
            current = []
            i = match.end()
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
        r"<<-?\s*['\"]?(\w+)['\"]?[^\r\n]*\r?\n(?:.*?\r?\n)*?[ \t]*\1[ \t]*(?=\r?\n|$)",
        "",
        command,
        flags=re.DOTALL,
    )


def _format_bash_result(returncode: int, output: str, max_output_chars: int) -> str:
    if returncode != 0:
        logger.debug("Bash command exited with code %s", returncode)
        msg = f"Error: command failed\nError (exit code {returncode}):"
        if output:
            msg += f"\n{output}"
        return _truncate_output(msg, max_output_chars)
    return _truncate_output(output, max_output_chars)


def _kill_process_group(process: subprocess.Popen) -> None:
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        else:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
            return

    try:
        process.kill()
    except OSError:  # pragma: no cover
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    LARGE_OVERWRITE_LINE_THRESHOLD = 80
    LARGE_OVERWRITE_RATIO_THRESHOLD = 0.25

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> None:
        self.stream_callback = stream_callback
        self.ask_user_question_callback = ask_user_question_callback
        self._read_snapshots: dict[Path, _ReadSnapshot] = {}
        # Resolve shell once at startup so behavior is stable for this session.
        self.shell_prefix = _detect_shell_prefix()

    def _load_tracked_file(
        self,
        resolved: Path,
        action_name: str,
    ) -> tuple[str, _ReadSnapshot] | str:
        if not resolved.exists():
            return f"Error: File not found: {resolved}"
        if not resolved.is_file():
            return f"Error: Path is not a file: {resolved}"

        previous_snapshot = self._read_snapshots.get(resolved)
        if previous_snapshot is None:
            return (
                f"Error: File must be read with Read() before {action_name}(). "
                f"Read {resolved} and try again."
            )

        text, current_snapshot = _read_text_and_snapshot(resolved)
        if current_snapshot.content_hash != previous_snapshot.content_hash:
            return (
                f"Error: File changed since last Read(); read {resolved} again "
                f"before {action_name}()."
            )
        return text, current_snapshot

    def _confirm_large_overwrite(
        self,
        resolved: Path,
        changed_lines: int,
        total_lines: int,
    ) -> str | None:
        if self.ask_user_question_callback is None:
            return (
                "Error: Overwrite() refused a large rewrite and no "
                "ask_user_question_callback is available for confirmation"
            )

        question = (
            f"Confirm large overwrite of {resolved}? "
            f"Estimated changed lines: {changed_lines} of {total_lines}. "
            "Reply yes to continue."
        )
        answer = self.ask_user_question_callback(question).strip().lower()
        if answer not in {"y", "yes"}:
            return "Error: Overwrite cancelled by user"
        return None

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
            resolved = _resolve_tool_path(file_path)
            text, snapshot = _read_text_and_snapshot(resolved)
            self._read_snapshots[resolved] = snapshot
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
        """Write content to a new file.

        Args:
            file_path: Path to the file to write.
            content: The full content to write to the file. The file must not exist yet.
        """
        try:
            resolved = _resolve_tool_path(file_path)
            if resolved.exists():
                return (
                    "Error: Write() only creates new files and will not overwrite "
                    f"existing paths: {resolved}. Use Overwrite() after Read()."
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _write_text_preserving_newlines(resolved, content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Overwrite(  # noqa: N802
        self,
        file_path: str,
        content: str,
    ) -> str:
        """Replace the full contents of an existing file after it has been read.

        Args:
            file_path: Absolute path to the file to replace.
            content: The full replacement content for the file.
        """
        try:
            resolved = _resolve_tool_path(file_path)
            tracked = self._load_tracked_file(resolved, "Overwrite")
            if isinstance(tracked, str):
                return tracked

            current_text, current_snapshot = tracked
            new_line_count = _count_lines(content)
            changed_lines = _count_changed_lines(current_text, content)
            max_lines = max(current_snapshot.line_count, new_line_count)
            is_large_rewrite = (
                changed_lines > self.LARGE_OVERWRITE_LINE_THRESHOLD
                or (
                    max_lines > 0
                    and changed_lines / max_lines > self.LARGE_OVERWRITE_RATIO_THRESHOLD
                )
            )
            if is_large_rewrite:
                confirmation_error = self._confirm_large_overwrite(
                    resolved,
                    changed_lines,
                    max_lines,
                )
                if confirmation_error is not None:
                    return confirmation_error

            _write_text_preserving_newlines(resolved, content)
            return f"Successfully overwrote {file_path} with {len(content)} characters"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        timeout_seconds: float = 30,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.
            timeout_seconds: Timeout in seconds for the edit command.

        Returns:
            The output of the edit operation.
        """
        try:
            resolved = _resolve_tool_path(file_path)
            if old_string == new_string:
                return "Error: new_string must be different from old_string"
            if timeout_seconds < 0.001:
                return "Error: Command execution timeout"
            tracked = self._load_tracked_file(resolved, "Edit")
            if isinstance(tracked, str):
                return tracked

            current_text, _ = tracked
            outcome = _run_with_timeout(
                timeout_seconds,
                lambda: _compute_edit_outcome(
                    current_text,
                    old_string,
                    new_string,
                    replace_all,
                ),
            )
            if outcome is None:
                return "Error: Command execution timeout"

            _write_text_preserving_newlines(resolved, outcome.updated_text)
            count = current_text.count(old_string)
            parts = [
                f"File: {resolved}",
                f"Looking for: '{old_string}'",
                f"Replacing with: '{new_string}'",
                f"Occurrences found: {count}",
                f"Replace all: {str(replace_all).lower()}",
                "",
            ]
            if outcome.replaced_count == 1:
                parts.append("Successfully replaced 1 occurrence")
            else:
                parts.append(f"Successfully replaced {outcome.replaced_count} occurrence(s)")
            parts.append("")
            parts.append(_format_changed_section(outcome.updated_text, new_string))
            return "\n".join(parts)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _start_bash_process(self, command: str) -> subprocess.Popen[str]:
        command = _rewrite_windows_bash_command(command, self.shell_prefix)
        if os.name == "nt":
            return subprocess.Popen(
                [*self.shell_prefix, command],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

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
            process = self._start_bash_process(command)
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.debug("Exception caught", exc_info=True)
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:  # pragma: no cover
                    pass
                return "Error: Command execution timeout"
            except BaseException:
                _kill_process_group(process)
                try:
                    process.communicate(timeout=5)
                except Exception:  # pragma: no cover
                    pass
                raise
            return _format_bash_result(process.returncode, stdout, max_output_chars)
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _bash_streaming(
        self,
        command: str,
        timeout_seconds: float,
        max_output_chars: int,
    ) -> str:
        assert self.stream_callback is not None
        process = self._start_bash_process(command)
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
            except subprocess.TimeoutExpired:  # pragma: no cover
                _kill_process_group(process)
        except BaseException:
            _kill_process_group(process)
            raise
        finally:
            timer.cancel()
            if process.stdout is not None:
                process.stdout.close()

        if timed_out:
            logger.debug("Bash command timed out while streaming")
            return "Error: Command execution timeout"

        output = "".join(chunks)
        return _format_bash_result(process.returncode, output, max_output_chars)
