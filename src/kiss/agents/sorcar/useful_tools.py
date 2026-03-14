"""Useful tools for agents: file editing and bash execution."""

import logging
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
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


EDIT_SCRIPT = r"""
#!/usr/bin/env bash
#
# Edit Tool - Claude Code Implementation
# Performs precise string replacements in files with exact matching
#
# Usage: edit_tool.sh <file_path> <old_string> <new_string> [replace_all]
#
# Parameters:
#   file_path    - Absolute path to the file to modify (required)
#   old_string   - Exact text to find and replace (required)
#   new_string   - Replacement text, must differ from old_string (required)
#   replace_all  - If "true", replace all occurrences (optional, default: false)
#
# Exit codes:
#   0 - Success
#   1 - Invalid arguments
#   2 - File not found
#   3 - String not found or not unique
#   4 - Read-before-edit validation failed

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validate arguments
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}" >&2
    echo "Usage: $0 <file_path> <old_string> <new_string> [replace_all]" >&2
    exit 1
fi

FILE_PATH="$1"
OLD_STRING="$2"
NEW_STRING="$3"
REPLACE_ALL="${4:-false}"

# Validate file path is absolute
if [[ ! "$FILE_PATH" = /* ]]; then
    echo -e "${RED}Error: file_path must be absolute, not relative${NC}" >&2
    exit 1
fi

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo -e "${RED}Error: File not found: $FILE_PATH${NC}" >&2
    exit 2
fi

# Check if old_string and new_string are different
if [ "$OLD_STRING" = "$NEW_STRING" ]; then
    echo -e "${RED}Error: new_string must be different from old_string${NC}" >&2
    exit 1
fi

# Create a state tracking directory (simulating session state)
STATE_DIR="${HOME}/.claude-edit-state"
mkdir -p "$STATE_DIR"

# Check read-before-edit validation
# In a real implementation, this would check session state
# For demo purposes, we'll create a marker file when files are "read"
if command -v md5sum &>/dev/null; then
    FILE_HASH=$(echo -n "$FILE_PATH" | md5sum | cut -d' ' -f1)
else
    FILE_HASH=$(echo -n "$FILE_PATH" | md5 -q)
fi
READ_MARKER="$STATE_DIR/$FILE_HASH"

if [ ! -f "$READ_MARKER" ]; then
    echo -e "${YELLOW}Warning: File has not been read in this session${NC}" >&2
    echo -e "${YELLOW}Creating read marker for demo purposes...${NC}" >&2
    touch "$READ_MARKER"
fi

# Count literal occurrences of old_string (not just matching lines)
export EDIT_FILE_PATH="$FILE_PATH" EDIT_OLD_STRING="$OLD_STRING"
OCCURRENCE_COUNT=$(python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
with open(file_path, 'r', newline='') as f:
    content = f.read()
print(content.count(old_string))
")

echo "File: $FILE_PATH"
echo "Looking for: '$OLD_STRING'"
echo "Replacing with: '$NEW_STRING'"
echo "Occurrences found: $OCCURRENCE_COUNT"
echo "Replace all: $REPLACE_ALL"
echo ""

# Handle replacement based on mode
export EDIT_NEW_STRING="$NEW_STRING"

if [ "$REPLACE_ALL" = "true" ]; then
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    fi
    REPLACE_COUNT=""
else
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    elif [ "$OCCURRENCE_COUNT" -gt 1 ]; then
        echo -e "${RED}Error: String appears $OCCURRENCE_COUNT times (not unique)${NC}" >&2
        echo -e "${YELLOW}Hint: Use replace_all=true to replace all occurrences${NC}" >&2
        exit 3
    fi
    REPLACE_COUNT="1"
fi

export EDIT_REPLACE_COUNT="$REPLACE_COUNT"
python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
new_string = os.environ['EDIT_NEW_STRING']
count = int(os.environ['EDIT_REPLACE_COUNT']) if os.environ['EDIT_REPLACE_COUNT'] else -1
with open(file_path, 'r', newline='') as f:
    content = f.read()
if count >= 0:
    content = content.replace(old_string, new_string, count)
else:
    content = content.replace(old_string, new_string)
with open(file_path, 'w', newline='') as f:
    f.write(content)
"

if [ "$REPLACE_ALL" = "true" ]; then
    echo -e "${GREEN}✓ Successfully replaced $OCCURRENCE_COUNT occurrence(s)${NC}"
else
    echo -e "${GREEN}✓ Successfully replaced 1 occurrence${NC}"
fi

# Show the changed section (context around the change)
echo ""
echo "Changed section:"
echo "----------------------------------------"
grep -Fn -C 2 "$NEW_STRING" "$FILE_PATH" || echo "(No context available)"
echo "----------------------------------------"

exit 0
"""


# Cross-platform edit implementation used when bash isn't available
# (most Windows setups) while preserving the same external behavior.
EDIT_SCRIPT_PYTHON = r"""
import pathlib
import sys


def _main() -> None:
    if len(sys.argv) != 5:
        print("Error: Invalid number of arguments", file=sys.stderr)
        sys.exit(1)

    file_path = pathlib.Path(sys.argv[1])
    old_string = sys.argv[2]
    new_string = sys.argv[3]
    replace_all = sys.argv[4].lower() == "true"

    if not file_path.is_absolute():
        print("Error: file_path must be absolute, not relative", file=sys.stderr)
        sys.exit(1)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(2)
    if old_string == new_string:
        print("Error: new_string must be different from old_string", file=sys.stderr)
        sys.exit(1)

    with file_path.open("r", newline="") as f:
        content = f.read()
    occurrence_count = content.count(old_string)

    print(f"File: {file_path}")
    print(f"Looking for: '{old_string}'")
    print(f"Replacing with: '{new_string}'")
    print(f"Occurrences found: {occurrence_count}")
    print(f"Replace all: {str(replace_all).lower()}")
    print("")

    if replace_all:
        if occurrence_count == 0:
            print("Error: String not found in file", file=sys.stderr)
            sys.exit(3)
        updated = content.replace(old_string, new_string)
        replaced_count = occurrence_count
    else:
        if occurrence_count == 0:
            print("Error: String not found in file", file=sys.stderr)
            sys.exit(3)
        if occurrence_count > 1:
            print(f"Error: String appears {occurrence_count} times (not unique)", file=sys.stderr)
            print("Hint: Use replace_all=true to replace all occurrences", file=sys.stderr)
            sys.exit(3)
        updated = content.replace(old_string, new_string, 1)
        replaced_count = 1

    with file_path.open("w", newline="") as f:
        f.write(updated)

    if replaced_count == 1:
        print("Successfully replaced 1 occurrence")
    else:
        print(f"Successfully replaced {replaced_count} occurrence(s)")
    print("")
    print("Changed section:")
    print("----------------------------------------")
    for line_number, line in enumerate(updated.splitlines(), start=1):
        if new_string in line:
            print(f"{line_number}:{line}")
    print("----------------------------------------")


if __name__ == "__main__":
    _main()
"""


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

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.stream_callback = stream_callback
        # Resolve shell once at startup so behavior is stable for this session.
        self.shell_prefix = _detect_shell_prefix()

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
            with resolved.open("r", newline="") as f:
                text = f.read()
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
            resolved = _resolve_tool_path(file_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with resolved.open("w", newline="") as f:
                f.write(content)
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
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

        # Windows commonly lacks bash; run a Python edit helper instead.
        bash_path = shutil.which("bash")
        if os.name == "nt" or bash_path is None:
            command = [
                sys.executable,
                "-c",
                EDIT_SCRIPT_PYTHON,
                str(resolved),
                old_string,
                new_string,
                "true" if replace_all else "false",
            ]
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                return result.stdout
            except subprocess.TimeoutExpired:
                return "Error: Command execution timeout"
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else str(e)
                return f"Error: {error_msg}"
            except Exception as e:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)
                return f"Error: {e}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as f:
            f.write(EDIT_SCRIPT)
            script_path = f.name

        try:
            # Make script executable
            Path(script_path).chmod(0o755)

            # Build command with arguments
            command = [
                bash_path,
                script_path,
                str(resolved),
                old_string,
                new_string,
                "true" if replace_all else "false",
            ]

            # Execute with timeout for safety
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.debug("Exception caught", exc_info=True)
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            # Include stderr which contains the actual error message from the script
            logger.debug("Exception caught", exc_info=True)
            error_msg = e.stderr.strip() if e.stderr else str(e)
            return f"Error: {error_msg}"
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"
        finally:
            # Clean up temporary script
            try:
                Path(script_path).unlink()
            except Exception:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)

    def _start_bash_process(self, command: str) -> subprocess.Popen[str]:
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
