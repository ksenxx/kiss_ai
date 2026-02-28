"""Useful tools for agents: file editing, bash execution, web search, and URL fetching."""

import re
import shlex
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

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
with open(file_path, 'r') as f:
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
if [ "$REPLACE_ALL" = "true" ]; then
    # Replace all occurrences
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    fi

    # Use python for literal string replacement (handles special chars)
    # Pass strings via environment variables to handle embedded quotes safely
    export EDIT_FILE_PATH="$FILE_PATH" EDIT_OLD_STRING="$OLD_STRING"
    export EDIT_NEW_STRING="$NEW_STRING"
    python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
new_string = os.environ['EDIT_NEW_STRING']
with open(file_path, 'r') as f:
    content = f.read()
content = content.replace(old_string, new_string)
with open(file_path, 'w') as f:
    f.write(content)
"

    echo -e "${GREEN}✓ Successfully replaced $OCCURRENCE_COUNT occurrence(s)${NC}"

else
    # Single replacement mode - requires exactly one occurrence
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    elif [ "$OCCURRENCE_COUNT" -gt 1 ]; then
        echo -e "${RED}Error: String appears $OCCURRENCE_COUNT times (not unique)${NC}" >&2
        echo -e "${YELLOW}Hint: Use replace_all=true to replace all occurrences${NC}" >&2
        exit 3
    fi

    # Exactly one occurrence - safe to replace
    # Pass strings via environment variables to handle embedded quotes safely
    export EDIT_FILE_PATH="$FILE_PATH" EDIT_OLD_STRING="$OLD_STRING"
    export EDIT_NEW_STRING="$NEW_STRING"
    python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
new_string = os.environ['EDIT_NEW_STRING']
with open(file_path, 'r') as f:
    content = f.read()
content = content.replace(old_string, new_string, 1)
with open(file_path, 'w') as f:
    f.write(content)
"

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


DISALLOWED_BASH_COMMANDS = {
    ".",
    "env",
    "eval",
    "exec",
}


def _extract_leading_command_name(part: str) -> str | None:
    try:
        tokens = shlex.split(part)
    except ValueError:
        return None
    if not tokens:
        return None

    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", tokens[i]):
        i += 1
    if i >= len(tokens):
        return None
    return tokens[i].split("/")[-1]


def _extract_command_names(command: str) -> list[str]:
    names: list[str] = []
    stripped_command = _strip_heredocs(command)
    segments = re.split(r"&&|\|\||;", stripped_command)
    for segment in segments:
        for part in re.split(r"(?<!>)\|(?!\|)", segment):
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
        r"<<-?\s*'?\"?(\w+)'?\"?\s*\n.*?\n\s*\1\b",
        "",
        command,
        flags=re.DOTALL,
    )


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
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except Exception as e:
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

        resolved = Path(file_path).resolve()

        # Create a temporary script file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(EDIT_SCRIPT)
            script_path = f.name

        try:
            # Make script executable
            Path(script_path).chmod(0o755)

            # Build command with arguments
            replace_all_str = "true" if replace_all else "false"
            command = [
                "/bin/bash",
                script_path,
                str(resolved),
                old_string,
                new_string,
                replace_all_str,
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
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            # Include stderr which contains the actual error message from the script
            error_msg = e.stderr.strip() if e.stderr else str(e)
            return f"Error: {error_msg}"
        except Exception as e:  # pragma: no cover
            return f"Error: {e}"
        finally:
            # Clean up temporary script
            try:
                Path(script_path).unlink()
            except Exception:  # pragma: no cover
                pass

    def MultiEdit(  # noqa: N802
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
        return self.Edit(file_path, old_string, new_string, replace_all, timeout_seconds)

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
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = result.stdout
            if len(output) > max_output_chars:
                half = max_output_chars // 2
                output = (
                    output[:half]
                    + f"\n\n... [truncated {len(output) - max_output_chars} chars] ...\n\n"
                    + output[-half:]
                )
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"
        except Exception as e:  # pragma: no cover
            return f"Error: {e}"

    def _bash_streaming(
        self, command: str, timeout_seconds: float, max_output_chars: int
    ) -> str:
        assert self.stream_callback is not None
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = False

        def _kill() -> None:
            nonlocal timed_out
            timed_out = True
            process.kill()

        timer = threading.Timer(timeout_seconds, _kill)
        timer.start()
        try:
            chunks: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                chunks.append(line)
                self.stream_callback(line)
            process.wait()
        finally:
            timer.cancel()

        if timed_out:
            return "Error: Command execution timeout"

        output = "".join(chunks)

        if process.returncode != 0:
            return f"Error: {subprocess.CalledProcessError(process.returncode, command)}"

        if len(output) > max_output_chars:
            half = max_output_chars // 2
            output = (
                output[:half]
                + f"\n\n... [truncated {len(output) - max_output_chars} chars] ...\n\n"
                + output[-half:]
            )
        return output
