"""File tools (Read, Write, Edit) that execute inside a Docker container via bash."""

import base64
import shlex
from collections.abc import Callable


class DockerTools:
    """File tools that execute inside a Docker container via bash.

    Each method generates a shell command and executes it via the provided
    bash function (typically DockerManager.Bash or RelentlessAgent._docker_bash).
    """

    def __init__(self, bash_fn: Callable[[str, str], str]) -> None:
        """Initialize with a bash execution function.

        Args:
            bash_fn: Callable(command, description) -> output string.
                     Executes a bash command inside the Docker container.
        """
        self.bash = bash_fn

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
        path = shlex.quote(file_path)
        cmd = (
            f'FILE={path}\n'
            f'if [ ! -f "$FILE" ]; then echo "Error: File not found: $FILE"; exit 1; fi\n'
            f'TOTAL=$(wc -l < "$FILE")\n'
            f'head -n {max_lines} "$FILE"\n'
            f'if [ "$TOTAL" -gt {max_lines} ]; then\n'
            f'  echo "[truncated: $((TOTAL - {max_lines})) more lines]"\n'
            f'fi'
        )
        return self.bash(cmd, f"Read {file_path}")

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
        encoded = base64.b64encode(content.encode()).decode()
        path = shlex.quote(file_path)
        # Use heredoc to avoid ARG_MAX limits for large files
        cmd = (
            f'mkdir -p "$(dirname {path})" && base64 -d > {path} << \'KISS_B64_EOF\'\n'
            f'{encoded}\n'
            f'KISS_B64_EOF'
        )
        result = self.bash(cmd, f"Write {file_path}")
        # DockerManager.Bash appends "[exit code: N]" on non-zero exit
        if "[exit code:" in result:
            return result
        return f"Successfully wrote {len(content)} characters to {file_path}"

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
        """
        b64_old = base64.b64encode(old_string.encode()).decode()
        b64_new = base64.b64encode(new_string.encode()).decode()
        path = shlex.quote(file_path)
        ra = "True" if replace_all else "False"

        # Use python3 or python, whichever is available
        cmd = (
            f'PYTHON=$(command -v python3 || command -v python) || '
            f'{{ echo "Error: Python required for Edit"; exit 1; }}; '
            f'"$PYTHON" -c "\n'
            f"import base64, sys\n"
            f"old = base64.b64decode('{b64_old}').decode()\n"
            f"new = base64.b64decode('{b64_new}').decode()\n"
            f"if old == new:\n"
            f"    print('Error: new_string must be different from old_string'); sys.exit(1)\n"
            f"path = sys.argv[1]\n"
            f"try:\n"
            f"    content = open(path).read()\n"
            f"except FileNotFoundError:\n"
            f"    print(f'Error: File not found: {{path}}'); sys.exit(1)\n"
            f"count = content.count(old)\n"
            f"if count == 0:\n"
            f"    print('Error: String not found in file'); sys.exit(1)\n"
            f"ra = {ra}\n"
            f"if not ra and count > 1:\n"
            f"    print(f'Error: String appears {{count}} times (not unique). "
            f"Use replace_all=True to replace all occurrences.'); sys.exit(1)\n"
            f"new_content = content.replace(old, new) if ra else content.replace(old, new, 1)\n"
            f"open(path, 'w').write(new_content)\n"
            f"replaced = count if ra else 1\n"
            f"print(f'Successfully replaced {{replaced}} occurrence(s) in {{path}}')\n"
            f'" {path}'
        )
        return self.bash(cmd, f"Edit {file_path}")
