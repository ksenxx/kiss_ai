# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File tools (Read, Write, Edit) that execute inside a Docker container via bash."""

import base64
import shlex
from collections.abc import Callable

#: Marker the Write script echoes only after the bytes really landed.
#: Success must be proven positively: the bash function's *other* failure
#: return — ``"Error: command timed out after Ns"`` — carries no
#: ``[exit code:`` marker, so sniffing for that marker reported a write
#: that never happened as a success.
_WRITE_OK = "__KISS_WRITE_OK__"
#: Marker the Edit script prints when the container has no Python interpreter.
_NO_PYTHON = "Error: Python required for Edit"


def _perl_edit_command(file_path: str, old_string: str, new_string: str, replace_all: bool) -> str:
    """The Edit replacement as a Perl one-liner, for containers without Python.

    Args:
        file_path: Absolute path to the file to modify.
        old_string: Exact text to find and replace.
        new_string: Replacement text.
        replace_all: If True, replace all occurrences.

    Returns:
        A bash command whose output matches the Python implementation's messages.
    """
    ra = "1" if replace_all else "0"
    return (
        f"KISS_OLD={shlex.quote(old_string)} KISS_NEW={shlex.quote(new_string)} perl -e '\n"
        f"my $old = $ENV{{KISS_OLD}}; my $new = $ENV{{KISS_NEW}};\n"
        f"my $path = $ARGV[0];\n"
        f"open(my $fh, \"<:raw\", $path)\n"
        f"  or do {{ print \"Error: File not found: $path\\n\"; exit 1 }};\n"
        f"local $/; my $content = <$fh>; close $fh;\n"
        f"my $count = () = $content =~ /\\Q$old\\E/g;\n"
        f"if ($count == 0) {{ print \"Error: String not found in file\\n\"; exit 1 }}\n"
        f"if (!{ra} && $count > 1) {{ print \"Error: String appears $count times (not unique). "
        f"Use replace_all=True to replace all occurrences.\\n\"; exit 1 }}\n"
        f"if ({ra}) {{ $content =~ s/\\Q$old\\E/$new/g }}\n"
        f"else {{ $content =~ s/\\Q$old\\E/$new/ }}\n"
        f"open($fh, \">:raw\", $path) or do {{ print \"Error: cannot write $path\\n\"; exit 1 }};\n"
        f"print $fh $content; close $fh;\n"
        f"my $replaced = {ra} ? $count : 1;\n"
        f"print \"Successfully replaced $replaced occurrence(s) in $path\\n\";\n"
        f"' {shlex.quote(file_path)}"
    )


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
        start_line: int = 1,
    ) -> str:
        """Read file contents.

        Args:
            file_path: Path to the file; a relative path resolves against
                the container's working directory (the task's work dir in
                a container kiss started from an image).
            max_lines: Maximum number of lines to return.
            start_line: 1-indexed line at which to begin the returned
                window.  ``start_line=1`` (the default) reads from the
                top of the file and is backward-compatible.  Values
                less than 1 are rejected; values beyond EOF return an
                explicit sentinel rather than empty content.
        """
        if start_line < 1:
            return (
                f"Error: start_line must be >= 1 (got {start_line}); the "
                f"parameter is 1-indexed."
            )
        if max_lines < 1:
            return f"Error: max_lines must be >= 1 (got {max_lines})."
        path = shlex.quote(file_path)
        cmd = (
            f'FILE={path}\n'
            f'if [ ! -f "$FILE" ]; then echo "Error: File not found: $FILE"; exit 1; fi\n'
            f'TOTAL=$(awk \'END{{print NR}}\' "$FILE")\n'
            f'if [ "$TOTAL" -eq 0 ]; then echo "(file is empty)"; exit 0; fi\n'
            f'START={start_line}\n'
            f'MAX={max_lines}\n'
            f'if [ "$START" -gt "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then\n'
            f'  echo "Error: start_line=$START is past EOF (file has $TOTAL lines)."\n'
            f'  exit 0\n'
            f'fi\n'
            f'tail -n +"$START" "$FILE" | head -n "$MAX"\n'
            f'REMAINING=$((TOTAL - START + 1 - MAX))\n'
            f'if [ "$REMAINING" -gt 0 ]; then\n'
            f'  echo "[truncated: $REMAINING more lines]"\n'
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
        cmd = (
            f'mkdir -p "$(dirname {path})" && base64 -d > {path} '
            f'<< \'KISS_B64_EOF\' && echo {_WRITE_OK}\n'
            f'{encoded}\n'
            f'KISS_B64_EOF'
        )
        result = self.bash(cmd, f"Write {file_path}")
        if _WRITE_OK not in result:
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
            file_path: Path to the file to modify; a relative path resolves
                against the container's working directory (the task's work
                dir in a container kiss started from an image).
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.
        """
        if old_string == "":
            return (
                "Error: old_string must not be empty. "
                "Use the Write tool to create or overwrite a file."
            )
        if old_string == new_string:
            return "Error: new_string must be different from old_string"
        b64_old = base64.b64encode(old_string.encode()).decode()
        b64_new = base64.b64encode(new_string.encode()).decode()
        path = shlex.quote(file_path)
        ra = "True" if replace_all else "False"

        cmd = (
            f'PYTHON=$(command -v python3 || command -v python) || '
            f'{{ echo "{_NO_PYTHON}"; exit 1; }}; '
            f'"$PYTHON" -c "\n'
            f"import base64, sys\n"
            f"old = base64.b64decode('{b64_old}').decode()\n"
            f"new = base64.b64decode('{b64_new}').decode()\n"
            f"if old == new:\n"
            f"    print('Error: new_string must be different from old_string'); sys.exit(1)\n"
            f"path = sys.argv[1]\n"
            f"try:\n"
            f"    content = open(path, encoding='utf-8').read()\n"
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
            f"open(path, 'w', encoding='utf-8').write(new_content)\n"
            f"replaced = count if ra else 1\n"
            f"print(f'Successfully replaced {{replaced}} occurrence(s) in {{path}}')\n"
            f'" {path}'
        )
        result = self.bash(cmd, f"Edit {file_path}")
        if _NO_PYTHON not in result:
            return result
        # Images without Python (Rust, R, C toolchains) still have perl-base, so
        # the same replacement is done in Perl.  The strings travel in
        # environment variables (perl-base has no MIME::Base64), which cannot
        # carry NUL bytes.
        if "\0" in old_string or "\0" in new_string:
            return "Error: Edit needs Python in this container for strings containing NUL bytes"
        command = _perl_edit_command(file_path, old_string, new_string, replace_all)
        return self.bash(command, f"Edit {file_path}")
