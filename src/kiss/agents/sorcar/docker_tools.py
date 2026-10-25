# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File tools (Read, Write, Edit) that execute inside a Docker container via bash."""

import base64
import secrets
import shlex
from collections.abc import Callable

from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE
from kiss.agents.sorcar.useful_tools import _edit_pairs

#: Marker the Write script echoes only after the bytes really landed.
#: Success must be proven positively: the bash function's *other* failure
#: return — ``"Error: command timed out after Ns"`` — carries no
#: ``[exit code:`` marker, so sniffing for that marker reported a write
#: that never happened as a success.
_WRITE_OK = "__KISS_WRITE_OK__"
#: Marker the Edit script prints when the container has no Python interpreter.
_NO_PYTHON = "Error: Python required for Edit"
#: Payload bytes one shell command carries.  The whole command is a
#: single ``bash -c`` argument, which Linux caps at 128 KiB
#: (``MAX_ARG_STRLEN``); base64 grows the payload by a third, so this
#: many raw bytes leave room for the script around them.
_CHUNK_BYTES = 72_000


def _store_commands(staging: str, target: str, data: bytes) -> list[str]:
    """Return the shell commands that together store *data* at *target*.

    Each command carries one base64 piece of at most :data:`_CHUNK_BYTES`
    bytes, so it fits a single argv string whatever the payload size.
    The pieces accumulate in *staging*, a file beside the target that
    nobody else names, and the last command publishes it with one
    ``mv`` (keeping the mode of the file it replaces), so a reader of
    the target sees the previous file or the whole new one and a second
    writer of the same target cannot splice its bytes into this one.
    Each command echoes :data:`_WRITE_OK` once its piece landed and
    nothing (non-zero status) when it did not.

    Args:
        staging: The staging path, already shell-quoted.
        target: The target path, already shell-quoted.
        data: The bytes to store; empty data makes an empty file.
    """
    pieces = [data[i:i + _CHUNK_BYTES] for i in range(0, len(data), _CHUNK_BYTES)] or [b""]
    publish = (
        f'{{ [ ! -f {target} ] || chmod "$(stat -c %a {target})" {staging}; }} && '
        f"mv -fT {staging} {target} && "
    )
    commands = []
    for index, piece in enumerate(pieces):
        redirect = ">" if index == 0 else ">>"
        prefix = f'mkdir -p "$(dirname {target})" && ' if index == 0 else ""
        then = publish if index == len(pieces) - 1 else ""
        encoded = base64.b64encode(piece).decode()
        commands.append(
            f"{prefix}base64 -d {redirect} {staging} << 'KISS_B64_EOF' && {then}echo {_WRITE_OK}\n"
            f"{encoded}\nKISS_B64_EOF"
        )
    return commands


def _python_edit_command(file_path: str, payload: str, old_len: int, replace_all: bool) -> str:
    """The Edit replacement in Python, reading ``old``/``new`` from *payload*.

    The payload file holds ``old`` followed by ``new`` (UTF-8); the first
    *old_len* bytes are ``old``.  It is removed once the replacement ran,
    whatever its outcome, and kept when there is no Python so the Perl
    fallback can read it.

    Args:
        file_path: Absolute path to the file to modify.
        payload: Path of the payload file inside the container.
        old_len: Byte length of ``old`` within the payload.
        replace_all: If True, replace all occurrences.

    Returns:
        A bash command printing the outcome message, or :data:`_NO_PYTHON`.
    """
    ra = "True" if replace_all else "False"
    return (
        f'PYTHON=$(command -v python3 || command -v python) || '
        f'{{ echo "{_NO_PYTHON}"; exit 1; }}; '
        f'"$PYTHON" -c "\n'
        f"import sys\n"
        f"blob = open(sys.argv[2], 'rb').read()\n"
        f"old = blob[:{old_len}].decode()\n"
        f"new = blob[{old_len}:].decode()\n"
        f"path = sys.argv[1]\n"
        f"try:\n"
        f"    content = open(path, encoding='utf-8', newline='').read()\n"
        f"except FileNotFoundError:\n"
        f"    print(f'Error: File not found: {{path}}'); sys.exit(1)\n"
        f"count = content.count(old)\n"
        f"if count == 0 and '\\r\\n' in content and '\\r\\n' not in old:\n"
        f"    old = old.replace('\\n', '\\r\\n')\n"
        f"    new = new.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')\n"
        f"    count = content.count(old)\n"
        f"if count == 0:\n"
        f"    print('Error: String not found in file'); sys.exit(1)\n"
        f"ra = {ra}\n"
        f"if not ra and count > 1:\n"
        f"    print(f'Error: String appears {{count}} times (not unique). "
        f"Use replace_all=True to replace all occurrences.'); sys.exit(1)\n"
        f"new_content = content.replace(old, new) if ra else content.replace(old, new, 1)\n"
        f"open(path, 'w', encoding='utf-8', newline='').write(new_content)\n"
        f"replaced = count if ra else 1\n"
        f"print(f'Successfully replaced {{replaced}} occurrence(s) in {{path}}')\n"
        f'" {shlex.quote(file_path)} {payload}; KISS_STATUS=$?; rm -f {payload}; exit $KISS_STATUS'
    )


def _perl_edit_command(file_path: str, payload: str, old_len: int, replace_all: bool) -> str:
    """The Edit replacement as a Perl one-liner, for containers without Python.

    Reads the same *payload* file as :func:`_python_edit_command`, as
    raw bytes, and removes it afterwards.

    Args:
        file_path: Absolute path to the file to modify.
        payload: Path of the payload file inside the container.
        old_len: Byte length of ``old`` within the payload.
        replace_all: If True, replace all occurrences.

    Returns:
        A bash command whose output matches the Python implementation's messages.
    """
    ra = "1" if replace_all else "0"
    return (
        f"perl -e '\n"
        f"my ($path, $payload) = @ARGV;\n"
        f"open(my $pf, \"<:raw\", $payload) or die \"cannot read $payload\";\n"
        f"local $/; my $blob = <$pf>; close $pf;\n"
        f"my $old = substr($blob, 0, {old_len}); my $new = substr($blob, {old_len});\n"
        f"open(my $fh, \"<:raw\", $path)\n"
        f"  or do {{ print \"Error: File not found: $path\\n\"; exit 1 }};\n"
        f"my $content = <$fh>; close $fh;\n"
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
        f"' {shlex.quote(file_path)} {payload}; KISS_STATUS=$?; rm -f {payload}; exit $KISS_STATUS"
    )


class DockerTools:
    """File tools that execute inside a Docker container via bash.

    Each method generates a shell command and executes it via the provided
    bash function (typically DockerManager.Bash or SorcarAgent._docker_bash).
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
            f'if [ "$START" -gt "$TOTAL" ]; then\n'
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

    def _store(self, file_path: str, data: bytes, description: str) -> str | None:
        """Store *data* at *file_path* in one piece, whatever happens on the way.

        The pieces land in a staging file beside the target that is
        published with the last one (:func:`_store_commands`).  A piece
        that did not land leaves the target as it was and removes the
        staging file.  When the piece was killed by a Stop, or a bash
        call raised (the container or the task is going away), the
        discard would end the same way and is skipped: the target is
        still untouched and only the uniquely named staging file stays.

        Returns:
            None once the data is published, else the output of the
            command that failed, for the caller to return as is.
        """
        staging = shlex.quote(f"{file_path}.kiss-partial-{secrets.token_hex(8)}")
        commands = _store_commands(staging, shlex.quote(file_path), data)
        for index, command in enumerate(commands):
            part = f" (part {index + 1}/{len(commands)})" if len(commands) > 1 else ""
            result = self.bash(command, f"{description}{part}")
            if _WRITE_OK in result:
                continue
            if STOPPED_MESSAGE not in result:
                self.bash(f"rm -f {staging}", f"{description} (discard)")
            return result
        return None

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
        failure = self._store(file_path, content.encode(), f"Write {file_path}")
        if failure is not None:
            return failure
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str = "",
        new_string: str = "",
        replace_all: bool = False,
        edits: str = "",
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Path to the file to modify; a relative path resolves
                against the container's working directory (the task's work
                dir in a container kiss started from an image).
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.
            edits: Several replacements in one call, as a JSON list of
                ``{"old_string": ..., "new_string": ..., "replace_all": bool}``
                objects applied in order; the first that fails stops the
                batch and the result names the edits already applied.
                ``old_string`` / ``new_string`` are then ignored.
        """
        if edits:
            pairs = _edit_pairs(edits)
            if isinstance(pairs, str):
                return pairs
            results: list[str] = []
            for index, (old_text, new_text, every) in enumerate(pairs, 1):
                result = self.Edit(file_path, old_text, new_text, every)
                if result.startswith("Error"):
                    applied = f"edits 1-{index - 1} applied; " if index > 1 else ""
                    return f"Error: edit {index}: {applied}{result.removeprefix('Error: ')}"
                results.append(f"edit {index}: {result}")
            return "\n".join(results)
        if old_string == "":
            return (
                "Error: old_string must not be empty. "
                "Use the Write tool to create or overwrite a file."
            )
        if old_string == new_string:
            return "Error: new_string must be different from old_string"
        # The strings travel in a payload file of their own, so their
        # size is not bounded by the shell command's argv limit.
        old = old_string.encode()
        payload = f"/tmp/.kiss-edit-{secrets.token_hex(8)}"
        failure = self._store(payload, old + new_string.encode(), f"Edit {file_path}")
        if failure is not None:
            return failure
        result = self.bash(
            _python_edit_command(file_path, payload, len(old), replace_all),
            f"Edit {file_path}",
        )
        if _NO_PYTHON not in result:
            return result
        # Images without Python (Rust, R, C toolchains) still have
        # perl-base, so the same replacement is done in Perl.
        return self.bash(
            _perl_edit_command(file_path, payload, len(old), replace_all),
            f"Edit {file_path}",
        )
