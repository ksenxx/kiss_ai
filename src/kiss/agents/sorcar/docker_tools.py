# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File tools (Read, Write, Edit) that execute inside a Docker container via bash."""

import base64
import mimetypes
import secrets
import shlex
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, PendingAppend
from kiss.agents.sorcar.useful_tools import _edit_pairs
from kiss.core.models.model import READ_TOOL_BINARY_MIME_TYPES, encode_binary_attachment

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


def _pieces(data: bytes) -> list[bytes]:
    """Split *data* into the base64 pieces one shell command each carries.

    Each piece holds at most :data:`_CHUNK_BYTES` bytes, so its command
    fits a single argv string whatever the payload size; empty data is
    one empty piece.
    """
    return [data[i:i + _CHUNK_BYTES] for i in range(0, len(data), _CHUNK_BYTES)] or [b""]


def _store_commands(staging: str, target: str, data: bytes) -> list[str]:
    """Return the shell commands that together store *data* at *target*.

    Each command carries one base64 piece (:func:`_pieces`).  The pieces
    accumulate in *staging*, a file beside the target that nobody else
    names, and the last command publishes it with one ``mv`` (keeping
    the mode of the file it replaces), so a reader of the target sees
    the previous file or the whole new one and a second writer of the
    same target cannot splice its bytes into this one.  Each command
    echoes :data:`_WRITE_OK` once its piece landed and nothing
    (non-zero status) when it did not.

    Args:
        staging: The staging path, already shell-quoted.
        target: The target path, already shell-quoted.
        data: The bytes to store; empty data makes an empty file.
    """
    pieces = _pieces(data)
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


def _append_commands(target: str, data: bytes) -> list[str]:
    """Return the shell commands that together append *data* to *target*.

    Each command appends one base64 piece (:func:`_pieces`) through the
    shell's ``>>``: the target is opened for appending, so what another
    writer appends before the first piece or between two stays in front
    of them, as after the ``cat >> target`` a redirected copy stands
    for (:meth:`DockerTools.Write`).  Each command echoes
    :data:`_WRITE_OK` once its piece landed.

    Args:
        target: The target path, already shell-quoted.
        data: The bytes to append; empty data creates a missing target.
    """
    return [
        f"base64 -d >> {target} << 'KISS_B64_EOF' && echo {_WRITE_OK}\n"
        f"{base64.b64encode(piece).decode()}\nKISS_B64_EOF"
        for piece in _pieces(data)
    ]


def _python_edit_command(file_path: str, payload: str, old_len: int, replace_all: bool) -> str:
    """The Edit replacement in Python, reading ``old``/``new`` from *payload*.

    The payload file holds ``old`` followed by ``new`` (UTF-8); the first
    *old_len* bytes are ``old``.  The caller (:meth:`DockerTools.Edit`)
    removes it once the replacement ran, whatever its outcome.

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
        f'" {shlex.quote(file_path)} {payload}'
    )


def _perl_edit_command(file_path: str, payload: str, old_len: int, replace_all: bool) -> str:
    """The Edit replacement as a Perl one-liner, for containers without Python.

    Reads the same *payload* file as :func:`_python_edit_command`, as
    raw bytes.

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
        f"' {shlex.quote(file_path)} {payload}"
    )


class DockerTools:
    """File tools that execute inside a Docker container via bash.

    Each method generates a shell command and executes it via the provided
    bash function (typically DockerManager.Bash or SorcarAgent._docker_bash).
    """

    def __init__(
        self,
        bash_fn: Callable[[str, str], str],
        read_bytes: Callable[[str], bytes] | None = None,
    ) -> None:
        """Initialize with a bash execution function.

        Args:
            bash_fn: Callable(command, description) -> output string.
                     Executes a bash command inside the Docker container.
            read_bytes: Fetches a file's bytes from the container
                (``DockerManager.read_bytes``), raising ``OSError`` when
                it cannot: what ``Read`` attaches an image or PDF with.
                Without it ``Read`` reports such a file as not readable.
        """
        self.bash = bash_fn
        self.read_bytes = read_bytes
        # The manager whose ``read_bytes`` this is (``DockerManager`` or
        # ``ExecBridge``): its ``Bash`` redirects a ``>>`` copy to a Write
        # and keeps the append as ``pending_append`` for that Write.
        self.manager: Any = getattr(read_bytes, "__self__", None)

    def _take_pending_append(self) -> PendingAppend | None:
        """Return the manager's pending append and forget it: it belongs to this call alone.

        ``Bash`` redirected a ``>>`` copy to a Write and kept the append
        (:class:`~kiss.agents.sorcar.docker_manager.PendingAppend`) for
        that Write.  Whichever call comes next takes it (as a Bash of
        the manager drops it), so a redirect the agent's hook refused
        never turns a later Write into an append.
        """
        pending: PendingAppend | None = getattr(self.manager, "pending_append", None)
        if pending is not None:
            self.manager.pending_append = None
        return pending

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
        self._take_pending_append()
        if start_line < 1:
            return (
                f"Error: start_line must be >= 1 (got {start_line}); the "
                f"parameter is 1-indexed."
            )
        if max_lines < 1:
            return f"Error: max_lines must be >= 1 (got {max_lines})."
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type in READ_TOOL_BINARY_MIME_TYPES:
            return self._read_binary(file_path, mime_type or "")
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

    def _read_binary(self, file_path: str, mime_type: str) -> str:
        """Attach the image, PDF, audio or video *file_path* as the host ``Read`` does.

        The bytes come through :attr:`read_bytes` (one exec, no text
        decoding, no output cap) and go out as the attachment encoding
        a vision model sees as the image and a text-only model gets
        described (:func:`encode_binary_attachment`).
        """
        if self.read_bytes is None:
            return (
                f"Error: Cannot read binary file {file_path} (mime={mime_type}): this "
                "environment fetches no file bytes; use Bash tools to inspect it."
            )
        try:
            data = self.read_bytes(file_path)
        except OSError as e:
            return f"Error: Cannot read binary file {file_path} (mime={mime_type}): {e}"
        return (
            f"Read binary file {file_path} as {mime_type} ({len(data)} bytes); "
            "content attached below.\n" + encode_binary_attachment(mime_type, data)
        )

    def _store(
        self, file_path: str, data: bytes, description: str, append: bool = False,
    ) -> str | None:
        """Store *data* at *file_path* in one piece, whatever happens on the way, or append it.

        The pieces land in a staging file beside the target that is
        published with the last one (:func:`_store_commands`).  A piece
        that did not land leaves the target as it was and removes the
        staging file.  When the piece was killed by a Stop, or a bash
        call raised (the container or the task is going away), the
        discard would end the same way and is skipped: the target is
        still untouched and only the uniquely named staging file stays.
        With *append* the pieces go after the target's current bytes
        through the shell's ``>>`` instead (:func:`_append_commands`),
        as the ``cat >> target`` the Write stands for would put them;
        a piece that did not land leaves the ones before it, as that
        command would.

        Returns:
            None once the data is published, else the output of the
            command that failed, for the caller to return as is.
        """
        target = shlex.quote(file_path)
        staging = shlex.quote(f"{file_path}.kiss-partial-{secrets.token_hex(8)}")
        commands = (
            _append_commands(target, data) if append else _store_commands(staging, target, data)
        )
        for index, command in enumerate(commands):
            part = f" (part {index + 1}/{len(commands)})" if len(commands) > 1 else ""
            result = self.bash(command, f"{description}{part}")
            if _WRITE_OK in result:
                continue
            if not append and STOPPED_MESSAGE not in result:
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
        pending = self._take_pending_append()
        if pending is not None and (file_path, content) == (
            pending.path, pending.snapshot.decode("utf-8") + pending.body
        ):
            # ``Bash`` redirected a ``>>`` copy to this very Write: the body
            # goes after whatever the file holds when it lands, through the
            # shell's ``>>``, as the command would have put it.  The agent's
            # hook ran in between, and what another writer appended there,
            # or appends before the store, must not be overwritten by the
            # stale snapshot.
            failure = self._store(file_path, pending.body.encode(), f"Write {file_path}", True)
        else:
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
        self._take_pending_append()
        if edits:
            pairs = _edit_pairs(edits)
            if isinstance(pairs, str):
                return pairs
            results: list[str] = []
            for index, (old_text, new_text, every) in enumerate(pairs, 1):
                result = self.Edit(file_path, old_text, new_text, every)
                if STOPPED_MESSAGE in result:
                    # A task Stop: every later edit would be killed the
                    # same way (an exec and a reaper wait each).
                    return result
                # Only the success line proves the edit landed: a traceback
                # (a non-UTF-8 or unwritable file) or a failed payload store
                # does not start with "Error".
                if not result.startswith("Successfully"):
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
        if _NO_PYTHON in result:
            # Images without Python (Rust, R, C toolchains) still have
            # perl-base, so the same replacement is done in Perl.
            result = self.bash(
                _perl_edit_command(file_path, payload, len(old), replace_all),
                f"Edit {file_path}",
            )
        # The payload goes whatever the outcome, a timed-out replacement
        # included; after a Stop the removal would be killed the same way
        # (as in _store), so only the uniquely named payload file stays.
        if STOPPED_MESSAGE not in result:
            self.bash(f"rm -f {payload}", f"Edit {file_path} (discard)")
        return result
