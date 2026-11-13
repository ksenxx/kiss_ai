# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File tools (Read, Write, Edit) that execute inside a Docker container via bash."""

import base64
import hashlib
import logging
import mimetypes
import re
import secrets
import shlex
import threading
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.docker_manager import (
    STOPPED_MESSAGE,
    PendingAppend,
    PendingEdit,
    PendingRewrite,
)
from kiss.agents.sorcar.edit_matching import nearest_region, reindent, tolerant_matches
from kiss.agents.sorcar.useful_tools import _edit_pairs
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.locked import locked_call
from kiss.core.models.model import READ_TOOL_BINARY_MIME_TYPES, encode_binary_attachment

logger = logging.getLogger(__name__)

# Held by ``Edit`` for the script that reads, modifies and publishes a
# file and by ``Write`` for each command that publishes or appends to
# one, as the host tools hold theirs
# (:data:`~kiss.agents.sorcar.useful_tools._FILE_WRITE_LOCK`): the tool
# instances of a task and its sub-agents (threads of one process attached
# to one container) would otherwise undo each other's changes, both
# calls reporting success, when one's write landed between another's
# read and write of the same file.  Taken through :func:`locked_call`.
_FILE_WRITE_LOCK = threading.RLock()

#: Marker the Write script echoes only after the bytes really landed.
#: Success must be proven positively: the bash function's *other* failure
#: return — ``"Error: command timed out after Ns"`` — carries no
#: ``[exit code:`` marker, so sniffing for that marker reported a write
#: that never happened as a success.
_WRITE_OK = "__KISS_WRITE_OK__"
#: Marker the Edit script prints when the container has no Python interpreter.
_NO_PYTHON = "Error: Python required for Edit"
#: What the Edit scripts print when ``old`` is not in the file verbatim.
_NOT_FOUND = "Error: String not found in file"
#: The note the Read command prints after a window that stops short of EOF.
_TRUNCATION_NOTE = re.compile(r"\[truncated: \d+ more lines\]")
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


def _operand(file_path: str) -> str:
    """Return *file_path* shell-quoted as a file operand: a leading dash gets a ``./``.

    A relative path such as ``-sample.txt`` would otherwise be read as
    options by the utilities the scripts run (``dirname``, ``tail``,
    ``mv``, ``perl``).
    """
    return shlex.quote(f"./{file_path}" if file_path.startswith("-") else file_path)


def _edited(result: str) -> bool:
    """Whether an editor script's *result* holds its success line.

    A line, not the start of the result: the shell of a container whose
    locale is not installed warns before every command's own output.
    """
    return any(line.startswith("Successfully replaced ") for line in result.splitlines())


def _failure(output: str) -> str:
    """Return the *output* of a command that failed as the result of a failed call.

    A result opening with ``Error`` is the tools' word for a call that
    did not go through: the agent then leaves the rest of a compound
    command unmade and a digest does not replay the write.  A Stop's
    message is reported as it is.
    """
    if output.startswith("Error") or STOPPED_MESSAGE in output:
        return output
    return f"Error: {output}"


# Shell refusing a resolved target that exists and is not a regular file, in the host
# Write's words: the publishing ``mv`` would turn a device or a FIFO into a regular file.
_REGULAR_OR_NEW = (
    '{ [ ! -e "$KISS_TARGET" ] || [ -f "$KISS_TARGET" ] || { echo "Error: $KISS_TARGET exists'
    ' and is not a regular file (directory/FIFO/device/socket); refusing to write to it.";'
    " false; }; } && "
)


def _resolved(path_command: str, staging: str) -> str:
    """Shell setting ``KISS_TARGET`` to the path *path_command* prints, ``KISS_STAGING`` beside it.

    A command substitution drops the trailing newlines of what it
    captures, and a path may end in one (``name\\n`` is another file
    than ``name``): an ``x`` is printed after the path, inside the
    substitution, and taken off again.  The staging file is *staging*
    in the target's directory; its name does not carry the target's,
    as a name near the file system's limit leaves no room for a longer
    sibling.
    """
    return (
        f'KISS_TARGET="$({path_command} && printf x)" && KISS_TARGET="${{KISS_TARGET%x}}" && '
        f'KISS_STAGING="${{KISS_TARGET%/*}}/{staging}" && '
    )


def _note(staging: str) -> str:
    """The file noting the resolved target of the store staged as *staging*.

    Not ``/tmp/<staging>``: that is the staging file itself when the
    target lies in ``/tmp``.
    """
    return f"/tmp/{staging}.note"


def _store_commands(staging: str, target: str, data: bytes) -> list[str]:
    """Return the shell commands that together store *data* at *target*.

    Each command carries one base64 piece (:func:`_pieces`).  The pieces
    accumulate in a staging file that nobody else names, and the last
    command publishes it with one ``mv`` (keeping the mode of the file
    it replaces, its owner and its group), so a reader of the target
    sees the previous file or the whole new one and a second writer of
    the same target cannot splice its bytes into this one; a file with
    other hard links gets the staged bytes copied into it instead, as
    the host Write and the Edit scripts write one, since the rename
    would cut it from its other names, which would keep the old bytes.
    So does a file this user may not write (the rename takes the
    directory's permission only: a read-only file is refused, as the
    ``cat > target`` refuses it), one whose owner the staged file
    cannot be given and one the ``mv`` is refused over (a file mounted
    into the container, as Docker mounts ``/etc/hosts``: a mount point
    is not renamed over).  A target that is a symlink is published to
    the file it points to (``readlink -f``, made when the link dangles), as
    the ``cat > target`` a redirected copy stands for writes through
    it, so the link stays; the staging file is made beside that file
    (*staging* is its name there), on its file system, so the ``mv`` is
    a rename and never a copy.  A directory its user may not write
    takes no sibling: the pieces are staged as ``/tmp/<staging>``
    instead and copied into the file, as the shell's ``>`` writes a
    writable file there; the later commands find that staging file by
    its presence.
    The first command resolves the target, refuses one that exists and
    is not a regular file (:data:`_REGULAR_OR_NEW`) and, when more
    follow, notes the resolved path under ``/tmp`` (:func:`_note`)
    for them (each command is an exec of its own and shares no shell
    variable): a link repointed between two commands would otherwise
    send the later pieces, the publication and the discard to another
    file, publishing a suffix of the data as the whole and leaving the
    first pieces behind, while the ``cat > target`` keeps the file it
    opened.  The paths are captured whole, a trailing newline included
    (:func:`_resolved`).  Each command echoes :data:`_WRITE_OK` once its
    piece landed and nothing (non-zero status) when it did not.

    Args:
        staging: The staging file's name in the resolved target's
            directory (no shell-special character).
        target: The target path, already shell-quoted.
        data: The bytes to store; empty data makes an empty file.
    """
    pieces = _pieces(data)
    note = _note(staging)
    noted = len(pieces) > 1
    away = f"/tmp/{staging}"
    publish = (
        f'if [ "$KISS_STAGING" != "${{KISS_TARGET%/*}}/{staging}" ]'
        ' || { [ -f "$KISS_TARGET" ] && { [ "$(stat -c %h "$KISS_TARGET")" -gt 1 ]'
        ' || [ ! -w "$KISS_TARGET" ]'
        ' || ! chown "$(stat -c %u:%g "$KISS_TARGET")" "$KISS_STAGING" 2>/dev/null'
        ' || ! chmod "$(stat -c %a "$KISS_TARGET")" "$KISS_STAGING"; }; }'
        ' || ! mv -fT "$KISS_STAGING" "$KISS_TARGET" 2>/dev/null; then'
        ' cat "$KISS_STAGING" > "$KISS_TARGET" && rm -f "$KISS_STAGING"; fi && '
    ) + (f"rm -f {note} && " if noted else "")
    commands = []
    for index, piece in enumerate(pieces):
        if index == 0:
            # dirname ends its answer with a newline of its own: it goes with the x.
            resolve = (
                f'KISS_DIR="$(dirname {target} && echo x)" && mkdir -p "${{KISS_DIR%??}}" && '
                + _resolved(f"readlink -fn {target}", staging)
                + _REGULAR_OR_NEW
                + (f'printf %s "$KISS_TARGET" > {note} && ' if noted else "")
                # Only a directory that refuses its user: one that takes no
                # new file for another reason (no inode left) fails the
                # store, which a copy into the file could leave cut.
                + f'{{ [ -w "${{KISS_STAGING%/*}}/" ] || KISS_STAGING={away}; }} && '
            )
        else:
            resolve = (
                _resolved(f"cat {note}", staging) + '[ -n "$KISS_TARGET" ] && '
                + f"{{ [ ! -e {away} ] || KISS_STAGING={away}; }} && "
            )
        redirect = ">" if index == 0 else ">>"
        then = publish if index == len(pieces) - 1 else ""
        encoded = base64.b64encode(piece).decode()
        commands.append(
            f"{resolve}base64 -d {redirect} \"$KISS_STAGING\" << 'KISS_B64_EOF' && "
            f"{then}echo {_WRITE_OK}\n{encoded}\nKISS_B64_EOF"
        )
    return commands


def _discard_staging_command(staging: str, target: str) -> str:
    """The shell command removing the staging file of *target* and its note.

    The staging file (:func:`_store_commands`) is beside the file the
    note names when the store wrote one (a later piece or the
    publication failed), else beside the file the target resolves to
    now (a one-piece store, or a first command that failed before
    noting it); the ``x`` keeps the newline that file's name may end
    in, as in :func:`_resolved`, and goes with the name.  The one of a
    directory that takes no sibling is under ``/tmp``.
    """
    note = _note(staging)
    path = f"{{ cat {note} 2>/dev/null || readlink -fn {target}; }}"
    beside = f'"$(p="$({path}; printf x)"; printf %s "${{p%/*}}/{staging}")"'
    return f"rm -f {beside} {note} /tmp/{staging}"


def _append_commands(staging: str, target: str, data: bytes) -> list[str]:
    """Return the shell commands that together append *data* to *target*.

    One base64 piece (:func:`_pieces`) goes through the shell's ``>>``
    as it is.  Several accumulate in ``/tmp/<staging>``, one command
    each, and a last command appends that file through one ``>>``.
    Either way the target is opened once, for appending, so the data
    lands whole in one file (a link repointed or a file replaced between
    two commands cannot get a part of it), after what another writer
    appended before, as after the ``cat >> target`` a redirected copy
    stands for (:meth:`DockerTools.Write`).  Each command echoes
    :data:`_WRITE_OK` once its part landed.

    Args:
        staging: The staging file's name under ``/tmp`` (no shell-special character).
        target: The target path, already shell-quoted.
        data: The bytes to append; empty data creates a missing target.
    """
    pieces = _pieces(data)
    stage = f"/tmp/{staging}"
    into = target if len(pieces) == 1 else stage
    commands = [
        f"base64 -d >> {into} << 'KISS_B64_EOF' && echo {_WRITE_OK}\n"
        f"{base64.b64encode(piece).decode()}\nKISS_B64_EOF"
        for piece in pieces
    ]
    if len(pieces) > 1:
        commands.append(f"cat {stage} >> {target} && rm -f {stage} && echo {_WRITE_OK}")
    return commands


def _python_edit_command(
    file_path: str, payload: str, old_len: int, replace_all: bool, exact: bool = False,
    at: int | None = None,
) -> str:
    """The Edit replacement in Python, reading ``old``/``new`` from *payload*.

    The payload file holds ``old`` followed by ``new`` (UTF-8); the first
    *old_len* bytes are ``old``.  The caller (:meth:`DockerTools.Edit`)
    removes it once the replacement ran, whatever its outcome.  A path
    that is not a regular file is answered as a missing one, as the
    host ``Edit`` answers it (reading a FIFO would block).  An LF
    ``old`` missing from a CRLF file is retried as CRLF (``new`` made
    CRLF with it) and the ``new`` of a one-line ``old`` found as it is
    takes the file's CRLF too, unless *exact*.  With *at*, the ``old``
    at that byte of the file is the one replaced, wherever else its
    text occurs; a file that no longer holds it there (changed since
    the caller read it) is searched as without *at*.  The new text is
    published as the host ``Edit`` publishes it
    (:func:`~kiss.agents.sorcar.useful_tools._write_file`): a sibling
    file renamed over the file (through a symlink, keeping the mode,
    the owner and the group), so a reader sees the old text or the new
    and a write that fails leaves the file as it was; a file with other
    hard links is written in place, as is one whose directory refuses
    the sibling, one whose owner the sibling cannot be given, one the
    rename is refused over (a file mounted into the container) and one
    this user may not write (the rename takes the directory's
    permission only: a read-only file is refused, to an *exact* edit
    too; the ``sed -i`` that replaces one is not made as an Edit,
    :func:`~kiss.agents.sorcar.docker_manager.probe_command`).  The
    file written in place is closed before the outcome is printed: a
    write the disk refuses then (no space left) is an error, not a
    success over a cut file.

    Args:
        file_path: Absolute path to the file to modify.
        payload: Path of the payload file inside the container.
        old_len: Byte length of ``old`` within the payload.
        replace_all: If True, replace all occurrences.
        exact: If True, match ``old`` verbatim only (no CRLF retry).
        at: Byte offset in the file of the occurrence to replace.

    Returns:
        A bash command printing the outcome message, or :data:`_NO_PYTHON`.
    """
    ra = "True" if replace_all else "False"
    return (
        f'PYTHON=$(command -v python3 || command -v python) || '
        f'{{ echo "{_NO_PYTHON}"; exit 1; }}; '
        f'"$PYTHON" -c "\n'
        f"import os, stat, sys, uuid\n"
        f"blob = open(sys.argv[2], 'rb').read()\n"
        f"old = blob[:{old_len}].decode()\n"
        f"new = blob[{old_len}:].decode()\n"
        f"path = sys.argv[1]\n"
        f"try:\n"
        f"    if not os.path.isfile(path):\n"
        f"        raise FileNotFoundError\n"
        f"    content = open(path, encoding='utf-8', newline='').read()\n"
        f"except FileNotFoundError:\n"
        f"    print(f'Error: File not found: {{path}}'); sys.exit(1)\n"
        f"ra = {ra}\n"
        f"at = {at}\n"
        f"raw = content.encode() if at is not None else b''\n"
        f"if at is not None and old and raw[at:at + {old_len}] == blob[:{old_len}]:\n"
        f"    count = 1\n"
        f"    new_content = (raw[:at] + blob[{old_len}:] + raw[at + {old_len}:]).decode()\n"
        f"else:\n"
        f"    count = content.count(old)\n"
        f"    if {not exact} and (count == 0 or '\\n' not in old) and '\\r\\n' in content "
        f"and '\\r\\n' not in old:\n"
        f"        old = old.replace('\\n', '\\r\\n')\n"
        f"        new = new.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')\n"
        f"        count = content.count(old)\n"
        f"    if count == 0:\n"
        f"        print('Error: String not found in file'); sys.exit(1)\n"
        f"    if not ra and count > 1:\n"
        f"        print(f'Error: String appears {{count}} times (not unique). "
        f"Use replace_all=True to replace all occurrences.'); sys.exit(1)\n"
        f"    new_content = content.replace(old, new) if ra else content.replace(old, new, 1)\n"
        f"real = os.path.realpath(path)\n"
        f"st = os.stat(real)\n"
        f"tmp = os.path.join(os.path.dirname(real), '.' + uuid.uuid4().hex + '.tmp')\n"
        f"try:\n"
        f"    if st.st_nlink > 1 or not os.access(real, os.W_OK):\n"
        f"        raise PermissionError\n"
        f"    with open(tmp, 'x', encoding='utf-8', newline='') as f:\n"
        f"        f.write(new_content)\n"
        f"    made = os.stat(tmp)\n"
        f"    if (made.st_uid, made.st_gid) != (st.st_uid, st.st_gid):\n"
        f"        os.chown(tmp, st.st_uid, st.st_gid)\n"
        f"    os.chmod(tmp, stat.S_IMODE(st.st_mode))\n"
        f"    try:\n"
        f"        os.replace(tmp, real)\n"
        f"    except OSError:\n"
        f"        raise PermissionError\n"
        f"except PermissionError:\n"
        f"    try:\n"
        f"        with open(real, 'w', encoding='utf-8', newline='') as f:\n"
        f"            f.write(new_content)\n"
        f"    except OSError as e:\n"
        f"        print(f'Error: {{e}}'); sys.exit(1)\n"
        f"finally:\n"
        f"    if os.path.lexists(tmp):\n"
        f"        os.unlink(tmp)\n"
        f"replaced = count if ra else 1\n"
        f"print(f'Successfully replaced {{replaced}} occurrence(s) in {{path}}')\n"
        f'" {_operand(file_path)} {payload}'
    )


def _perl_edit_command(
    file_path: str, payload: str, old_len: int, replace_all: bool, exact: bool = False,
    at: int | None = None,
) -> str:
    """The Edit replacement as a Perl one-liner, for containers without Python.

    Reads the same *payload* file as :func:`_python_edit_command`, as
    raw bytes, opens a regular file only, matches as it does (an LF
    ``old`` is retried as CRLF against a CRLF file and the ``new`` of
    a one-line ``old`` made CRLF, unless *exact*; the ``old`` at byte
    *at* first) and publishes the same way (a sibling file renamed
    over the file, with its mode, owner and group; in place for a file
    with other hard links, a directory refusing the sibling, an owner
    the sibling cannot be given, a refused rename or a file this user
    may not write), with a write or close that failed reported as an
    error, not as a success.
    The sibling is created under a random name that must not exist yet,
    so nothing waiting at a guessable name (a symlink to another file)
    is written through.

    Args:
        file_path: Absolute path to the file to modify.
        payload: Path of the payload file inside the container.
        old_len: Byte length of ``old`` within the payload.
        replace_all: If True, replace all occurrences.
        exact: If True, match ``old`` verbatim only (no CRLF retry).
        at: Byte offset in the file of the occurrence to replace.

    Returns:
        A bash command whose output matches the Python implementation's messages.
    """
    ra = "1" if replace_all else "0"
    spot = -1 if at is None else at
    return (
        f"perl -MCwd -MFcntl -e '\n"
        f"my ($path, $payload) = @ARGV;\n"
        f"open(my $pf, \"<:raw\", $payload) or die \"cannot read $payload\";\n"
        f"local $/; my $blob = <$pf>; close $pf;\n"
        f"my $old = substr($blob, 0, {old_len}); my $new = substr($blob, {old_len});\n"
        f"my $fh;\n"
        f"(-f $path and open($fh, \"<:raw\", $path))\n"
        f"  or do {{ print \"Error: File not found: $path\\n\"; exit 1 }};\n"
        f"my $content = <$fh>; close $fh;\n"
        f"my $count = 1;\n"
        f"if ({spot} >= 0 && length($old) && substr($content, {spot}, length($old)) eq $old) {{\n"
        f"  substr($content, {spot}, length($old), $new);\n"
        f"}} else {{\n"
        f"  $count = () = $content =~ /\\Q$old\\E/g;\n"
        f"  if (!{int(exact)} && ($count == 0 || index($old, \"\\n\") < 0) "
        f"&& index($content, \"\\r\\n\") >= 0 && index($old, \"\\r\\n\") < 0) {{\n"
        f"    $old =~ s/\\n/\\r\\n/g; $new =~ s/\\r\\n/\\n/g; $new =~ s/\\n/\\r\\n/g;\n"
        f"    $count = () = $content =~ /\\Q$old\\E/g;\n"
        f"  }}\n"
        f"  if ($count == 0) {{ print \"Error: String not found in file\\n\"; exit 1 }}\n"
        f"  if (!{ra} && $count > 1) {{ print \"Error: String appears $count times (not unique). "
        f"Use replace_all=True to replace all occurrences.\\n\"; exit 1 }}\n"
        f"  if ({ra}) {{ $content =~ s/\\Q$old\\E/$new/g }}\n"
        f"  else {{ $content =~ s/\\Q$old\\E/$new/ }}\n"
        f"}}\n"
        f"my $real = Cwd::realpath($path); my @st = stat($real);\n"
        f"my $tmp = $real;\n"
        f"$tmp =~ s{{[^/]*$}}{{sprintf(\".%08x%08x.tmp\", rand(2**32), rand(2**32))}}e;\n"
        f"my $staged = $st[3] <= 1 && -w $real "
        f"&& sysopen($fh, $tmp, O_WRONLY|O_CREAT|O_EXCL, 0600);\n"
        f"if ($staged) {{ my @made = stat($fh);\n"
        f"  unless (($made[4] == $st[4] && $made[5] == $st[5]) || chown($st[4], $st[5], $fh)) {{\n"
        f"    close($fh); unlink($tmp); $staged = 0 }} }}\n"
        f"if (!$staged) {{ open($fh, \">:raw\", $real) "
        f"or do {{ print \"Error: cannot write $path: $!\\n\"; exit 1 }} }}\n"
        f"binmode($fh);\n"
        f"unless (print $fh $content and (!$staged or chmod($st[2] & 07777, $fh)) "
        f"and close($fh) and (!$staged or rename($tmp, $real) "
        f"or (open($fh, \">:raw\", $real) and print $fh $content and close($fh) "
        f"and unlink($tmp)))) {{\n"
        f"  my $why = $!; unlink($tmp) if $staged;\n"
        f"  print \"Error: cannot write $path: $why\\n\"; exit 1 }}\n"
        f"my $replaced = {ra} ? $count : 1;\n"
        f"print \"Successfully replaced $replaced occurrence(s) in $path\\n\";\n"
        f"' {_operand(file_path)} {payload}"
    )


def _is_window(result: str) -> bool:
    """Whether a ``Read`` result is file content: not an error, a sentinel or a killed command.

    The errors and the sentinels of the Read command start with
    ``Error:`` or ``(file is empty)``; a command that failed, timed out
    or was stopped ends with the ``[exit code: N]`` line (or starts
    with ``Error:``).
    """
    if result.startswith(("Error:", "(file is empty)")):
        return False
    return not result.rstrip("\n").rsplit("\n", 1)[-1].startswith("[exit code:")


class DockerTools:
    """File tools that execute inside a Docker container via bash.

    Each method generates a shell command and executes it via the provided
    bash function (typically DockerManager.Bash or SorcarAgent._docker_bash).
    """

    def __init__(
        self,
        bash_fn: Callable[[str, str], str],
        read_bytes: Callable[[str], bytes] | None = None,
        step_count: Callable[[], int] | None = None,
    ) -> None:
        """Initialize with a bash execution function.

        Args:
            bash_fn: Callable(command, description) -> output string.
                     Executes a bash command inside the Docker container.
            read_bytes: Fetches a file's bytes from the container
                (``DockerManager.read_bytes``), raising ``OSError`` when
                it cannot: what ``Read`` attaches an image or PDF with and
                what ``Edit`` fetches the file's text with for its
                whitespace-tolerant retry.  Without it ``Read`` reports
                such a file as not readable and ``Edit`` stops at the
                exact match.
            step_count: Returns the agent's current step, the number the
                model sees in its result footers: what the "Unchanged
                since your earlier Read at step N" note of ``Read`` cites.
                Without it the note cites the Read's ordinal in this
                session.
        """
        self.bash = bash_fn
        self.read_bytes = read_bytes
        self.step_count = step_count
        # The manager whose ``read_bytes`` this is (``DockerManager`` or
        # ``ExecBridge``): its ``Bash`` redirects a ``>>`` copy to a Write
        # and a mechanical rewrite to an Edit, and keeps the append or the
        # edit as ``pending_append`` for that call.
        self.manager: Any = getattr(read_bytes, "__self__", None)
        self._reads = 0
        # sha256 of each window text ``Read`` returned -> the step it was
        # returned at, its first and last line, its path (the latest
        # showing): the content-keyed dedupe of :meth:`Read`.
        self._reads_shown: dict[str, tuple[int, int, int, str]] = {}

    def reset_reads(self) -> None:
        """Forget which file windows the model has already been shown.

        For the harness's context-reset path (``context_reset_hook``: a
        hand-off to a new session, a compaction): the earlier outputs
        left the model's context, so the next ``Read`` of an unchanged
        window returns its content instead of the "Unchanged since" note.
        """
        self._reads_shown.clear()

    def _take_pending(self) -> PendingRewrite | None:
        """Return the manager's pending append or edit and forget it: it belongs to this call alone.

        ``Bash`` redirected a ``>>`` copy to a Write, or a mechanical
        rewrite to an Edit, and kept what the command stands for
        (:class:`~kiss.agents.sorcar.docker_manager.PendingAppend`,
        :class:`~kiss.agents.sorcar.docker_manager.PendingEdit`) for
        that call.  Whichever call comes next takes it (as a Bash of
        the manager drops it), so a redirect the agent's hook refused
        never turns a later Write into an append.
        """
        pending: PendingRewrite | None = getattr(self.manager, "pending_append", None)
        if pending is not None:
            self.manager.pending_append = None
        return pending

    def Read(  # noqa: N802
        self,
        file_path: str,
        max_lines: int = 2000,
        start_line: int = 1,
        force: bool = False,
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
            force: Re-send a window whose text an earlier Read of this
                session already returned (any range of any path); by
                default such a Read returns the one-line "Unchanged
                since your earlier Read at step N (lines a-b of path)"
                note instead of the content, which is still in your
                context above.
        """
        self._take_pending()
        self._reads += 1
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
        path = _operand(file_path)
        # awk takes a relative operand shaped ``name=value`` (date=2024-01-01/part.csv)
        # for a variable assignment and reads its empty standard input: a ./ makes it a path.
        counted = shlex.quote(file_path if file_path.startswith("/") else f"./{file_path}")
        cmd = (
            f'FILE={path}\n'
            f'if [ ! -f "$FILE" ]; then echo "Error: File not found: $FILE"; exit 1; fi\n'
            f'TOTAL=$(awk \'END{{print NR}}\' {counted}) || '
            f'{{ echo "Error: cannot read $FILE"; exit 1; }}\n'
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
        result = self.bash(cmd, f"Read {file_path}")
        if not _is_window(result):
            return result
        return self._dedupe(result, file_path, start_line, max_lines, force)

    def _dedupe(
        self, window: str, file_path: str, start_line: int, max_lines: int, force: bool,
    ) -> str:
        """Return *window* or, when its lines were returned earlier in this session, the note.

        The key is the sha256 of the window's lines (the truncation
        note aside), so a window of another range or another path with
        the same lines counts.  A returned window is recorded (the
        latest showing wins, a forced one included); the note cites the
        recorded step, lines and path.
        """
        lines = window.removesuffix("\n").split("\n")
        if len(lines) == max_lines + 1 and _TRUNCATION_NOTE.fullmatch(lines[-1]):
            lines.pop()
        digest = hashlib.sha256("\n".join(lines).encode("utf-8", "surrogatepass")).hexdigest()
        earlier = self._reads_shown.get(digest)
        if earlier is not None and not force and DEFAULT_CONFIG.read_dedupe:
            step, first, last, path = earlier
            return (
                f"Unchanged since your earlier Read at step {step} (lines {first}-{last} of {path})"
            )
        step = self.step_count() if self.step_count is not None else self._reads
        self._reads_shown[digest] = (step, start_line, start_line + len(lines) - 1, file_path)
        return window

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

        The pieces land in a staging file beside the target (under
        ``/tmp`` when its directory takes none) that is published with
        the last one (:func:`_store_commands`).  A piece
        that did not land leaves the target as it was and removes the
        staging file (:func:`_discard_staging_command`); a bash call
        that raised instead (a transient executor failure, a container
        going away) gets the same discard, best effort, before the
        error goes on (:meth:`_discard`).  When the piece was killed by
        a Stop the discard would be killed the same way and is skipped:
        the target is still untouched and only the uniquely named
        staging file (and the note of its place) stays.
        With *append* the data goes after the target's current bytes
        through one ``>>`` of the last command instead
        (:func:`_append_commands`), as the ``cat >> target`` the Write
        stands for would put it; several pieces are staged under
        ``/tmp`` first, and that file is discarded the same way.

        Returns:
            None once the data is published, else the output of the
            command that failed, for the caller to report
            (:func:`_failure`).
        """
        target = _operand(file_path)
        staging = f".kiss-partial-{secrets.token_hex(8)}"
        if append:
            commands = _append_commands(staging, target, data)
            # a one-piece append stages nothing
            discard = f"rm -f /tmp/{staging}" if len(commands) > 1 else None
        else:
            commands = _store_commands(staging, target, data)
            discard = _discard_staging_command(staging, target)
        for index, command in enumerate(commands):
            part = f" (part {index + 1}/{len(commands)})" if len(commands) > 1 else ""
            try:
                # Only the last command (the publication, or the append)
                # touches the target.
                if index == len(commands) - 1:
                    result = locked_call(
                        _FILE_WRITE_LOCK, self.bash, command, f"{description}{part}",
                    )
                else:
                    result = self.bash(command, f"{description}{part}")
            except BaseException:
                self._discard(discard, f"{description} (discard)")
                raise
            # The marker counts as a whole line: a diagnostic naming the
            # target (``mv: cannot overwrite directory '.../__KISS_WRITE_OK__'``)
            # is no proof the bytes landed.
            if _WRITE_OK in result.splitlines():
                continue
            if discard is not None and STOPPED_MESSAGE not in result:
                self.bash(discard, f"{description} (discard)")
            return result
        return None

    def _discard(self, command: str | None, description: str) -> None:
        """Run the discard *command* after a bash call raised; its own failure is only logged.

        The executor that raised may refuse this call too (the container
        or the task is going away); the staging or payload file it would
        remove is uniquely named and harms nothing, so the error that
        interrupted the caller is the one worth reporting.
        """
        if command is None:
            return
        try:
            self.bash(command, description)
        except Exception:
            logger.debug("%s did not run", description, exc_info=True)

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
        pending = self._take_pending()
        if isinstance(pending, PendingAppend) and (file_path, content) == (
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
            return _failure(failure)
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str = "",
        new_string: str = "",
        replace_all: bool = False,
        edits: str = "",
        exact: bool = False,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Path to the file to modify; a relative path resolves
                against the container's working directory (the task's work
                dir in a container kiss started from an image).
            old_string: Exact text to find and replace.  When it is not in
                the file verbatim, the lines that equal its lines once
                runs of blanks are collapsed and trailing whitespace
                dropped are replaced instead, when exactly one such place
                exists (the file's indentation is kept); when none does,
                the error names the region of the file closest to it with
                its line numbers and text, to use as the next old_string.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences (verbatim only).
            edits: Several replacements in one call, as a JSON list of
                ``{"old_string": ..., "new_string": ..., "replace_all": bool}``
                objects applied in order; the first that fails stops the
                batch and the result names the edits already applied.
                ``old_string`` / ``new_string`` are then ignored.
            exact: If True, match verbatim only: no retry ignoring
                whitespace or line endings.  Set on the Edit a shell
                rewrite (a ``sed -i``, a python ``.replace``) is applied as.
        """
        pending = self._take_pending()
        if isinstance(pending, PendingEdit):
            # One critical section for the check and the replacements: no
            # other Write or Edit of this process lands between them.
            return locked_call(
                _FILE_WRITE_LOCK, self._edit_as_rewrite, pending, file_path, old_string,
                new_string, replace_all, edits, exact,
            )
        if edits:
            pairs = _edit_pairs(edits)
            if isinstance(pairs, str):
                return pairs
            results: list[str] = []
            for index, (old_text, new_text, every) in enumerate(pairs, 1):
                result = self.Edit(file_path, old_text, new_text, every, exact=exact)
                if STOPPED_MESSAGE in result:
                    # A task Stop: every later edit would be killed the
                    # same way (an exec and a reaper wait each).
                    return result
                # Only the success line proves the edit landed.
                if not _edited(result):
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
        result = self._replace(file_path, old_string, new_string, replace_all, exact)
        if not result.startswith(_NOT_FOUND):
            return result
        text = self._text(file_path)
        if text is None:
            return result
        if not replace_all and not exact:
            matches = tolerant_matches(text, old_string)
            if len(matches) == 1:
                # The exact replacement again, of the original text of the
                # matched lines at their place in the file (the container
                # scripts stay exact; the same text inside a longer line
                # is not the match), with the file's indentation and line
                # endings on the new text.
                start = matches[0].start
                matched = text[start:matches[0].end]
                new = reindent(new_string, matches[0].indent)
                if "\r\n" in text and "\r\n" not in new:
                    new = new.replace("\n", "\r\n")
                return self._replace(
                    file_path, matched, new, False, at=len(text[:start].encode()),
                )
            if matches:
                return (
                    f"{_NOT_FOUND}; ignoring runs of blanks and trailing whitespace it "
                    f"matches {len(matches)} places, so add surrounding lines to "
                    f"old_string to make it unique"
                )
        region = nearest_region(text, old_string)
        if region is None:
            return result
        first, last, region_text = region
        region_text = region_text.removesuffix("\n")
        return (
            f"{_NOT_FOUND}. The closest region is lines {first}-{last} of {file_path}; "
            f"pass its text verbatim as old_string:\n{region_text}"
        )

    def _edit_as_rewrite(
        self, pending: PendingEdit, file_path: str, old_string: str, new_string: str,
        replace_all: bool, edits: str, exact: bool,
    ) -> str:
        """Make the Edit that follows the redirect of the shell rewrite *pending*.

        When the call is the one the rewrite was redirected to, its
        pairs are made only while they are the ones exact for the text
        the file holds now (:meth:`MechanicalEdit.edits_for
        <kiss.agents.sorcar.edit_guards.MechanicalEdit.edits_for>`), as
        the host ``Edit`` makes them
        (:func:`~kiss.agents.sorcar.useful_tools._edit_locked`): a
        writer since the command was read may have left a text the
        command changes otherwise, or one that cannot be fetched any
        more.  Any other Edit is made as it is.
        """
        pairs = _edit_pairs(edits) if edits else [(old_string, new_string, replace_all)]
        if (file_path, pairs) == (pending.path, pending.edits):
            text = self._text(file_path)
            if text is None or pending.rewrite.edits_for(text) != pairs:
                return (
                    f"Error: {file_path} changed since the {pending.rewrite.what} was read as "
                    "this Edit, which the command no longer stands for: nothing was written. "
                    "Re-issue the command."
                )
        return self.Edit(file_path, old_string, new_string, replace_all, edits, exact)

    def _text(self, file_path: str) -> str | None:
        """Return the UTF-8 text of *file_path* in the container, None when it cannot be fetched.

        None without :attr:`read_bytes`, when the fetch fails (the file
        is missing, unreadable or over the fetch's size cap) and when
        the bytes are not UTF-8: ``Edit`` then stops at the exact match.
        """
        if self.read_bytes is None:
            return None
        try:
            return self.read_bytes(file_path).decode()
        except (OSError, UnicodeDecodeError):
            return None

    def _replace(
        self, file_path: str, old: str, new: str, replace_all: bool, exact: bool = False,
        at: int | None = None,
    ) -> str:
        """Replace *old* with *new* in *file_path* inside the container, verbatim.

        An LF *old* missing from a CRLF file is retried as CRLF unless
        *exact*.  With *at* (a byte offset in the file) the *old* found
        there is the one replaced; a file that changed since and holds
        other bytes there is searched as without it.

        Returns:
            The script's outcome message (``Successfully replaced ...``,
            :data:`_NOT_FOUND`, the not-unique error, ...); any other
            output (a traceback of a non-UTF-8 or unwritable file, a
            payload store that failed) as a failure (:func:`_failure`).
        """
        # The strings travel in a payload file of their own, so their
        # size is not bounded by the shell command's argv limit.
        old_bytes = old.encode()
        payload = f"/tmp/.kiss-edit-{secrets.token_hex(8)}"
        failure = self._store(payload, old_bytes + new.encode(), f"Edit {file_path}")
        if failure is not None:
            return _failure(failure)
        try:
            result = locked_call(
                _FILE_WRITE_LOCK, self.bash,
                _python_edit_command(file_path, payload, len(old_bytes), replace_all, exact, at),
                f"Edit {file_path}",
            )
            if _NO_PYTHON in result:
                # Images without Python (Rust, R, C toolchains) still have
                # perl-base, so the same replacement is done in Perl.
                result = locked_call(
                    _FILE_WRITE_LOCK, self.bash,
                    _perl_edit_command(
                        file_path, payload, len(old_bytes), replace_all, exact, at,
                    ),
                    f"Edit {file_path}",
                )
        except BaseException:
            self._discard(f"rm -f {payload}", f"Edit {file_path} (discard)")
            raise
        # The payload goes whatever the outcome, a timed-out replacement
        # included; after a Stop the removal would be killed the same way
        # (as in _store), so only the uniquely named payload file stays.
        if STOPPED_MESSAGE not in result:
            self.bash(f"rm -f {payload}", f"Edit {file_path} (discard)")
        return result if _edited(result) else _failure(result)
