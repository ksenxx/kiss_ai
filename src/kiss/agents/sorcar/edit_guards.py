# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helpers that make the honest edit the easy edit (P7 of the 2026-10-24 trajectory audit).

* :func:`repo_read_key` keys the read-before-modify bookkeeping by the
  repository-relative path, so a file shown at the main checkout stays
  "read" at its worktree path (and the other way round).
* :func:`shell_view_target` names the single file a ``cat`` / ``sed -n``
  / ``grep -n`` / ``nl`` / ``head`` / ``tail`` command shows, so a
  Bash view counts as a read.
* :func:`rewrite_guard` refuses the shell rewrites the audit saw used
  to dodge ``Edit`` (``sed -i`` with a newline in its script, a python
  heredoc doing ``open(...).read()`` and ``.replace(``) and names the
  ``Edit`` call to make instead.
* :func:`heredoc_guard` refuses a ``cat``/``tee`` here-document that
  writes or appends to an existing file, or to a path only the shell
  can resolve (shared with the coding SEA and the Docker tools); a
  here-document may still create a new file.
* :func:`numbered_region` renders the lines around a string so an
  ``Edit`` refusal, or an ``Edit`` result, carries the context the
  next call needs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from kiss.agents.sorcar.shell_scan import (
    Group,
    SimpleCommand,
    Word,
    bash_path,
    host_path,
    receiver,
    resolve,
    simple_commands,
)

VIEW_COMMANDS = frozenset({"cat", "sed", "grep", "nl", "head", "tail"})
"""Commands whose single file operand is shown to the model."""
# Commands that take a non-file first operand (sed's script, grep's pattern).
_FIRST_OPERAND_IS_NOT_A_FILE = frozenset({"sed", "grep"})
# An in-place option of sed: ``-i``, ``-i.bak``, ``-Ei``, ``--in-place[=.bak]``.
_SED_INPLACE = re.compile(r"-[a-zA-Z]*i|--in-place(?:=|$)")
_PYTHON_HEREDOC = re.compile(r"python3?\s+-\s*<<")
_RE_SUB = re.compile(r"\bre\.sub\(")
# Evidence that a python heredoc writes a file back: ``.write(``,
# ``write_text(``, ``write_bytes(`` or ``open(..., "w"/"a")``.
_PYTHON_WRITE = re.compile(
    r"(?<!stdout)(?<!stderr)\.write(?:_text|_bytes)?\(|open\([^)]*,\s*['\"][wa]"
)
# ...and that it read one first: ``open(`` or ``read_text(``/``read_bytes(``.
_PYTHON_READ = re.compile(r"\bopen\(|\.read_(?:text|bytes)\(")


def repo_read_key(path: Path) -> str:
    """Return the key the read-before-modify bookkeeping uses for *path*.

    Inside a git checkout or worktree the key is the repository's
    identity (its main ``.git`` directory's parent) joined with the
    path relative to the checkout root, so the same file at the main
    checkout and at a ``.kiss-worktrees/`` worktree share one key.
    Outside any repository the key is the resolved path itself.

    Args:
        path: A resolved absolute path.
    """
    for root in (path, *path.parents):
        marker = root / ".git"
        if marker.is_dir():
            identity = root
        elif marker.is_file():
            identity = _main_checkout(marker) or root
        else:
            continue
        return f"{identity}::{path.relative_to(root).as_posix()}"
    return str(path)


def _main_checkout(git_file: Path) -> Path | None:
    """Return the main checkout of the linked worktree whose ``.git`` file is *git_file*.

    A linked worktree's ``.git`` file names ``<main>/.git/worktrees/<name>``
    (absolute, or relative to the worktree).  A submodule's names
    ``<superproject>/.git/modules/<name>`` and a ``--separate-git-dir``
    checkout's some other directory: each is a repository of its own,
    so ``None``.
    """
    try:
        text = git_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = (git_file.parent / text[len("gitdir:"):].strip()).resolve()
    if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
        return gitdir.parent.parent.parent
    return None


def shell_view_target(command: str, cwd: str | None) -> Path | None:
    """Return the one file *command* shows, or ``None``.

    The command is read as bash reads it
    (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`: quotes,
    comments, subshells, ``cd``); the last view command
    (:data:`VIEW_COMMANDS`) with exactly one literal operand naming an
    existing file names the file.  Options are skipped, as is the
    script of ``sed`` and the pattern of ``grep``.

    Args:
        command: The shell command as the model wrote it.
        cwd: The directory it runs in; ``None`` for this process's.
    """
    target: Path | None = None
    start = bash_path(os.path.abspath(cwd or os.getcwd()))
    for simple in simple_commands(command, start):
        program = simple.program()
        if program not in VIEW_COMMANDS:
            continue
        operands = _operands(program, simple.argv()[1:])
        path = resolve(simple.cwd, operands[0]) if len(operands) == 1 else None
        file = None if path is None else Path(host_path(path))
        if file is not None and file.is_file():
            target = file.resolve()
    return target


_LINE_PREFIX = re.compile(r"^\s*\d+\s*[:\t-]\s?")
"""The ``12:`` / ``12-`` of ``grep -n`` and the ``    12\\t`` of ``nl``."""
_SHOWN_LINES_CHECKED = 200
_SHOWN_FILE_BYTES = 4_000_000
"""How much of each end of a file :func:`output_shows_file` reads."""


def output_shows_file(output: str, path: Path) -> bool:
    """Return whether *output* contains at least one line of the file at *path*.

    What ``_note_shell_view`` needs before it counts a shell view as a
    read: ``echo label; true || cat a.py`` names ``a.py`` but shows
    nothing of it.  The first :data:`_SHOWN_LINES_CHECKED` non-blank
    output lines are looked up in the file's text, with the line-number
    prefix of ``grep -n`` / ``nl`` stripped; one-character lines are
    ignored.  Of a file over twice :data:`_SHOWN_FILE_BYTES` only the
    first and the last :data:`_SHOWN_FILE_BYTES` are read, where a
    ``head`` or a ``tail`` of it shows.

    Args:
        output: What the command printed.
        path: The file the command named.
    """
    try:
        text = _file_ends(path)
    except OSError:
        return False
    checked = 0
    for line in output.splitlines():
        shown = _LINE_PREFIX.sub("", line, count=1).strip()
        if not shown:
            continue
        if len(shown) > 1 and shown in text:
            return True
        checked += 1
        if checked >= _SHOWN_LINES_CHECKED:
            break
    return False


def _file_ends(path: Path) -> str:
    """Return the text of *path*: all of it, or its first and last :data:`_SHOWN_FILE_BYTES`.

    The two ends of a longer file are joined by a newline.  Bytes
    appended after the size was taken are not read.

    Raises:
        OSError: When the file cannot be read.
    """
    with open(path, "rb") as file:
        head = file.read(_SHOWN_FILE_BYTES)
        size = file.seek(0, os.SEEK_END)
        offset = file.seek(max(len(head), size - _SHOWN_FILE_BYTES))
        tail = file.read(max(0, size - offset))
    gap = b"\n" if size > len(head) + len(tail) else b""
    return (head + gap + tail).decode("utf-8", errors="replace")


_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "sed": frozenset({"-e", "-f", "--expression", "--file"}),
    "grep": frozenset({"-e", "-f", "-A", "-B", "-C", "-m", "--regexp", "--file"}),
    "head": frozenset({"-n", "-c"}),
    "tail": frozenset({"-n", "-c"}),
    "nl": frozenset({"-n", "-b", "-s", "-w"}),
    "cat": frozenset(),
}
"""Per view command, the options whose value is the next token."""
_SCRIPT_OPTIONS = frozenset({"--expression", "--regexp"})
"""Long options carrying the script/pattern; ``-e`` (joined or separate) is matched by prefix."""


def _operands(program: str, args: list[Word]) -> list[Word]:
    """Return the file operands among the *args* of the view command *program*.

    Those are its words that are no option or option value and do not
    come before the script of ``sed`` / the pattern of ``grep``.
    """
    operands: list[Word] = []
    script_given = False
    skip_next = False
    for word in args:
        token = word.text
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-") and token != "-":
            if token in _SCRIPT_OPTIONS or token.startswith("-e"):
                script_given = True
            skip_next = token in _VALUE_OPTIONS[program]
            continue
        operands.append(word)
    if program in _FIRST_OPERAND_IS_NOT_A_FILE and not script_given:
        operands = operands[1:]
    return operands


def rewrite_guard(command: str) -> str | None:
    """Return the refusal of a shell command that rewrites a file the way ``Edit`` should.

    Refused: ``sed -i`` whose script contains a newline escape (a
    multi-line substitution no one can review) and a ``python3 -``
    heredoc that reads a file, calls ``.replace(`` or ``re.sub(`` on
    it and writes a file back.  Both are the audit's "silent
    rewrites": they skip the read-before-modify rule and leave no
    reviewable diff in the tool log.  A heredoc that only prints the
    transformed text (read-only analysis) is not a rewrite.

    Args:
        command: The shell command as the model wrote it.

    Returns:
        The refusal text naming the ``Edit`` call to make, or ``None``.
    """
    if any(_is_multiline_sed_inplace(simple) for simple in simple_commands(command, None)):
        return (
            "REFUSED: the `sed -i` call did NOT run: it writes a multi-line substitution no "
            "one can review. Make the same change with Edit(file_path, old_string=<the exact "
            "current text>, new_string=<the replacement>) — or pass several pairs in one call "
            "through Edit(file_path, edits=[{\"old_string\": ..., \"new_string\": ...}, ...])."
        )
    if (
        _PYTHON_HEREDOC.search(command)
        and _PYTHON_READ.search(command)
        and (".replace(" in command or _RE_SUB.search(command))
        and _PYTHON_WRITE.search(command)
    ):
        return (
            "REFUSED: the python heredoc did NOT run: it rewrites a file with "
            ".replace()/re.sub(), which "
            "skips the read-before-modify rule and leaves no reviewable change. Make the "
            "same change with Edit(file_path, old_string=..., new_string=...), or several "
            "at once with Edit(file_path, edits=[{\"old_string\": ..., \"new_string\": ...}])."
        )
    return None


def _is_multiline_sed_inplace(command: SimpleCommand) -> bool:
    """Whether *command* is a ``sed`` editing in place with a newline escape in its words."""
    if command.program() != "sed":
        return False
    args = [word.text for word in command.argv()[1:]]
    return any(_SED_INPLACE.match(arg) for arg in args) and any("\\n" in arg for arg in args)


# The commands that copy their input (a here-document) to their outputs verbatim.
_COPIERS = frozenset({"cat", "tee"})
HEREDOC_VERDICT = (
    "REFUSED: the command did not run: it writes the existing file {path} through a "
    "here-document (an append counts). Use Edit(file_path, old_string, new_string) for a "
    "change (Read the region first) or Write(file_path, content) for the whole file: they "
    "are logged and reviewable, a heredoc is not. A here-document may create a new file."
)
"""The refusal :func:`heredoc_guard` returns for the existing target ``{path}``."""
HEREDOC_UNRESOLVED_VERDICT = (
    "REFUSED: the command did not run: it writes {path} through a here-document, and only "
    "the shell knows which file that is, so it may be an existing one. Use "
    "Write(file_path, content) with the literal path, or Edit(file_path, old_string, "
    "new_string) for a change to an existing file. A here-document may create a new file "
    "at a literal path."
)
"""The refusal :func:`heredoc_guard` returns for a target ``{path}`` only the shell resolves."""


def _output_files(
    reader: SimpleCommand, copies_only: bool, commands: list[SimpleCommand],
) -> list[tuple[str | None, Word]]:
    """Return ``(directory, file)`` for each file the output of *reader* reaches.

    The output goes to the files it is redirected to, else on to its
    :func:`~kiss.agents.sorcar.shell_scan.receiver` (the command it is
    piped to, or its group); ``tee`` also writes it to its operands.
    Piped into a group, it reaches what the group's commands that read
    the group's input write (*commands* holds them all).  With
    *copies_only*, it stops at the first command that is not ``cat`` or
    ``tee``: that one writes output of its own.
    """
    writes: list[tuple[str | None, Word]] = []
    node: SimpleCommand | Group | None = reader
    while node is not None:
        if isinstance(node, SimpleCommand):
            program = node.program()
            if copies_only and program not in _COPIERS:
                break
            if program == "tee":
                operands = node.argv()[1:]
                writes.extend((node.cwd, word) for word in operands if word.text[:1] != "-")
        if node.outputs:
            writes.extend((node.cwd, word) for word in node.outputs)
            break
        if isinstance(node.reader, Group):
            for member in _input_readers(node.reader, commands):
                writes.extend(_output_files(member, copies_only, commands))
            break
        node = receiver(node)
    return writes


def _input_readers(group: Group, commands: list[SimpleCommand]) -> list[SimpleCommand]:
    """Return the commands of *commands* in *group* that read the group's standard input.

    One whose own input, or that of a group between it and *group*,
    is a pipe or a here-document reads that instead.
    """
    readers: list[SimpleCommand] = []
    for command in commands:
        node: SimpleCommand | Group | None = command
        while node is not None and node is not group and not (node.piped or node.heredoc):
            node = node.group
        if node is group:
            readers.append(command)
    return readers


def _heredoc_writes(
    command: str, cwd: str | None, copies_only: bool,
) -> list[tuple[str | None, Word]]:
    """Return ``(directory, file)`` for each file a here-document pipeline of *command* writes.

    Each command reading a here-document writes the :func:`_output_files`
    of its output.  With *copies_only*, only ``cat`` and ``tee`` count,
    which copy the here-document's text: a heredoc fed to an interpreter
    (``python3 - <<EOF > out.json``) writes the program's output, and
    a pipeline stops copying at its first other command.

    Args:
        command: The shell command text.
        cwd: The directory it starts in (``None``: unknown).
        copies_only: Whether only ``cat``/``tee`` copies count.
    """
    writes: list[tuple[str | None, Word]] = []
    commands = simple_commands(command, cwd)
    for reader in commands:
        if reader.reads_heredoc():
            writes.extend(_output_files(reader, copies_only, commands))
    return writes


def heredoc_targets(command: str) -> list[str]:
    """Return the files the here-document pipelines of *command* write, unquoted, as written.

    A program's output counts too: the coding SEA's ledger gate treats
    every such write as an edit.

    Args:
        command: The shell command as the model wrote it.
    """
    return [word.text for _, word in _heredoc_writes(command, None, copies_only=False)]


def heredoc_guard(command: str, cwd: str | None, exists: Callable[[str], bool]) -> str | None:
    """Refuse a ``cat``/``tee`` here-document copy into a file that exists or may exist.

    A copy into a file that does not exist yet creates it and may run.
    A copy into an existing file, an append (``>>``, ``tee -a``)
    included, is a rewrite for ``Edit``/``Write``: it skips the
    read-before-modify rule and leaves no reviewable change in the tool
    log.  A target holding an expansion (``"$OUT"``, ``~/x``), or a
    relative one after a ``cd`` only the shell can resolve, cannot be
    checked and is refused too.  ``/dev/`` targets are not files.

    Args:
        command: The shell command as the model wrote it.
        cwd: The absolute POSIX directory the command starts in;
            ``None`` when unknown (every relative target is refused).
        exists: Whether a path exists where the command runs (the host,
            a container or an exec bridge's environment).

    Returns:
        :data:`HEREDOC_VERDICT` or :data:`HEREDOC_UNRESOLVED_VERDICT`, or
        ``None`` when the command may run.
    """
    for directory, target in _heredoc_writes(command, cwd, copies_only=True):
        path = resolve(directory, target)
        if path is None:
            return HEREDOC_UNRESOLVED_VERDICT.format(path=target.text)
        if not path.startswith("/dev/") and exists(path):
            return HEREDOC_VERDICT.format(path=path)
    return None


def numbered_region(content: str, start_line: int, end_line: int) -> str:
    """Return lines *start_line*..*end_line* (1-based, clipped) of *content* with line numbers."""
    lines = content.splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    return "\n".join(f"{n:5d} | {lines[n - 1]}" for n in range(start, end + 1))


def line_of(content: str, position: int) -> int:
    """Return the 1-based line number of character *position* in *content*."""
    return content.count("\n", 0, position) + 1


def region_around_string(content: str, needle: str, context: int = 10) -> str:
    """Return the numbered lines around the first line of *needle* in *content*.

    When *needle* does not occur, the lines around its first line's
    best match (the first line containing the needle's first line
    stripped) are shown; when nothing matches, the file's head.

    Args:
        content: The file text.
        needle: The ``old_string`` the model wanted to replace.
        context: Lines shown on each side.
    """
    first_line = next((ln.strip() for ln in needle.splitlines() if ln.strip()), "")
    position = content.find(needle)
    if position < 0 and first_line:
        position = content.find(first_line)
    if position < 0:
        return numbered_region(content, 1, 2 * context)
    anchor = line_of(content, position)
    needle_lines = max(1, len(needle.rstrip("\n").splitlines()))
    return numbered_region(content, anchor - context, anchor + needle_lines - 1 + context)
