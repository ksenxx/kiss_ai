# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helpers that make the honest edit the easy edit.

* :func:`repo_read_key` keys the read-before-modify bookkeeping by the
  repository-relative path, so a file shown at the main checkout stays
  "read" at its worktree path (and the other way round).
* :func:`shell_view_target` names the single file a ``cat`` / ``sed -n``
  / ``grep -n`` / ``nl`` / ``head`` / ``tail`` command shows, so a
  Bash view counts as a read.
* :func:`rewrite_guard` refuses the shell rewrites models use to dodge
  ``Edit`` (``sed -i`` with a newline in its script, a python
  heredoc doing ``open(...).read()`` and ``.replace(``) and names the
  ``Edit`` call to make instead.
* :func:`heredoc_guard` refuses a ``cat``/``tee`` here-document that
  writes or appends to an existing file, or to a path only the shell
  can resolve (shared with the coding SEA and the Docker tools); a
  here-document may still create a new file.
* :func:`heredoc_copy` and :func:`mechanical_edit` read the rewrites
  whose accepted alternative is mechanical (a ``cat``/``tee``
  here-document copy with a literal body, a python ``.replace`` rewrite,
  a one-substitution ``sed -i``) as the ``Write`` / ``Edit`` call they
  stand for, so the host ``Bash`` tool applies that call instead of
  refusing.  A conversion is exact or does not happen: what sed,
  python or bash would have done differently is left to the refusal.
  The rewrite may be one step of ``[cd DIR &&] <rewrite> [&& | ; |
  newline] <rest>`` (:func:`_rewrite_segment`): the result then carries
  the directory and the rest of the command for the caller to run after
  the call.
* :func:`numbered_region` renders the lines around a string so an
  ``Edit`` refusal, or an ``Edit`` result, carries the context the
  next call needs.
"""

from __future__ import annotations

import ast
import codecs
import json
import os
import posixpath
import re
import shlex
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kiss.agents.sorcar.edit_matching import tolerant_matches
from kiss.agents.sorcar.shell_scan import (
    Group,
    SimpleCommand,
    Word,
    _continued,
    bash_path,
    cd_path,
    handed_text,
    host_path,
    receiver,
    resolve,
    simple_commands,
)

VIEW_COMMANDS = frozenset({"cat", "sed", "grep", "nl", "head", "tail"})
"""Commands whose single file operand is shown to the model."""
# Commands that take a non-file first operand (sed's script, grep's pattern).
_FIRST_OPERAND_IS_NOT_A_FILE = frozenset({"sed", "grep"})
# An in-place option of sed: ``-i``, ``-i.bak``, ``-Ei``, ``--in-place[=.bak]`` or a
# prefix of it (``--in-pl``, ``--i=.bak``: GNU sed's only long option in ``i``).
_SED_INPLACE = re.compile(r"-[a-zA-Z]*i|--i[a-z-]*(?:=|$)")
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
    "grep": frozenset({
        "-e", "-f", "-A", "-B", "-C", "-m", "--regexp", "--file",
        "--after-context", "--before-context", "--context", "--max-count",
    }),
    "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "tail": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "nl": frozenset({"-n", "-b", "-s", "-w"}),
    "cat": frozenset(),
}
"""Per view command, the options whose value is the next token."""
_SCRIPT_OPTIONS = frozenset({"--expression", "--regexp", "--file"})
"""Long options carrying the script/pattern or its file, the value attached with ``=`` or not.

``-e`` and ``-f`` (joined or separate) are matched by prefix.
"""


def _operands(program: str, args: list[Word]) -> list[Word]:
    """Return the file operands among the *args* of the view command *program*.

    Those are its words that are no option or option value and do not
    come before the script of ``sed`` / the pattern of ``grep``; every
    word after ``--`` is an operand.
    """
    operands: list[Word] = []
    script_given = False
    skip_next = False
    for index, word in enumerate(args):
        token = word.text
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            operands += args[index + 1:]
            break
        if token.startswith("-") and token != "-":
            if token.split("=", 1)[0] in _SCRIPT_OPTIONS or token.startswith(("-e", "-f")):
                script_given = True
            skip_next = token in _VALUE_OPTIONS[program]
            continue
        operands.append(word)
    if program in _FIRST_OPERAND_IS_NOT_A_FILE and not script_given:
        operands = operands[1:]
    return operands


def _unguarded(
    cwd: str | None, targets: list[Word], guarded: Callable[[str], bool] | None,
) -> bool:
    """Whether a rewrite of every file of *targets* may run as written.

    With *guarded* (see :func:`rewrite_guard`) that is so when each
    target resolves in *cwd* to a path *guarded* does not guard; a
    target only the shell resolves is guarded.  Without it nothing is.
    """
    if guarded is None or not targets:
        return False
    paths = [resolve(cwd, target) for target in targets]
    return all(path is not None and not guarded(path) for path in paths)


def _handed_program(reader: SimpleCommand) -> str | None:
    """The text of the here-document the python command *reader* reads, or ``None``."""
    node = reader.heredoc_reader()
    return node.heredoc_body if node is not None else None


def _rewrites_unguarded(
    reader: SimpleCommand, program: str | None, guarded: Callable[[str], bool] | None,
) -> bool:
    """Whether the python *program* the command *reader* runs rewrites unguarded files only.

    The files it writes must all be known (:func:`python_written_paths`)
    and pass :func:`_unguarded` in the command's directory.
    """
    if program is None:
        return False
    written = python_written_paths(program)
    return written is not None and _unguarded(
        reader.cwd, [Word(path, True) for path in written], guarded)


def rewrite_guard(
    command: str, cwd: str | None = None, guarded: Callable[[str], bool] | None = None,
    isdir: Callable[[str], bool] | None = None,
) -> str | None:
    """Return the refusal of a shell command that rewrites a file the way ``Edit`` should.

    Refused: ``sed -i`` whose script contains a newline escape (a
    multi-line substitution no one can review) and a ``python3 -``
    heredoc that reads a file, calls ``.replace(`` or ``re.sub(`` on
    it and writes a file back.  Both are "silent rewrites": they skip
    the read-before-modify rule and leave no reviewable diff in the
    tool log.  A heredoc that only prints the transformed text
    (read-only analysis) is not a rewrite.

    With *guarded*, a rewrite of files that are not guarded runs as
    written: the ``sed -i`` when every file operand resolves to such a
    path, the python heredoc when every file it writes
    (:func:`python_written_paths`) is a literal resolving to one; a
    path the program builds at run time (an f-string, ``sys.argv``) may
    name any file and keeps the refusal.  The 2026-11-01 audit counted
    63 of 68 such refusals on scratch files.

    Args:
        command: The shell command as the model wrote it.
        cwd: The absolute POSIX directory it starts in, where a relative
            file resolves; ``None`` when unknown.
        guarded: Whether a rewrite of the absolute bash path given is
            refused (an existing file outside the scratch directories);
            ``None`` refuses every rewrite, as before.
        isdir: Whether an absolute bash path is a directory where the
            command runs: a ``cd`` bash does not walk as written then
            leaves a directory only the shell knows
            (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`),
            where every file is guarded.

    Returns:
        The refusal text naming the ``Edit`` call to make (with the file
        and the strings when the command names them literally), or ``None``.
    """
    commands = simple_commands(command, cwd, isdir=isdir)
    for simple in commands:
        if _is_multiline_sed_inplace(simple):
            if _unguarded(simple.cwd, _operands("sed", simple.argv()[1:]), guarded):
                continue
            operand, edits = sed_edit(simple)
            file = "<the file>" if operand is None else json.dumps(operand.text)
            call = (
                edit_call(operand.text, edits) if operand is not None and edits else
                f"Edit(file_path={file}, old_string=<the exact current text>, "
                "new_string=<the replacement>)"
            )
            return (
                "REFUSED: the `sed -i` call did NOT run: it writes a multi-line substitution no "
                f"one can review. Make the same change with {call} — or pass several pairs in "
                f"one call through Edit(file_path={file}, edits=[{{\"old_string\": ..., "
                "\"new_string\": ...}, ...])."
            )
    if (
        _PYTHON_HEREDOC.search(command)
        and _PYTHON_READ.search(command)
        and (".replace(" in command or _RE_SUB.search(command))
        and _PYTHON_WRITE.search(command)
    ):
        python = [c for c in commands if c.reads_heredoc() and c.program().startswith("python")]
        if not python:
            # No python reads a here-document: a ``python3 - <<`` in the
            # body of a here-document a copier copies into files
            # (``cat`` copying a script into a scratch file,
            # :func:`_copies_to_files`) is data; one anywhere else (the
            # body a shell runs, one a copier pipes on to a shell, a
            # quoted string or an array value the scanner does not read
            # into) is code it cannot see into and keeps the refusal.
            # Counted, not cut out of the text: a copy of the text does
            # not hide the one that runs.
            copied = sum(
                len(_PYTHON_HEREDOC.findall(simple.heredoc_body))
                for simple in commands
                if simple.heredoc_body is not None and simple.heredoc_body in command
                and _copies_to_files(simple, commands)
            )
            if copied == len(_PYTHON_HEREDOC.findall(command)):
                return None
        whole = heredoc_text(command)
        # The program each python reads: the here-document that is the
        # whole command, else the text the scanner kept on the node
        # reading it (a ``cd`` before it, commands after it, a second
        # here-document).  The command runs as written only when EVERY
        # python's program is known and rewrites unguarded files alone.
        programs = [whole if whole is not None else _handed_program(reader) for reader in python]
        if python and all(_rewrites_unguarded(reader, program, guarded)
                          for reader, program in zip(python, programs)):
            return None
        program = whole if whole is not None else next(
            (text for text in programs if text is not None), None)
        path, edits, _ = python_edit(program) if program is not None else (None, [], False)
        file = "<the literal path>" if path is None else json.dumps(path)
        call = (
            edit_call(path, edits) if path is not None and edits else
            f"Edit(file_path={file}, old_string=..., new_string=...)"
        )
        return (
            "REFUSED: the python heredoc did NOT run: it rewrites "
            f"{'a file' if path is None else path} with .replace()/re.sub(), which "
            "skips the read-before-modify rule and leaves no reviewable change. Make the "
            f"same change with {call}, or several at once with Edit(file_path={file}, "
            "edits=[{\"old_string\": ..., \"new_string\": ...}])."
        )
    return None


def _is_sed_inplace(command: SimpleCommand) -> bool:
    """Whether *command* is a ``sed`` editing in place (``-i``, ``-i.bak``, ``--in-place``)."""
    if command.program() != "sed":
        return False
    return any(_SED_INPLACE.match(word.text) for word in command.argv()[1:])


def _is_multiline_sed_inplace(command: SimpleCommand) -> bool:
    """Whether *command* is a ``sed`` editing in place with a newline escape in its words."""
    return _is_sed_inplace(command) and any("\\n" in word.text for word in command.argv()[1:])


PREVIEW_CHARS = 80
"""Characters of a string an accepted call's argument shows before an ellipsis."""


def preview(text: str) -> str:
    """Return *text* as a JSON string literal, cut to :data:`PREVIEW_CHARS` characters."""
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "…"
    return json.dumps(text, ensure_ascii=False)


def edit_call(file_path: str, edits: list[tuple[str, str, bool]]) -> str:
    """Render the ``Edit`` call applying *edits* (``(old, new, replace_all)``) to *file_path*.

    Strings are cut to :data:`PREVIEW_CHARS`, so the call is a template
    the model fills from the file when a string was cut.
    """
    if len(edits) == 1:
        old, new, every = edits[0]
        flag = ", replace_all=True" if every else ""
        return (
            f"Edit(file_path={json.dumps(file_path)}, old_string={preview(old)}, "
            f"new_string={preview(new)}{flag})"
        )
    items = ", ".join(
        f'{{"old_string": {preview(old)}, "new_string": {preview(new)}'
        + (', "replace_all": true}' if every else "}")
        for old, new, every in edits
    )
    return f"Edit(file_path={json.dumps(file_path)}, edits=[{items}])"


# ``<<`` or ``<<-`` that is no here-string (``<<<``) and no part of one.
_HEREDOC_OPERATOR = re.compile(r"(?<!<)<<(?!<)(-?)[ \t]*")
_HEREDOC_DELIMITER_END = frozenset(" \t\n;&|<>()")


def _heredoc_delimiter(text: str) -> tuple[str, bool] | None:
    """Return the delimiter word at the start of *text*, unquoted, and whether it was quoted."""
    parts: list[str] = []
    quoted = False
    position = 0
    while position < len(text) and text[position] not in _HEREDOC_DELIMITER_END:
        char = text[position]
        if char in "'\"":
            end = text.find(char, position + 1)
            if end < 0:
                return None
            parts.append(text[position + 1:end])
            quoted, position = True, end + 1
        elif char == "\\":
            if position + 1 >= len(text):
                return None
            parts.append(text[position + 1])
            quoted, position = True, position + 2
        else:
            parts.append(char)
            position += 1
    return ("".join(parts), quoted) if parts else None


def _first_line(command: str) -> tuple[str, str]:
    """Return the first non-blank line of *command* and the text after it."""
    head, _, rest = command.lstrip(" \t\n").partition("\n")
    return head, rest


def _heredoc_body(line: str, rest: str) -> tuple[str, str] | None:
    """Return the body of the here-document the command *line* opens, and the text after it.

    *line* must hold one ``<<`` / ``<<-`` operator and its delimiter,
    with no ``<`` after the delimiter: a later ``< file``, ``<<< word``,
    ``<>``, ``<&`` or second here-document would replace this one as
    the standard input (bash honours the last redirection).  The body
    is read from *rest*, the command's lines after *line*, up to the
    delimiter line; a quoted delimiter keeps the body as written, an
    unquoted one joins a line ending in a backslash with the next
    (:func:`_continued`) before the delimiter is looked for, as bash
    does, and keeps the body only without ``$`` or backticks
    (:func:`~kiss.agents.sorcar.shell_scan.handed_text`); ``<<-`` strips the leading tabs of
    the body and delimiter lines (of a joined line, those of its start).

    Returns:
        The body, every line newline-terminated, and the text of *rest*
        after the delimiter line; ``None`` when *line* opens no such
        here-document, *rest* does not end it, or the shell would expand it.
    """
    match = _HEREDOC_OPERATOR.search(line)
    if match is None or line.startswith("#") or line.endswith("\\"):
        return None
    delimiter = _heredoc_delimiter(line[match.end():])
    if delimiter is None or "<" in line[match.end():]:
        return None
    word, quoted = delimiter
    strip_tabs = match.group(1) == "-"
    lines = rest.split("\n")
    body: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        while not quoted and _continued(raw) and index + 1 < len(lines):
            index += 1
            raw = raw[:-1] + lines[index]
        text = raw.lstrip("\t") if strip_tabs else raw
        if text == word:
            joined = "".join(part + "\n" for part in body)
            handed = handed_text(joined, not quoted)
            return None if handed is None else (handed, "\n".join(lines[index + 1:]))
        body.append(text)
        index += 1
    return None


def heredoc_text(command: str) -> str | None:
    """Return the body of the one here-document of *command* as the shell hands it on.

    The here-document must open on the first non-blank line of the
    command, be the only input redirection from its ``<<`` on
    (:func:`_heredoc_body`), and end before any other text.

    Args:
        command: The shell command text.

    Returns:
        The body, every line newline-terminated, or ``None`` when the
        command has no such here-document or the shell would expand it.
    """
    head, rest = _first_line(command)
    parsed = _heredoc_body(head, rest)
    if parsed is None or parsed[1].strip():
        return None
    return parsed[0]


# The control operators of a command line, and the ``cd`` opening a segment.
_CONTROL_OPERATOR = re.compile(r"&&|\|\||;;&?|;&|;|\|&|\||&")
_CD_SEGMENT = re.compile(r"cd(?:[ \t]|$)")


def _segments(line: str) -> list[tuple[str, str]] | None:
    """Split the one-line command *line* at its control operators into ``(text, operator)`` pairs.

    The operator (``&&``, ``;``, ``||``, ``|``, ``&``...) ends the text
    before it; the last text's operator is ``""``.  Quotes and
    backslashes are honoured, the ``&`` or ``|`` of a redirection
    (``2>&1``, ``<&0``, ``&>``, ``>|``) is no operator, and a comment
    ends the line.  ``None`` when the line holds a subshell, a command
    substitution, a backtick or an unterminated quote: its texts are
    then not the simple commands the scanner lists, one each, in order.
    """
    segments: list[tuple[str, str]] = []
    quote = ""
    start = position = 0
    while position < len(line):
        char = line[position]
        if char == "\\" and quote != "'":
            position += 2
        elif quote:
            if char == quote:
                quote = ""
            elif quote == '"' and (char == "`" or line.startswith("$(", position)):
                return None
            position += 1
        elif char in "'\"":
            quote, position = char, position + 1
        elif char in "()`":
            return None
        elif char in "&|" and line[position - 1:position] in ("<", ">"):
            position += 1
        elif line.startswith("&>", position):
            position += 2
        elif char == "#" and (position == 0 or line[position - 1] in " \t;&|"):
            break
        elif operator := _CONTROL_OPERATOR.match(line, position):
            segments.append((line[start:position], operator[0]))
            start = position = operator.end()
        else:
            position += 1
    if quote:
        return None
    segments.append((line[start:], ""))
    return segments


# The redirection operators of standard output to a file, and of both
# standard output and standard error to one.
_STDOUT_OPERATORS = frozenset({">", ">>", ">|", ">&"})
_BOTH_OPERATORS = frozenset({"&>", "&>>"})


def _exact_redirections(simple: SimpleCommand) -> bool:
    """Whether every redirection of *simple* is one the conversions reproduce.

    Those are its here-document on standard input, its standard output
    to a file (``>``, ``>>``, ``>|``, ``>&`` with a file operand, ``&>``,
    ``&>>``), its standard error to standard output (``2>&1``) or to
    ``/dev/null``, and ``/dev/null`` as its standard input (the one file
    that always opens; a here-document after it is the input that
    counts, one before it is not: :func:`_heredoc_body`): a
    ``Write``/``Edit`` of the file leaves what they leave.  Any other is
    not reproduced: an input file (bash runs nothing when it cannot open
    it), a duplication such as ``1>&2`` (the output goes elsewhere), a
    standard-error file (the shell creates it), a descriptor of the
    command's own (``3> g``, ``3<< EOF``).
    """
    for fd, operator, operand in simple.redirections:
        if operator in ("<<", "<<-"):
            exact = fd in ("", "0")
        elif fd in ("", "0") and operator == "<":
            exact = operand.text == "/dev/null"
        elif operator in _BOTH_OPERATORS:
            exact = True
        elif fd in ("", "1") and operator in _STDOUT_OPERATORS:
            exact = not (operator == ">&" and (operand.text.isdigit() or operand.text == "-"))
        elif fd == "2" and operator == ">&":
            exact = operand.text == "1"
        elif fd == "2" and operator == ">":
            exact = operand.text == "/dev/null"
        else:
            exact = False
        if not exact:
            return False
    return True


@dataclass(frozen=True)
class _Rewrite:
    """The one rewrite step of a command, as :func:`_rewrite_segment` reads it.

    Attributes:
        simple: The rewrite's simple command.
        text: Its text on the command's first line.
        body: Its here-document's body as the shell hands it on; ``None``
            without one.
        cwd: The absolute path of the ``cd`` before it, as written
            (:func:`~kiss.agents.sorcar.shell_scan.cd_path`); ``""``
            without a ``cd``.
        remainder: The rest of the command after it; ``""`` without one.
    """

    simple: SimpleCommand
    text: str
    body: str | None
    cwd: str
    remainder: str


def _rewrite_segment(command: str, cwd: str | None) -> _Rewrite | None:
    """Read *command* as ``[cd DIR &&] <rewrite> [&& | ; | newline] <rest>``; return the rewrite.

    The grammar, on the first non-blank line of the command:

    * an optional ``cd DIR`` segment, ended by ``&&``, whose ``DIR``
      resolves to an absolute directory (a ``cd`` to a path only the
      shell knows, or ended by ``;``, is not read);
    * then the rewrite: the first simple command after it, named by its
      program word (no ``!``, wrapper or reserved word before it), at
      the top level (no subshell, group, pipe or ``&``), ended by
      ``&&``, ``;`` or the end of the line, with its here-document, if
      any, read as :func:`_heredoc_body` reads it;
    * then the rest: the remainder of the line after that operator and
      the lines after the here-document (or after the first line),
      which hold no here-document and no ``sed`` in place.

    A line holding a subshell, a command substitution, a backtick, an
    unterminated quote or a line continuation is not read
    (:func:`_segments`), nor is a rewrite that is not the first segment
    after the ``cd`` (an earlier segment may change the file).  A
    command that is one simple command without a here-document, at the
    top level, not negated and not in the background, is the rewrite as
    it is (a line continuation or a comment line before it is fine
    then).  The rewrite's redirections must be the ones the conversions
    model (:func:`_exact_redirections`).

    Args:
        command: The shell command text.
        cwd: The absolute POSIX directory it starts in; ``None`` when unknown.

    Returns:
        The rewrite with the path of its ``cd`` as written (a ``..``
        stays with the name before it) and the rest of the command, or
        ``None`` when *command* is not of that shape.  A caller
        applying the call should first check that bash walks that path
        (:func:`~kiss.agents.sorcar.shell_scan.walks`: it runs nothing
        when the ``cd`` fails, and enters another directory than the
        rewrite's when it gets there through a link only) and run the
        rest only when the call went through.
    """
    commands = simple_commands(command, cwd)
    if len(commands) == 1 and not commands[0].heredoc and commands[0].group is None:
        # One command: its text is the whole command, whose operators
        # (a trailing ``&``) the segments of the whole text show.
        segments = _segments(command)
        if (
            segments is None
            or commands[0].negated  # the shell inverts its status: not an Edit's
            or any(operator not in ("", ";") for _, operator in segments)
            or not _exact_redirections(commands[0])
        ):
            return None
        return _Rewrite(commands[0], command, None, "", "")
    head, rest = _first_line(command)
    segments = _segments(head)
    if segments is None or not commands or head.endswith("\\"):
        return None
    index = 0
    directory = ""
    if _CD_SEGMENT.match(segments[0][0].lstrip()):
        # Another command follows the cd (a lone one is read above), so commands[1] exists.
        if segments[0][1] != "&&" or commands[0].heredoc:
            return None
        index = 1
    text, operator = segments[index]
    rewrite = commands[index]
    if index:
        entered = cd_path(commands[0].cwd, commands[0].argv()[1:])
        if rewrite.cwd is None or entered is None:
            return None
        directory = entered
    if (
        rewrite.group is not None
        or not rewrite.words
        or not text.lstrip().startswith(rewrite.words[0].text)
        or operator not in ("", "&&", ";")
        or not _exact_redirections(rewrite)
    ):
        return None
    for other in commands[index + 1:]:
        if other.reads_heredoc() or _is_sed_inplace(other):
            return None
    body = None
    after = rest
    if rewrite.heredoc:
        parsed = _heredoc_body(text.strip(), rest)
        if parsed is None:
            return None
        body, after = parsed
    rest_of_line = "".join(part + operator for part, operator in segments[index + 1:])
    remainder = "\n".join(part for part in (rest_of_line.strip(), after.strip()) if part)
    return _Rewrite(rewrite, text, body, directory, remainder)


@dataclass(frozen=True)
class HeredocCopy:
    """A ``cat``/``tee`` here-document copy into one file: the ``Write`` it stands for.

    Attributes:
        target: The file as the command names it, quotes removed.
        path: Its absolute POSIX path; ``None`` when only the shell knows it.
        body: The here-document's text as the shell hands it to the command.
        append: Whether the copy appends (``>>``, ``tee -a``) rather than overwrites.
        cwd: The absolute POSIX path of the ``cd`` before the copy, as
            written (:attr:`_Rewrite.cwd`), where the ``remainder`` runs;
            ``""`` when there is no ``cd``.
        remainder: The rest of the command, to run after the ``Write``:
            the first line's part after the ``&&`` / ``;`` and the lines
            after the here-document, joined with a newline; ``""`` when
            the copy is the whole command.
    """

    target: str
    path: str | None
    body: str
    append: bool
    cwd: str = ""
    remainder: str = ""


_TEE_OPTIONS = frozenset({"-a", "--append", "-i", "--ignore-interrupts"})


def _names_a_directory(target: str) -> bool:
    """Whether the path *target* can only name a directory.

    Its last component is then empty (``f/``), ``.`` or ``..``.  The
    system opens no file by such a name (``f/`` for a file ``f``: not a
    directory), while :func:`resolve` drops the ending and names ``f``:
    no ``Write`` or ``Edit`` stands for a command writing it.
    """
    return posixpath.basename(target) in ("", ".", "..")


def _home_path(target: Word, home: str) -> str | None:
    """Return the absolute POSIX path of a ``~`` / ``~/x`` *target* in the home directory *home*.

    ``None`` when more than the tilde may expand in it (a variable, a
    pattern, braces: only the shell knows the file then).  A ``..``
    stays in the path for the filesystem, as in any other
    (:func:`~kiss.agents.sorcar.shell_scan.resolve`).
    """
    rest = target.text[1:]
    if target.literal or target.text[:1] != "~" or rest[:1] not in ("", "/"):
        return None
    if any(char in rest for char in "$`*?[{"):
        return None
    return resolve(None, Word(home + rest, literal=True))


def heredoc_copy(command: str, cwd: str | None, home: str | None = "~") -> HeredocCopy | None:
    """Return the one literal here-document copy that *command* is, or ``None``.

    Qualifies: ``cat`` (no option) or ``tee`` (``-a``/``--append``/``-i``
    at most, no wrapper) with the here-document on its standard input
    and one output file (``cat``'s one redirection; ``tee``'s one
    operand, its standard output at most to ``/dev/null``), whose body
    the shell copies verbatim (:func:`_heredoc_body`), as the whole
    command or as the rewrite step of ``[cd DIR &&] <copy> [&& | ; |
    newline] <rest>`` (:func:`_rewrite_segment`; the result then names
    the ``cd`` directory and the rest).  A ``~`` / ``~/x`` target is
    resolved in *home*; one that can only name a directory (``f/``) is
    no copy, the shell opens no file by it (:func:`_names_a_directory`).

    Args:
        command: The shell command text.
        cwd: The absolute POSIX directory it starts in; ``None`` when unknown.
        home: The home directory of the shell that runs the command,
            which its ``~`` names: this process's own by default
            (``"~"``, the host shell's), a container's or a remote
            shell's when the command runs there, or ``None`` when that
            is unknown, which leaves a ``~`` target's ``path`` ``None``.
    """
    rewrite = _rewrite_segment(command, cwd)
    if rewrite is None or rewrite.body is None:
        return None
    simple = rewrite.simple
    argv = simple.argv()
    if simple.words != argv:
        return None
    options = [word.text for word in argv[1:] if word.text.startswith("-") and word.text != "-"]
    operands = [word for word in argv[1:] if not word.text.startswith("-") or word.text == "-"]
    program = simple.program()
    if program == "cat" and not options and all(word.text == "-" for word in operands):
        if len(simple.outputs) != 1:
            return None
        target = simple.outputs[0]
    elif program == "tee" and set(options) <= _TEE_OPTIONS and len(operands) == 1:
        if any(word.text != "/dev/null" for word in simple.outputs):
            return None
        target = operands[0]
    else:
        return None
    if ">>" in target.text or _names_a_directory(target.text):
        return None
    if program == "tee":
        append = bool({"-a", "--append"} & set(options))
    else:
        # The one standard-output redirection (_exact_redirections): its
        # operator, not the text, which may say ``>>`` in a comment or a delimiter.
        append = any(operator in (">>", "&>>") for _, operator, _ in simple.redirections)
    path = resolve(simple.cwd, target)
    if path is None and home is not None:
        path = _home_path(target, bash_path(os.path.expanduser(home)))
    return HeredocCopy(target.text, path, rewrite.body, append, rewrite.cwd, rewrite.remainder)


@dataclass(frozen=True)
class MechanicalEdit:
    """The ``Edit`` call a python here-document or a ``sed -i`` stands for.

    Attributes:
        what: The command, for its result or refusal: ``"python heredoc"``
            or ``"`sed -i` call"``.
        path: The absolute POSIX path of the file; ``None`` when only the
            shell knows it.
        target: The file as the command names it.
        edits: ``(old, new, replace_all)`` per replacement, in order.
        cwd: The absolute POSIX path of the ``cd`` before the rewrite, as
            written (:attr:`_Rewrite.cwd`), where the ``remainder`` runs;
            ``""`` when there is no ``cd``.
        remainder: The rest of the command, to run after the ``Edit``: the
            first line's part after the ``&&`` / ``;`` and the lines after
            the here-document (or the first line), joined with a newline;
            ``""`` when the rewrite is the whole command.
        per_line: Whether the one substitution is a ``sed`` without
            ``g``, which replaces the first match of every record (a
            line, or with ``-z`` a NUL-separated record: *separator*):
            ``edits`` then replaces one match, which is exact only while
            the file holds one; :meth:`edits_for` settles the exact call
            from the file's text.
        separator: The record separator of a ``sed``: ``"\\n"``, or
            ``"\\0"`` with ``-z``.
        universal_newlines: Whether the python rewrite reads the file
            with universal newlines (a ``newline`` keyword left out or
            ``None``), which turns a ``\\r\\n`` or ``\\r`` of the file into
            ``\\n`` before the replacements and writes it back so: exact
            only while the file holds no ``\\r``; :meth:`edits_for`
            settles it from the file's text.
    """

    what: str
    path: str | None
    target: str
    edits: list[tuple[str, str, bool]]
    cwd: str = ""
    remainder: str = ""
    per_line: bool = False
    universal_newlines: bool = False
    separator: str = "\n"

    @property
    def text_dependent(self) -> bool:
        """Whether :meth:`edits_for` depends on the file's text (else it is :attr:`edits`)."""
        return self.per_line or self.universal_newlines or len(self.edits) > 1 or any(
            "\n" in old and "\r\n" not in old for old, _, _ in self.edits
        )

    def edits_for(self, content: str) -> list[tuple[str, str, bool]] | None:
        """Return the ``(old, new, replace_all)`` pairs exact for the file text *content*.

        A :attr:`universal_newlines` read has no ``Edit`` equivalent for
        a file holding a ``\\r``: the rewrite would turn its line endings
        into ``\\n``, which ``Edit`` keeps.  A :attr:`per_line`
        substitution depends on the text too: with several matches, each
        alone in its record (:attr:`separator`), sed replaces them all,
        so the ``Edit`` gets ``replace_all``; a record holding the
        pattern twice has no ``Edit`` equivalent (sed would replace its
        first match only).  The pairs
        are judged in order, each against the text the earlier ones
        leave, as ``Edit`` applies them: one matching nothing is left
        out, as ``str.replace`` and sed leave the text as it is where
        ``Edit`` would refuse the string it did not find.  When no pair
        matches, the rewrite changes nothing and the pairs stand as
        they are, so that the ``Edit`` reports the first as not found;
        not over a CRLF file when such a pair holds a bare ``\\n``:
        ``Edit`` retries it with ``\\r\\n``, which ``str.replace`` and
        sed never do, and no ``Edit`` is exact.  Nor when the first
        pair replaces one match the text holds whitespace aside:
        ``Edit`` retries it with runs of blanks and trailing whitespace
        ignored (:func:`~kiss.agents.sorcar.edit_matching.tolerant_matches`),
        and would change what sed left alone.

        Args:
            content: The text of the file the rewrite targets.

        Returns:
            The pairs to apply, or ``None`` when no ``Edit`` is exact.
        """
        if self.universal_newlines and "\r" in content:
            return None
        pairs = self.edits
        if self.per_line:
            old, new, _ = pairs[0]
            counts = [record.count(old) for record in content.split(self.separator)]
            if max(counts) > 1:
                return None
            pairs = [(old, new, sum(counts) > 1)]
        exact: list[tuple[str, str, bool]] = []
        text = content
        for old, new, every in pairs:
            if old not in text:
                continue
            exact.append((old, new, every))
            text = text.replace(old, new) if every else text.replace(old, new, 1)
        if exact:
            return exact
        if "\r\n" in content and any("\n" in old and "\r\n" not in old for old, _, _ in pairs):
            return None
        old, _, every = pairs[0]
        if not every and tolerant_matches(content, old):
            return None
        return pairs


_TEXT_KEYWORDS = frozenset({"encoding", "errors", "newline"})
# ``Path.write_text`` took ``newline`` in Python 3.10 and ``read_text`` in
# 3.13: an environment's python may reject it, so the pattern does not.
_PATH_TEXT_KEYWORDS = frozenset({"encoding", "errors"})
_READ_MODES = frozenset({"r", "rt"})
_WRITE_MODES = frozenset({"w", "wt"})


def _is_utf8(encoding: str) -> bool:
    """Whether the codec name *encoding* (``utf8``, ``UTF-8``...) is UTF-8.

    A name python looks up no codec by is not: an unknown one
    (``LookupError``), one holding a null character or a lone surrogate
    (``ValueError``).
    """
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except (LookupError, ValueError):
        return False


def _exact_text_io(node: ast.expr) -> bool:
    """Whether the keywords of the text I/O call *node* keep the bytes as ``Edit`` does.

    ``Edit`` reads and writes UTF-8 with strict errors and no newline
    translation, so ``encoding`` may only name UTF-8, ``errors`` be
    ``"strict"`` and ``newline`` be ``""`` or ``"\\n"``; a keyword left
    out, or ``None``, is the default, which agrees on a UTF-8 file with
    ``\\n`` line endings.  Any other value, or one that is no constant,
    changes the bytes read or written.
    """
    keywords = node.keywords if isinstance(node, ast.Call) else []
    for keyword in keywords:
        if not isinstance(keyword.value, ast.Constant):
            return False
        value = keyword.value.value
        if value is None:
            continue
        if keyword.arg == "encoding":
            exact = isinstance(value, str) and _is_utf8(value)
        elif keyword.arg == "errors":
            exact = value == "strict"
        else:  # newline: the only other keyword _open_call / _method_call let through
            exact = value in ("", "\n")
        if not exact:
            return False
    return True


def _universal_newlines(node: ast.expr) -> bool:
    """Whether the text read *node* (one :func:`_exact_text_io` accepted) translates line endings.

    A ``newline`` keyword left out, or ``None``, is universal newlines:
    a ``\\r\\n`` or ``\\r`` in the file is read as ``\\n``, so the text
    the replacements see is the file's only while it holds no ``\\r``.
    """
    keywords = node.keywords if isinstance(node, ast.Call) else []
    return not any(
        keyword.arg == "newline"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is not None
        for keyword in keywords
    )


def _method_call(
    node: ast.expr, name: str, args: int, keywords: frozenset[str] = frozenset(),
) -> tuple[ast.expr, list[ast.expr]] | None:
    """Return the receiver and positionals of ``<receiver>.<name>(...)`` in *node*.

    Only a call with *args* positionals and keywords among *keywords* matches.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and len(node.args) == args
        and all(keyword.arg in keywords for keyword in node.keywords)
    ):
        return node.func.value, node.args
    return None


def _open_call(node: ast.expr, modes: frozenset[str]) -> ast.expr | None:
    """Return the path argument of ``open(path[, mode][, encoding=...])`` for a mode in *modes*.

    A missing mode is a read (``"r"``).
    """
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "open"
        or not 1 <= len(node.args) <= 2
        or not all(keyword.arg in _TEXT_KEYWORDS for keyword in node.keywords)
    ):
        return None
    if len(node.args) == 1:
        return node.args[0] if "r" in modes else None
    mode = node.args[1]
    if isinstance(mode, ast.Constant) and mode.value in modes:
        return node.args[0]
    return None


_INERT_MODULES = frozenset({
    "collections", "datetime", "functools", "io", "itertools", "json", "math", "os", "os.path",
    "pathlib", "re", "shutil", "string", "subprocess", "sys", "textwrap", "time", "typing",
})
"""Modules every python has whose import does nothing a rewrite could show.

The ones a rewrite program plausibly imports: an ``import`` of any other
module may fail or act before the file is written (:meth:`_PythonRewrite.imports`).
The standard library's, that is: a file of that name where the program runs
is another module (the ``shadowed`` oracle of :func:`mechanical_edit`).
"""


def _imported(program: str) -> set[str]:
    """Return the top-level names of the modules the python *program* imports."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(program)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.partition(".")[0])
    return modules


# Run by a python started with ``-I -S`` (its own library: nothing of the directory, the
# environment or ``site`` is on its path) with the module names: imports them and prints
# ``name origin`` for every module that loaded, the ones they import and the submodules of
# their packages included (``__main__`` has no spec).
_LIBRARY_ORIGINS = (
    "import sys\n"
    "for name in sys.argv[1:]:\n"
    "    __import__(name)\n"
    "specs = [getattr(module, '__spec__', None) for module in list(sys.modules.values())]\n"
    "lines = {'%s %s' % (spec.name, spec.origin) for spec in specs if spec is not None}\n"
    "print('\\n'.join(sorted(lines)))\n"
)
# Run by the python a command runs, as the command runs it, with those lines: prints the
# same lines for where it would load each module from, asking its finders as an import
# does (a submodule with the search path of its package); no module runs.
_ORIGINS_HERE = (
    "import sys\n"
    "def spec(name):\n"
    "    if name in sys.modules:\n"
    "        return sys.modules[name].__spec__\n"
    "    package = name.rpartition('.')[0]\n"
    "    path = spec(package).submodule_search_locations if package else None\n"
    "    for finder in sys.meta_path:\n"
    "        found = finder.find_spec(name, path)\n"
    "        if found is not None:\n"
    "            return found\n"
    "for line in sys.argv[1].splitlines():\n"
    "    name = line.split()[0]\n"
    "    print(name, spec(name).origin)\n"
)


def shadow_command(program: str, path: str, modules: set[str]) -> str:
    """Return the shell command telling whether python, run in *path*, shadows one of *modules*.

    The interpreter is asked, since only it knows its path: the
    directory a program read from stdin runs in, the ``PYTHONPATH``
    entries as it normalises them, an archive among them, and whatever
    its own startup puts first (an executable line of a ``.pth``, a
    ``sitecustomize``).  *program* (a bare name or an absolute path, as
    the command names it) says first, isolated from all of those
    (:data:`_LIBRARY_ORIGINS`), where its library holds the modules and
    every module they import in turn (``json`` imports ``enum`` through
    ``re``, and its own ``json.decoder``: an ``enum.py`` in the directory,
    or a finder that takes the submodule, is what that runs), then, run
    in *path* as the command runs it (:data:`_ORIGINS_HERE`), where it
    would load each of them from.  The status is 1 when the answers
    are the same (nothing shadows), or *path* is no directory, which the
    command's ``cd`` fails on; any other status stands for a shadow:
    another origin, or a python that gave no answer.
    """
    python = shlex.quote(program)
    names = " ".join(shlex.quote(module) for module in sorted(modules))
    return (
        f"cd -P -- {shlex.quote(path)} 2>/dev/null || exit 1; "
        f"library=$({python} -I -S -c {shlex.quote(_LIBRARY_ORIGINS)} {names}) && "
        f'here=$({python} -c {shlex.quote(_ORIGINS_HERE)} "$library") && '
        '[ -n "$here" ] && [ "$here" = "$library" ] && exit 1; exit 0'
    )


class _PythonRewrite:
    """Reads a python program statement by statement for the read-replace-write pattern.

    Attributes:
        paths: Names assigned one string (or ``Path(<string>)``) so far.
        objects: Those of them assigned a ``Path(...)``, which alone
            have ``read_text`` / ``write_text``.
        pathlib: Whether ``from pathlib import Path`` ran: a ``Path``
            called before it raises ``NameError``.
        read: The literal path read; ``written``: the one written back.
        named: The first literal path a statement outside the pattern
            opens, for naming the file in a refusal.
        text: The name holding the text read, after its replacements.
        edits: The ``(old, new, replace_all)`` of the ``.replace`` calls.
        exact: Whether every statement so far fits the pattern.
        universal_newlines: Whether the read translates line endings
            (:func:`_universal_newlines`).
    """

    def __init__(self) -> None:
        self.paths: dict[str, str] = {}
        self.objects: set[str] = set()
        self.pathlib = False
        self.read: str | None = None
        self.written: str | None = None
        self.named: str | None = None
        self.text: str | None = None
        self.edits: list[tuple[str, str, bool]] = []
        self.exact = True
        self.universal_newlines = False

    def path(self, node: ast.expr, text_io: bool = False) -> str | None:
        """Return the path *node* names: a string, ``Path(<string>)`` or a name assigned one.

        With *text_io* only one that has ``read_text`` / ``write_text``:
        a ``Path(...)`` or a name assigned one (:attr:`objects`), not a
        string.  A ``Path`` called before its import raises: the path is
        named all the same, the exactness ends.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return None if text_io else node.value
        if isinstance(node, ast.Name):
            return None if text_io and node.id not in self.objects else self.paths.get(node.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and not node.keywords
        ):
            self.exact = self.exact and self.pathlib
            return self.path(node.args[0])
        return None

    def statement(self, node: ast.stmt) -> None:
        """Account for one top-level statement; a shape outside the pattern ends the exactness.

        So does one binding ``open`` or ``Path`` (``open = ...``, ``with
        ... as Path:``): they are then not the calls the pattern reads.
        """
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.imports(node)
            return
        fits = False
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            fits = self.assignment(node.targets[0].id, node.value)
        elif isinstance(node, ast.Expr):
            fits = self.write(node.value)
        elif isinstance(node, ast.With):
            fits = self.with_open(node)
        if not fits or _bound_names(node).keys() & {"open", "Path"}:
            self.exact = False
            self.note_paths(node)

    def imports(self, node: ast.Import | ast.ImportFrom) -> None:
        """Account for an import: one that may fail or act, or rebinds a name read, is not exact.

        Python runs an import before the rewrite, and one that fails
        ends the program with the file as it was.  Part of the pattern
        are only the import of a module every python has and that does
        nothing when imported (:data:`_INERT_MODULES`) and ``from
        pathlib import Path``: any other module may be missing or act,
        any other ``from`` import may name what its module lacks (a
        ``*`` binds anything), and a relative one fails in a program
        read from stdin, which has no parent package.  A path or text
        name bound to a module (``import os as p``) no longer holds
        what it was read as, and an ``open`` or a ``Path`` bound to a
        module is not the call the pattern reads.  ``pathlib``'s
        ``Path`` is noted as imported (:attr:`pathlib`).
        """
        if isinstance(node, ast.ImportFrom):
            pathlib = not node.level and node.module == "pathlib"
            inert = pathlib and all(alias.name == "Path" for alias in node.names)
        else:
            pathlib = False
            inert = all(alias.name in _INERT_MODULES for alias in node.names)
        if not inert:
            self.exact = False
        for alias in node.names:
            name = (alias.asname or alias.name).partition(".")[0]
            if (
                name in self.paths or name == self.text or name == "open"
                or (name == "Path" and not pathlib)
            ):
                self.exact = False
            elif name == "Path":
                self.pathlib = True

    def assignment(self, name: str, value: ast.expr) -> bool:
        """Account for ``name = value``: a path, the text read, or its replacements.

        A name keeps one role: the text name assigned a path, or a path
        name assigned the text, a read or a replacement, is a program
        outside the pattern (the old binding would be read as standing).
        """
        literal = self.path(value)
        if literal is not None:
            if name in self.paths or name == self.text:
                return False
            self.paths[name] = literal
            if self.path(value, True) is not None:
                self.objects.add(name)
            return True
        if name in self.paths:
            return False
        read = self.read_path(value)
        if read is not None:
            if self.read is not None:
                return False
            self.read, self.text = read, name
            return True
        edits = self.replacements(value)
        # A replacement after the write never reaches the file.
        if edits is not None and self.written is None:
            self.edits.extend(edits)
            self.text = name
            return True
        return False

    def read_path(self, node: ast.expr) -> str | None:
        """Return the path of ``open(p).read()``, ``Path(p).read_text()`` or ``p.read_text()``.

        The call must read the bytes as ``Edit`` does (:func:`_exact_text_io`);
        :attr:`universal_newlines` records whether it translates line endings.
        ``read_text`` (the call is then *node* itself) needs a ``Path``.
        """
        read = _method_call(node, "read", 0)
        if read is not None:
            path, call = _open_call(read[0], _READ_MODES), read[0]
        else:
            read = _method_call(node, "read_text", 0, _PATH_TEXT_KEYWORDS)
            path, call = (None if read is None else read[0]), node
        if path is None or not _exact_text_io(call):
            return None
        self.universal_newlines = _universal_newlines(call)
        return self.path(path, call is node)

    def replacements(self, node: ast.expr) -> list[tuple[str, str, bool]] | None:
        """Return the pairs of a ``text.replace(old, new)...`` chain on the text read, in order.

        ``None`` when *node* is no such chain, or one no ``Edit`` is
        exact for: ``replace("", x)`` inserts *x* between every
        character.  A ``replace(x, x)`` changes nothing and gives no
        pair (``Edit`` refuses equal strings), so a chain of such calls
        alone gives an empty list.
        """
        edits: list[tuple[str, str, bool]] = []
        links = 0
        while True:
            replace = _method_call(node, "replace", 2)
            if replace is None:
                break
            links += 1
            node, (old, new) = replace
            if not (
                isinstance(old, ast.Constant) and isinstance(old.value, str)
                and isinstance(new, ast.Constant) and isinstance(new.value, str)
            ) or old.value == "":
                return None
            # str.replace replaces every occurrence: so must the Edit.
            if old.value != new.value:
                edits.append((old.value, new.value, True))
        if links and isinstance(node, ast.Name) and node.id == self.text and self.text is not None:
            return edits[::-1]
        return None

    def write(self, node: ast.expr) -> bool:
        """Account for ``open(p, "w").write(s)``, ``Path(p).write_text(s)``, ``p.write_text(s)``.

        The call must write the bytes as ``Edit`` does (:func:`_exact_text_io`);
        ``write_text`` (the call is then *node* itself) needs a ``Path``.
        """
        write = _method_call(node, "write", 1)
        if write is not None:
            path, call = _open_call(write[0], _WRITE_MODES), write[0]
        else:
            write = _method_call(node, "write_text", 1, _PATH_TEXT_KEYWORDS)
            path, call = (None if write is None else write[0]), node
        if write is None or path is None or not _exact_text_io(call):
            return False
        return self.written_to(self.path(path, call is node), write[1][0])

    def written_to(self, path: str | None, value: ast.expr) -> bool:
        """Record the write of *value* (the text read) to *path*; only one write fits."""
        if (
            path is None
            or self.written is not None
            or not isinstance(value, ast.Name)
            or value.id != self.text
        ):
            return False
        self.written = path
        return True

    def with_open(self, node: ast.With) -> bool:
        """Account for ``with open(p[, "w"]) as f:`` holding one ``s = f.read()``/``f.write(s)``."""
        if len(node.items) != 1 or len(node.body) != 1:
            return False
        item, body = node.items[0], node.body[0]
        if not isinstance(item.optional_vars, ast.Name) or not _exact_text_io(item.context_expr):
            return False
        handle = item.optional_vars.id
        # The handle is a file object: a path name or the text name bound to
        # it no longer holds what the pattern reads it as.
        if handle in self.paths or handle == self.text:
            return False
        read = _open_call(item.context_expr, _READ_MODES)
        if read is not None and isinstance(body, ast.Assign):
            call = _method_call(body.value, "read", 0)
            if (
                call is None or self.read is not None
                or len(body.targets) != 1 or not isinstance(body.targets[0], ast.Name)
                or body.targets[0].id in self.paths
                or not isinstance(call[0], ast.Name) or call[0].id != handle
            ):
                return False
            self.read = self.path(read)
            self.text = body.targets[0].id
            self.universal_newlines = _universal_newlines(item.context_expr)
            return self.read is not None
        written = _open_call(item.context_expr, _WRITE_MODES)
        if written is not None and isinstance(body, ast.Expr):
            call = _method_call(body.value, "write", 1)
            if call is None or not isinstance(call[0], ast.Name) or call[0].id != handle:
                return False
            return self.written_to(self.path(written), call[1][0])
        return False

    def note_paths(self, node: ast.stmt) -> None:
        """Record the first file a statement outside the pattern opens, to name it in a refusal."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            receiver = _open_call(child, _READ_MODES | _WRITE_MODES)
            for name, args in (("read_text", 0), ("write_text", 1)):
                call = _method_call(child, name, args, _TEXT_KEYWORDS)
                receiver = receiver if call is None else call[0]
            literal = None if receiver is None else self.path(receiver)
            if literal is not None and self.named is None:
                self.named = literal


def python_edit(program: str) -> tuple[str | None, list[tuple[str, str, bool]], bool]:
    """Return the path a python rewrite *program* edits, its ``Edit`` pairs and its newline mode.

    The pairs are empty unless the program is exactly: read one file
    (``open(p).read()``, ``Path(p).read_text()``, ``with open(p) as
    f: s = f.read()``), apply ``.replace(<str>, <str>)`` to the text
    one or more times, and write it back to the same file
    (``open(p, "w").write(s)``, ``Path(p).write_text(s)``, ``with
    open(p, "w") as f: f.write(s)``), *p* a string literal, possibly
    through a name assigned once; an import binding no name the
    pattern reads (:meth:`_PythonRewrite.imports`) is allowed.  The
    I/O calls may only carry the ``encoding``/``errors``/``newline``
    keywords that keep the bytes as ``Edit`` does
    (:func:`_exact_text_io`), and a replacement after the write is no
    part of it.  The path is still returned, when one is found, for a
    program outside the pattern (``re.sub``, a loop, two files).  The
    last value says whether the read translates line endings
    (:func:`_universal_newlines`): the pairs are then exact only for a
    file holding no ``\\r`` (:meth:`MechanicalEdit.edits_for`).

    A program python itself would not run as far as its write is
    outside the pattern too: one the parser gives up on
    (:data:`_UNREADABLE`), one that parses but does not compile (a
    repeated keyword, a late ``__future__`` import), a relative import,
    a string used as a ``Path``, a ``Path`` not imported yet, ``open``
    or ``Path`` bound to something else, a path holding a null
    character.

    Args:
        program: The python source.
    """
    try:
        tree = ast.parse(program)
        compile(tree, "<stdin>", "exec")
    except _UNREADABLE:
        return None, [], False
    rewrite = _PythonRewrite()
    for statement in tree.body:
        rewrite.statement(statement)
    path = rewrite.read or rewrite.written or rewrite.named
    if (
        rewrite.exact
        and rewrite.edits
        and rewrite.read is not None
        and rewrite.read == rewrite.written
        and "\0" not in rewrite.read
    ):
        return path, rewrite.edits, rewrite.universal_newlines
    return path, [], False


def python_written_paths(program: str) -> list[str] | None:
    """Return the literal paths of every file the python *program* writes, or ``None``.

    A file is written by ``open(p, <mode>)`` for a mode holding ``w``,
    ``a``, ``x`` or ``+`` (as an argument or the ``mode`` keyword), by
    ``p.open(<such a mode>)``, or by ``p.write_text(...)`` /
    ``p.write_bytes(...)``, anywhere in the program; *p* is a string
    literal, ``Path(<literal>)`` or a name bound exactly once in the
    whole program, by an assignment.  ``None`` when the program does
    not parse, writes no file this reading sees, writes one whose
    path it builds at run time (an f-string, ``sys.argv``, a name bound
    twice: assigned again, or also a parameter, loop target or import),
    or changes its working directory (``os.chdir``, ``os.fchdir``),
    after which a relative path names a file in a directory the caller
    resolving the paths does not know: it may then write any file, and
    a guard must assume the worst.

    Args:
        program: The python source.
    """
    try:
        tree = ast.parse(program)
    except _UNREADABLE:
        return None
    stores = _bound_names(tree)
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and stores[node.targets[0].id] == 1
        ):
            literal = _literal_path(node.value, {})
            if literal is not None:
                names[node.targets[0].id] = literal
    written: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node) in _CHDIR:
            return None
        target = _write_target(node)
        if target is None:
            continue
        literal = _literal_path(target, names) if target is not _UNKNOWN else None
        if literal is None:
            return None
        written.append(literal)
    return written or None


#: What python raises for a source it cannot parse or compile: a syntax error, a null
#: byte, or nesting past the parser's limits (a chain of tens of thousands of calls).
_UNREADABLE = (SyntaxError, ValueError, RecursionError, MemoryError)

#: A write whose target or mode is not a literal: it may name any file.
_UNKNOWN = ast.Name("?")

#: The calls that change the working directory (``os.chdir(d)``, a bare ``chdir(d)``).
_CHDIR = frozenset({"chdir", "fchdir"})


def _called_name(call: ast.Call) -> str:
    """Return the name *call* calls by: the ``chdir`` of ``os.chdir(d)`` and of a bare ``chdir(d)``.

    The last name alone, read off the node (``""`` when the target ends in none):
    ``ast.unparse`` of the target recurses once per link of a chain of calls and
    overflows the stack on a long one.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _bound_names(tree: ast.AST) -> Counter[str]:
    """Count how many times each name is bound anywhere in *tree*, in any scope.

    Assignment and loop targets, ``with ... as``, walrus targets,
    function and lambda parameters, ``except ... as``, ``global`` /
    ``nonlocal`` declarations, function and class names, imports and
    the captures of a ``match`` pattern (``case p:``, ``case [*p]:``,
    ``case {**p}:``) all bind a name; a name bound more than once may
    hold a value this reading cannot follow (a parameter shadowing a
    module-level path).
    """
    bound: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound[node.id] += 1
        elif isinstance(node, ast.arg):
            bound[node.arg] += 1
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            bound[node.name] += 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound[node.rest] += 1
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound.update(node.names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound[node.name] += 1
        elif isinstance(node, ast.alias):
            bound[(node.asname or node.name).partition(".")[0]] += 1
    return bound


def _write_target(call: ast.Call) -> ast.expr | None:
    """Return the path expression *call* writes a file at (:data:`_UNKNOWN` when unclear).

    ``open(p, mode)`` / ``p.open(mode)`` for a mode that writes;
    ``p.write_text(...)`` / ``p.write_bytes(...)``.  ``None`` for a call
    that writes no file (a read, another function).
    """
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
        return func.value
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if isinstance(func, ast.Name) and func.id == "open":
        if None in keywords:
            return _UNKNOWN  # ``**kwargs`` may carry the file and the mode
        if not call.args and "file" not in keywords:
            return None  # no file to open
        receiver = call.args[0] if call.args else keywords["file"]
        arguments = call.args[1:]
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        receiver, arguments = func.value, call.args
    else:
        return None
    if None in keywords:
        return _UNKNOWN  # ``**kwargs`` may carry the mode
    mode = keywords.get("mode", arguments[0] if arguments else None)
    if mode is None:
        return None  # a read
    if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
        return _UNKNOWN
    return receiver if set(mode.value) & set("wax+") else None


def _literal_path(node: ast.expr, names: dict[str, str]) -> str | None:
    """Return the path *node* names: a string, ``Path(<string>)`` or a name of *names*.

    ``Path`` is the bare name or ``pathlib.Path``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        func = node.func
        qualified = (
            isinstance(func, ast.Attribute) and func.attr == "Path"
            and isinstance(func.value, ast.Name) and func.value.id == "pathlib"
        )
        if qualified or (isinstance(func, ast.Name) and func.id == "Path"):
            return _literal_path(node.args[0], names)
    return None


_SED_NULL_DATA = frozenset({"-z", "--null-data"})
_SED_EDIT_OPTIONS = frozenset({"-i", "--in-place"}) | _SED_NULL_DATA
_SED_PATTERN_METACHARACTERS = ".*[]^$"


def _sed_parts(script: str, delimiter: str) -> list[str]:
    """Split the sed *script* after its ``s<d>`` on the unescaped *delimiter*."""
    parts: list[str] = []
    start = position = 0
    while position < len(script):
        char = script[position]
        if char == "\\":
            position += 2
        elif char == delimiter:
            parts.append(script[start:position])
            start = position = position + 1
        else:
            position += 1
    parts.append(script[start:])
    return parts


def _sed_text(raw: str, delimiter: str, pattern: bool) -> str | None:
    """Return the literal text of a sed ``s`` *pattern* or replacement, ``None`` when it is none.

    ``\\n``, ``\\t``, an escaped delimiter and ``\\\\`` are unescaped; in
    the pattern so are escaped BRE metacharacters, while a bare one
    makes it a regular expression; in the replacement ``\\&`` is ``&``,
    while a bare ``&`` or a ``\\N`` back-reference is no literal text.
    Nor is the escaped delimiter of a pattern when it is a BRE
    metacharacter: sed unescapes it to the metacharacter
    (``s.a\\.b.c.`` replaces ``aXb``).  Nor is a newline no backslash
    precedes: sed ends the command there, unterminated.
    """
    parts: list[str] = []
    position = 0
    while position < len(raw):
        char = raw[position]
        if char == "\\":
            escaped = raw[position + 1:position + 2]
            if escaped == "n":
                parts.append("\n")
            elif escaped == "t":
                parts.append("\t")
            elif pattern and escaped == delimiter and escaped in _SED_PATTERN_METACHARACTERS:
                return None
            elif escaped in (delimiter, "\\") and escaped:
                parts.append(escaped)
            elif pattern and escaped and escaped in _SED_PATTERN_METACHARACTERS:
                parts.append(escaped)
            elif not pattern and escaped == "&":
                parts.append("&")
            else:
                return None
            position += 2
        elif char == "\n" or (char in _SED_PATTERN_METACHARACTERS if pattern else char == "&"):
            return None
        else:
            parts.append(char)
            position += 1
    return "".join(parts)


def sed_edit(simple: SimpleCommand) -> tuple[Word | None, list[tuple[str, str, bool]]]:
    """Return the file operand of the ``sed`` *simple* and the ``Edit`` pair its script stands for.

    The pair is empty unless the options are ``-i``/``--in-place`` and
    at most ``-z``/``-e`` (an ``-e`` ending the command has no script:
    sed runs nothing), and the one script, a word the shell passes
    on as written (no ``$``, backtick or glob in it), is
    ``s<d>old<d>new<d>`` or ``s<d>old<d>new<d>g`` (then ``replace_all``)
    whose *old* and *new* are literal text (:func:`_sed_text`) and whose
    delimiter is one byte, as sed requires (``s§a§b§`` is an error).  On
    macOS, the separate empty backup suffix of ``sed -i ''`` is consumed as
    an option value, not mistaken for the script.  Without
    ``-z`` sed matches one line at a time, so *old* may not hold a
    newline (it would never match), and without ``g`` it replaces the
    first match of every line: the pair replaces one match, exact only
    while the file holds one (:meth:`MechanicalEdit.edits_for`).  The
    operand is ``None`` unless exactly one word is neither an option
    nor the script.

    Args:
        simple: A simple command whose program is ``sed``.
    """
    options: list[str] = []
    scripts: list[str] = []
    operands: list[Word] = []
    literal = True
    value_next = False
    bsd_in_place_suffix = False
    for word in simple.argv()[1:]:
        text = word.text
        if bsd_in_place_suffix:
            bsd_in_place_suffix = False
            if not text and word.literal:
                continue
        if value_next:
            scripts.append(text)
            literal = literal and word.literal
            value_next = False
        elif text in ("-e", "--expression"):
            value_next = True
        elif text.startswith("--expression="):
            scripts.append(text[len("--expression="):])
            literal = literal and word.literal
        elif text.startswith("-e"):
            scripts.append(text[2:])
            literal = literal and word.literal
        elif text.startswith("-") and text != "-":
            options.append(text)
            bsd_in_place_suffix = sys.platform == "darwin" and text == "-i"
        else:
            operands.append(word)
    if not scripts and operands:
        script_word = operands.pop(0)
        scripts.append(script_word.text)
        literal = script_word.literal
    operand = operands[0] if len(operands) == 1 else None
    in_place = {"-i", "--in-place"} & set(options)
    if (
        operand is None
        or value_next
        or len(scripts) != 1
        or not literal
        or not in_place
        or not set(options) <= _SED_EDIT_OPTIONS
    ):
        return operand, []
    script = scripts[0]
    if (
        len(script) < 2
        or script[0] != "s"
        or script[1] in "\\\n"
        or script[1].isalnum()
        or not script[1].isascii()
    ):
        return operand, []
    parts = _sed_parts(script[2:], script[1])
    if len(parts) != 3 or parts[2] not in ("", "g"):
        return operand, []
    old = _sed_text(parts[0], script[1], pattern=True)
    new = _sed_text(parts[1], script[1], pattern=False)
    if not old or new is None or old == new:
        return operand, []
    if "\n" in old and not _SED_NULL_DATA & set(options):
        return operand, []
    return operand, [(old, new, parts[2] == "g")]


def _program(word: Word, cwd: str | None) -> str | None:
    """The program *word* names in *cwd*, as the oracles of :func:`mechanical_edit` are asked.

    A bare name as written (bash looks it up on ``PATH``) and a path
    made absolute; ``None`` for a relative path in an unknown directory.
    """
    return word.text if "/" not in word.text else resolve(cwd, word)


def _found(program: str | None, runs: Callable[[str], bool] | None) -> bool:
    """Whether bash finds *program* (:func:`_program`), as *runs* tells; without it, always."""
    return runs is None or (program is not None and runs(program))


def mechanical_edit(
    command: str, cwd: str | None, shadowed: Callable[[str, str, set[str]], bool] | None = None,
    runs: Callable[[str], bool] | None = None,
) -> MechanicalEdit | None:
    """Return the ``Edit`` a python heredoc rewrite or a ``sed -i`` *command* stands for.

    The rewrite is one simple command (no wrapper, no assignment, no
    output redirection), the whole command or the rewrite step of
    ``[cd DIR &&] <rewrite> [&& | ; | newline] <rest>``
    (:func:`_rewrite_segment`; the result then names the ``cd``
    directory and the rest).  ``None`` when *command* is not of that
    shape, the rewrite is not mechanical (:func:`python_edit`,
    :func:`sed_edit`) or its file can only name a directory
    (:func:`_names_a_directory`).

    Args:
        command: The shell command text.
        cwd: The absolute POSIX directory it starts in; ``None`` when unknown.
        shadowed: Whether the python given (a program as *runs* gets
            it), run in the absolute bash directory given, loads one of
            the modules named, or one they import, from elsewhere than
            its own library (``json.py`` in the directory a program read
            from stdin runs in is what ``import json`` runs, and it may
            fail or act; :func:`shadow_command` asks the interpreter).
            With it a python rewrite importing such a module, or run in
            an unknown directory, is not mechanical; without it every
            import is taken for the standard library's.
        runs: Whether bash finds the program given: a bare name on the
            ``PATH`` of the command's environment, or an absolute bash
            path.  With it a rewrite whose program is not found is not
            mechanical (bash fails with status 127 and leaves the file
            alone); without it the program is taken to be there.
    """
    rewrite = _rewrite_segment(command, cwd)
    if rewrite is None:
        return None
    simple = rewrite.simple
    argv = simple.argv()
    if simple.words != argv or simple.outputs:
        return None
    program = simple.program()
    words = [word.text for word in argv[1:]]
    if program in ("python", "python3") and rewrite.body is not None and words == ["-"]:
        path, edits, universal_newlines = python_edit(rewrite.body)
        python = _program(argv[0], simple.cwd)
        if path is None or not edits or _names_a_directory(path) or not _found(python, runs):
            return None
        if shadowed is not None and (modules := _imported(rewrite.body)) and (
                python is None or simple.cwd is None or shadowed(python, simple.cwd, modules)):
            return None
        return MechanicalEdit(
            "python heredoc", resolve(simple.cwd, Word(path, True)), path, edits,
            rewrite.cwd, rewrite.remainder, universal_newlines=universal_newlines,
        )
    if program == "sed" and rewrite.body is None:
        operand, edits = sed_edit(simple)
        if (operand is None or not edits or _names_a_directory(operand.text)
                or not _found(_program(argv[0], simple.cwd), runs)):
            return None
        # -z makes NUL the record separator; without g the first match of
        # every record goes, whatever separates them.
        separator = "\0" if _SED_NULL_DATA & set(words) else "\n"
        return MechanicalEdit(
            "`sed -i` call", resolve(simple.cwd, operand), operand.text, edits,
            rewrite.cwd, rewrite.remainder, not edits[0][2], separator=separator,
        )
    return None


# The commands that copy their input (a here-document) to their outputs verbatim.
_COPIERS = frozenset({"cat", "tee"})
# File names that are a process's own descriptors, not files of their own.
_DESCRIPTOR_ALIASES = ("/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/tty", "/dev/fd/",
                       "/proc/self/fd/", "/proc/thread-self/fd/")
# The redirections that open their operand as a file on the descriptor they name.
_FILE_OPENERS = frozenset({"<", "<>", ">", ">>", ">|", "&>", "&>>"})
HEREDOC_VERDICT = (
    "REFUSED: the command did not run: it writes the existing file {path} through a "
    "here-document (an append counts). Use Edit(file_path=\"{path}\", old_string=<the exact "
    "current text>, new_string=<the replacement>) for a change (Read the region first) or "
    "Write(file_path=\"{path}\", content=<the here-document body>) for the whole file: they "
    "are logged and reviewable, a heredoc is not. A here-document may create a new file."
)
"""The refusal :func:`heredoc_guard` returns for the existing target ``{path}``."""
HEREDOC_UNRESOLVED_VERDICT = (
    "REFUSED: the command did not run: it writes {path} through a here-document, and only "
    "the shell knows which file that is, so it may be an existing one. Use "
    "Write(file_path=<the literal path {path} names>, content=<the here-document body>), "
    "or Edit(file_path=<that path>, old_string=<the exact current text>, new_string=<the "
    "replacement>) for a change to an existing file. A here-document may create a new file "
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
    Piped into a group, or into a command that runs literal text (its
    ``script``: a shell's ``-c``, ``eval``), it reaches what the commands
    in it that read its input write (*commands* holds them all).  With
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
        into = node.reader.script if isinstance(node.reader, SimpleCommand) else node.reader
        if isinstance(into, Group):
            for member in _input_readers(into, commands):
                writes.extend(_output_files(member, copies_only, commands))
            break
        node = receiver(node)
    return writes


def _names_a_file(cwd: str | None, word: Word) -> bool:
    """Return whether *word* names a file of its own in the directory *cwd*.

    The path must be one the scan resolves (:func:`resolve`: literal,
    normalised, in a known directory) and not a descriptor alias:
    ``/dev/stdout``, ``/dev//fd/3`` and the like are the command's own
    descriptors under a file name, a stream, not a sink.
    """
    path = resolve(cwd, word)
    return path is not None and not ("/" + path.lstrip("/")).startswith(_DESCRIPTOR_ALIASES)


def _stdout_sink(node: SimpleCommand | Group) -> Word | str:
    """Return where the standard output of *node* ends once its redirections are applied.

    The redirections are replayed in order over the descriptors they
    name (the operator's default when unnumbered: ``0`` for ``<``,
    ``1`` for ``>``), so a duplication (``2>&1``, ``1>&3``, ``1<&3``)
    copies what its source holds then.  A literal file :class:`Word` is
    a sink the scan can name; ``"stream"`` is the pipe or standard
    output the shell gave the command (where ``3>&1 >/dev/null 1>&3``
    ends again); ``"unknown"`` is anything else: a descriptor alias such
    as ``/dev/stdout``, a word only the shell knows, a closed descriptor,
    one nothing opened, a here-document or a here-string, and both
    standard descriptors after a ``01<&3``, ``11>x`` or ``{name}>x``
    (bash's descriptor 1, 11 or a new one; sh's default one).
    """
    table: dict[str, Word | str] = {"1": "stream", "2": "stream"}
    for fd, operator, operand in node.redirections:
        value: Word | str = "unknown"
        targets: tuple[str, ...] = ("1", "2")
        if len(fd) <= 1:
            if operator in ("<&", ">&") and (operand.text.isdigit() or operand.text == "-"):
                value = table.get(operand.text, "unknown")
            elif operator in _FILE_OPENERS and _names_a_file(node.cwd, operand):
                value = operand
            default = "0" if operator.startswith("<") else "1"
            targets = ("1", "2") if operator in _BOTH_OPERATORS else (fd or default,)
        for target in targets:
            table[target] = value
    return table["1"]


def _copies_to_files(reader: SimpleCommand, commands: list[SimpleCommand]) -> bool:
    """Return whether *reader* copies its input into files the scan can name, and nowhere else.

    Its output, followed as :func:`_output_files` follows it, must end
    in files (:func:`_stdout_sink`; ``tee``'s operands too); a shell
    or interpreter it reaches runs the text, a process substitution
    (``>(bash)``) or a word only the shell knows may be one, a
    substitution hands it to another command as a word, and the
    command's own standard output may be sent anywhere (``exec >``).
    """
    node: SimpleCommand | Group | None = reader
    while node is not None:
        if isinstance(node, SimpleCommand):
            program = node.program()
            if program not in _COPIERS:
                return False
            if program == "tee" and not all(
                    _names_a_file(node.cwd, word)
                    for word in node.argv()[1:] if word.text[:1] != "-"):
                return False
        sink = _stdout_sink(node)
        if sink != "stream":
            return isinstance(sink, Word)
        if isinstance(node.reader, Group):
            return all(_copies_to_files(member, commands)
                       for member in _input_readers(node.reader, commands))
        node = receiver(node)
    return False


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
    isdir: Callable[[str], bool] | None = None,
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
        isdir: Whether a path is a directory where the command runs
            (:func:`heredoc_guard`).
    """
    writes: list[tuple[str | None, Word]] = []
    commands = simple_commands(command, cwd, isdir=isdir)
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


def heredoc_guard(
    command: str, cwd: str | None, guarded: Callable[[str], bool],
    isdir: Callable[[str], bool] | None = None,
) -> str | None:
    """Refuse a ``cat``/``tee`` here-document copy into a guarded file, or one that may be.

    A copy into a file that does not exist yet creates it and may run.
    A copy into an existing file, an append (``>>``, ``tee -a``)
    included, is a rewrite for ``Edit``/``Write``: it skips the
    read-before-modify rule and leaves no reviewable change in the tool
    log.  A target holding an expansion the command did not assign
    itself (``"$OUT"``, ``~/x``; ``T=tmp/x && cat > $T/f`` names
    ``tmp/x/f``), or a relative one after a ``cd`` only the shell can
    resolve, cannot be checked and is refused too.  ``/dev/`` targets
    are not files.

    Args:
        command: The shell command as the model wrote it.
        cwd: The absolute POSIX directory the command starts in;
            ``None`` when unknown (every relative target is refused).
        guarded: Whether a rewrite of the absolute bash path given is
            refused where the command runs (the host, a container or an
            exec bridge's environment): the file exists, and is not a
            scratch file where the caller exempts those.
        isdir: Whether an absolute bash path is a directory there, its
            links followed.  With it a ``cd link/../deep`` that bash
            cannot walk as written, and so takes through the link,
            leaves a directory only the shell can resolve
            (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`);
            without it the directory is the one the text names.

    Returns:
        :data:`HEREDOC_VERDICT` or :data:`HEREDOC_UNRESOLVED_VERDICT`, or
        ``None`` when the command may run.
    """
    for directory, target in _heredoc_writes(command, cwd, copies_only=True, isdir=isdir):
        path = resolve(directory, target)
        if path is None:
            return HEREDOC_UNRESOLVED_VERDICT.format(path=target.text)
        if not path.startswith("/dev/") and guarded(path):
            return HEREDOC_VERDICT.format(path=path)
    return None


def numbered_region(content: str, start_line: int, end_line: int) -> str:
    """Return lines *start_line*..*end_line* (1-based, clipped) of *content* with line numbers.

    A line ends at ``\\n`` only, as :func:`line_of` counts them and as Read,
    ``sed -n`` and ``grep -n`` number them (``str.splitlines`` would also end
    one at a form feed, a lone ``\\r`` and the Unicode separators, and move
    every number after it); a ``\\r\\n`` is shown as ``\\n``, as Read shows it.
    """
    lines = content.replace("\r\n", "\n").split("\n")
    if not lines[-1]:  # no text after the last newline, or none at all: not a line
        del lines[-1]
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
    stripped) are shown; when nothing matches, the file's head.  The
    needle's lines end at ``\\n`` only, as the content's do
    (:func:`numbered_region`).

    Args:
        content: The file text.
        needle: The ``old_string`` the model wanted to replace.
        context: Lines shown on each side.
    """
    first_line = next((ln.strip() for ln in needle.split("\n") if ln.strip()), "")
    position = content.find(needle)
    if position < 0 and first_line:
        position = content.find(first_line)
    if position < 0:
        return numbered_region(content, 1, 2 * context)
    anchor = line_of(content, position)
    needle_lines = needle.rstrip("\n").count("\n") + 1
    return numbered_region(content, anchor - context, anchor + needle_lines - 1 + context)
