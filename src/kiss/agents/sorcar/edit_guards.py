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
* :func:`numbered_region` renders the lines around a string so an
  ``Edit`` refusal, or an ``Edit`` result, carries the context the
  next call needs.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

VIEW_COMMANDS = frozenset({"cat", "sed", "grep", "nl", "head", "tail"})
"""Commands whose single file operand is shown to the model."""
# Commands that take a non-file first operand (sed's script, grep's pattern).
_FIRST_OPERAND_IS_NOT_A_FILE = frozenset({"sed", "grep"})
_SEGMENT_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_SED_INPLACE = re.compile(r"(?:^|\s)sed\s+(?:-[a-zA-Z]*i|--in-place)")
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
    """Return the main checkout a worktree's ``.git`` file points into, or ``None``."""
    try:
        text = git_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:"):].strip())
    for parent in (gitdir, *gitdir.parents):
        if parent.name == ".git":
            return parent.parent
    return None


def shell_view_target(command: str, cwd: str | None) -> Path | None:
    """Return the one file *command* shows, or ``None``.

    The command is read segment by segment (``&&``, ``;``, ``|``);
    ``cd`` segments move the working directory, and the last segment
    that is a view command (:data:`VIEW_COMMANDS`) with exactly one
    existing file operand names the file.  Options are skipped, as is
    the script of ``sed`` and the pattern of ``grep``.

    Args:
        command: The shell command as the model wrote it.
        cwd: The directory relative operands resolve against.
    """
    directory = Path(cwd) if cwd else Path.cwd()
    target: Path | None = None
    for segment in _SEGMENT_SPLIT.split(command.strip()):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return None
        if not tokens:
            continue
        if tokens[0] == "cd" and len(tokens) == 2:
            directory = (directory / tokens[1]).resolve()
            continue
        if tokens[0] not in VIEW_COMMANDS:
            continue
        operands = _operands(tokens)
        if len(operands) != 1:
            continue
        candidate = (directory / operands[0]).resolve()
        if candidate.is_file():
            target = candidate
    return target


_LINE_PREFIX = re.compile(r"^\s*\d+\s*[:\t-]\s?")
"""The ``12:`` / ``12-`` of ``grep -n`` and the ``    12\\t`` of ``nl``."""
_SHOWN_LINES_CHECKED = 200
_SHOWN_FILE_CHARS = 4_000_000


def output_shows_file(output: str, path: Path) -> bool:
    """Return whether *output* contains at least one line of the file at *path*.

    What ``_note_shell_view`` needs before it counts a shell view as a
    read: ``echo label; true || cat a.py`` names ``a.py`` but shows
    nothing of it.  The first :data:`_SHOWN_LINES_CHECKED` non-blank
    output lines are looked up in the file's text, with the line-number
    prefix of ``grep -n`` / ``nl`` stripped; one-character lines are
    ignored.

    Args:
        output: What the command printed.
        path: The file the command named.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:_SHOWN_FILE_CHARS]
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


def _operands(tokens: list[str]) -> list[str]:
    """Return the file operands of a view command: non-option tokens after the script/pattern."""
    command = tokens[0]
    operands: list[str] = []
    script_given = False
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-") and token != "-":
            if token in _SCRIPT_OPTIONS or token.startswith("-e"):
                script_given = True
            skip_next = token in _VALUE_OPTIONS[command]
            continue
        operands.append(token)
    if command in _FIRST_OPERAND_IS_NOT_A_FILE and not script_given:
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
    if _SED_INPLACE.search(command) and "\\n" in command:
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
