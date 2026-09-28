# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The user's standing instructions in ``~/.kiss/SORCAR.md``.

``RelentlessAgent.perform_task`` appends the whole of
``$KISS_HOME/SORCAR.md`` (``~/.kiss/SORCAR.md`` by default) to the
system prompt of every Sorcar task, so a line written there is an
instruction the agent follows in every later task.  This module is the
storage layer shared by the ``/remember`` and ``/forget`` agents
(:mod:`kiss.agents.seas.remember.remember_sea`, :mod:`kiss.agents.seas.forget.forget_sea`):
each instruction is one Markdown bullet line (``- <instruction>``), the
file is created with a short heading on first use, and any other text
the user wrote in the file by hand is left byte for byte as it was
(undecodable bytes and whatever line endings each line has included).
Every update runs under an inter-process lock (``SORCAR.md.lock``
beside the file), so two tasks remembering at once cannot lose each
other's line, and the new content is moved into place atomically, so a
task reading the file for its system prompt never sees it half-written.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock

HEADER = ["# User instructions", ""]
"""The lines written above the first bullet when ``/remember`` creates the file."""

_BULLET = re.compile(r"^\s*[-*+]\s+(.*\S)\s*$")
_LEADING_MARKERS = re.compile(r"^(?:[-*+](?:\s+|$))+")


def sorcar_md_path() -> Path:
    """Return the path of the user's instruction file, ``$KISS_HOME/SORCAR.md``."""
    return kiss_home() / "SORCAR.md"


def normalize(instruction: str) -> str:
    """Return *instruction* as one line: whitespace collapsed, bullet markers dropped.

    Args:
        instruction: The text the user typed after ``/remember`` or ``/forget``,
            or the text of a stored bullet line.

    Returns:
        The single-line instruction text, or ``""`` when nothing but
        whitespace and bullet markers was given.
    """
    text = " ".join(instruction.split())
    return _LEADING_MARKERS.sub("", text)


def _key(instruction: str) -> str:
    """Return the comparison key of an instruction: normalized and case-folded."""
    return normalize(instruction).casefold()


def _read_lines(path: Path) -> list[str]:
    """Return the file's lines, each with its own line terminator.

    Every byte of the file is in exactly one returned line: undecodable
    bytes survive as surrogate escapes and terminators are kept as they
    are (``\\n``, ``\\r\\n`` or ``\\r``, mixed or missing on the last
    line), so :func:`_write_lines` puts unchanged lines back byte for
    byte.  A file that does not exist reads as no lines.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="surrogateescape", newline="")
    return text.splitlines(keepends=True)


def _terminator(line: str) -> str:
    """Return the line terminator ending *line* (``""`` when it has none)."""
    return line[len(line.rstrip("\r\n")):]


def _newline(lines: list[str]) -> str:
    """Return the terminator new lines should use: the file's last one, else ``\\n``."""
    for line in reversed(lines):
        if _terminator(line):
            return _terminator(line)
    return "\n"


def _write_lines(path: Path, lines: list[str]) -> None:
    """Replace the file's content with *lines* (terminators included) atomically.

    The text is written to a sibling temporary file and moved over the
    real one, so a task reading ``SORCAR.md`` for its system prompt at
    the same moment sees either the old or the new content, never a
    truncated file.  Callers hold the ``SORCAR.md.lock`` lock, so the
    temporary file's fixed name is never contended.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text("".join(lines), encoding="utf-8", errors="surrogateescape", newline="")
    os.replace(staging, path)


def _bullet_text(line: str) -> str | None:
    """Return the text of a bullet *line*, or ``None`` when it is not a bullet."""
    match = _BULLET.match(line.rstrip("\r\n"))
    return match.group(1) if match else None


def _instructions(lines: list[str]) -> list[str]:
    """Return the text of every bullet line in *lines*, in order."""
    return [text for text in map(_bullet_text, lines) if text is not None]


def read_instructions() -> list[str]:
    """Return the instructions stored in the file: every bullet line's text, in order."""
    return _instructions(_read_lines(sorcar_md_path()))


def _format_list(instructions: list[str], path: Path) -> str:
    """Return *instructions* numbered one per line, or a sentence when empty."""
    if not instructions:
        return f"No instructions are stored in {path}."
    return "\n".join(f"{i}. {text}" for i, text in enumerate(instructions, 1))


def list_instructions() -> str:
    """List the standing instructions currently stored in ~/.kiss/SORCAR.md.

    Returns:
        One numbered line per instruction, exactly as stored, or a
        sentence saying the file holds no instructions.
    """
    return _format_list(read_instructions(), sorcar_md_path())


def add_instruction(instruction: str) -> str:
    """Append *instruction* as a bullet line to the file.

    Args:
        instruction: The instruction text; collapsed to one line.

    Returns:
        A one-line report: added, already present, or rejected as empty.
    """
    text = normalize(instruction)
    if not text:
        return "Error: the instruction is empty; nothing was remembered."
    path = sorcar_md_path()
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        lines = _read_lines(path)
        if any(_key(existing) == text.casefold() for existing in _instructions(lines)):
            return f"Already remembered in {path}: {text}"
        newline = _newline(lines)
        if not lines:
            lines = [f"{line}{newline}" for line in HEADER]
        elif not _terminator(lines[-1]):
            lines[-1] += newline  # the bullet must start on a line of its own
        _write_lines(path, lines + [f"- {text}{newline}"])
    return f"Remembered in {path}: {text}"


def remove_instruction(instruction: str) -> str:
    """Delete the bullet line(s) whose text equals *instruction*.

    The comparison ignores case, surrounding whitespace, bullet markers
    and the amount of whitespace between words.

    Args:
        instruction: The stored instruction text to remove.

    Returns:
        A one-line report of what was removed, or an error naming the
        instructions that are stored when none matches.
    """
    text = normalize(instruction)
    if not text:
        return "Error: the instruction is empty; nothing was forgotten."
    path = sorcar_md_path()
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        lines = _read_lines(path)
        kept = []
        removed = []
        for line in lines:
            stored = _bullet_text(line)
            if stored is not None and _key(stored) == text.casefold():
                removed.append(stored)
            else:
                kept.append(line)
        if not removed:
            return (
                f"Error: no instruction in {path} matches: {text}\n"
                f"Stored instructions:\n{_format_list(_instructions(lines), path)}"
            )
        _write_lines(path, kept)
    return f"Forgot from {path}: {removed[0]}"
