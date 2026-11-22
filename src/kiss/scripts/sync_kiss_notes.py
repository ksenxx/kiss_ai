#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two-way semantic sync of the KISS notes between this machine and a remote host.

The notes are two Markdown files in the KISS home directory (``$KISS_HOME``,
else ``~/<home_dir>`` of the brand -- ``~/.kiss`` for stock KISS Sorcar; see
:func:`brand_home_dir`) that the agents write on every machine they run on
and that a plain copy would clobber:

* ``AUTOROUTER.md`` -- the observed model evidence the autorouter SEA
  splices into its prompt: a stamp line ``_Observed in the task history,
  refreshed YYYY-MM-DD by /rsi7d._``, a window paragraph, one table (a row
  per model, its task count in the second column) and bullets.  ``/rsi7d``
  rewrites it wholesale on each machine from that machine's copy of the
  task history.  Merge rule: the copy with the later stamp wins outright
  (evidence from an older window is stale, not complementary); two copies
  with the same stamp are combined -- table rows are united by model,
  keeping for a model both have the row that counts more tasks (the copy
  that saw more of the shared history), sorted by task count, and bullets
  are united, keeping each distinct bullet once.  The combination is the
  same whichever machine computes it (the copy with more models, then the
  greater layout, sets the layout), so both machines end up with one text.
* ``MODEL_DECISIONS.md`` -- the routing ledger: a title and one table to
  which every autorouter task appends a row ``| time | task_id | unit |
  tier | model | reason | outcome |``.  Merge rule: the union of the rows
  of both copies in time order, each distinct row once; the title and any
  other prose keep this machine's lines followed by the other's new ones.

Usage::

    sync_kiss_notes.py user@host          # both notes, both ways
    sync_kiss_notes.py merge NAME INTO FROM  # merge the note FROM into the note INTO

Called by ``./rsorcar`` on every deploy (step 4d).  For each note the
remote's copy is fetched over ssh and merged into this machine's, and the
result is sent back: the remote merges it into whatever it holds by then
(so a ledger row appended there meanwhile survives).  A merge reads,
merges and rewrites the note under the lock the agents take on it to
append a row (see :func:`merge_into_file`), so a row appended while the
sync runs is never lost.  The remote side runs this file's own source
under its system ``python3`` (shipped inline on the ssh command line), so
it needs nothing installed.  Only the standard library is used here for
the same reason.  A note that could not be synced is reported and makes
the exit status 1; the other note is still synced.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from typing import IO

try:
    import fcntl
except ImportError:  # pragma: no cover -- Windows has no fcntl
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # POSIX has no msvcrt
    msvcrt = None  # type: ignore[assignment]

NOTES = ("AUTOROUTER.md", "MODEL_DECISIONS.md")
"""The notes synced, by file name inside the KISS home directory."""

DEFAULT_HOME_DIR = ".kiss"
"""The stock brand's state directory name under ``$HOME``."""

PHASE_MERGE = "_merge"
"""Internal sub-command the remote runs: merge stdin into its copy of a note."""

MERGE_ATTEMPTS = 3
"""Merges of a note replaced under each of them before :func:`merge_into_file` gives up."""

LOCK_SECONDS = 10
"""The longest a read or a merge waits for a note's lock before it gives up (:func:`lock_note`)."""

SSH_DEADLINES = (
    "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"
)
"""ssh options that fail a command on a host that does not answer instead of hanging it.

At most ConnectTimeout seconds to connect and shake hands, and a connection
that goes silent later is dropped after ServerAliveCountMax probes
ServerAliveInterval seconds apart.
"""

REMOTE_READ = f"""import fcntl, sys, time
note = open(sys.argv[1], "rb")
deadline = time.monotonic() + {LOCK_SECONDS}
while True:
    try:
        fcntl.flock(note, fcntl.LOCK_SH | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() >= deadline:
            sys.exit("still locked by another process after {LOCK_SECONDS} s")
        time.sleep(0.05)
sys.stdout.buffer.write(note.read())
"""
"""What the remote's ``python3`` runs to print a note: the locked read of :func:`read_note`."""

_STAMP = re.compile(r"refreshed (\d{4}-\d{2}-\d{2})")
_BULLET = re.compile(r"^[-*+] ")
_CONTINUATION = re.compile(r"^\s{2,}\S")


class SyncError(Exception):
    """A note could not be synced; both copies are left as they were."""


# --------------------------------------------------------------------------
# Markdown pieces
# --------------------------------------------------------------------------


def is_table_row(line: str) -> bool:
    """Return whether *line* is a Markdown table row (starts with ``|``)."""
    return line.strip().startswith("|")


def is_separator_row(line: str) -> bool:
    """Return whether *line* is a table's header separator (``|---|---|``)."""
    return is_table_row(line) and set(line.strip()) <= set("|-: ")


def cells(row: str) -> list[str]:
    """Return the stripped cells of a table *row* (the outer bars dropped)."""
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def normalized(text: str) -> str:
    """Return *text* with runs of whitespace collapsed, for comparing lines."""
    return " ".join(text.split())


def split_table(lines: list[str]) -> tuple[int, int, int]:
    """Locate the first table in *lines*.

    Returns:
        ``(start, body, end)``: the table's first line, the first body row
        and the line after the table; ``(-1, -1, -1)`` when there is no
        table.  The table has a header (``body == start + 2``) only when
        a separator row follows its first row; otherwise every row is a
        body row (``body == start``).
    """
    start = next((i for i, line in enumerate(lines) if is_table_row(line)), -1)
    if start < 0:
        return -1, -1, -1
    end = start
    while end < len(lines) and is_table_row(lines[end]):
        end += 1
    body = start + 2 if start + 1 < end and is_separator_row(lines[start + 1]) else start
    return start, body, end


def split_bullets(lines: list[str]) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` line spans of the top-level bullet items in *lines*.

    An item is a ``- `` line and the indented lines that continue it.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _BULLET.match(lines[i]):
            end = i + 1
            while end < len(lines) and _CONTINUATION.match(lines[end]):
                end += 1
            spans.append((i, end))
            i = end
        else:
            i += 1
    return spans


# --------------------------------------------------------------------------
# the ledger: MODEL_DECISIONS.md
# --------------------------------------------------------------------------


def merge_ledger(ours: str, theirs: str) -> str:
    """Return the union of two routing ledgers.

    Rows (``| time | task_id | ... |`` lines after the table header) are
    united, each distinct row once, and ordered by their first cell, the
    UTC time stamp ``YYYY-MM-DD HH:MM``; rows with the same time keep
    ours before theirs.  The header comes from ours (theirs when ours has
    none) and the prose outside the table keeps our lines followed by
    the lines only theirs has.

    Args:
        ours: This machine's ledger text (``""`` when it has none).
        theirs: The other machine's ledger text (``""`` when it has none).

    Returns:
        The merged ledger text, ending in a newline; ``""`` only when both
        inputs are blank.
    """
    if not theirs.strip():
        return ours
    if not ours.strip():
        return theirs
    our_prose, our_header, our_rows = _ledger_parts(ours)
    their_prose, their_header, their_rows = _ledger_parts(theirs)
    prose = list(our_prose)
    seen = {normalized(line) for line in prose}
    for line in their_prose:
        if normalized(line) not in seen:
            prose.append(line)
            seen.add(normalized(line))
    rows: dict[str, str] = {}
    for row in our_rows + their_rows:
        rows.setdefault(normalized(row), row.strip())
    ordered = sorted(rows.values(), key=lambda row: cells(row)[0])
    header = our_header or their_header
    out = "\n".join(prose)
    if header or ordered:
        out += "\n\n" if out else ""
        out += "\n".join(header + ordered)
    return out + "\n"


def _ledger_parts(text: str) -> tuple[list[str], list[str], list[str]]:
    """Split a ledger into its prose lines, table header lines and body rows.

    Every table row of the file counts, whichever table it is in, so a
    ledger split by a stray blank line still yields all of its rows.
    """
    lines = text.splitlines()
    start, body, end = split_table(lines)
    header = [line.strip() for line in lines[start:body]] if start >= 0 else []
    prose = [line.rstrip() for line in lines if line.strip() and not is_table_row(line)]
    rows = [
        line
        for i, line in enumerate(lines)
        if is_table_row(line) and not (start <= i < body) and not is_separator_row(line)
    ]
    return prose, header, rows


# --------------------------------------------------------------------------
# the evidence: AUTOROUTER.md
# --------------------------------------------------------------------------


def stamp_of(text: str) -> str:
    """Return the ``refreshed YYYY-MM-DD`` date of an evidence *text*, or ``""``."""
    match = _STAMP.search(text)
    return match.group(1) if match else ""


def merge_evidence(ours: str, theirs: str) -> str:
    """Return the merge of two copies of the observed model evidence.

    The copy with the later ``refreshed`` stamp wins outright (a copy
    without a stamp counts as the oldest).  Two copies stamped the same
    day -- or both unstamped -- are combined, and both machines must
    arrive at the same text whichever side they merge from, so the
    *base* is chosen by a rule symmetric in the two copies: the one with
    more models, then more bullets, then the greater layout.  The
    base's prose and layout are kept; its table body becomes the union of
    both tables' rows keyed by the first cell (the model) -- a model both
    list keeps the row with the larger second cell (tasks), the greater
    text on a tie, so the choice does not depend on the base -- sorted by
    that count, largest first, rows without a count last; its bullets are
    followed by the other copy's bullets it lacks, appended when the base
    has none (a copy with table rows always outranks one without, so the
    base never lacks a table the other copy has).  Two tables with
    different header rows cannot be combined cell by cell: the base wins
    outright (a table without a header row goes with any header).

    Args:
        ours: This machine's evidence text (``""`` when it has none).
        theirs: The other machine's evidence text (``""`` when it has none).

    Returns:
        The merged evidence text.
    """
    if not theirs.strip():
        return ours
    if not ours.strip():
        return theirs
    if ours == theirs:
        return ours  # the same text bar surrounding blank lines is merged: one text after a sync
    our_stamp, their_stamp = stamp_of(ours), stamp_of(theirs)
    if our_stamp != their_stamp:
        return theirs if their_stamp > our_stamp else ours
    other, base = sorted((ours, theirs), key=_evidence_rank)
    lines = base.splitlines()
    start, body, _end = split_table(lines)
    other_lines = other.splitlines()
    other_start, other_body, _other_end = split_table(other_lines)
    header = [normalized(line) for line in lines[start:body]]
    other_header = [normalized(line) for line in other_lines[other_start:other_body]]
    if header and other_header and header != other_header:
        return base
    return _combine(lines, other_lines)


def _combine(lines: list[str], other_lines: list[str]) -> str:
    """Return the base's *lines* with its rows and bullets united with the other copy's.

    With no other lines, the base as a merge writes it: its own repeated rows and
    bullets united, its rows sorted, its bullet groups moved together.
    """
    start, body, end = split_table(lines)
    _other_start, other_body, other_end = split_table(other_lines)
    rows = _merge_rows(lines[body:end] if start >= 0 else [], other_lines[other_body:other_end])
    bullets = _merge_bullets(lines, other_lines)
    out: list[str] = []
    spans = split_bullets(lines)
    table_done, bullets_done = start < 0, not spans
    i = 0
    while i < len(lines):
        if not table_done and i == body:
            out.extend(rows)
            i, table_done = end, True
        elif not bullets_done and i == spans[0][0]:
            out.extend(bullets)
            i, bullets_done = spans[0][1], True
        elif any(s <= i < e for s, e in spans):
            i += 1  # a later bullet item: already emitted with the first
        else:
            out.append(lines[i])
            i += 1
    if not spans and bullets:
        out += ["", *bullets]
    return "\n".join(out).rstrip("\n") + "\n"


def _evidence_rank(text: str) -> tuple[int, int, list[str], list[str], str]:
    """Return the key that picks the base of a same-day merge.

    Taken of the text as a merge writes it (:func:`_combine` with nothing): its rows,
    its bullets, its layout -- the lines before and after the table body -- and last the
    text itself.  A merge result holds every row and bullet of both inputs in the layout
    of its base, so it ranks below neither input in rows, bullets and layout, and
    merging it again with one of them keeps it as it is.
    """
    lines = _combine(text.splitlines(), []).splitlines()
    _start, body, end = split_table(lines)
    return (max(end - body, 0), len(split_bullets(lines)), lines[:body], lines[end:], text)


def _task_count(row: str) -> int | None:
    """Return the integer in a row's second cell (``1,809`` counts too), or ``None``.

    ``None`` also for a row whose first cell starts with "other" (the
    "others (7 models)" tally ``/rsi7d`` writes last), so it stays last.
    """
    parts = cells(row)
    if len(parts) < 2 or parts[0].lower().startswith("other"):
        return None
    digits = parts[1].replace(",", "")
    return int(digits) if digits.isdigit() else None


def _row_rank(row: str) -> tuple[int, str]:
    """Return the key that picks between two rows of one model: task count, then the text."""
    count = _task_count(row)
    return (-1 if count is None else count, row)


def _merge_rows(base: list[str], other: list[str]) -> list[str]:
    """Unite two tables' body rows by first cell; see :func:`merge_evidence`."""
    merged: dict[str, str] = {}
    for row in base + other:
        key = cells(row)[0].lower()
        current = merged.get(key)
        merged[key] = row.strip() if current is None else max(current, row.strip(), key=_row_rank)
    # Largest count first; rows tied on the count, and the count-less rows at
    # the end, in text order, so the order does not depend on the base either.
    counted = sorted(
        (row for row in merged.values() if _task_count(row) is not None),
        key=lambda row: (-(_task_count(row) or 0), row),
    )
    return counted + sorted(row for row in merged.values() if _task_count(row) is None)


def _merge_bullets(base: list[str], other: list[str]) -> list[str]:
    """Return the base's bullet items followed by the other copy's items it lacks."""
    out: list[str] = []
    seen: set[str] = set()
    for lines in (base, other):
        for start, end in split_bullets(lines):
            item = lines[start:end]
            key = normalized(" ".join(item))
            if key not in seen:
                seen.add(key)
                out.extend(item)
    return out


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

MERGERS = {"AUTOROUTER.md": merge_evidence, "MODEL_DECISIONS.md": merge_ledger}


def merge_note(name: str, ours: str, theirs: str) -> str:
    """Merge two copies of the note *name* with the rule of that note.

    Args:
        name: One of :data:`NOTES`.
        ours: This side's text (``""`` for no file).
        theirs: The other side's text (``""`` for no file).

    Returns:
        The merged text.

    Raises:
        SyncError: For a name that is not a synced note.
    """
    if name not in MERGERS:
        raise SyncError(f"{name!r} is not a synced note (one of {', '.join(NOTES)})")
    return MERGERS[name](ours, theirs)


def brand_home_dir() -> str:
    """Return the name of the brand's state directory under ``$HOME``.

    Read as ``install.sh`` reads it (``brand_home_dir_name`` there):
    ``home_dir`` of the checkout's ``.brand/brand.json`` overlay when
    there is one, else of the ``brand.json`` next to the ``kiss`` package
    this script belongs to -- the file ``kiss.core.brand`` reads, so the
    answer is the directory the agents write the notes in (``.s10s`` for
    Seamless Loop, :data:`DEFAULT_HOME_DIR` for stock KISS Sorcar).  A
    value that is not a single path component, or no file at all, falls
    back to :data:`DEFAULT_HOME_DIR`.  The remote runs this source with
    no file of its own and is told the name instead (:data:`PHASE_MERGE`).
    """
    script = globals().get("__file__")
    if not script:
        return DEFAULT_HOME_DIR
    package = os.path.dirname(os.path.dirname(os.path.realpath(script)))
    checkout = os.path.dirname(os.path.dirname(package))
    for candidate in (
        os.path.join(checkout, ".brand", "brand.json"),
        os.path.join(package, "webapp", "media", "brand.json"),
    ):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                name = json.load(handle).get("home_dir")
        except (OSError, ValueError, AttributeError):
            return DEFAULT_HOME_DIR
        one_component = (
            isinstance(name, str) and name not in ("", ".", "..") and not set(name) & set("/\\")
        )
        return name if one_component else DEFAULT_HOME_DIR
    return DEFAULT_HOME_DIR


def kiss_dir(home_dir: str | None = None) -> str:
    """Return the KISS home directory: ``$KISS_HOME`` when set, else ``~/<home_dir>``.

    Args:
        home_dir: The brand's state directory name; :func:`brand_home_dir`
            when ``None``.
    """
    return os.environ.get("KISS_HOME") or os.path.expanduser("~/" + (home_dir or brand_home_dir()))


def _open_without_waiting(path: str, flags: int) -> int:
    """Opener for ``open``: a named pipe nobody writes to opens at once instead of never."""
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def read_note(path: str) -> str | None:
    """Return the text of the file at *path*, or ``None`` when there is none.

    Read under the note's lock (:func:`lock_note`, as a reader), so a merge
    rewriting the note in place is seen before it or after it, never
    between its write and its truncate.

    Raises:
        SyncError: When the note stays locked (:func:`lock_note`), or *path*
            names something other than a regular file (a named pipe, a
            device), which is not read at all.
    """
    try:
        with open(path, encoding="utf-8", opener=_open_without_waiting) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise SyncError(f"{path} is not a regular file")
            lock_note(handle, shared=True)
            return handle.read()
    except FileNotFoundError:
        return None


def lock_note(handle: IO[str], shared: bool = False) -> None:
    """Take the lock the agents take to append to a note: ``flock`` on the open note.

    The same lock as ``kiss.core.file_lock.lock_exclusive`` (``flock`` on
    POSIX, the note's first byte on Windows), so that the autorouter's
    ``log_decision`` and this script's merges take turns on the ledger.
    Waits for the holder, :data:`LOCK_SECONDS` at most (on Windows the ten
    tries of ``LK_LOCK``, a second apart); released by :func:`unlock_note`
    or by closing the file.  *shared* is the lock of a reader: it waits
    for a writer, not for another reader, and a file opened for reading
    only may hold it (over NFS ``flock`` grants the exclusive one to a
    writable file only).

    Raises:
        SyncError: When another process still holds the lock at the
            deadline (``OSError`` on Windows).
    """
    if fcntl is not None:
        mode = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
        deadline = time.monotonic() + LOCK_SECONDS
        while True:
            try:
                fcntl.flock(handle.fileno(), mode)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SyncError(
                        f"still locked by another process after {LOCK_SECONDS} s"
                    ) from None
                time.sleep(0.05)
    elif msvcrt is not None:  # pragma: no cover -- Windows-only branch
        os.lseek(handle.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # pyright: ignore[reportAttributeAccessIssue]


def unlock_note(handle: IO[str]) -> None:
    """Release the lock :func:`lock_note` took on *handle*."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover -- Windows-only branch
        os.lseek(handle.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]


@contextmanager
def rename_lock(path: str) -> Iterator[bool]:
    """Hold the lock file beside *path*, ``<path>.lock``, for the block.

    The lock of whoever replaces what is at *path* by a rename, held from
    before the note is read until after the rename: a merge of the
    evidence (:func:`merge_into_file`) and ``/rsi7d``, which rewrites the
    evidence wholesale.  They take turns, so neither renames its text over
    one the other wrote after it had read the note.  The lock is
    :func:`lock_note`'s, on the lock file, which is created empty and
    private to its user when missing and never deleted.  A merge through
    a link that leads to the lock file renames over it, though: the lock
    is held once the locked file is still the one at that name.

    Args:
        path: The path a file is about to be renamed over.

    Yields:
        Whether the lock is held.  ``False`` where there is no lock to be
        had: the lock file cannot be opened (a directory that takes no new
        file or does not exist, a lock file of another user), it is not a
        regular file with this one name (a link may lead to the note
        itself, or to a lock the caller holds), or it is the file *path*
        leads to: to wait for those is to wait for oneself.  The caller
        then writes as it did before there was a lock file, and a merge
        renames nothing.

    Raises:
        SyncError: When the lock stays held (:func:`lock_note`).
    """
    with ExitStack() as stack:
        held = False
        try:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            while not held:
                fd = os.open(path + ".lock", flags, 0o600)
                lock = os.fdopen(fd, encoding="utf-8")
                stack.callback(lock.close)  # closing it releases it
                status = os.fstat(fd)
                if (
                    not stat.S_ISREG(status.st_mode)
                    or status.st_nlink != 1
                    or (os.path.exists(path) and os.path.samefile(fd, path))
                ):
                    break
                lock_note(lock)
                held = os.path.samestat(status, os.stat(path + ".lock"))
                if not held:  # renamed over while this waited: the lock is the file there now
                    lock.close()
        except OSError:
            pass
        yield held  # after the handler: an error of the block would carry that one along


@contextmanager
def _rename_locks(path: str) -> Iterator[str | None]:
    """Hold the lock file of every name on the way from *path* to its file, for the block.

    That of *path* and of each symlink followed from it (:func:`rename_lock`),
    in that order.  ``/rsi7d`` replaces the note, symlink or not, and a merge
    the file a symlinked note leads to: both wait, so *path* leads to one
    file from here until the block ends, by whichever of its names a merge
    was given.

    Yields:
        The path of that file, or ``None`` where one of the locks is not held.
    """
    with ExitStack() as stack:
        locked = True
        for _hop in range(41):  # the 40 links the kernel follows at most, then the file
            locked = stack.enter_context(rename_lock(path)) and locked
            try:
                target = os.readlink(path)
            except OSError:  # no symlink: the file itself, or nothing yet
                break
            path = os.path.join(os.path.dirname(path), target)
        yield path if locked else None


def _extended_attributes(fd: int) -> dict[str, bytes]:
    """Return the extended attributes of the open file *fd*; an ACL is one of them.

    Raises:
        OSError: Off Linux, where Python cannot read them.
    """
    if sys.platform != "linux":  # ``os.listxattr`` is Linux's
        raise OSError("Python reads extended attributes on Linux only")
    return {name: os.getxattr(fd, name) for name in os.listxattr(fd)}


def replace_note(path: str, text: str, like: IO[str] | None = None, mode: int = 0o666) -> None:
    """Put *text* at *path* by renaming a new file over whatever is there.

    The text is staged beside *path* in a file created for it, under a name
    nothing has (so never through a link to the note), and that file is
    renamed over *path*: whoever reads the note gets the old text or the
    new one, whole, with the note's lock or without.  The caller holds the
    lock file of *path* (:func:`rename_lock`).

    Args:
        path: Where the text goes; a symlink there is replaced, not followed.
        text: The note's new text.
        like: The open note the new file is to stand in for.  ``None`` leaves
            the new file what it is, a file its user just made: what
            ``/rsi7d`` has always renamed over the evidence.
        mode: The permission bits the new file is made with (less the
            umask's); those of *like* replace them.

    Raises:
        OSError: When nothing was renamed and the note is as it was: the
            directory takes no new file, or the new file cannot stand in
            for *like* -- *like* is not the file at *path* or has other
            names as well (hard links), the new file cannot be given its
            owner, group and permission bits, or lacks an extended
            attribute it has (an ACL is one) -- or may lack one, off Linux,
            where Python cannot read them.
    """
    staged = f"{path}.{os.urandom(8).hex()}"
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stage:
            stage.write(text)
            if like is not None:
                note = like.fileno()
                status = os.fstat(note)
                os.fchown(fd, status.st_uid, status.st_gid)
                os.fchmod(fd, stat.S_IMODE(status.st_mode))
                if (
                    not os.path.samestat(status, os.lstat(path))
                    or status.st_nlink != 1
                    or _extended_attributes(fd) != _extended_attributes(note)
                ):
                    raise OSError(f"no new file can stand in for {path}")
        os.replace(staged, path)
    finally:
        with suppress(OSError):  # gone once renamed; stays where the directory lets nothing go
            os.unlink(staged)


def merge_into_file(name: str, path: str, theirs: str | None) -> str:
    """Merge *theirs* into the note at *path* and write the result when it changed.

    The read, the merge and the write are one step under the lock the
    agents take on the note to append a row to it (:func:`lock_note`): a
    row appended meanwhile lands after the merged text, and two merges at
    once take turns.

    The ledger is rewritten in place rather than replaced by a rename: an
    agent that opened it to append and is waiting for the lock keeps the
    file it opened, so its row lands after the merged text instead of in
    an orphaned copy.  The evidence has no appender, so it is replaced by
    a rename (:func:`replace_note`), which shows a reader the old text or
    the new one and never a mix of them.  The file renamed over is the one
    *path* leads to (a symlinked note stays a symlink), and the merge
    holds the lock file of each name on the way to it from before it opens
    the note (:func:`_rename_locks`), as ``/rsi7d`` holds the note's to
    replace the evidence: the two take turns as well.  Evidence that is
    not there yet is made the same way, so no reader finds it empty.
    Where no new file can stand in for the evidence -- a lock file cannot be had, the
    directory takes no new file, the note has an ACL or a second name, or
    off Linux, where Python cannot tell whether it has -- it is rewritten
    in place like the ledger, as it always was, under the lock files the
    merge holds; a reader is sure not to see it between the write and the
    truncate only when it takes the note's lock (:func:`read_note`).
    Windows renames over no file that is open: the evidence is merged
    there as the ledger is, with no lock file.

    Whoever replaces a note by a rename without these locks (an editor, a
    daemon from before ``/rsi7d`` took the lock file) is not waited for: a
    merge that then finds another file at *path* than the one it merged
    into, in place, merges again, into that one.

    Args:
        name: The note's name (picks the merge rule).
        path: The file holding this side's copy (may not exist yet).
        theirs: The other side's text, or ``None`` when it has no file.

    Returns:
        ``"unchanged"``, ``"created"`` or ``"updated"``.

    Raises:
        SyncError: When the note was replaced during each of
            :data:`MERGE_ATTEMPTS` merges, or it or its lock file stays
            locked (:func:`lock_note`).
    """
    try:
        os.stat(path)
        existed = True
    except FileNotFoundError:
        existed = False
    if not existed and not merge_note(name, "", theirs or "").strip():
        return "unchanged"  # nothing to create
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    replaceable = name == "AUTOROUTER.md" and os.name == "posix"
    with _rename_locks(path) if replaceable else nullcontext() as real:
        if real is not None and not os.path.lexists(real):
            with suppress(OSError):  # nowhere to stage it: made in place below, as it always was
                replace_note(real, merge_note(name, "", theirs or ""), mode=0o600)
                return "created"
        for _attempt in range(MERGE_ATTEMPTS):
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+", encoding="utf-8") as handle:
                lock_note(handle)
                try:
                    current = handle.read()
                    merged = merge_note(name, current, theirs or "")
                    renamed = False
                    if merged != current:
                        if real is not None:
                            with suppress(OSError):  # no new file can stand in for this note
                                replace_note(real, merged, handle, 0o600)
                                renamed = True
                        if not renamed:
                            handle.seek(0)
                            handle.write(merged)
                            handle.truncate()
                            handle.flush()
                    if not renamed and not os.path.samefile(handle.fileno(), path):
                        continue  # merged into a file a rename has replaced meanwhile
                finally:
                    unlock_note(handle)
            if merged == current:
                return "unchanged"
            return "updated" if existed else "created"
    raise SyncError(f"{path} was replaced during each of {MERGE_ATTEMPTS} merges into it")


# --------------------------------------------------------------------------
# ssh
# --------------------------------------------------------------------------


def _ssh(host: str, command: str, stdin: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    """Run *command* on *host* through ssh in batch mode (no prompts) and return the result.

    A host that does not answer fails the command (:data:`SSH_DEADLINES`), and what
    the remote runs gives up on a locked note by itself (:data:`LOCK_SECONDS`): no cap
    on the whole command, which would cut short a note that only takes long to travel.
    """
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", *SSH_DEADLINES, host, command],
        input=stdin,
        capture_output=True,
        check=False,
    )


def fetch_remote(host: str, name: str, home_dir: str) -> str | None:
    """Return the remote's copy of the note *name*, or ``None`` when it has no such file.

    The remote's KISS home is resolved as :func:`kiss_dir` resolves it
    there: ``$KISS_HOME`` of the ssh session when set, else
    ``$HOME/<home_dir>``.  The note is read under its lock there
    (:data:`REMOTE_READ`), as :func:`read_note` reads this machine's.

    Args:
        host: ``user@host`` of the remote machine.
        name: One of :data:`NOTES`.
        home_dir: The brand's state directory name (:func:`brand_home_dir`).

    Raises:
        SyncError: When ssh or the remote command fails, or the remote's
            copy is not UTF-8: decoded with replacement characters it
            would be merged into this machine's copy.
    """
    command = (
        f'h=$HOME/{shlex.quote(home_dir)}; f="${{KISS_HOME:-$h}}/{name}";'
        f' if [ -f "$f" ]; then python3 -c {shlex.quote(REMOTE_READ)} "$f"; else exit 3; fi'
    )
    done = _ssh(host, command)
    if done.returncode == 3:
        return None
    if done.returncode != 0:
        raise SyncError(
            f"could not read {name} on {host} (ssh exited {done.returncode}):"
            f" {done.stderr.decode('utf-8', 'replace').strip()}"
        )
    try:
        return done.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyncError(f"{name} on {host} is not UTF-8: {exc}") from None


def push_remote(host: str, name: str, text: str, home_dir: str) -> str:
    """Merge *text* into the remote's copy of the note *name* and return the remote's verdict.

    The remote runs this file's source under its ``python3`` with the
    text on stdin, so whatever it appended to the note since it was
    fetched is merged rather than overwritten.  It is told *home_dir*,
    the directory :func:`fetch_remote` read, since it has no checkout to
    read the brand from.

    Args:
        host: ``user@host`` of the remote machine.
        name: One of :data:`NOTES`.
        text: The merged note to merge into the remote's copy.
        home_dir: The brand's state directory name (:func:`brand_home_dir`).

    Raises:
        SyncError: When ssh or the remote merge fails.
    """
    payload = base64.b64encode(_script_source()).decode("ascii")
    bootstrap = f"import base64;exec(base64.b64decode('{payload}'))"
    command = " ".join(
        shlex.quote(p) for p in ("python3", "-c", bootstrap, PHASE_MERGE, name, home_dir)
    )
    done = _ssh(host, command, text.encode("utf-8"))
    stderr = done.stderr.decode("utf-8", "replace").strip()
    if done.returncode != 0:
        raise SyncError(
            f"could not merge {name} into {host}'s copy (exit {done.returncode})"
            + (f": {stderr}" if stderr else "")
        )
    return done.stdout.decode("utf-8", "replace").strip() or "unchanged"


def _script_source() -> bytes:
    """Return this file's source, for running the merge on the remote."""
    path = globals().get("__file__")
    if not path:
        raise SyncError("cannot locate this script's source for remote execution")
    with open(path, "rb") as handle:
        return handle.read()


def sync(host: str, home: str, home_dir: str) -> int:
    """Sync every note of :data:`NOTES` both ways with *host*.

    Args:
        host: ``user@host`` of the remote machine.
        home: This machine's KISS home directory.
        home_dir: The brand's state directory name, which places the
            notes on the remote (:func:`brand_home_dir`).

    Returns:
        0 when every note is in sync, 1 when at least one could not be.
    """
    failed = 0
    for name in NOTES:
        path = os.path.join(home, name)
        try:
            theirs = fetch_remote(host, name, home_dir)
            local_state = merge_into_file(name, path, theirs)
            merged = read_note(path)
            if merged is None:
                print(f"{name}: neither machine has it")
                continue
            remote_state = push_remote(host, name, merged, home_dir)
        except (SyncError, OSError, UnicodeError) as exc:  # UnicodeError: a note not in UTF-8
            print(f"warning: {name}: {exc}", file=sys.stderr)
            failed = 1
            continue
        print(f"{name}: here {local_state}, {host} {remote_state}")
    return failed


USAGE = (
    "usage: sync_kiss_notes.py user@host          (sync both notes both ways)\n"
    "       sync_kiss_notes.py merge NAME INTO FROM  (merge the note file FROM into INTO)"
)


def main(argv: list[str] | None = None) -> int:
    """Run the sync, a local merge or the remote merge phase; return the exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) == 3 and args[0] == PHASE_MERGE:
            name, home_dir = args[1], args[2]
            # The sender writes UTF-8 (push_remote); sys.stdin would decode it in this locale's.
            theirs = sys.stdin.buffer.read().decode("utf-8")
            print(merge_into_file(name, os.path.join(kiss_dir(home_dir), name), theirs))
            return 0
        if len(args) == 4 and args[0] == "merge":
            print(merge_into_file(args[1], args[2], read_note(args[3])))
            return 0
        is_host = len(args) == 1 and not args[0].startswith("-")
        if is_host and args[0] not in ("merge", PHASE_MERGE):
            home_dir = brand_home_dir()
            return sync(args[0], kiss_dir(home_dir), home_dir)
    except (SyncError, OSError, UnicodeError) as exc:
        print(f"sync_kiss_notes: {exc}", file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
