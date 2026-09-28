#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two-way semantic sync of the KISS notes between this machine and a remote host.

The notes are two Markdown files in the KISS home directory (``~/.kiss``)
that the agents write on every machine they run on and that a plain copy
would clobber:

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
  same whichever machine computes it (the copy with more rows, then the
  greater text, sets the layout), so both machines end up with one text.
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
(so a ledger row appended there meanwhile survives) and writes the file
in one atomic rename.  The remote side runs this file's own source under
its system ``python3`` (shipped inline on the ssh command line), so it
needs nothing installed.  Only the standard library is used here for the
same reason.  A note that could not be synced is reported and makes the
exit status 1; the other note is still synced.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
import sys
import tempfile

NOTES = ("AUTOROUTER.md", "MODEL_DECISIONS.md")
"""The notes synced, by file name inside the KISS home directory."""

PHASE_MERGE = "_merge"
"""Internal sub-command the remote runs: merge stdin into its copy of a note."""

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
    more table rows, then more bullets, then the greater text.  The
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
    if ours.strip() == theirs.strip():
        return max(ours, theirs)  # the same text: settle the surrounding whitespace too
    our_stamp, their_stamp = stamp_of(ours), stamp_of(theirs)
    if our_stamp != their_stamp:
        return theirs if their_stamp > our_stamp else ours
    other, base = sorted((ours, theirs), key=_evidence_rank)
    lines = base.splitlines()
    start, body, end = split_table(lines)
    other_lines = other.splitlines()
    other_start, other_body, other_end = split_table(other_lines)
    header = [normalized(line) for line in lines[start:body]]
    other_header = [normalized(line) for line in other_lines[other_start:other_body]]
    if header and other_header and header != other_header:
        return base
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


def _evidence_rank(text: str) -> tuple[int, int, str]:
    """Return the key that picks the base of a same-day merge: rows, bullets, then the text.

    A merge result never ranks below either input (it holds every row and
    bullet of both), so merging it again with one of them keeps it as it is.
    """
    lines = text.splitlines()
    _start, body, end = split_table(lines)
    return (max(end - body, 0), len(split_bullets(lines)), text)


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


def kiss_dir() -> str:
    """Return the KISS home directory: ``$KISS_HOME`` when set, else ``~/.kiss``."""
    return os.environ.get("KISS_HOME") or os.path.expanduser("~/.kiss")


def read_note(path: str) -> str | None:
    """Return the text of the file at *path*, or ``None`` when there is none."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return None


def write_note(path: str, text: str) -> None:
    """Write *text* to *path* in one atomic rename (a reader never sees a partial file)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(partial, path)
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise


def merge_into_file(name: str, path: str, theirs: str | None) -> str:
    """Merge *theirs* into the note at *path* and write the result when it changed.

    Args:
        name: The note's name (picks the merge rule).
        path: The file holding this side's copy (may not exist yet).
        theirs: The other side's text, or ``None`` when it has no file.

    Returns:
        ``"unchanged"``, ``"created"`` or ``"updated"``.
    """
    ours = read_note(path)
    merged = merge_note(name, ours or "", theirs or "")
    if merged == (ours or "") or (ours is None and not merged.strip()):
        return "unchanged"
    write_note(path, merged)
    return "created" if ours is None else "updated"


# --------------------------------------------------------------------------
# ssh
# --------------------------------------------------------------------------


def _ssh(host: str, command: str, stdin: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    """Run *command* on *host* through ssh in batch mode (no prompts) and return the result."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        input=stdin,
        capture_output=True,
        check=False,
    )


def fetch_remote(host: str, name: str) -> str | None:
    """Return the remote's copy of the note *name*, or ``None`` when it has no such file.

    The remote's KISS home is resolved as :func:`kiss_dir` does there.

    Raises:
        SyncError: When ssh or the remote command fails.
    """
    command = (
        f'f="${{KISS_HOME:-$HOME/.kiss}}/{name}"; if [ -f "$f" ]; then cat "$f"; else exit 3; fi'
    )
    done = _ssh(host, command)
    if done.returncode == 3:
        return None
    if done.returncode != 0:
        raise SyncError(
            f"could not read {name} on {host} (ssh exited {done.returncode}):"
            f" {done.stderr.decode('utf-8', 'replace').strip()}"
        )
    return done.stdout.decode("utf-8", "replace")


def push_remote(host: str, name: str, text: str) -> str:
    """Merge *text* into the remote's copy of the note *name* and return the remote's verdict.

    The remote runs this file's source under its ``python3`` with the
    text on stdin, so whatever it appended to the note since it was
    fetched is merged rather than overwritten.

    Raises:
        SyncError: When ssh or the remote merge fails.
    """
    payload = base64.b64encode(_script_source()).decode("ascii")
    bootstrap = f"import base64;exec(base64.b64decode('{payload}'))"
    command = " ".join(shlex.quote(p) for p in ("python3", "-c", bootstrap, PHASE_MERGE, name))
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


def sync(host: str, home: str) -> int:
    """Sync every note of :data:`NOTES` both ways with *host*.

    Args:
        host: ``user@host`` of the remote machine.
        home: This machine's KISS home directory.

    Returns:
        0 when every note is in sync, 1 when at least one could not be.
    """
    failed = 0
    for name in NOTES:
        path = os.path.join(home, name)
        try:
            theirs = fetch_remote(host, name)
            local_state = merge_into_file(name, path, theirs)
            merged = read_note(path)
            if merged is None:
                print(f"{name}: neither machine has it")
                continue
            remote_state = push_remote(host, name, merged)
        except (SyncError, OSError) as exc:
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
        if len(args) == 2 and args[0] == PHASE_MERGE:
            print(merge_into_file(args[1], os.path.join(kiss_dir(), args[1]), sys.stdin.read()))
            return 0
        if len(args) == 4 and args[0] == "merge":
            print(merge_into_file(args[1], args[2], read_note(args[3])))
            return 0
        is_host = len(args) == 1 and not args[0].startswith("-")
        if is_host and args[0] not in ("merge", PHASE_MERGE):
            return sync(args[0], kiss_dir())
    except (SyncError, OSError) as exc:
        print(f"sync_kiss_notes: {exc}", file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
