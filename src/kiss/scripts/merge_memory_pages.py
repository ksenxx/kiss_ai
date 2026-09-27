#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge one directory of memory pages into another, newest page wins.

The agent's persistent memory (``kiss.core.memoryfield``) is a flat directory
of Markdown pages, ``<name>.md``, each starting with a YAML frontmatter block
that carries an ``updated`` timestamp, plus its domain memories: flat page
directories nested one level down, ``<name>/`` (the memory of a repository,
for example).  Deleting a page leaves a tombstone, ``.tombstones/<name>``,
holding the time of the deletion.  scripts/sync-memory.sh copies the pages
and tombstones of one machine next to the memory of the other and runs this
to fold them in.

Usage:
    python3 merge_memory_pages.py [--tolerance SECONDS] SOURCE_DIR DEST_DIR

For every page in ``SOURCE_DIR`` and in each of its domain memories:

* a page ``DEST_DIR`` does not have is added -- unless ``DEST_DIR`` holds a
  tombstone for it that is newer than the page: then the page was deleted
  after it was last written, and it stays out;
* a page both have with the same bytes is left alone;
* a page both have with different content is replaced when the source's
  ``updated`` is the newer (a page without a parsable ``updated`` counts its
  file modification time instead); an older source is kept out;
* a page both have, different, with the same ``updated`` on both sides was
  edited on one side without the agent -- by hand, in an editor -- which
  moves the file's modification time but not the stamp: the copy modified
  later wins.  Modified in the same second on both sides, it is a conflict:
  the destination keeps its own copy and the page is named on stdout, so a
  person can decide.

For every tombstone in ``SOURCE_DIR``: a page ``DEST_DIR`` has that is older
than the deletion is removed, and the tombstone is copied so the deletion
travels on to the next machine (over an older tombstone of the destination,
never over a newer one); a page newer than the deletion was written again
after it and wins -- the tombstone is dropped.  A page and a tombstone of
the same second are a conflict, resolved in favour of the page.  A memory
holding both a page and its tombstone -- a write interrupted before it could
clear the tombstone -- keeps the page; the tombstone is stale and removed.

An edit made without the agent is seen only against another copy of the page
carrying the same stamp.  Measured against a later deletion, or a later
agent write, of the page on the other machine, the stamp is what counts and
the edit loses: the file's modification time alone cannot tell such an edit
from a copy made without preserving times, which must not beat real changes.

``--tolerance SECONDS`` widens what "the same second" means: two times that
are closer together than this are treated as equal.  scripts/sync-memory.sh
passes the difference it measured between the two machines' clocks, so that
a clock running ahead cannot make an older edit overwrite a newer one; the
pair is reported as a conflict instead.

No file other than a page or a tombstone is touched -- in particular not the
``*.sqlite3`` vector index that lives in the same directory: it is a cache
keyed by page content, and the next agent that opens the memory re-embeds
what changed.  Pages are put in place by an atomic rename, so an agent
reading the memory meanwhile sees either the old page or the new one, never
a half-written file.  An agent that rewrites the very page being merged in
the instant between the comparison and the rename loses that write, the
same way it would to a second agent writing the page: the memory has no page
locks, and this script takes no more than an agent does.

Stdlib-only and self-contained, so it can be copied to a remote machine and
run there with the system ``python3`` before the project's environment exists.

Prints one summary line -- ``added N updated N kept N conflicts N removed N``
-- followed by the name of every conflicting page, one per line
(``<memory>/<name>`` for a page of a domain memory).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# The page filename rules of kiss.core.memoryfield.pages (PAGE_NAME_RE,
# DEBRIS_NAMES, is_debris, TOMBSTONES_DIR), repeated here because this file
# cannot import the package on a machine that does not have it yet.
PAGE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
DEBRIS_NAMES = frozenset({".DS_Store", "desktop.ini", "Thumbs.db"})
TOMBSTONES_DIR = ".tombstones"
# A leading ``---`` line, YAML, a closing ``---`` line -- the same shape
# kiss.core.memoryfield.pages.split_frontmatter accepts.
FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
UPDATED_RE = re.compile(r"^updated:[ \t]*(.+?)[ \t\r]*$", re.MULTILINE)
# Written by kiss.core.memoryfield.pages.now_iso.
UPDATED_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def is_page(path: Path) -> bool:
    """Return True when *path* is a memory page and not an index, debris or a symlink.

    Args:
        path: An entry of a memory directory.
    """
    name = path.name
    if name in DEBRIS_NAMES or name.endswith("~") or ".sync-conflict-" in name:
        return False
    if path.suffix != ".md" or not PAGE_NAME_RE.match(path.stem):
        return False
    return not path.is_symlink() and path.is_file()


def parse_stamp(raw: str) -> float | None:
    """Return the epoch seconds of an ``updated`` stamp, or None when it does not parse.

    Args:
        raw: The stamp as written by kiss.core.memoryfield.pages.now_iso,
            YAML quotes allowed.
    """
    try:
        # timezone.utc, not datetime.UTC: the remote's python3 may be 3.10.
        stamp = datetime.strptime(raw.strip("'\""), UPDATED_FORMAT)
        return stamp.replace(tzinfo=timezone.utc).timestamp()  # noqa: UP017
    except ValueError:
        return None


def updated_at(path: Path, text: str) -> float:
    """Return when the page was last written, as seconds since the epoch.

    The ``updated`` field of the frontmatter block is used when it is there
    and parsable; otherwise the file's modification time.  As in
    kiss.core.memoryfield.pages.split_frontmatter, a block with no closing
    ``---`` is not frontmatter.

    Args:
        path: The page file, for its modification time.
        text: The page's full text.
    """
    block = FRONTMATTER_RE.match(text)
    if block:
        match = UPDATED_RE.search(block.group(1))
        if match:
            stamp = parse_stamp(match.group(1))
            if stamp is not None:
                return stamp
    return path.stat().st_mtime


def modified_at(path: Path) -> int:
    """Return the file's modification time in whole seconds.

    Whole seconds because a page that travelled in a tar stream keeps its
    time only to the second, and a copy must compare equal to its original.

    Args:
        path: The file.
    """
    return int(path.stat().st_mtime)


def compare(first: float, second: float, tolerance: float) -> int:
    """Say which of two times is the later: 1 for *first*, -1 for *second*, 0 for neither.

    Times closer together than *tolerance* -- and equal times -- are
    "neither": too close to call.

    Args:
        first: Seconds since the epoch.
        second: Seconds since the epoch.
        tolerance: The difference, in seconds, below which two times count
            as the same moment.
    """
    if abs(first - second) <= tolerance:
        return 0
    return 1 if first > second else -1


def install_file(source: Path, dest: Path) -> None:
    """Put *source* in place as *dest* by an atomic rename, keeping its mtime.

    The temporary file sits in the destination directory (a rename across
    filesystems is not atomic) and does not end in ``.md``, so a memory
    listing meanwhile does not take it for a page; a copy that fails half-way
    takes it away again.

    Args:
        source: The page or tombstone to copy.
        dest: Where it goes.
    """
    staging = dest.with_name(dest.name + ".incoming")
    try:
        shutil.copy2(source, staging)
        os.replace(staging, dest)
    finally:
        if staging.exists():
            staging.unlink()


def is_domain_memory(path: Path) -> bool:
    """Return True when *path* is a domain memory: a sub-directory named like a page.

    A memory's domain memories (the repository memory, for example) are
    flat page directories nested one level down, ``<memory>/<name>/``
    (``kiss.core.memoryfield.tools.MemoryTools``).

    Args:
        path: An entry of a memory directory.
    """
    return bool(PAGE_NAME_RE.match(path.name)) and not path.is_symlink() and path.is_dir()


def tombstones(memory_dir: Path) -> dict[str, tuple[Path, float]]:
    """Map the name of every deleted page of *memory_dir* to its tombstone and deletion time.

    A tombstone is ``.tombstones/<name>`` holding the deletion time as an
    ``updated`` stamp (kiss.core.memoryfield.pages.MemoryDir.delete); one
    that holds something else counts its modification time.  Entries that
    are not named like a page, or are symlinks, are not tombstones.

    Args:
        memory_dir: A memory or domain memory directory.
    """
    found: dict[str, tuple[Path, float]] = {}
    directory = memory_dir / TOMBSTONES_DIR
    if not directory.is_dir() or directory.is_symlink():
        return found
    for path in directory.iterdir():
        if not PAGE_NAME_RE.match(path.name) or path.is_symlink() or not path.is_file():
            continue
        first_line = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        stamp = parse_stamp(first_line.strip())
        found[path.name] = (path, path.stat().st_mtime if stamp is None else stamp)
    return found


def merge(
    source_dir: Path, dest_dir: Path, tolerance: float, prefix: str = ""
) -> tuple[dict[str, int], list[str]]:
    """Fold the pages and tombstones of *source_dir* into *dest_dir*, domain memories included.

    Args:
        source_dir: Pages to merge in.  A directory that does not exist holds
            no pages.
        dest_dir: The memory receiving them; created when missing.
        tolerance: Seconds within which two times count as the same moment
            (see :func:`compare`).
        prefix: Prepended to conflicting page names, ``"<memory>/"`` when
            merging a domain memory.

    Returns:
        The counts of ``added``, ``updated``, ``kept``, ``conflicts`` and
        ``removed`` pages, and the names of the conflicting pages.
    """
    counts = {"added": 0, "updated": 0, "kept": 0, "conflicts": 0, "removed": 0}
    conflicts: list[str] = []
    if not source_dir.is_dir():
        return counts, conflicts
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_tombstones = tombstones(dest_dir)
    for name, (tombstone, _) in list(dest_tombstones.items()):
        if is_page(dest_dir / f"{name}.md"):
            # A page beside its own tombstone: a write interrupted before it
            # could clear the tombstone.  The page is what the memory shows;
            # the tombstone is stale.
            tombstone.unlink()
            del dest_tombstones[name]
    source_tombstones = {
        name: found
        for name, found in tombstones(source_dir).items()
        if not is_page(source_dir / f"{name}.md")
    }
    for source in sorted(source_dir.iterdir()):
        if not prefix and is_domain_memory(source):
            sub_counts, sub_conflicts = merge(
                source, dest_dir / source.name, tolerance, source.name + "/"
            )
            for key, value in sub_counts.items():
                counts[key] += value
            conflicts.extend(sub_conflicts)
            continue
        if not is_page(source):
            continue
        dest = dest_dir / source.name
        source_bytes = source.read_bytes()
        source_time = updated_at(source, source_bytes.decode("utf-8", errors="replace"))
        if not dest.exists():
            deleted = dest_tombstones.get(source.stem)
            if deleted is not None:
                # The destination deleted this page: the page stays out
                # unless it was written again after the deletion.
                verdict = compare(source_time, deleted[1], tolerance)
                if verdict < 0:
                    counts["kept"] += 1
                    continue
                if verdict == 0:
                    counts["conflicts"] += 1
                    conflicts.append(prefix + source.stem)
                    continue
                deleted[0].unlink()
            install_file(source, dest)
            counts["added"] += 1
            continue
        dest_bytes = dest.read_bytes()
        if source_bytes == dest_bytes:
            counts["kept"] += 1
            continue
        dest_time = updated_at(dest, dest_bytes.decode("utf-8", errors="replace"))
        if source_time == dest_time:
            # One side was edited without the agent, which moves the file's
            # time but not the stamp: the copy modified later is the edit.
            verdict = compare(modified_at(source), modified_at(dest), tolerance)
        else:
            verdict = compare(source_time, dest_time, tolerance)
        if verdict > 0:
            install_file(source, dest)
            counts["updated"] += 1
        elif verdict < 0:
            counts["kept"] += 1
        else:
            counts["conflicts"] += 1
            conflicts.append(prefix + source.stem)
    for name, (tombstone, deleted_at) in sorted(source_tombstones.items()):
        dest = dest_dir / f"{name}.md"
        if dest.exists():
            page_time = updated_at(dest, dest.read_bytes().decode("utf-8", errors="replace"))
            verdict = compare(deleted_at, page_time, tolerance)
            if verdict == 0:
                counts["conflicts"] += 1
                conflicts.append(prefix + name)
            elif verdict > 0:
                dest.unlink()
                counts["removed"] += 1
            if verdict <= 0:
                # A page written again after the deletion outlives it; a
                # page too close to call is kept, deletion being the
                # destructive choice.
                continue
        # The deletion travels on, unless the destination knows a later one.
        known = dest_tombstones.get(name)
        if known is None or deleted_at > known[1]:
            install_tombstone(tombstone, dest_dir, name)
    return counts, conflicts


def install_tombstone(tombstone: Path, dest_dir: Path, name: str) -> None:
    """Copy *tombstone* into *dest_dir*'s tombstone directory as *name*.

    A tombstone directory that is a symlink is not one (:func:`tombstones`
    reads none from it), and nothing is written through it either.

    Args:
        tombstone: The tombstone to copy.
        dest_dir: The memory receiving it.
        name: The deleted page's name.
    """
    directory = dest_dir / TOMBSTONES_DIR
    if directory.is_symlink():
        return
    directory.mkdir(exist_ok=True)
    install_file(tombstone, directory / name)


def main(argv: list[str]) -> int:
    """Run the merge named on the command line and print its summary.

    Args:
        argv: ``[--tolerance SECONDS] SOURCE_DIR DEST_DIR``.
    """
    tolerance = 0.0
    if len(argv) == 4 and argv[0] == "--tolerance":
        try:
            tolerance = float(argv[1])
        except ValueError:
            tolerance = -1.0
        argv = argv[2:]
    if len(argv) != 2 or tolerance < 0:
        print(
            f"Usage: {Path(sys.argv[0]).name} [--tolerance SECONDS] SOURCE_DIR DEST_DIR",
            file=sys.stderr,
        )
        return 2
    counts, conflicts = merge(Path(argv[0]), Path(argv[1]), tolerance)
    print(" ".join(f"{key} {value}" for key, value in counts.items()))
    for name in conflicts:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
