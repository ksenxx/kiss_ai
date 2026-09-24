#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge one directory of memory pages into another, newest page wins.

The agent's persistent memory (``kiss.core.memoryfield``) is a flat directory
of Markdown pages, ``<name>.md``, each starting with a YAML frontmatter block
that carries an ``updated`` timestamp.  scripts/sync-memory.sh copies the
pages of one machine next to the memory of the other and runs this to fold
them in.

Usage:
    python3 merge_memory_pages.py SOURCE_DIR DEST_DIR

For every page in ``SOURCE_DIR``:

* a page ``DEST_DIR`` does not have is added;
* a page both have with the same bytes is left alone;
* a page both have with different content is replaced when the source's
  ``updated`` is the newer (a page without a parsable ``updated`` counts its
  file modification time instead); an older source is kept out;
* a page both have, different, with the same timestamp on both sides is a
  conflict: the destination keeps its own copy and the page is named on
  stdout, so a person can decide.

Nothing is ever deleted from ``DEST_DIR``, and no file other than a page is
touched -- in particular not the ``*.sqlite3`` vector index that lives in the
same directory: it is a cache keyed by page content, and the next agent that
opens the memory re-embeds what changed.  Pages are put in place by an
atomic rename, so an agent reading the memory meanwhile sees either the old
page or the new one, never a half-written file.  An agent that rewrites the
very page being merged in the instant between the comparison and the rename
loses that write, the same way it would to a second agent writing the page:
the memory has no page locks, and this script takes no more than an agent
does.

Stdlib-only and self-contained, so it can be copied to a remote machine and
run there with the system ``python3`` before the project's environment exists.

Prints one summary line -- ``added N updated N kept N conflicts N`` -- followed
by the name of every conflicting page, one per line.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# The page filename rules of kiss.core.memoryfield.pages (PAGE_NAME_RE,
# DEBRIS_NAMES, is_debris), repeated here because this file cannot import the
# package on a machine that does not have it yet.
PAGE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
DEBRIS_NAMES = frozenset({".DS_Store", "desktop.ini", "Thumbs.db"})
# A leading ``---`` line, YAML, a closing ``---`` line -- the same shape
# kiss.core.memoryfield.pages.split_frontmatter accepts.
FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
UPDATED_RE = re.compile(r"^updated:[ \t]*(.+?)[ \t\r]*$", re.MULTILINE)
# Written by kiss.core.memoryfield.pages.utc_now_iso.
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


def updated_at(path: Path, text: str) -> float:
    """Return when the page was last written, as seconds since the epoch.

    The ``updated`` field of the frontmatter block is used when it is there
    and parsable (YAML quoting stripped); otherwise the file's modification
    time.  As in kiss.core.memoryfield.pages.split_frontmatter, a block with
    no closing ``---`` is not frontmatter.

    Args:
        path: The page file, for its modification time.
        text: The page's full text.
    """
    block = FRONTMATTER_RE.match(text)
    if block:
        match = UPDATED_RE.search(block.group(1))
        if match:
            raw = match.group(1).strip("'\"")
            try:
                # timezone.utc, not datetime.UTC: the remote's python3 may be 3.10.
                stamp = datetime.strptime(raw, UPDATED_FORMAT).replace(
                    tzinfo=timezone.utc  # noqa: UP017
                )
                return stamp.timestamp()
            except ValueError:
                pass
    return path.stat().st_mtime


def install_page(source: Path, dest: Path) -> None:
    """Put *source* in place as *dest* by an atomic rename, keeping its mtime.

    The temporary file sits in the destination directory (a rename across
    filesystems is not atomic) and does not end in ``.md``, so a memory
    listing meanwhile does not take it for a page; a copy that fails half-way
    takes it away again.

    Args:
        source: The page to copy.
        dest: Where it goes.
    """
    staging = dest.with_name(dest.name + ".incoming")
    try:
        shutil.copy2(source, staging)
        os.replace(staging, dest)
    finally:
        if staging.exists():
            staging.unlink()


def merge(source_dir: Path, dest_dir: Path) -> tuple[dict[str, int], list[str]]:
    """Fold the pages of *source_dir* into *dest_dir*.

    Args:
        source_dir: Pages to merge in.  A directory that does not exist holds
            no pages.
        dest_dir: The memory receiving them; created when missing.

    Returns:
        The counts of ``added``, ``updated``, ``kept`` and ``conflicts`` pages,
        and the names of the conflicting pages.
    """
    counts = {"added": 0, "updated": 0, "kept": 0, "conflicts": 0}
    conflicts: list[str] = []
    if not source_dir.is_dir():
        return counts, conflicts
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if not is_page(source):
            continue
        dest = dest_dir / source.name
        if not dest.exists():
            install_page(source, dest)
            counts["added"] += 1
            continue
        source_bytes = source.read_bytes()
        dest_bytes = dest.read_bytes()
        if source_bytes == dest_bytes:
            counts["kept"] += 1
            continue
        source_time = updated_at(source, source_bytes.decode("utf-8", errors="replace"))
        dest_time = updated_at(dest, dest_bytes.decode("utf-8", errors="replace"))
        if source_time > dest_time:
            install_page(source, dest)
            counts["updated"] += 1
        elif source_time < dest_time:
            counts["kept"] += 1
        else:
            counts["conflicts"] += 1
            conflicts.append(source.stem)
    return counts, conflicts


def main(argv: list[str]) -> int:
    """Run the merge named on the command line and print its summary.

    Args:
        argv: ``[SOURCE_DIR, DEST_DIR]``.
    """
    if len(argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} SOURCE_DIR DEST_DIR", file=sys.stderr)
        return 2
    counts, conflicts = merge(Path(argv[0]), Path(argv[1]))
    print(" ".join(f"{key} {value}" for key, value in counts.items()))
    for name in conflicts:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
