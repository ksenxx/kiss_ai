#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The part of the memory sync that runs on either machine: fingerprints, pack, unpack-and-merge.

A memory (``~/.kiss/memories``) is a flat directory of Markdown pages plus
its domain memories one level down and a ``.tombstones/`` directory per
memory recording deletions (see ``kiss/scripts/merge_memory_pages.py``,
which holds the merge rules this module reuses).  Two machines bring their
memories together without copying everything: each side lists the content
hash of every page and tombstone (:func:`fingerprints`), only the entries
whose hashes differ travel as one gzip'd tar stream (:func:`pack`), and the
receiving side unpacks the stream next to its memory and folds it in with
``merge_memory_pages.merge`` (:func:`unpack_and_merge`), newest page wins.

Stdlib-only and self-contained so it can be shipped to a remote machine as
text and run there with the system ``python3``; ``merge_memory_pages.py``
travels with it (``SHIPPED_SOURCES``, as ``kiss.scripts.sync_db`` ships its
helpers).  Run as a script it takes one of three modes::

    probe DIR              -> JSON {"now": epoch seconds,
                                    "entries": {relative path: sha1}}
    pack DIR               <- JSON list of relative paths on stdin
                           -> gzip'd tar of those entries on stdout
    merge DIR TOLERANCE    <- gzip'd tar on stdin
                           -> JSON {"counts": {...}, "conflicts": [...]}

Relative paths are ``<name>.md``, ``<domain>/<name>.md``,
``.tombstones/<name>`` and ``<domain>/.tombstones/<name>``; nothing else
is ever packed or unpacked, so a stream cannot write outside the memory.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

# The same two primitives ``kiss.core.file_lock`` uses: ``flock`` on POSIX,
# ``msvcrt.locking`` of the first byte on Windows.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]

LOCK_TIMEOUT = 60.0
"""Longest wait for an agent holding a memory's lock (``MemoryDir.MUTATION_LOCK_TIMEOUT``)."""

LOCK_POLL = 0.05
"""Seconds between attempts at a held lock."""

ENTRY_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/)?"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.md|\.tombstones/[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)$"
)
"""A page or tombstone of the memory or of one of its domain memories, as a relative path."""

MERGE_MODULE = "merge_memory_pages"
MERGE_MODULE_PATH = os.path.join("scripts", "merge_memory_pages.py")
"""Where ``merge_memory_pages.py`` lives, relative to the ``kiss`` package."""


def merge_module() -> Any:
    """Return ``kiss.scripts.merge_memory_pages`` wherever this module is running.

    On a remote machine the module's text arrives with this script, bound
    to the ``SHIPPED_SOURCES`` global by the ssh bootstrap; in the checkout
    it is read from the file.  Either way it is built from the text, so it
    needs neither the package on ``sys.path`` nor anything installed.
    """
    shipped: dict[str, bytes] = globals().get("SHIPPED_SOURCES") or {}
    source = shipped.get(MERGE_MODULE)
    if source is None:
        here = os.path.dirname(os.path.abspath(__file__))
        source = Path(here, "..", "..", "..", MERGE_MODULE_PATH).resolve().read_bytes()
    module = types.ModuleType(MERGE_MODULE)
    exec(compile(source, MERGE_MODULE + ".py", "exec"), module.__dict__)
    return module


def fingerprints(memory_dir: Path) -> dict[str, str]:
    """Map every page and tombstone of *memory_dir* (domain memories included) to its content hash.

    Args:
        memory_dir: The memory; a directory that does not exist has no entries.

    Returns:
        ``{relative path: sha1 hex digest}``; only files the merge would
        read are listed (no index, debris, symlink or stray file).  A
        tombstone beside a live page of the same name is left out: it is
        the stale remains of an interrupted deletion, which the merge
        ignores when it sees the page -- sent on its own it would read as
        a deletion.
    """
    rules = merge_module()
    found: dict[str, str] = {}
    if not memory_dir.is_dir():
        return found
    for memory, prefix in [(memory_dir, "")] + [
        (entry, entry.name + "/")
        for entry in sorted(memory_dir.iterdir())
        if rules.is_domain_memory(entry)
    ]:
        for entry in sorted(memory.iterdir()):
            if rules.is_page(entry):
                found[prefix + entry.name] = _digest(entry)
        for name, (tombstone, _) in rules.tombstones(memory).items():
            if f"{prefix}{name}.md" not in found:
                found[f"{prefix}{rules.TOMBSTONES_DIR}/{name}"] = _digest(tombstone)
    return found


def _digest(path: Path) -> str:
    """Return the sha1 hex digest of a file's bytes."""
    return hashlib.sha1(path.read_bytes()).hexdigest()  # noqa: S324 - content id, not security


def differing(mine: dict[str, str], theirs: dict[str, str]) -> set[str]:
    """Return the entries whose content differs between two fingerprint maps, or is on one side.

    Args:
        mine: One machine's :func:`fingerprints`.
        theirs: The other machine's.
    """
    return {key for key in mine.keys() | theirs.keys() if mine.get(key) != theirs.get(key)}


def pack(memory_dir: Path, entries: list[str], out: BinaryIO) -> int:
    """Write the named pages and tombstones of *memory_dir* to *out* as a gzip'd tar.

    An entry that is not a valid relative path, or that is gone by the
    time it is read (a page deleted since the fingerprints were taken),
    is left out; modification times travel, the merge compares them for
    pages that carry the same ``updated`` stamp.

    Args:
        memory_dir: The memory.
        entries: Relative paths, as the keys of :func:`fingerprints`.
        out: Receives the stream.

    Returns:
        The number of entries written.
    """
    written = 0
    with tarfile.open(fileobj=out, mode="w|gz") as archive:
        for entry in sorted(set(entries)):
            path = memory_dir / entry
            if not ENTRY_RE.match(entry) or path.is_symlink() or not path.is_file():
                continue
            archive.add(path, arcname=entry, recursive=False)
            written += 1
    return written


def unpack(stream: BinaryIO, into: Path) -> int:
    """Extract a :func:`pack` stream into *into*, admitting only page and tombstone paths.

    Args:
        stream: The gzip'd tar.
        into: An empty directory.

    Returns:
        The number of entries extracted.
    """
    extracted = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not ENTRY_RE.match(member.name):
                continue
            target = into / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if (
                source is None
            ):  # pragma: no cover - extractfile is None only for non-files, filtered above
                continue
            with open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            os.utime(target, (member.mtime, member.mtime))
            extracted += 1
    return extracted


def unpack_and_merge(stream: BinaryIO, memory_dir: Path, tolerance: float) -> dict[str, Any]:
    """Fold a :func:`pack` stream into *memory_dir*, newest page wins.

    The stream is unpacked into a scratch directory and merged with
    ``merge_memory_pages.merge`` (which puts every page in place by an
    atomic rename inside the memory); the scratch directory is removed.
    The merge holds the mutation lock of the memory and of every domain
    memory it touches (:func:`memory_locks`): an agent writing a page in
    the instant between the merge's comparison and its rename would
    otherwise lose that write.

    Args:
        stream: The gzip'd tar of the other machine's differing entries.
        memory_dir: The memory receiving them; created when missing.
        tolerance: Seconds within which two times count as the same moment
            (the measured clock skew between the machines).

    Returns:
        ``{"received": n, "counts": {added, updated, kept, conflicts,
        removed}, "conflicts": [page names]}``.

    Raises:
        TimeoutError: When a lock is still held after :data:`LOCK_TIMEOUT`.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    rules = merge_module()
    staging = Path(tempfile.mkdtemp(prefix="memory-delta-"))
    try:
        received = unpack(stream, staging)
        domains = [entry.name for entry in staging.iterdir() if rules.is_domain_memory(entry)]
        with memory_locks(memory_dir, domains):
            counts, conflicts = rules.merge(staging, memory_dir, tolerance)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"received": received, "counts": counts, "conflicts": conflicts}


@contextmanager
def memory_locks(memory_dir: Path, domains: list[str]) -> Iterator[None]:
    """Hold the mutation locks of a memory and of the named domain memories for the block.

    The locks are the ones ``kiss.core.memoryfield.pages.MemoryDir`` takes
    around every write and delete (``kiss.core.file_lock``): an exclusive
    lock on ``<memory>.lock`` beside the memory directory and on
    ``<memory>/<domain>.lock`` for a domain memory, ``flock`` on POSIX and
    ``msvcrt.locking`` of the file's first byte on Windows.  An agent never
    holds two of them at once, so taking several here, in a fixed order,
    cannot deadlock with one.

    Args:
        memory_dir: The memory.
        domains: Names of its domain memories to lock as well.

    Raises:
        TimeoutError: When a lock is still held after :data:`LOCK_TIMEOUT`.
    """
    paths = [Path(f"{memory_dir}.lock")] + [memory_dir / f"{name}.lock" for name in sorted(domains)]
    descriptors: list[int] = []
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            descriptors.append(descriptor)
            _lock(descriptor, path)
        yield
    finally:
        for descriptor in reversed(descriptors):
            _unlock(descriptor)
            os.close(descriptor)


def _lock(descriptor: int, path: Path) -> None:
    """Take the exclusive lock on *descriptor*, polling for up to :data:`LOCK_TIMEOUT` seconds."""
    deadline = time.monotonic() + LOCK_TIMEOUT
    while not _try_lock(descriptor):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{path} still locked by another process after {LOCK_TIMEOUT:g}s")
        time.sleep(LOCK_POLL)


def _try_lock(descriptor: int) -> bool:
    """Probe the lock once without waiting; True when it is now held."""
    if fcntl is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True
    return True  # pragma: no cover - a platform with neither primitive has no lock to take


def _unlock(descriptor: int) -> None:
    """Release the lock :func:`_try_lock` took on *descriptor*."""
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]


def main(argv: list[str]) -> int:
    """Run one mode (see the module docstring) with binary stdin/stdout.

    Args:
        argv: ``[mode, memory_dir, ...]``; ``~`` in the directory is expanded here.

    Returns:
        The process exit status.
    """
    if len(argv) < 2:
        print("usage: memory_delta.py probe|pack|merge DIR [TOLERANCE]", file=sys.stderr)
        return 2
    mode, memory_dir = argv[0], Path(os.path.expanduser(argv[1]))
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    if mode == "probe":
        report = {"now": time.time(), "entries": fingerprints(memory_dir)}
        stdout.write(json.dumps(report).encode("utf-8"))
    elif mode == "pack":
        entries = json.loads(stdin.read().decode("utf-8"))
        pack(memory_dir, entries, stdout)
    elif mode == "merge":
        tolerance = float(argv[2]) if len(argv) > 2 else 0.0
        result = unpack_and_merge(io.BytesIO(stdin.read()), memory_dir, tolerance)
        stdout.write(json.dumps(result).encode("utf-8"))
    else:
        print(f"memory_delta.py: unknown mode {mode!r}", file=sys.stderr)
        return 2
    stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
