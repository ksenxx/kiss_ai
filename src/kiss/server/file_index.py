# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Persistent, self-refreshing file index behind the ``@``-mention picker.

The picker used to walk a tab's working directory on the first ``@``
(seconds for a large tree) and re-rank the whole list on every
keystroke (hundreds of milliseconds for a home directory).  This module
replaces both with an index that is

* **built once per root and persisted** under ``$KISS_HOME/file-index``
  so a daemon restart reloads it instead of rescanning;
* **refreshed by directory mtime**: a rescan ``stat``s every known
  directory and re-lists only those whose mtime changed or was recent when
  last read, so keeping a 100k-entry tree fresh costs a fraction of a second;
* **ordered once, at build time**, by how likely an entry is to be what
  a user wants to hand to an agent (source before bulk data, files
  outside generated-run containers first, shallow before deep), so a
  query is a C-speed substring scan that stops at the first
  :data:`MATCH_CAP` hits instead of a Python sort of every path;
* **shared by every tab under the home directory**: the home index
  serves any work_dir below it through a cached :class:`FileView`, and
  roots outside it (or inside a skipped directory) get an index of
  their own;
* **home-wide**: a :class:`Picker` answers with ``./path`` mentions
  from the work dir followed by ``~/path`` mentions from the rest of
  the home directory, so any file under ``~`` can be handed to an
  agent from any tab.

Scan rules: dot-directories, :data:`JUNK_DIR_NAMES` and the non-glob
entries of every ``.gitignore`` met on the way down (nested repositories
included) are skipped, depth is capped at :data:`MAX_DEPTH`, and bulk
data dumps (:func:`_bulk_data_dirs`) are collapsed to their ``dir/``
entry.  The home index also skips the operating system's application
state tree (Windows-hidden directories such as ``AppData``; ``~/Library``
on macOS, see :data:`_HOME_RULES`), and a directory the operating
system refuses to let the daemon read (macOS privacy protection, see
:data:`_refused`) is left out for the rest of the process instead of
stalling every scan.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import posixpath
import queue
import stat
import sys
import threading
import time
from array import array
from bisect import bisect_right
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from kiss.core.config import kiss_home
from kiss.core.utils import atomic_write_text, is_root_dir
from kiss.server.helpers import SUGGESTION_LIMIT

logger = logging.getLogger(__name__)

# Directories deeper than this below the index root are listed as ``dir/``
# entries but not descended into.
MAX_DEPTH = 12

# Hard cap on the number of entries one index holds.  The picker filters
# by substring, so a too-small cap silently hides whole subtrees.
_SCAN_FILES_CAP = 1_000_000

# Directory names that never hold anything worth mentioning to an agent,
# whether or not a ``.gitignore`` says so.
JUNK_DIR_NAMES = frozenset({"node_modules", "__pycache__", "venv", "site-packages"})

# ``stat.FILE_ATTRIBUTE_HIDDEN``: Windows' counterpart of a dot-directory.
_FILE_ATTRIBUTE_HIDDEN = 0x2

# Suffixes of machine-generated data: run results, logs, traces, dumps,
# serialized tensors.  Data files rank after everything else and a
# directory made almost entirely of them is collapsed (``_bulk_data_dirs``),
# so the list errs towards data formats: a suffix missing from it merely
# keeps a data dump in the picker, whereas listing a source suffix would
# hide code.
_DATA_SUFFIXES = frozenset({
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".log", ".txt", ".out",
    ".err", ".csv", ".tsv", ".parquet", ".arrow", ".npy", ".npz", ".pkl",
    ".pickle", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".db",
    ".sqlite", ".sqlite3", ".patch", ".diff",
})
# A directory is a bulk data dump when its subtree (minus bulk children)
# holds at least this many files and at least this fraction of them carry a
# data suffix.
_BULK_DATA_MIN_FILES = 200
_BULK_DATA_MIN_FRACTION = 0.9

# A directory with at least this many immediate subdirectories is a container
# of generated runs (``artifacts/<run_id>/``, ``jobs/<date>/``): source trees
# rarely have more than a couple of dozen sibling packages, while such
# containers hold hundreds of near-identical copies of the same files.
WIDE_DIR_MIN_CHILDREN = 50

# Matches collected (in index order) before ranking by match position.
# Beyond this many hits the query is too unspecific for position to matter.
MATCH_CAP = 1000

# Slots of a picker reply kept for matches outside the work dir, so the
# rest of the home directory stays reachable when the work dir alone
# would fill the list.
HOME_SLOTS = 5

# A view served more than this many seconds after its index was built
# schedules a background refresh, so the next keystroke sees the change.
STALE_AFTER = 60.0

# Persisted listing format version; bump when the JSON layout changes.
_CACHE_VERSION = 1

# A directory or ``.gitignore`` whose mtime is less than this many
# nanoseconds old can change again without its mtime moving: file systems
# stamp a change with a clock that ticks every few milliseconds, or every
# two seconds (FAT); the third second covers that clock running behind
# ``time.time_ns()`` (the kernel's tick, a file server's clock).  What a
# scan reads from one is recorded under an mtime it does not have, so the
# next scan reads it again.
_RACY_NS = 3_000_000_000

# Per directory: (mtime_ns, sorted file names, sorted subdirectory names).
Listings = dict[str, tuple[int, list[str], list[str]]]


def _bulk_data_dirs(own_counts: dict[str, list[int]]) -> set[str]:
    """Pick the directories whose subtree is almost entirely data files.

    Directories are folded bottom-up: a bulk subtree is recorded and NOT
    added to its parent's totals, so ``bench/`` with five scripts and a
    250-file ``bench/results/`` dump yields ``{"bench/results"}`` — the
    scripts stay visible.  Small dumps spread over many subdirectories
    still add up at the first ancestor that crosses the threshold.

    Args:
        own_counts: Map from relative directory path (``""`` for the
            root) to ``[files, data_files]`` directly inside it, with
            every parent inserted before its children (the scan's
            breadth-first order); the parent of every non-root entry is
            itself an entry.

    Returns:
        Relative paths of directories with at least ``_BULK_DATA_MIN_FILES``
        files of which at least ``_BULK_DATA_MIN_FRACTION`` are data files.
        The root is never returned: a repository that is itself a data set
        should still be browsable.
    """
    bulk: set[str] = set()
    totals = {d: list(c) for d, c in own_counts.items()}
    for d in reversed(totals):  # children before parents
        if d == "":
            continue
        files, data_files = totals[d]
        if files >= _BULK_DATA_MIN_FILES and data_files >= _BULK_DATA_MIN_FRACTION * files:
            bulk.add(d)
            continue
        parent = totals[d.rpartition("/")[0]]
        parent[0] += files
        parent[1] += data_files
    return bulk


def _parse_gitignore(text: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Extract the non-glob entries of a ``.gitignore``.

    Following gitignore semantics, an entry containing a slash anywhere
    other than at its end is anchored to the ignore file's directory
    (``/build``, ``src/generated``), while a bare name matches at any
    depth below it — ``build/`` only directories, ``secret.txt`` files
    and directories alike.

    Args:
        text: Contents of the ``.gitignore`` file.

    Returns:
        ``(names, dir_names, anchored, anchored_dirs)`` — bare names to
        skip at any depth, bare names to skip at any depth when they are
        directories, and paths (relative to the ignore file's directory)
        to skip at their exact location only, whatever they are
        (``/build``) or only when they are directories (``/build/``).
        Negations, comments and glob entries are ignored.
    """
    names: set[str] = set()
    dir_names: set[str] = set()
    anchored: set[str] = set()
    anchored_dirs: set[str] = set()
    for raw_line in text.splitlines():
        # More lenient than git, which keeps leading whitespace in the
        # pattern: an indented entry is taken for the name it was meant as.
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        # Git's own test (``is_glob_special``): a bracket expression, closed
        # or not, and a backslash escape never mean the name as written.
        if "*" in line or "?" in line or "[" in line or "\\" in line:
            continue
        entry = line.rstrip("/")
        if "/" in entry:
            (anchored_dirs if line.endswith("/") else anchored).add(entry.lstrip("/"))
        elif line.endswith("/"):
            dir_names.add(entry)
        else:
            names.add(entry)
    return names, dir_names, anchored, anchored_dirs


def _is_hidden_dir(entry: os.DirEntry[str]) -> bool:
    """Whether directory *entry* carries the Windows ``hidden`` attribute.

    Windows marks ``AppData`` (caches, program state: most of a home
    directory's files) and similar folders hidden the way POSIX tools
    dot-prefix them.  ``entry.stat(follow_symlinks=False)`` is answered
    from the listing on Windows, so this costs no system call; on other
    platforms the answer is always ``False``.
    """
    if sys.platform != "win32":
        return False
    return bool(entry.stat(follow_symlinks=False).st_file_attributes & _FILE_ATTRIBUTE_HIDDEN)


def _hidden_dir_on_path(base: str, parts: list[str]) -> bool:
    """Whether a directory from *base* down through *parts* is Windows-hidden."""
    if sys.platform != "win32":
        return False
    path = base
    for part in parts:
        path = path + os.sep + part
        try:
            if os.stat(path).st_file_attributes & _FILE_ATTRIBUTE_HIDDEN:
                return True
        except (OSError, ValueError):  # unreadable, or a NUL byte in the name
            return False
    return False


def _list_dir(abs_dir: str) -> tuple[list[str], list[str]] | None:
    """List *abs_dir* as ``(sorted file names, sorted subdirectory names)``.

    Symbolic links and Windows junctions count as files (they are listed
    but never followed: ``Application Data`` -> ``AppData\\Local`` would
    otherwise recurse to :data:`MAX_DEPTH`); Windows-hidden directories
    are dropped like dot-directories; a directory that cannot be listed
    gives ``None``, and one the OS refused (see :data:`_refused`) is
    answered so without asking again.  ``is_dir`` reads the type the kernel
    returned with the entry, so no per-entry ``stat`` system call happens.
    """
    files: list[str] = []
    dirs: list[str] = []
    if abs_dir in _refused:
        return None
    try:
        with os.scandir(abs_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False) or entry.is_junction():
                    files.append(entry.name)
                elif not _is_hidden_dir(entry):
                    dirs.append(entry.name)
    except OSError as exc:
        if exc.errno in _REFUSED_ERRNOS:
            _refused.add(abs_dir)
            logger.info(
                "%s cannot be listed (%s); left out of the file picker until restart",
                abs_dir, exc.strerror,
            )
        else:
            logger.debug("cannot list %s", abs_dir, exc_info=True)
        return None
    return sorted(files), sorted(dirs)


# Extensions (without the dot, lower- and upper-case) of ``_DATA_SUFFIXES``,
# so classifying a name is one ``rpartition`` and one set lookup.
_DATA_EXTS = frozenset(
    ext for s in _DATA_SUFFIXES for ext in (s[1:], s[1:].upper())
)


def _encode(text: str) -> bytes:
    """UTF-8 encode *text* for the search blob; every string has an encoding.

    A surrogate is encoded as it is: the escaped byte of an undecodable
    file name, and the lone one a client's query may carry, which a
    strict encoding refuses.
    """
    return text.encode("utf-8", "surrogatepass")


def _end_dist(path: str, query: str) -> int:
    """Distance from the last occurrence of lowercase *query* to the end of *path*.

    Zero for an empty query, so an unspecific query keeps index order.
    """
    if not query:
        return 0
    return len(path) - path.lower().rfind(query) - len(query)


# Entries of one directory split by rank class: 0 = source/dirs, 1 = data
# files and dotfiles, 2 and 3 = the same inside a wide container.
_Buckets = tuple[list[str], list[str], list[str], list[str]]
# Skip rules in force below a directory: gitignore bare names (any entry),
# bare names that only skip directories, root-relative anchored paths (any
# entry) and root-relative anchored paths that only skip directories.
_Rules = tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]
_ROOT_RULES: _Rules = (frozenset({".git"}), frozenset(), frozenset(), frozenset())

# Skip rules of the home root.  macOS keeps application state in
# ``~/Library`` (caches, sandbox containers, the mail, message and photo
# stores: a third of a home directory's entries, most of them behind the
# privacy protection described at ``_refused``), the way Windows keeps it
# in the hidden ``AppData`` that ``_is_hidden_dir`` already drops; Finder
# hides the folder by default.  A work dir inside it (iCloud Drive lives
# at ``~/Library/Mobile Documents``) is a root of its own, as a work dir
# inside any skipped directory is (``FileIndexRegistry.root_for``).
_HOME_SKIPPED_DIRS = frozenset({"Library"}) if sys.platform == "darwin" else frozenset()
_HOME_RULES: _Rules = (_ROOT_RULES[0], _ROOT_RULES[1], _HOME_SKIPPED_DIRS, frozenset())

# Absolute paths of the directories this process may not read.  macOS
# answers a daemon without "Files and Folders" permission (a launchd
# agent cannot show the permission prompt) with a five-second wait and
# ``EINTR`` for each ``scandir`` of a protected directory (``~/Desktop``,
# ``~/Downloads``, ``~/Library/Messages``, ...) and holds an ``open`` of
# a file below one for as long as the prompt stays unanswered -- for
# ever, on the only index worker thread.  A directory refused once is
# refused until the daemon restarts (so a scan pays each wait once) and
# is never persisted, so a daemon granted the permission lists it on its
# next start.  Only the index worker thread writes the set.
_refused: set[str] = set()
_REFUSED_ERRNOS = frozenset({errno.EINTR, errno.EPERM, errno.EACCES})

# Seconds a ``.gitignore`` read may take before its directory counts as
# refused: the OS answers a permitted read in microseconds and a refused
# ``scandir`` in five seconds.
_READ_DEADLINE = 5.0


class _DerivedDir(NamedTuple):
    """What the scan derived from one directory's listing.

    Valid for reuse while the directory's ``mtime``, the wide flag, the
    rules its parent passed down and the mtime of its own ``.gitignore``
    are unchanged.  Rule objects are passed down by identity, so an
    unchanged tree compares them with ``is`` all the way down.
    """

    mtime: int
    in_wide: bool
    rules: _Rules
    gitignore_mtime: int
    child_rules: _Rules
    buckets: _Buckets
    kept_dirs: list[str]
    counts: list[int]
    wide: bool


def _gitignore_mtime(abs_dir: str) -> int:
    """mtime of ``abs_dir/.gitignore`` in ns, ``-2`` (no file's mtime) when it cannot be read."""
    try:
        return os.stat(abs_dir + "/.gitignore").st_mtime_ns
    except OSError:
        return -2


# Bytes of the longest ``.gitignore`` that is read; hand-written ones stay far below.
_GITIGNORE_MAX_BYTES = 1 << 20

# How a ``.gitignore`` is opened: without waiting for the other end of a pipe, and
# without taking a terminal it may name as the daemon's own.
_READ_ONLY = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
    | getattr(os, "O_BINARY", 0)
)


def _read_file_into(path: str, out: list[str]) -> None:
    """Append the text of the ignore file *path* to *out*; nothing when it cannot be read.

    Nor when it is no ignore file.  The name may be a link to anything,
    so only a regular file is read (a device's bytes never end, a pipe's
    may never come), and of that at most :data:`_GITIGNORE_MAX_BYTES`: a
    longer one is left unread.
    """
    try:
        fd = os.open(path, _READ_ONLY)
    except OSError:
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
        with open(fd, "rb", closefd=False) as file:
            data = file.read(_GITIGNORE_MAX_BYTES + 1)
        if len(data) <= _GITIGNORE_MAX_BYTES:
            out.append(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError):
        pass
    finally:
        os.close(fd)


def _read_gitignore(abs_dir: str) -> str | None:
    """Return the text of ``abs_dir/.gitignore``, or ``None`` if it cannot be read.

    The read runs on a helper thread and is given :data:`_READ_DEADLINE`
    seconds.  A directory listed from the persisted listings was never
    opened by this process, and if the OS protects it (see
    :data:`_refused`) the ``open`` never returns; the deadline turns
    that into a refusal of the directory, and the helper thread is left
    behind blocked (at most one per refused directory).
    """
    if abs_dir in _refused:
        return None
    text: list[str] = []
    reader = threading.Thread(
        target=_read_file_into, args=(abs_dir + "/.gitignore", text), daemon=True,
    )
    reader.start()
    reader.join(_READ_DEADLINE)
    if reader.is_alive():
        _refused.add(abs_dir)
        logger.warning("%s cannot be read (permission prompt pending); skipped", abs_dir)
        return None
    return text[0] if text else None


def _child_rules(abs_dir: str, rel: str, rules: _Rules) -> _Rules | None:
    """Merge ``abs_dir/.gitignore`` into *rules* for the directories below.

    ``None`` when the file cannot be read.
    """
    text = _read_gitignore(abs_dir)
    if text is None:
        return None
    names, dir_names, anchored, anchored_dirs = _parse_gitignore(text)
    if not names and not dir_names and not anchored and not anchored_dirs:
        return rules
    skip_names, skip_dir_names, skip_paths, skip_dir_paths = rules
    return (
        skip_names | names,
        skip_dir_names | dir_names,
        skip_paths | {posixpath.join(rel, a) for a in anchored},
        skip_dir_paths | {posixpath.join(rel, a) for a in anchored_dirs},
    )


def _anchored_here(paths: frozenset[str], rel: str) -> set[str]:
    """The names of the anchored *paths* that point into directory *rel* itself."""
    return {p.rpartition("/")[2] for p in paths if p.rpartition("/")[0] == rel}


def _derive(
    rel: str,
    files: list[str],
    subdirs: list[str],
    in_wide: bool,
    rules: _Rules,
) -> tuple[_Buckets, list[str], list[int], bool]:
    """Apply the skip rules to one directory listing and bucket its entries.

    Args:
        rel: Directory path relative to the root (``""`` for the root).
        files: Sorted file names in the directory.
        subdirs: Sorted subdirectory names in the directory.
        in_wide: Whether an ancestor is a wide container.
        rules: Names skipped at any depth and root-relative paths skipped.

    Returns:
        ``(buckets, kept_dirs, [files, data_files], wide)`` — see
        ``_Buckets``; ``wide`` tells whether entries below this directory
        sit inside a wide container (the root itself is never one: a
        repository with many top-level packages is not a run dump).
    """
    prefix = rel + "/" if rel else ""
    skip, skip_dirs, skip_paths, skip_dir_paths = rules
    if skip_paths:
        skip = skip | _anchored_here(skip_paths, rel)
    if skip_dir_paths:
        skip_dirs = skip_dirs | _anchored_here(skip_dir_paths, rel)
    kept_files = [f for f in files if f not in skip]
    kept_dirs = [
        d for d in subdirs
        if d not in skip and d not in skip_dirs and d[0] != "." and d not in JUNK_DIR_NAMES
    ]
    wide = in_wide or (rel != "" and len(kept_dirs) >= WIDE_DIR_MIN_CHILDREN)
    buckets: _Buckets = ([], [], [], [])
    base = 2 if wide else 0
    plain, data = buckets[base], buckets[base + 1]
    exts = _DATA_EXTS
    for f in kept_files:
        # Dotfiles and data files rank last; inlined, this loop runs once
        # per file in the tree.  A name without a dot has no suffix.
        _, dot, ext = f.rpartition(".")
        (data if f[0] == "." or (dot and ext in exts) else plain).append(prefix + f)
    n_data = len(data)
    plain.extend(prefix + d + "/" for d in kept_dirs)
    return buckets, kept_dirs, [len(kept_files), n_data], wide


class FileView:
    """The entries of one :class:`FileIndex` below one directory.

    Paths are relative to the view's directory and keep the index's
    priority order: entries outside wide containers first (``wide_from``
    is the index of the first entry inside one), each class shallow
    first.  Searching is a case-insensitive substring scan of a single
    newline-joined UTF-8 blob (C speed, one byte per ASCII character
    however exotic a stray file name is) that stops after
    :data:`MATCH_CAP` hits; the hits are then ordered by wide-container
    membership, then by how close to the end of the path the query
    occurs, ties keeping index order.
    """

    def __init__(self, paths: list[str], wide_from: int | None = None) -> None:
        """Build the search blob for *paths* (index order, view-relative).

        Args:
            paths: The entries, non-wide ones first.
            wide_from: Index of the first entry inside a wide container;
                ``None`` when there is none.
        """
        self.paths = paths
        self.wide_from = len(paths) if wide_from is None else wide_from
        joined = "\n".join(paths)
        if joined.isascii():
            # Lowercasing ASCII keeps every length, so offsets come free.
            self._blob = ("\n" + joined.lower() + "\n").encode("ascii")
            lengths = [len(p) for p in paths]
        else:
            parts = [_encode(p.lower()) for p in paths]
            self._blob = b"\n" + b"\n".join(parts) + b"\n"
            lengths = [len(part) for part in parts]
        starts = array("Q")
        pos = 1
        for n in lengths:
            starts.append(pos)
            pos += n + 1
        starts.append(pos)
        self._starts = starts

    def __contains__(self, path: str) -> bool:
        """Return whether *path* (exact spelling) is an entry of the view."""
        needle = b"\n" + _encode(path.lower()) + b"\n"
        pos = self._blob.find(needle)
        while pos >= 0:
            if self.paths[bisect_right(self._starts, pos + 1) - 1] == path:
                return True
            pos = self._blob.find(needle, pos + 1)
        return False

    def _hits(self, query: bytes, want: int) -> list[tuple[int, int, int]]:
        """``(wide, end distance, index)`` of the entries containing *query*.

        Entries are scanned in index order and at most :data:`MATCH_CAP`
        are collected; beyond that the query is too unspecific for the
        match position to matter.  The scan also stops as soon as *want*
        non-wide hits have distance 0 (the query ends the path, e.g. a
        file name typed in full), or as soon as it reaches the wide
        region holding *want* non-wide hits already: no later entry can
        outrank those, since ties are broken by index order.  The
        distance is measured in the blob, so no per-hit lowercasing.
        """
        wide_from = self.wide_from
        if not query:
            return [(int(i >= wide_from), 0, i) for i in range(min(len(self.paths), MATCH_CAP))]
        blob, starts, n = self._blob, self._starts, len(query)
        out: list[tuple[int, int, int]] = []
        exact = 0
        pos = 0
        while len(out) < MATCH_CAP and exact < want:
            pos = blob.find(query, pos)
            if pos < 0:
                break
            i = bisect_right(starts, pos) - 1
            # Only on reaching the wide region, while ``out`` holds non-wide
            # hits alone: after that the wide hits compete among themselves.
            if i >= wide_from and len(out) >= want and not out[-1][0]:
                break
            end = starts[i + 1] - 1
            dist = end - blob.rfind(query, pos, end) - n
            if dist == 0 and i < wide_from:
                exact += 1
            out.append((int(i >= wide_from), dist, i))
            pos = end + 1
        return out

    def search(
        self,
        query: str,
        usage: dict[str, int],
        limit: int = SUGGESTION_LIMIT,
    ) -> list[dict[str, str]]:
        """Rank the view's entries for the picker.

        Entries the user has mentioned before (``usage`` count > 0) come
        first as ``frequent`` items ordered by match position, recency
        (later ``usage`` keys are more recent) and count; the remaining
        matches follow as ``file`` items ordered by match position and
        then index priority.

        Args:
            query: Substring to look for, matched case-insensitively
                anywhere in the path.  Whitespace never occurs in an
                ``@``-mention query; a query containing a newline
                matches nothing.
            usage: Mention counts keyed by view-relative path, oldest
                first (see ``_load_file_usage``).
            limit: Maximum number of items returned.

        Returns:
            Ranked ``{"type": "frequent" | "file", "text": path}`` dicts.
        """
        q = query.lower()
        if "\n" in q:
            return []
        n_usage = len(usage)
        frequent = sorted(
            (_end_dist(p, q), n_usage - rank, -count, p)
            for rank, (p, count) in enumerate(usage.items())
            if count > 0 and q in p.lower() and p in self
        )
        out = [{"type": "frequent", "text": item[3]} for item in frequent[:limit]]
        if len(out) >= limit:
            return out
        seen = {item[3] for item in frequent}
        hits = sorted(self._hits(_encode(q), limit + len(seen)))
        for _, _, i in hits:
            p = self.paths[i]
            if p in seen:
                continue
            out.append({"type": "file", "text": p})
            if len(out) >= limit:
                break
        return out


class FileIndex:
    """Every mentionable entry below one root, in picker priority order.

    Build one with :meth:`scan`; instances are immutable afterwards (a
    refresh produces a new index) apart from the lazily cached
    :meth:`view` objects.

    Attributes:
        paths: Relative entries in priority order; directories end with
            ``/``.
        dirs: Relative paths (no trailing ``/``) of every directory the
            scan descended into and did not collapse — the work dirs
            this index can serve.
        listings: Raw per-directory listings keyed by relative path
            (``""`` for the root), the input of the next mtime-pruned
            rescan and what gets persisted.
        relisted: Whether the scan read a listing that differs from the
            cached one (so the persisted copy is stale).
        wide_from: Index in ``paths`` of the first entry inside a wide
            container (``len(paths)`` when there is none).
        built_at: ``time.monotonic()`` when the scan started — anything
            changed after that instant may be missing.  ``-inf`` for an
            index built from persisted listings alone (``trusted``), which
            verified nothing: it is stale from birth, so every job queued
            for its root rescans and the registry asks for a rescan
            whenever it serves it.
    """

    def __init__(
        self,
        paths: list[str],
        dirs: frozenset[str],
        listings: Listings,
        derived: dict[str, _DerivedDir],
        relisted: bool,
        wide_from: int,
    ) -> None:
        """Wrap the results of :meth:`scan`; see the class attributes."""
        self.paths = paths
        self.wide_from = wide_from
        self.dirs = dirs
        self.listings = listings
        self._derived = derived
        self.relisted = relisted
        self.built_at = time.monotonic()
        self._views: dict[tuple[str, bool], FileView] = {}
        self._served: set[tuple[str, bool]] = set()
        self._views_lock = threading.Lock()

    @classmethod
    def scan(
        cls,
        root: str,
        listings: Listings | None = None,
        previous: FileIndex | None = None,
        rules: _Rules = _ROOT_RULES,
        trusted: bool = False,
    ) -> FileIndex:
        """Walk *root* and build an index, reusing what has not changed.

        Directories are visited breadth-first.  One whose mtime equals
        the cached one reuses the cached names without a ``scandir``;
        when in addition the skip rules in force at that directory are
        the ones *previous* saw, its derived entries are reused verbatim,
        so a rescan of an unchanged tree is one ``stat`` per directory
        plus list concatenation.  Skip rules (dot-directories,
        :data:`JUNK_DIR_NAMES`, ``.gitignore`` entries accumulated on the
        way down) apply to the listed names, so a rule change never
        needs a re-listing.  A directory or ``.gitignore`` read less than
        :data:`_RACY_NS` after its last change is read again by the next
        scan: a change within the same clock tick leaves the mtime alone.

        A directory the OS refuses to let this process read (see
        :data:`_refused`), or whose listing fails for now, is indexed as its
        bare ``dir/`` entry: nothing below it, and no listing of it is kept.

        Args:
            root: Absolute directory to index.
            listings: Persisted listings of the same root, used when
                *previous* is ``None``.
            previous: The index this scan refreshes; its listings and
                derived per-directory results are reused where valid.
            rules: Skip rules in force at *root* (:data:`_HOME_RULES`
                for the home directory).
            trusted: Build from *listings* alone, without touching the
                file system: every cached directory counts as unchanged,
                one missing from the cache is left out, and ``.gitignore``
                files are not read (their directories are marked so the
                next scan derives their rules).  A daemon that just
                started serves this index while its first real scan,
                which may wait on the OS, runs behind it.

        Returns:
            A new index; ``relisted`` tells whether any listing changed.
        """
        # Not wrapped in gc.disable() for the walk's ~15% saving: the
        # switch is process-wide, and a cold scan of a large tree (up to
        # _SCAN_FILES_CAP entries, minutes on a network file system) would
        # stop every other thread's cyclic garbage from being collected.
        started = time.monotonic()
        index = cls._scan(root, listings or {}, previous, rules, trusted)
        index.built_at = float("-inf") if trusted else started
        return index

    @classmethod
    def _scan(
        cls,
        root: str,
        listings: Listings,
        previous: FileIndex | None,
        root_rules: _Rules,
        trusted: bool,
    ) -> FileIndex:
        """Body of :meth:`scan`; see there."""
        old = previous.listings if previous is not None else listings
        old_derived = previous._derived if previous is not None else {}
        new: Listings = {}
        derived: dict[str, _DerivedDir] = {}
        relisted = False
        own_counts: dict[str, list[int]] = {}
        groups: list[tuple[str, _Buckets]] = []
        total = 0
        # rel_dir, depth, inside a wide container, skip rules
        pending: deque[tuple[str, int, bool, _Rules]] = deque([("", 0, False, root_rules)])
        stat = os.stat
        while pending and total < _SCAN_FILES_CAP:
            rel, depth, in_wide, rules = pending.popleft()
            abs_dir = os.path.join(root, *rel.split("/")) if rel else root
            # A refused directory (see ``_refused``) contributes nothing of
            # its own -- its parent's listing already holds its ``dir/``
            # entry -- and keeps no listing: neither a later scan nor the
            # persisted cache may take its emptiness for its contents.
            # Checked before any cached listing or derived entry is
            # reused, so a refusal recorded by another scan holds here.
            # Dropping a listing it had changes what is persisted.
            cached = old.get(rel)
            if abs_dir in _refused:
                relisted = relisted or cached is not None
                continue
            if trusted:
                if cached is None:
                    continue
                mtime = cached[0]
            else:
                try:
                    mtime = stat(abs_dir).st_mtime_ns
                except OSError:
                    continue
            d = old_derived.get(rel)
            if cached is not None and cached[0] == mtime:
                new[rel] = cached
                files, subdirs = cached[1], cached[2]
                # The listing is the same, so whether it has a .gitignore
                # is already recorded in the derived entry.
                has_gitignore = (
                    d.gitignore_mtime != -1 if d is not None else ".gitignore" in files
                )
            else:
                # Asked before the listing is read; see ``_RACY_NS``.
                if time.time_ns() - mtime < _RACY_NS:
                    mtime -= 1
                listing = _list_dir(abs_dir)
                if listing is None:
                    # Refused, or not listable for now (no descriptor left,
                    # an I/O error): an empty listing would stand for its
                    # contents.  Dropping the cached listing it had changes
                    # what is persisted; one never cached changes nothing.
                    relisted = relisted or cached is not None
                    continue
                files, subdirs = listing
                # A fresh listing is derived afresh: under ``mtime - 1`` the
                # previous scan's entry could pass for current.
                d = None
                new[rel] = (mtime, files, subdirs)
                # A racy listing found unchanged is persisted already.
                relisted = relisted or new[rel] != cached
                has_gitignore = ".gitignore" in files
            # A trusted build records an unread .gitignore as -2, an mtime
            # no file has, so the scan that follows derives its rules; so
            # does a scan that reads a racy one (``_RACY_NS``).
            if not has_gitignore:
                gitignore_mtime = -1
            elif trusted:
                gitignore_mtime = -2
            else:
                gitignore_mtime = _gitignore_mtime(abs_dir)
                if time.time_ns() - gitignore_mtime < _RACY_NS:
                    gitignore_mtime, d = -2, None
            if (
                d is None
                or d.mtime != mtime
                or d.in_wide != in_wide
                or d.rules != rules
                or d.gitignore_mtime != gitignore_mtime
            ):
                child_rules = rules
                if has_gitignore and not trusted:
                    merged = _child_rules(abs_dir, rel, rules)
                    if abs_dir in _refused:
                        # Reading its .gitignore was refused, so nothing
                        # below it can be read either: drop its listing.
                        del new[rel]
                        relisted = True
                        continue
                    if merged is None:
                        # Not readable now (no permission, a link without a
                        # target), and no mtime tells when it will be: the
                        # next scan tries again.
                        gitignore_mtime = -2
                    else:
                        child_rules = merged
                d = _DerivedDir(
                    mtime, in_wide, rules, gitignore_mtime, child_rules,
                    *_derive(rel, files, subdirs, in_wide, child_rules),
                )
            derived[rel] = d
            kept_dirs = d.kept_dirs
            own_counts[rel] = d.counts
            groups.append((rel, d.buckets))
            total += d.counts[0] + len(kept_dirs)
            if kept_dirs and depth < MAX_DEPTH:
                prefix = rel + "/" if rel else ""
                wide, child_rules = d.wide, d.child_rules
                for name in kept_dirs:
                    pending.append((prefix + name, depth + 1, wide, child_rules))
        bulk = _bulk_data_dirs(own_counts)
        out: _Buckets = ([], [], [], [])
        dirs: set[str] = set()
        dropped: set[str] = set()
        # Breadth-first order lists every parent before its children, so
        # "inside a bulk dump" is inherited from the parent in O(1).
        for rel, buckets in groups:
            if rel in bulk or rel.rpartition("/")[0] in dropped:
                dropped.add(rel)
                continue
            dirs.add(rel)
            for target, bucket in zip(out, buckets, strict=True):
                target.extend(bucket)
        # Class order, and within a class the breadth-first order (shallow
        # first, alphabetical within a directory): what the picker shows
        # for an empty query.
        paths = out[0] + out[1] + out[2] + out[3]
        del paths[_SCAN_FILES_CAP:]
        wide_from = min(len(out[0]) + len(out[1]), len(paths))
        return cls(paths, frozenset(dirs), new, derived, relisted, wide_from)

    @classmethod
    def empty(cls) -> FileIndex:
        """Return an index with no entries (a failed scan)."""
        return cls([], frozenset({""}), {}, {}, False, 0)

    def view_keys(self) -> list[tuple[str, bool]]:
        """Return the ``(sub_dir, complement)`` keys of the views :meth:`view` served.

        A view only :meth:`prebuild` built is not among them: the next
        index prebuilds these keys alone, so a view no tab asked for
        during one index's lifetime is dropped instead of being rebuilt
        on every rescan for the daemon's lifetime.
        """
        with self._views_lock:
            return list(self._served)

    def view(self, sub_dir: str, complement: bool = False) -> FileView:
        """Return the (cached) view of the entries below *sub_dir*.

        Args:
            sub_dir: ``""`` for the whole index, else a relative directory
                path with a trailing ``/`` (``"kiss/src/"``).
            complement: Return the entries NOT below *sub_dir* instead,
                as root-relative paths (the ``~/`` part of a picker
                rooted at *sub_dir*).  Ignored when *sub_dir* is ``""``.
        """
        key = (sub_dir, complement and bool(sub_dir))
        with self._views_lock:
            self._served.add(key)
            return self._cached_view(key)

    def prebuild(self, keys: list[tuple[str, bool]]) -> None:
        """Build the views of *keys* (from :meth:`view_keys`) before anyone asks.

        Prebuilt views are not served views (see :meth:`view_keys`)
        until :meth:`view` hands them out.
        """
        with self._views_lock:
            for key in keys:
                self._cached_view(key)

    def _cached_view(self, key: tuple[str, bool]) -> FileView:
        """Return the view of the normalised *key*, building it once (``_views_lock`` held)."""
        view = self._views.get(key)
        if view is not None:
            return view
        sub_dir, complement = key
        if not sub_dir:
            view = FileView(self.paths, self.wide_from)
        else:
            head, tail = self.paths[:self.wide_from], self.paths[self.wide_from:]
            if complement:
                head = [p for p in head if not p.startswith(sub_dir)]
                tail = [p for p in tail if not p.startswith(sub_dir)]
            else:
                # ``len(p) > n`` drops *sub_dir*'s own entry, which is in
                # the tail when its parent is a wide container.
                n = len(sub_dir)
                head = [p[n:] for p in head if p.startswith(sub_dir) and len(p) > n]
                tail = [p[n:] for p in tail if p.startswith(sub_dir) and len(p) > n]
            view = FileView(head + tail, len(head))
        self._views[key] = view
        return view


def _search_prefixed(
    view: FileView | None,
    query: str,
    usage: dict[str, int],
    prefix: str,
    kind: str,
    limit: int,
) -> list[dict[str, str]]:
    """Search *view* and return its items as *prefix*-ed mentions.

    Only the *usage* entries recorded under *prefix* count as frequent
    here (a ``./`` mention and a ``~/`` mention are different paths).
    Plain ``file`` hits are retyped *kind*; ``frequent`` ones keep their
    type so both parts of a picker share one "Frequent" section.
    """
    if view is None:
        return []
    n = len(prefix)
    own = {p[n:]: count for p, count in usage.items() if p.startswith(prefix)}
    items = view.search(query, own, limit)
    for item in items:
        item["text"] = prefix + item["text"]
        if item["type"] == "file":
            item["type"] = kind
    return items


class Picker:
    """What one tab's ``@``-mention picker searches.

    Matches inside the work dir come back as ``./path`` items (types
    ``frequent`` / ``file``), matches elsewhere below the home directory
    as ``~/path`` items (types ``frequent`` / ``home``), so the text of
    an item is exactly the mention to insert and the usage key to
    record.  A query starting with ``./`` searches the work dir only; one
    starting with ``~`` (``~/Doc``) searches the whole home directory
    only and every hit is a ``~/`` item (the work dir included when the
    home index holds it; a work dir inside a skipped directory such as
    ``~/.cache/x`` is reachable through its ``./`` items only).

    Attributes:
        local: View of the work dir.
        home_rest: View of the home directory minus the work dir, or
            ``None`` when there is nothing outside the work dir to offer
            (the work dir is home itself, or home is not indexed yet).
        home_all: View of the whole home directory, or ``None`` when it
            is not indexed yet.
    """

    def __init__(
        self, local: FileView, home_rest: FileView | None, home_all: FileView | None,
    ) -> None:
        """Wrap the views; see the class attributes."""
        self.local = local
        self.home_rest = home_rest
        self.home_all = home_all

    def search(
        self,
        query: str,
        usage: dict[str, int],
        limit: int = SUGGESTION_LIMIT,
    ) -> list[dict[str, str]]:
        """Rank at most *limit* mentions for *query*.

        Frequent mentions of either part lead, then the work dir's
        files, then home's; when both parts together exceed *limit*,
        the work dir yields up to :data:`HOME_SLOTS` places to the home
        part so it never disappears entirely.

        Args:
            query: The text after ``@``, optionally led by ``./`` or ``~/``.
            usage: Mention counts keyed by mention text, oldest first
                (see ``_load_file_usage``).
            limit: Maximum number of items returned.
        """
        if query.startswith("~"):
            q = query[2:] if query.startswith("~/") else query[1:]
            return _search_prefixed(self.home_all, q, usage, "~/", "home", limit)
        if query.startswith("./"):
            return _search_prefixed(self.local, query[2:], usage, "./", "file", limit)
        local = _search_prefixed(self.local, query, usage, "./", "file", limit)
        home = _search_prefixed(self.home_rest, query, usage, "~/", "home", limit)
        if len(local) + len(home) > limit:
            local = local[:limit - min(len(home), HOME_SLOTS)]
            home = home[:limit - len(local)]
        items = local + home
        frequent = [item for item in items if item["type"] == "frequent"]
        return frequent + [item for item in items if item["type"] != "frequent"]


def _load_listings(path: Path, root: str) -> Listings:
    """Read the persisted listings of *root* from *path*; ``{}`` if unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION or data.get("root") != root:
            return {}
        return {
            rel: (int(item[0]), list(item[1]), list(item[2]))
            for rel, item in data["listings"].items()
        }
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        logger.debug("no usable file-index cache at %s", path, exc_info=True)
        return {}


def _save_listings(path: Path, root: str, listings: Listings) -> None:
    """Atomically persist *listings* of *root* to *path* (best effort)."""
    payload = {"version": _CACHE_VERSION, "root": root, "listings": listings}
    try:
        # Staged in a private temporary file: two daemons persisting
        # the same root must not interleave their writes before the rename.
        atomic_write_text(path, json.dumps(payload, separators=(",", ":")))
    except OSError:
        logger.debug("cannot persist file index to %s", path, exc_info=True)


class FileIndexRegistry:
    """Owns the indexes of every root the picker has been asked about.

    One daemon thread builds and refreshes indexes so two scans of the
    same root never race; :meth:`picker_for` never blocks — it serves
    the current index (scheduling a refresh when it is older than
    :data:`STALE_AFTER`) or ``None`` when the root has not been indexed
    yet, in which case :meth:`ensure` builds it and runs a callback.

    Work dirs below the home directory are served by the home index
    (see :meth:`root_for`); every other work dir is a root of its own.
    """

    def __init__(self, home: str = "", cache_dir: Path | None = None) -> None:
        """Create an idle registry.

        Args:
            home: The user's home directory; defaults to ``Path.home()``.
            cache_dir: Where persisted listings live; defaults to
                ``$KISS_HOME/file-index`` (resolved lazily on each save
                so a ``KISS_HOME`` change in tests is honoured).
        """
        self.home = os.path.abspath(home or str(Path.home()))
        self._cache_dir = cache_dir
        self._indexes: dict[str, FileIndex] = {}
        # Roots with a queued job, and when each was last asked for.
        self._pending: dict[str, float] = {}
        self._building: str | None = None
        self._lock = threading.Lock()
        self._jobs: queue.Queue[tuple[str, float, Callable[[], object] | None] | None] = (
            queue.Queue()
        )
        self._worker: threading.Thread | None = None
        self._stopped = False

    def root_for(self, work_dir: str) -> tuple[str, str]:
        """Resolve *work_dir* to ``(index root, sub_dir)``.

        A work dir at or below :attr:`home` maps to the home index with
        ``sub_dir`` its home-relative path plus ``/`` (``""`` for home
        itself), unless a path component is one the scan skips
        (dot-directory, Windows-hidden directory, :data:`JUNK_DIR_NAMES`,
        the home's application state tree of :data:`_HOME_RULES`) or —
        once the home index exists — the directory is not among its
        :attr:`FileIndex.dirs` (gitignored, collapsed or too deep).
        Everything else, and a filesystem root (which is never scanned),
        maps to a root of its own: ``(work_dir, "")``.
        """
        wd = self.home if is_root_dir(work_dir) else os.path.abspath(work_dir)
        if is_root_dir(wd):
            wd = self.home
        if wd == self.home:
            return self.home, ""
        if wd.startswith(self.home + os.sep):
            rel = wd[len(self.home) + 1:].replace(os.sep, "/")
            parts = rel.split("/")
            skipped = parts[0] in _HOME_SKIPPED_DIRS or any(
                p.startswith(".") or p in JUNK_DIR_NAMES for p in parts
            )
            hidden = _hidden_dir_on_path(self.home, parts)
            if len(parts) <= MAX_DEPTH and not skipped and not hidden:
                with self._lock:
                    home_index = self._indexes.get(self.home)
                if home_index is None or rel in home_index.dirs:
                    return self.home, rel + "/"
        return wd, ""

    def _current(self, root: str) -> FileIndex | None:
        """Return the index of *root*, or ``None`` if it was never built.

        Serving an index older than :data:`STALE_AFTER` also queues a
        background refresh of its root, so the next request sees the
        change without anyone waiting on the rescan.  A root whose
        rescan is already queued or running is left alone: the stale
        index is still the one served until that rescan is published,
        and queueing another walk behind it would scan the tree twice
        for the same request.
        """
        with self._lock:
            index = self._indexes.get(root)
            if (
                index is not None
                and root != self._building
                and time.monotonic() - index.built_at > STALE_AFTER
            ):
                self._enqueue_locked(root, None)
        return index

    def picker_for(self, work_dir: str) -> Picker | None:
        """Return the :class:`Picker` of *work_dir*, or ``None`` if unindexed.

        ``None`` only while the root covering *work_dir* itself is not
        built yet.  The home part is whatever the home index holds at
        the moment: the complement of *work_dir* when the home index
        serves it, all of home when *work_dir* is a root of its own
        beside home, nothing when *work_dir* is home itself or contains
        it (its own index lists home's files already) or the home index
        does not exist yet (the web server pre-warms it at start-up and
        a missing one is never waited for).
        """
        root, sub_dir = self.root_for(work_dir)
        index = self._current(root)
        if index is None:
            return None
        local = index.view(sub_dir)
        if root != self.home:
            home = self._current(self.home)
            home_all = None if home is None else home.view("")
            # A work dir above home (``/home``) lists home's files itself.
            above_home = self.home.startswith(os.path.join(root, ""))
            return Picker(local, None if above_home else home_all, home_all)
        home_rest = index.view(sub_dir, complement=True) if sub_dir else None
        return Picker(local, home_rest, index.view(""))

    def ensure(self, work_dir: str, on_ready: Callable[[], object] | None = None) -> bool:
        """Build (or refresh) the index covering *work_dir* in the background.

        Args:
            work_dir: Directory whose covering root should be indexed.
            on_ready: Called on the worker thread once the root's index
                is at least as new as this call.  It is invoked even when
                the build failed (the root then has an empty index), so a
                caller waiting on a picker reply always gets one; only a
                job still queued when :meth:`stop` runs is dropped with
                its callback (the daemon is shutting down).

        Returns:
            Whether the job was queued.  ``False`` after :meth:`stop` or
            when the worker thread cannot be started; *on_ready* will
            then never run and the caller must not wait for it.
        """
        root, _ = self.root_for(work_dir)
        with self._lock:
            return self._enqueue_locked(root, on_ready)

    def refresh(self, work_dir: str) -> None:
        """Queue a rescan of the root covering *work_dir* if it is indexed.

        A root nobody has asked about yet is left alone: the first
        request will scan it from scratch anyway.
        """
        root, _ = self.root_for(work_dir)
        with self._lock:
            if root in self._indexes:
                self._enqueue_locked(root, None)

    def _enqueue_locked(self, root: str, on_ready: Callable[[], object] | None) -> bool:
        """Queue a build of *root*, :attr:`_lock` held; a duplicate callback-less job is dropped.

        Returns whether a job will run for this request (a dropped
        duplicate counts: the pending job covers it).
        """
        if self._stopped:
            return False
        now = time.monotonic()
        if on_ready is None and root in self._pending:
            # The pending job answers this request too: no scan that
            # started before now may count as its answer.
            self._pending[root] = now
            return True
        if self._worker is None:
            worker = threading.Thread(
                target=self._worker_loop, name="file-index", daemon=True,
            )
            try:
                worker.start()
            except RuntimeError:
                # Thread exhaustion: leave ``_worker`` unset so the
                # next request tries again instead of queueing jobs
                # nobody will ever run.
                logger.exception("file index worker failed to start")
                return False
            self._worker = worker
        self._pending[root] = now
        # Under the lock, so no job can land behind the stop sentinel.
        self._jobs.put((root, now, on_ready))
        return True

    def stop(self) -> None:
        """Stop the worker thread; later requests are ignored."""
        with self._lock:
            self._stopped = True
            worker = self._worker
        if worker is not None:
            self._jobs.put(None)
            worker.join(timeout=5.0)

    def _cache_path(self, root: str) -> Path:
        """Return the persisted-listings file of *root*."""
        base = self._cache_dir if self._cache_dir is not None else kiss_home() / "file-index"
        # ``fsencode`` round-trips surrogate-escaped (undecodable) path
        # bytes that ``str.encode("utf-8")`` would reject.
        return base / (hashlib.sha1(os.fsencode(root)).hexdigest()[:16] + ".json")

    def _build(self, root: str) -> None:
        """Scan *root* (reusing cached listings) and publish the index.

        The first build of a root with persisted listings publishes an
        index made from the listings alone (``trusted``: no system call,
        well under a second for a home directory) and queues the real
        rescan as the next job: the first ``@`` after a daemon restart
        then answers at once, while the rescan -- which on macOS can
        wait seconds on each protected directory it must re-list -- runs
        behind it and publishes the fresh index when done.  The trusted
        index is born stale (``built_at`` is ``-inf``), so a job that was
        queued for the root while the listings loaded, and absorbed the
        rescan request, rescans too instead of taking it for current.
        """
        with self._lock:
            previous = self._indexes.get(root)
        rules = _HOME_RULES if root == self.home else _ROOT_RULES
        started = time.monotonic()
        try:
            listings = {} if previous is not None else _load_listings(self._cache_path(root), root)
            if listings:
                index = FileIndex.scan(root, listings, None, rules, trusted=True)
                with self._lock:
                    self._indexes[root] = index
                    self._enqueue_locked(root, None)
                if index.relisted:
                    # The listings held a directory this process has
                    # already been refused: it was dropped, and the rescan
                    # (which starts from this index) would not notice.
                    _save_listings(self._cache_path(root), root, index.listings)
                logger.info(
                    "file index of %s: %d entries in %.2fs (persisted listings; rescan queued)",
                    root, len(index.paths), time.monotonic() - started,
                )
                return
            index = FileIndex.scan(root, {}, previous, rules)
        except Exception:
            logger.exception("file index scan of %s failed", root)
            index = FileIndex.empty()
        # Rebuild the views tabs used since the previous build, here on
        # the worker thread, so the next keystroke does not pay for them.
        if previous is not None:
            index.prebuild(previous.view_keys())
        with self._lock:
            self._indexes[root] = index
        if index.relisted:
            _save_listings(self._cache_path(root), root, index.listings)
        logger.info(
            "file index of %s: %d entries in %.2fs (%s)",
            root, len(index.paths), time.monotonic() - started,
            "rescanned" if index.relisted else "unchanged",
        )

    def _worker_loop(self) -> None:
        """Serve build jobs until :meth:`stop`."""
        while True:
            job = self._jobs.get()
            if job is None or self._stopped:
                return
            root, queued_at, on_ready = job
            # ``_pending`` is cleared here, not when the build ends, so
            # a ``refresh`` during the walk (the tree changed after it
            # started) still queues the rescan it needs; ``_building``
            # keeps ``_current`` from queueing a walk for staleness
            # alone while this one runs.  The job takes the time of the
            # last request it absorbed while pending.
            with self._lock:
                queued_at = max(queued_at, self._pending.pop(root, queued_at))
                self._building = root
                current = self._indexes.get(root)
            if current is None or current.built_at < queued_at:
                # The only worker: one failing build must not end it.
                try:
                    self._build(root)
                except Exception:
                    logger.exception("file index build of %s failed", root)
            with self._lock:
                self._building = None
            if on_ready is not None:
                try:
                    on_ready()
                except Exception:
                    logger.exception("file index callback for %s failed", root)
