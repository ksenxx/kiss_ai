# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SQLite vector index over a directory of memory pages.

The index is a single SQLite file inside the memory directory, named after
the embedding model (``text-embedding-3-small.sqlite3``). It stores one row
per page: the page's frontmatter, its sha256, its unit-length embedding
as a float32 BLOB and a ``revision`` number that is unique to each write
of the index. Rows are refreshed incrementally: a page whose stat data is
unchanged is not even read, a page is re-embedded only when its sha256
changes, changed pages are embedded in batched requests, and rows for
deleted pages are dropped. Every agent process syncs the same index file,
so writes are compare-and-swaps on the ``revision`` token (see
:meth:`VectorIndex.sync`).

Search is an exhaustive cosine scan: one numpy float32 matrix-vector product
over the stored embeddings selects candidates, which are then rescored
exactly with :func:`math.sumprod`. Each :class:`VectorIndex` keeps the
decoded rows in memory until a write to the index changes its change token
(see :meth:`VectorIndex._rows`). No SQLite extension is needed
(``sqlite-vec`` cannot be loaded by the macOS system Python). The index is
a cache: deleting the ``.sqlite3`` file loses nothing.
"""

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from array import array
from collections import OrderedDict
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir, split_frontmatter
from kiss.core.utils import read_bytes_waiting_for_writer

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

Embedder = Callable[[str], list[float]]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

HASHED_EMBEDDING_MODEL_CODE = "hashed-bow-v1"
HASHED_EMBEDDING_DIMS = 1024

# Version of the page-to-embedding-input mapping (see
# :func:`embedding_text`), stored per row in the ``input_format`` column.
# :meth:`VectorIndex.sync` re-embeds any row carrying a different value —
# including rows that a still-running older process writes AFTER an
# upgraded process has synced (rows written before this column existed
# read as the column's default, ``'1'``).  The ``sha256`` column keeps the
# plain content hash, so a pre-format process sees rows written by current
# code as unchanged and the two versions never rebuild the index back and
# forth.  Bump this whenever the mapping changes.
EMBEDDING_INPUT_FORMAT = "2"

# Columns added to the ``pages`` table after its first release, with the
# definition used both in CREATE TABLE and in the ALTER TABLE migration
# that :meth:`VectorIndex._connect` runs on older index files.
_ADDED_COLUMNS = (
    ("input_format", "TEXT NOT NULL DEFAULT '1'"),
    ("revision", "INTEGER NOT NULL DEFAULT 0"),
    ("stat_key", "TEXT NOT NULL DEFAULT ''"),
)

# Stored in ``meta`` once :meth:`VectorIndex._connect` has created the
# tables and added every column in :data:`_ADDED_COLUMNS`.  A connection
# that finds it (and the model code and generation) skips all DDL and
# writes, so opening the index for a search takes no write lock.  An index
# file created by an older process lacks the key and gets migrated.
SCHEMA_VERSION = "3"

# Rows per block of the pairwise similarity matrix in
# :meth:`VectorIndex.near_duplicates` (bounds its memory to
# ``_PAIR_BLOCK_ROWS * rows * 4`` bytes).
_PAIR_BLOCK_ROWS = 1024

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens of *text*."""
    return _WORD_RE.findall(text.lower())


def hashed_embedding(text: str, dims: int = HASHED_EMBEDDING_DIMS) -> list[float]:
    """Offline feature-hashing embedding (unigrams + bigrams, log-scaled counts).

    A real, deterministic bag-of-words embedding that needs no network or
    API key. It captures lexical overlap only, so it is the fallback when no
    embedding model is configured and the reference point that a neural
    embedding must beat.

    Args:
        text: Text to embed.
        dims: Vector length.

    Returns:
        A unit-length vector of *dims* floats (all zeros for empty text).
    """
    vector = [0.0] * dims
    words = _tokens(text)
    features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
    counts: dict[str, int] = {}
    for feature in features:
        counts[feature] = counts.get(feature, 0) + 1
    for feature, count in counts.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        vector[(value >> 1) % dims] += sign * (1.0 + math.log(count))
    return normalize(vector)


def embedding_text(raw: str) -> str:
    """The searchable text of a page: title, summary and Markdown body.

    Volatile frontmatter — the ``uuid`` and the ``created``/``updated``
    timestamps — is excluded: embedding it made two writes of identical
    content produce different vectors, and its random tokens leaked into
    similarity scores as noise.

    Args:
        raw: Complete page text including frontmatter.
    """
    frontmatter, body = split_frontmatter(raw)
    parts = [str(frontmatter[key]) for key in ("title", "summary") if frontmatter.get(key)]
    return "\n".join([*parts, body])


def normalize(vector: list[float]) -> list[float]:
    """Scale *vector* to unit length (zero vectors are returned unchanged)."""
    norm = math.sqrt(math.sumprod(vector, vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def serialize_float32(vector: list[float]) -> bytes:
    """Pack *vector* as little-endian float32 bytes (the sqlite-vec BLOB layout)."""
    packed = array("f", vector)
    if packed.itemsize != 4:  # pragma: no cover - platform without 4-byte floats
        raise RuntimeError("array('f') is not 32-bit on this platform")
    return packed.tobytes()


def deserialize_float32(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB produced by :func:`serialize_float32`."""
    unpacked = array("f")
    unpacked.frombytes(blob)
    return unpacked.tolist()


class ModelEmbedder:
    """Embedder backed by a KISS framework embedding model.

    Thread-safe: the model client is created once, under a lock, and then
    shared, so its HTTP connection pool is reused across calls.

    Args:
        model_name: Embedding model name from the model catalog, e.g.
            ``text-embedding-3-small`` or ``gemini-embedding-001``.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _client(self) -> Any:
        """The initialised model, created on first use."""
        with self._lock:
            if self._model is None:
                from kiss.core.models.model_info import model as create_model

                model = create_model(self.model_name)
                model.initialize("")
                self._model = model
            return self._model

    def __call__(self, text: str) -> list[float]:
        """Embed *text* with the configured model (lazily initialised)."""
        vector: list[float] = self._client().get_embedding(text, embedding_model=self.model_name)
        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in one request when the model supports batching.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text, in order.
        """
        vectors: list[list[float]] = self._client().get_embeddings(
            texts, embedding_model=self.model_name
        )
        return vectors



# Query embeddings kept by each VectorIndex (an agent typically runs
# memory_search and then memory_pull with the same query).
QUERY_CACHE_SIZE = 64


def default_embedder() -> Embedder:
    """The best embedder the current process can actually run.

    :class:`ModelEmbedder` with :data:`DEFAULT_EMBEDDING_MODEL` when an
    ``OPENAI_API_KEY`` is present in the environment (the daemon loads the
    key store into the environment at startup), else the fully offline
    :func:`hashed_embedding`.  Each embedder names its own index file
    (``text-embedding-3-small.sqlite3`` vs ``hashed-bow-v1.sqlite3``), so a
    store used first without a key and later with one never mixes vectors:
    the second embedder simply builds its own index beside the first.

    Returns:
        An :data:`Embedder` callable safe to invoke in this process.
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return ModelEmbedder()
    return hashed_embedding


def model_code_for_filename(model_name: str) -> str:
    """Map a model name to the filename-safe prefix of its index file.

    Args:
        model_name: Embedding model name, possibly with a provider prefix
            such as ``BAAI/bge-base-en-v1.5``.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-.")


# A page's stat key is recorded only when its mtime and ctime are at least
# this much older than the scan (covers 2-second FAT timestamps and small
# clock differences); younger pages are hashed on every sync.
RACY_WINDOW_NS = 3_000_000_000

# Limits of one embedding request made by VectorIndex.sync (OpenAI accepts
# 2048 inputs and 300k tokens per request; a token is at least one byte).
EMBED_BATCH_INPUTS = 256
EMBED_BATCH_BYTES = 250_000


def stat_key(st: os.stat_result, scan_ns: int, digest: bytes) -> str:
    """Identify a page file's version by its stat data, as git's index does.

    The key ends with the sha256 *digest* of the content it was verified
    against, so it vouches for a row only while the row still holds that
    digest: a still-running older process that rewrites a row's content
    without knowing about the ``stat_key`` column leaves a key that no
    longer matches.

    Any content write changes the size or mtime, and on POSIX also the
    ctime (which user space cannot set); an atomic replace changes the
    inode.  On Windows ``st_ctime`` is the creation time (deprecated since
    Python 3.12) and ``os.scandir`` reports ``st_ino`` as 0, so the key
    there is size and mtime — NTFS stamps mtime in 100 ns ticks.  Returns
    ``''`` — "must hash" — while the timestamps fall within
    :data:`RACY_WINDOW_NS` of *scan_ns*, when a same-size rewrite in the
    same timestamp tick (1 s on HFS+, 2 s on FAT) would be invisible.

    Args:
        st: The page's ``lstat`` result, taken before its content is read.
        scan_ns: :func:`time.time_ns` taken before the directory scan.
        digest: sha256 of the page content read after *st* was taken.
    """
    changed_ns = st.st_mtime_ns if os.name == "nt" else max(st.st_mtime_ns, st.st_ctime_ns)
    if changed_ns >= scan_ns - RACY_WINDOW_NS:
        return ""
    if os.name == "nt":
        # No inode: ``os.scandir`` entries report 0 while ``os.stat`` reports
        # the real file index, so the same file would get two keys.
        return f"{st.st_size}:{st.st_mtime_ns}#{digest.hex()}"
    return f"{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}#{digest.hex()}"


def float32_dot_error(dims: int) -> float:
    """Upper bound on the error of a float32 dot product of two unit vectors.

    Rounding the query to float32 and accumulating *dims* products in
    float32 (any summation order, with or without FMA) each contribute at
    most one unit roundoff, 2**-24, per term.

    Args:
        dims: Vector length.
    """
    return (dims + 2) * 2.0**-24


_Pending = tuple[str, bytes, str, str, str]
"""A page awaiting embedding: filename, sha256, raw text, stat key, embedding input."""


def _batches(pending: list[_Pending]) -> list[list[_Pending]]:
    """Split pending pages into embedding requests within the batch limits."""
    batches: list[list[_Pending]] = []
    size = 0
    for item in pending:
        length = len(item[4].encode("utf-8"))
        if (
            not batches
            or len(batches[-1]) >= EMBED_BATCH_INPUTS
            or size + length > EMBED_BATCH_BYTES
        ):
            batches.append([])
            size = 0
        batches[-1].append(item)
        size += length
    return batches


def _embed_all(embed: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed *texts* with one ``embed_many`` call when *embed* has one, else one by one."""
    embed_many = getattr(embed, "embed_many", None)
    if embed_many is not None:
        vectors: list[list[float]] = embed_many(texts)
        return vectors
    return [embed(text) for text in texts]


@dataclass(frozen=True)
class SearchHit:
    """One search result.

    Attributes:
        name: Page name (without ``.md``).
        score: Cosine similarity in ``[-1, 1]``; higher is closer.
        frontmatter: The page's frontmatter as stored in the index.
    """

    name: str
    score: float
    frontmatter: dict[str, Any]

    @property
    def title(self) -> str:
        """Title from frontmatter, or the page name."""
        title = self.frontmatter.get("title")
        return str(title) if title else self.name

    @property
    def summary(self) -> str:
        """Summary from frontmatter, or an empty string."""
        summary = self.frontmatter.get("summary")
        return str(summary) if summary else ""


@dataclass(frozen=True)
class SyncReport:
    """Counts from one :meth:`VectorIndex.sync` run."""

    added: int
    updated: int
    removed: int
    unchanged: int


@dataclass(frozen=True)
class _Rows:
    """Decoded index rows that share one embedding dimension, in filename order.

    Attributes:
        names: Page names (without ``.md``).
        frontmatter: Each row's frontmatter as stored (JSON text).
        vectors: ``len(names) x dims`` float32 matrix of unit embeddings.
    """

    names: list[str]
    frontmatter: list[str]
    vectors: "np.ndarray"


class VectorIndex:
    """Incrementally maintained cosine-similarity index over a :class:`MemoryDir`.

    Args:
        memory: The page directory to index.
        embed: Function mapping text to an embedding vector. Defaults to
            :func:`default_embedder`.
        model_code: Identifier of the embedding model, used to name the index
            file so indexes from different models never mix. Defaults to the
            embedder's ``model_name`` when it has one, else
            :data:`HASHED_EMBEDDING_MODEL_CODE`.
    """

    def __init__(
        self,
        memory: MemoryDir,
        embed: Embedder | None = None,
        model_code: str | None = None,
    ) -> None:
        self.memory = memory
        if embed is None:
            embed = default_embedder()
        self.embed = embed
        if model_code is None:
            model_code = str(getattr(embed, "model_name", "") or HASHED_EMBEDDING_MODEL_CODE)
        self.model_code: str = model_code
        self.path = memory.root / f"{model_code_for_filename(self.model_code)}.sqlite3"
        # (change token, rows grouped by dimension) from the last _rows()
        # load; replaced as one tuple so concurrent threads never see a
        # token paired with another load's rows.
        self._cache: tuple[tuple[Any, ...], dict[int, _Rows]] | None = None
        self._query_vectors: OrderedDict[str, list[float]] = OrderedDict()
        self._query_lock = threading.Lock()

    def embed_query(self, query: str) -> list[float]:
        """Return the unit embedding of *query*, from a small LRU cache when possible.

        Args:
            query: Natural-language query.
        """
        with self._query_lock:
            vector = self._query_vectors.get(query)
            if vector is not None:
                self._query_vectors.move_to_end(query)
                return vector
        vector = normalize(self.embed(query))
        with self._query_lock:
            self._query_vectors[query] = vector
            if len(self._query_vectors) > QUERY_CACHE_SIZE:
                self._query_vectors.popitem(last=False)
        return vector

    def _connect(self) -> sqlite3.Connection:
        """Open the index database, creating or migrating the schema when needed.

        An index already at :data:`SCHEMA_VERSION` is opened with reads
        only; otherwise the tables, missing columns and ``meta`` keys are
        created first.  The returned connection holds no open transaction,
        so readers never block writers. Callers must close it (use
        ``closing(...)``).

        Raises:
            ValueError: If the index path is a symlink (it would be followed
                to a file outside the memory directory) or if the file was
                built by a different embedding model.
        """
        self.memory.root.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError(f"Index path {self.path} is a symlink; refusing to use it.")
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            try:
                meta = dict(
                    conn.execute(
                        "SELECT key, value FROM meta"
                        " WHERE key IN ('model_code', 'schema_version', 'generation')"
                    ).fetchall()
                )
            except sqlite3.OperationalError:
                meta = {}  # new file: no meta table yet
            if meta.get("schema_version") != SCHEMA_VERSION or "generation" not in meta:
                self._create_schema(conn)
                meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("model_code") != self.model_code:
                raise ValueError(
                    f"Index {self.path} was built with embedding model {meta.get('model_code')!r}, "
                    f"not {self.model_code!r}; delete it or use a distinct model_code."
                )
        except BaseException:
            conn.close()
            raise
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the tables, add missing columns and seed the ``meta`` keys.

        Idempotent and safe to race with other processes doing the same.
        The random ``generation`` identifies this index file, so a deleted
        and rebuilt index never matches a change token cached from the old
        file (see :meth:`_rows`).
        """
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pages ("
            " filename TEXT PRIMARY KEY,"
            " frontmatter TEXT NOT NULL,"
            " last_modified REAL NOT NULL,"
            " sha256 BLOB NOT NULL,"
            " embedding BLOB NOT NULL,"
            " input_format TEXT NOT NULL DEFAULT '1',"
            " revision INTEGER NOT NULL DEFAULT 0,"
            " stat_key TEXT NOT NULL DEFAULT '')"
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(pages)")}
        for column, definition in _ADDED_COLUMNS:
            if column in columns:
                continue
            # Index created before this column existed.  input_format's
            # default '1' marks old rows for re-embedding (they were
            # embedded from the whole raw file, frontmatter included);
            # revision's default 0 is just a valid CAS token.
            try:
                conn.execute(f"ALTER TABLE pages ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError:
                # Fine only if another process added the column between
                # the PRAGMA read and the ALTER (unreachable in a
                # single-process test run); anything else, such as a lock
                # timeout, must not be recorded as a finished migration.
                if column not in {row[1] for row in conn.execute("PRAGMA table_info(pages)")}:
                    raise
        # Covering index for sync's snapshot query: revision and stat_key sit
        # after the embedding BLOB in each row, so without it every sync
        # would read every row's overflow pages (the whole file).
        conn.execute(
            "CREATE INDEX IF NOT EXISTS pages_sync"
            " ON pages (filename, sha256, input_format, revision, stat_key)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            [("model_code", self.model_code), ("revision", "0"), ("generation", uuid.uuid4().hex)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def embedding_input(self, raw: str) -> str:
        """The text that gets embedded for a page, capped at 8192 bytes.

        :func:`embedding_text` (title, summary and Markdown body — no
        volatile frontmatter), truncated to :data:`MAX_PAGE_BYTES` of UTF-8.

        Args:
            raw: Complete page text including frontmatter.
        """
        text = embedding_text(raw)
        data = text.encode("utf-8")
        if len(data) <= MAX_PAGE_BYTES:
            return text
        return data[:MAX_PAGE_BYTES].decode("utf-8", errors="ignore")

    def sync(self, verify: bool = False) -> SyncReport:
        """Bring the index in line with the pages on disk.

        A page is skipped when its row was embedded under the current
        :data:`EMBEDDING_INPUT_FORMAT` and either its stored stat key (size,
        mtime, ctime and inode, see :func:`stat_key`) equals the file's
        current one — no read at all — or its stored sha256 matches the
        file's content.  A stat key is stored only once the file's
        timestamps are :data:`RACY_WINDOW_NS` older than the scan that
        recorded them (git's "racy clean" rule), so a same-size rewrite
        within one timestamp tick of a scan is still caught by hashing.
        New or modified pages and rows embedded under an older input mapping
        — even ones written by a still-running old process after this code
        has synced — are re-embedded, in batches of at most
        :data:`EMBED_BATCH_INPUTS` inputs / :data:`EMBED_BATCH_BYTES` bytes
        (one request each when the embedder has an ``embed_many`` method).
        Rows for pages that no longer exist are removed.  Embeddings are
        computed with no database transaction open, and each batch's rows
        are then written in one short transaction, so an interrupted build
        keeps the batches it finished.  A page that changes while it is being
        embedded is left for the next sync rather than stored with a stale
        vector.

        Every write is an optimistic compare-and-swap against the snapshot
        of the index this sync started from.  Each row carries a
        ``revision`` drawn from a counter in the ``meta`` table that only
        ever increases, so no two writes to the index ever share one; a row
        is inserted only if still absent, and updated or deleted only while
        its ``revision`` still equals the snapshot's.  A row that another
        sync inserted, updated, deleted or deleted-and-re-inserted after the
        snapshot — even with byte-identical content — therefore never
        matches, so a sync never overwrites or removes a row written by a
        concurrent sync.  Writes skipped this way are counted in no
        category.  (Recording a stat key for an unchanged row is not a
        content write: it applies only while the row still holds the digest
        it was verified against.)  Two things the CAS cannot see: the disk —
        a page deleted after its post-embed check keeps (or gains) a row
        until the next sync drops it, exactly as with a single sync — and a
        still-running process on the pre-``revision`` code, whose rows land
        with the column default ``0`` and are matched (or blindly replaced
        by that process) until every writer runs this version.  Both are
        healed by the next sync because the ``sha256`` check is unaffected.

        Args:
            verify: Hash every page even when its stat key matches (used by
                ``memory_refresh`` to catch edits that preserved size and
                timestamps).

        Returns:
            A :class:`SyncReport` with per-category counts of the writes
            that were actually applied.
        """
        unchanged = 0
        with closing(self._connect()) as conn:
            stored = {
                row[0]: (bytes(row[1]), row[2], row[3], row[4])
                for row in conn.execute(
                    "SELECT filename, sha256, input_format, revision, stat_key FROM pages"
                )
            }
        scan_ns = time.time_ns()
        pages = self.memory.page_stats()
        on_disk = {f"{name}.md" for name in pages}
        restats: list[tuple[str, str, bytes]] = []
        pending: list[_Pending] = []
        for name in sorted(pages):
            filename = f"{name}.md"
            snapshot = stored.get(filename)
            # The row, if it was embedded under the current input mapping.
            current = (
                snapshot if snapshot is not None and snapshot[1] == EMBEDDING_INPUT_FORMAT else None
            )
            if (
                current is not None
                and not verify
                and current[3]
                and current[3] == stat_key(pages[name], scan_ns, current[0])
            ):
                unchanged += 1
                continue
            try:
                path = self.memory.page_path(name)  # refuses a page swapped for a symlink
                data = read_bytes_waiting_for_writer(path)
            except FileNotFoundError:
                # Deleted by another process since the scan: its row (if
                # any) is stale and is removed below.
                on_disk.discard(filename)
                continue
            digest = hashlib.sha256(data).digest()
            key = stat_key(pages[name], scan_ns, digest)
            if current is not None and current[0] == digest:
                unchanged += 1
                if key != current[3]:
                    restats.append((key, filename, digest))
                continue
            raw = data.decode("utf-8", errors="replace")
            pending.append((filename, digest, raw, key, self.embedding_input(raw)))

        stale = sorted(stored.keys() - on_disk)
        report = SyncReport(added=0, updated=0, removed=0, unchanged=unchanged)
        if restats:
            report = self._write(stored, [], restats, [], report)
        for batch in _batches(pending):
            vectors = _embed_all(self.embed, [item[4] for item in batch])
            rows = []
            for (filename, digest, raw, key, _), vector in zip(batch, vectors, strict=True):
                path = self.memory.root / filename  # validated by page_path above
                try:
                    changed = (
                        hashlib.sha256(read_bytes_waiting_for_writer(path)).digest() != digest
                    )
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    changed, mtime = True, 0.0  # deleted mid-embed; skipped below
                if changed:
                    logger.info(
                        "%s changed while being embedded; it will be indexed on the next sync",
                        filename,
                    )
                    continue
                rows.append(
                    (
                        filename,
                        json.dumps(split_frontmatter(raw)[0], default=str),
                        mtime,
                        digest,
                        serialize_float32(normalize(vector)),
                        EMBEDDING_INPUT_FORMAT,
                        key,
                    )
                )
            report = self._write(stored, rows, [], [], report)
        if stale:
            report = self._write(stored, [], [], stale, report)
        return report

    def _write(
        self,
        stored: dict[str, tuple[bytes, str, int, str]],
        rows: list[tuple[str, str, float, bytes, bytes, str, str]],
        restats: list[tuple[str, str, bytes]],
        stale: list[str],
        report: SyncReport,
    ) -> SyncReport:
        """Apply one batch of :meth:`sync` writes in a single transaction.

        Args:
            stored: The snapshot the sync started from (the CAS tokens).
            rows: Embedded rows to insert or update.
            restats: ``(stat_key, filename, sha256)`` for unchanged rows.
            stale: Filenames whose rows should be deleted.
            report: Counts so far.

        Returns:
            *report* plus the writes that this transaction applied.
        """
        added, updated, removed = report.added, report.updated, report.removed
        skipped_message = "%s row changed or was removed after this sync started; skipping it"
        with closing(self._connect()) as conn, conn:
            # Take the write lock up front so reading and advancing the
            # revision counter is atomic with the row writes it numbers.
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "UPDATE pages SET stat_key = ? WHERE filename = ? AND sha256 = ?"
                f" AND input_format = '{EMBEDDING_INPUT_FORMAT}'",
                restats,
            )
            if not rows and not stale:
                return report
            revision = int(
                conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0]
            )
            for row in rows:
                filename = row[0]
                snapshot = stored.get(filename)
                revision += 1
                if snapshot is None:
                    applied = conn.execute(
                        "INSERT OR IGNORE INTO pages"
                        " (filename, frontmatter, last_modified, sha256, embedding, input_format,"
                        " stat_key, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (*row, revision),
                    ).rowcount
                    added += applied
                else:
                    applied = conn.execute(
                        "UPDATE pages SET frontmatter = ?, last_modified = ?, sha256 = ?,"
                        " embedding = ?, input_format = ?, stat_key = ?, revision = ?"
                        " WHERE filename = ? AND revision = ?",
                        (*row[1:], revision, filename, snapshot[2]),
                    ).rowcount
                    updated += applied
                if not applied:
                    logger.info(skipped_message, filename)
            for filename in stale:
                revision += 1  # a delete changes the token that _rows() caches on
                applied = conn.execute(
                    "DELETE FROM pages WHERE filename = ? AND revision = ?",
                    (filename, stored[filename][2]),
                ).rowcount
                removed += applied
                if not applied:
                    logger.info(skipped_message, filename)
            conn.execute("UPDATE meta SET value = ? WHERE key = 'revision'", (str(revision),))
        return SyncReport(
            added=added, updated=updated, removed=removed, unchanged=report.unchanged
        )

    def _rows(self) -> dict[int, _Rows]:
        """All index rows, decoded and grouped by embedding dimension.

        The decoded rows are cached until the index's change token — its
        ``generation``, its ``revision`` counter (advanced by every insert,
        update and delete) and its row count (which also catches deletes by
        pre-token processes) — changes.  The token is read before the rows,
        so a write landing in between only makes the cache newer than its
        token and the next call reloads.
        """
        import numpy as np

        with closing(self._connect()) as conn:
            token = conn.execute(
                "SELECT (SELECT value FROM meta WHERE key = 'generation'),"
                " (SELECT value FROM meta WHERE key = 'revision'), COUNT(*) FROM pages"
            ).fetchone()
            cache = self._cache
            if cache is not None and cache[0] == token:
                return cache[1]
            rows = conn.execute(
                "SELECT filename, frontmatter, embedding FROM pages ORDER BY filename"
            ).fetchall()
        by_dims: dict[int, list[tuple[str, str, bytes]]] = {}
        for filename, frontmatter_json, blob in rows:
            by_dims.setdefault(len(blob) // 4, []).append((filename[:-3], frontmatter_json, blob))
        groups = {
            dims: _Rows(
                names=[row[0] for row in group],
                frontmatter=[row[1] for row in group],
                vectors=np.frombuffer(b"".join(row[2] for row in group), dtype=np.float32).reshape(
                    len(group), dims
                ),
            )
            for dims, group in by_dims.items()
        }
        self._cache = (token, groups)
        return groups

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
        """Return the *k* pages most similar to *query*.

        Rows whose stored dimension differs from the query's (written by a
        different embedder into a shared index file) are skipped with a
        warning.

        Args:
            query: Natural-language query.
            k: Maximum number of hits.
            min_score: Drop hits whose cosine similarity is not above this
                value. The default drops orthogonal pages (no overlap at all);
                pass ``-1.0`` to always return *k* hits when the index has them.

        Returns:
            Hits sorted by descending score (ties in page-name order).
        """
        import numpy as np

        if k <= 0:
            return []
        query_vector = self.embed_query(query)
        groups = self._rows()
        for dims, group in groups.items():
            if dims != len(query_vector):
                logger.warning(
                    "Skipping %d row(s): stored dimension %d != query dimension %d "
                    "(delete %s to rebuild the index)",
                    len(group.names),
                    dims,
                    len(query_vector),
                    self.path,
                )
        rows = groups.get(len(query_vector))
        if rows is None:
            return []
        # float32 BLAS scores only select candidates: every row within the
        # float32 error bound of the k-th best (or of min_score) is rescored
        # exactly with math.sumprod, so scores, ties and the min_score cut
        # are the same as an exact scan (an orthogonal page scores 0.0).
        approx = rows.vectors @ np.asarray(query_vector, dtype=np.float32)
        slack = float32_dot_error(len(query_vector))
        floor = min_score - slack
        if len(approx) > k:
            kth_best = float(np.partition(approx, len(approx) - k)[len(approx) - k])
            floor = max(floor, kth_best - slack)
        scored = sorted(
            (-math.sumprod(rows.vectors[i].tolist(), query_vector), int(i))
            for i in np.flatnonzero(approx >= floor)
        )
        return [
            SearchHit(
                name=rows.names[i],
                score=-negated,
                frontmatter=json.loads(rows.frontmatter[i]),
            )
            for negated, i in scored[:k]
            if -negated > min_score
        ]

    def near_duplicates(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Return page pairs whose embeddings are at least *threshold* similar.

        An exact pairwise scan computed as blocks of a matrix product, so
        thousands of pages take well under a second. Pairs whose stored
        dimensions differ (rows written by different embedders into a
        shared index file) are skipped, matching :meth:`search`.

        Args:
            threshold: Minimum cosine similarity for a pair to be reported.

        Returns:
            ``(name_a, name_b, score)`` triples, names in lexical order within
            each pair, sorted by descending score.
        """
        import numpy as np

        pairs: list[tuple[str, str, float]] = []
        for rows in self._rows().values():
            vectors = rows.vectors
            for start in range(0, len(rows.names), _PAIR_BLOCK_ROWS):
                block = vectors[start : start + _PAIR_BLOCK_ROWS] @ vectors.T
                floor = threshold - float32_dot_error(vectors.shape[1])
                for i, j in zip(*np.nonzero(block >= floor), strict=True):
                    if j <= start + i:
                        continue
                    # Exact rescoring, as in search().
                    score = math.sumprod(vectors[start + i].tolist(), vectors[j].tolist())
                    if score >= threshold:
                        pairs.append((rows.names[start + i], rows.names[j], score))
        pairs.sort(key=lambda pair: pair[2], reverse=True)
        return pairs

    def count(self) -> int:
        """Number of pages currently in the index."""
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0])

    def clear(self) -> None:
        """Delete the index file; the next :meth:`sync` rebuilds it from the pages."""
        if self.path.exists():
            self.path.unlink()
