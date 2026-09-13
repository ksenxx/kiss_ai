# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SQLite vector index over a directory of memory pages.

The index is a single SQLite file inside the memory directory, named after
the embedding model (``text-embedding-3-small.sqlite3``). It stores one row
per page: the page's frontmatter, its sha256 and its unit-length embedding
as a float32 BLOB. Rows are refreshed incrementally: a page is re-embedded
only when its sha256 changes, and rows for deleted pages are dropped.

Search is an exhaustive cosine scan using :func:`math.sumprod`. At
memoryfield scale (hundreds to a few thousand pages) this runs in
milliseconds and avoids any native extension (``sqlite-vec`` cannot be loaded
by the macOS system Python). The index is a cache: deleting the ``.sqlite3``
file loses nothing.
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
from array import array
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

from kiss.agents.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir, split_frontmatter

logger = logging.getLogger(__name__)

Embedder = Callable[[str], list[float]]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

HASHED_EMBEDDING_MODEL_CODE = "hashed-bow-v1"
HASHED_EMBEDDING_DIMS = 1024

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

    Args:
        model_name: Embedding model name from the model catalog, e.g.
            ``text-embedding-3-small`` or ``gemini-embedding-001``.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def __call__(self, text: str) -> list[float]:
        """Embed *text* with the configured model (lazily initialised)."""
        if self._model is None:
            from kiss.core.models.model_info import model as create_model

            self._model = create_model(self.model_name)
            self._model.initialize("")
        vector: list[float] = self._model.get_embedding(text, embedding_model=self.model_name)
        return vector


def model_code_for_filename(model_name: str) -> str:
    """Map a model name to the filename-safe prefix of its index file.

    Args:
        model_name: Embedding model name, possibly with a provider prefix
            such as ``BAAI/bge-base-en-v1.5``.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-.")


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


class VectorIndex:
    """Incrementally maintained cosine-similarity index over a :class:`MemoryDir`.

    Args:
        memory: The page directory to index.
        embed: Function mapping text to an embedding vector. Defaults to
            :class:`ModelEmbedder` with :data:`DEFAULT_EMBEDDING_MODEL`.
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
            embed = ModelEmbedder()
        self.embed = embed
        if model_code is None:
            model_code = str(getattr(embed, "model_name", "") or HASHED_EMBEDDING_MODEL_CODE)
        self.model_code: str = model_code
        self.path = memory.root / f"{model_code_for_filename(self.model_code)}.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        """Open the index database, creating the schema on first use.

        The returned connection holds no open transaction, so readers never
        block writers. Callers must close it (use ``closing(...)``).

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
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pages ("
                " filename TEXT PRIMARY KEY,"
                " frontmatter TEXT NOT NULL,"
                " last_modified REAL NOT NULL,"
                " sha256 BLOB NOT NULL,"
                " embedding BLOB NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('model_code', ?)",
                (self.model_code,),
            )
            conn.commit()
            stored_code = conn.execute(
                "SELECT value FROM meta WHERE key = 'model_code'"
            ).fetchone()[0]
            if stored_code != self.model_code:
                raise ValueError(
                    f"Index {self.path} was built with embedding model {stored_code!r}, "
                    f"not {self.model_code!r}; delete it or use a distinct model_code."
                )
        except BaseException:
            conn.close()
            raise
        return conn

    def embedding_input(self, raw: str) -> str:
        """The text that gets embedded for a page: the whole file, capped at 8192 bytes.

        Args:
            raw: Complete page text including frontmatter.
        """
        data = raw.encode("utf-8")
        if len(data) <= MAX_PAGE_BYTES:
            return raw
        return data[:MAX_PAGE_BYTES].decode("utf-8", errors="ignore")

    def sync(self) -> SyncReport:
        """Bring the index in line with the pages on disk.

        Pages whose sha256 is unchanged are skipped; new or modified pages
        are re-embedded; rows for pages that no longer exist are removed.
        Embeddings are computed with no database transaction open, and the
        rows are then written in one short transaction. A page that changes
        while it is being embedded is left for the next sync rather than
        stored with a stale vector.

        Returns:
            A :class:`SyncReport` with per-category counts.
        """
        added = updated = removed = unchanged = 0
        with closing(self._connect()) as conn:
            stored = {
                row[0]: bytes(row[1]) for row in conn.execute("SELECT filename, sha256 FROM pages")
            }

        rows: list[tuple[str, str, float, bytes, bytes]] = []
        on_disk: set[str] = set()
        for name in self.memory.page_names():
            filename = f"{name}.md"
            on_disk.add(filename)
            path = self.memory.page_path(name)
            data = path.read_bytes()
            digest = hashlib.sha256(data).digest()
            if stored.get(filename) == digest:
                unchanged += 1
                continue
            raw = data.decode("utf-8", errors="replace")
            frontmatter, _ = split_frontmatter(raw)
            vector = normalize(self.embed(self.embedding_input(raw)))
            if hashlib.sha256(path.read_bytes()).digest() != digest:
                logger.info(
                    "%s changed while being embedded; it will be indexed on the next sync", filename
                )
                continue
            rows.append(
                (
                    filename,
                    json.dumps(frontmatter, default=str),
                    path.stat().st_mtime,
                    digest,
                    serialize_float32(vector),
                )
            )
            if filename in stored:
                updated += 1
            else:
                added += 1

        stale = stored.keys() - on_disk
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT OR REPLACE INTO pages"
                " (filename, frontmatter, last_modified, sha256, embedding) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.executemany("DELETE FROM pages WHERE filename = ?", [(f,) for f in stale])
        removed = len(stale)
        return SyncReport(added=added, updated=updated, removed=removed, unchanged=unchanged)

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
        """Return the *k* pages most similar to *query*.

        Args:
            query: Natural-language query.
            k: Maximum number of hits.
            min_score: Drop hits whose cosine similarity is not above this
                value. The default drops orthogonal pages (no overlap at all);
                pass ``-1.0`` to always return *k* hits when the index has them.

        Returns:
            Hits sorted by descending score.
        """
        if k <= 0:
            return []
        query_vector = normalize(self.embed(query))
        hits: list[SearchHit] = []
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT filename, frontmatter, embedding FROM pages").fetchall()
        for filename, frontmatter_json, blob in rows:
            vector = deserialize_float32(bytes(blob))
            if len(vector) != len(query_vector):
                logger.warning(
                    "Skipping %s: stored dimension %d != query dimension %d "
                    "(delete %s to rebuild the index)",
                    filename,
                    len(vector),
                    len(query_vector),
                    self.path,
                )
                continue
            score = math.sumprod(vector, query_vector)
            if score <= min_score:
                continue
            hits.append(
                SearchHit(
                    name=filename[:-3],
                    score=score,
                    frontmatter=json.loads(frontmatter_json),
                )
            )
        hits.sort(key=attrgetter("score"), reverse=True)
        return hits[:k]

    def near_duplicates(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Return page pairs whose embeddings are at least *threshold* similar.

        An exhaustive pairwise scan over the stored embeddings — fine for the
        hundreds of pages a memoryfield is designed for. Pairs whose stored
        dimensions differ (rows written by different embedders into a shared
        index file) are skipped, matching :meth:`search`.

        Args:
            threshold: Minimum cosine similarity for a pair to be reported.

        Returns:
            ``(name_a, name_b, score)`` triples, names in lexical order within
            each pair, sorted by descending score.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT filename, embedding FROM pages ORDER BY filename"
            ).fetchall()
        vectors = [
            (str(filename)[:-3], deserialize_float32(bytes(blob)))
            for filename, blob in rows
        ]
        pairs: list[tuple[str, str, float]] = []
        for i, (name_a, vector_a) in enumerate(vectors):
            for name_b, vector_b in vectors[i + 1 :]:
                if len(vector_a) != len(vector_b):
                    continue
                score = math.sumprod(vector_a, vector_b)
                if score >= threshold:
                    pairs.append((name_a, name_b, score))
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
