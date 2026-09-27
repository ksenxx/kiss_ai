# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SQLite full-text block store behind the ``/git_extract_knowledge`` SEA.

A *block* is one unit of durable knowledge about a repository: a file, a
chunk of a file, a symbol definition, a commit, one commit's change to
one file, a tag, a branch, a contributor, a directory, or a note written
by the agent.  Blocks live in one SQLite database with an FTS5 index
(:class:`KnowledgeStore`); the database holds millions of rows without
loading anything into memory, which is what sets it apart from the
Markdown page memory of :mod:`kiss.core.memoryfield` (whose vector index
is decoded into RAM and is meant for thousands of curated pages).

Search is BM25 over the block title, text and key (:data:`RANK`).  A query is
tokenized here, never handed to FTS5 raw, so any text is a valid query:
first every token must match (precise), and when that leaves fewer than
*k* hits the union of the tokens fills the rest (recall).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocks (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    sha TEXT NOT NULL DEFAULT '',
    updated TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS blocks_kind ON blocks (kind);
CREATE INDEX IF NOT EXISTS blocks_path ON blocks (path);
CREATE INDEX IF NOT EXISTS blocks_sha ON blocks (sha);
CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
    title, text, key, content='blocks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS blocks_ai AFTER INSERT ON blocks BEGIN
    INSERT INTO blocks_fts(rowid, title, text, key)
    VALUES (new.id, new.title, new.text, new.key);
END;
CREATE TRIGGER IF NOT EXISTS blocks_ad AFTER DELETE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, title, text, key)
    VALUES ('delete', old.id, old.title, old.text, old.key);
END;
CREATE TRIGGER IF NOT EXISTS blocks_au AFTER UPDATE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, title, text, key)
    VALUES ('delete', old.id, old.title, old.text, old.key);
    INSERT INTO blocks_fts(rowid, title, text, key)
    VALUES (new.id, new.title, new.text, new.key);
END;
"""

KINDS = (
    "repo", "dir", "file", "chunk", "symbol", "commit", "change", "tag", "branch",
    "author", "note",
)
"""Every block kind the indexer and the agent write, in display order."""

WRITE_BATCH = 2000
"""Rows per ``executemany`` inside one transaction."""

RANK = "bm25(blocks_fts, 4.0, 1.0, 2.0)"
"""BM25 with the title weighted 4x and the key 2x over the text, so a definition or a
file named after the query outranks a block that merely mentions it often."""

HISTORY_RANK_FACTOR = {"change": 0.6, "commit": 0.8}
"""Rank factors of the history kinds (BM25 scores are negative: nearer zero ranks
lower).  A ``change`` block is a short patch and a ``commit`` block a short message,
which BM25 favours over the longer current-state blocks (``chunk``, ``symbol``,
``file``, ``note``), so a question about the code as it is found mostly old diffs
and tests that mention it: with these factors, a history block outranks a
current-state block only when its BM25 score is clearly better (1/0.6 = 1.7x for a
change, 1.25x for a commit).  ``kinds=["change"]`` still returns history alone."""

_ORDER = f"{RANK} * CASE b.kind " + " ".join(
    f"WHEN '{kind}' THEN {factor}" for kind, factor in HISTORY_RANK_FACTOR.items()
) + " ELSE 1.0 END"

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Block:
    """One unit of knowledge.

    Attributes:
        kind: One of :data:`KINDS`.
        key: Unique key, ``<kind>:<identifier>`` (e.g. ``file:src/a.py``,
            ``commit:abc123``, ``change:abc123:src/a.py``).
        title: One-line title shown in search results.
        text: Searchable body.
        path: Repository path the block is about, when it is about one file.
        sha: Blob or commit hash the block was built from (for change detection).
    """

    kind: str
    key: str
    title: str
    text: str
    path: str = ""
    sha: str = ""


@dataclass(frozen=True)
class Hit:
    """One search result: the block plus its rank (scaled BM25, lower is better) and a snippet."""

    block: Block
    rank: float
    snippet: str


STOP_WORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the this to what"
    " when where which who why with".split()
)
"""Question words dropped from a query when other words remain: they match almost every
block, so they only slow the search down and blur the ranking."""


def query_tokens(query: str) -> list[str]:
    """Split free text into the FTS tokens it consists of.

    Splits on every non-alphanumeric character (including ``_``, which the
    ``unicode61`` tokenizer also treats as a separator), lowercases, and
    drops duplicates while keeping order.  A token ending in ``*`` in the
    query keeps a trailing ``*`` for prefix matching.  :data:`STOP_WORDS`
    are dropped unless the query consists of nothing else.

    Args:
        query: Free text such as a question or an identifier.
    """
    tokens: list[str] = []
    for raw in query.split():
        prefix = raw.endswith("*")
        for token in _TOKEN_RE.findall(raw.lower()):
            if prefix and raw.lower().endswith(token + "*"):
                token += "*"
            if token not in tokens:
                tokens.append(token)
    content = [token for token in tokens if token not in STOP_WORDS]
    return content or tokens


def fts_expression(tokens: list[str], operator: str) -> str:
    """Join tokens into an FTS5 MATCH expression, quoting each as a phrase."""
    parts = []
    for token in tokens:
        if token.endswith("*"):
            parts.append(f'"{token[:-1]}"*')
        else:
            parts.append(f'"{token}"')
    return f" {operator} ".join(parts)


class KnowledgeStore:
    """The block store of one repository.

    Args:
        path: The SQLite database file; created with its parent directory
            on first use.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ----- meta -------------------------------------------------------------

    def get_meta(self, key: str, default: str = "") -> str:
        """Return the stored value of *key*, or *default*."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_meta(self, **values: str) -> None:
        """Store every keyword argument as a meta key."""
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                list(values.items()),
            )

    # ----- writes -----------------------------------------------------------

    def upsert(self, blocks: Iterable[Block], updated: str) -> int:
        """Insert or replace blocks by key, in batches of :data:`WRITE_BATCH`.

        Args:
            blocks: The blocks to store.
            updated: ISO timestamp recorded on every written row.

        Returns:
            The number of rows written.
        """
        sql = (
            "INSERT INTO blocks(kind, key, title, text, path, sha, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET kind = excluded.kind, title = excluded.title,"
            " text = excluded.text, path = excluded.path, sha = excluded.sha,"
            " updated = excluded.updated"
        )
        written = 0
        with closing(self._connect()) as conn:
            batch: list[tuple[str, ...]] = []
            for block in blocks:
                batch.append(
                    (block.kind, block.key, block.title, block.text, block.path, block.sha,
                     updated)
                )
                if len(batch) >= WRITE_BATCH:
                    with conn:
                        conn.executemany(sql, batch)
                    written += len(batch)
                    batch = []
            if batch:
                with conn:
                    conn.executemany(sql, batch)
                written += len(batch)
        return written

    def delete(self, kind: str = "", path: str = "", key: str = "") -> int:
        """Delete the blocks matching every given filter.

        Args:
            kind: Restrict to this kind.
            path: Restrict to blocks about this repository path.
            key: Delete exactly this key.

        Returns:
            The number of deleted rows.
        """
        clauses, params = _filters(kind=kind, path=path, key=key)
        if not clauses:
            raise ValueError("delete() needs at least one filter")
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(f"DELETE FROM blocks WHERE {' AND '.join(clauses)}", params)
            return int(cursor.rowcount)

    def paths_of_shas(self, shas: Iterable[str]) -> set[str]:
        """The paths whose ``change`` blocks belong to any commit in *shas*."""
        paths: set[str] = set()
        with closing(self._connect()) as conn:
            for sha in shas:
                rows = conn.execute(
                    "SELECT DISTINCT path FROM blocks WHERE sha = ? AND kind = 'change'", (sha,),
                ).fetchall()
                paths.update(str(row[0]) for row in rows)
        return paths

    def delete_shas(self, shas: Iterable[str]) -> int:
        """Delete the ``commit`` and ``change`` blocks of every commit in *shas*."""
        deleted = 0
        with closing(self._connect()) as conn, conn:
            for sha in shas:
                cursor = conn.execute(
                    "DELETE FROM blocks WHERE sha = ? AND kind IN ('commit', 'change')", (sha,),
                )
                deleted += int(cursor.rowcount)
        return deleted

    def delete_paths(self, paths: Iterable[str]) -> int:
        """Delete every block about any of *paths* (file, chunk and symbol blocks)."""
        deleted = 0
        with closing(self._connect()) as conn, conn:
            for path in paths:
                cursor = conn.execute(
                    "DELETE FROM blocks WHERE path = ? AND kind IN ('file', 'chunk', 'symbol')",
                    (path,),
                )
                deleted += int(cursor.rowcount)
        return deleted

    # ----- reads ------------------------------------------------------------

    def get(self, key: str) -> Block | None:
        """Return the block with *key*, or None."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT kind, key, title, text, path, sha FROM blocks WHERE key = ?", (key,)
            ).fetchone()
        return Block(*row) if row else None

    def file_shas(self) -> dict[str, str]:
        """Map every indexed file path to the blob hash its blocks were built from."""
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT path, sha FROM blocks WHERE kind = 'file'").fetchall()
        return {str(path): str(sha) for path, sha in rows}

    def keys(self, kind: str) -> list[str]:
        """Return the sorted keys of every block of *kind*."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key FROM blocks WHERE kind = ? ORDER BY key", (kind,)
            ).fetchall()
        return [str(row[0]) for row in rows]

    def iter_blocks(self, kind: str) -> Iterator[Block]:
        """Yield every block of *kind* in key order."""
        with closing(self._connect()) as conn:
            for row in conn.execute(
                "SELECT kind, key, title, text, path, sha FROM blocks WHERE kind = ? ORDER BY key",
                (kind,),
            ):
                yield Block(*row)

    def counts(self) -> dict[str, int]:
        """Return ``{kind: number of blocks}`` for every kind with at least one block."""
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT kind, COUNT(*) FROM blocks GROUP BY kind").fetchall()
        return {str(kind): int(count) for kind, count in rows}

    def search(
        self, query: str, k: int = 10, kinds: Iterable[str] = (), path_prefix: str = "",
    ) -> list[Hit]:
        """Return the *k* blocks that best match *query*.

        Args:
            query: Free text; see :func:`query_tokens`.
            k: Maximum number of hits.
            kinds: Restrict to these block kinds (empty: all).
            path_prefix: Restrict to blocks whose path starts with this prefix.

        Returns:
            Hits in rank order (BM25 scaled by :data:`HISTORY_RANK_FACTOR`
            for the history kinds) — every hit matching all tokens first,
            then hits matching some of them when the first pass returned
            fewer than *k* rows.  An empty query returns no hits.
        """
        tokens = query_tokens(query)
        if not tokens or k <= 0:
            return []
        hits = self._match(fts_expression(tokens, "AND"), k, tuple(kinds), path_prefix)
        if len(hits) < k and len(tokens) > 1:
            seen = {hit.block.key for hit in hits}
            for hit in self._match(fts_expression(tokens, "OR"), k, tuple(kinds), path_prefix):
                if hit.block.key not in seen and len(hits) < k:
                    hits.append(hit)
        return hits

    def _match(
        self, expression: str, k: int, kinds: tuple[str, ...], path_prefix: str,
    ) -> list[Hit]:
        clauses = ["blocks_fts MATCH ?"]
        params: list[object] = [expression]
        if kinds:
            clauses.append(f"b.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if path_prefix:
            clauses.append("substr(b.path, 1, ?) = ?")
            params.extend([len(path_prefix), path_prefix])
        params.append(k)
        sql = (
            f"SELECT b.kind, b.key, b.title, b.text, b.path, b.sha, {_ORDER},"
            " snippet(blocks_fts, 1, '[', ']', ' … ', 24)"
            " FROM blocks_fts JOIN blocks AS b ON b.id = blocks_fts.rowid"
            f" WHERE {' AND '.join(clauses)} ORDER BY {_ORDER} LIMIT ?"
        )
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Hit(Block(*row[:6]), float(row[6]), str(row[7])) for row in rows]


def _filters(**filters: str) -> tuple[list[str], list[str]]:
    """Turn non-empty keyword filters into SQL clauses and parameters."""
    clauses = [f"{column} = ?" for column, value in filters.items() if value]
    params = [value for value in filters.values() if value]
    return clauses, params
