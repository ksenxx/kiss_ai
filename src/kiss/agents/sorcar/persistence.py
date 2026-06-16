# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SQLite persistence for task history, chat events, model and file usage.

All data is stored in a single SQLite database at ``~/.kiss/sorcar.db``
using WAL mode for concurrent access.  Four tables hold task history,
chat events, model usage counters, and file usage counters.

Thread safety is achieved with:
- **Per-thread connections** via ``threading.local()`` so concurrent
  threads never share a Python ``sqlite3.Connection`` object (which
  avoids cursor-state interference).
- A **read-write lock** (``_rw_lock``) that allows concurrent readers
  but gives writers exclusive access, matching SQLite's own WAL
  constraint of at most one writer at a time.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read-write lock
# ---------------------------------------------------------------------------

class _RWLock:
    """Writer-preferring read-write lock.

    Multiple readers can hold the lock concurrently.  A writer gets
    exclusive access — no readers or other writers may proceed while a
    write lock is held.  Pending writers block new readers to prevent
    writer starvation.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._pending_writers = 0

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        """Acquire shared read access."""
        with self._cond:
            while self._writer or self._pending_writers > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        """Acquire exclusive write access."""
        with self._cond:
            self._pending_writers += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._pending_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


_rw_lock = _RWLock()

# Dedicated mutex for first-time DDL (``_init_tables``).  We intentionally
# do NOT reuse ``_rw_lock`` because most query helpers acquire the read
# lock *before* calling ``_get_db()`` — if ``_get_db()`` then tried to
# upgrade to ``_rw_lock.write_lock()`` for ``_init_tables`` it would
# self-deadlock against its own read lock.  Concurrent CREATE TABLE
# IF NOT EXISTS statements across threads are serialized through this
# small lock instead, which is independent of the reader/writer queue.
_init_tables_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Chat-context text cache
# ---------------------------------------------------------------------------
#
# Autocomplete (`_complete_from_active_file`) calls ``_load_chat_context`` on
# every keystroke when ``chat_id`` is supplied — that runs a SELECT against
# ``task_history`` and re-joins every prior task/result text in the session.
# The chat context only changes when a task is added or a task result is
# saved, so we cache the joined text per ``chat_id`` and invalidate on both
# write paths.  The cache lives at module scope (not on the server instance)
# so any caller — VS Code autocomplete, parallel sub-agents reusing the same
# chat — benefits transparently, and tests that swap the database connection
# don't accumulate stale invalidator callbacks.

_chat_context_text_cache: dict[str, str] = {}
_chat_context_cache_lock = threading.Lock()
# Invalidation generation counter.  Bumped on every cache invalidation
# (per-chat or global).  Readers capture the generation *before* the
# SQL read and only store their result if the generation hasn't
# advanced — preventing a slow reader from overwriting a fresh cache
# entry that a faster reader produced after a concurrent
# write+invalidate.
_chat_context_cache_gen: int = 0


def _invalidate_chat_context_cache(chat_id: str = "") -> None:
    """Drop the cached chat-context text for *chat_id*.

    When *chat_id* is empty, the entire cache is cleared (used by test
    fixtures that swap the underlying database file).
    """
    global _chat_context_cache_gen
    with _chat_context_cache_lock:
        if chat_id:
            _chat_context_text_cache.pop(chat_id, None)
        else:
            _chat_context_text_cache.clear()
        _chat_context_cache_gen += 1


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

def _default_kiss_dir() -> Path:
    """Return the KISS data directory, respecting ``KISS_HOME`` env var."""
    env = os.environ.get("KISS_HOME")
    return Path(env) if env else Path.home() / ".kiss"


_KISS_DIR = _default_kiss_dir()
_DB_PATH = _KISS_DIR / "sorcar.db"


def _current_db_path() -> str:
    """Return the active database path as a string.

    Used by asynchronous producers (the background event writer's
    enqueue path, the VS Code server's fire-and-forget follow-up
    thread) to stamp each pending write with the database it was
    produced against, so a late write can never land in a *different*
    database after ``_DB_PATH`` has been reassigned (test fixtures,
    daemon restarts pointed at another home dir).  Numeric
    ``task_history`` ids are only unique within one database file —
    AUTOINCREMENT prevents reuse inside a database, but a swapped
    database restarts the counter, so a stale id would otherwise
    resolve to an unrelated task's row.
    """
    return str(_DB_PATH)

_MAX_FILE_USAGE_ENTRIES = 10000

_MAX_FREQUENT_TASKS = 100


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


_HistoryEntry = dict[str, object]


# ---------------------------------------------------------------------------
# ``extra`` JSON encoding / decoding
# ---------------------------------------------------------------------------
#
# The ``task_history.extra`` column must always hold *valid RFC 8259*
# JSON: the SQL predicate ``_HISTORY_NOT_SUBAGENT`` classifies rows
# with SQLite's ``json_valid``/``json_type``, which reject the bare
# ``NaN`` / ``Infinity`` / ``-Infinity`` tokens that Python's
# ``json.dumps`` emits by default (``allow_nan=True``) and that
# ``json.loads`` accepts by default.  A non-finite float (e.g. a NaN
# ``cost``) written through plain ``json.dumps`` would therefore make
# the SQL-side and Python-side sub-agent detectors disagree — a
# sub-agent row would leak into the history sidebar while
# ``_list_recent_chats`` burned a limit slot on its chat.  All writers
# go through :func:`_dumps_extra` and all sub-agent classification
# goes through :func:`_parse_extra_dict` so the two sides can never
# diverge.

def _sanitize_non_finite(value: object) -> object:
    """Recursively replace non-finite floats (NaN/±Inf) with ``None``."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(v) for v in value]
    return value


def _dumps_extra(extra: dict[str, object]) -> str:
    """JSON-encode *extra* guaranteeing valid RFC 8259 output.

    Non-finite floats anywhere in the payload are replaced with
    ``None`` — plain ``json.dumps`` would serialise them as the bare
    ``NaN``/``Infinity`` tokens, which SQLite's ``json_valid`` (used by
    ``_HISTORY_NOT_SUBAGENT``) rejects.

    Args:
        extra: Metadata dict to serialise.

    Returns:
        A valid JSON object string.
    """
    try:
        return json.dumps(extra, allow_nan=False)
    except ValueError:
        sanitized = _sanitize_non_finite(extra)
        return json.dumps(sanitized, allow_nan=False)


def _reject_json_constant(token: str) -> object:
    """``parse_constant`` hook: reject ``NaN``/``Infinity`` tokens."""
    raise ValueError(f"non-RFC8259 JSON constant: {token}")


def _parse_extra_dict(extra: object) -> dict[str, object] | None:
    """Parse an ``extra`` column value as a strict-JSON dict.

    Mirrors SQLite's ``json_valid`` semantics (used by the
    ``_HISTORY_NOT_SUBAGENT`` predicate): the bare ``NaN`` /
    ``Infinity`` / ``-Infinity`` tokens — accepted by default by
    Python's ``json.loads`` but invalid RFC 8259 — are rejected, so a
    legacy corrupt row is classified identically by the SQL side and
    the Python side.

    Args:
        extra: The raw column value (``None``, ``""``, or a string).

    Returns:
        The parsed dict, or ``None`` when *extra* is empty, not a
        string, malformed, non-strict, or not a JSON object.
    """
    if not extra or not isinstance(extra, str):
        return None
    try:
        parsed = json.loads(extra, parse_constant=_reject_json_constant)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Per-thread connection management
# ---------------------------------------------------------------------------

# Legacy backward-compat variable: tests save/restore/None-ify this.
# Setting to None signals _get_db() to reconnect on the next call.
_db_conn: sqlite3.Connection | None = None

# Per-thread connection storage.  Each thread gets its own
# sqlite3.Connection keyed by (_db_generation, _DB_PATH).
_thread_local = threading.local()
_db_generation: int = 0


def _close_db() -> None:
    """Close all database connections and invalidate cached handles.

    Bumps the generation counter so that thread-local connections in
    other threads are detected as stale on their next ``_get_db()``
    call and replaced with fresh connections.
    """
    global _db_conn, _db_generation
    # Drain and stop the background writer first so it doesn't keep a
    # connection or seq cache that references the soon-to-be-replaced DB.
    _stop_event_writer()
    _db_generation += 1
    # Close current thread's connection
    tl_conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if tl_conn is not None:
        try:
            tl_conn.close()
        except Exception:
            pass
    _thread_local.conn = None
    _thread_local.gen = -1
    _thread_local.path = None
    _db_conn = None


_HISTORY_SELECT = (
    "SELECT id, timestamp, task, has_events, result, chat_id, extra "
    "FROM task_history "
)

# SQL predicate that is TRUE for every row that is NOT a sub-agent row.
# Must match :func:`_is_subagent_row` exactly: a sub-agent row is a row
# whose ``extra`` column is valid JSON whose TOP-LEVEL object carries a
# ``subagent`` key.  A raw substring test (``NOT LIKE '%"subagent"%'``)
# would wrongly hide regular rows whose extra merely *contains* the
# quoted word — e.g. a nested ``{"opts": {"subagent": false}}`` or a
# legacy malformed value.  ``CASE`` guarantees ``json_type`` is only
# evaluated on valid JSON (it raises on malformed input); for non-object
# top-level JSON, ``json_type(extra, '$.subagent')`` is NULL, matching
# ``_is_subagent_row``'s ``isinstance(parsed, dict)`` guard.
_HISTORY_NOT_SUBAGENT = (
    "(CASE WHEN json_valid(COALESCE(extra, '')) "
    "THEN json_type(COALESCE(extra, ''), '$.subagent') IS NULL "
    "ELSE 1 END)"
)


def _is_failed_result(result: str) -> bool:
    """Return True when the ``task_history.result`` text represents a
    failed task that should be flagged with a red dot in the history
    sidebar (``.sidebar-item-failed``).

    Recognized failure markers:

    * ``Task failed*`` — the standard in-process failure prefix
      written by ``_save_task_result`` for ``task_error`` events.
    * ``Agent Failed Abruptly`` — the sentinel inserted by
      ``_add_task`` that survives only when the host process was
      SIGKILL'd / OOM-killed / VS Code-reloaded mid-task before any
      Python ``finally`` could run.
    * ``Task terminated unexpectedly (process killed)`` — the rewrite
      that ``_recover_orphaned_tasks`` applies to surviving sentinel
      rows on fresh-server boot.

    User-stopped runs (``Task stopped by user``) are intentionally
    NOT marked failed here: the user explicitly halted them.
    """
    return (
        result.startswith("Task failed")
        or result == "Agent Failed Abruptly"
        or result == "Task terminated unexpectedly (process killed)"
    )


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id CHAR(32) DEFAULT '',
            extra TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES task_history(id),
            seq INTEGER NOT NULL,
            event_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL UNIQUE,
            count INTEGER DEFAULT 0,
            -- ``is_last`` is retained in the schema for backward
            -- compatibility with existing databases, but is no longer
            -- read or written: the last-selected model is now a user
            -- preference stored in ``config.json`` (see _load_last_model
            -- / _save_last_model).  Keeping the column here ensures the
            -- table schema does not change for new or existing databases.
            is_last INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS file_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            count INTEGER DEFAULT 0,
            last_used REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS frequent_tasks (
            task TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL DEFAULT 0
        );
    """)
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_th_timestamp
            ON task_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_th_task
            ON task_history(task);
        CREATE INDEX IF NOT EXISTS idx_th_chat_id
            ON task_history(chat_id);
        CREATE INDEX IF NOT EXISTS idx_ev_task_id
            ON events(task_id);
    """)


def _get_db() -> sqlite3.Connection:
    """Return a per-thread database connection, creating one if needed.

    Each calling thread gets its own ``sqlite3.Connection`` so that
    concurrent threads never share cursor state.  Connections are
    cached in ``threading.local()`` and invalidated when:

    * ``_db_generation`` is bumped (via ``_close_db()``),
    * ``_DB_PATH`` changes (test redirects), or
    * ``_db_conn`` is set to ``None`` (legacy test pattern).
    """
    global _db_conn
    tl = _thread_local
    tl_conn: sqlite3.Connection | None = getattr(tl, "conn", None)
    tl_gen: int = getattr(tl, "gen", -1)
    tl_path: str | None = getattr(tl, "path", None)
    current_path = str(_DB_PATH)
    _maybe_reset_caches(current_path)

    if (
        tl_conn is not None
        and tl_gen == _db_generation
        and tl_path == current_path
        and _db_conn is not None
    ):
        return tl_conn

    # Stale or missing — close old thread-local connection
    if tl_conn is not None:
        try:
            tl_conn.close()
        except Exception:
            pass

    _ensure_kiss_dir()
    if not _DB_PATH.exists():
        for suffix in ("-wal", "-shm"):
            stale_file = _DB_PATH.with_name(_DB_PATH.name + suffix)
            stale_file.unlink(missing_ok=True)
    conn = sqlite3.connect(
        current_path,
        check_same_thread=False,
        timeout=10,
        isolation_level=None,  # true autocommit — no implicit BEGIN
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    # Serialize first-time DDL across concurrent threads so CREATE TABLE
    # statements (run by every fresh thread-local connection) don't race
    # and trigger "database is locked" errors under heavy parallelism.
    # NOTE: must use a dedicated lock (not ``_rw_lock``) — many callers
    # already hold ``_rw_lock.read_lock()`` when invoking ``_get_db()``,
    # so reusing the RW lock for DDL would self-deadlock.
    with _init_tables_lock:
        _init_tables(conn)

    tl.conn = conn
    tl.gen = _db_generation
    tl.path = current_path
    _db_conn = conn  # backward compat: expose last-created connection
    return conn


def _most_recent_task_id(db: sqlite3.Connection, task: str | None) -> int | None:
    """Return the row id of the most recent run of *task*, or the latest row.

    Uses the total order ``(timestamp, id)`` — the ``id`` tiebreak keeps
    rows with equal timestamps (coarse clock ticks, imported databases)
    resolving to the genuinely latest insert, consistent with
    :func:`_load_latest_chat_events_by_chat_id`.
    """
    if task is not None:
        row = db.execute(
            "SELECT id FROM task_history WHERE task = ? "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (task,),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT id FROM task_history "
            "ORDER BY timestamp DESC, id DESC LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def _add_task(
    task: str,
    chat_id: str = "",
    extra: dict[str, object] | None = None,
) -> tuple[int, str]:
    """Append a task to the history and return ``(task_id, chat_id)``.

    When *chat_id* is ``""`` (new session), a new UUID-style string
    is generated as the chat session identifier.
    Otherwise the given *chat_id* is stored directly (continuation task).

    When *extra* is provided, the JSON-encoded dict is written into the
    ``extra`` column in the same INSERT so that values known at task
    creation time (model, work_dir, version, toggles) are immediately
    visible in the history sidebar — even before the task completes.
    Callers that need to add post-completion values (tokens, cost) can
    later call :func:`_save_task_extra` which rewrites the column
    (preserving any ``is_favorite`` flag set in the meantime).

    Thread-safe: all writes are protected by ``_rw_lock.write_lock()``.

    Args:
        task: The task description string.
        chat_id: Chat session identifier.  ``""`` starts a new session.
        extra: Optional dict of metadata to store immediately.

    Returns:
        ``(task_id, chat_id)`` — the inserted row id and the
        chat session identifier.
    """
    import uuid
    db = _get_db()
    extra_str = _dumps_extra(extra) if extra else ""
    with _rw_lock.write_lock():
        if chat_id == "":
            chat_id = uuid.uuid4().hex
        cursor = db.execute(
            "INSERT INTO task_history (timestamp, task, chat_id, result, extra) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), task, chat_id, "Agent Failed Abruptly", extra_str),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("sqlite did not return lastrowid")
        db.commit()
    # Invalidate the autocomplete chat-context cache so the new task's
    # text becomes visible to ghost-text suggestions on the next
    # keystroke.  Done outside the write lock to keep the critical
    # section short.
    _invalidate_chat_context_cache(chat_id)
    return row_id, chat_id


def _allocate_chat_id() -> str:
    """Pre-allocate a chat session id without keeping a task row.

    Generates a new UUID-style string that can be used as a unique
    chat session identifier.

    This is used by ``WorktreeSorcarAgent`` to name worktree branches
    *before* the first task in a session is persisted.

    Returns:
        A unique 32-character string suitable for use as a ``chat_id``.
    """
    import uuid
    return uuid.uuid4().hex


def _get_task_chat_id(task_id: int) -> str:
    """Return the chat_id of the task with the given row id, or ``""``.

    Args:
        task_id: The primary key of the task_history row.

    Returns:
        The chat_id string, or ``""`` if the row is not found or its
        chat_id column is empty.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        return str(row["chat_id"]) if row and row["chat_id"] else ""


def _chat_has_tasks(chat_id: str) -> bool:
    """Return True if the given chat_id has at least one task row.

    Args:
        chat_id: The chat session identifier string.

    Returns:
        True when at least one ``task_history`` row carries this
        ``chat_id``, otherwise False.  Returns False for ``""``.
    """
    if not chat_id:
        return False
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT 1 FROM task_history WHERE chat_id = ? LIMIT 1", (chat_id,),
        ).fetchone()
        return row is not None


def _subagent_child_ids(
    db: sqlite3.Connection, parent_task_id: int,
) -> list[int]:
    """Return ids of persisted sub-agent rows whose parent is *parent_task_id*.

    A sub-agent row carries ``{"subagent": {"parent_task_id": <id>}}``
    in its ``extra`` JSON (written by
    ``ChatSorcarAgent._run_tasks_parallel``).  The SQL ``LIKE``
    pre-filter is re-validated by ``json.loads`` — the same
    false-positive defense as
    :func:`_load_subagent_rows_by_parent_task_id`.  Callers must hold
    ``_rw_lock`` (read or write).

    Args:
        db: Active database connection.
        parent_task_id: Primary key of the parent ``task_history`` row.

    Returns:
        List of child row ids (possibly empty).
    """
    like = f'%"parent_task_id": {parent_task_id}%'
    rows = db.execute(
        "SELECT id, extra FROM task_history WHERE extra LIKE ?", (like,),
    ).fetchall()
    out: list[int] = []
    for r in rows:
        parsed = _parse_extra_dict(r["extra"])
        if parsed is None:
            continue
        sub = parsed.get("subagent")
        if isinstance(sub, dict) and sub.get("parent_task_id") == parent_task_id:
            out.append(int(r["id"]))
    return out


def _delete_task(task_id: int) -> bool:
    """Delete a task, its events, and its persisted sub-agent rows.

    Removes the events table rows that reference the given task_id,
    then removes the task_history row itself.  Sub-agent rows spawned
    by this task's ``run_parallel`` call
    (``extra.subagent.parent_task_id == task_id``) are cascade-deleted
    together with their events — recursively, because a sub-agent is a
    full ``ChatSorcarAgent`` that can itself fan out nested sub-agents
    whose rows point at the *child's* id, not the top-level parent's.
    They are reachable only through the parent chain (via
    :func:`_load_subagent_rows_by_parent_task_id`), so leaving any
    level behind would leak unreachable rows and make
    :func:`_chat_has_tasks` report a visually-empty chat as non-empty.

    Args:
        task_id: The primary key of the task_history row to delete.

    Returns:
        True if the task existed and was deleted, False otherwise.
    """
    # Drain pending queued events for this (or any) task_id before we
    # delete the row — otherwise the writer could insert an event row
    # referencing a deleted task_history id.
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        chat_id = (row["chat_id"] or "") if row is not None else ""
        doomed_ids = [task_id]
        if row is not None:
            # Breadth-first walk: a sub-agent is a full ChatSorcarAgent
            # that can itself call ``run_parallel``, so grandchildren
            # (and deeper) rows exist and must be cascade-deleted too.
            # The ``seen`` set guards against corrupt self/cyclic
            # ``parent_task_id`` references.
            seen = {task_id}
            frontier = [task_id]
            while frontier:
                next_frontier: list[int] = []
                for parent_id in frontier:
                    for child_id in _subagent_child_ids(db, parent_id):
                        if child_id not in seen:
                            seen.add(child_id)
                            next_frontier.append(child_id)
                doomed_ids.extend(next_frontier)
                frontier = next_frontier
        for did in doomed_ids:
            db.execute("DELETE FROM events WHERE task_id = ?", (did,))
        cursor = db.execute(
            "DELETE FROM task_history WHERE id = ?", (task_id,)
        )
        for did in doomed_ids[1:]:
            db.execute("DELETE FROM task_history WHERE id = ?", (did,))
        for did in doomed_ids:
            _next_seq_cache.pop(did, None)
            _marked_has_events.discard(did)
        db.commit()
        deleted = (cursor.rowcount or 0) > 0
    # Invalidate the autocomplete chat-context cache so the deleted
    # task/result text stops being served to ghost-text suggestions.
    # Done outside the write lock to keep the critical section short.
    if deleted:
        _invalidate_chat_context_cache(chat_id)
    return deleted


def _load_history(limit: int = 0, offset: int = 0) -> list[_HistoryEntry]:
    """Load task history entries (most-recent-first). Thread-safe.

    Args:
        limit: Maximum number of entries to return.
            0 returns all entries (no cap).
        offset: Number of entries to skip before returning results.

    Returns:
        List of history entry dicts with ``id``, ``timestamp``,
        ``task``, ``has_events``, ``result``, and ``chat_id`` keys.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        effective_limit = limit if limit > 0 else -1
        sql = (
            _HISTORY_SELECT
            + f"WHERE {_HISTORY_NOT_SUBAGENT} "
            + "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?"
        )
        rows = db.execute(sql, (effective_limit, offset)).fetchall()
        return [dict(r) for r in rows]


def _prefix_match_task(query: str) -> str:
    """Find the most recent task starting with *query* (case-sensitive).

    Uses a SQL ``GLOB`` query for case-sensitive prefix matching,
    avoiding the need to load many rows into Python for prefix scanning.

    Args:
        query: The prefix string to match against task text.

    Returns:
        The full task string of the most recent match, or ``""`` if none.
    """
    matches = _prefix_match_tasks(query, limit=1)
    return matches[0] if matches else ""


def _prefix_match_tasks(query: str, limit: int = 8) -> list[str]:
    """Find recent unique tasks starting with *query* (case-sensitive).

    The SQL ``GLOB`` filter does case-sensitive prefix matching server
    side; in Python we then deduplicate identical task strings while
    preserving their most-recent-first ordering so the dropdown menu
    never shows the same suggestion twice.

    Args:
        query: The prefix string to match against task text.
        limit: Maximum number of distinct matches to return.

    Returns:
        Up to *limit* full task strings, most recent first.  Empty when
        *query* is empty or no task matches.
    """
    if not query or limit <= 0:
        return []
    with _rw_lock.read_lock():
        db = _get_db()
        escaped = query.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")
        # Over-fetch so that duplicate task strings (a single task run
        # many times) still leave room for *limit* distinct entries.
        rows = db.execute(
            "SELECT task FROM task_history "
            "WHERE task GLOB ? AND LENGTH(task) > ? "
            f"AND {_HISTORY_NOT_SUBAGENT} "
            "ORDER BY timestamp DESC, id DESC LIMIT ?",
            (escaped + "*", len(query), limit * 4),
        ).fetchall()
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        task = row["task"]
        if task in seen:
            continue
        seen.add(task)
        out.append(task)
        if len(out) >= limit:
            break
    return out


def _search_history(
    query: str, limit: int = 50, offset: int = 0
) -> list[_HistoryEntry]:
    """Search history entries by substring match. Thread-safe.

    Args:
        query: Case-insensitive substring to match against task text.
        limit: Maximum number of matching entries to return.
        offset: Number of entries to skip before returning results.

    Returns:
        List of matching entries, most-recent-first.
    """
    if not query:
        return _load_history(limit=limit, offset=offset)
    with _rw_lock.read_lock():
        db = _get_db()
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = db.execute(
            _HISTORY_SELECT
            + "WHERE task LIKE ? ESCAPE '\\' "
            + f"AND {_HISTORY_NOT_SUBAGENT} "
            + "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (f"%{escaped}%", limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def _get_history_entry(idx: int) -> _HistoryEntry | None:
    """Get a single history entry by its index (0 = most recent). Thread-safe.

    Returns:
        The entry dict, or ``None`` if the index is out of range.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            _HISTORY_SELECT
            + f"WHERE {_HISTORY_NOT_SUBAGENT} "
            + "ORDER BY timestamp DESC, id DESC LIMIT 1 OFFSET ?",
            (idx,),
        ).fetchone()
        return dict(row) if row else None


def _resolve_task_id(
    db: sqlite3.Connection,
    task_id: int | None,
    task: str | None,
) -> int | None:
    """Resolve a stable row id, falling back to the most recent task.

    Args:
        db: Active database connection.
        task_id: Explicit row id when available.
        task: Fallback task description for legacy callers.

    Returns:
        The resolved row id, or ``None`` if not found.
    """
    if task_id is not None:
        row = db.execute(
            "SELECT id FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        return row["id"] if row else None
    return _most_recent_task_id(db, task)


def _log_orphaned_task_forensics(
    db: sqlite3.Connection,
    not_in_clause: str,
) -> None:
    """Log diagnostic info for each row still carrying the orphan sentinel.

    Called by :func:`_recover_orphaned_tasks` (under the write lock)
    before it rewrites the sentinel rows, so the startup log captures
    exactly which tasks were interrupted and their last recorded
    state.  This is the primary forensic evidence when a kill
    (SIGKILL / OOM / VS Code reload) prevents the normal
    ``_save_task_result`` → ``_append_chat_event`` finally block from
    running.

    Args:
        db: Active database connection.
        not_in_clause: SQL fragment excluding still-active task ids
            (``""`` or ``"AND id NOT IN (...)"``).
    """
    diag_rows = db.execute(
        "SELECT id, task, chat_id, extra FROM task_history "
        "WHERE result = ? " + not_in_clause,
        ("Agent Failed Abruptly",),
    ).fetchall()
    for row in diag_rows:
        task_id_val = row["id"]
        # Fetch the last few events to show where the agent was
        last_events = db.execute(
            "SELECT seq, event_json, timestamp FROM events "
            "WHERE task_id = ? ORDER BY seq DESC LIMIT 3",
            (task_id_val,),
        ).fetchall()
        last_event_summaries = []
        for ev in last_events:
            try:
                ev_data = json.loads(ev["event_json"])
                ev_type = ev_data.get("type", "unknown")
                ev_ts = ev["timestamp"]
            except Exception:
                ev_type = "parse_error"
                ev_ts = 0
            last_event_summaries.append(
                f"seq={ev['seq']} type={ev_type} ts={ev_ts:.1f}"
            )
        extra_str = row["extra"] or "{}"
        try:
            extra = json.loads(extra_str)
            model_name = extra.get("model", "unknown")
            start_ts = extra.get("startTs", 0)
            steps = extra.get("steps", "?")
            cost = extra.get("cost", "?")
        except Exception:
            model_name = "unknown"
            start_ts = 0
            steps = "?"
            cost = "?"
        task_preview = (row["task"] or "")[:120]
        logger.warning(
            "Orphaned task recovered: id=%s chat_id=%s model=%s "
            "startTs=%s steps=%s cost=%s task=%r last_events=[%s]",
            task_id_val,
            row["chat_id"] or "",
            model_name,
            start_ts,
            steps,
            cost,
            task_preview,
            "; ".join(last_event_summaries),
        )


def _recover_orphaned_tasks(active_task_ids: set[int]) -> int:
    """Replace the ``"Agent Failed Abruptly"`` sentinel on dead rows.

    The sentinel is written by :func:`_add_task` at task-creation
    time and is supposed to be overwritten by
    :func:`_save_task_result` from ``_TaskRunnerMixin._run_task_inner``'s
    cleanup ``finally``.  When the host process is killed externally
    (SIGKILL, VS Code extension reload, OOM) mid-task the Python
    ``finally`` never runs and the sentinel survives in the history
    sidebar verbatim as "Agent Failed Abruptly".  Catching the
    in-process ``BaseException`` variants in the task runner cannot
    cover this case because no Python code runs to catch anything.

    The remedy is a startup-time sweep: on every fresh ``VSCodeServer``
    instantiation (one per Python process) we scan ``task_history``
    for any row that still carries the sentinel AND whose id is not
    in *active_task_ids* (the currently-running tasks in THIS
    process), and rewrite ``result`` to a diagnostic message that
    truthfully describes what happened.

    Args:
        active_task_ids: Row ids that are still being processed in
            the current process and must therefore NOT be rewritten.
            Pass an empty set at fresh-server startup — by then any
            row carrying the sentinel must belong to a prior,
            now-dead process.

    Returns:
        The number of rows whose ``result`` column was rewritten.
    """
    db = _get_db()
    # Inline the ids; SQLite has a hard limit on bound parameters
    # (~999) but we never have that many live tasks, and the ids are
    # integers so there is no injection surface.
    not_in_clause = (
        f"AND id NOT IN ({','.join(str(int(t)) for t in active_task_ids)})"
        if active_task_ids
        else ""
    )
    sql = (
        "UPDATE task_history SET result = ? WHERE result = ? " + not_in_clause
    )
    with _rw_lock.write_lock():
        _log_orphaned_task_forensics(db, not_in_clause)
        cursor = db.execute(
            sql,
            (
                "Task terminated unexpectedly (process killed)",
                "Agent Failed Abruptly",
            ),
        )
        rowcount = cursor.rowcount or 0
        db.commit()
    if rowcount:
        logger.warning(
            "Recovered %d orphaned task(s) from prior process kill",
            rowcount,
        )
        # Updated rows could belong to any chat — clear the entire
        # autocomplete chat-context cache so stale entries don't
        # surface a stale "Agent Failed Abruptly" line.
        _invalidate_chat_context_cache("")
    return rowcount


def _shutdown_persist_in_flight_results(task_ids: set[int]) -> int:
    """Pre-emptive sentinel rewrite for in-flight tasks during shutdown.

    Called by :meth:`RemoteAccessServer._stop_active_agent_tasks` BEFORE
    the worker threads are signalled to stop.  For each row in
    *task_ids* that still carries the ``"Agent Failed Abruptly"``
    sentinel (set by :func:`_add_task` at task creation time and
    normally overwritten by :func:`_save_task_result` from
    ``_TaskRunnerMixin._run_task_inner``'s cleanup ``finally``), the
    column is rewritten to ``"Task interrupted by server
    restart/shutdown"``.

    This is a safety net for the failure mode where the worker thread
    cannot reach ``_save_task_result`` before the process exits — e.g.
    because it is wedged in C code (a blocking LLM API call ignoring
    ``KeyboardInterrupt``) or its cleanup ``finally`` exceeds the
    shutdown timeout.  Without the pre-emptive rewrite, the row stays
    at the sentinel and the next startup's orphan sweep
    (:func:`_recover_orphaned_tasks`) rewrites it to ``"Task
    terminated unexpectedly (process killed)"`` — the silent failure
    mode users report as "the agent was killed mid-task".

    Workers that *do* manage to finish their cleanup will overwrite
    this placeholder with a more detailed message (e.g. the per-task
    summary or "Task interrupted by server restart/shutdown" from the
    same cleanup path) — that ordering is fine because we set the
    placeholder BEFORE signalling the workers.

    Only rows still at the sentinel are touched, so a task that
    already completed cleanly (its row carrying a real result) is
    never clobbered.

    Args:
        task_ids: Row ids whose still-pending sentinel rows should be
            pre-emptively rewritten.

    Returns:
        The number of rows whose ``result`` column was rewritten.
    """
    if not task_ids:
        return 0
    db = _get_db()
    # Inline ids (integers, no injection surface; SQLite parameter
    # limit is ~999 and the active-task set is always far smaller).
    ids_clause = ",".join(str(int(t)) for t in task_ids)
    sql = (
        f"UPDATE task_history SET result = ? "
        f"WHERE id IN ({ids_clause}) AND result = ?"
    )
    affected_chat_ids: list[str] = []
    with _rw_lock.write_lock():
        # Capture chat_ids of rows we are about to rewrite so we can
        # invalidate the autocomplete chat-context cache afterwards.
        rows = db.execute(
            f"SELECT chat_id FROM task_history "
            f"WHERE id IN ({ids_clause}) AND result = ?",
            ("Agent Failed Abruptly",),
        ).fetchall()
        affected_chat_ids = [r["chat_id"] or "" for r in rows]
        cursor = db.execute(
            sql,
            (
                "Task interrupted by server restart/shutdown",
                "Agent Failed Abruptly",
            ),
        )
        rowcount = cursor.rowcount or 0
        db.commit()
    if rowcount:
        logger.warning(
            "Pre-emptively persisted shutdown result for %d in-flight task(s)",
            rowcount,
        )
        for chat_id in set(affected_chat_ids):
            _invalidate_chat_context_cache(chat_id)
    return rowcount


def _update_task_column(
    column: str,
    value: str,
    task_id: int | None,
    task: str | None,
) -> str | None:
    """Write *value* into *column* of the resolved ``task_history`` row.

    Drains pending queued events first so the column update is ordered
    after every event the task has emitted so far, then performs the
    UPDATE under the process-wide write lock.

    Args:
        column: Column name to update.  Must be a trusted literal
            (``"result"`` or ``"extra"``) — never user input.
        value: The new column value.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.

    Returns:
        The updated row's ``chat_id`` (possibly ``""``), or ``None``
        when no row could be resolved.
    """
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return None
        db.execute(
            f"UPDATE task_history SET {column} = ? WHERE id = ?",
            (value, resolved),
        )
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE id = ?", (resolved,),
        ).fetchone()
        db.commit()
        return (row["chat_id"] or "") if row is not None else ""


def _save_task_result(
    result: str,
    task_id: int | None = None,
    task: str | None = None,
) -> None:
    """Save just the result summary for a task (no event table changes).

    Args:
        result: The task result text to store in the history entry.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
    """
    affected_chat_id = _update_task_column("result", result, task_id, task)
    if affected_chat_id is None:
        return
    # Invalidate the autocomplete chat-context cache so the updated
    # result text becomes visible to ghost-text suggestions on the next
    # keystroke.  Done outside the write lock to keep the critical
    # section short.
    _invalidate_chat_context_cache(affected_chat_id)


def _set_task_favorite(task_id: int, is_favorite: bool) -> bool:
    """Toggle the ``is_favorite`` flag inside ``task_history.extra``.

    Performs a merge-update on the JSON-encoded ``extra`` column:
    loads any existing extra dict, sets ``is_favorite`` to the given
    boolean, and writes the merged object back.  Unlike
    :func:`_save_task_extra` (which overwrites the entire column),
    this preserves other keys such as ``tokens``, ``cost``, ``steps``,
    ``model``, and ``subagent``.

    Thread-safe: drains the background event queue first so the
    favourite flag write is ordered after any in-flight event inserts
    for the same task, then acquires the process-wide write lock.

    Args:
        task_id: Primary key of the ``task_history`` row to update.
        is_favorite: New value for the ``is_favorite`` flag.

    Returns:
        True when the row existed and was updated, False otherwise.
    """
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        row = db.execute(
            "SELECT extra FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        extra_raw = row["extra"] or ""
        extra_obj: dict[str, object]
        if extra_raw:
            strict = _parse_extra_dict(extra_raw)
            if strict is not None:
                extra_obj = strict
            else:
                # Legacy corrupt extra (e.g. a bare ``NaN`` token written
                # by the pre-``_dumps_extra`` code path): strict parsing
                # fails, so every reader and the ``_HISTORY_NOT_SUBAGENT``
                # SQL predicate classify this row as NOT a sub-agent row.
                # Recover the metadata leniently (the history sidebar
                # displays it via plain ``json.loads``) but drop the
                # never-effective top-level ``subagent`` key — otherwise
                # re-encoding through ``_dumps_extra`` would mint VALID
                # JSON whose ``subagent`` key suddenly starts classifying,
                # and starring the row would make it vanish from
                # ``_load_history`` / ``_list_recent_chats``.
                try:
                    parsed = json.loads(extra_raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                extra_obj = parsed if isinstance(parsed, dict) else {}
                extra_obj.pop("subagent", None)
        else:
            extra_obj = {}
        extra_obj["is_favorite"] = bool(is_favorite)
        db.execute(
            "UPDATE task_history SET extra = ? WHERE id = ?",
            (_dumps_extra(extra_obj), task_id),
        )
        db.commit()
        return True


def _save_task_extra(
    extra: dict[str, object],
    task_id: int | None = None,
    task: str | None = None,
) -> None:
    """Save extra metadata for a task as a JSON string.

    Stores a JSON-serialized dict in the ``extra`` column of
    ``task_history``.  Typical keys: ``model``, ``work_dir``,
    ``version``, ``tokens``, ``cost``, ``is_parallel``, ``is_worktree``.

    The ``is_favorite`` flag written by :func:`_set_task_favorite` is
    preserved: when the stored ``extra`` JSON carries ``is_favorite``
    and the new *extra* payload does not, the stored value is merged
    into the payload before writing.  Otherwise starring a task while
    it is still running would be silently undone by the
    completion-time extra write.

    Args:
        extra: Dictionary of metadata to persist.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
    """
    _flush_chat_events()
    db = _get_db()
    merged = dict(extra)
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        row = db.execute(
            "SELECT extra FROM task_history WHERE id = ?", (resolved,),
        ).fetchone()
        existing_raw = (row["extra"] or "") if row is not None else ""
        if existing_raw and "is_favorite" not in merged:
            try:
                existing = json.loads(existing_raw)
            except (json.JSONDecodeError, TypeError):
                existing = None
            if isinstance(existing, dict) and "is_favorite" in existing:
                merged["is_favorite"] = existing["is_favorite"]
        db.execute(
            "UPDATE task_history SET extra = ? WHERE id = ?",
            (_dumps_extra(merged), resolved),
        )
        db.commit()


# ---------------------------------------------------------------------------
# Background event writer
# ---------------------------------------------------------------------------
#
# Hot-path callers (every printer broadcast from every running agent + every
# sub-agent thread) used to call ``_append_chat_event`` synchronously.  That
# acquired the process-wide write lock per event and ran three SQL statements
# plus a commit per call.  Under N parallel sub-agents this collapsed all
# event traffic onto a single mutex, dropping throughput from ~10k ev/s at
# N=1 to ~3k ev/s at N=8.
#
# The fix is a single background writer thread that drains events from a
# queue in batches: one ``executemany`` insert plus at most one
# ``UPDATE task_history`` per batch, under exactly one write-lock acquisition.
# Producers call ``_queue_chat_event`` (sub-microsecond enqueue).  Callers
# that depend on event-ordering relative to a subsequent ``UPDATE
# task_history`` (``_save_task_result``, ``_save_task_extra``,
# ``_append_chat_event``) call ``_flush_chat_events()`` first.

_event_queue: queue.Queue = queue.Queue()
_event_writer_thread: threading.Thread | None = None
_event_writer_lock = threading.Lock()
_event_writer_stop = threading.Event()
_next_seq_cache: dict[int, int] = {}
_marked_has_events: set[int] = set()
# Path the seq/has_events caches were last populated against.  When
# ``_DB_PATH`` is reassigned (test fixtures), the caches are stale and
# must be cleared on the next ``_get_db()`` reconnect.
_caches_db_path: str | None = None
_caches_lock = threading.Lock()


def _maybe_reset_caches(current_path: str) -> None:
    """Clear seq/has_events caches when ``_DB_PATH`` changes (test fixtures)."""
    global _caches_db_path
    if _caches_db_path == current_path:
        return
    with _caches_lock:
        if _caches_db_path == current_path:
            return
        _next_seq_cache.clear()
        _marked_has_events.clear()
        _caches_db_path = current_path

# Per-batch tuning.  256 events at 50 µs JSON encoding ≈ 13 ms of producer
# work bundled into one SQL transaction.  20 ms collection window keeps
# end-to-end latency under ~25 ms for any single event.
_BATCH_MAX = 256
_BATCH_WINDOW_S = 0.020


def _start_event_writer() -> None:
    """Lazily spawn the background event writer thread (idempotent)."""
    global _event_writer_thread
    if _event_writer_thread is not None and _event_writer_thread.is_alive():
        return
    with _event_writer_lock:
        if _event_writer_thread is not None and _event_writer_thread.is_alive():
            return
        _event_writer_stop.clear()
        t = threading.Thread(
            target=_event_writer_loop,
            name="kiss-event-writer",
            daemon=True,
        )
        _event_writer_thread = t
        t.start()


def _event_writer_loop() -> None:
    """Drain the event queue in batches and persist them."""
    while not _event_writer_stop.is_set():
        try:
            first = _event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if first is None:
            # Shutdown sentinel
            _event_queue.task_done()
            if _event_writer_stop.is_set():
                return
            continue
        batch: list[tuple[int, str, float, str]] = [first]
        deadline = time.monotonic() + _BATCH_WINDOW_S
        shutdown_pending = False
        while len(batch) < _BATCH_MAX:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = _event_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                _event_queue.task_done()
                shutdown_pending = _event_writer_stop.is_set()
                break
            batch.append(item)
        try:
            _write_event_batch(batch)
        except Exception:
            logger.debug("event writer batch failed", exc_info=True)
        finally:
            for _ in batch:
                _event_queue.task_done()
        if shutdown_pending:
            return


def _write_event_batch(batch: list[tuple[int, str, float, str]]) -> None:
    """Persist a batch of (task_id, event_json, timestamp, origin_db_path) rows.

    Rows whose ``origin_db_path`` no longer matches the active
    ``_DB_PATH`` are dropped: their numeric ``task_id`` belongs to the
    database that was active when they were enqueued, so writing them
    into the current database would attach them to an unrelated task
    that merely shares the same row id.
    """
    if not batch:
        return
    current_path = _current_db_path()
    batch = [row for row in batch if row[3] == current_path]
    if not batch:
        return
    db = _get_db()
    task_ids = {tid for (tid, _ej, _ts, _op) in batch}
    with _rw_lock.write_lock():
        # Initialise next-seq cache for any new task_ids.  All inserts for
        # a given task_id are serialised through this single writer thread,
        # so the cached counter is authoritative once seeded from the DB.
        for tid in task_ids:
            if tid not in _next_seq_cache:
                # Validate the task_id exists in task_history; skip events
                # whose task row was deleted.  Inserting a dangling event
                # would raise an IntegrityError (FK violation) that aborts
                # the whole batch, losing every other task's events.
                exists = db.execute(
                    "SELECT 1 FROM task_history WHERE id = ?", (tid,),
                ).fetchone()
                if exists is None:
                    continue
                row = db.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq "
                    "FROM events WHERE task_id = ?",
                    (tid,),
                ).fetchone()
                _next_seq_cache[tid] = row["next_seq"] if row else 0
        rows: list[tuple[int, int, str, float]] = []
        for tid, ev_json, ts, _op in batch:
            seq = _next_seq_cache.get(tid)
            if seq is None:
                # Task row deleted — drop the dangling event, keep the rest.
                continue
            _next_seq_cache[tid] = seq + 1
            rows.append((tid, seq, ev_json, ts))
        db.executemany(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        to_mark = [
            tid for tid in task_ids
            if tid in _next_seq_cache and tid not in _marked_has_events
        ]
        if to_mark:
            placeholders = ",".join("?" * len(to_mark))
            db.execute(
                f"UPDATE task_history SET has_events = 1 "
                f"WHERE id IN ({placeholders})",
                to_mark,
            )
            _marked_has_events.update(to_mark)
        db.commit()


def _queue_chat_event(
    event: dict[str, object],
    task_id: int,
    origin_db_path: str | None = None,
) -> None:
    """Asynchronously persist an event for a known task_id.

    Sub-microsecond enqueue from the producer's perspective.  A
    background writer thread (started lazily) batches enqueued events
    and persists them with one transaction per batch.

    Callers that need ordering relative to a subsequent synchronous
    write to ``task_history`` (``_save_task_result``, ``_save_task_extra``)
    must call ``_flush_chat_events()`` first.

    Args:
        event: The event dict to persist.
        task_id: Stable ``task_history`` row id.  Must be non-None.
        origin_db_path: Database path *task_id* was resolved against.
            Defaults to the active ``_DB_PATH``.  The background
            writer drops the event if the active database has changed
            since, because the numeric id would then point at an
            unrelated task in the new database (see
            :func:`_current_db_path`).
    """
    _event_queue.put((
        task_id,
        json.dumps(event),
        time.time(),
        origin_db_path or _current_db_path(),
    ))
    if _event_writer_thread is None or not _event_writer_thread.is_alive():
        _start_event_writer()


def _flush_chat_events() -> None:
    """Block until all queued events have been persisted.

    Safe to call when no events are queued (returns immediately).  MUST
    be called BEFORE acquiring ``_rw_lock.write_lock()`` from the same
    thread — the writer thread also takes that lock per batch, so
    calling this while holding the write lock would deadlock.
    """
    if _event_writer_thread is None:
        return
    # Mirror ``_queue_chat_event``: if the writer thread died (e.g. it was
    # stopped during a DB swap) while events are still queued, restart it
    # so the backlog gets drained.  Otherwise ``_event_queue.join()`` would
    # block forever waiting for a ``task_done()`` the dead writer can never
    # call.
    if not _event_writer_thread.is_alive() and _event_queue.unfinished_tasks:
        _start_event_writer()
    _event_queue.join()


def _stop_event_writer() -> None:
    """Drain and stop the writer thread.  Used by ``_close_db``/tests."""
    global _event_writer_thread, _caches_db_path
    t = _event_writer_thread
    if t is not None:
        try:
            _event_queue.join()
        except Exception:
            pass
        _event_writer_stop.set()
        try:
            _event_queue.put_nowait(None)
        except queue.Full:  # pragma: no cover — unbounded queue
            pass
        t.join(timeout=5)
        _event_writer_thread = None
        _event_writer_stop.clear()
    _next_seq_cache.clear()
    _marked_has_events.clear()
    _caches_db_path = None


def _append_chat_event(
    event: dict[str, object],
    task_id: int | None = None,
    task: str | None = None,
    origin_db_path: str | None = None,
) -> None:
    """Append a single event to the saved chat events for a task.

    Synchronous: completes the write before returning.  Callers that
    can tolerate asynchronous persistence should prefer
    ``_queue_chat_event`` instead.

    Args:
        event: The event dict to append.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
        origin_db_path: Database path *task_id* was resolved against
            (see :func:`_queue_chat_event`).  Late asynchronous
            callers (e.g. the follow-up suggestion thread) pass the
            path captured when the task completed so the event is
            dropped — instead of attached to an unrelated task with
            the same row id — if the active database has changed.
    """
    if origin_db_path is not None and origin_db_path != _current_db_path():
        return
    # Resolve (and validate) the row id up front — the queued write
    # path requires a concrete, existing ``task_history`` id.
    with _rw_lock.read_lock():
        db = _get_db()
        resolved = _resolve_task_id(db, task_id, task)
    if resolved is None:
        return
    # Reuse the single batched write path so the sync and async event
    # writers can never diverge.  The FIFO queue guarantees this event
    # lands AFTER any earlier ``_queue_chat_event`` calls for the same
    # task; the flush makes the write synchronous.
    _queue_chat_event(event, resolved, origin_db_path)
    _flush_chat_events()


def _task_has_events(task_id: int) -> bool:
    """Return whether any chat events are persisted for *task_id*.

    Flushes the asynchronous event queue first so events enqueued by a
    recording printer (which land on the events table via the background
    writer) are visible before the check.  Used by
    :meth:`ChatSorcarAgent.run` to decide whether it must synthesize a
    minimal replayable event stream (prompt + result) for runs that
    happened outside a chat webview — i.e. without a recording printer
    that would have persisted the live event stream.

    Args:
        task_id: Stable ``task_history`` row id.

    Returns:
        ``True`` if at least one row exists in ``events`` for *task_id*.
    """
    _flush_chat_events()
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT 1 FROM events WHERE task_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        return row is not None


def _load_task_chat_id(task: str) -> str:
    """Return the chat_id for the most recent run of *task*, or ``""``.

    Args:
        task: The task description string.

    Returns:
        The string chat_id, or ``""`` if not found.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE task = ? "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (task,),
        ).fetchone()
        return str(row["chat_id"]) if row and row["chat_id"] else ""


def _load_last_chat_id() -> str:
    """Return the chat_id of the most recently added task, or ``""``.

    Useful for resuming the last CLI session without manually tracking
    the chat_id.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT chat_id FROM task_history "
            "ORDER BY timestamp DESC, id DESC LIMIT 1"
        ).fetchone()
        return str(row["chat_id"]) if row and row["chat_id"] else ""


def _list_recent_chats(limit: int = 10) -> list[dict[str, object]]:
    """List recent chat sessions with their tasks and results.

    Returns the most recent *limit* distinct chat sessions, ordered by
    most-recent-first.  Each entry contains the ``chat_id`` and a list
    of ``tasks`` (each with ``task``, ``result``, ``timestamp``) in
    chronological order.

    Sub-agent rows (``extra.subagent``, see :func:`_is_subagent_row`)
    are excluded — they are an internal implementation detail of the
    parent's ``run_parallel`` tool call, exactly as in every other
    chat/history reader (:func:`_load_history`,
    :func:`_load_chat_context`, ...).  A chat whose only rows are
    sub-agent rows is omitted entirely.

    Args:
        limit: Maximum number of chat sessions to return.

    Returns:
        List of dicts, each with ``chat_id`` (str) and ``tasks``
        (list of dicts with ``task``, ``result``, ``timestamp``).
    """
    with _rw_lock.read_lock():
        db = _get_db()
        # Filter sub-agent rows inside the chat-selection query itself:
        # a chat whose only rows are sub-agent rows must not consume one
        # of the *limit* slots (it would then be skipped below, silently
        # returning fewer chats than exist), and a chat's recency must
        # be anchored to its latest REAL task, not a sub-agent row.
        chat_rows = db.execute(
            "SELECT chat_id, MAX(timestamp) AS latest "
            "FROM task_history WHERE chat_id != '' "
            f"AND {_HISTORY_NOT_SUBAGENT} "
            "GROUP BY chat_id ORDER BY latest DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result: list[dict[str, object]] = []
        for cr in chat_rows:
            cid = cr["chat_id"]
            tasks = db.execute(
                "SELECT task, result, timestamp, extra FROM task_history "
                "WHERE chat_id = ? ORDER BY timestamp ASC, id ASC",
                (cid,),
            ).fetchall()
            task_dicts = [
                {"task": t["task"], "result": t["result"],
                 "timestamp": t["timestamp"]}
                for t in tasks
                if not _is_subagent_row(t["extra"])
            ]
            if not task_dicts:
                continue
            result.append({"chat_id": cid, "tasks": task_dicts})
        return result


def _fetch_events_for_task_id(
    db: sqlite3.Connection, task_id: int,
) -> list[dict[str, object]]:
    """Load and decode the event rows for *task_id* in seq order.

    Each surviving event dict has its ``_timestamp`` field injected
    from the matching ``events.timestamp`` column.  Rows whose
    ``event_json`` fails to decode are silently dropped (logged at
    DEBUG level).  Callers must hold ``_rw_lock.read_lock()`` (or
    ``write_lock()``) when invoking this helper.

    Args:
        db: Active database connection.
        task_id: Primary key of the ``task_history`` row.

    Returns:
        List of event dicts with ``_timestamp`` injected.
    """
    event_rows = db.execute(
        "SELECT event_json, timestamp FROM events "
        "WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    events: list[dict[str, object]] = []
    for r in event_rows:
        try:
            ev = json.loads(r["event_json"])
            ev["_timestamp"] = r["timestamp"]
            events.append(ev)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Exception caught", exc_info=True)
    return events


def _events_session_dict(
    db: sqlite3.Connection,
    task_id: int,
    task: str,
    chat_id: str,
    extra: object,
) -> dict[str, object]:
    """Build the replay-session dict shared by both chat-events loaders.

    Callers must hold ``_rw_lock.read_lock()`` (or ``write_lock()``)
    because this fetches the event rows via
    :func:`_fetch_events_for_task_id`.

    Args:
        db: Active database connection.
        task_id: Primary key of the ``task_history`` row.
        task: The row's task text.
        chat_id: The session's chat id (possibly ``""``).
        extra: The raw ``extra`` column value.

    Returns:
        Dict with ``task``, ``task_id``, ``events``, ``chat_id``, and
        ``extra`` keys.
    """
    return {
        "task": task,
        "task_id": task_id,
        "events": _fetch_events_for_task_id(db, task_id),
        "chat_id": chat_id,
        "extra": extra or "",
    }


def _load_latest_chat_events_by_chat_id(
    chat_id: str,
) -> dict[str, object] | None:
    """Load the latest task and its events for a chat session.

    Finds the most recent NON-sub-agent task in the given chat session
    and returns its task description string and recorded events.
    Sub-agent rows (``extra.subagent`` present, see
    :func:`_is_subagent_row`) share the parent's ``chat_id`` and are
    persisted AFTER the parent row, so a chat-id-only lookup (e.g. the
    webview's post-restart ``resumeSession``) must skip them —
    otherwise a restored parent tab would replay the last sub-agent's
    events and be styled as a sub-agent tab.  Sub-agent rows are only
    ever loaded explicitly by task id
    (:func:`_load_chat_events_by_task_id`).

    Args:
        chat_id: The string chat session identifier.

    Returns:
        A dict with ``task`` (str), ``task_id`` (int), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if chat_id is ``""`` or has no
        non-sub-agent tasks.
    """
    if not chat_id:
        return None
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT id, task, extra FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp DESC, id DESC",
            (chat_id,),
        ).fetchall()
        for row in rows:
            if _is_subagent_row(row["extra"]):
                continue
            return _events_session_dict(
                db, row["id"], row["task"], chat_id, row["extra"]
            )
        return None


def _load_chat_events_by_task_id(
    task_id: int,
) -> dict[str, object] | None:
    """Load a specific task and its events by the task row ID.

    Unlike ``_load_latest_chat_events_by_chat_id`` which always picks
    the most recent task in a chat session, this loads the exact task
    identified by *task_id*.

    Args:
        task_id: The primary key of the ``task_history`` row.

    Returns:
        A dict with ``task`` (str), ``task_id`` (int), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if no such row exists.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT id, task, chat_id, extra FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        return _events_session_dict(
            db, task_id, row["task"], str(row["chat_id"] or ""), row["extra"]
        )


def _load_subagent_rows_by_parent_task_id(
    parent_task_id: int,
) -> list[dict[str, object]]:
    """Return persisted sub-agent rows whose parent is *parent_task_id*.

    Used by :meth:`VSCodeServer._replay_session` when the user clicks
    a parent task in the history sidebar: every sub-agent fanned out
    by the parent's ``run_parallel`` tool call is reopened in its own
    sub-agent tab so the loaded view mirrors the live execution
    layout.

    A sub-agent row is identified by ``task_history.extra`` parsing
    to a JSON object containing
    ``{"subagent": {"parent_task_id": <parent_task_id>}}`` — exactly
    the shape written by
    :meth:`ChatSorcarAgent._run_tasks_parallel`'s worker thread.

    Args:
        parent_task_id: Primary key of the parent ``task_history`` row.

    Returns:
        List of dicts ordered by ``task_history.id`` ASC (the order in
        which the parent enqueued sub-agents).  Each dict has
        ``task_id`` (int), ``task`` (str), ``chat_id`` (str),
        ``events`` (list of event dicts), and ``extra`` (str, the raw
        JSON column).  Empty list when no sub-agent rows exist.
    """
    if not isinstance(parent_task_id, int):
        return []
    out: list[dict[str, object]] = []
    with _rw_lock.read_lock():
        db = _get_db()
        # Pre-filter at the SQL level on the substring of the JSON
        # ``extra`` to avoid loading every history row into Python.
        # The exact match (parent id + integer-valued field) is then
        # re-validated by ``json.loads`` to defend against false
        # positives (e.g. a row whose unrelated free-form metadata
        # happened to embed the same substring).
        like = f'%"parent_task_id": {parent_task_id}%'
        rows = db.execute(
            "SELECT id, task, chat_id, extra FROM task_history "
            "WHERE extra LIKE ? ORDER BY id ASC",
            (like,),
        ).fetchall()
        for r in rows:
            extra_str = r["extra"] or ""
            parsed = _parse_extra_dict(extra_str)
            if parsed is None:
                continue
            sub = parsed.get("subagent")
            if not isinstance(sub, dict):
                continue
            if sub.get("parent_task_id") != parent_task_id:
                continue
            sub_task_id = int(r["id"])
            out.append({
                "task_id": sub_task_id,
                "task": r["task"],
                "chat_id": str(r["chat_id"] or ""),
                "events": _fetch_events_for_task_id(db, sub_task_id),
                "extra": extra_str,
            })
    return out


def _get_adjacent_task_by_chat_id(
    chat_id: str, current_task_id: int | None, direction: str
) -> dict[str, object] | None:
    """Return the adjacent task within a chat session, relative to *current_task_id*.

    Args:
        chat_id: The string chat session identifier.
        current_task_id: The DB row id of the current task used to find
            the reference timestamp within the chat.  Using the row id
            (rather than the task description string) ensures that
            duplicate task texts within the same chat are navigated
            unambiguously.
        direction: ``"prev"`` for the earlier task, ``"next"`` for the
            later task in the same chat session.

    Returns:
        A dict with ``task`` (str), ``task_id`` (int) and ``events``
        (list of event dicts), or ``None`` if no adjacent task exists.
    """
    if not chat_id or current_task_id is None:
        return None
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT id, timestamp FROM task_history "
            "WHERE id = ? AND chat_id = ?",
            (current_task_id, chat_id),
        ).fetchone()
        if not row:
            return None
        ts = row["timestamp"]

        # Sub-agent rows (those carrying a ``subagent`` marker in
        # ``extra``) are internal implementation detail of the parent's
        # ``run_parallel`` tool call and must NEVER appear when the user
        # adjacent-scrolls between tasks in the chat webview — they
        # already render inside the parent task's panel.  Filter them
        # out at the SQL level so the LIMIT 1 lands on the next *parent*
        # row rather than a sub-agent that happens to sit between two
        # parent tasks chronologically.
        # Adjacency uses the total order ``(timestamp, id)`` — the same
        # id tiebreak as ``_load_latest_chat_events_by_chat_id``'s
        # ``ORDER BY timestamp DESC, id DESC`` — so rows that share a
        # timestamp value (concurrent inserts, imported databases)
        # remain mutually reachable instead of strict ``<`` / ``>``
        # timestamp comparison silently skipping them.
        if direction == "prev":
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? "
                "AND (timestamp < ? OR (timestamp = ? AND id < ?)) "
                f"AND {_HISTORY_NOT_SUBAGENT} "
                "ORDER BY timestamp DESC, id DESC LIMIT 1",
                (chat_id, ts, ts, row["id"]),
            ).fetchone()
        else:
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? "
                "AND (timestamp > ? OR (timestamp = ? AND id > ?)) "
                f"AND {_HISTORY_NOT_SUBAGENT} "
                "ORDER BY timestamp ASC, id ASC LIMIT 1",
                (chat_id, ts, ts, row["id"]),
            ).fetchone()

        if not adj:
            return None

        adj_id = adj["id"]
        return {
            "task": adj["task"],
            "task_id": adj_id,
            "events": _fetch_events_for_task_id(db, adj_id),
        }


def _load_chat_context(chat_id: str) -> list[_HistoryEntry]:
    """Load all tasks and results for a chat session in chronological order.

    Sub-agent rows (those whose ``extra`` column carries a
    ``subagent`` key — set by :class:`ChatSorcarAgent._run_tasks_parallel`
    on every worker thread's task row) are filtered out.  Sub-agent
    tasks/results are an internal implementation detail of the
    parent's ``run_parallel`` tool call; surfacing them in the chat
    context would (a) pollute the LLM's "Previous tasks and results"
    augmentation built by
    :meth:`ChatSorcarAgent.build_chat_prompt` with N copies of every
    fan-out task, and (b) cause the parent tab's history panel to
    list the sub-agent rows alongside the parent task that already
    summarises them.

    Args:
        chat_id: The string chat session identifier.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered by
        timestamp ascending (oldest first), excluding sub-agent rows.
    """
    if not chat_id:
        return []
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT task, result, extra FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp ASC, id ASC",
            (chat_id,),
        ).fetchall()
        entries: list[_HistoryEntry] = []
        for r in rows:
            if _is_subagent_row(r["extra"]):
                continue
            entries.append({"task": r["task"], "result": r["result"]})
        return entries


def _is_subagent_row(extra: object) -> bool:
    """Return ``True`` when *extra* JSON encodes a sub-agent task row.

    A sub-agent row is identified by the presence of a ``subagent``
    key in the JSON object stored in ``task_history.extra`` — this
    key is written by
    :meth:`ChatSorcarAgent._run_tasks_parallel`'s worker thread for
    every fan-out task.  Empty / missing / malformed ``extra``
    values yield ``False`` so legacy rows that pre-date the
    ``subagent`` marker continue to be treated as regular parent
    tasks.

    Args:
        extra: The raw ``extra`` column value (``None``, ``""``, or a
            JSON-encoded string).

    Returns:
        ``True`` if *extra* parses as a dict containing a ``subagent``
        key, ``False`` otherwise.
    """
    parsed = _parse_extra_dict(extra)
    return parsed is not None and "subagent" in parsed


def _load_chat_context_text(chat_id: str) -> str:
    """Return the joined task+result text for *chat_id* with caching.

    Concatenates the ``task`` and ``result`` strings of every entry
    returned by :func:`_load_chat_context` with newline separators.
    The joined string is cached in ``_chat_context_text_cache`` and
    automatically invalidated by :func:`_add_task` and
    :func:`_save_task_result` so callers (notably the ghost-text
    autocomplete, which calls this on every keystroke) never re-run
    the SQL or rejoin the text while the chat context is unchanged.

    Args:
        chat_id: The string chat session identifier.  Empty string
            short-circuits to ``""``.

    Returns:
        Newline-joined concatenation of every prior task and result
        in the session, or ``""`` when *chat_id* is empty or no prior
        rows exist.
    """
    if not chat_id:
        return ""
    # Capture the per-chat generation BEFORE the SQL read so that any
    # concurrent ``_invalidate_chat_context_cache(chat_id)`` (driven by
    # a writer that committed during our read) bumps the counter and
    # makes us skip the store — preventing a stale overwrite of a
    # fresher value another reader may have just published.
    with _chat_context_cache_lock:
        cached = _chat_context_text_cache.get(chat_id)
        snapshot_gen = _chat_context_cache_gen
    if cached is not None:
        return cached
    parts: list[str] = []
    for entry in _load_chat_context(chat_id):
        task = entry.get("task")
        result = entry.get("result")
        if isinstance(task, str):
            parts.append(task)
        if isinstance(result, str):
            parts.append(result)
    text = "\n".join(parts)
    with _chat_context_cache_lock:
        # Only publish our result if no invalidation occurred between
        # the pre-read snapshot and now.  Otherwise our data is
        # potentially stale — leave whatever fresher entry (or empty
        # slot) is already there alone.
        if _chat_context_cache_gen == snapshot_gen:
            _chat_context_text_cache[chat_id] = text
    return text


def _load_model_usage() -> dict[str, int]:
    """Return model usage counts as ``{model_name: count}``."""
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute("SELECT model, count FROM model_usage").fetchall()
        return {r["model"]: r["count"] for r in rows}


def _load_last_model() -> str:
    """Return the name of the most recently selected model, or ``""``.

    The last-selected model is a persistent **user preference** stored
    in ``~/.kiss/config.json`` (under the ``last_model`` key) — *not*
    in the SQLite ``model_usage`` table, which now tracks only per-model
    usage counts.
    """
    from kiss.agents.vscode.vscode_config import load_config

    return str(load_config().get("last_model", "") or "")


def _save_last_model(model: str) -> None:
    """Persist the selected model name as a user preference.

    Writes the ``last_model`` key to ``~/.kiss/config.json`` (atomic).
    Does **not** touch the SQLite usage counters.

    Args:
        model: The model name to save as the last-selected model.
    """
    from kiss.agents.vscode.vscode_config import load_config, save_config

    cfg = load_config()
    cfg["last_model"] = model
    save_config(cfg)


def _record_model_usage(model: str) -> None:
    """Increment a model's usage counter and mark it as last-used.

    The usage ``count`` lives in the SQLite ``model_usage`` table; the
    "last selected" pointer is persisted separately to
    ``config.json`` via :func:`_save_last_model`.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute(
            "INSERT INTO model_usage (model, count) VALUES (?, 1) "
            "ON CONFLICT(model) DO UPDATE SET count = count + 1",
            (model,),
        )
        db.commit()
    _save_last_model(model)


def _load_file_usage() -> dict[str, int]:
    """Return file usage counts ordered oldest-first (by last_used).

    The returned dict preserves insertion order so that callers can
    derive recency from key position.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT path, count FROM file_usage ORDER BY last_used ASC"
        ).fetchall()
        return {r["path"]: r["count"] for r in rows}


def _record_file_usage(path: str) -> None:
    """Increment the access count for a file path atomically."""
    db = _get_db()
    now = time.time()
    with _rw_lock.write_lock():
        db.execute(
            "INSERT INTO file_usage (path, count, last_used) VALUES (?, 1, ?) "
            "ON CONFLICT(path) DO UPDATE SET count = count + 1, last_used = ?",
            (path, now, now),
        )
        row = db.execute("SELECT COUNT(*) FROM file_usage").fetchone()
        if row[0] > _MAX_FILE_USAGE_ENTRIES:
            db.execute(
                "DELETE FROM file_usage WHERE path NOT IN "
                "(SELECT path FROM file_usage ORDER BY last_used DESC LIMIT ?)",
                (_MAX_FILE_USAGE_ENTRIES,),
            )
        db.commit()


def _record_frequent_task(task: str) -> None:
    """Increment the run-count of *task* and refresh its timestamp.

    Upserts a row in the ``frequent_tasks`` table so that subsequent
    calls with the same *task* increment its ``count`` and update its
    ``timestamp`` to ``time.time()``.

    The table is capped at ``_MAX_FREQUENT_TASKS`` rows.  When inserting
    a brand-new task would exceed the cap, the row with the lowest
    ``count`` (and, on a count tie, the oldest ``timestamp``) is
    evicted before the insert completes.

    Args:
        task: The task description string.  Empty strings are ignored.
    """
    if not task:
        return
    db = _get_db()
    now = time.time()
    with _rw_lock.write_lock():
        existing = db.execute(
            "SELECT 1 FROM frequent_tasks WHERE task = ?", (task,),
        ).fetchone()
        if existing is None:
            row = db.execute("SELECT COUNT(*) FROM frequent_tasks").fetchone()
            if row[0] >= _MAX_FREQUENT_TASKS:
                db.execute(
                    "DELETE FROM frequent_tasks WHERE task = "
                    "(SELECT task FROM frequent_tasks "
                    "ORDER BY count ASC, timestamp ASC LIMIT 1)"
                )
        db.execute(
            "INSERT INTO frequent_tasks (task, count, timestamp) "
            "VALUES (?, 1, ?) "
            "ON CONFLICT(task) DO UPDATE SET "
            "count = count + 1, timestamp = ?",
            (task, now, now),
        )
        db.commit()


def _delete_frequent_task(task: str) -> bool:
    """Delete a row from the ``frequent_tasks`` table by task text.

    Args:
        task: The exact task description string identifying the row.

    Returns:
        True if a matching row existed and was deleted, False otherwise.
    """
    if not task:
        return False
    db = _get_db()
    with _rw_lock.write_lock():
        cursor = db.execute(
            "DELETE FROM frequent_tasks WHERE task = ?", (task,)
        )
        db.commit()
        return (cursor.rowcount or 0) > 0


def _load_frequent_tasks(limit: int = 50) -> list[dict[str, object]]:
    """Return the top *limit* most-frequent tasks (highest count first).

    On a tie in ``count``, the more recently used task (larger
    ``timestamp``) is returned first.

    Args:
        limit: Maximum number of rows to return.

    Returns:
        A list of dicts with keys ``task`` (str), ``count`` (int) and
        ``timestamp`` (float), ordered by ``count`` descending.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT task, count, timestamp FROM frequent_tasks "
            "ORDER BY count DESC, timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"task": r["task"], "count": r["count"], "timestamp": r["timestamp"]}
            for r in rows
        ]


