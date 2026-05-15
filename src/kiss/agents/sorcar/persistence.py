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


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

def _default_kiss_dir() -> Path:
    """Return the KISS data directory, respecting ``KISS_HOME`` env var."""
    env = os.environ.get("KISS_HOME")
    return Path(env) if env else Path.home() / ".kiss"


_KISS_DIR = _default_kiss_dir()
_DB_PATH = _KISS_DIR / "sorcar.db"

_MAX_FILE_USAGE_ENTRIES = 10000

_MAX_FREQUENT_TASKS = 100


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


_HistoryEntry = dict[str, object]


# ---------------------------------------------------------------------------
# Per-thread connection management
# ---------------------------------------------------------------------------

# Legacy backward-compat variable: tests save/restore/None-ify this.
# Setting to None signals _get_db() to reconnect on the next call.
_db_conn: sqlite3.Connection | None = None

# Kept for backward compatibility — some callers import it directly.
_db_lock = threading.Lock()

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

_CLEAR_LAST_MODEL = "UPDATE model_usage SET is_last = 0 WHERE is_last = 1"


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
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _init_tables(conn)

    tl.conn = conn
    tl.gen = _db_generation
    tl.path = current_path
    _db_conn = conn  # backward compat: expose last-created connection
    return conn


def _most_recent_task_id(db: sqlite3.Connection, task: str | None) -> int | None:
    """Return the row id of the most recent run of *task*, or the latest row."""
    if task is not None:
        row = db.execute(
            "SELECT id FROM task_history WHERE task = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (task,),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT id FROM task_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def _add_task(task: str, chat_id: str = "") -> tuple[int, str]:
    """Append a task to the history and return ``(task_id, chat_id)``.

    When *chat_id* is ``""`` (new session), a new UUID-style string
    is generated as the chat session identifier.
    Otherwise the given *chat_id* is stored directly (continuation task).

    Thread-safe: all writes are protected by ``_rw_lock.write_lock()``.

    Args:
        task: The task description string.
        chat_id: Chat session identifier.  ``""`` starts a new session.

    Returns:
        ``(task_id, chat_id)`` — the inserted row id and the
        chat session identifier.
    """
    import uuid
    db = _get_db()
    with _rw_lock.write_lock():
        if chat_id == "":
            chat_id = uuid.uuid4().hex
        cursor = db.execute(
            "INSERT INTO task_history (timestamp, task, chat_id, result) VALUES (?, ?, ?, ?)",
            (time.time(), task, chat_id, "Agent Failed Abruptly"),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("sqlite did not return lastrowid")
        db.commit()
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


def _delete_task(task_id: int) -> bool:
    """Delete a task and its associated events from the database.

    Removes the events table rows that reference the given task_id,
    then removes the task_history row itself.

    Args:
        task_id: The primary key of the task_history row to delete.

    Returns:
        True if the task existed and was deleted, False otherwise.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        row = db.execute(
            "SELECT id FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return False
        db.execute("DELETE FROM events WHERE task_id = ?", (task_id,))
        db.execute("DELETE FROM task_history WHERE id = ?", (task_id,))
        db.commit()
        return True


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
        sql = _HISTORY_SELECT + "ORDER BY timestamp DESC LIMIT ? OFFSET ?"
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
    if not query:
        return ""
    with _rw_lock.read_lock():
        db = _get_db()
        escaped = query.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")
        row = db.execute(
            "SELECT task FROM task_history "
            "WHERE task GLOB ? AND LENGTH(task) > ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (escaped + "*", len(query)),
        ).fetchone()
        return row["task"] if row else ""


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
            + "WHERE task LIKE ? ESCAPE '\\' ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (f"%{escaped}%", limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def _get_history_entry(idx: int) -> _HistoryEntry | None:
    """Get a single history entry by its index (0 = most recent). Thread-safe.

    Args:
        idx: Zero-based index into the history list.

    Returns:
        The entry dict, or ``None`` if the index is out of range.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            _HISTORY_SELECT + "ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
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
    db = _get_db()
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        db.execute(
            "UPDATE task_history SET result = ? WHERE id = ?",
            (result, resolved),
        )
        db.commit()


def _save_task_extra(
    extra: dict[str, object],
    task_id: int | None = None,
    task: str | None = None,
) -> None:
    """Save extra metadata for a task as a JSON string.

    Stores a JSON-serialized dict in the ``extra`` column of
    ``task_history``.  Typical keys: ``model``, ``work_dir``,
    ``version``, ``tokens``, ``cost``, ``is_parallel``, ``is_worktree``.

    Args:
        extra: Dictionary of metadata to persist.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        db.execute(
            "UPDATE task_history SET extra = ? WHERE id = ?",
            (json.dumps(extra), resolved),
        )
        db.commit()


def _append_chat_event(
    event: dict[str, object],
    task_id: int | None = None,
    task: str | None = None,
) -> None:
    """Append a single event to the saved chat events for a task.

    Args:
        event: The event dict to append.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        row = db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM events WHERE task_id = ?",
            (resolved,),
        ).fetchone()
        next_seq = row["next_seq"] if row else 0
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
            (resolved, next_seq, json.dumps(event), time.time()),
        )
        db.execute(
            "UPDATE task_history SET has_events = 1 WHERE id = ?",
            (resolved,),
        )
        db.commit()


def _load_task_chat_id(task: str) -> str:
    """Return the chat_id for the most recent run of *task*, or ``""``.

    Args:
        task: The task description string.

    Returns:
        The string chat_id, or ``""`` if not found.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        task_id = _most_recent_task_id(db, task)
        if task_id is None:
            return ""
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE id = ?", (task_id,)
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
            "SELECT chat_id FROM task_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return str(row["chat_id"]) if row and row["chat_id"] else ""


def _list_recent_chats(limit: int = 10) -> list[dict[str, object]]:
    """List recent chat sessions with their tasks and results.

    Returns the most recent *limit* distinct chat sessions, ordered by
    most-recent-first.  Each entry contains the ``chat_id`` and a list
    of ``tasks`` (each with ``task``, ``result``, ``timestamp``) in
    chronological order.

    Args:
        limit: Maximum number of chat sessions to return.

    Returns:
        List of dicts, each with ``chat_id`` (str) and ``tasks``
        (list of dicts with ``task``, ``result``, ``timestamp``).
    """
    with _rw_lock.read_lock():
        db = _get_db()
        chat_rows = db.execute(
            "SELECT chat_id, MAX(timestamp) AS latest "
            "FROM task_history WHERE chat_id != '' "
            "GROUP BY chat_id ORDER BY latest DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result: list[dict[str, object]] = []
        for cr in chat_rows:
            cid = cr["chat_id"]
            tasks = db.execute(
                "SELECT task, result, timestamp FROM task_history "
                "WHERE chat_id = ? ORDER BY timestamp ASC",
                (cid,),
            ).fetchall()
            result.append({
                "chat_id": cid,
                "tasks": [
                    {"task": t["task"], "result": t["result"],
                     "timestamp": t["timestamp"]}
                    for t in tasks
                ],
            })
        return result


def _load_latest_chat_events_by_chat_id(
    chat_id: str,
) -> dict[str, object] | None:
    """Load the latest task and its events for a chat session.

    Finds the most recent task in the given chat session and returns
    its task description string and recorded events.

    Args:
        chat_id: The string chat session identifier.

    Returns:
        A dict with ``task`` (str), ``task_id`` (int), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if chat_id is ``""`` or has no
        tasks.
    """
    if not chat_id:
        return None
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT id, task, extra FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if not row:
            return None
        task_id = row["id"]
        task = row["task"]
        extra_str = row["extra"] or ""
        event_rows = db.execute(
            "SELECT event_json, timestamp FROM events WHERE task_id = ? ORDER BY seq",
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
        return {
            "task": task,
            "task_id": task_id,
            "events": events,
            "chat_id": chat_id,
            "extra": extra_str,
        }


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
        task = row["task"]
        chat_id = str(row["chat_id"] or "")
        extra_str = row["extra"] or ""
        event_rows = db.execute(
            "SELECT event_json, timestamp FROM events WHERE task_id = ? ORDER BY seq",
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
        return {
            "task": task,
            "task_id": task_id,
            "events": events,
            "chat_id": chat_id,
            "extra": extra_str,
        }


def _get_adjacent_task_by_chat_id(
    chat_id: str, current_task: str, direction: str
) -> dict[str, object] | None:
    """Return the adjacent task within a chat session, relative to *current_task*.

    Args:
        chat_id: The string chat session identifier.
        current_task: The current task description string used to find
            the reference timestamp within the chat.
        direction: ``"prev"`` for the earlier task, ``"next"`` for the
            later task in the same chat session.

    Returns:
        A dict with ``task`` (str), ``task_id`` (int) and ``events``
        (list of event dicts), or ``None`` if no adjacent task exists.
    """
    if not chat_id or not current_task:
        return None
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT id, timestamp FROM task_history "
            "WHERE chat_id = ? AND task = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (chat_id, current_task),
        ).fetchone()
        if not row:
            return None
        ts = row["timestamp"]

        if direction == "prev":
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? AND timestamp < ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (chat_id, ts),
            ).fetchone()
        else:
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? AND timestamp > ? "
                "ORDER BY timestamp ASC LIMIT 1",
                (chat_id, ts),
            ).fetchone()

        if not adj:
            return None

        adj_id = adj["id"]
        adj_task = adj["task"]
        event_rows = db.execute(
            "SELECT event_json, timestamp FROM events WHERE task_id = ? ORDER BY seq",
            (adj_id,),
        ).fetchall()
        events: list[dict[str, object]] = []
        for r in event_rows:
            try:
                ev = json.loads(r["event_json"])
                ev["_timestamp"] = r["timestamp"]
                events.append(ev)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Exception caught", exc_info=True)
        return {"task": adj_task, "task_id": adj_id, "events": events}


def _load_chat_context(chat_id: str) -> list[_HistoryEntry]:
    """Load all tasks and results for a chat session in chronological order.

    Args:
        chat_id: The string chat session identifier.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered by
        timestamp ascending (oldest first).
    """
    if not chat_id:
        return []
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT task, result FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,),
        ).fetchall()
        return [{"task": r["task"], "result": r["result"]} for r in rows]


def _load_model_usage() -> dict[str, int]:
    """Return model usage counts as ``{model_name: count}``."""
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute("SELECT model, count FROM model_usage").fetchall()
        return {r["model"]: r["count"] for r in rows}


def _load_last_model() -> str:
    """Return the name of the most recently selected model, or ``""``."""
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT model FROM model_usage WHERE is_last = 1 LIMIT 1"
        ).fetchone()
        return row["model"] if row else ""


def _save_last_model(model: str) -> None:
    """Persist the selected model name without incrementing usage count.

    Multi-statement transaction — uses _db_lock for atomicity.

    Args:
        model: The model name to save as the last-selected model.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute(_CLEAR_LAST_MODEL)
        db.execute(
            "INSERT INTO model_usage (model, count, is_last) VALUES (?, 0, 1) "
            "ON CONFLICT(model) DO UPDATE SET is_last = 1",
            (model,),
        )
        db.commit()


def _record_model_usage(model: str) -> None:
    """Increment a model's usage counter and mark it as last-used."""
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute(_CLEAR_LAST_MODEL)
        db.execute(
            "INSERT INTO model_usage (model, count, is_last) VALUES (?, 1, 1) "
            "ON CONFLICT(model) DO UPDATE SET count = count + 1, is_last = 1",
            (model,),
        )
        db.commit()


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


