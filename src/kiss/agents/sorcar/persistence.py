"""SQLite persistence for task history, chat events, model and file usage.

All data is stored in a single SQLite database at ``~/.kiss/history.db``
using WAL mode for concurrent access.  Four tables hold task history,
chat events, model usage counters, and file usage counters.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _log_exc() -> None:
    logger.debug("Exception caught", exc_info=True)

_KISS_DIR = Path.home() / ".kiss"
_DB_PATH = _KISS_DIR / "history.db"

_MAX_FILE_USAGE_ENTRIES = 1000


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


_HistoryEntry = dict[str, object]


# ---------------------------------------------------------------------------
# Database connection (singleton)
# ---------------------------------------------------------------------------

_db_conn: sqlite3.Connection | None = None
_db_lock = threading.Lock()


_HISTORY_SELECT = (
    "SELECT id, timestamp, task, has_events, result, chat_id "
    "FROM task_history "
)

_CLEAR_LAST_MODEL = "UPDATE model_usage SET is_last = 0 WHERE is_last = 1"


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_th_timestamp
            ON task_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_th_task
            ON task_history(task);
        CREATE INDEX IF NOT EXISTS idx_th_chat_id
            ON task_history(chat_id);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES task_history(id),
            seq INTEGER NOT NULL,
            event_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ev_task_id
            ON events(task_id);

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
    """)


def _get_db() -> sqlite3.Connection:
    """Return the singleton database connection, creating it on first call.

    Sets WAL journal mode, enables foreign keys, and creates tables
    if they do not already exist.
    """
    global _db_conn
    if _db_conn is not None:
        return _db_conn
    with _db_lock:
        if _db_conn is not None:  # pragma: no cover — double-check lock race
            return _db_conn
        _ensure_kiss_dir()
        if not _DB_PATH.exists():
            for suffix in ("-wal", "-shm"):
                stale = _DB_PATH.with_name(_DB_PATH.name + suffix)
                stale.unlink(missing_ok=True)
        conn = sqlite3.connect(
            str(_DB_PATH), check_same_thread=False, timeout=10,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _init_tables(conn)
        _db_conn = conn
        return _db_conn


# ---------------------------------------------------------------------------
# Task history
# ---------------------------------------------------------------------------

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


def _generate_chat_id() -> str:
    """Generate a unique hex chat session ID. Thread-safe.

    Produces a 32-character lowercase hex string via UUID4 and verifies
    it does not already exist in the task_history table before returning.
    """
    db = _get_db()
    with _db_lock:
        while True:
            candidate = uuid.uuid4().hex
            row = db.execute(
                "SELECT 1 FROM task_history WHERE chat_id = ? LIMIT 1",
                (candidate,),
            ).fetchone()
            if row is None:  # pragma: no branch — UUID4 collision virtually impossible
                return candidate


def _add_task(task: str, chat_id: str = "") -> None:
    """Append a task to the history. Thread-safe.

    Single-statement write — SQLite WAL + busy_timeout handles contention.

    Args:
        task: The task description string.
        chat_id: Chat session identifier to associate this task with.
    """
    db = _get_db()
    db.execute(
        "INSERT INTO task_history (timestamp, task, chat_id, result) VALUES (?, ?, ?, ?)",
        (time.time(), task, chat_id, "Agent Failed Abrubptly"),
    )
    db.commit()


def _load_history(limit: int = 0, offset: int = 0) -> list[_HistoryEntry]:
    """Load task history entries (most-recent-first). Thread-safe.

    Args:
        limit: Maximum number of entries to return.
            0 returns all entries.
        offset: Number of entries to skip before returning results.

    Returns:
        List of history entry dicts with ``id``, ``timestamp``,
        ``task``, ``has_events``, ``result``, and ``chat_id`` keys.
    """
    db = _get_db()
    sql = _HISTORY_SELECT + "ORDER BY timestamp DESC"
    if limit > 0:
        sql += " LIMIT ? OFFSET ?"
        rows = db.execute(sql, (limit, offset)).fetchall()
    else:
        rows = db.execute(sql).fetchall()
    return [dict(r) for r in rows]


def _prefix_match_task(query: str) -> str:
    """Find the most recent task starting with *query* (case-insensitive).

    Uses a SQL ``LIKE 'prefix%'`` query with the ``idx_th_task`` index,
    avoiding the need to load many rows into Python for prefix scanning.

    Args:
        query: The prefix string to match against task text.

    Returns:
        The full task string of the most recent match, or ``""`` if none.
    """
    if not query:
        return ""
    db = _get_db()
    # Escape LIKE wildcards in the query, then append '%'
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    row = db.execute(
        "SELECT task FROM task_history "
        "WHERE task LIKE ? ESCAPE '\\' AND LENGTH(task) > ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (escaped + "%", len(query)),
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
    db = _get_db()
    rows = db.execute(
        _HISTORY_SELECT + "WHERE task LIKE ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (f"%{query}%", limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_history_entry(idx: int) -> _HistoryEntry | None:
    """Get a single history entry by its index (0 = most recent). Thread-safe.

    Args:
        idx: Zero-based index into the history list.

    Returns:
        The entry dict, or ``None`` if the index is out of range.
    """
    db = _get_db()
    row = db.execute(
        _HISTORY_SELECT + "ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
        (idx,),
    ).fetchone()
    return dict(row) if row else None


def _load_task_chat_events(
    task: str,
) -> list[dict[str, object]]:
    """Load chat events for a specific task.

    Returns the events for the most recent run of *task*.

    Args:
        task: The task description string.

    Returns:
        List of chat event dicts, or empty list if none stored.
    """
    db = _get_db()
    task_id = _most_recent_task_id(db, task)
    if task_id is None:
        return []
    rows = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    result: list[dict[str, object]] = []
    for r in rows:
        try:
            result.append(json.loads(r["event_json"]))
        except (json.JSONDecodeError, TypeError):
            _log_exc()
    return result


def _save_task_result(
    task: str,
    result: str,
) -> None:
    """Save just the result summary for a task (no event table changes).

    Updates only the ``result`` column of the target task_history row.
    Single-statement write — SQLite WAL + busy_timeout handles contention.

    Args:
        task: The task description string to look up.
        result: The task result text to store in the history entry.
    """
    db = _get_db()
    task_id = _most_recent_task_id(db, task)
    if task_id is None:
        return
    db.execute(
        "UPDATE task_history SET result = ? WHERE id = ?",
        (result, task_id),
    )
    db.commit()


def _set_latest_chat_events(
    events: list[dict[str, object]],
    task: str | None = None,
    result: str = "",
) -> None:
    """Save chat events for a task.

    Updates the ``has_events`` and ``result`` columns of the target
    task_history row and replaces all rows in the events table for
    that task.

    Args:
        events: The chat events to store.
        task: If given, find the history entry by task name.
              Otherwise update the most recent entry.
        result: The task result text to store in the history entry.
    """
    db = _get_db()
    task_id = _most_recent_task_id(db, task)
    if task_id is None:
        return
    has_ev = 1 if events else 0
    with _db_lock:
        db.execute(
            "UPDATE task_history SET has_events = ?, result = ? WHERE id = ?",
            (has_ev, result, task_id),
        )
        db.execute("DELETE FROM events WHERE task_id = ?", (task_id,))
        if has_ev:
            db.executemany(
                "INSERT INTO events (task_id, seq, event_json) VALUES (?, ?, ?)",
                [(task_id, i, json.dumps(ev)) for i, ev in enumerate(events)],
            )
        db.commit()


def _append_chat_event(task: str, event: dict[str, object]) -> None:
    """Append a single event to the saved chat events for a task.

    Adds the event at the end (next sequence number) of the events
    already stored for the most recent run of *task*.

    Args:
        task: The task description string.
        event: The event dict to append.
    """
    db = _get_db()
    task_id = _most_recent_task_id(db, task)
    if task_id is None:
        return
    with _db_lock:
        row = db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM events WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        next_seq = row["next_seq"] if row else 0
        db.execute(
            "INSERT INTO events (task_id, seq, event_json) VALUES (?, ?, ?)",
            (task_id, next_seq, json.dumps(event)),
        )
        db.commit()


def _load_task_chat_id(task: str) -> str:
    """Return the chat_id for the most recent run of *task*, or ``""``.

    Args:
        task: The task description string.

    Returns:
        The chat_id string, or empty string if not found.
    """
    db = _get_db()
    task_id = _most_recent_task_id(db, task)
    if task_id is None:
        return ""
    row = db.execute(
        "SELECT chat_id FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    return row["chat_id"] if row and row["chat_id"] else ""


def _load_last_chat_id() -> str:
    """Return the chat_id of the most recently added task, or ``""``.

    Useful for resuming the last CLI session without manually tracking
    the chat_id.
    """
    db = _get_db()
    row = db.execute(
        "SELECT chat_id FROM task_history ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    return row["chat_id"] if row and row["chat_id"] else ""


def _load_chat_context(chat_id: str) -> list[_HistoryEntry]:
    """Load all tasks and results for a chat session in chronological order.

    Args:
        chat_id: The chat session identifier.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered by
        timestamp ascending (oldest first).
    """
    if not chat_id:
        return []
    db = _get_db()
    rows = db.execute(
        "SELECT task, result FROM task_history "
        "WHERE chat_id = ? ORDER BY timestamp ASC",
        (chat_id,),
    ).fetchall()
    return [{"task": r["task"], "result": r["result"]} for r in rows]


# ---------------------------------------------------------------------------
# Model usage
# ---------------------------------------------------------------------------

def _load_model_usage() -> dict[str, int]:
    """Return model usage counts as ``{model_name: count}``."""
    db = _get_db()
    rows = db.execute("SELECT model, count FROM model_usage").fetchall()
    return {r["model"]: r["count"] for r in rows}


def _load_last_model() -> str:
    """Return the name of the most recently selected model, or ``""``."""
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
    with _db_lock:
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
    with _db_lock:
        db.execute(_CLEAR_LAST_MODEL)
        db.execute(
            "INSERT INTO model_usage (model, count, is_last) VALUES (?, 1, 1) "
            "ON CONFLICT(model) DO UPDATE SET count = count + 1, is_last = 1",
            (model,),
        )
        db.commit()


# ---------------------------------------------------------------------------
# File usage
# ---------------------------------------------------------------------------

def _load_file_usage() -> dict[str, int]:
    """Return file usage counts ordered oldest-first (by last_used).

    The returned dict preserves insertion order so that callers can
    derive recency from key position.
    """
    db = _get_db()
    rows = db.execute(
        "SELECT path, count FROM file_usage ORDER BY last_used ASC"
    ).fetchall()
    return {r["path"]: r["count"] for r in rows}


def _record_file_usage(path: str) -> None:
    """Increment the access count for a file path.

    Updates the ``last_used`` timestamp for recency ordering and
    evicts the least recently used entries when the table exceeds
    ``_MAX_FILE_USAGE_ENTRIES`` rows.

    Single atomic INSERT; eviction uses _db_lock for multi-statement safety.
    """
    db = _get_db()
    now = time.time()
    db.execute(
        "INSERT INTO file_usage (path, count, last_used) VALUES (?, 1, ?) "
        "ON CONFLICT(path) DO UPDATE SET count = count + 1, last_used = ?",
        (path, now, now),
    )
    row = db.execute("SELECT COUNT(*) FROM file_usage").fetchone()
    if row[0] > _MAX_FILE_USAGE_ENTRIES:
        with _db_lock:
            db.execute(
                "DELETE FROM file_usage WHERE path NOT IN "
                "(SELECT path FROM file_usage ORDER BY last_used DESC LIMIT ?)",
                (_MAX_FILE_USAGE_ENTRIES,),
            )
    db.commit()


# ---------------------------------------------------------------------------
# Stale directory cleanup (unchanged)
# ---------------------------------------------------------------------------

def _cleanup_stale_cs_dirs(max_age_hours: int = 24) -> int:
    """Remove the sorcar data directory if stale.

    Checks ``~/.kiss/sorcar-data`` and removes it if older than
    ``max_age_hours`` and no process is listening on its port.
    Also removes any legacy ``~/.kiss/cs-*`` per-workdir directories,
    ``cs-port-*`` files, and the old ``cs-data`` directory.

    Args:
        max_age_hours: Maximum age in hours before the directory is
            eligible for cleanup.

    Returns:
        Number of directories removed.
    """
    threshold = time.time() - max_age_hours * 3600
    removed = 0
    for p in sorted(_KISS_DIR.glob("cs-port-*")):
        if p.is_file():
            try:
                p.unlink()
            except OSError:  # pragma: no cover — OS-level unlink failure
                _log_exc()
    for d in sorted(_KISS_DIR.glob("cs-*")):
        if not d.is_dir() or d.name == "cs-extensions":
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    d = _KISS_DIR / "sorcar-data"
    if not d.is_dir():
        return removed
    try:
        if d.stat().st_mtime > threshold:
            return removed
        _pf = d / "cs-port"
        if _pf.exists():
            try:
                port = int(_pf.read_text().strip())
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    return removed
            except (ConnectionRefusedError, OSError, ValueError):
                pass
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    except OSError:  # pragma: no cover
        _log_exc()
    return removed
