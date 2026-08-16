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

import atexit
import json
import logging
import math
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl
    _fcntl = None  # type: ignore[assignment]


def _race_delay() -> None:
    """Sleep briefly when ``KISS_RACE_DELAY`` is set (no-op by default).

    Concurrency tests need to widen a read-modify-write window to make
    a cross-process race deterministic.  The delay is opt-in via an
    environment variable that production never sets, and is capped at
    100 ms so a stray value can never stall a real run.
    """
    raw = os.environ.get("KISS_RACE_DELAY")
    if not raw:
        return
    try:
        time.sleep(min(float(raw), 0.1))
    except ValueError:
        pass


@contextmanager
def _immediate_txn(db: sqlite3.Connection) -> Iterator[None]:
    """Run a read-modify-write sequence as ONE atomic transaction.

    Connections are opened with ``isolation_level=None`` (autocommit),
    so without an explicit transaction every statement of a
    read-modify-write sequence commits on its own and another PROCESS
    can interleave between them — ``_rw_lock`` is a ``threading``
    primitive and provides no cross-process exclusion.  ``BEGIN
    IMMEDIATE`` takes SQLite's write lock for the whole block, which
    every process on the database respects.

    Args:
        db: The connection to run the transaction on.

    Yields:
        ``None`` — run the statements inside the ``with`` block.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        try:
            db.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — rollback of a dead conn
            pass
        raise
    db.execute("COMMIT")


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
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
            except BaseException:
                self._pending_writers -= 1
                self._cond.notify_all()
                raise
            self._pending_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


_rw_lock = _RWLock()

_init_tables_lock = threading.Lock()


_chat_context_text_cache: dict[str, str] = {}
_chat_context_cache_lock = threading.Lock()
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


def _default_kiss_dir() -> Path:
    """Return the KISS data directory, respecting ``KISS_HOME`` env var."""
    return kiss_home()


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

_OWNER_DIR_NAME = "task-owners"

_owner_state: tuple[str, str, IO[Any]] | None = None
_owner_state_lock = threading.Lock()


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


def _owner_dir() -> Path:
    """Return the directory holding one liveness marker per process."""
    return _KISS_DIR / _OWNER_DIR_NAME


def _process_owner_token() -> str:
    """Return this process's owner token, publishing its liveness marker.

    The database is shared by every Sorcar process on the machine (the
    ``kiss-web`` daemon, a ``kiss`` CLI run, a VS Code reload), so
    "is the task that wrote this row still running?" cannot be answered
    from Python memory.  Each process therefore creates
    ``<KISS_HOME>/task-owners/<token>.lock`` and holds an exclusive
    ``flock`` on it for its whole lifetime: the kernel releases that
    lock when the process dies, however violently, so any other
    process can test liveness by trying to take the lock.

    The token is re-minted when ``_KISS_DIR`` is redirected (test
    fixtures, a daemon pointed at another home) so the marker always
    lives next to the database the rows are written to; the marker of
    the home being left is deleted, and so is this process's marker
    when the interpreter exits normally (:func:`_release_owner_marker`).
    Without that, every daemon and CLI lifecycle would leave one file
    behind forever, since a marker is otherwise only unlinked when
    some *other* process happens to test a sentinel row of the dead
    owner.

    Without ``fcntl`` there is no way to test whether the owning
    process is still alive — file existence would make every crashed
    process's rows look live forever — so no token is minted at all
    and liveness degrades to the timestamp-only heuristic.

    Returns:
        The token to store in ``task_history.owner``, or ``""`` when
        the marker could not be created (liveness then degrades to the
        previous timestamp-only heuristic).
    """
    global _owner_state
    if _fcntl is None:  # pragma: no cover — Windows has no flock
        return ""
    current_dir = str(_owner_dir())
    with _owner_state_lock:
        if _owner_state is not None and _owner_state[0] == current_dir:
            return _owner_state[1]
        _discard_owner_state()
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        try:
            _ensure_kiss_dir()
            Path(current_dir).mkdir(parents=True, exist_ok=True)
            handle = open(
                Path(current_dir) / f"{token}.lock", "w", encoding="utf-8",
            )
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except OSError:
            logger.warning(
                "could not publish task-owner liveness marker", exc_info=True,
            )
            return ""
        _owner_state = (current_dir, token, handle)
        return token


def _discard_owner_state() -> None:
    """Close and delete this process's current liveness marker.

    Caller holds :data:`_owner_state_lock` (or is the ``atexit`` hook,
    which takes it).  Deleting the file is what keeps
    ``task-owners/`` from growing by one entry per process lifetime;
    the kernel lock is released by the close either way.
    """
    global _owner_state
    if _owner_state is None:
        return
    directory, token, handle = _owner_state
    _owner_state = None
    try:
        handle.close()
    except OSError:  # pragma: no cover — close of a dead handle
        pass
    try:
        (Path(directory) / f"{token}.lock").unlink()
    except OSError:  # pragma: no cover — already swept by another process
        pass


def _release_owner_marker() -> None:
    """``atexit`` hook: drop this process's liveness marker on exit."""
    with _owner_state_lock:
        _discard_owner_state()


atexit.register(_release_owner_marker)


def _owner_is_alive(token: str) -> bool:
    """Return True when the process that recorded *token* is still running.

    A missing marker file, or one whose ``flock`` can be taken, means
    the owning process is gone.  Stale markers are unlinked as they are
    discovered so ``task-owners/`` cannot grow without bound.

    Args:
        token: The value of a row's ``owner`` column.  ``""`` (legacy
            rows written before owner tracking) reports not alive.

    Returns:
        True when the owning process still holds its marker.
    """
    if not token:
        return False
    if _fcntl is None:  # pragma: no cover — Windows has no flock
        # No token is stamped without a usable cross-process lock, so
        # any token seen here was written by a process on another
        # platform.  Its marker's mere existence proves nothing (a
        # crashed owner leaves it behind forever), so treat the owner
        # as gone and let the timestamp heuristic decide.
        return False
    if _owner_state is not None and _owner_state[1] == token:
        return True
    marker = _owner_dir() / f"{token}.lock"
    try:
        with open(marker, "r+", encoding="utf-8") as handle:
            try:
                _fcntl.flock(
                    handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB,
                )
            except OSError:
                return True
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
    except OSError:
        return False
    try:
        marker.unlink()
    except OSError:  # pragma: no cover — concurrent sweep won the unlink
        pass
    return False


_HistoryEntry = dict[str, object]


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce *value* to ``int``, returning *default* on failure.

    Non-finite floats (NaN/Inf) yield *default* rather than raising
    ``OverflowError``.  Any object whose ``__eq__`` raises an
    arbitrary exception is treated as the default rather than
    propagating — this keeps the task-completion finally robust
    against caller-supplied misbehaving objects.
    """
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)  # type: ignore[arg-type, call-overload, no-any-return]
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to ``float``, returning *default* on failure.

    Non-finite floats (NaN/Inf) yield *default* so the value never
    leaks into a JSON-serialised payload that would break SQLite's
    ``json_valid``.  Any object whose ``__eq__`` raises an arbitrary
    exception is treated as the default rather than propagating.
    """
    try:
        if value is None or value == "":
            return default
        result = float(value)  # type: ignore[arg-type]
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _safe_str(value: object, default: str = "") -> str:
    """Coerce *value* to ``str`` for JSON-bound payloads.

    bughunt8: SQLite's dynamic typing lets a hand-edited /
    3rd-party-source DB store a BLOB in a TEXT column, and
    ``json.dumps`` raises ``TypeError`` on ``bytes`` — uncaught by
    ``_dumps_extra``'s ``ValueError`` handler — which blanked the
    whole history sidebar over one corrupt row.  BLOBs are decoded as
    UTF-8 with replacement so the result is always JSON-serialisable;
    other non-string scalars go through ``str()``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _sanitize_non_finite(value: object) -> object:
    """Recursively replace non-finite floats (NaN/±Inf) with ``None``."""
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


_db_conn: sqlite3.Connection | None = None

_thread_local = threading.local()
_db_generation: int = 0


def _close_db() -> None:
    """Close the calling thread's connection and invalidate cached handles.

    Only the current thread's connection is closed eagerly.  Bumping
    the generation counter invalidates every other thread's cached
    connection *lazily*: each is detected as stale on that thread's
    next ``_get_db()`` call and replaced (and closed) there.
    """
    global _db_conn, _db_generation
    _stop_event_writer()
    _db_generation += 1
    tl_conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if tl_conn is not None:
        try:
            tl_conn.close()
        except Exception:
            pass
    _thread_local.conn = None
    _thread_local.gen = -1
    _thread_local.path = None
    _thread_local.file_id = None
    _db_conn = None


def _close_thread_db() -> None:
    """Close and forget the CALLING thread's cached connection only.

    Unlike :func:`_close_db`, the global generation counter is left
    untouched so every other thread's cached connection stays valid.
    Used by short-lived background threads (e.g. the startup
    orphan-task sweep in ``VSCodeServer.__init__``) to release their
    per-thread SQLite connection when they finish, instead of leaking
    an open connection for the life of the process.
    """
    global _db_conn
    tl_conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if tl_conn is not None:
        try:
            tl_conn.close()
        except Exception:
            pass
        if _db_conn is tl_conn:
            _db_conn = None
    _thread_local.conn = None
    _thread_local.gen = -1
    _thread_local.path = None
    _thread_local.file_id = None


_HISTORY_SELECT = (
    "SELECT id, timestamp, task, has_events, result, chat_id, "
    "model, work_dir, version, tokens, cost, steps, "
    "is_parallel, is_worktree, auto_commit_mode, "
    "start_ts, end_ts, is_favorite, parent_task_id "
    "FROM task_history "
)

_HISTORY_NOT_SUBAGENT = "(parent_task_id IS NULL OR parent_task_id = '')"

_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_task_history_id(value: object) -> bool:
    """Return True when *value* is shaped like a ``task_history.id``.

    The canonical id format is the un-hyphenated 32-character
    lowercase hex string produced by ``uuid.uuid4().hex``.  Callers
    use this guard at IPC / SQL boundaries to reject malformed or
    legacy-int payloads before they propagate.
    """
    return isinstance(value, str) and _TASK_ID_RE.fullmatch(value) is not None


def _coerce_parent_task_id(value: object) -> str:
    """Return a canonical ``parent_task_id`` column value.

    Accepts only a 32-char lowercase-hex UUID string.  Any other
    shape (None, empty, int, list, dict, non-UUID string) maps to the
    empty-string sentinel that ``_HISTORY_NOT_SUBAGENT`` treats as
    "not a sub-agent" — preventing garbage parent ids from being
    silently persisted as text that never matches any real UUID.
    """
    if isinstance(value, str) and _TASK_ID_RE.fullmatch(value):
        return value
    return ""


def _row_to_extra_json(row: sqlite3.Row) -> str:
    """Build the legacy-compat ``extra`` JSON string from typed columns.

    Many consumers (history sidebar, replay) read ``entry["extra"]`` as
    a JSON-encoded string.  This helper synthesizes the same shape from
    the new flat columns so those consumers continue to work unchanged.
    """
    payload: dict[str, object] = {}
    try:
        payload["model"] = _safe_str(row["model"])
        payload["work_dir"] = _safe_str(row["work_dir"])
        payload["version"] = _safe_str(row["version"])
        payload["auto_commit_mode"] = bool(row["auto_commit_mode"])
        payload["tokens"] = _safe_int(row["tokens"], 0)
        payload["cost"] = _safe_float(row["cost"], 0.0)
        payload["steps"] = _safe_int(row["steps"], 0)
        payload["is_parallel"] = bool(row["is_parallel"])
        payload["is_worktree"] = bool(row["is_worktree"])
        payload["startTs"] = _safe_int(row["start_ts"], 0)
        payload["endTs"] = _safe_int(row["end_ts"], 0)
        payload["is_favorite"] = bool(row["is_favorite"])
        if row["parent_task_id"]:
            payload["subagent"] = {
                "parent_task_id": _safe_str(row["parent_task_id"]),
            }
    except (KeyError, IndexError):
        return ""
    return _dumps_extra(payload) if payload else ""


def _history_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    """Convert a ``_HISTORY_SELECT`` row into a consumer-friendly dict.

    Exposes every selected typed column (``model``, ``cost``, etc.) so
    callers that switched to the new flat schema can read them
    directly, AND synthesises the legacy ``extra`` JSON string so
    callers that still parse ``entry["extra"]`` continue to work
    without any migration on their end.
    """
    out: dict[str, object] = {col: row[col] for col in row.keys()}
    out["extra"] = _row_to_extra_json(row)
    return out


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
    * ``Task stopped by user`` — an explicit user cancellation.  It is
      not a successful completion, so the history sidebar should mark
      it with the same red status dot used for failed runs.
    * ``Task interrupted*`` — an interrupted run.  Covers both the
      bare ``Task interrupted`` marker persisted by
      ``ChatSorcarAgent.run``'s ``except BaseException`` handler (a
      ``KeyboardInterrupt`` reaching a sub-agent / channel-agent
      run) and ``Task interrupted by server restart/shutdown`` (a
      graceful daemon/server shutdown cancellation).  Both are
      incomplete task outcomes rather than successes.
    """
    return (
        result.startswith("Task failed")
        or result == "Agent Failed Abruptly"
        or result == "Task terminated unexpectedly (process killed)"
        or result == "Task stopped by user"
        or result.startswith("Task interrupted")
    )


_INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_th_timestamp ON task_history(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_th_task ON task_history(task)",
    "CREATE INDEX IF NOT EXISTS idx_th_chat_id ON task_history(chat_id)",
    "CREATE INDEX IF NOT EXISTS idx_th_parent_task_id "
    "ON task_history(parent_task_id)",
    "CREATE INDEX IF NOT EXISTS idx_ev_task_id ON events(task_id)",
)


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_history (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id CHAR(32) DEFAULT '',
            model TEXT DEFAULT '',
            work_dir TEXT DEFAULT '',
            version TEXT DEFAULT '',
            tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            steps INTEGER DEFAULT 0,
            -- The framework (vscode_config.DEFAULTS, AgentState,
            -- sorcar.run()) defaults all three toggles to ON, so a row
            -- whose producer did not state a value must read as ON
            -- too; DEFAULT 0 would label it "no worktree / no
            -- parallel / no auto-commit" in the history sidebar.
            is_parallel INTEGER DEFAULT 1,
            is_worktree INTEGER DEFAULT 1,
            auto_commit_mode INTEGER DEFAULT 1,
            start_ts INTEGER DEFAULT 0,
            end_ts INTEGER DEFAULT 0,
            is_favorite INTEGER DEFAULT 0,
            parent_task_id TEXT DEFAULT '',
            -- Token of the process that created the row; see
            -- _process_owner_token / _recover_orphaned_tasks.
            owner TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES task_history(id),
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
    for ddl in _INDEX_DDL:
        conn.execute(ddl)
    _add_missing_columns(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    ``CREATE TABLE IF NOT EXISTS`` cannot extend an existing table, so
    every column added to :func:`_init_tables` after the first release
    needs a matching ``ALTER TABLE`` here.  A concurrent process may
    have added the same column between the ``PRAGMA`` and the
    ``ALTER``; the resulting "duplicate column name" error is benign.

    Args:
        conn: Connection whose ``task_history`` table to extend.
    """
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
    }
    if "owner" in cols:
        return
    try:
        conn.execute(
            "ALTER TABLE task_history ADD COLUMN owner TEXT DEFAULT ''"
        )
    except sqlite3.OperationalError:  # pragma: no cover — lost the race
        logger.debug("owner column already present", exc_info=True)


def _migrate_old_schema_if_needed(conn: sqlite3.Connection) -> bool:
    """Port a pre-UUID task_history DB to the new schema in-place.

    Detects the old schema (``task_history.id`` is ``INTEGER`` and the
    ``extra`` column exists), creates new-shaped tables under temporary
    names, assigns each row a fresh ``uuid.uuid4().hex``, copies row
    data into the typed columns, remaps every ``events.task_id`` to the
    new UUID, then atomically replaces the old tables.

    Returns ``True`` when migration was performed, ``False`` when the
    DB already has the new schema or no ``task_history`` table yet.
    """
    cols = {
        r[1]: (r[2] or "").upper()
        for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
    }
    if not cols:
        return False
    if cols.get("id") == "TEXT":
        return False
    if "extra" not in cols:
        return False

    def _sx(v: object) -> str:
        if v is None or v == "":
            return ""
        return v if isinstance(v, str) else str(v)

    def _bx(v: object, missing: int = 0) -> int:
        """Coerce a legacy ``extra`` value to a 0/1 column value.

        *missing* is returned for a key the old row never recorded —
        the framework toggles predate their own persistence, so
        mapping their absence to 0 would assert that a legacy run had
        worktree/parallel/auto-commit explicitly DISABLED, the exact
        opposite of the framework default at the time.
        """
        if v is None:
            return missing
        if isinstance(v, str):
            return 0 if v.strip().lower() in {"", "0", "false", "no"} else 1
        return 1 if bool(v) else 0

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        cols_locked = {
            r[1]: (r[2] or "").upper()
            for r in conn.execute(
                "PRAGMA table_info(task_history)"
            ).fetchall()
        }
        if (
            not cols_locked
            or cols_locked.get("id") == "TEXT"
            or "extra" not in cols_locked
        ):
            conn.execute("ROLLBACK")
            conn.execute("PRAGMA foreign_keys=ON")
            return False
        conn.execute("DROP TABLE IF EXISTS task_history__new")
        conn.execute("DROP TABLE IF EXISTS events__new")
        conn.execute(
            "CREATE TABLE task_history__new ("
            "id TEXT PRIMARY KEY, "
            "timestamp REAL NOT NULL, "
            "task TEXT NOT NULL, "
            "has_events INTEGER DEFAULT 0, "
            "result TEXT DEFAULT '', "
            "chat_id CHAR(32) DEFAULT '', "
            "model TEXT DEFAULT '', "
            "work_dir TEXT DEFAULT '', "
            "version TEXT DEFAULT '', "
            "tokens INTEGER DEFAULT 0, "
            "cost REAL DEFAULT 0.0, "
            "steps INTEGER DEFAULT 0, "
            "is_parallel INTEGER DEFAULT 1, "
            "is_worktree INTEGER DEFAULT 1, "
            "auto_commit_mode INTEGER DEFAULT 1, "
            "start_ts INTEGER DEFAULT 0, "
            "end_ts INTEGER DEFAULT 0, "
            "is_favorite INTEGER DEFAULT 0, "
            "parent_task_id TEXT DEFAULT '', "
            "owner TEXT DEFAULT ''"
            ")"
        )
        conn.execute(
            "CREATE TABLE events__new ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id TEXT NOT NULL REFERENCES task_history__new(id), "
            "seq INTEGER NOT NULL, "
            "event_json TEXT NOT NULL, "
            "timestamp REAL NOT NULL"
            ")"
        )
        rows = conn.execute(
            "SELECT id, timestamp, task, has_events, result, chat_id, "
            "extra FROM task_history ORDER BY id ASC"
        ).fetchall()
        id_map: dict[int, str] = {int(r[0]): uuid.uuid4().hex for r in rows}
        dropped_unknown_keys = 0
        known_extra_keys = {
            "model", "work_dir", "version", "tokens", "cost", "steps",
            "is_parallel", "is_worktree", "auto_commit_mode",
            "startTs", "endTs", "is_favorite", "subagent",
        }
        for r in rows:
            old_id = int(r[0])
            extra_raw = r[6] or ""
            try:
                extra = json.loads(extra_raw) if extra_raw else {}
            except (json.JSONDecodeError, TypeError):
                extra = {}
            if not isinstance(extra, dict):
                extra = {}
            parent_task_id = ""
            sub = extra.get("subagent")
            if isinstance(sub, dict):
                old_parent = sub.get("parent_task_id")
                if isinstance(old_parent, int):
                    parent_task_id = id_map.get(old_parent, "")
                elif (
                    isinstance(old_parent, str)
                    and _TASK_ID_RE.fullmatch(old_parent)
                ):
                    parent_task_id = old_parent
            for k in extra:
                if k not in known_extra_keys:
                    dropped_unknown_keys += 1
            conn.execute(
                "INSERT INTO task_history__new (id, timestamp, task, "
                "has_events, result, chat_id, model, work_dir, version, "
                "tokens, cost, steps, is_parallel, is_worktree, "
                "auto_commit_mode, start_ts, end_ts, is_favorite, "
                "parent_task_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    id_map[old_id], r[1], r[2],
                    r[3] or 0, r[4] or "", r[5] or "",
                    _sx(extra.get("model")), _sx(extra.get("work_dir")),
                    _sx(extra.get("version")), _safe_int(extra.get("tokens")),
                    _safe_float(extra.get("cost")),
                    _safe_int(extra.get("steps")),
                    _bx(extra.get("is_parallel"), missing=1),
                    _bx(extra.get("is_worktree"), missing=1),
                    _bx(extra.get("auto_commit_mode"), missing=1),
                    _safe_int(extra.get("startTs")),
                    _safe_int(extra.get("endTs")),
                    _bx(extra.get("is_favorite")), parent_task_id,
                ),
            )
        has_events_table = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='events'"
        ).fetchone() is not None
        dropped_events = 0
        if has_events_table:
            ev_rows = conn.execute(
                "SELECT task_id, seq, event_json, timestamp FROM events"
            ).fetchall()
            for er in ev_rows:
                try:
                    new_tid = id_map.get(int(er[0]))
                except (TypeError, ValueError):
                    new_tid = None
                if new_tid is None:
                    dropped_events += 1
                    continue
                conn.execute(
                    "INSERT INTO events__new "
                    "(task_id, seq, event_json, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (new_tid, er[1], er[2], er[3]),
                )
            conn.execute("DROP TABLE events")
        conn.execute("DROP TABLE task_history")
        conn.execute(
            "ALTER TABLE task_history__new RENAME TO task_history"
        )
        conn.execute(
            "ALTER TABLE events__new RENAME TO events"
        )
        for ddl in _INDEX_DDL:
            conn.execute(ddl)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error:
            pass
        raise
    conn.execute("PRAGMA foreign_keys=ON")
    if dropped_unknown_keys:
        logger.warning(
            "task_history migration dropped %d unknown extra key(s)",
            dropped_unknown_keys,
        )
    if dropped_events:
        logger.warning(
            "task_history migration dropped %d orphan event row(s) "
            "whose task_id had no surviving parent",
            dropped_events,
        )
    return True


#: Sentinel returned by :func:`_db_file_identity` when ``os.stat``
#: failed for a reason OTHER than the file being absent (EACCES, EIO,
#: EMFILE, ...).  Impossible as a real ``(st_dev, st_ino)`` pair.
_FILE_ID_UNKNOWN: tuple[int, int] = (-1, -1)


def _db_file_identity(path: str) -> tuple[int, int] | None:
    """Return the ``(st_dev, st_ino)`` identity of *path*.

    ``None`` means the file is CONFIRMED absent — ``os.stat`` raised
    ``FileNotFoundError`` (ENOENT/ENOTDIR).  Any other ``OSError`` is a
    transient or environmental failure that says nothing about whether
    the file exists, so the distinct sentinel :data:`_FILE_ID_UNKNOWN`
    is returned instead; callers must treat it as "identity unknown,
    assume unchanged" — never as "file deleted or replaced".

    Conflating the two was catastrophic: one transient stat failure
    under heavy load made every thread treat its healthy connection as
    stale, and the reconnect path then deleted the LIVE ``-wal``/
    ``-shm`` sidecars out from under the remaining connections,
    corrupting the database and losing every commit still in the WAL.

    The identity distinguishes a file that was deleted and recreated
    at the same pathname from the original file — a plain existence
    check cannot.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "transient stat failure on %s; assuming file unchanged",
            path,
            exc_info=True,
        )
        return _FILE_ID_UNKNOWN
    return (st.st_dev, st.st_ino)


def _get_db() -> sqlite3.Connection:
    """Return a per-thread database connection, creating one if needed.

    Each calling thread gets its own ``sqlite3.Connection`` so that
    concurrent threads never share cursor state.  Connections are
    cached in ``threading.local()`` and invalidated when:

    * ``_db_generation`` is bumped (via ``_close_db()``),
    * ``_DB_PATH`` changes (test redirects), or
    * the file at ``_DB_PATH`` is deleted or replaced on disk (its
      ``(st_dev, st_ino)`` identity no longer matches the one the
      cached connection was opened against).

    The process-global ``_db_conn`` is deliberately NOT part of that
    validity test: it names whichever connection was created last by
    ANY thread, so a short-lived thread calling ``_close_thread_db()``
    used to force every other thread to close a healthy connection and
    re-run the WAL pragma and the migration check.
    """
    global _db_conn
    tl = _thread_local
    tl_conn: sqlite3.Connection | None = getattr(tl, "conn", None)
    tl_gen: int = getattr(tl, "gen", -1)
    tl_path: str | None = getattr(tl, "path", None)
    gen_snapshot = _db_generation
    current_path = str(_DB_PATH)
    current_id = _db_file_identity(current_path)
    if current_id is not _FILE_ID_UNKNOWN:
        _maybe_reset_caches(current_path, current_id)

    if (
        tl_conn is not None
        and tl_gen == gen_snapshot
        and tl_path == current_path
        and (
            # A transient stat failure proves nothing about the file;
            # keep using the healthy cached connection rather than
            # tearing it down and reconnecting to a database that is
            # momentarily unreadable.
            current_id is _FILE_ID_UNKNOWN
            or (
                current_id is not None
                and getattr(tl, "file_id", None) == current_id
            )
        )
    ):
        return tl_conn

    if tl_conn is not None:
        # Forget the handle BEFORE closing it: when the reconnect
        # below raises (the database file is unreadable right now) the
        # cache must not keep pointing at a closed connection that a
        # later call would hand out as valid.
        tl.conn = None
        tl.file_id = None
        try:
            tl_conn.close()
        except Exception:
            pass

    _ensure_kiss_dir()
    # Deliberately NO manual cleanup of stale ``-wal``/``-shm``
    # sidecars here.  Application-level unlink of SQLite's sidecar
    # files is impossible to make safe: any check-then-unlink is racy
    # against other threads and OTHER PROCESSES concurrently opening
    # or holding the same database, and unlinking a live WAL destroys
    # committed-but-uncheckpointed pages (the root cause of the
    # 2026-08-15 sorcar.db corruption).  SQLite itself handles a
    # leftover sidecar of a deleted-and-recreated database safely: a
    # WAL whose salt/checksums do not match is ignored and reset on
    # the first write, so no cleanup is needed for correctness.
    conn = sqlite3.connect(
        current_path,
        check_same_thread=False,
        timeout=10,
        isolation_level=None,
    )
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    with _init_tables_lock:
        wal_deadline = time.monotonic() + 30.0
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                code = getattr(exc, "sqlite_errorcode", None)
                busy = (
                    code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                    if code is not None
                    else "locked" in str(exc).lower()
                    or "busy" in str(exc).lower()
                )
                if not busy or time.monotonic() >= wal_deadline:
                    raise
                time.sleep(0.05)
        _migrate_old_schema_if_needed(conn)
        _init_tables(conn)

    tl.conn = conn
    tl.gen = gen_snapshot
    tl.path = current_path
    tl.file_id = _db_file_identity(current_path)
    _db_conn = conn
    return conn


def _most_recent_task_id(db: sqlite3.Connection, task: str | None) -> str | None:
    """Return the row id of the most recent run of *task*, or the latest row.

    Uses the total order ``(timestamp, rowid)`` — the ``rowid`` tiebreak keeps
    rows with equal timestamps (coarse clock ticks, imported databases)
    resolving to the genuinely latest insert, consistent with
    :func:`_load_latest_chat_events_by_chat_id`.
    """
    if task is not None:
        row = db.execute(
            "SELECT id FROM task_history WHERE task = ? "
            "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
            (task,),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT id FROM task_history "
            "ORDER BY timestamp DESC, rowid DESC LIMIT 1"
        ).fetchone()
    return str(row["id"]) if row else None


def _add_task(
    task: str,
    chat_id: str = "",
    extra: dict[str, object] | None = None,
) -> tuple[str, str]:
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
    db = _get_db()
    payload = dict(extra) if extra else {}
    parent_task_id = ""
    sub = payload.get("subagent")
    flat_parent = payload.get("parent_task_id")
    if sub is not None and flat_parent is not None:
        raise ValueError(
            "Cannot pass both 'parent_task_id' and 'subagent' to _add_task",
        )
    if isinstance(sub, dict):
        parent_task_id = _coerce_parent_task_id(sub.get("parent_task_id"))
    elif isinstance(sub, str):
        parent_task_id = _coerce_parent_task_id(sub)
    elif flat_parent is not None:
        parent_task_id = _coerce_parent_task_id(flat_parent)
    with _rw_lock.write_lock():
        if chat_id == "":
            chat_id = _allocate_chat_id()
        task_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, chat_id, result, "
            "model, work_dir, version, tokens, cost, steps, is_parallel, "
            "is_worktree, auto_commit_mode, start_ts, end_ts, is_favorite, "
            "parent_task_id, owner) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, time.time(), task, chat_id,
                "Agent Failed Abruptly",
                _safe_str(payload.get("model", "") or ""),
                _safe_str(payload.get("work_dir", "") or ""),
                _safe_str(payload.get("version", "") or ""),
                _safe_int(payload.get("tokens"), 0),
                _safe_float(payload.get("cost"), 0.0),
                _safe_int(payload.get("steps"), 0),
                1 if payload.get("is_parallel") else 0,
                1 if payload.get("is_worktree") else 0,
                1 if payload.get("auto_commit_mode") else 0,
                _safe_int(payload.get("startTs"), 0),
                _safe_int(payload.get("endTs"), 0),
                1 if payload.get("is_favorite") else 0,
                parent_task_id,
                _process_owner_token(),
            ),
        )
    _invalidate_chat_context_cache(chat_id)
    return task_id, chat_id


def _allocate_chat_id() -> str:
    """Pre-allocate a chat session id without keeping a task row.

    Generates a new UUID-style string that can be used as a unique
    chat session identifier.

    This is used by ``WorktreeSorcarAgent`` to name worktree branches
    *before* the first task in a session is persisted.

    Returns:
        A unique 32-character string suitable for use as a ``chat_id``.
    """
    return uuid.uuid4().hex


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
            + "ORDER BY timestamp DESC, rowid DESC LIMIT ? OFFSET ?"
        )
        rows = db.execute(sql, (effective_limit, offset)).fetchall()
        return [_history_row_to_dict(r) for r in rows]


def _history_date_range() -> tuple[float | None, float | None]:
    """Return the first and last task timestamps in the history.

    Computes ``(MIN(timestamp), MAX(timestamp))`` over the same row
    set the History sidebar lists (i.e. excluding sub-agent rows) so
    the sidebar's From/To date inputs can be pre-filled with the
    first and last task dates.  Thread-safe.

    Returns:
        ``(min_ts, max_ts)`` in epoch seconds, or ``(None, None)``
        when no listable rows exist.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT MIN(timestamp) AS mn, MAX(timestamp) AS mx "
            f"FROM task_history WHERE {_HISTORY_NOT_SUBAGENT}"
        ).fetchone()
    if row is None or row["mn"] is None or row["mx"] is None:
        return (None, None)
    return (float(row["mn"]), float(row["mx"]))


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
        rows = db.execute(
            "SELECT task FROM task_history "
            "WHERE task GLOB ? AND LENGTH(task) > ? "
            f"AND {_HISTORY_NOT_SUBAGENT} "
            "GROUP BY task "
            "ORDER BY MAX(timestamp) DESC, MAX(rowid) DESC LIMIT ?",
            (escaped + "*", len(query), limit),
        ).fetchall()
    return [row["task"] for row in rows]


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
            + "ORDER BY timestamp DESC, rowid DESC LIMIT ? OFFSET ?",
            (f"%{escaped}%", limit, offset),
        ).fetchall()
        return [_history_row_to_dict(r) for r in rows]


def _resolve_task_id(
    db: sqlite3.Connection,
    task_id: str | None,
    task: str | None,
) -> str | None:
    """Resolve a stable row id, falling back to the most recent task.

    Args:
        db: Active database connection.
        task_id: Explicit row id when available.
        task: Fallback task description for legacy callers.

    Returns:
        The resolved row id, or ``None`` if not found.
    """
    if isinstance(task_id, str) and task_id != "":
        if not is_task_history_id(task_id):
            return _most_recent_task_id(db, task)
        row = db.execute(
            "SELECT id FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        if row is not None:
            return str(row["id"])
        return None
    return _most_recent_task_id(db, task)


def _log_orphaned_task_forensics(
    db: sqlite3.Connection,
    rowids: list[int],
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
        rowids: The ``rowid``s about to be rewritten.  Empty is a
            no-op.
    """
    if not rowids:
        return
    placeholders = ",".join(["?"] * len(rowids))
    diag_rows = db.execute(
        "SELECT id, task, chat_id, model, start_ts, steps, cost "
        f"FROM task_history WHERE rowid IN ({placeholders})",
        list(rowids),
    ).fetchall()
    for row in diag_rows:
        task_id_val = row["id"]
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
                ev_ts = _safe_float(ev["timestamp"], 0.0)
            except Exception:
                ev_type = "parse_error"
                ev_ts = 0
            last_event_summaries.append(
                f"seq={ev['seq']} type={ev_type} ts={ev_ts:.1f}"
            )
        model_name = row["model"] or "unknown"
        start_ts = row["start_ts"] or 0
        steps = row["steps"] if row["steps"] is not None else "?"
        cost = row["cost"] if row["cost"] is not None else "?"
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


def _recover_orphaned_tasks(
    active_task_ids: set[str],
    created_before: float | None = None,
) -> int:
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

    Liveness is decided from the DATABASE, not from process memory:
    every row records the ``owner`` token of the process that created
    it (see :func:`_process_owner_token`), and a row whose owner still
    holds its liveness marker is skipped.  Without that check a second
    Sorcar process — a ``kiss`` CLI run, a VS Code extension reload, a
    restarted daemon — rewrote the rows of tasks that were still
    RUNNING in the first process, painting a red failure dot on a live
    task; and because the sentinel was gone, neither the shutdown
    safety net nor a later sweep could ever record the real outcome.

    Args:
        active_task_ids: Row ids that are still being processed in
            the current process and must therefore NOT be rewritten.
            Pass an empty set at fresh-server startup — by then any
            row carrying the sentinel must belong to a prior process,
            live or dead.
        created_before: Optional epoch-seconds cut-off.  When given,
            only sentinel rows whose ``timestamp`` column is strictly
            older are rewritten.  ``VSCodeServer`` passes its boot
            timestamp here because its sweep runs on a BACKGROUND
            thread: a task legitimately started after boot (inserting
            a fresh sentinel row) must never be mistaken for an
            orphan of a prior process by a sweep whose UPDATE races
            past the insertion — that mislabels a live task as
            "process killed" and defeats the pre-emptive shutdown
            persistence in ``_stop_active_agent_tasks`` (which only
            rewrites rows still carrying the sentinel).  ``None``
            applies no time filter.

    Returns:
        The number of rows whose ``result`` column was rewritten.
    """
    db = _get_db()
    active_ids = {str(t) for t in active_task_ids}
    # ``rowid`` rather than ``id``: ``id`` is a TEXT primary key, so
    # SQLite accepts NULL there (a partial INSERT from an older
    # release or an external tool), and such a row could then never be
    # targeted by an id-based UPDATE.  Every non-WITHOUT-ROWID table
    # has a unique, non-NULL rowid.
    select_sql = (
        "SELECT rowid AS rid, id, owner FROM task_history WHERE result = ? "
    )
    params: list[object] = ["Agent Failed Abruptly"]
    if created_before is not None:
        select_sql += "AND timestamp < ? "
        params.append(float(created_before))
    with _rw_lock.write_lock(), _immediate_txn(db):
        candidates = db.execute(select_sql, params).fetchall()
        dead_rowids = [
            int(row["rid"])
            for row in candidates
            if str(row["id"] or "") not in active_ids
            and not _owner_is_alive(row["owner"] or "")
        ]
        _log_orphaned_task_forensics(db, dead_rowids)
        rowcount = 0
        if dead_rowids:
            placeholders = ",".join(["?"] * len(dead_rowids))
            cursor = db.execute(
                "UPDATE task_history SET result = ? "
                f"WHERE rowid IN ({placeholders}) AND result = ?",
                [
                    "Task terminated unexpectedly (process killed)",
                    *dead_rowids,
                    "Agent Failed Abruptly",
                ],
            )
            rowcount = cursor.rowcount or 0
    if rowcount:
        logger.warning(
            "Recovered %d orphaned task(s) from prior process kill",
            rowcount,
        )
        _invalidate_chat_context_cache("")
    return rowcount


def _shutdown_persist_in_flight_results(task_ids: set[str]) -> int:
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
    id_list = [str(t) for t in task_ids]
    placeholders = ",".join(["?"] * len(id_list))
    sql = (
        f"UPDATE task_history SET result = ? "
        f"WHERE id IN ({placeholders}) AND result = ?"
    )
    affected_chat_ids: list[str] = []
    with _rw_lock.write_lock(), _immediate_txn(db):
        rows = db.execute(
            f"SELECT chat_id FROM task_history "
            f"WHERE id IN ({placeholders}) AND result = ?",
            [*id_list, "Agent Failed Abruptly"],
        ).fetchall()
        affected_chat_ids = [r["chat_id"] or "" for r in rows]
        cursor = db.execute(
            sql,
            ["Task interrupted by server restart/shutdown",
             *id_list, "Agent Failed Abruptly"],
        )
        rowcount = cursor.rowcount or 0
    if rowcount:
        logger.warning(
            "Pre-emptively persisted shutdown result for %d in-flight task(s)",
            rowcount,
        )
        for chat_id in set(affected_chat_ids):
            _invalidate_chat_context_cache(chat_id)
    return rowcount


_UPDATABLE_COLUMNS = frozenset({"result"})


def _update_task_column(
    column: str,
    value: str,
    task_id: str | None,
    task: str | None,
) -> str | None:
    """Write *value* into *column* of the resolved ``task_history`` row.

    Drains pending queued events first so the column update is ordered
    after every event the task has emitted so far, then performs the
    UPDATE under the process-wide write lock.

    Args:
        column: Column name to update.  The name is interpolated into
            the SQL, so it must be one of :data:`_UPDATABLE_COLUMNS`;
            anything else raises ``ValueError``.
        value: The new column value.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.

    Returns:
        The updated row's ``chat_id`` (possibly ``""``), or ``None``
        when no row could be resolved.

    Raises:
        ValueError: When *column* is not an allowed column name.
    """
    if column not in _UPDATABLE_COLUMNS:
        raise ValueError(
            f"_update_task_column refuses column {column!r}; "
            f"allowed: {sorted(_UPDATABLE_COLUMNS)}"
        )
    _flush_chat_events(task_id if is_task_history_id(task_id) else None)
    db = _get_db()
    with _rw_lock.write_lock(), _immediate_txn(db):
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
        return (row["chat_id"] or "") if row is not None else ""


def _save_task_result(
    result: str,
    task_id: str | None = None,
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
    _invalidate_chat_context_cache(affected_chat_id)


def _set_task_favorite(task_id: str, is_favorite: bool) -> bool:
    """Toggle the ``is_favorite`` column for a task row.

    Thread-safe: drains the background event queue first so the
    favourite flag write is ordered after any in-flight event inserts
    for the same task, then acquires the process-wide write lock.

    Args:
        task_id: Primary key of the ``task_history`` row to update.
        is_favorite: New value for the ``is_favorite`` flag.

    Returns:
        True when the row existed and was updated, False otherwise.
    """
    _flush_chat_events(task_id)
    db = _get_db()
    with _rw_lock.write_lock():
        cursor = db.execute(
            "UPDATE task_history SET is_favorite = ? WHERE id = ?",
            (1 if is_favorite else 0, task_id),
        )
        return (cursor.rowcount or 0) > 0


_EXTRA_COL_MAP: dict[str, tuple[str, object, object]] = {
    "model": ("model", str, ""),
    "work_dir": ("work_dir", str, ""),
    "version": ("version", str, ""),
    "auto_commit_mode": ("auto_commit_mode", lambda v: 1 if v else 0, 0),
    "tokens": ("tokens", int, 0),
    "cost": ("cost", float, 0.0),
    "steps": ("steps", int, 0),
    "is_parallel": ("is_parallel", lambda v: 1 if v else 0, 0),
    "is_worktree": ("is_worktree", lambda v: 1 if v else 0, 0),
    "startTs": ("start_ts", int, 0),
    "endTs": ("end_ts", int, 0),
}


def _save_task_extra(
    extra: dict[str, object],
    task_id: str | None = None,
    task: str | None = None,
) -> None:
    """Save extra metadata for a task into typed columns.

    Writes each known key from *extra* to its column in
    ``task_history``.  Unknown keys are silently ignored.  Keys absent
    from *extra* are NOT included in the UPDATE — so the
    ``is_favorite`` flag (set independently by
    :func:`_set_task_favorite`) is automatically preserved.

    The legacy nested ``{"subagent": {"parent_task_id": <uuid>}}``
    payload is translated to a write of the ``parent_task_id`` column
    (only when the payload contains the dotted shape).

    Args:
        extra: Dictionary of metadata to persist.
        task_id: Stable row id to update when available.
        task: Fallback task description string for legacy callers.
    """
    _flush_chat_events(task_id if is_task_history_id(task_id) else None)
    db = _get_db()
    with _rw_lock.write_lock(), _immediate_txn(db):
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        pairs: list[tuple[str, object]] = []
        for k, v in extra.items():
            if k == "is_favorite":
                raise ValueError(
                    "_save_task_extra does not write 'is_favorite'; "
                    "use _set_task_favorite() instead"
                )
            mapping = _EXTRA_COL_MAP.get(k)
            if mapping is None:
                if k == "parent_task_id":
                    if "subagent" in extra:
                        raise ValueError(
                            "Cannot pass both 'parent_task_id' and "
                            "'subagent' to _save_task_extra"
                        )
                    coerced = _coerce_parent_task_id(v)
                    if coerced:
                        pairs.append(("parent_task_id = ?", coerced))
                    continue
                if k == "subagent":
                    if isinstance(v, dict):
                        raw_parent = v.get("parent_task_id")
                    else:
                        raw_parent = v
                    coerced = _coerce_parent_task_id(raw_parent)
                    if coerced:
                        pairs.append(("parent_task_id = ?", coerced))
                continue
            col, cast, default = mapping
            try:
                if v is None or v == "":
                    val: object = default
                elif isinstance(v, float) and not math.isfinite(v):
                    val = default
                else:
                    result = cast(v)  # type: ignore[operator]
                    if (
                        isinstance(result, float)
                        and not math.isfinite(result)
                    ):
                        val = default
                    else:
                        val = result
            except Exception:
                val = default
            pairs.append((f"{col} = ?", val))
        if not pairs:
            return
        sets = [s for s, _ in pairs]
        vals = [v for _, v in pairs]
        vals.append(resolved)
        db.execute(
            f"UPDATE task_history SET {', '.join(sets)} WHERE id = ?", vals
        )


_event_queue: queue.Queue = queue.Queue()
_event_writer_thread: threading.Thread | None = None
_event_writer_lock = threading.Lock()
_event_writer_stop = threading.Event()
_journal_lock = threading.Lock()
_pending_cond = threading.Condition()
_pending_by_task: dict[str, int] = {}
_next_seq_cache: dict[str, int] = {}
_marked_has_events: set[str] = set()
_caches_db_key: tuple[str, tuple[int, int] | None] | None = None
_caches_lock = threading.Lock()


def _maybe_reset_caches(
    current_path: str, file_id: tuple[int, int] | None = None,
) -> None:
    """Clear seq/has_events caches when the database is swapped out.

    Triggered both by a ``_DB_PATH`` reassignment (test fixtures) and
    by an on-disk replacement of the SAME pathname (*file_id* — the
    file's ``(st_dev, st_ino)`` — changes): the cached sequence
    counters and has-events marks were seeded from the previous
    database and are meaningless for the new one.
    """
    global _caches_db_key
    key = (current_path, file_id)
    if _caches_db_key == key:
        return
    with _caches_lock:
        if _caches_db_key == key:
            return
        _next_seq_cache.clear()
        _marked_has_events.clear()
        _caches_db_key = key
    _invalidate_chat_context_cache("")

_BATCH_MAX = 256
_BATCH_WINDOW_S = 0.020


def _start_event_writer() -> None:
    """Lazily spawn the background event writer thread (idempotent)."""
    global _event_writer_thread, _event_writer_stop
    if _event_writer_thread is not None and _event_writer_thread.is_alive():
        return
    with _event_writer_lock:
        if _event_writer_thread is not None and _event_writer_thread.is_alive():
            return
        stop = threading.Event()
        t = threading.Thread(
            target=_event_writer_loop,
            args=(stop,),
            name="kiss-event-writer",
            daemon=True,
        )
        _event_writer_stop = stop
        _event_writer_thread = t
        t.start()


def _event_writer_loop(stop: threading.Event) -> None:
    """Drain the event queue in batches and persist them.

    Args:
        stop: This writer's private stop event (set by
            ``_stop_event_writer``); private so a concurrent
            ``_start_event_writer`` can never un-stop it.
    """
    while not stop.is_set():
        try:
            first = _event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if first is None:
            _event_queue.task_done()
            if stop.is_set():
                return
            continue
        batch: list[tuple[str, str, float, str]] = [first]
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
                shutdown_pending = stop.is_set()
                break
            batch.append(item)
        try:
            _persist_batch_with_retry(batch)
        finally:
            _release_pending(batch)
        if shutdown_pending:
            return


def _persist_batch_with_retry(batch: list[tuple[str, str, float, str]]) -> None:
    """Persist *batch*, retrying a few times before giving up.

    Transient failures (e.g. external lock contention beyond the busy
    timeout) must not silently drop events; a bounded retry keeps the
    writer from stalling forever while still recovering the common
    case.  Events are only abandoned — with an ``error`` log — after
    every attempt fails.
    """
    attempts = 4
    for attempt in range(attempts):
        try:
            _write_event_batch(batch)
            return
        except Exception:
            if attempt < attempts - 1:
                logger.warning("event writer batch failed; retrying", exc_info=True)
                time.sleep(0.05 * (attempt + 1))
            else:
                _journal_failed_events(batch, attempts)


def _failed_events_path(db_path: str) -> str:
    """Return the journal path holding unwritable events for *db_path*."""
    return db_path + ".failed_events.jsonl"


#: Suffix of a journal file a replayer has taken ownership of.  The
#: live sidecar is renamed to ``<sidecar>.consumed-<pid>-<uuid>``
#: before it is replayed, so the file that is finally deleted is
#: exactly the one that was written to the database — never a file
#: another process has appended to since.
_JOURNAL_CONSUMED_SUFFIX = ".consumed-"


@contextmanager
def _journal_file_lock(sidecar: str) -> Iterator[None]:
    """Hold an inter-process lock on *sidecar* for the whole block.

    The database and its journal are shared by every Sorcar process on
    the machine (the ``kiss-web`` daemon, a ``kiss`` CLI run, a VS Code
    reload), so the module-level :data:`_journal_lock` — a plain
    thread lock — cannot order an append in one process against a
    replay in another.  Without that ordering a replayer deletes
    batches a peer appended after the replayer read the file, and two
    replayers write the same batch twice.

    The lock lives in a sibling ``<sidecar>.lock`` file rather than in
    the journal itself, because the journal is renamed and deleted
    while the lock is held.  Closing the handle releases the kernel
    lock, and the kernel releases it anyway if the process dies.

    Args:
        sidecar: Path of the journal file being appended or replayed.

    Yields:
        Nothing; the lock is held for the duration of the block.  On
        platforms without ``fcntl`` the block runs unserialised (the
        in-process lock still applies), which is why replay also
        consumes by rename.
    """
    if _fcntl is None:  # pragma: no cover — Windows has no flock
        yield
        return
    try:
        handle = open(sidecar + ".lock", "a+", encoding="utf-8")
    except OSError:  # pragma: no cover — unwritable journal directory
        yield
        return
    try:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


def _journal_failed_events(
    batch: list[tuple[str, str, float, str]], attempts: int,
) -> None:
    """Preserve a permanently unwritable batch in a durable sidecar file.

    A batch that failed every write attempt must not be silently
    acknowledged and lost — ``_flush_chat_events`` would then report
    completion for events that were never persisted.  The rows are
    appended as JSON lines to ``<db>.failed_events.jsonl`` and replayed
    by :func:`_replay_failed_events` as soon as the database accepts
    writes again, so the transcript is recoverable instead of merely
    inspectable.

    Each row is journalled next to the database it was PRODUCED
    against (its own ``origin_db_path``), not next to whichever
    database happens to be active now: after a database swap the two
    differ, and a journal written next to the new database would be
    replayed into it and dropped.

    The append is serialised against every other process's append and
    replay by :func:`_journal_file_lock`, so a batch can never be
    written into a file another process is in the middle of consuming.
    """
    by_origin: dict[str, list[tuple[str, str, float, str]]] = {}
    for row in batch:
        by_origin.setdefault(row[3], []).append(row)
    with _journal_lock:
        for origin, rows in by_origin.items():
            sidecar = _failed_events_path(origin)
            try:
                with _journal_file_lock(sidecar), open(
                    sidecar, "a", encoding="utf-8",
                ) as stream:
                    for task_id, event_json, timestamp, origin_path in rows:
                        stream.write(json.dumps({
                            "task_id": task_id,
                            "event_json": event_json,
                            "timestamp": timestamp,
                            "origin_db_path": origin_path,
                        }) + "\n")
                logger.error(
                    "%d chat events could not be written after %d attempts; "
                    "journalled in %s for replay",
                    len(rows), attempts, sidecar, exc_info=True,
                )
            except OSError:
                logger.error(
                    "dropping %d chat events after %d failed write attempts "
                    "(sidecar %s also unwritable)",
                    len(rows), attempts, sidecar, exc_info=True,
                )


def _replay_failed_events() -> None:
    """Re-insert events journalled while the database was unwritable.

    Called from :func:`_flush_chat_events`, so the recovery happens on
    the very next write-ordering barrier after the database becomes
    writable again — before ``_task_has_events`` decides that a task
    produced no transcript and a stub stream must be synthesized in
    its place.

    A snapshot is only removed once every row it holds has landed in
    the database; a still-failing replay leaves it on disk for the next
    attempt.

    Concurrency: the journal is shared by every Sorcar process, so the
    whole claim → write → delete sequence runs under
    :func:`_journal_file_lock`, and the rows are claimed by *renaming*
    the sidecar aside first.  Deleting the renamed snapshot can
    therefore never destroy a batch a peer appended in the meantime —
    that batch goes to a fresh sidecar — and no batch is ever replayed
    by two processes at once.
    """
    path = _failed_events_path(_current_db_path())
    if not _journal_has_pending_rows(path):
        # The overwhelmingly common case: nothing ever failed to write.
        # Checked before locking so a healthy database never pays for
        # the lock file or the lock itself.
        return
    with _journal_lock, _journal_file_lock(path):
        for snapshot in _claim_journal_snapshots(path):
            if not _replay_journal_snapshot(snapshot):
                _restore_journal_snapshot(snapshot, path)


def _journal_has_pending_rows(path: str) -> bool:
    """Return True when a journal or an unfinished snapshot exists.

    Args:
        path: The live sidecar path for the active database.
    """
    if os.path.exists(path):
        return True
    directory = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + _JOURNAL_CONSUMED_SUFFIX
    try:
        return any(name.startswith(prefix) for name in os.listdir(directory))
    except OSError:  # pragma: no cover — unreadable journal directory
        return False


def _claim_journal_snapshots(path: str) -> list[str]:
    """Take ownership of *path* and return every snapshot to replay.

    The live sidecar is renamed to a unique
    ``.consumed-<pid>-<uuid>`` sibling, which is what makes the later
    delete safe.  Snapshots a previous replayer left behind — it
    crashed, or the database was still refusing writes — are picked up
    too, so a rename is never a way to lose events.

    Args:
        path: The live sidecar path for the active database.

    Returns:
        Snapshot paths to replay, oldest name first.  Caller holds
        :func:`_journal_file_lock`.
    """
    if os.path.exists(path):
        claimed = (
            f"{path}{_JOURNAL_CONSUMED_SUFFIX}{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            os.replace(path, claimed)
        except OSError:  # pragma: no cover — unrenamable journal
            logger.warning("could not claim journal %s", path, exc_info=True)
    directory = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + _JOURNAL_CONSUMED_SUFFIX
    try:
        names = os.listdir(directory)
    except OSError:  # pragma: no cover — unreadable journal directory
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(names)
        if name.startswith(prefix)
    ]


def _restore_journal_snapshot(snapshot: str, path: str) -> None:
    """Put a snapshot the database still refuses back under the live name.

    ``<db>.failed_events.jsonl`` stays the single place an operator —
    and the next replay — looks for pending rows, instead of the
    pending transcript hiding under a ``.consumed-*`` name after every
    failed attempt.

    Args:
        snapshot: The claimed file whose replay failed.
        path: The live sidecar path for the active database.
    """
    if os.path.exists(path):
        # Another snapshot was restored first (a previous replayer
        # died before it could restore its own).  Leaving this one as
        # a snapshot loses nothing: the next replay claims it too.
        return
    try:
        os.replace(snapshot, path)
    except OSError:  # pragma: no cover — unrenamable snapshot
        logger.warning(
            "could not restore journal snapshot %s", snapshot, exc_info=True,
        )


def _replay_journal_snapshot(snapshot: str) -> bool:
    """Write one claimed journal *snapshot* to the database and delete it.

    Args:
        snapshot: Path of a ``.consumed-*`` file this process owns.

    Returns:
        True when the snapshot was replayed (or held nothing usable)
        and removed; False when the database still refuses the write,
        in which case the snapshot is kept for the next attempt.
    """
    try:
        with open(snapshot, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError:  # pragma: no cover — unreadable snapshot
        return True
    batch: list[tuple[str, str, float, str]] = []
    for line in lines:
        try:
            record = json.loads(line)
            batch.append((
                str(record["task_id"]),
                str(record["event_json"]),
                _safe_float(record["timestamp"], 0.0),
                str(record["origin_db_path"]),
            ))
        except (ValueError, TypeError, KeyError):
            logger.warning("skipping malformed journal line", exc_info=True)
    if batch:
        try:
            _write_event_batch(batch)
        except Exception:
            logger.warning(
                "replay of %d journalled events failed; keeping %s",
                len(batch), snapshot, exc_info=True,
            )
            return False
        logger.info("replayed %d journalled chat events", len(batch))
    try:
        os.unlink(snapshot)
    except OSError:  # pragma: no cover — concurrent unlink
        pass
    return True


def _write_event_batch(batch: list[tuple[str, str, float, str]]) -> None:
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
    with _rw_lock.write_lock(), _caches_lock:
        try:
            _write_event_batch_locked(db, batch, task_ids)
        except Exception:
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            # The seq cache may have advanced past rows that were
            # rolled back; recompute from the database on retry.
            for tid in task_ids:
                _next_seq_cache.pop(tid, None)
                _marked_has_events.discard(tid)
            raise


def _write_event_batch_locked(
    db: sqlite3.Connection,
    batch: list[tuple[str, str, float, str]],
    task_ids: set[str],
) -> None:
    """Insert *batch* inside one explicit transaction.

    Caller holds ``_rw_lock.write_lock()`` and ``_caches_lock`` and
    rolls back + invalidates the seq caches on any failure, so a
    mid-batch error can never diverge the cache from the database.
    """
    db.execute("BEGIN IMMEDIATE")
    for tid in task_ids:
        if tid not in _next_seq_cache:
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
    rows: list[tuple[str, int, str, float]] = []
    for tid, ev_json, ts, _op in batch:
        seq = _next_seq_cache.get(tid)
        if seq is None:
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
    db.execute("COMMIT")


def _queue_chat_event(
    event: dict[str, object],
    task_id: str,
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
    _reserve_pending(task_id)
    _event_queue.put((
        task_id,
        json.dumps(event),
        time.time(),
        origin_db_path or _current_db_path(),
    ))
    t = _event_writer_thread
    if t is None or not t.is_alive():
        _start_event_writer()


def _reserve_pending(task_id: str) -> None:
    """Record one queued-but-unwritten event for *task_id*."""
    with _pending_cond:
        _pending_by_task[task_id] = _pending_by_task.get(task_id, 0) + 1


def _release_pending(batch: list[tuple[str, str, float, str]]) -> None:
    """Mark every event in *batch* as no longer pending and wake waiters."""
    with _pending_cond:
        for task_id, _ev, _ts, _origin in batch:
            remaining = _pending_by_task.get(task_id, 0) - 1
            if remaining > 0:
                _pending_by_task[task_id] = remaining
            else:
                _pending_by_task.pop(task_id, None)
        _pending_cond.notify_all()
    for _ in batch:
        _event_queue.task_done()


def _pending_count(task_id: str | None) -> int:
    """Return the number of unwritten events, optionally for one task."""
    if task_id is None:
        return int(_event_queue.unfinished_tasks)
    return _pending_by_task.get(task_id, 0)


def _flush_chat_events(task_id: str | None = None) -> None:
    """Block until queued events have been persisted.

    Waits on the writer's condition variable rather than polling, and
    re-plays any events that were journalled while the database was
    unwritable so the caller's "everything is persisted" assumption
    actually holds.

    Safe to call when no events are queued (returns immediately).  MUST
    be called BEFORE acquiring ``_rw_lock.write_lock()`` from the same
    thread — the writer thread also takes that lock per batch, so
    calling this while holding the write lock would deadlock.

    Args:
        task_id: When given, only that task's events are waited for.
            Write ordering only matters within one task, so a caller
            updating task A must not be delayed by a high-volume run
            in task B.  ``None`` waits for every queued event.
    """
    while True:
        with _pending_cond:
            if _pending_count(task_id) == 0:
                break
            writer = _event_writer_thread
            if writer is not None and writer.is_alive():
                _pending_cond.wait(0.05)
                continue
        _start_event_writer()
    _replay_failed_events()


def _stop_event_writer() -> None:
    """Drain and stop the writer thread.  Used by ``_close_db``/tests.

    A producer can enqueue an event *after* the drain below but
    *before* the old writer observes its stop flag, in which case the
    stopped writer exits without consuming it.  The loop re-checks the
    queue after each stopped writer and drains again (starting a fresh
    writer if needed) until nothing is left unfinished, so no queued
    event is ever stranded.
    """
    global _event_writer_thread, _caches_db_key
    while True:
        _flush_chat_events()
        with _event_writer_lock:
            t = _event_writer_thread
            stop = _event_writer_stop
        if t is not None:
            stop.set()
            try:
                _event_queue.put_nowait(None)
            except queue.Full:  # pragma: no cover — unbounded queue
                pass
            t.join(timeout=5)
            if t.is_alive():
                logger.warning(
                    "event writer thread did not stop within 5s; "
                    "deferring writer cleanup until it exits"
                )
                return
            with _event_writer_lock:
                if _event_writer_thread is t:
                    _event_writer_thread = None
        if not _event_queue.unfinished_tasks:
            break
    with _caches_lock:
        _next_seq_cache.clear()
        _marked_has_events.clear()
        _caches_db_key = None


def _drain_events_at_exit() -> None:
    """``atexit`` hook: flush queued events before interpreter teardown.

    The writer is a daemon thread, so a process exiting right after
    ``JsonPrinter`` enqueued events would otherwise lose them.  Thread
    creation can fail during interpreter shutdown, hence best-effort.
    """
    try:
        _stop_event_writer()
    except Exception:  # pragma: no cover — interpreter shutdown edge
        logger.debug("event drain at exit failed", exc_info=True)


atexit.register(_drain_events_at_exit)


def _append_chat_event(
    event: dict[str, object],
    task_id: str | None = None,
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
    with _rw_lock.read_lock():
        db = _get_db()
        resolved = _resolve_task_id(db, task_id, task)
    if resolved is None:
        return
    _queue_chat_event(event, resolved, origin_db_path)
    _flush_chat_events()


def _task_has_events(task_id: str) -> bool:
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
    _flush_chat_events(task_id)
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT 1 FROM events WHERE task_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        return row is not None


def _descendant_task_ids(root_task_id: str) -> list[str]:
    """Return the ids of all tasks below *root_task_id*.

    Walks the ``parent_task_id`` tree breadth-first (cycle-guarded).
    The root itself is NOT included.

    Args:
        root_task_id: Stable ``task_history`` row id of the parent task.

    Returns:
        List of descendant task-id strings (children first, then
        grandchildren, ...).  Empty when the task spawned no sub-tasks.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        ids: list[str] = []
        seen: set[str] = {str(root_task_id)}
        frontier = [str(root_task_id)]
        while frontier:
            marks = ",".join("?" * len(frontier))
            rows = db.execute(
                "SELECT id FROM task_history "
                f"WHERE parent_task_id IN ({marks})",
                frontier,
            ).fetchall()
            frontier = [
                str(r["id"]) for r in rows if str(r["id"]) not in seen
            ]
            seen.update(frontier)
            ids.extend(frontier)
        return ids


def _changed_paths_of_tasks(task_ids: list[str]) -> set[str]:
    """Return file paths the given tasks changed, from persisted events.

    Collects the ``path`` of every persisted ``Write`` / ``Edit``
    ``tool_call`` event of *task_ids*.  The asynchronous event queue is
    flushed first so the very last writes of a just-finished task are
    visible.

    Used with :func:`_descendant_task_ids` by the end-of-task
    auto-commit to also commit files that sub-agents changed outside
    the tab's work_dir repository.

    Args:
        task_ids: Stable ``task_history`` row ids.

    Returns:
        Set of path strings (as recorded in the events, i.e. absolute
        for the standard file tools).  Empty when none of the tasks
        changed a file.
    """
    if not task_ids:
        return set()
    _flush_chat_events()
    with _rw_lock.read_lock():
        db = _get_db()
        paths: set[str] = set()
        for start in range(0, len(task_ids), 100):
            batch = task_ids[start:start + 100]
            marks = ",".join("?" * len(batch))
            rows = db.execute(
                "SELECT DISTINCT json_extract(event_json, '$.path') AS p "
                f"FROM events WHERE task_id IN ({marks}) "
                "AND json_extract(event_json, '$.type') = 'tool_call' "
                "AND json_extract(event_json, '$.name') IN ('Write', 'Edit')",
                batch,
            ).fetchall()
            paths.update(str(r["p"]) for r in rows if r["p"])
        return paths


def _fetch_events_for_task_id(
    db: sqlite3.Connection, task_id: str,
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
    task_id: str,
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


def _load_events_session_row(
    where_sql: str,
    params: tuple[object, ...],
) -> dict[str, object] | None:
    """Load one ``task_history`` row and its events as a session dict.

    Shared engine of :func:`_load_latest_chat_events_by_chat_id` and
    :func:`_load_chat_events_by_task_id` — runs ``_HISTORY_SELECT``
    plus *where_sql* under the read lock and converts the first
    matching row via :func:`_events_session_dict`.

    Args:
        where_sql: SQL appended to ``_HISTORY_SELECT`` (the
            WHERE/ORDER BY/LIMIT clauses).
        params: Bind parameters for *where_sql*.

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` when no row matches.
    """
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(_HISTORY_SELECT + where_sql, params).fetchone()
        if row is None:
            return None
        return _events_session_dict(
            db, str(row["id"]), row["task"], str(row["chat_id"] or ""),
            _row_to_extra_json(row),
        )


def _load_latest_chat_events_by_chat_id(
    chat_id: str,
) -> dict[str, object] | None:
    """Load the latest task and its events for a chat session.

    Finds the most recent NON-sub-agent task in the given chat session
    and returns its task description string and recorded events.
    Sub-agent rows (``extra.subagent`` present, identified by a
    non-NULL ``parent_task_id`` column) share the parent's ``chat_id``
    and are persisted AFTER the parent row, so a chat-id-only lookup
    (e.g. the
    webview's post-restart ``resumeSession``) must skip them —
    otherwise a restored parent tab would replay the last sub-agent's
    events and be styled as a sub-agent tab.  Sub-agent rows are only
    ever loaded explicitly by task id
    (:func:`_load_chat_events_by_task_id`).

    Args:
        chat_id: The string chat session identifier.

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if chat_id is ``""`` or has no
        non-sub-agent tasks.
    """
    if not chat_id:
        return None
    return _load_events_session_row(
        f"WHERE chat_id = ? AND {_HISTORY_NOT_SUBAGENT} "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (chat_id,),
    )


def _load_chat_events_by_task_id(
    task_id: str,
) -> dict[str, object] | None:
    """Load a specific task and its events by the task row ID.

    Unlike ``_load_latest_chat_events_by_chat_id`` which always picks
    the most recent task in a chat session, this loads the exact task
    identified by *task_id*.

    Args:
        task_id: The primary key of the ``task_history`` row.

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if no such row exists.
    """
    return _load_events_session_row("WHERE id = ?", (task_id,))


def _load_subagent_rows_by_parent_task_id(
    parent_task_id: str,
) -> list[dict[str, object]]:
    """Return persisted sub-agent rows whose parent is *parent_task_id*.

    Used by :meth:`VSCodeServer._replay_session` when the user clicks
    a parent task in the history sidebar: every sub-agent fanned out
    by the parent's ``run_parallel`` tool call is reopened in its own
    sub-agent tab so the loaded view mirrors the live execution
    layout.

    A sub-agent row is identified by its ``parent_task_id`` column
    matching *parent_task_id* — the dedicated column written by
    :meth:`ChatSorcarAgent._run_tasks_parallel`'s worker thread (the
    ``extra`` payload's ``subagent`` object is synthesized back from
    this column by :func:`_row_to_extra_json`).

    Args:
        parent_task_id: Primary key (32-hex UUID TEXT id) of the
            parent ``task_history`` row.

    Returns:
        List of dicts ordered by ``rowid`` ASC (the order in which
        the parent enqueued sub-agents).  Each dict has ``task_id``
        (str, the row's UUID TEXT id), ``task`` (str), ``chat_id``
        (str), ``events`` (list of event dicts), and ``extra`` (str,
        JSON metadata synthesized by :func:`_row_to_extra_json`).
        Empty list when no sub-agent rows exist.
    """
    if not isinstance(parent_task_id, str) or not parent_task_id:
        return []
    out: list[dict[str, object]] = []
    with _rw_lock.read_lock():
        db = _get_db()
        rows = db.execute(
            _HISTORY_SELECT
            + "WHERE parent_task_id = ? ORDER BY rowid ASC",
            (parent_task_id,),
        ).fetchall()
        for r in rows:
            sub_task_id = str(r["id"])
            out.append({
                "task_id": sub_task_id,
                "task": r["task"],
                "chat_id": str(r["chat_id"] or ""),
                "events": _fetch_events_for_task_id(db, sub_task_id),
                "extra": _row_to_extra_json(r),
            })
    return out


def _get_adjacent_task_by_chat_id(
    chat_id: str, current_task_id: str | None, direction: str
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
        A dict with ``task`` (str), ``task_id`` (str) and ``events``
        (list of event dicts), or ``None`` if no adjacent task exists.
    """
    if not chat_id or current_task_id is None:
        return None
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT rowid, id, timestamp FROM task_history "
            "WHERE id = ? AND chat_id = ?",
            (current_task_id, chat_id),
        ).fetchone()
        if not row:
            return None
        ts = row["timestamp"]
        cur_rowid = row["rowid"]

        if direction == "prev":
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? "
                "AND (timestamp < ? OR (timestamp = ? AND rowid < ?)) "
                f"AND {_HISTORY_NOT_SUBAGENT} "
                "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
                (chat_id, ts, ts, cur_rowid),
            ).fetchone()
        else:
            adj = db.execute(
                "SELECT id, task FROM task_history "
                "WHERE chat_id = ? "
                "AND (timestamp > ? OR (timestamp = ? AND rowid > ?)) "
                f"AND {_HISTORY_NOT_SUBAGENT} "
                "ORDER BY timestamp ASC, rowid ASC LIMIT 1",
                (chat_id, ts, ts, cur_rowid),
            ).fetchone()

        if not adj:
            return None

        adj_id = str(adj["id"])
        return {
            "task": adj["task"],
            "task_id": adj_id,
            "events": _fetch_events_for_task_id(db, adj_id),
        }


def _load_chat_context(chat_id: str) -> list[_HistoryEntry]:
    """Load all tasks and results for a chat session in chronological order.

    Sub-agent rows (those with a non-empty ``parent_task_id`` column —
    set by :class:`ChatSorcarAgent._run_tasks_parallel`
    on every worker thread's task row) are filtered out via the shared
    ``_HISTORY_NOT_SUBAGENT`` SQL predicate.  Sub-agent
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
            "SELECT task, result FROM task_history "
            f"WHERE chat_id = ? AND {_HISTORY_NOT_SUBAGENT} "
            "ORDER BY timestamp ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
        return [{"task": r["task"], "result": r["result"]} for r in rows]


def _load_task_chain_context(task_id: str) -> list[_HistoryEntry]:
    """Load the tasks and results along a task's parent chain.

    Starting at *task_id*, follows ``parent_task_id`` links upward
    (task → parent → grandparent → …) and returns the collected
    entries in chronological order (root ancestor first, the *task_id*
    row last) — the same shape :func:`_load_chat_context` returns, so
    :meth:`ChatSorcarAgent.build_chat_prompt` can consume either
    interchangeably.  Used when a tab is opened by a specific task id
    and the first task subsequently issued in that tab needs the
    opened task's lineage — rather than its whole chat — as context.

    Traversal is cycle-safe (each row id is visited at most once) and
    stops silently at a missing row or an empty ``parent_task_id``.

    Args:
        task_id: The ``task_history.id`` to start the traversal at.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered
        oldest ancestor first.  Empty when *task_id* is empty or has
        no persisted row.
    """
    if not task_id:
        return []
    with _rw_lock.read_lock():
        db = _get_db()
        entries: list[_HistoryEntry] = []
        seen: set[str] = set()
        current = task_id
        while current and current not in seen:
            seen.add(current)
            row = db.execute(
                "SELECT task, result, parent_task_id FROM task_history "
                "WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            entries.append({"task": row["task"], "result": row["result"]})
            current = _safe_str(row["parent_task_id"])
        entries.reverse()
        return entries


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
    current_path = _current_db_path()
    _maybe_reset_caches(current_path, _db_file_identity(current_path))
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
    from kiss.core.vscode_config import load_config

    return str(load_config().get("last_model", "") or "")


def _save_last_model(model: str) -> None:
    """Persist the selected model name as a user preference.

    Writes the ``last_model`` key to ``~/.kiss/config.json`` (atomic).
    Does **not** touch the SQLite usage counters.

    Passes *only* the changed key to :func:`save_config` — whose
    locked read-merge-write overlays every DEFAULTS key present in its
    argument onto a fresh re-read of the file — so a stale full-config
    snapshot taken here can never clobber a concurrent update to an
    unrelated key (e.g. a settings toggle saved from another thread).

    Args:
        model: The model name to save as the last-selected model.
    """
    from kiss.core.vscode_config import save_config

    save_config({"last_model": model})


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
    with _rw_lock.write_lock(), _immediate_txn(db):
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


def _record_frequent_task(task: str) -> None:
    """Increment the run-count of *task* and refresh its timestamp.

    Upserts a row in the ``frequent_tasks`` table so that subsequent
    calls with the same *task* increment its ``count`` and update its
    ``timestamp`` to ``time.time()``.

    The table is capped at ``_MAX_FREQUENT_TASKS`` rows.  When inserting
    a brand-new task would exceed the cap, the row with the lowest
    ``count`` (and, on a count tie, the oldest ``timestamp``) is
    evicted before the insert completes.  The whole probe → count →
    evict → upsert sequence runs in one ``BEGIN IMMEDIATE``
    transaction: as separate autocommit statements, two PROCESSES
    could both observe "cap not reached" and both insert, pushing the
    table permanently over the cap.

    Args:
        task: The task description string.  Empty strings are ignored.
    """
    if not task:
        return
    db = _get_db()
    now = time.time()
    with _rw_lock.write_lock(), _immediate_txn(db):
        existing = db.execute(
            "SELECT 1 FROM frequent_tasks WHERE task = ?", (task,),
        ).fetchone()
        if existing is None:
            row = db.execute("SELECT COUNT(*) FROM frequent_tasks").fetchone()
            _race_delay()
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
