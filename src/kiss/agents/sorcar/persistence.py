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

from kiss.core.config import kiss_home

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
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
            except BaseException:
                # An exception delivered while blocked in ``wait()`` (e.g.
                # a ``KeyboardInterrupt`` injected into a task thread to
                # stop it) must not leak the pending-writer count —
                # otherwise every future ``read_lock()`` blocks forever on
                # ``self._pending_writers > 0``.  Wake blocked readers so
                # they re-check the predicate.
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
# go through :func:`_dumps_extra` so the two sides can never diverge.

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
    """Close the calling thread's connection and invalidate cached handles.

    Only the current thread's connection is closed eagerly.  Bumping
    the generation counter invalidates every other thread's cached
    connection *lazily*: each is detected as stale on that thread's
    next ``_get_db()`` call and replaced (and closed) there.
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
        # ``_db_conn`` is a backward-compat alias for the last-created
        # connection; never leave it pointing at a closed handle.
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

# SQL predicate that is TRUE for every row that is NOT a sub-agent row.
# A sub-agent row carries a non-empty ``parent_task_id`` column.
_HISTORY_NOT_SUBAGENT = "(parent_task_id IS NULL OR parent_task_id = '')"

# Pattern for a task_history.id minted by ``uuid.uuid4().hex``.
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
        # r3-H3: emit every typed column consistently — including the
        # falsy / zero cases — so consumers that test for key presence
        # do not see a behaviour change between two rows that differ
        # only in whether a numeric field happens to be zero.  Only
        # the ``subagent`` nested dict remains gated on a non-empty
        # ``parent_task_id`` because its absence is the canonical
        # marker for a top-level (non-sub-agent) task in the
        # downstream classifier.
        # bughunt8: coerce through ``_safe_str`` — a BLOB stored in
        # one of these TEXT columns made ``json.dumps`` raise
        # ``TypeError`` out of ``_load_history``/``_search_history``.
        payload["model"] = _safe_str(row["model"])
        payload["work_dir"] = _safe_str(row["work_dir"])
        payload["version"] = _safe_str(row["version"])
        payload["auto_commit_mode"] = bool(row["auto_commit_mode"])
        # bughunt2: use the finite-aware safe coercers instead of bare
        # ``int()``/``float()``.  SQLite's dynamic typing lets a
        # hand-edited / 3rd-party-source DB store TEXT (e.g. ``'abc'``)
        # in the REAL/INTEGER columns; bare coercion raised
        # ``ValueError`` — uncaught by the (KeyError, IndexError)
        # handler below — which propagated out of ``_load_history`` /
        # ``_search_history`` / ``_load_chat_events_by_task_id`` and
        # blanked the whole history sidebar over one corrupt row.
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
    # Route through ``_dumps_extra`` so any non-finite ``cost`` (e.g.
    # from a hand-edited / 3rd-party-source DB) gets normalised
    # rather than emitted as a bare ``NaN``/``Infinity`` token that
    # SQLite's ``json_valid`` and downstream strict parsers reject.
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
    * ``Task interrupted by server restart/shutdown`` — a graceful
      daemon/server shutdown cancellation.  It is likewise an
      incomplete task outcome rather than a success.
    """
    return (
        result.startswith("Task failed")
        or result == "Agent Failed Abruptly"
        or result == "Task terminated unexpectedly (process killed)"
        or result == "Task stopped by user"
        or result == "Task interrupted by server restart/shutdown"
    )


# Index DDL shared by ``_init_tables`` and the legacy-schema migration.
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
            is_parallel INTEGER DEFAULT 0,
            is_worktree INTEGER DEFAULT 0,
            auto_commit_mode INTEGER DEFAULT 0,
            start_ts INTEGER DEFAULT 0,
            end_ts INTEGER DEFAULT 0,
            is_favorite INTEGER DEFAULT 0,
            parent_task_id TEXT DEFAULT ''
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
    # r3-C2: do a FAST autocommit probe first so we can early-return
    # without entering a write transaction when the DB already has
    # the new schema.  The DEFINITIVE probe must re-run inside the
    # write transaction below to defeat a TOCTOU race against a
    # concurrent process that migrates between the probe and our
    # ``BEGIN IMMEDIATE``.
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

    def _bx(v: object) -> int:
        # r6-persistence-H3: ``bool("false") == True`` and
        # ``bool("0") == True`` — a naive ``bool(v)`` corrupts legacy
        # JSON-extra payloads that happen to encode their flags as
        # string literals ("false", "0", "False").  Normalise the
        # common false-y string forms before falling back to
        # ``bool()`` for everything else (int / real-bool / dict).
        if isinstance(v, str):
            return 0 if v.strip().lower() in {"", "0", "false", "no"} else 1
        return 1 if bool(v) else 0

    # Wrap the entire migration body in an explicit transaction so a
    # mid-migration crash leaves the DB unchanged.  The connection is
    # in autocommit mode (``isolation_level=None``); ``BEGIN IMMEDIATE``
    # opens a write transaction that is rolled back unless ``COMMIT``
    # runs at the end.  IMPORTANT: only ``execute()`` calls may run
    # inside the transaction — ``executescript()`` issues an implicit
    # COMMIT before its body which would silently end the
    # transaction and defeat the atomicity guarantee.
    # r6-persistence-H7: temporarily disable foreign-key enforcement
    # during the rename dance.  The CREATE TABLE for ``events__new``
    # declares ``REFERENCES task_history__new(id)``; after we
    # ALTER ... RENAME TO ``task_history`` the FK target name no
    # longer matches under SQLite < 3.26 or with
    # ``legacy_alter_table=1``, which can leave a stale FK and break
    # later inserts.  Restoring ``foreign_keys=ON`` after the
    # rename is safe because the orphan-events pre-scan above
    # already excluded events that would fail FK.
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # r3-C2: re-probe inside the transaction so another process
        # that migrated between our autocommit probe and our
        # BEGIN IMMEDIATE is detected before we corrupt its work.
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
        # r3-H4: drop any leftover ``__new`` tables from a prior
        # crashed migration INSIDE the transaction so the DROPs are
        # atomic with the rest of the migration.
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
            "is_parallel INTEGER DEFAULT 0, "
            "is_worktree INTEGER DEFAULT 0, "
            "auto_commit_mode INTEGER DEFAULT 0, "
            "start_ts INTEGER DEFAULT 0, "
            "end_ts INTEGER DEFAULT 0, "
            "is_favorite INTEGER DEFAULT 0, "
            "parent_task_id TEXT DEFAULT ''"
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
        # Read every legacy task_history row in stable insertion order
        # so the post-migration ``rowid`` tiebreaker preserves
        # chronology across the upgrade.
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
                    # Already-canonical UUID-shaped string survives.
                    # Garbage strings (e.g. ``"123"`` from a buggy
                    # 3rd-party migration) are rejected so they don't
                    # land in the TEXT column as a value no future
                    # query can resolve.
                    parent_task_id = old_parent
            # Count any unknown keys for an after-the-fact warning so
            # the upgrade is auditable; the new schema has no overflow
            # column so unknown keys are necessarily lost.
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
                    _bx(extra.get("is_parallel")),
                    _bx(extra.get("is_worktree")),
                    _bx(extra.get("auto_commit_mode")),
                    _safe_int(extra.get("startTs")),
                    _safe_int(extra.get("endTs")),
                    _bx(extra.get("is_favorite")), parent_task_id,
                ),
            )
        # Probe for the events table — early/manually-wiped legacy DBs
        # may have a task_history but no events table.  Without this
        # guard the SELECT below raises and the migration aborts.
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
        # r6-persistence-H7: restore FK enforcement on the error path
        # so the connection state remains symmetric with the OFF set
        # above.
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error:
            pass
        raise
    # r6-persistence-H7: restore FK enforcement on the success path.
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


def _db_file_identity(path: str) -> tuple[int, int] | None:
    """Return the ``(st_dev, st_ino)`` identity of *path*, or ``None``.

    ``None`` means the file does not currently exist (or cannot be
    stat'ed).  The identity distinguishes a file that was deleted and
    recreated at the same pathname from the original file — a plain
    existence check cannot.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _get_db() -> sqlite3.Connection:
    """Return a per-thread database connection, creating one if needed.

    Each calling thread gets its own ``sqlite3.Connection`` so that
    concurrent threads never share cursor state.  Connections are
    cached in ``threading.local()`` and invalidated when:

    * ``_db_generation`` is bumped (via ``_close_db()``),
    * ``_DB_PATH`` changes (test redirects),
    * ``_db_conn`` is set to ``None`` (legacy test pattern), or
    * the file at ``_DB_PATH`` is deleted or replaced on disk (its
      ``(st_dev, st_ino)`` identity no longer matches the one the
      cached connection was opened against).
    """
    global _db_conn
    tl = _thread_local
    tl_conn: sqlite3.Connection | None = getattr(tl, "conn", None)
    tl_gen: int = getattr(tl, "gen", -1)
    tl_path: str | None = getattr(tl, "path", None)
    # Snapshot the generation ONCE, before any connection work.  A
    # concurrent ``_close_db()`` can bump ``_db_generation`` at any
    # point while this thread is creating a connection below; stamping
    # ``tl.gen`` from a *re-read* of the global would tag a connection
    # created under the old generation with the new one, making it
    # survive an invalidation it should not.  With the snapshot, a
    # bump that lands mid-creation leaves ``tl.gen`` stale, so the
    # next ``_get_db()`` call detects it and reconnects.
    gen_snapshot = _db_generation
    current_path = str(_DB_PATH)
    current_id = _db_file_identity(current_path)
    _maybe_reset_caches(current_path, current_id)

    if (
        tl_conn is not None
        and tl_gen == gen_snapshot
        and tl_path == current_path
        and _db_conn is not None
        # The file on disk must be the SAME file the cached connection
        # was opened against.  A cached connection whose backing file
        # was deleted — and possibly recreated at the same pathname by
        # an independent ``sqlite3.connect`` (log rotation, a test
        # cleaning up ``$KISS_HOME``, a user removing
        # ``~/.kiss/sorcar.db`` while the daemon runs) — keeps writing
        # into the orphaned inode: every subsequent INSERT silently
        # disappears, while any NEW reader of ``_DB_PATH`` sees an
        # empty database with no ``task_history`` table.  A bare
        # existence check is not enough (the recreated file exists),
        # so the ``(st_dev, st_ino)`` identity recorded when the
        # connection was opened is compared instead; a mismatch falls
        # through to the reconnect path, which (re)creates the file
        # and its schema.
        and current_id is not None
        and getattr(tl, "file_id", None) == current_id
    ):
        return tl_conn

    # Stale or missing — close old thread-local connection
    if tl_conn is not None:
        try:
            tl_conn.close()
        except Exception:
            pass

    _ensure_kiss_dir()
    # Stale -wal/-shm cleanup for a deleted main database file.  Both
    # the existence check and the unlink targets are derived from the
    # ONE ``current_path`` snapshot taken above — never from a re-read
    # of the ``_DB_PATH`` global.  Re-reading it here is a TOCTOU race:
    # a test (or embedder) can redirect ``_DB_PATH`` to a scratch
    # directory, delete that scratch database, and restore the original
    # path at any moment.  With a re-read, this thread could observe
    # "missing" for the SCRATCH file but compute the unlink targets
    # from the freshly RESTORED path — deleting the live shared
    # database's -wal/-shm side files out from under every open
    # connection.  SQLite then fails all NEW connections to that
    # database with a permanent ``disk I/O error`` for as long as any
    # old connection keeps the unlinked -shm mapped.
    if not os.path.exists(current_path):
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(current_path + suffix)
            except OSError:
                pass
    conn = sqlite3.connect(
        current_path,
        check_same_thread=False,
        timeout=10,
        isolation_level=None,  # true autocommit — no implicit BEGIN
    )
    # Arm the busy handler BEFORE the journal-mode switch: a fresh
    # connection whose very first statement is ``PRAGMA
    # journal_mode=WAL`` would otherwise race a concurrent writer with
    # only the ``connect(timeout=...)`` handler, and the WAL switch is
    # exactly the statement most likely to collide on a brand-new
    # database (see below).
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    # Serialize first-time DDL across concurrent threads so CREATE TABLE
    # statements (run by every fresh thread-local connection) don't race
    # and trigger "database is locked" errors under heavy parallelism.
    # NOTE: must use a dedicated lock (not ``_rw_lock``) — many callers
    # already hold ``_rw_lock.read_lock()`` when invoking ``_get_db()``,
    # so reusing the RW lock for DDL would self-deadlock.
    #
    # The ``PRAGMA journal_mode=WAL`` switch must sit INSIDE the same
    # critical section: entering WAL mode takes an exclusive lock on
    # the database file, and SQLite returns SQLITE_BUSY *immediately*
    # (without consulting the busy handler) when another connection
    # holds a conflicting lock during the mode change.  Two parallel
    # sub-agent threads opening their first connections against a
    # fresh ``sorcar.db`` deterministically hit this: one thread is
    # mid-DDL while the other executes the WAL pragma and dies with
    # ``sqlite3.OperationalError: database is locked``.  The lock
    # serialises in-process racers; the retry loop absorbs the same
    # collision from OTHER processes sharing the database file (e.g.
    # the kiss-web daemon starting while a CLI task initialises).
    with _init_tables_lock:
        wal_deadline = time.monotonic() + 30.0
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                # Retry ONLY lock contention (SQLITE_BUSY /
                # SQLITE_LOCKED) — other operational errors (readonly
                # database, disk I/O, corrupt header) are permanent and
                # must surface immediately.  The deadline mirrors the
                # 30s ``busy_timeout`` configured above.
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
    # Record the identity of the file this connection is bound to.
    # ``_init_tables`` above has materialised the main DB file even
    # for a brand-new database, so the stat cannot miss.
    tl.file_id = _db_file_identity(current_path)
    _db_conn = conn  # backward compat: expose last-created connection
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
    # r4-persistence-C1/C2/H5: accept all three shapes symmetric with
    # ``_save_task_extra``.  Reject collisions explicitly.
    if sub is not None and flat_parent is not None:
        raise ValueError(
            "Cannot pass both 'parent_task_id' and 'subagent' to _add_task",
        )
    if isinstance(sub, dict):
        parent_task_id = _coerce_parent_task_id(sub.get("parent_task_id"))
    elif isinstance(sub, str):
        # convenience ``{"subagent": "<uuid>"}`` shape.
        parent_task_id = _coerce_parent_task_id(sub)
    elif flat_parent is not None:
        parent_task_id = _coerce_parent_task_id(flat_parent)
    with _rw_lock.write_lock():
        if chat_id == "":
            # Mint through the SAME :func:`_allocate_chat_id` helper the
            # agents use so the mint paths can never drift (e.g. if the
            # helper ever gains reservation / collision-check side
            # effects).
            chat_id = _allocate_chat_id()
        task_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, chat_id, result, "
            "model, work_dir, version, tokens, cost, steps, is_parallel, "
            "is_worktree, auto_commit_mode, start_ts, end_ts, is_favorite, "
            "parent_task_id) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        db.commit()
    # Invalidate the autocomplete chat-context cache so the new task's
    # text becomes visible to ghost-text suggestions on the next
    # keystroke.  Done outside the write lock to keep the critical
    # section short.
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


def _get_task_chat_id(task_id: str) -> str:
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


def _get_task_start_ts(task_id: str) -> int:
    """Return the ``start_ts`` of the task with the given row id, or 0.

    Args:
        task_id: The primary key of the task_history row.

    Returns:
        The task's recorded start timestamp (epoch seconds), or ``0``
        when the row is missing, its ``start_ts`` column is empty, or
        *task_id* is ``""``.
    """
    if not task_id:
        return 0
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT start_ts FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        return _safe_int(row["start_ts"], 0) if row else 0


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
    db: sqlite3.Connection, parent_task_id: str,
) -> list[str]:
    """Return ids of persisted sub-agent rows whose parent is *parent_task_id*.

    Children are identified by the dedicated ``parent_task_id`` column.
    Callers must hold ``_rw_lock`` (read or write).

    Args:
        db: Active database connection.
        parent_task_id: Primary key of the parent ``task_history`` row.

    Returns:
        List of child row ids (possibly empty).
    """
    if not parent_task_id:
        return []
    rows = db.execute(
        "SELECT id FROM task_history WHERE parent_task_id = ?",
        (parent_task_id,),
    ).fetchall()
    return [str(r["id"]) for r in rows]


def _delete_task(task_id: str) -> bool:
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
        doomed_ids: list[str] = [task_id]
        if row is not None:
            # Breadth-first walk: a sub-agent is a full ChatSorcarAgent
            # that can itself call ``run_parallel``, so grandchildren
            # (and deeper) rows exist and must be cascade-deleted too.
            # The ``seen`` set guards against corrupt self/cyclic
            # ``parent_task_id`` references.
            seen: set[str] = {task_id}
            frontier: list[str] = [task_id]
            while frontier:
                next_frontier: list[str] = []
                for parent_id in frontier:
                    for child_id in _subagent_child_ids(db, parent_id):
                        if child_id not in seen:
                            seen.add(child_id)
                            next_frontier.append(child_id)
                doomed_ids.extend(next_frontier)
                frontier = next_frontier
        deleted = False
        for did in doomed_ids:
            db.execute("DELETE FROM events WHERE task_id = ?", (did,))
            cursor = db.execute(
                "DELETE FROM task_history WHERE id = ?", (did,)
            )
            if did == task_id:
                deleted = (cursor.rowcount or 0) > 0
            _next_seq_cache.pop(did, None)
            _marked_has_events.discard(did)
        db.commit()
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
        # Over-fetch so that duplicate task strings (a single task run
        # many times) still leave room for *limit* distinct entries.
        rows = db.execute(
            "SELECT task FROM task_history "
            "WHERE task GLOB ? AND LENGTH(task) > ? "
            f"AND {_HISTORY_NOT_SUBAGENT} "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
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
    # r4-persistence-H3: reject non-string / malformed task_id values
    # rather than letting them silently TEXT-coerce inside SQLite and
    # never match.  Fall back to ``_most_recent_task_id`` so legacy
    # JSON-RPC clients with stale int task_id are still resolvable.
    if isinstance(task_id, str) and task_id != "":
        if not is_task_history_id(task_id):
            return _most_recent_task_id(db, task)
        row = db.execute(
            "SELECT id FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
        if row is not None:
            return str(row["id"])
        # bughunt2: a WELL-FORMED ``uuid4().hex`` id that matches no
        # row belongs to a DELETED task (user removed a running task
        # from the history sidebar, or a stale finished-tab id).  The
        # write must be dropped — falling back to
        # ``_most_recent_task_id`` here redirected the deleted task's
        # result/extra/event writes onto an unrelated task (callers
        # pass ``task=None``, so the fallback resolved to the most
        # recent row overall).  The legacy fallback above is only for
        # malformed (pre-UUID) ids that can never collide this way.
        return None
    return _most_recent_task_id(db, task)


def _log_orphaned_task_forensics(
    db: sqlite3.Connection,
    extra_clause: str,
    extra_params: list[object] | None = None,
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
        extra_clause: SQL fragment further restricting the sentinel
            rows (``""``, or any combination of
            ``"AND id NOT IN (?,?,...)"`` and
            ``"AND timestamp < ?"``).
        extra_params: Bound-parameter values matching the placeholders
            embedded in ``extra_clause``.  Must be supplied (and
            ordered) when the clause is non-empty.
    """
    params: list[object] = ["Agent Failed Abruptly"]
    if extra_params:
        params.extend(extra_params)
    diag_rows = db.execute(
        "SELECT id, task, chat_id, model, start_ts, steps, cost "
        "FROM task_history WHERE result = ? " + extra_clause,
        params,
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
                # bughunt8: coerce through the finite-aware helper —
                # SQLite's dynamic typing lets a hand-edited /
                # 3rd-party-source DB store TEXT/BLOB in the REAL
                # ``timestamp`` column, and the ``:.1f`` format below
                # runs OUTSIDE this try, so a raw read raised
                # ValueError out of ``_recover_orphaned_tasks`` and
                # crashed the whole startup sweep over one corrupt row.
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

    Args:
        active_task_ids: Row ids that are still being processed in
            the current process and must therefore NOT be rewritten.
            Pass an empty set at fresh-server startup — by then any
            row carrying the sentinel must belong to a prior,
            now-dead process.
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
    # r3-H5: use bound parameter placeholders for the active-task
    # id list rather than inlining them as SQL string literals.
    # SQLite's bound-parameter limit (~999) is far above any
    # realistic active-task count and ``?`` placeholders sidestep
    # the SQL-injection surface entirely.
    active_ids = [str(t) for t in active_task_ids]
    extra_clause = ""
    extra_params: list[object] = []
    if active_ids:
        placeholders = ",".join(["?"] * len(active_ids))
        extra_clause += f"AND id NOT IN ({placeholders}) "
        extra_params.extend(active_ids)
    if created_before is not None:
        extra_clause += "AND timestamp < ? "
        extra_params.append(float(created_before))
    sql = (
        "UPDATE task_history SET result = ? WHERE result = ? " + extra_clause
    )
    params: list[object] = [
        "Task terminated unexpectedly (process killed)",
        "Agent Failed Abruptly",
    ]
    params.extend(extra_params)
    with _rw_lock.write_lock():
        _log_orphaned_task_forensics(db, extra_clause, extra_params)
        cursor = db.execute(sql, params)
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
    # r3-H5: use ``?`` placeholders for the id list rather than
    # inlining string literals.  Eliminates the SQL-injection
    # surface (defensive even though valid hex never contains a
    # quote) and tightens contract.
    id_list = [str(t) for t in task_ids]
    placeholders = ",".join(["?"] * len(id_list))
    sql = (
        f"UPDATE task_history SET result = ? "
        f"WHERE id IN ({placeholders}) AND result = ?"
    )
    affected_chat_ids: list[str] = []
    with _rw_lock.write_lock():
        # Capture chat_ids of rows we are about to rewrite so we can
        # invalidate the autocomplete chat-context cache afterwards.
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
    task_id: str | None,
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
    # Invalidate the autocomplete chat-context cache so the updated
    # result text becomes visible to ghost-text suggestions on the next
    # keystroke.  Done outside the write lock to keep the critical
    # section short.
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
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        cursor = db.execute(
            "UPDATE task_history SET is_favorite = ? WHERE id = ?",
            (1 if is_favorite else 0, task_id),
        )
        db.commit()
        return (cursor.rowcount or 0) > 0


# Maps each legacy ``extra`` key to its (column_name, caster, default)
# tuple.  Keys absent from *extra* are NOT included in the UPDATE,
# which automatically preserves any other column (e.g. ``is_favorite``
# set independently via :func:`_set_task_favorite`).
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
    # r3-H1: ``is_favorite`` is intentionally NOT in this map.
    # ``_set_task_favorite`` is the only sanctioned writer for the
    # favorite flag; including it here would let a future caller of
    # ``_save_task_extra({"is_favorite": False})`` silently clear a
    # previously-set star.
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
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        resolved = _resolve_task_id(db, task_id, task)
        if resolved is None:
            return
        pairs: list[tuple[str, object]] = []
        for k, v in extra.items():
            # r5-persistence-C2: ``is_favorite`` is intentionally NOT
            # in ``_EXTRA_COL_MAP`` so the favorite flag (owned by
            # ``_set_task_favorite``) is preserved across normal
            # metadata updates.  Silently dropping a caller's
            # ``{"is_favorite": True}`` would leave the caller
            # convinced the flag was set when it wasn't.  Raise so
            # the bug surfaces.
            if k == "is_favorite":
                raise ValueError(
                    "_save_task_extra does not write 'is_favorite'; "
                    "use _set_task_favorite() instead"
                )
            mapping = _EXTRA_COL_MAP.get(k)
            if mapping is None:
                # Top-level parent_task_id passthrough (new flat shape).
                # r3-C1: only emit the UPDATE when the coerced value
                # is a real UUID.  Writing ``""`` here would silently
                # re-parent an existing sub-agent row to the
                # top-level history sidebar (the same trap fixed for
                # the ``subagent`` nested branch below).
                if k == "parent_task_id":
                    # r3-H2: refuse to silently honour both the flat
                    # and nested shapes when both are present.
                    if "subagent" in extra:
                        raise ValueError(
                            "Cannot pass both 'parent_task_id' and "
                            "'subagent' to _save_task_extra"
                        )
                    coerced = _coerce_parent_task_id(v)
                    if coerced:
                        pairs.append(("parent_task_id = ?", coerced))
                    continue
                # Legacy nested {"subagent": {"parent_task_id": ...}}
                # OR convenience {"subagent": "<uuid>"} shape.
                if k == "subagent":
                    if isinstance(v, dict):
                        raw_parent = v.get("parent_task_id")
                    else:
                        raw_parent = v
                    coerced = _coerce_parent_task_id(raw_parent)
                    # CRITICAL: only emit the UPDATE when the coerced
                    # value is a real UUID.  Writing ``""`` here would
                    # silently re-parent an existing sub-agent row to
                    # the top-level history sidebar.
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
_next_seq_cache: dict[str, int] = {}
_marked_has_events: set[str] = set()
# (path, file identity) key the seq/has_events caches were last
# populated against.  When ``_DB_PATH`` is reassigned (test fixtures)
# OR the file at the same pathname is replaced on disk, the caches are
# stale and must be cleared on the next ``_get_db()`` reconnect.
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
    # The chat-context text cache was seeded from the previous database
    # too — and ``_load_chat_context_text`` consults it BEFORE any
    # ``_get_db()`` reconnect, so without this it would keep serving the
    # old database's task/result text forever after a swap or removal.
    # ``_chat_context_cache_lock`` is a leaf lock, safe to take here.
    _invalidate_chat_context_cache("")

# Per-batch tuning.  256 events at 50 µs JSON encoding ≈ 13 ms of producer
# work bundled into one SQL transaction.  20 ms collection window keeps
# end-to-end latency under ~25 ms for any single event.
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
        # Each writer gets its OWN stop event (passed as an argument and
        # published in ``_event_writer_stop`` alongside the thread, both
        # under ``_event_writer_lock``).  Re-using one shared event
        # required a ``clear()`` here, which could un-stop a concurrent
        # ``_stop_event_writer``'s target writer and leave two live
        # writers draining the same FIFO queue.
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
            # Shutdown sentinel
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
            _write_event_batch(batch)
        except Exception:
            # WARNING, not DEBUG: a persistent failure (disk full, schema
            # mismatch, corruption) silently discards up to _BATCH_MAX
            # user-visible chat events per batch — it must be visible in
            # default logging configurations.
            logger.warning("event writer batch failed", exc_info=True)
        finally:
            for _ in batch:
                _event_queue.task_done()
        if shutdown_pending:
            return


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
    # ``_caches_lock`` is held for the whole seed-then-consume sequence so a
    # concurrent ``_maybe_reset_caches`` (triggered by a ``_DB_PATH`` swap in
    # another thread's ``_get_db()``) cannot clear ``_next_seq_cache`` between
    # seeding and consumption — which silently dropped events (``seq is
    # None``) whose batch had already passed the origin-path filter.
    # ``_caches_lock`` is a leaf lock: it is never held while acquiring
    # ``_rw_lock``, so taking it inside the write lock cannot deadlock.
    with _rw_lock.write_lock(), _caches_lock:
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
        rows: list[tuple[str, int, str, float]] = []
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
    _event_queue.put((
        task_id,
        json.dumps(event),
        time.time(),
        origin_db_path or _current_db_path(),
    ))
    t = _event_writer_thread
    if t is None or not t.is_alive():
        _start_event_writer()


def _flush_chat_events() -> None:
    """Block until all queued events have been persisted.

    Safe to call when no events are queued (returns immediately).  MUST
    be called BEFORE acquiring ``_rw_lock.write_lock()`` from the same
    thread — the writer thread also takes that lock per batch, so
    calling this while holding the write lock would deadlock.
    """
    # Poll instead of a single liveness-check + ``Queue.join()``: a
    # concurrent ``_stop_event_writer()`` can terminate the writer AFTER
    # this thread's one-shot ``is_alive()`` check, stranding queued items
    # with no live writer — ``join()`` would then block forever waiting
    # for a ``task_done()`` no writer will ever call.  Re-checking writer
    # liveness on every iteration (and restarting it while items remain)
    # guarantees the backlog is always drained.
    while _event_queue.unfinished_tasks:
        t = _event_writer_thread
        if t is None or not t.is_alive():
            _start_event_writer()
        time.sleep(0.002)


def _stop_event_writer() -> None:
    """Drain and stop the writer thread.  Used by ``_close_db``/tests."""
    global _event_writer_thread, _caches_db_key
    # Drain first: ``_flush_chat_events`` restarts a dead writer while
    # items remain, so this cannot block forever on a queue no writer
    # is draining (unlike a bare ``_event_queue.join()``).
    _flush_chat_events()
    # Snapshot thread + its paired stop event under the lock so a
    # concurrent ``_start_event_writer`` cannot swap in a new pair
    # between the two reads.  The join below deliberately happens
    # OUTSIDE the lock so a wedged writer never blocks producers'
    # ``_start_event_writer`` calls for the 5s join timeout.
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
            # The writer is wedged (e.g. a slow batch on a locked DB).
            # Leave the stop flag set so it exits as soon as it unwedges,
            # and keep the thread reference so ``_start_event_writer``
            # cannot spawn a second concurrent writer draining the same
            # FIFO queue — two writers would interleave batches and break
            # the per-task event ordering guarantee.  Caches are left for
            # the zombie's in-flight batch; ``_maybe_reset_caches`` resets
            # them on the next ``_get_db()`` after any path change.
            logger.warning(
                "event writer thread did not stop within 5s; "
                "deferring writer cleanup until it exits"
            )
            return
        with _event_writer_lock:
            # CAS-style: only null the reference if it still names the
            # writer we just stopped.  A concurrent ``_start_event_writer``
            # may have already published a fresh live writer (with its own
            # stop event) that must not be clobbered — clobbering it would
            # let a later start spawn a SECOND concurrent writer.
            if _event_writer_thread is t:
                _event_writer_thread = None
    with _caches_lock:
        _next_seq_cache.clear()
        _marked_has_events.clear()
        _caches_db_key = None


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


def _amend_last_talk_tool_call_audio(
    task_id: str, audio_b64: str, audio_mime: str,
) -> bool:
    """Attach a synthesized clip to the last persisted ``talk`` tool call.

    The agentic loop persists a ``tool_call`` event for the ``talk``
    tool BEFORE the tool executes, so the recorded event can never
    carry the audio clip synthesized DURING execution.  Demo-mode
    replay plays exactly ``extras.audioB64`` of that persisted event
    (the synthesis fallback is gone), so without this amendment every
    replayed ``talk`` narration would be silent.  Called by
    :meth:`JsonPrinter.attach_talk_audio` right after synthesis
    succeeds; finds the most recent ``talk`` ``tool_call`` event row
    for *task_id* that does not yet carry audio and rewrites its
    ``extras`` with the clip in place (an UPDATE — appending a second
    copy would render the panel twice on replay).

    Args:
        task_id: Stable ``task_history`` row id the event was
            persisted under.
        audio_b64: Base64-encoded synthesized clip bytes.
        audio_mime: The clip's MIME type (e.g. ``"audio/mpeg"``).

    Returns:
        ``True`` when a persisted event row was amended, ``False``
        when no matching audio-less ``talk`` tool call exists.
    """
    if not task_id or not audio_b64:
        return False
    # The talk tool_call event reaches the events table through the
    # asynchronous background writer; flush so the row we must amend
    # is guaranteed to be visible (and so the UPDATE cannot be
    # overtaken by its own INSERT).
    _flush_chat_events()
    with _rw_lock.write_lock():
        db = _get_db()
        rows = db.execute(
            "SELECT id, event_json FROM events "
            "WHERE task_id = ? AND event_json LIKE '%\"tool_call\"%' "
            "AND event_json LIKE '%\"talk\"%' "
            "ORDER BY seq DESC",
            (task_id,),
        ).fetchall()
        for row in rows:
            try:
                event = json.loads(row["event_json"])
            except (TypeError, ValueError):
                continue
            if event.get("type") != "tool_call" or event.get("name") != "talk":
                continue
            extras = event.get("extras")
            if not isinstance(extras, dict):
                extras = {}
                event["extras"] = extras
            if extras.get("audioB64"):
                # Already amended (e.g. a retried synthesis); the most
                # recent audio-less call is the one being spoken now.
                continue
            extras["audioB64"] = audio_b64
            extras["audioMime"] = audio_mime
            db.execute(
                "UPDATE events SET event_json = ? WHERE id = ?",
                (json.dumps(event), row["id"]),
            )
            db.commit()
            return True
    return False


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
    _flush_chat_events()
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            "SELECT 1 FROM events WHERE task_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        return row is not None


def _list_recent_chats(limit: int = 10) -> list[dict[str, object]]:
    """List recent chat sessions with their tasks and results.

    Returns the most recent *limit* distinct chat sessions, ordered by
    most-recent-first.  Each entry contains the ``chat_id`` and a list
    of ``tasks`` (each with ``task``, ``result``, ``timestamp``,
    ``task_id``, and ``parent_task_id``) in chronological order.

    Sub-agent rows (``extra.subagent``, identified by a non-NULL
    ``parent_task_id`` column) are excluded — they are an internal
    implementation detail of the parent's ``run_parallel`` tool call,
    exactly as in every other
    chat/history reader (:func:`_load_history`,
    :func:`_load_chat_context`, ...).  A chat whose only rows are
    sub-agent rows is omitted entirely.

    Args:
        limit: Maximum number of chat sessions to return.

    Returns:
        List of dicts, each with ``chat_id`` (str) and ``tasks``
        (list of dicts with ``task``, ``result``, ``timestamp``,
        ``task_id``, and ``parent_task_id``).
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
                "SELECT id, task, result, timestamp, parent_task_id "
                "FROM task_history "
                f"WHERE chat_id = ? AND {_HISTORY_NOT_SUBAGENT} "
                "ORDER BY timestamp ASC, rowid ASC",
                (cid,),
            ).fetchall()
            # Surface both the row's own ``task_id`` (``id`` column)
            # and its ``parent_task_id`` so callers — chiefly the CLI
            # ``/resume`` listing — can display per-task identity and
            # the sub-agent parent relationship.  Sub-agent rows are
            # excluded by the shared ``_HISTORY_NOT_SUBAGENT`` SQL
            # predicate (same classifier as every other reader) so the
            # resume picker stays focused on the user-driven tasks;
            # consequently every returned ``parent_task_id`` is the
            # empty string here, but the key is always present for a
            # stable schema.
            task_dicts = [
                {"task": t["task"], "result": t["result"],
                 "timestamp": t["timestamp"],
                 "task_id": t["id"],
                 "parent_task_id": t["parent_task_id"] or ""}
                for t in tasks
            ]
            if not task_dicts:
                continue
            result.append({"chat_id": cid, "tasks": task_dicts})
        return result


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
    with _rw_lock.read_lock():
        db = _get_db()
        # Sub-agent rows are excluded by the shared
        # ``_HISTORY_NOT_SUBAGENT`` SQL predicate — the same
        # classifier every other history/chat reader uses — so only
        # the single newest real row is fetched.
        row = db.execute(
            _HISTORY_SELECT
            + f"WHERE chat_id = ? AND {_HISTORY_NOT_SUBAGENT} "
            "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        return _events_session_dict(
            db, str(row["id"]), row["task"], chat_id,
            _row_to_extra_json(row),
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
    with _rw_lock.read_lock():
        db = _get_db()
        row = db.execute(
            _HISTORY_SELECT + "WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        return _events_session_dict(
            db, str(row["id"]), row["task"], str(row["chat_id"] or ""),
            _row_to_extra_json(row),
        )


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

        # Sub-agent rows (those carrying a ``subagent`` marker in
        # ``extra``) are internal implementation detail of the parent's
        # ``run_parallel`` tool call and must NEVER appear when the user
        # adjacent-scrolls between tasks in the chat webview — they
        # already render inside the parent task's panel.  Filter them
        # out at the SQL level so the LIMIT 1 lands on the next *parent*
        # row rather than a sub-agent that happens to sit between two
        # parent tasks chronologically.
        # Adjacency uses the total order ``(timestamp, rowid)`` — the same
        # rowid tiebreak as ``_load_latest_chat_events_by_chat_id``'s
        # ``ORDER BY timestamp DESC, rowid DESC`` — so rows that share a
        # timestamp value (concurrent inserts, imported databases)
        # remain mutually reachable instead of strict ``<`` / ``>``
        # timestamp comparison silently skipping them.
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
    # Detect an on-disk swap/removal of the database BEFORE consulting
    # the cache: the cache-hit path performs no ``_get_db()`` call, so
    # without this probe a chat_id cached against a deleted or replaced
    # database file would be served forever (``_maybe_reset_caches``
    # clears the cache when the path or file identity changes).
    current_path = _current_db_path()
    _maybe_reset_caches(current_path, _db_file_identity(current_path))
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


