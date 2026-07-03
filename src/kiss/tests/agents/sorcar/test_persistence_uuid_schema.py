# Author: Koushik Sen (ksen@berkeley.edu)
"""End-to-end tests for the UUID/flat-column task_history schema."""

# ruff: noqa: N806, N812
from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
import time
import uuid

import pytest


@contextlib.contextmanager
def _fresh_persistence_module(tmp_path, db_path=None):
    """Yield a freshly re-imported persistence module bound to *tmp_path*.

    On exit, the freshly-imported module is removed from ``sys.modules``
    and the ORIGINAL module object (with its original ``_DB_PATH``) is
    restored, so later tests in the same process that lazily import
    ``kiss.agents.sorcar.persistence`` see the real module again instead
    of a leftover copy bound to this test's temp database.
    """
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("kiss.agents.sorcar.persistence")
    }
    for name in saved:
        del sys.modules[name]
    from kiss.agents.sorcar import persistence as P
    P._KISS_DIR = tmp_path
    P._DB_PATH = db_path if db_path is not None else tmp_path / "sorcar.db"
    P._close_db()
    try:
        yield P
    finally:
        P._close_db()
        for name in list(sys.modules):
            if name.startswith("kiss.agents.sorcar.persistence"):
                del sys.modules[name]
        sys.modules.update(saved)
        original = saved.get("kiss.agents.sorcar.persistence")
        if original is not None:
            import kiss.agents.sorcar as _sorcar_pkg
            setattr(_sorcar_pkg, "persistence", original)  # noqa: B010


@pytest.fixture
def fresh_kiss_db(tmp_path, monkeypatch):
    """Provide a freshly-initialized persistence module bound to *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    with _fresh_persistence_module(tmp_path) as P:
        yield P


def test_add_task_returns_uuid_string_id(fresh_kiss_db):
    P = fresh_kiss_db
    tid, cid = P._add_task("hello world")
    assert isinstance(tid, str)
    assert len(tid) == 32
    uuid.UUID(tid)  # raises if invalid
    assert isinstance(cid, str)
    assert len(cid) == 32


def test_add_task_writes_extra_to_columns(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _cid = P._add_task(
        "t",
        extra={
            "model": "gpt-5", "work_dir": "/x", "version": "v1",
            "tokens": 100, "cost": 0.5, "steps": 3,
            "is_parallel": True, "is_worktree": True,
            "auto_commit_mode": "auto",
            "startTs": 1700000000000, "endTs": 1700000001000,
        },
    )
    db = P._get_db()
    row = db.execute(
        "SELECT * FROM task_history WHERE id = ?", (tid,)
    ).fetchone()
    assert row["model"] == "gpt-5"
    assert row["work_dir"] == "/x"
    assert row["version"] == "v1"
    assert row["tokens"] == 100
    assert row["cost"] == 0.5
    assert row["steps"] == 3
    assert row["is_parallel"] == 1
    assert row["is_worktree"] == 1
    assert row["auto_commit_mode"] == 1
    assert row["start_ts"] == 1700000000000
    assert row["end_ts"] == 1700000001000
    assert row["is_favorite"] == 0
    assert row["parent_task_id"] == ""


def test_save_task_extra_updates_columns_preserving_is_favorite(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _ = P._add_task("t")
    assert P._set_task_favorite(tid, True)
    P._save_task_extra({"tokens": 50, "cost": 0.1}, task_id=tid)
    db = P._get_db()
    row = db.execute(
        "SELECT * FROM task_history WHERE id = ?", (tid,)
    ).fetchone()
    assert row["tokens"] == 50
    assert row["cost"] == 0.1
    assert row["is_favorite"] == 1


def test_set_task_favorite_toggles_column(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _ = P._add_task("t")
    assert P._set_task_favorite(tid, True) is True
    db = P._get_db()
    fav = db.execute(
        "SELECT is_favorite FROM task_history WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert fav == 1
    P._set_task_favorite(tid, False)
    fav = db.execute(
        "SELECT is_favorite FROM task_history WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert fav == 0
    assert P._set_task_favorite("does-not-exist", True) is False


def test_subagent_child_ids_uses_parent_task_id_column(fresh_kiss_db):
    P = fresh_kiss_db
    parent, _ = P._add_task("parent")
    child, _ = P._add_task(
        "child", extra={"subagent": {"parent_task_id": parent}}
    )
    db = P._get_db()
    children = P._subagent_child_ids(db, parent)
    assert children == [child]


def test_delete_task_cascades_to_nested_subagents(fresh_kiss_db):
    P = fresh_kiss_db
    parent, _ = P._add_task("parent")
    child, _ = P._add_task(
        "c", extra={"subagent": {"parent_task_id": parent}}
    )
    P._add_task(
        "gc", extra={"subagent": {"parent_task_id": child}}
    )
    assert P._delete_task(parent) is True
    db = P._get_db()
    assert db.execute("SELECT COUNT(*) FROM task_history").fetchone()[0] == 0


def test_load_history_excludes_subagents(fresh_kiss_db):
    P = fresh_kiss_db
    parent, _ = P._add_task("parent")
    P._add_task("c", extra={"subagent": {"parent_task_id": parent}})
    entries = P._load_history()
    ids = [e["id"] for e in entries]
    assert parent in ids
    assert len(entries) == 1


def test_history_entry_synthesizes_extra_json_string(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _ = P._add_task("t", extra={"model": "m", "tokens": 5})
    entries = P._load_history()
    e = next(x for x in entries if x["id"] == tid)
    assert isinstance(e["extra"], str)
    obj = json.loads(e["extra"])
    assert obj["model"] == "m"
    assert obj["tokens"] == 5


def test_load_chat_events_by_task_id_returns_str_id(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _ = P._add_task("t")
    P._append_chat_event({"type": "x"}, task_id=tid)
    sess = P._load_chat_events_by_task_id(tid)
    assert sess is not None
    assert sess["task_id"] == tid
    assert isinstance(sess["task_id"], str)


def test_load_subagent_rows_by_parent_task_id(fresh_kiss_db):
    P = fresh_kiss_db
    parent, _ = P._add_task("parent")
    c1, _ = P._add_task(
        "c1", extra={"subagent": {"parent_task_id": parent}}
    )
    c2, _ = P._add_task(
        "c2", extra={"subagent": {"parent_task_id": parent}}
    )
    rows = P._load_subagent_rows_by_parent_task_id(parent)
    ids = [r["task_id"] for r in rows]
    assert c1 in ids
    assert c2 in ids
    assert all(isinstance(r["task_id"], str) for r in rows)


def test_recover_orphaned_tasks_string_ids(fresh_kiss_db):
    P = fresh_kiss_db
    t1, _ = P._add_task("t1")
    t2, _ = P._add_task("t2")
    # t1 active, t2 should be recovered
    n = P._recover_orphaned_tasks({t1})
    assert n == 1
    db = P._get_db()
    r1 = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (t1,)
    ).fetchone()[0]
    r2 = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (t2,)
    ).fetchone()[0]
    assert r1 == "Agent Failed Abruptly"
    assert "process killed" in r2


def test_shutdown_persist_in_flight_results_string_ids(fresh_kiss_db):
    P = fresh_kiss_db
    t1, _ = P._add_task("t1")
    t2, _ = P._add_task("t2")
    n = P._shutdown_persist_in_flight_results({t1, t2})
    assert n == 2
    db = P._get_db()
    for tid in (t1, t2):
        r = db.execute(
            "SELECT result FROM task_history WHERE id = ?", (tid,)
        ).fetchone()[0]
        assert "shutdown" in r or "restart" in r


def test_migration_from_old_schema(tmp_path, monkeypatch):
    """Build an old-schema DB, then connect and verify migration."""
    db_path = tmp_path / "sorcar.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id CHAR(32) DEFAULT '',
            extra TEXT DEFAULT ''
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES task_history(id),
            seq INTEGER NOT NULL,
            event_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE TABLE model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT UNIQUE, count INTEGER, is_last INTEGER
        );
        CREATE TABLE file_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE, count INTEGER, last_used REAL
        );
        CREATE TABLE frequent_tasks (
            task TEXT PRIMARY KEY, count INTEGER, timestamp REAL
        );
    """)
    extra_parent = json.dumps({
        "model": "gpt", "work_dir": "/p", "version": "v1",
        "tokens": 10, "cost": 1.5, "steps": 2,
        "is_parallel": True, "is_worktree": False,
        "auto_commit_mode": "auto",
        "startTs": 100, "endTs": 200, "is_favorite": True,
    })
    extra_child = json.dumps({
        "subagent": {"parent_task_id": 1}, "model": "claude"
    })
    conn.execute(
        "INSERT INTO task_history (timestamp, task, has_events, result, "
        "chat_id, extra) VALUES (?,?,?,?,?,?)",
        (time.time(), "parent task", 1, "ok", "chat1", extra_parent),
    )
    conn.execute(
        "INSERT INTO task_history (timestamp, task, has_events, result, "
        "chat_id, extra) VALUES (?,?,?,?,?,?)",
        (time.time(), "child task", 0, "", "chat1", extra_child),
    )
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) "
        "VALUES (?,?,?,?)",
        (1, 0, json.dumps({"type": "start"}), time.time()),
    )
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) "
        "VALUES (?,?,?,?)",
        (2, 0, json.dumps({"type": "step"}), time.time()),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    with _fresh_persistence_module(tmp_path, db_path) as P:
        db = P._get_db()  # triggers migration
        cols = {
            r[1]: r[2].upper()
            for r in db.execute("PRAGMA table_info(task_history)").fetchall()
        }
        assert cols["id"] == "TEXT"
        assert "model" in cols
        assert "parent_task_id" in cols
        assert "extra" not in cols
        rows = db.execute(
            "SELECT * FROM task_history ORDER BY task"
        ).fetchall()
        assert len(rows) == 2
        child_row = next(r for r in rows if r["task"] == "child task")
        parent_row = next(r for r in rows if r["task"] == "parent task")
        assert parent_row["model"] == "gpt"
        assert parent_row["tokens"] == 10
        assert parent_row["cost"] == 1.5
        assert parent_row["is_favorite"] == 1
        assert parent_row["is_parallel"] == 1
        assert parent_row["start_ts"] == 100
        assert parent_row["end_ts"] == 200
        assert child_row["parent_task_id"] == parent_row["id"]
        assert child_row["model"] == "claude"
        ev = db.execute("SELECT * FROM events ORDER BY id").fetchall()
        assert len(ev) == 2
        assert ev[0]["task_id"] == parent_row["id"]
        assert ev[1]["task_id"] == child_row["id"]


def test_migration_idempotent_on_new_schema(fresh_kiss_db):
    P = fresh_kiss_db
    tid, _ = P._add_task("t")
    db = P._get_db()
    assert P._migrate_old_schema_if_needed(db) is False
    row = db.execute(
        "SELECT id FROM task_history WHERE id = ?", (tid,)
    ).fetchone()
    assert row is not None


def test_migration_skips_empty_db(tmp_path, monkeypatch):
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    with _fresh_persistence_module(tmp_path) as P:
        db = P._get_db()
        cols = {
            r[1]: r[2].upper()
            for r in db.execute("PRAGMA table_info(task_history)").fetchall()
        }
        assert cols["id"] == "TEXT"


def test_resume_after_migration_uses_new_uuid_ids(tmp_path, monkeypatch):
    """End-to-end: after migration, queries that originally took int ids
    must accept str UUIDs and round-trip correctly."""
    db_path = tmp_path / "sorcar.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id CHAR(32) DEFAULT '',
            extra TEXT DEFAULT ''
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            event_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, chat_id, extra) "
        "VALUES (?,?,?,?,?)",
        (time.time(), "legacy", "done", "chatX", json.dumps({"model": "m"})),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    with _fresh_persistence_module(tmp_path, db_path) as P:
        entries = P._load_history()
        assert len(entries) == 1
        new_id = entries[0]["id"]
        assert isinstance(new_id, str)
        assert len(new_id) == 32
        # _get_task_chat_id should accept the new string id
        assert P._get_task_chat_id(new_id) == "chatX"
