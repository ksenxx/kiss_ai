"""End-to-end reproducing tests for bugs flagged by the gpt-5.5 review.

Each test reproduces a CRITICAL or HIGH bug from
``tmp/review_persistence.md``, ``tmp/review_vscode.md`` or
``tmp/review_sorcar_other.md``.  After the fix is in place every test
should pass; before the fix, each test failed deterministically.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import persistence

# ---------------------------------------------------------------------------
# Fixture: redirect persistence to a temp DB and reset thread-local caches.
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path]:
    """Point persistence at a temp DB and reset per-thread connection cache."""
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    yield db_path
    persistence._close_db()


def _make_legacy_db(path: Path) -> None:
    """Create an old-schema (INTEGER id + extra JSON) DB at *path*."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.executescript(
        """
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
        """
    )
    conn.close()


# ---------------------------------------------------------------------------
# Persist#1 — migration is atomic (BEGIN IMMEDIATE / ROLLBACK)
# ---------------------------------------------------------------------------


def test_persist_bug1_migration_rolls_back_on_failure(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-migration must leave the legacy DB structurally intact.

    Inject an exception after the new tables are created but before the
    final rename; the legacy ``task_history`` (INTEGER id, ``extra``
    column) must still be reachable, not a half-converted hybrid.
    """
    _make_legacy_db(temp_db)
    # Seed one legacy row.
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "legacy task", "ok", json.dumps({"model": "gpt-5"})),
    )
    conn.close()

    # Patch ``json.loads`` (used inside the migration's per-row
    # parse) to raise a non-JSON exception — escapes the inner
    # try/except and aborts the migration, triggering ROLLBACK.
    original_loads = persistence.json.loads

    def _crash_loads(s: str, *a: Any, **kw: Any) -> Any:
        if s == '{"model": "gpt-5"}':
            raise RuntimeError("simulated mid-migration crash")
        return original_loads(s, *a, **kw)

    monkeypatch.setattr(persistence.json, "loads", _crash_loads)
    with pytest.raises(RuntimeError, match="simulated"):
        persistence._get_db()
    monkeypatch.undo()
    persistence._close_db()

    # Direct sqlite check: the legacy table must still exist with its
    # original INTEGER id and ``extra`` column.  The __new tables may
    # remain (idempotent CREATE will drop them on retry).
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    cols = {
        r[1]: (r[2] or "").upper()
        for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
    }
    conn.close()
    assert cols.get("id", "").upper() == "INTEGER"
    assert "extra" in cols


def test_persist_bug2_migration_is_idempotent_after_prior_crash(
    temp_db: Path,
) -> None:
    """Leftover ``task_history__new`` from a prior crash must not block retry.

    Simulates: a previous migration crashed AFTER creating the temp
    tables but BEFORE the rename.  The next ``_get_db()`` boot must
    drop the stale tables and re-attempt cleanly.
    """
    _make_legacy_db(temp_db)
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "legacy", "ok", "{}"),
    )
    # Drop a stale ``task_history__new`` to mimic a crashed prior
    # migration.  Use a minimal schema; the DROP IF EXISTS preamble
    # in ``_migrate_old_schema_if_needed`` doesn't care about its
    # shape.
    conn.execute(
        "CREATE TABLE task_history__new (id TEXT PRIMARY KEY)"
    )
    conn.execute(
        "CREATE TABLE events__new (id INTEGER PRIMARY KEY)"
    )
    conn.close()

    # Should not raise — must drop the stale tables and migrate.
    db = persistence._get_db()
    rows = db.execute(
        "SELECT id, task, result FROM task_history"
    ).fetchall()
    assert len(rows) == 1
    assert persistence.is_task_history_id(rows[0]["id"])
    assert rows[0]["task"] == "legacy"


def test_persist_bug3_migration_handles_missing_events_table(
    temp_db: Path,
) -> None:
    """A legacy DB with no ``events`` table must still migrate cleanly."""
    _make_legacy_db(temp_db)
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "task A", "ok", "{}"),
    )
    conn.execute("DROP TABLE events")
    conn.close()

    # Must not raise.
    db = persistence._get_db()
    rows = db.execute("SELECT id, task FROM task_history").fetchall()
    assert len(rows) == 1
    assert persistence.is_task_history_id(rows[0]["id"])
    # Events table must have been re-created in the new schema.
    cols = {
        r[1]: (r[2] or "").upper()
        for r in db.execute("PRAGMA table_info(events)").fetchall()
    }
    assert cols.get("task_id", "").upper() == "TEXT"


# ---------------------------------------------------------------------------
# Persist#11 — _add_task / _save_task_extra reject non-UUID parent_task_id
# ---------------------------------------------------------------------------


def test_persist_bug11_add_task_rejects_legacy_int_parent_task_id(
    temp_db: Path,
) -> None:
    """A legacy integer ``parent_task_id`` must NOT be written as ``"123"``.

    The new column is UUID-hex; a numeric string would never match any
    real id, silently breaking ``_subagent_child_ids`` lookups.
    """
    task_id, _ = persistence._add_task(
        "child task", "", extra={"subagent": {"parent_task_id": 123}}
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    # Reject — the column must be empty, not "123".
    assert row["parent_task_id"] == ""


def test_persist_bug11_add_task_accepts_valid_uuid_parent_task_id(
    temp_db: Path,
) -> None:
    """A real 32-char hex UUID parent must round-trip."""
    parent = uuid.uuid4().hex
    child_id, _ = persistence._add_task(
        "child", "", extra={"subagent": {"parent_task_id": parent}}
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["parent_task_id"] == parent


def test_persist_bug11_save_task_extra_rejects_bogus_parent(
    temp_db: Path,
) -> None:
    """``_save_task_extra`` must also validate the parent id shape."""
    task_id, _ = persistence._add_task("task", "")
    persistence._save_task_extra(
        {"subagent": {"parent_task_id": "not-a-uuid"}},
        task_id=task_id,
        task=None,
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["parent_task_id"] == ""


# ---------------------------------------------------------------------------
# Persist#14 — _history_row_to_dict exposes typed columns
# ---------------------------------------------------------------------------


def test_persist_bug14_history_dict_exposes_typed_columns(
    temp_db: Path,
) -> None:
    """Consumers must see ``model``/``cost``/``tokens`` as top-level keys."""
    task_id, _ = persistence._add_task(
        "task A", "",
        extra={"model": "gpt-5", "cost": 1.25, "tokens": 4242},
    )
    entries = persistence._load_history()
    matched = [e for e in entries if e["id"] == task_id]
    assert len(matched) == 1
    e = matched[0]
    assert e["model"] == "gpt-5"
    assert e["cost"] == pytest.approx(1.25)
    assert e["tokens"] == 4242
    # Legacy ``extra`` JSON synthesis must still be present.
    assert "extra" in e
    extra_parsed = json.loads(cast(str, e["extra"]))
    assert extra_parsed["model"] == "gpt-5"


# ---------------------------------------------------------------------------
# VS#1 — server.py _replay_session sub-agent parent linking accepts str
# ---------------------------------------------------------------------------


def test_vs_bug1_replay_uses_str_parent_task_id() -> None:
    """The ``parent_tid`` extraction in ``_replay_session`` must accept str.

    Before the fix: ``isinstance(parent_tid_raw, int)`` — always False
    for the str UUID round-trip.  The coercion is now centralised in
    ``_coerce_id``: str UUIDs pass through unchanged (the primary,
    post-refactor canonical contract) and the r3-vscode-H2 int
    fallback stringifies legacy ids rather than replacing the str path.
    """
    from kiss.server.server import _coerce_id

    src = Path(
        "src/kiss/server/server.py"
    ).read_text()
    assert (
        'parent_tid = _coerce_id(subagent_info.get("parent_task_id"))' in src
    )
    assert _coerce_id("deadbeef" * 4) == "deadbeef" * 4
    assert _coerce_id(5) == "5"
    assert _coerce_id("") is None
    assert _coerce_id([5]) is None


# ---------------------------------------------------------------------------
# VS#2 — web_server.py shutdown safety net accepts UUID-str task ids
# ---------------------------------------------------------------------------


def test_vs_bug2_shutdown_helper_accepts_uuid_strings() -> None:
    """``active_task_history_ids`` must be a ``set[str]`` — no ``int(...)``.

    Reproduces by reading the source and asserting the
    ``set[int]``/``int(th_id)`` patterns are gone.  A direct functional
    repro would require spinning up a live `_RunningAgentState` /
    websocket / shutdown sequence — covered by the existing E2E test
    suite's daemon-shutdown tests.
    """
    src = Path(
        "src/kiss/server/web_server.py"
    ).read_text()
    assert "active_task_history_ids: set[int]" not in src
    assert "active_task_history_ids: set[str]" in src
    assert "active_task_history_ids.add(int(th_id))" not in src
    assert "active_task_history_ids.add(str(th_id))" in src


# ---------------------------------------------------------------------------
# VS#3 — commands.py rejects non-string taskId payloads
# ---------------------------------------------------------------------------


def test_vs_bug3_commands_reject_non_string_taskid() -> None:
    """A non-string ``taskId`` payload must be dropped before SQL.

    The previous pattern ``str(raw_task_id) if raw_task_id else None``
    accepted dicts and lists and stringified them.  All four relevant
    handlers now validate through the shared ``_opt_str`` guard, which
    rejects every non-string payload.
    """
    from kiss.server.commands import _opt_str

    src = Path(
        "src/kiss/server/commands.py"
    ).read_text()
    occurrences = src.count('task_id = _opt_str(cmd.get("taskId"))')
    assert occurrences == 4, (
        f"expected 4 hardened taskId guards, found {occurrences}"
    )
    assert _opt_str({"a": 1}) is None
    assert _opt_str([1]) is None
    assert _opt_str(7) is None
    assert _opt_str("") is None
    assert _opt_str("tid") == "tid"


# ---------------------------------------------------------------------------
# VS#4 — web_server.py cliTaskStart/End reject non-string taskId
# ---------------------------------------------------------------------------


def test_vs_bug4_cli_task_envelopes_reject_non_string_taskid() -> None:
    """``cliTaskStart`` / ``cliTaskEnd`` must reject non-str payloads.

    Functional repro: both branches validate the ``taskId`` through the
    shared ``_validated_cli_task_id`` helper, which must return ``""``
    for missing, empty, or non-string payloads so the daemon never
    registers a bogus task in ``_cli_running_tasks``.
    """
    from kiss.server.web_server import RemoteAccessServer

    validate = RemoteAccessServer._validated_cli_task_id
    assert validate({"type": "cliTaskStart", "taskId": ["evil"]}) == ""
    assert validate({"type": "cliTaskEnd", "taskId": 123}) == ""
    assert validate({"type": "cliTaskStart", "taskId": ""}) == ""
    assert validate({"type": "cliTaskStart"}) == ""
    assert validate({"type": "cliTaskEnd", "taskId": "abc123"}) == "abc123"


# ---------------------------------------------------------------------------
# Sorcar#1 — cli_printer is_task_history_id contract & lowercase normalize
# ---------------------------------------------------------------------------


def test_sorcar_bug1_is_task_history_id_contract() -> None:
    """``is_task_history_id`` is the canonical id-shape predicate."""
    assert persistence.is_task_history_id(uuid.uuid4().hex)
    assert not persistence.is_task_history_id("")
    assert not persistence.is_task_history_id(None)
    assert not persistence.is_task_history_id(123)
    # Uppercase hex is NOT accepted (cli_printer lowercases first).
    assert not persistence.is_task_history_id(
        uuid.uuid4().hex.upper()
    )
    # Hyphenated UUID is NOT accepted.
    assert not persistence.is_task_history_id(str(uuid.uuid4()))


def test_sorcar_bug1_cli_printer_normalizes_case() -> None:
    """``CliPrinter._broadcast_event`` lowercases the task id key.

    Reading the source guarantees the new ``.lower()`` is in place and
    the heuristic char-by-char check has been removed.
    """
    src = Path(
        "src/kiss/agents/sorcar/cli_printer.py"
    ).read_text()
    # Heuristic gone.
    assert "0123456789abcdef" not in src
    # New contract in place.
    assert "is_task_history_id" in src
    assert ".lower()" in src


# ---------------------------------------------------------------------------
# Sorcar#3 — _run_tasks_parallel must not leak "task-None__sub_*" keys
# ---------------------------------------------------------------------------


def test_sorcar_bug3_run_tasks_parallel_guards_none_parent() -> None:
    """When ``self._last_task_id is None`` the sub-tab key cannot be
    ``"task-None__sub_*"`` — assert by reading the source guard.
    """
    src = Path(
        "src/kiss/agents/sorcar/chat_sorcar_agent.py"
    ).read_text()
    # Session 12 split the value into ``persisted_parent_task_id``
    # (real or "") and ``routing_parent_key`` (real OR a fresh uuid
    # hex synthesised when no real parent exists).  The guard runs
    # BEFORE the ``sub_tab_id`` f-string is built so the literal
    # ``task-None__sub_*`` cannot leak.
    guard_idx = src.find(
        "routing_parent_key = uuid.uuid4().hex"
    )
    assert guard_idx != -1, "Sorcar#3 guard missing"
    sub_tab_idx = src.find(
        'sub_tab_id = f"task-{parent_task_id}__sub_{idx}"'
    )
    assert sub_tab_idx != -1
    assert guard_idx < sub_tab_idx, (
        "guard must run BEFORE sub_tab_id is constructed"
    )
