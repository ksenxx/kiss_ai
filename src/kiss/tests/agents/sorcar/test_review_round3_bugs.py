# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end reproducing tests for Phase-5 ROUND 3 review bugs.

Each test reproduces a CRITICAL or HIGH finding from
``tmp/review_*_r3.md`` against the post-round-3-fix code.  The tests
assert the FIXED behavior; running them on the pre-fix source raises
``AssertionError`` (or, in a few cases, the underlying bug itself).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence


@pytest.fixture
def temp_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path]:
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    yield db_path
    persistence._close_db()


def test_save_task_extra_top_level_parent_task_id_garbage_does_not_clear(
    temp_db: Path,
) -> None:
    """A non-UUID top-level ``parent_task_id`` value must not overwrite."""
    from kiss.agents.sorcar import persistence as P

    parent_real = uuid.uuid4().hex
    sub_id, _ = P._add_task(
        "sub", extra={"subagent": {"parent_task_id": parent_real}},
    )
    P._save_task_extra({"parent_task_id": "not-a-uuid"}, task_id=sub_id)
    db = P._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?", (sub_id,)
    ).fetchone()
    assert row["parent_task_id"] == parent_real



def test_save_task_extra_rejects_both_parent_and_subagent_keys(
    temp_db: Path,
) -> None:
    from kiss.agents.sorcar import persistence as P

    tid, _ = P._add_task("x")
    parent1 = uuid.uuid4().hex
    parent2 = uuid.uuid4().hex
    with pytest.raises(ValueError, match="parent_task_id.*subagent"):
        P._save_task_extra(
            {
                "parent_task_id": parent1,
                "subagent": {"parent_task_id": parent2},
            },
            task_id=tid,
        )



def test_migration_handles_non_finite_extra_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy DB with ``cost=NaN`` (encoded as 'NaN') must migrate cleanly."""
    from kiss.agents.sorcar import persistence as P

    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(P, "_DB_PATH", db_path)
    P._close_db()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE task_history (id INTEGER PRIMARY KEY, "
        "timestamp REAL NOT NULL, task TEXT NOT NULL, "
        "has_events INTEGER DEFAULT 0, result TEXT DEFAULT '', "
        "chat_id CHAR(32) DEFAULT '', extra TEXT DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "task_id INTEGER NOT NULL, seq INTEGER NOT NULL, "
        "event_json TEXT NOT NULL, timestamp REAL NOT NULL)"
    )
    bad_extra = '{"cost": NaN, "tokens": 7}'
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, extra) "
        "VALUES (1, 1.0, 'old', ?)", (bad_extra,)
    )
    conn.commit()
    conn.close()

    db = P._get_db()
    rows = db.execute(
        "SELECT id, cost, tokens FROM task_history"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["cost"] == 0.0
    assert rows[0]["tokens"] == 7



def test_save_task_extra_does_not_clear_favorite_via_is_favorite_payload(
    temp_db: Path,
) -> None:
    """r3-H1 + r5-persistence-C2: writing ``is_favorite`` via
    ``_save_task_extra`` is forbidden.  Previously silently dropped;
    now raises ``ValueError`` so the bug surfaces at the caller.
    """
    from kiss.agents.sorcar import persistence as P

    tid, _ = P._add_task("x")
    assert P._set_task_favorite(tid, True) is True
    with pytest.raises(ValueError, match="_set_task_favorite"):
        P._save_task_extra(
            {"tokens": 5, "is_favorite": False}, task_id=tid,
        )
    db = P._get_db()
    row = db.execute(
        "SELECT is_favorite FROM task_history WHERE id = ?", (tid,)
    ).fetchone()
    assert row["is_favorite"] == 1



def test_row_to_extra_json_emits_all_typed_columns(
    temp_db: Path,
) -> None:
    from kiss.agents.sorcar import persistence as P

    tid, _ = P._add_task("x")
    entries = P._load_history(limit=1)
    payload = json.loads(str(entries[0]["extra"]))
    for k in (
        "model", "work_dir", "version",
        "auto_commit_mode", "tokens", "cost", "steps",
        "is_parallel", "is_worktree",
        "startTs", "endTs", "is_favorite",
    ):
        assert k in payload, f"missing key {k!r}"



def test_recover_orphaned_tasks_uses_placeholders() -> None:
    """Structural check: SQL is built with ``?`` placeholders, not f-strings."""
    src = Path(
        "src/kiss/agents/sorcar/persistence.py"
    ).read_text()
    assert "'\" + str(t).replace(\"'\", \"''\") + \"'\"" not in src
    assert "AND id NOT IN ({placeholders})" in src


def test_shutdown_persist_in_flight_works_with_uuid_str(
    temp_db: Path,
) -> None:
    from kiss.agents.sorcar import persistence as P

    tid, _ = P._add_task("running")
    db = P._get_db()
    db.execute(
        "UPDATE task_history SET result = ? WHERE id = ?",
        ("Agent Failed Abruptly", tid),
    )
    db.commit()
    n = P._shutdown_persist_in_flight_results({tid})
    assert n == 1
    row = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (tid,)
    ).fetchone()
    assert "interrupted" in row["result"].lower()



def test_migration_reprobes_inside_transaction() -> None:
    """Structural: the migration body contains a second ``PRAGMA table_info`` call after BEGIN IMMEDIATE."""
    src = Path(
        "src/kiss/agents/sorcar/persistence.py"
    ).read_text()
    begin_idx = src.find("BEGIN IMMEDIATE")
    assert begin_idx != -1
    after_begin = src[begin_idx:]
    assert "PRAGMA table_info(task_history)" in after_begin, (
        "Migration must re-probe table_info inside the write transaction"
    )



def test_migration_drop_table_inside_transaction() -> None:
    src = Path(
        "src/kiss/agents/sorcar/persistence.py"
    ).read_text()
    begin_idx = src.find("BEGIN IMMEDIATE")
    drop_idx = src.find("DROP TABLE IF EXISTS task_history__new")
    assert begin_idx != -1 and drop_idx != -1
    assert drop_idx > begin_idx, (
        "DROP TABLE preamble must occur AFTER BEGIN IMMEDIATE"
    )



def test_cli_printer_lowercases_event_taskid_before_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.server.json_printer import JsonPrinter
    from kiss.ui.cli import cli_daemon_bridge, cli_printer

    monkeypatch.setattr(
        cli_daemon_bridge, "send_event", lambda ev: None,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_start", lambda _t: None,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_end", lambda _t: None,
    )
    captured: list[dict[str, Any]] = []

    def _spy_super_broadcast(self: Any, event: dict[str, Any]) -> None:
        captured.append(dict(event))

    monkeypatch.setattr(
        JsonPrinter, "broadcast", _spy_super_broadcast,
    )
    upper = uuid.uuid4().hex.upper()
    printer = cli_printer.RecordingConsolePrinter()
    printer.broadcast({"type": "step", "taskId": upper})
    assert captured, "super().broadcast must run"
    assert captured[0]["taskId"] == upper.lower()



def test_cli_printer_releases_lock_before_daemon_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.ui.cli import cli_daemon_bridge, cli_printer

    printer = cli_printer.RecordingConsolePrinter()

    held_during_send: list[bool] = []

    def _start_spy(_t: str) -> None:
        acquired = printer._cli_task_lock.acquire(blocking=False)
        held_during_send.append(not acquired)
        if acquired:
            printer._cli_task_lock.release()

    monkeypatch.setattr(
        cli_daemon_bridge, "send_event", lambda ev: None,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_start", _start_spy,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_end",
        lambda _t: _start_spy(_t),
    )

    tid = uuid.uuid4().hex
    printer._inject_task_id = lambda event: {**event, "taskId": tid}  # type: ignore[method-assign]
    printer.broadcast({"type": "step"})
    printer.broadcast({"type": "result"})
    assert held_during_send == [False, False], (
        f"Daemon-send must occur OUTSIDE _cli_task_lock; held={held_during_send}"
    )



def test_cli_printer_imports_is_task_history_id_at_module_level() -> None:
    from kiss.ui.cli import cli_printer

    src = Path(str(cli_printer.__file__)).read_text()
    assert (
        "from kiss.agents.sorcar.persistence import is_task_history_id" in src
    )
    broadcast_idx = src.find("def broadcast(")
    assert broadcast_idx != -1
    after_broadcast = src[broadcast_idx:]
    next_def = after_broadcast.find("\n    def ", 1)
    if next_def == -1:
        next_def = len(after_broadcast)
    method_body = after_broadcast[:next_def]
    assert "from kiss.agents.sorcar.persistence import" not in method_body



def test_cli_printer_running_task_ids_docstring_updated() -> None:
    from kiss.ui.cli import cli_printer

    src = Path(str(cli_printer.__file__)).read_text()
    assert "Set of int" not in src
    assert "32-char lowercase-hex" in src



def test_task_runner_rejects_non_string_task_id() -> None:
    """Non-string ``taskId`` payloads are rejected by the shared guard.

    The guard was centralised into :func:`_client_task_id_of` (bughunt
    round 9); exercise its behaviour directly instead of asserting on
    source-code text.
    """
    from kiss.server.task_runner import _client_task_id_of

    assert _client_task_id_of({"taskId": "abc123"}) == "abc123"
    assert _client_task_id_of({}) == ""
    for bad in ([1], {"x": 1}, True, 7, 3.5, None):
        assert _client_task_id_of({"taskId": bad}) == ""



def test_server_accepts_legacy_int_parent_task_id() -> None:
    from kiss.server.server import _coerce_id

    src = Path(
        "src/kiss/server/server.py"
    ).read_text()
    assert (
        'parent_tid = _coerce_id(subagent_info.get("parent_task_id"))' in src
    )
    assert 'pid = _coerce_id(sub.get("parent_task_id"))' in src
    assert _coerce_id("a" * 32) == "a" * 32
    assert _coerce_id(99) == "99"
