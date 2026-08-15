# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: E501, N812
"""End-to-end reproducing tests for Phase-5 ROUND 5 and ROUND 6 review bugs.

Each test reproduces a CRITICAL or HIGH finding from
``tmp/review_*_r5.md`` and ``tmp/review_*_r6.md`` against the
post-fix code.  Tests assert the FIXED behavior; running them
against the pre-fix source fails.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

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


def test_save_task_extra_raises_on_is_favorite_payload(temp_db: Path) -> None:  # noqa: ARG001
    """Caller must not be allowed to flip ``is_favorite`` via
    ``_save_task_extra`` — that flag is owned by
    ``_set_task_favorite``.  Silently dropping the key would leave
    the caller convinced the flag was set when it wasn't.
    """
    tid, _ = persistence._add_task("alpha")
    with pytest.raises(ValueError, match="_set_task_favorite"):
        persistence._save_task_extra(
            {"is_favorite": True, "tokens": 5}, task_id=tid,
        )


def test_bx_handles_falsy_string_literals_during_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy JSON-extra payloads sometimes encode boolean flags as
    string literals.  ``bool("false") == True`` in vanilla Python
    would silently flip every ``is_parallel`` / ``is_worktree`` /
    ``auto_commit_mode`` flag during migration.  Verify the fixed
    coercion handles all common false-y string forms.
    """
    import json
    import sqlite3

    legacy_db = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            "CREATE TABLE task_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp REAL NOT NULL, task TEXT NOT NULL, "
            "has_events INTEGER DEFAULT 0, result TEXT DEFAULT '', "
            "chat_id CHAR(32) DEFAULT '', extra TEXT DEFAULT ''"
            ")"
        )
        conn.execute(
            "CREATE TABLE events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id INTEGER NOT NULL, seq INTEGER NOT NULL, "
            "event_json TEXT NOT NULL, timestamp REAL NOT NULL"
            ")"
        )
        for label, flags in [
            ("falsy_string_false", {"is_parallel": "false", "is_worktree": "0"}),
            ("falsy_string_no", {"is_parallel": "no", "is_worktree": ""}),
            ("truthy_string_true", {"is_parallel": "true", "is_worktree": "1"}),
        ]:
            conn.execute(
                "INSERT INTO task_history "
                "(timestamp, task, has_events, result, chat_id, extra) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1.0, label, 0, "", "x" * 32, json.dumps(flags)),
            )

    import kiss.agents.sorcar.persistence as P

    P._close_db()
    monkeypatch.setattr(P, "_DB_PATH", legacy_db)
    db = P._get_db()
    try:
        rows = list(db.execute(
            "SELECT task, is_parallel, is_worktree FROM task_history",
        ))
        result = {r["task"]: (r["is_parallel"], r["is_worktree"]) for r in rows}
        assert result["falsy_string_false"] == (0, 0), (
            f"'false'/'0' must coerce to 0, got {result['falsy_string_false']}"
        )
        assert result["falsy_string_no"] == (0, 0), (
            f"'no'/'' must coerce to 0, got {result['falsy_string_no']}"
        )
        assert result["truthy_string_true"] == (1, 1), (
            f"'true'/'1' must coerce to 1, got {result['truthy_string_true']}"
        )
    finally:
        P._close_db()


def test_migration_toggles_foreign_keys_off_during_rename() -> None:
    """Inspect source to verify the migration encloses its body with
    ``PRAGMA foreign_keys=OFF`` / ``PRAGMA foreign_keys=ON`` so the
    ``ALTER TABLE __new RENAME TO task_history`` does not leave a
    stale FK target on SQLite < 3.26.
    """
    src = Path("src/kiss/agents/sorcar/persistence.py").read_text()
    assert "PRAGMA foreign_keys=OFF" in src
    assert src.count("PRAGMA foreign_keys=ON") >= 2
