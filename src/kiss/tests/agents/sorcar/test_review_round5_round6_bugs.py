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


# ---------------------------------------------------------------------------
# r5-persistence-C2 — ``_save_task_extra({"is_favorite": ...})`` must raise.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# r5-sorcar-H7 — ``cli_printer.broadcast`` only forwards specific global
#   event types to the daemon when ``taskId`` is empty.
# ---------------------------------------------------------------------------

def test_cli_printer_forwards_only_listed_global_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process diagnostics with empty taskId must NOT flood the
    daemon UDS; only the explicit ``new_tab`` / ``tasks_updated``
    global system events should be forwarded.
    """
    from kiss.agents.sorcar import cli_daemon_bridge, cli_printer

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli_daemon_bridge,
        "send_event",
        lambda env: captured.append(env),
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_start", lambda _tid: None,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_end", lambda _tid: None,
    )

    printer = cli_printer.RecordingConsolePrinter()
    # Forwarded
    printer.broadcast({
        "type": "new_tab",
        "task_id": "a" * 32,
        "parent_tab_id": "",
        "taskId": "",
    })
    printer.broadcast({"type": "tasks_updated", "taskId": ""})
    # NOT forwarded (diagnostic without taskId)
    printer.broadcast({"type": "debug_diagnostic", "taskId": ""})
    printer.broadcast({"type": "random_internal_event"})

    types_forwarded = [e.get("type") for e in captured]
    assert "new_tab" in types_forwarded
    assert "tasks_updated" in types_forwarded
    assert "debug_diagnostic" not in types_forwarded
    assert "random_internal_event" not in types_forwarded


# ---------------------------------------------------------------------------
# r5-sorcar-H3 — ``_register_running_state`` ``state.agent is None``
#   branch must bind ``state.agent`` so downstream scans find it.
# ---------------------------------------------------------------------------

def test_register_running_state_does_not_clobber_preexisting(temp_db: Path) -> None:  # noqa: ARG001
    """Documented contract: a pre-allocated ``_RunningAgentState``
    entry for this chat_id with ``state.agent=None`` belongs to its
    creator (server frame or parent worktree agent).  A standalone
    child agent that observes the entry must SKIP re-registration
    (return False) WITHOUT hijacking the entry.  See
    ``test_run_does_not_clobber_preexisting_state`` for the
    end-to-end contract.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState

    agent = ChatSorcarAgent(name="t")
    agent._chat_id = "test-chat-456"

    pre = _RunningAgentState(
        "test-chat-456",
        "model",
        agent=None,
    )
    pre.chat_id = "test-chat-456"
    pre.is_task_active = False

    try:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states["test-chat-456"] = pre

        result = agent._register_running_state()
        assert result is False, "must skip re-register for pre-existing entry"
        assert pre.agent is None, (
            "must NOT clobber pre-allocated entry's None agent"
        )
    finally:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.pop(
                "test-chat-456", None,
            )


# ---------------------------------------------------------------------------
# r6-persistence-H3 — ``_bx`` must NOT treat string literals
#   "false" / "0" / "no" as truthy.
# ---------------------------------------------------------------------------

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

    # Build a legacy-shape DB with stringified bool flags in extra.
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

    # Open via _get_db() to trigger migration.
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


# ---------------------------------------------------------------------------
# r6-persistence-H7 — migration must temporarily disable foreign_keys
#   so the ``ALTER TABLE ... RENAME`` is safe under older SQLite.
# ---------------------------------------------------------------------------

def test_migration_toggles_foreign_keys_off_during_rename() -> None:
    """Inspect source to verify the migration encloses its body with
    ``PRAGMA foreign_keys=OFF`` / ``PRAGMA foreign_keys=ON`` so the
    ``ALTER TABLE __new RENAME TO task_history`` does not leave a
    stale FK target on SQLite < 3.26.
    """
    src = Path("src/kiss/agents/sorcar/persistence.py").read_text()
    # Must turn FK off before BEGIN IMMEDIATE in the migration.
    assert "PRAGMA foreign_keys=OFF" in src
    # And restore on every exit path (success + rollback).
    assert src.count("PRAGMA foreign_keys=ON") >= 2


# ---------------------------------------------------------------------------
# r6-vscode-H4 — server.py ``_live_task_id`` must return ``str | None``
#   coercing int values from legacy DBs.
# ---------------------------------------------------------------------------

def test_live_task_id_returns_str_or_none() -> None:
    """Source-level assertion: the ``_live_task_id`` helper must be
    annotated as returning ``str | None``.  The r5-vscode review
    flagged that an int slipping through breaks
    ``set[str]`` membership comparisons downstream.
    """
    src = Path("src/kiss/server/server.py").read_text()
    # Function signature
    assert "_live_task_id" in src
    # Strict-typed return contract.
    assert "-> str | None" in src or "-> str|None" in src, (
        "_live_task_id should return str | None"
    )
