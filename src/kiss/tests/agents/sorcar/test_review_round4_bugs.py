# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end reproducing tests for Phase-5 ROUND 4 review bugs.

Each test reproduces a CRITICAL or HIGH finding from
``tmp/review_*_r4.md`` against the post-round-4-fix code.  The tests
assert the FIXED behavior; running them on the pre-fix source raises
``AssertionError`` (or the underlying bug itself).
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
# persistence r4 C1 / C2 / H5:
#   ``_add_task`` must accept the flat ``{"parent_task_id": <uuid>}``
#   shape and the ``{"subagent": "<uuid>"}`` string-shorthand shape,
#   and must raise ``ValueError`` when both shapes are passed.
# ---------------------------------------------------------------------------

def test_add_task_accepts_flat_parent_task_id_shape(temp_db: Path) -> None:  # noqa: ARG001
    parent_id, _ = persistence._add_task("parent")
    sub_id, _ = persistence._add_task(
        "sub", extra={"parent_task_id": parent_id},
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (sub_id,),
    ).fetchone()
    assert row["parent_task_id"] == parent_id


def test_add_task_accepts_subagent_string_shorthand(temp_db: Path) -> None:  # noqa: ARG001
    parent_id, _ = persistence._add_task("parent")
    sub_id, _ = persistence._add_task(
        "sub", extra={"subagent": parent_id},
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (sub_id,),
    ).fetchone()
    assert row["parent_task_id"] == parent_id


def test_add_task_rejects_collision_of_parent_task_id_and_subagent(temp_db: Path) -> None:  # noqa: ARG001
    parent_id, _ = persistence._add_task("parent")
    with pytest.raises(ValueError, match=r"both 'parent_task_id' and 'subagent'"):
        persistence._add_task(
            "sub",
            extra={
                "parent_task_id": parent_id,
                "subagent": {"parent_task_id": parent_id},
            },
        )


# ---------------------------------------------------------------------------
# persistence r4 H3:
#   ``_resolve_task_id`` must not silently no-op when the caller
#   passes a stale int (or any non-UUID-hex) ``task_id`` — it must
#   fall back to ``_most_recent_task_id(db, task)`` so legacy
#   JSON-RPC clients can still resolve.
# ---------------------------------------------------------------------------

def test_resolve_task_id_falls_back_on_int_task_id(temp_db: Path) -> None:  # noqa: ARG001
    persistence._add_task("alpha")
    real_id, _ = persistence._add_task("beta")
    db = persistence._get_db()
    # Pretend a legacy JSON-RPC client passes an int task_id with the
    # right task name.  The function must fall back to the most-recent
    # task with that name rather than silently return ``None``.
    resolved = persistence._resolve_task_id(db, 42, "beta")  # type: ignore[arg-type]
    assert resolved == real_id


def test_resolve_task_id_falls_back_on_non_uuid_string(temp_db: Path) -> None:  # noqa: ARG001
    real_id, _ = persistence._add_task("gamma")
    db = persistence._get_db()
    resolved = persistence._resolve_task_id(db, "not-a-uuid", "gamma")
    assert resolved == real_id


def test_resolve_task_id_returns_real_id_on_valid_match(temp_db: Path) -> None:  # noqa: ARG001
    real_id, _ = persistence._add_task("delta")
    db = persistence._get_db()
    resolved = persistence._resolve_task_id(db, real_id, "delta")
    assert resolved == real_id


# ---------------------------------------------------------------------------
# cli_printer r4 H5:
#   ``broadcast`` must forward events with empty ``taskId`` (global
#   system events like ``new_tab`` / ``tasks_updated``) to the daemon
#   so subscribed webviews still receive the broadcast.
# ---------------------------------------------------------------------------

def test_cli_printer_forwards_empty_taskid_global_event_to_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
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
    # Global system event — taskId deliberately empty.
    printer.broadcast({
        "type": "new_tab",
        "task_id": "a" * 32,
        "parent_tab_id": "",
        "taskId": "",
    })
    assert any(e.get("type") == "new_tab" for e in captured), (
        "expected new_tab envelope to reach daemon despite empty taskId"
    )


# ---------------------------------------------------------------------------
# r4-sorcar-H3: paired ``self._last_task_id`` clear + register guarded
#   by per-instance lock.  Verify the lock exists and is exercised.
# ---------------------------------------------------------------------------

def test_chat_sorcar_agent_has_per_instance_task_id_lock() -> None:
    import threading

    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    agent = ChatSorcarAgent(name="t")
    assert hasattr(agent, "_task_id_lock"), (
        "ChatSorcarAgent must expose ``_task_id_lock`` per r4-sorcar-H3"
    )
    # A reentrant lock is required so a recursive ``run()`` invocation
    # (e.g. resume during shutdown) does not self-deadlock.
    assert isinstance(agent._task_id_lock, type(threading.RLock())), (
        "``_task_id_lock`` must be a re-entrant ``threading.RLock``"
    )
    # Acquireable -> releaseable round-trip.
    with agent._task_id_lock:
        with agent._task_id_lock:
            pass


# ---------------------------------------------------------------------------
# r4-vscode-H1: ``_replay_session`` rebound_task_id extraction must
#   accept legacy int payloads (mirror r3-vscode-H2's defence applied
#   to ``parent_task_id``).
# ---------------------------------------------------------------------------

def test_vscode_server_source_accepts_int_rebound_task_id() -> None:
    from pathlib import Path

    src = Path("src/kiss/agents/vscode/server.py").read_text()
    # The fixed code path must coerce int -> str rather than dropping
    # the rebound id to None.
    assert "isinstance(_raw_rebound_tid, int)" in src, (
        "r4-vscode-H1: ``rebound_task_id`` extraction must accept int"
    )
    assert "str(_raw_rebound_tid)" in src, (
        "r4-vscode-H1: int rebound task_id must be stringified"
    )


# ---------------------------------------------------------------------------
# r4-vscode-H2: ``_get_history`` ``entry_id`` extraction must coerce
#   legacy int task_ids to str rather than dropping them.
# ---------------------------------------------------------------------------

def test_vscode_server_source_accepts_int_entry_id() -> None:
    from pathlib import Path

    src = Path("src/kiss/agents/vscode/server.py").read_text()
    assert "isinstance(_raw_eid, int)" in src, (
        "r4-vscode-H2: ``entry_id`` must accept int rows from legacy DBs"
    )
    assert "entry_id = str(_raw_eid)" in src, (
        "r4-vscode-H2: int ``entry_id`` must be stringified"
    )
