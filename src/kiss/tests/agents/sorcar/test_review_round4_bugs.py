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


def test_resolve_task_id_falls_back_on_int_task_id(temp_db: Path) -> None:  # noqa: ARG001
    persistence._add_task("alpha")
    real_id, _ = persistence._add_task("beta")
    db = persistence._get_db()
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


def test_chat_sorcar_agent_has_per_instance_task_id_lock() -> None:
    import threading

    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    agent = ChatSorcarAgent(name="t")
    assert hasattr(agent, "_task_id_lock"), (
        "ChatSorcarAgent must expose ``_task_id_lock`` per r4-sorcar-H3"
    )
    assert isinstance(agent._task_id_lock, type(threading.RLock())), (
        "``_task_id_lock`` must be a re-entrant ``threading.RLock``"
    )
    with agent._task_id_lock:
        with agent._task_id_lock:
            pass
