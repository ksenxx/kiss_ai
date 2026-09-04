# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (cross-boundary): the early history row must carry the auto-commit toggle.

``ChatSorcarAgent.run`` inserts the task's history row at start (so
the sidebar can show a running task) and rewrites it at completion.
The early payload came from ``_build_extra_payload``, which never
included ``auto_commit_mode``; ``persistence._add_task`` maps an
absent toggle to ``0`` (manual commit) even though the schema default
and the legacy migration treat "absent" as ON.  So every task was
persisted as manual-commit from creation until the final save — a
task killed mid-run, or a task running in another process viewed
from the daemon sidebar, was labelled manual-commit with auto-commit
on.

These tests run a real ``WorktreeSorcarAgent`` / ``ChatSorcarAgent``
against the local stand-in model server with a real SQLite history
under an isolated ``KISS_HOME``.  The stand-in's responder reads the
history row through its own ``sqlite3`` connection while the agent is
blocked waiting for the model reply — exactly the cross-process view
the daemon sidebar has of a task running elsewhere — and the test
asserts that mid-run value against both the run's toggle and the
final persisted row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0902-xb-autocommit-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _read_auto_commit_rows() -> list[tuple[str, int]]:
    """Read ``(id, auto_commit_mode)`` of every history row via a fresh connection.

    A separate ``sqlite3`` connection (not the module's per-thread
    cached one) is what another process — the daemon's sidebar — sees.
    """
    conn = sqlite3.connect(str(persistence._DB_PATH), timeout=10)
    try:
        return [
            (str(row[0]), int(row[1]))
            for row in conn.execute(
                "SELECT id, auto_commit_mode FROM task_history"
            ).fetchall()
        ]
    finally:
        conn.close()


class _MidRunSnapshot:
    """Stand-in model that records the history rows before finishing.

    The responder runs on the HTTP server thread while the agent is
    blocked in its first model call: the early row has been inserted,
    the final save has not happened yet.
    """

    def __init__(self) -> None:
        self.mid_run_rows: list[tuple[str, int]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.mid_run_rows = _read_auto_commit_rows()
        return finish_response("done")


def _run_and_snapshot(
    env: IsolatedKissHome, agent: ChatSorcarAgent, **kwargs: Any
) -> tuple[int, int]:
    """Run *agent* once; return ``(mid_run_toggle, final_toggle)`` of its row."""
    model = _MidRunSnapshot()
    server = StandInModelServer(model)
    try:
        result = agent.run(
            prompt_template="say done",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            web_tools=False,
            is_parallel=False,
            verbose=False,
            **kwargs,
        )
    finally:
        server.stop()
    assert yaml.safe_load(result)["success"] is True, result
    task_id = agent.last_task_id
    assert task_id
    mid = dict(model.mid_run_rows)
    assert task_id in mid, "the early row was not visible during the run"
    final = dict(_read_auto_commit_rows())
    return mid[task_id], final[task_id]


def test_worktree_agent_auto_commit_on_is_visible_mid_run(
    env: IsolatedKissHome,
) -> None:
    """With the per-run toggle ON the early row must already say ON."""
    agent = WorktreeSorcarAgent("audit0902-xb-on")
    mid, final = _run_and_snapshot(env, agent, auto_commit=True)
    assert agent.auto_commit_enabled is True
    assert mid == 1, "early row labelled the running task manual-commit"
    assert final == 1
    assert mid == final


def test_worktree_agent_auto_commit_off_is_visible_mid_run(
    env: IsolatedKissHome,
) -> None:
    """With the per-run toggle OFF both rows must say OFF."""
    agent = WorktreeSorcarAgent("audit0902-xb-off")
    mid, final = _run_and_snapshot(env, agent, auto_commit=False)
    assert agent.auto_commit_enabled is False
    assert mid == 0
    assert final == 0


def test_worktree_agent_reads_persisted_config_toggle(
    env: IsolatedKissHome,
) -> None:
    """Without a per-run kwarg the early row follows ``config.json``."""
    env.write_config(auto_commit_mode=False)
    agent = WorktreeSorcarAgent("audit0902-xb-config")
    mid, final = _run_and_snapshot(env, agent)
    assert agent.auto_commit_enabled is False
    assert (mid, final) == (0, 0)


def test_plain_chat_agent_defaults_to_auto_commit_on(
    env: IsolatedKissHome,
) -> None:
    """A ``ChatSorcarAgent`` has no toggle attribute: absent means ON.

    That matches the schema default (``auto_commit_mode INTEGER
    DEFAULT 1``) and the legacy migration (``missing=1``), so the
    early row and the final row must both read ON.
    """
    agent = ChatSorcarAgent("audit0902-xb-chat")
    assert not hasattr(agent, "auto_commit_enabled")
    mid, final = _run_and_snapshot(env, agent)
    assert mid == 1, "early row labelled the running task manual-commit"
    assert final == 1
