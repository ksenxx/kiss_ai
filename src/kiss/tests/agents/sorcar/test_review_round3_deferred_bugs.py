# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end test for the ROUND-3 sorcar-other H5 TOCTOU bug.

See ``tmp/review_sorcar_other_r3.md`` for the original review.

  * **H5** — ``_run_tasks_parallel`` re-snapshots ``self._last_task_id``
    at the start of each ``_run_single`` worker (TOCTOU defeat), so a
    sub-agent launched after the parent's task row is persisted picks
    up the real parent task id instead of a stale pre-persist snapshot.

(The former H1/H2 tests verified the deleted per-tab
``_RunningAgentState`` self-registration mechanics and were removed
with that registry.)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import cast

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()
    yield
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()
    persistence._close_db()


def test_run_single_resnapshots_parent_task_id_at_worker_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later worker picks up a parent task_id that arrives mid-batch.

    With ``max_workers=1`` the executor runs task 1, then task 2.  We
    intentionally update ``parent._last_task_id`` from inside task 1's
    ``run()`` so that the snapshot taken inside ``_run_single`` BEFORE
    task 2's ``run()`` must observe the updated value.  Without the
    re-snapshot inside the closure, task 2 would inherit task 1's
    captured value — the fan-out's synthetic placeholder, which names
    no history row.
    """
    parent = ChatSorcarAgent("h5-toctou-probe")
    parent._chat_id = uuid.uuid4().hex  # noqa: SLF001
    parent._last_task_id = ""  # noqa: SLF001
    real_parent_tid = uuid.uuid4().hex

    captured: list[str | None] = []

    def _fake_run(self: ChatSorcarAgent, **_kwargs: object) -> str:
        info = self._subagent_info  # noqa: SLF001
        captured.append(
            cast("str | None", info["parent_task_id"]) if info else None,
        )
        if len(captured) == 1:
            parent._last_task_id = real_parent_tid  # noqa: SLF001
        return "summary: ok"

    monkeypatch.setattr(ChatSorcarAgent, "run", _fake_run)

    parent._run_tasks_parallel(["task 1", "task 2"], max_workers=1)  # noqa: SLF001

    assert captured[0] and captured[0] != real_parent_tid, (
        "task 1 ran before the parent had a row, so it must carry the "
        "fan-out's synthetic parent id — never a blank one (which would "
        f"make it a top-level history row); got {captured[0]!r}"
    )
    assert captured[1] == real_parent_tid, (
        "H5 re-snapshot failed: task 2 should have observed the parent's "
        f"published task_id; got {captured[1]!r}, expected {real_parent_tid!r}"
    )
