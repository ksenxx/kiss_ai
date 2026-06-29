# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: N812, E501
"""End-to-end reproducing tests for the 3 deferred ROUND-3 sorcar-other HIGH bugs.

See ``tmp/review_sorcar_other_r3.md`` for the full review.  These three
fixes were deferred out of session 15 because they required an invasive
ordering refactor of ``ChatSorcarAgent.run`` and ``_run_tasks_parallel``:

  * **H1** — ``_register_running_state()`` runs before ``_add_task``; a
    raise in ``_add_task`` would otherwise leave a stale entry wedged
    inside ``_RunningAgentState.running_agent_states`` forever.
  * **H2** — ``_RunningAgentState`` for sub-agents must be fully
    populated by its constructor; peer threads must never observe a
    half-built state under ``_registry_lock``.
  * **H5** — ``_run_tasks_parallel`` re-snapshots ``self._last_task_id``
    at the start of each ``_run_single`` worker (TOCTOU defeat).
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import cast

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    # Reset shared registries that survive across tests.
    with _RunningAgentState._registry_lock:
        _RunningAgentState.running_agent_states.clear()
    with ChatSorcarAgent._running_agents_lock:
        ChatSorcarAgent.running_agents.clear()
    yield
    with _RunningAgentState._registry_lock:
        _RunningAgentState.running_agent_states.clear()
    with ChatSorcarAgent._running_agents_lock:
        ChatSorcarAgent.running_agents.clear()
    persistence._close_db()


# ---------------------------------------------------------------------------
# Round-3 sorcar-other H1 (HIGH):
#   ``_register_running_state`` runs before ``_add_task``.  When
#   ``_add_task`` raises (DB error, disk-full, etc.) the agent must
#   not leak a stale entry in ``running_agent_states``.
# ---------------------------------------------------------------------------


def test_register_unregisters_when_add_task_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_add_task`` raises mid-``run``, the registered state is removed."""
    agent = ChatSorcarAgent("h1-leak-probe")
    # Force ``_add_task`` to raise.
    from kiss.agents.sorcar import chat_sorcar_agent as csa

    def _boom(*_args: object, **_kwargs: object) -> tuple[str, str]:
        raise RuntimeError("simulated DB write failure")

    monkeypatch.setattr(csa, "_add_task", _boom)
    # Stub out everything beyond ``_add_task`` so we don't run a model.
    monkeypatch.setattr(
        ChatSorcarAgent, "build_chat_prompt",
        lambda self, p: p,  # type: ignore[arg-type]
    )

    pre_keys = set(_RunningAgentState.running_agent_states)
    with pytest.raises(RuntimeError, match="simulated DB write failure"):
        agent.run(prompt_template="hello")
    post_keys = set(_RunningAgentState.running_agent_states)
    assert post_keys == pre_keys, (
        "register_running_state leaked a stale entry after _add_task raised. "
        f"new keys: {post_keys - pre_keys}"
    )


# ---------------------------------------------------------------------------
# Round-3 sorcar-other H2 (HIGH):
#   When a sub-agent state is constructed inside ``_run_tasks_parallel``,
#   peer threads holding ``_registry_lock`` must never see a half-built
#   ``_RunningAgentState``: all required fields (``chat_id``,
#   ``is_subagent``, ``parent_task_id``, ``is_task_active``,
#   ``model_name``, ``agent``) must be populated by the constructor.
# ---------------------------------------------------------------------------


def test_running_agent_state_subagent_kwargs_populate_in_constructor() -> None:
    """The constructor sets all subagent-related fields in one shot."""
    fake_agent = object()  # not a real ChatSorcarAgent but treated as opaque
    parent_uuid = uuid.uuid4().hex
    state = _RunningAgentState(
        "sub-tab-001",
        "gpt-x",
        agent=fake_agent,  # type: ignore[arg-type]
        chat_id="chat-xyz",
        is_subagent=True,
        parent_task_id=parent_uuid,
        is_task_active=True,
    )
    # The H2 fix requires every field below to be set by __init__,
    # not by post-construction attribute writes.
    assert state.chat_id == "chat-xyz"
    assert state.is_subagent is True
    assert state.parent_task_id == parent_uuid
    assert state.is_task_active is True
    assert state.selected_model == "gpt-x"
    assert state.agent is fake_agent


def test_run_tasks_parallel_constructs_subagent_state_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peer-thread observation under ``_registry_lock`` finds a fully-built state.

    A peer thread snapshots ``running_agent_states`` for every key
    matching ``sub-tab-*`` and asserts that the state already has
    ``is_subagent=True``, a non-empty ``chat_id``, and
    ``is_task_active=True``.  Pre-H2 (post-construct attribute writes
    between ``__init__`` and ``register``), a peer could observe
    ``is_subagent=False`` because the writes happened lock-free
    after ``register`` had already published the entry.
    """
    parent = ChatSorcarAgent("h2-parent")
    parent._chat_id = uuid.uuid4().hex  # noqa: SLF001
    parent._last_task_id = uuid.uuid4().hex  # noqa: SLF001

    observations: list[tuple[str, bool, str, bool]] = []
    stop = threading.Event()

    def _peer_observer() -> None:
        while not stop.is_set():
            with _RunningAgentState._registry_lock:
                for k, st in _RunningAgentState.running_agent_states.items():
                    if k.startswith("task-") and "__sub_" in k:
                        observations.append(
                            (k, st.is_subagent, st.chat_id, st.is_task_active),
                        )

    # Stub run() so we don't actually invoke a model.
    def _fake_run(self: ChatSorcarAgent, **_kwargs: object) -> str:
        # Brief pause to widen the observation window.
        import time
        time.sleep(0.005)
        return "summary: done"

    monkeypatch.setattr(ChatSorcarAgent, "run", _fake_run)

    observer = threading.Thread(target=_peer_observer, daemon=True)
    observer.start()
    try:
        parent._run_tasks_parallel(  # noqa: SLF001
            ["task A", "task B", "task C"], max_workers=3,
        )
    finally:
        stop.set()
        observer.join(timeout=2.0)

    assert observations, "peer thread never observed any sub-agent state"
    bad = [o for o in observations if not (o[1] and o[2] and o[3])]
    assert not bad, (
        "peer thread observed half-built sub-agent state(s): "
        f"{bad[:3]}"
    )


# ---------------------------------------------------------------------------
# Round-3 sorcar-other H5 (HIGH):
#   ``_run_single`` re-snapshots ``self._last_task_id`` at worker start
#   so a concurrent batch (or late-arriving parent ``_add_task``) does
#   not orphan the sub-agent row.
# ---------------------------------------------------------------------------


def test_run_single_resnapshots_parent_task_id_at_worker_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later worker picks up a parent task_id that arrives mid-batch.

    With ``max_workers=1`` the executor runs task 1, then task 2.  We
    intentionally update ``parent._last_task_id`` from inside task 1's
    ``run()`` so that the snapshot taken inside ``_run_single`` BEFORE
    task 2's ``run()`` must observe the updated value.  Pre-H5 (no
    re-snapshot inside the closure), task 2 would inherit task 1's
    captured value — empty.
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
        # On the first worker, simulate the parent's _add_task arriving
        # late: publish the real task_history.id BEFORE worker #2's
        # snapshot.
        if len(captured) == 1:
            parent._last_task_id = real_parent_tid  # noqa: SLF001
        return "summary: ok"

    monkeypatch.setattr(ChatSorcarAgent, "run", _fake_run)

    parent._run_tasks_parallel(["task 1", "task 2"], max_workers=1)  # noqa: SLF001

    assert captured[0] == "", (
        f"task 1's snapshot should be empty (parent had no id yet); got {captured[0]!r}"
    )
    assert captured[1] == real_parent_tid, (
        "H5 re-snapshot failed: task 2 should have observed the parent's "
        f"published task_id; got {captured[1]!r}, expected {real_parent_tid!r}"
    )
