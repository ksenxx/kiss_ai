# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: deleting a parent is refused while a just-admitted child is still starting.

``deleteTask`` / ``deleteChat`` refuse to delete a task whose sub-task
still runs (``_any_running``): a running child that has no
``task_history`` row yet is known only through its state's
``parent_task_id``.  ``_cmd_run`` knew the ``parentTaskId`` of a
dispatched child at admission but did not stamp it onto the child's
``AgentState``; the worker set it later, so in the startup window the
child had a temporary registry key, no parent id and no row, and the
deletion of its finished parent went through unchallenged while the
child was live (it could then persist a row under the deleted parent).

The state now carries the parent id from its construction, under the
admission lock.  Real pieces throughout: the production ``_cmd_run``
admission (a nonexistent agent script makes the worker fail at setup
without a model call), the real registry, ``_cmd_delete_task`` /
``_cmd_delete_chat`` and the real sqlite history.  A profile hook holds
the worker thread at ``_run_task``'s entry so the window stays open;
it replaces no code.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.persistence import _add_task, _load_history, _task_tree
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer

DEADLINE_S = 15.0


class _EntryGate:
    """A profile hook that parks the thread entering *code* until released."""

    def __init__(self, code: Any) -> None:
        self.code = code
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is self.code:
            self.entered.set()
            assert self.release.wait(DEADLINE_S), "gate timed out"
            sys.setprofile(None)


def _admit_gated(server: AgentServer, cmd: dict[str, Any]) -> tuple[_EntryGate, AgentState]:
    """Admit *cmd* through ``_cmd_run`` with its worker parked at ``_run_task``'s entry."""
    gate = _EntryGate(server._run_task.__func__.__code__)  # type: ignore[attr-defined]
    threading.setprofile(gate)
    try:
        server._cmd_run(cmd)
        assert gate.entered.wait(DEADLINE_S), "the worker never reached _run_task"
    finally:
        threading.setprofile(None)
    state = agent_state.agent_states[cmd["_state_key"]]
    assert state.task_thread is not None and state.task_thread.is_alive()
    return gate, state


def _finish_gated(gate: _EntryGate, state: AgentState) -> None:
    """Stop the parked run and let its worker finish."""
    if state.stop_event is not None:
        state.stop_event.set()
    thread = state.task_thread
    gate.release.set()
    if thread is not None:
        thread.join(DEADLINE_S)
        assert not thread.is_alive(), "the worker did not finish"


def _delete_gated(
    server: AgentServer, delete: Callable[[dict[str, Any]], None], cmd: dict[str, Any],
) -> tuple[_EntryGate, threading.Thread]:
    """Run ``delete(cmd)`` on a thread parked at ``_delete_history``'s entry."""
    gate = _EntryGate(server._delete_history.__func__.__code__)  # type: ignore[attr-defined]
    threading.setprofile(gate)
    try:
        thread = threading.Thread(target=delete, args=(cmd,), daemon=True)
        thread.start()
        assert gate.entered.wait(DEADLINE_S), "the delete never reached _delete_history"
    finally:
        threading.setprofile(None)
    return gate, thread


def _run_help_child(server: AgentServer, tmp_path: Path, parent_id: str, chat_id: str) -> str:
    """Run a model-free ``/sh help`` sub-task of *parent_id* to its end; its row id."""
    cmd = {
        "type": "run", "tabId": "help-child-tab", "prompt": "/sh help", "chatId": chat_id,
        "parentTaskId": parent_id, "parentTabId": "parent-tab", "workDir": str(tmp_path),
    }
    server._cmd_run(cmd)
    thread = agent_state.agent_states[cmd["_state_key"]].task_thread
    assert thread is not None
    thread.join(DEADLINE_S)
    assert not thread.is_alive(), "the /sh help child did not finish"
    tree = _task_tree(parent_id)
    assert len(tree) == 2 and tree[0] == parent_id, tree
    return tree[1]


def _row(task_id: str) -> dict[str, Any] | None:
    return next((r for r in _load_history() if r["id"] == task_id), None)


def _child_cmd(tmp_path: Path, parent_id: str, chat_id: str, tab_id: str) -> dict[str, Any]:
    return {
        "type": "run", "tabId": tab_id, "prompt": "child task", "chatId": chat_id,
        "parentTaskId": parent_id, "parentTabId": "parent-tab", "workDir": str(tmp_path),
        "seaPath": str(tmp_path / "does-not-exist.py"),
    }


def _server() -> tuple[AgentServer, list[dict[str, Any]], threading.Lock]:
    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


def _notices(events: list[dict[str, Any]], lock: threading.Lock) -> list[str]:
    with lock:
        return [e["text"] for e in events if e.get("type") == "notice"]


def test_delete_task_is_refused_while_its_child_is_still_starting(tmp_path: Path) -> None:
    """The child's state names its parent from admission on; the parent row survives."""
    parent_id, chat_id = _add_task("finished parent", chat_id="")
    server, events, lock = _server()
    gate, state = _admit_gated(server, _child_cmd(tmp_path, parent_id, chat_id, "child-tab"))
    try:
        assert state.running()

        server._cmd_delete_task({"taskId": parent_id, "tabId": "chats", "connId": "c1"})

        assert _row(parent_id) is not None, "the parent was deleted under its running child"
        assert _notices(events, lock) == [
            "Cannot delete the task: its task is still running. Stop it first.",
        ]
        assert state.parent_task_id == parent_id, "the child's state does not name its parent"
        assert state.is_subagent
    finally:
        _finish_gated(gate, state)
        for s in agent_state.snapshot():
            agent_state.unregister(s.task_id, s)


def test_delete_chat_is_refused_while_a_child_of_its_task_is_still_starting(
    tmp_path: Path,
) -> None:
    """The same guard for ``deleteChat`` of the parent's chat."""
    parent_id, chat_id = _add_task("finished parent", chat_id="")
    server, events, lock = _server()
    # The child is admitted into a chat of its own: only its parent id
    # ties it to the chat being deleted.
    gate, state = _admit_gated(
        server, _child_cmd(tmp_path, parent_id, "child-own-chat", "child-tab-2"),
    )
    try:
        assert state.chat_id != chat_id

        server._cmd_delete_chat({"chatId": chat_id, "tabId": "chats", "connId": "c2"})

        assert _row(parent_id) is not None, "the chat was deleted under a running child"
        assert _notices(events, lock) == [
            "Cannot delete the chat: its task is still running. Stop it first.",
        ]
    finally:
        _finish_gated(gate, state)
        for s in agent_state.snapshot():
            agent_state.unregister(s.task_id, s)


@pytest.mark.parametrize("what", ["task", "chat"])
def test_delete_sees_a_sub_task_admitted_after_it_read_the_tree(
    tmp_path: Path, what: str,
) -> None:
    """A child finishes and its own child starts while the delete is under way: refused.

    The delete used to read the task tree before it blocked new
    admissions, so a grandchild admitted meanwhile (its parent being
    the child that finished in between, not the root the stale tree
    named) escaped the liveness check, and the delete then removed
    the tree persistence recomputed, the running grandchild's rows
    included.
    """
    root_id, chat_id = _add_task("finished root", chat_id="")
    server, events, lock = _server()
    if what == "task":
        delete, cmd = server._cmd_delete_task, {"taskId": root_id}
    else:
        delete, cmd = server._cmd_delete_chat, {"chatId": chat_id}
    gate, deleting = _delete_gated(server, delete, {**cmd, "tabId": "chats", "connId": "c3"})
    grand: tuple[_EntryGate, AgentState] | None = None
    try:
        child_id = _run_help_child(server, tmp_path, root_id, chat_id)
        grand = _admit_gated(
            server, _child_cmd(tmp_path, child_id, "grandchild-own-chat", "grandchild-tab"),
        )
        assert grand[1].running() and grand[1].parent_task_id == child_id
        gate.release.set()
        deleting.join(DEADLINE_S)
        assert not deleting.is_alive()

        assert _row(root_id) is not None, f"the {what} was deleted under a running grandchild"
        assert _task_tree(root_id) == [root_id, child_id], "the grandchild's parent was deleted"
        assert _notices(events, lock) == [
            f"Cannot delete the {what}: its task is still running. Stop it first.",
        ]
        assert server._history_deletes == 0
    finally:
        gate.release.set()
        deleting.join(DEADLINE_S)
        if grand is not None:
            _finish_gated(*grand)
        for s in agent_state.snapshot():
            agent_state.unregister(s.task_id, s)
