# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Task-keyed registry of running agent and sub-agent states.

This module is the single place where the Sorcar server keeps the
state of live agent runs.  The registry maps a **task id** — the
``task_history`` row id once the run has allocated it, or the
client-minted run id before that — to one :class:`AgentState`.

Agents themselves (``kiss.agents.sorcar``) never import this module:
the layering invariant only lets them depend on ``kiss.core``.  They
reach the registry exclusively through the duck-typed bridge methods
on :class:`kiss.server.json_printer.JsonPrinter`
(``agent_task_allocated`` / ``agent_task_finished`` /
``drain_pending_user_messages`` / ``live_worktree_branches``).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

STATE_LOCK = threading.RLock()
"""Guards :data:`agent_states` and every :class:`AgentState` field."""

agent_states: dict[str, AgentState] = {}
"""The registry: task id → live agent state."""


class AgentState:
    """State of one agent (or sub-agent) task run.

    Created by the server when a ``run`` command arrives (keyed by the
    client-minted task id) or by the printer bridge when an agent
    allocates its ``task_history`` row (sub-agents, standalone runs).
    Re-keyed to the persisted task id via :func:`rekey` as soon as the
    row exists.

    A tab's state lives until its tab is explicitly closed (tabs are
    global state shared by every client, so a mere disconnect never
    disposes one) — in particular a server-owned worktree run whose
    pending worktree (or an in-flight merge/discard) outlives the task
    stays registered until the user merges or discards, because the
    merge flow needs the agent that owns the worktree.
    """

    __slots__ = (
        "task_id",
        "agent",
        "chat_id",
        "tab_id",
        "conn_id",
        "parent_task_id",
        "server_owned",
        "stop_event",
        "task_thread",
        "user_answer_queue",
        "pending_ask_question",
        "pending_user_messages",
        "unattributed_prompt_echoes",
        "is_task_active",
        "is_merging",
        "merge_thread",
        "is_running_non_wt",
        "non_wt_repo_root",
        "interrupted_by_shutdown",
        "frontend_closed",
        "use_worktree",
        "use_parallel",
        "auto_commit_mode",
        "last_user_prompt",
        "last_result_summary",
    )

    def __init__(
        self,
        task_id: str,
        *,
        agent: WorktreeSorcarAgent | None = None,
        chat_id: str = "",
        tab_id: str = "",
        conn_id: str = "",
        parent_task_id: str | None = None,
        server_owned: bool = False,
        stop_event: threading.Event | None = None,
        task_thread: threading.Thread | None = None,
        is_task_active: bool = False,
    ) -> None:
        self.task_id: str = task_id
        self.agent: WorktreeSorcarAgent | None = agent
        self.chat_id: str = chat_id
        self.tab_id: str = tab_id
        self.conn_id: str = conn_id
        self.parent_task_id: str | None = parent_task_id
        self.server_owned: bool = server_owned
        self.stop_event: threading.Event | None = stop_event
        self.task_thread: threading.Thread | None = task_thread
        self.user_answer_queue: queue.Queue[str] | None = None
        # The live ``ask_user_question`` text this task is currently
        # blocked on, or ``""`` when no question is pending.  Set by
        # the asking agent thread, cleared when an answer is consumed
        # or the wait aborts; session replays re-broadcast it so a
        # client that connects mid-question also shows the modal.
        # Read/written under the server's ``_state_lock``.
        self.pending_ask_question: str = ""
        self.pending_user_messages: list[str] = []
        self.unattributed_prompt_echoes: list[str] = []
        self.is_task_active: bool = is_task_active
        self.is_merging: bool = False
        # The thread executing an interactive merge/discard, so
        # shutdown can WAIT for it: a merge rewrites the repository and
        # must never be cut short.  It runs in the event loop's default
        # executor, not in ``task_thread``.
        self.merge_thread: threading.Thread | None = None
        self.is_running_non_wt: bool = False
        # Resolved main-repo root a non-worktree task occupies (set
        # when the task starts), so worktree merges in OTHER
        # repositories are not blocked by it (repo-aware busy guard).
        self.non_wt_repo_root: Path | None = None
        self.interrupted_by_shutdown: bool = False
        self.frontend_closed: bool = False
        self.use_worktree: bool = True
        self.use_parallel: bool = True
        self.auto_commit_mode: bool = True
        self.last_user_prompt: str = ""
        self.last_result_summary: str = ""

    @property
    def is_subagent(self) -> bool:
        """True when this state belongs to a parallel sub-agent run."""
        return self.parent_task_id is not None

    def thread_alive(self) -> bool:
        """True while the run's worker thread is installed and not dead.

        A thread that has been created but not yet started
        (``ident is None``) counts as alive: between submitting a run
        and the worker raising ``is_task_active`` the task is real and
        must not be treated as finished.
        """
        thread = self.task_thread
        return thread is not None and (thread.ident is None or thread.is_alive())

    def busy(self) -> bool:
        """True when the state is owned by a live task or merge/discard.

        Callers must hold :data:`STATE_LOCK` while acting on the
        result.
        """
        return self.is_task_active or self.is_merging or self.thread_alive()


def register(state: AgentState) -> None:
    """Install *state* in the registry under ``state.task_id``.

    Last registration wins; replacing a different live entry is legal
    (the old owner keeps its object and cleans up via the
    identity-checked :func:`unregister`).

    Args:
        state: The state to install.
    """
    with STATE_LOCK:
        agent_states[state.task_id] = state


def unregister(task_id: str, state: AgentState | None = None) -> None:
    """Remove *task_id* from the registry.

    Args:
        task_id: The registry key to remove.
        state: When provided, remove the entry only if it is this
            exact object — a replaced owner must not delete its
            replacement.
    """
    with STATE_LOCK:
        if state is not None and agent_states.get(task_id) is not state:
            return
        agent_states.pop(task_id, None)


def rekey(state: AgentState, new_task_id: str) -> None:
    """Move *state* to *new_task_id*, updating ``state.task_id``.

    Used when the persisted ``task_history`` row id becomes known for
    a run that was registered under a provisional client id, and for
    each subtask of a multi-``<task>`` prompt.

    Args:
        state: The registered state to move.
        new_task_id: The new registry key.
    """
    if not new_task_id:
        return
    with STATE_LOCK:
        if agent_states.get(state.task_id) is state:
            agent_states.pop(state.task_id, None)
        state.task_id = new_task_id
        agent_states[new_task_id] = state


def get(task_id: object) -> AgentState | None:
    """Return the state registered under *task_id*, or ``None``.

    Args:
        task_id: The task id (any type; coerced to ``str``).
    """
    if task_id is None or task_id == "":
        return None
    with STATE_LOCK:
        return agent_states.get(str(task_id))


def find_by_tab(tab_id: str) -> AgentState | None:
    """Return the server-owned state launched from *tab_id*, or ``None``.

    At most one server-owned state exists per tab (a new run on a tab
    replaces the tab's previous state), so the first match is the
    match.  Sub-agent states carry synthetic tab ids of their own.

    Args:
        tab_id: The frontend tab id to look up.
    """
    if not tab_id:
        return None
    with STATE_LOCK:
        for state in agent_states.values():
            if state.tab_id == tab_id and state.server_owned:
                return state
        for state in agent_states.values():
            if state.tab_id == tab_id:
                return state
    return None


def find_by_agent(agent: object) -> AgentState | None:
    """Return the state whose ``agent`` is *agent*, or ``None``.

    Args:
        agent: The live agent instance to look up (identity match).
    """
    if agent is None:
        return None
    with STATE_LOCK:
        for state in agent_states.values():
            if state.agent is agent:
                return state
    return None


def snapshot() -> list[AgentState]:
    """Return a stable snapshot of every registered state."""
    with STATE_LOCK:
        return list(agent_states.values())


def iter_items() -> Iterator[tuple[str, AgentState]]:
    """Yield ``(task_id, state)`` pairs from a locked snapshot."""
    with STATE_LOCK:
        items = list(agent_states.items())
    return iter(items)
