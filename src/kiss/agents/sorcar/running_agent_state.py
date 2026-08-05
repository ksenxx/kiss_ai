# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Per-tab state for the VS Code server.

Originally split out of ``server.py`` for organisation; moved into
the ``sorcar`` package so the per-tab state class lives alongside
its consumer :class:`kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`.

The process-global registry mapping frontend tab id →
:class:`_RunningAgentState` lives directly on this class as
:attr:`_RunningAgentState.running_agent_states` — a registry of its
own instances.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

logger = logging.getLogger(__name__)


def _tab_busy(tab: _RunningAgentState) -> bool:
    """True when *tab* is owned by somebody and must be left alone.

    A tab is busy while a task is active, a merge review is in
    progress, or its worker thread is installed but not yet started or
    still alive.

    The third clause is not redundant.  ``_cmd_run`` installs
    ``task_thread`` under the state lock and starts it only after
    releasing that lock, and the worker raises ``is_task_active``
    later still, so between submitting a task and it actually
    beginning there is a window in which both flags read False while
    the task is very much real (S3-05).

    Callers must hold the state lock while reading the result — the
    function itself only does plain attribute reads and is not
    internally locked.

    Args:
        tab: The per-tab state to inspect.

    Returns:
        True when any lifecycle flag is still raised.
    """
    return (
        tab.is_task_active
        or tab.is_merging
        or (
            tab.task_thread is not None
            and (
                tab.task_thread.ident is None
                or tab.task_thread.is_alive()
            )
        )
    )


class _RunningAgentState:
    """Per-tab state holding settings, runtime state, and the live agent (if any).

    The ``agent`` field is **transient** — populated only while a task
    is actively running (or its post-task worktree-merge UI is still
    in flight) and reset back to ``None`` once the task lifecycle
    completes.  A fresh :class:`WorktreeSorcarAgent` is created at
    :meth:`_TaskRunnerMixin._run_task_inner` (unless a caller
    pre-populated ``tab.agent``) and disposed in
    :meth:`_TaskRunnerMixin._run_task`'s outer ``finally``.
    Each worktree task creates its own fresh worktree
    and branch independent of any chat id; there is no cross-task
    restoration of worktree state from git.

    Long-lived per-tab state — the canonical ``chat_id``, the most
    recently completed ``last_task_id``, sticky UI flags
    (``use_worktree``, ``use_parallel``, ``selected_model``), and
    lifecycle bookkeeping — lives directly on
    this state so it survives across task boundaries without
    requiring an agent instance.
    """

    running_agent_states: dict[str, _RunningAgentState] = {}

    _registry_lock: threading.RLock = threading.RLock()

    @classmethod
    def register(cls, tab_id: str, state: _RunningAgentState) -> None:
        """Atomically install *state* in :attr:`running_agent_states` under *tab_id*.

        Holds :attr:`_registry_lock` so the insert is serialised
        against the VS Code server's iteration loops (which hold the
        very same lock via ``VSCodeServer._state_lock``) and against
        peer producers (parallel sub-agent spawners, worktree
        register / unregister helpers).

        Overwriting a *different* live entry for the same *tab_id*
        orphans the old state (its ``stop_event`` / ``task_thread``
        become unreachable through the registry, so the old task can
        no longer be stopped from here) — that is sometimes
        intentional (tab reuse) but always worth a trace, so it is
        logged at WARNING.  Semantics are unchanged: the last
        registration wins.
        """
        with cls._registry_lock:
            existing = cls.running_agent_states.get(tab_id)
            if existing is not None and existing is not state:
                logger.warning(
                    "register() overwriting existing running-agent state "
                    "for tab_id=%r (old chat_id=%r, new chat_id=%r); the "
                    "old state is no longer reachable via the registry",
                    tab_id,
                    existing.chat_id,
                    state.chat_id,
                )
            cls.running_agent_states[tab_id] = state

    @classmethod
    def unregister(cls, tab_id: str, state: _RunningAgentState | None = None) -> None:
        """Atomically remove *tab_id* from :attr:`running_agent_states`.

        No-op when no entry is present.  See :meth:`register` for
        the locking discipline.

        Args:
            tab_id: The registry key to remove.
            state: When provided, the entry is removed ONLY if it is
                this exact object.  :meth:`register` explicitly allows
                a different state to replace an entry under the same
                key; a stale owner's key-only cleanup would otherwise
                delete the replacement, orphaning ITS stop event,
                thread, and agent routing (ABA bug).  Pass the state
                you registered whenever you might have been replaced.
        """
        with cls._registry_lock:
            if (
                state is not None
                and cls.running_agent_states.get(tab_id) is not state
            ):
                return
            cls.running_agent_states.pop(tab_id, None)

    __slots__ = (
        "agent",
        "tab_id",
        "chat_id",
        "last_task_id",
        "last_user_prompt",
        "last_result_summary",
        "task_history_id",
        "use_worktree",
        "use_parallel",
        "auto_commit_mode",
        "selected_model",
        "stop_event",
        "task_thread",
        "user_answer_queue",
        "pending_user_messages",
        "unattributed_prompt_echoes",
        "is_merging",
        "is_running_non_wt",
        "is_task_active",
        "interrupted_by_shutdown",
        "frontend_closed",
        "is_subagent",
        "parent_task_id",
    )

    def __init__(
        self,
        tab_id: str,
        default_model: str,
        *,
        agent: WorktreeSorcarAgent | None = None,
        chat_id: str = "",
        is_subagent: bool = False,
        parent_task_id: str | None = None,
        is_task_active: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.agent: WorktreeSorcarAgent | None = agent
        self.tab_id: str = tab_id
        self.chat_id: str = chat_id
        self.last_task_id: str | None = None
        self.last_user_prompt: str = ""
        self.last_result_summary: str = ""
        self.task_history_id: str | None = None
        self.use_worktree: bool = False
        self.use_parallel: bool = True
        self.auto_commit_mode: bool = True
        self.selected_model: str = default_model
        self.stop_event: threading.Event | None = stop_event
        self.task_thread: threading.Thread | None = None
        self.user_answer_queue: queue.Queue[str] | None = None
        self.pending_user_messages: list[str] = []
        self.unattributed_prompt_echoes: list[str] = []
        self.is_merging: bool = False
        self.is_running_non_wt: bool = False
        self.is_task_active: bool = is_task_active
        self.interrupted_by_shutdown: bool = False
        self.frontend_closed: bool = False
        self.is_subagent: bool = is_subagent
        self.parent_task_id: str | None = parent_task_id
