# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Per-tab state and small text helpers for the VS Code server.

Originally split out of ``server.py`` for organisation; moved into
the ``sorcar`` package so the per-tab state class lives alongside
its consumer :class:`kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`.

The process-global registry mapping frontend tab id →
:class:`_RunningAgentState` lives directly on this class as
:attr:`_RunningAgentState.running_agent_states` — a registry of its
own instances.
"""

from __future__ import annotations

import ctypes
import queue
import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def parse_task_tags(text: str) -> list[str]:
    """Parse ``<task>...</task>`` tags from *text* and return individual tasks.

    When the input contains one or more ``<task>`` blocks with non-empty
    content, each block's content is returned as a separate list element.
    If no valid ``<task>`` blocks are found (or all are empty/whitespace),
    the original *text* is returned as a single-element list so that
    callers can always iterate without special-casing.

    Args:
        text: Input text potentially containing ``<task>...</task>`` tags.

    Returns:
        List of task strings.  Always contains at least one element.
    """
    tasks = [m.strip() for m in re.findall(r"<task>(.*?)</task>", text, re.DOTALL)]
    tasks = [t for t in tasks if t]
    return tasks if tasks else [text]


ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


class _RunningAgentState:
    """Per-tab state holding settings, runtime state, and the live agent (if any).

    The ``agent`` field is **transient** — populated only while a task
    is actively running (or its post-task worktree-merge UI is still
    in flight) and reset back to ``None`` once the task lifecycle
    completes.  A fresh :class:`WorktreeSorcarAgent` is created at
    :meth:`_CommandsMixin._cmd_run` before the worker thread starts
    and disposed in :meth:`_TaskRunnerMixin._run_task`'s outer
    ``finally``.  Each worktree task creates its own fresh worktree
    and branch independent of any chat id; there is no cross-task
    restoration of worktree state from git.

    Long-lived per-tab state — the canonical ``chat_id``, the most
    recently completed ``last_task_id``, sticky UI flags
    (``use_worktree``, ``use_parallel``, ``selected_model``), and
    lifecycle bookkeeping — lives directly on
    this state so it survives across task boundaries without
    requiring an agent instance.
    """

    # Process-global map of frontend tab id → live per-tab agent
    # runtime state.  Class attribute (shared across every
    # :class:`_RunningAgentState` instance) so any helper inside the
    # ``kiss.agents`` package can inspect or attach to a running
    # agent without holding a reference to either the agent or the
    # VS Code server.  Owned conceptually by the VS Code server,
    # which mutates it under its own ``_state_lock`` to coordinate
    # task lifecycle, merge, autocommit and worktree transitions.
    # Living on the state class itself avoids the prior import cycle
    # with :mod:`kiss.agents.sorcar.worktree_sorcar_agent` and makes
    # the registry self-typed (a dict of this very class).
    # Producers / consumers MUST hold ``VSCodeServer._state_lock``
    # for any multi-step access (read-then-modify, scan-then-modify).
    # ``VSCodeServer._state_lock`` is bound to the very same
    # :class:`threading.RLock` instance held in :attr:`_registry_lock`
    # below so callers outside the VS Code server (parallel sub-agent
    # spawners in :class:`ChatSorcarAgent`, registration helpers in
    # :class:`WorktreeSorcarAgent`) can guard their multi-step access
    # against the server's iteration loops without having to import
    # the server class.
    running_agent_states: dict[str, _RunningAgentState] = {}

    # Process-global lock guarding ``running_agent_states`` against
    # concurrent mutation / iteration.  ``RLock`` (re-entrant) to
    # support nested acquisition by the same thread — the VS Code
    # server occasionally calls helpers that themselves re-enter
    # ``with self._state_lock:`` from within an already-locked
    # critical section.  Bound to ``VSCodeServer._state_lock`` in
    # :meth:`VSCodeServer.__init__` so the two names reference the
    # exact same lock object.
    _registry_lock: threading.RLock = threading.RLock()

    @classmethod
    def register(cls, tab_id: str, state: _RunningAgentState) -> None:
        """Atomically install *state* in :attr:`running_agent_states` under *tab_id*.

        Holds :attr:`_registry_lock` so the insert is serialised
        against the VS Code server's iteration loops (which hold the
        very same lock via ``VSCodeServer._state_lock``) and against
        peer producers (parallel sub-agent spawners, worktree
        register / unregister helpers).
        """
        with cls._registry_lock:
            cls.running_agent_states[tab_id] = state

    @classmethod
    def unregister(cls, tab_id: str) -> None:
        """Atomically remove *tab_id* from :attr:`running_agent_states`.

        No-op when no entry is present.  See :meth:`register` for
        the locking discipline.
        """
        with cls._registry_lock:
            cls.running_agent_states.pop(tab_id, None)

    __slots__ = (
        "agent",
        "tab_id",
        "chat_id",
        "last_task_id",
        "last_user_prompt",
        "task_history_id",
        "use_worktree",
        "use_parallel",
        "auto_commit_mode",
        "selected_model",
        "stop_event",
        "task_thread",
        "user_answer_queue",
        "pending_user_messages",
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
    ) -> None:
        # ``agent`` is transient — the VS Code server flow leaves it
        # ``None`` until :meth:`_CommandsMixin._cmd_run` constructs a
        # fresh agent immediately before the worker thread starts.
        # The standalone :meth:`WorktreeSorcarAgent.run` flow passes
        # ``agent=self`` so its own per-run registration entry points
        # back at the running agent.
        self.agent: WorktreeSorcarAgent | None = agent
        # Frontend routing key for this tab.  Stored on the state so
        # consumers (e.g. multi-viewer subscribe in
        # :meth:`VSCodeServer._reattach_running_chat`) can recover the
        # source tab id without depending on the dict key.
        self.tab_id: str = tab_id
        # Canonical chat id for this tab.  Empty for brand-new tabs
        # that have not yet been associated with a chat; populated by
        # :meth:`_CommandsMixin._cmd_run` (fresh uuid) or by
        # :meth:`VSCodeServer._replay_session` (resumed history row).
        # Orthogonal to :attr:`tab_id` (frontend routing key): the
        # same chat may be live-viewed from multiple tabs.
        self.chat_id: str = ""
        # Primary-key id of the most recently *completed* task in this
        # tab's chat session — used by post-task hooks
        # (:meth:`_MergeFlowMixin._handle_autocommit_action`) that may
        # run after the agent has already been disposed.
        self.last_task_id: int | None = None
        # Most recent user task prompt submitted on this tab.
        # Populated by :meth:`_TaskRunnerMixin._run_task_inner` before
        # each agent run and read by post-task auto-commit hooks
        # (:meth:`_MergeFlowMixin._handle_autocommit_action`) so the
        # generated commit message can include the user's intent.
        # Empty string before the first task has run.
        self.last_user_prompt: str = ""
        # In-flight task id within the current ``_run_task_inner``
        # iteration — used as the persistence target for the
        # ``task_done`` / ``task_stopped`` / ``task_error`` event,
        # the result row, and the extra-payload row.  Reset to
        # ``None`` once the post-task finally block has cleaned up.
        self.task_history_id: int | None = None
        self.use_worktree: bool = False
        self.use_parallel: bool = True
        # ``auto_commit_mode`` mirrors the "Auto commit" menu toggle
        # sent by the frontend on each submit.  When True, the
        # post-task lifecycle in :class:`_TaskRunnerMixin` skips the
        # interactive merge/diff workflow and auto-commits the
        # agent's changes (and in worktree mode also auto-merges into
        # the original branch).
        self.auto_commit_mode: bool = True
        self.selected_model: str = default_model
        self.stop_event: threading.Event | None = None
        self.task_thread: threading.Thread | None = None
        self.user_answer_queue: queue.Queue[str] | None = None
        # Prompts submitted by the user via the task-input textbox
        # while a task is already running on this tab.  Drained into
        # the live agent's conversation as additional ``user`` role
        # messages immediately before each model call (see
        # :meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`'s
        # pre-step hook).  Mutated under
        # :attr:`_registry_lock` so the drain-then-clear sequence
        # cannot race a concurrent ``appendUserMessage`` command from
        # the frontend.  Cleared in :meth:`_TaskRunnerMixin._run_task`
        # 's outer ``finally`` so pending messages never leak across
        # successive tasks on the same tab.
        self.pending_user_messages: list[str] = []
        self.is_merging: bool = False
        self.is_running_non_wt: bool = False
        self.is_task_active: bool = False
        # ``True`` when this tab's in-flight task is being cancelled by
        # the server's graceful-shutdown path
        # (:meth:`RemoteAccessServer._stop_active_agent_tasks`, reached
        # only when ``kiss-web`` receives a ``SIGTERM`` — e.g. a daemon
        # / LaunchAgent restart triggered by an extension update) rather
        # than by the user clicking "Stop".  Both paths inject the same
        # ``KeyboardInterrupt`` into the worker thread, so this flag is
        # the only way the task-runner's ``except KeyboardInterrupt``
        # handler can tell them apart and persist the correct outcome
        # label ("Task interrupted by server restart/shutdown" with an
        # event type of ``task_interrupted`` vs. "Task stopped by
        # user" / ``task_stopped``).  Set BEFORE the interrupt is
        # injected so the handler always observes it.
        self.interrupted_by_shutdown: bool = False
        # ``True`` once the frontend has issued ``closeTab`` for this
        # tab while a task / merge was still in flight.  The tab state
        # is then kept alive (so the running agent can finish) and
        # disposed by :meth:`VSCodeServer._dispose_if_closed` when the
        # last lifecycle flag (``is_task_active`` / ``is_merging`` /
        # ``task_thread.is_alive()``) drops to false.
        self.frontend_closed: bool = False
        # Set by :meth:`ChatSorcarAgent._run_tasks_parallel` for the
        # per-thread state of each parallel sub-agent.  When True,
        # this state represents a sub-agent's live run (sharing the
        # parent's :attr:`chat_id`) and ``parent_task_id`` carries the
        # ``task_history.id`` of the parent task that spawned it.
        # These two fields let
        # :meth:`VSCodeServer._reattach_running_chat` disambiguate
        # which live state to subscribe a freshly-opened history-tab
        # to when several states share the same ``chat_id``.
        self.is_subagent: bool = False
        self.parent_task_id: int | None = None
