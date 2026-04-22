"""Per-tab state and small text helpers for the VS Code server.

Split out of ``server.py`` for organisation.  Imported and
re-exported from ``server`` for backwards compatibility.
"""

from __future__ import annotations

import ctypes
import queue
import re
import threading

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


class _TabState:
    """Per-tab state holding the agent, runtime state, and settings.

    Each chat tab owns a single ``WorktreeSorcarAgent`` so concurrent
    tabs never share mutable agent state (chat_id, last_task_id,
    worktree branch, etc.).  The ``use_worktree`` flag is passed to
    ``agent.run()`` per task — when ``False`` the agent short-circuits
    to the plain stateful code path, so no separate non-worktree agent
    instance is needed.  Runtime state (stop event, task thread,
    answer queue, merge flag) also lives here so the server needs
    only a single ``_tab_states`` dict.
    """

    __slots__ = (
        "agent",
        "use_worktree",
        "use_parallel",
        "task_history_id",
        "selected_model",
        "stop_event",
        "task_thread",
        "user_answer_queue",
        "is_merging",
        "is_running_non_wt",
        "is_task_active",
    )

    def __init__(self, tab_id: str, default_model: str) -> None:
        self.agent = WorktreeSorcarAgent("Sorcar VS Code")
        self.use_worktree: bool = False
        self.use_parallel: bool = False
        self.task_history_id: int | None = None
        self.selected_model: str = default_model
        self.stop_event: threading.Event | None = None
        self.task_thread: threading.Thread | None = None
        self.user_answer_queue: queue.Queue[str] | None = None
        self.is_merging: bool = False
        self.is_running_non_wt: bool = False
        self.is_task_active: bool = False
