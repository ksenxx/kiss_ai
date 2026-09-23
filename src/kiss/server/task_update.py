# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Periodic task updates for the task-info panel.

While a chat webview shows a running task, its task-info panel asks the
daemon (``getTaskUpdate``) for a progress report written by the
:mod:`~kiss.agents.seas.task_update_sea` agent.  :class:`TaskUpdateRunner`
keeps one report per task and runs the agent when the report is missing,
older than :data:`UPDATE_INTERVAL_S`, or explicitly refreshed — never
more than one run per task at a time, and never for a task nobody is
looking at (a poll is what triggers a run).

:func:`run_task_update_sea` runs the agent in-process the way
:func:`~kiss.server.merge_conflict_resolver.run_merge_sea` runs the merge
agent: as a sub-agent of the task, in the task's own chat, so it opens as
a nested tab under the task's tab and its history row nests under the
task's; its spend is attributed to the task.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.agents.seas import task_update_sea

log = logging.getLogger(__name__)

# How long a report stays fresh before a poll triggers the next run.
UPDATE_INTERVAL_S = 600.0
# Reports of tasks that stopped being polled are dropped after this long.
_PRUNE_AFTER_S = 3600.0

# ``(parent_agent, task_id) -> (report_html, cost_usd)``.
TaskUpdateSeaRunner = Callable[[Any, str], tuple[str, float]]


@dataclasses.dataclass
class TaskUpdate:
    """The task-info panel's report for one task.

    Attributes:
        text: The agent's report (HTML), ``""`` before the first run
            completes.  A failed run keeps the previous report.
        error: The failure of the last run, ``""`` when it succeeded.
        finished_at: ``time.time()`` when the last run ended (0 = never).
        cost: USD spent by the last run.
        running: Whether a run is in flight.
        polled_at: ``time.time()`` of the last poll (for pruning).
    """

    text: str = ""
    error: str = ""
    finished_at: float = 0.0
    cost: float = 0.0
    running: bool = False
    polled_at: float = 0.0

    @property
    def sig(self) -> str:
        """Return a fingerprint that changes whenever the panel must repaint."""
        return f"{self.finished_at:.3f}:{int(self.running)}"

    def payload(self) -> dict[str, Any]:
        """Return the wire fields of a ``taskUpdate`` reply."""
        return {
            "exists": bool(self.text or self.error or self.running),
            "content": self.text,
            "error": self.error,
            "running": self.running,
            "cost": self.cost,
            "updatedAt": int(self.finished_at * 1000),
            "sig": self.sig,
        }


def run_task_update_sea(parent_agent: Any, task_id: str) -> tuple[str, float]:
    """Run the task-update agent in-process for *task_id*.

    The child is a :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    stamped like a ``run_parallel`` child (``_tab_id`` /
    ``_subagent_info``) and resumed on *parent_agent*'s chat, so the
    frontend opens it as a nested tab of the task's tab and its history
    row nests under the task's row in the same chat.  It uses the parent's
    model unless the agent script defines ``model()``, the script's system
    prompt, tools and budget cap, and runs in the parent's work dir.

    Its spend is charged to the task whatever way the run ends.  While
    the task still runs it is banked on *parent_agent*'s live counters,
    bound to the usage epoch captured at the start (a parent that has
    since reset for its next task discards it instead of billing that
    task); the task's final save then writes those totals to its row.
    When the task finished while the update ran (its row already holds
    the final totals) the spend is added to the row instead
    (:func:`~kiss.agents.sorcar.persistence._add_task_usage`), so it is
    never counted twice.  Only an update that ends in the few
    milliseconds between the final save reading the counters and
    writing the row can lose its spend.

    Args:
        parent_agent: The agent of the running task to report on.
        task_id: That task's persisted ``task_history`` row id.

    Returns:
        ``(report, cost)``: the child's ``finish`` summary and the USD
        it spent.

    Raises:
        Exception: Whatever the child's run raised (after the spend was
            attributed).
    """
    from kiss.agents.sorcar.chat_sorcar_agent import (
        ChatSorcarAgent,
        _extract_result_summary,
    )
    from kiss.agents.sorcar.persistence import _add_task_usage, _task_is_finished
    from kiss.agents.sorcar.sorcar_agent import (
        _attribute_sub_usage,
        _broadcast_subagent_done,
        _live_agent_usage,
        _persisted_task_id,
    )

    printer = getattr(parent_agent, "printer", None)
    parent_tab_id = str(getattr(parent_agent, "_tab_id", "") or "")
    sub_tab_id = f"task-{task_id}__update-{int(time.time() * 1000)}"
    model_getter = getattr(task_update_sea, "model", None)
    model_name = str(
        model_getter() if callable(model_getter) else parent_agent.model_name
    )
    agent = ChatSorcarAgent("Task update")
    agent._tab_id = sub_tab_id
    agent._subagent_info = {
        "parent_task_id": task_id,
        "parent_tab_id": parent_tab_id,
        "reviewer": False,
    }
    agent.resume_chat_by_id(str(getattr(parent_agent, "chat_id", "") or ""))
    epoch_getter = getattr(parent_agent, "_usage_epoch", None)
    epoch = epoch_getter() if callable(epoch_getter) else None
    result = ""
    try:
        result = agent.run(
            prompt_template=task_update_sea.build_prompt(task_id),
            model_name=model_name,
            work_dir=str(getattr(parent_agent, "work_dir", "") or "."),
            printer=printer,
            tools=task_update_sea.tools(),
            tool_profile=task_update_sea.tool_profile(),
            is_parallel=task_update_sea.is_parallel(),
            max_budget=task_update_sea.max_budget(),
            model_config=(
                getattr(parent_agent, "model_config", None)
                if model_name == parent_agent.model_name else None
            ),
            base_system_prompt=task_update_sea.system_prompt(),
            web_tools=task_update_sea.use_web_tools(),
            use_memory=task_update_sea.use_memory(),
        )
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        if _task_is_finished(task_id):
            try:
                _add_task_usage(task_id, tokens, budget, steps)
            except Exception:
                log.warning(
                    "could not add task-update spend to task %s", task_id,
                    exc_info=True,
                )
        else:
            _attribute_sub_usage(parent_agent, budget, tokens, steps, epoch=epoch)
        if printer is not None:
            viewer_ids: list[str] = []
            fanout = getattr(printer, "_fanout_targets", None)
            sub_task_id = _persisted_task_id(agent)
            found = fanout(sub_task_id) if callable(fanout) and sub_task_id else None
            if isinstance(found, list):
                viewer_ids = [v for v in found if v]
            if sub_tab_id not in viewer_ids:
                viewer_ids.append(sub_tab_id)
            _broadcast_subagent_done(printer, viewer_ids, model_name)
    return _extract_result_summary(result), budget


class TaskUpdateRunner:
    """Keeps one :class:`TaskUpdate` per task and schedules the agent runs.

    Args:
        run_sea: Runs the agent for ``(parent_agent, task_id)`` and
            returns ``(report, cost)``; :func:`run_task_update_sea` by
            default.
    """

    def __init__(self, run_sea: TaskUpdateSeaRunner = run_task_update_sea) -> None:
        self._run_sea = run_sea
        self._updates: dict[str, TaskUpdate] = {}
        self._lock = threading.Lock()

    def poll(self, task_id: str, parent_agent: Any, force: bool = False) -> TaskUpdate:
        """Return the report for *task_id*, starting a run when one is due.

        A run is due when no report exists yet, when the last one ended
        :data:`UPDATE_INTERVAL_S` or more ago, or when *force* is set
        (the panel's refresh button); a run already in flight is never
        doubled.

        Args:
            task_id: The running task's persisted id.
            parent_agent: The running task's agent (the run's parent).
            force: Start a run now even if the report is fresh.

        Returns:
            A snapshot of the task's report state.
        """
        now = time.time()
        with self._lock:
            self._prune(now)
            upd = self._updates.setdefault(task_id, TaskUpdate())
            upd.polled_at = now
            due = force or upd.finished_at == 0.0 or (
                now - upd.finished_at >= UPDATE_INTERVAL_S
            )
            if due and not upd.running:
                upd.running = True
                threading.Thread(
                    target=self._run,
                    args=(task_id, parent_agent),
                    name=f"task-update-{task_id[:8]}",
                    daemon=True,
                ).start()
            return dataclasses.replace(upd)

    def _prune(self, now: float) -> None:
        """Drop idle reports nobody polled for :data:`_PRUNE_AFTER_S`."""
        stale = [
            tid for tid, upd in self._updates.items()
            if not upd.running and now - upd.polled_at >= _PRUNE_AFTER_S
        ]
        for tid in stale:
            del self._updates[tid]

    def _run(self, task_id: str, parent_agent: Any) -> None:
        """Thread body: run the agent and record its report."""
        text, cost, error = "", 0.0, ""
        try:
            text, cost = self._run_sea(parent_agent, task_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("task update for %s failed", task_id, exc_info=True)
        with self._lock:
            upd = self._updates.setdefault(task_id, TaskUpdate())
            if text:
                upd.text = text
            upd.error = error
            upd.cost = cost
            upd.running = False
            upd.finished_at = time.time()
