# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Periodic task updates for the task-info panel.

While a chat webview shows a running task, its task-info panel asks the
daemon (``getTaskUpdate``) for a short progress update.  The update is
the answer of the ``/ask`` agent
(:mod:`~kiss.agents.seas.ask.ask_sea`) to :data:`UPDATE_QUESTION`, the
same two-or-three-sentence answer a user gets by typing ``/ask`` into
the task's chat.  :class:`TaskUpdateRunner` keeps one update per task
and runs the agent once the task is :data:`FIRST_UPDATE_DELAY_S` old,
then whenever the update is older than :data:`UPDATE_INTERVAL_S` or is
explicitly refreshed — never more than one run per task at a time, and
never for a task nobody is looking at (a poll is what triggers a run).

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

from kiss.agents.seas.ask.ask_sea import AskSea
from kiss.agents.sorcar._concurrency import _race_delay
from kiss.agents.sorcar.persistence import TASK_USAGE_LOCK
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.server.json_printer import stamp_event_ts

log = logging.getLogger(__name__)

# The question the panel asks the ``/ask`` agent about the task.
UPDATE_QUESTION = "What has this task done so far, and what are its partial results?"
# A task's first update runs this long after the task started.
FIRST_UPDATE_DELAY_S = 60.0
# How long an update stays fresh before a poll triggers the next run.
UPDATE_INTERVAL_S = 600.0
# USD cap of one run.
UPDATE_BUDGET_USD = 1.0
# Updates of tasks that stopped being polled are dropped after this long.
_PRUNE_AFTER_S = 3600.0

# ``(parent_agent, task_id) -> (answer_html, cost_usd)``.
TaskUpdateSeaRunner = Callable[[Any, str], tuple[str, float]]


def build_prompt(task_id: str) -> str:
    """Return the ``/ask`` prompt the update run answers for *task_id*.

    What a ``/ask`` typed into the task's chat produces: the
    :class:`AskSea` ``prompt`` applied to the question, with
    ``{task_id}`` filled in as the daemon would.

    Args:
        task_id: The ``task_history`` row id of the task to report on.

    Returns:
        The prompt text.
    """
    return evaluate_sea([AskSea()], UPDATE_QUESTION, task_id).prompt


@dataclasses.dataclass
class TaskUpdate:
    """The task-info panel's update for one task.

    Attributes:
        text: The agent's answer (HTML), ``""`` before the first run
            completes.  A failed run keeps the previous answer.
        error: The failure of the last run, ``""`` when it succeeded.
        finished_at: ``time.time()`` when the last run ended (0 = never).
        cost: USD spent by the last run.
        running: Whether a run is in flight.
        started_at: ``time.time()`` when the task started (the first
            poll's time when the task's agent does not tell).
        polled_at: ``time.time()`` of the last poll (for pruning).
    """

    text: str = ""
    error: str = ""
    finished_at: float = 0.0
    cost: float = 0.0
    running: bool = False
    started_at: float = 0.0
    polled_at: float = 0.0

    @property
    def due_at(self) -> float:
        """Return the ``time.time()`` before which no unforced run starts.

        The task's start plus :data:`FIRST_UPDATE_DELAY_S` until the
        first run has ended, then the last run's end plus
        :data:`UPDATE_INTERVAL_S`.
        """
        if self.finished_at:
            return self.finished_at + UPDATE_INTERVAL_S
        return self.started_at + FIRST_UPDATE_DELAY_S

    @property
    def sig(self) -> str:
        """Return a fingerprint that changes whenever the panel must repaint."""
        return f"{self.finished_at:.3f}:{int(self.running)}"

    def payload(self) -> dict[str, Any]:
        """Return the wire fields of a ``taskUpdate`` reply.

        ``exists`` is true from the first poll on: before the first run
        the panel shows when that run is due (``dueAt``), so a task that
        is seconds old does not look like a task without an update.
        """
        return {
            "exists": True,
            "content": self.text,
            "error": self.error,
            "running": self.running,
            "cost": self.cost,
            "updatedAt": int(self.finished_at * 1000),
            "dueAt": int(self.due_at * 1000),
            "sig": self.sig,
        }


def mark_legacy_updates_as_side_channels() -> int:
    """Stamp task-update children persisted before the side-channel flag.

    Rows written by earlier releases of :func:`run_task_update_sea`
    (which ran the ``/task_update`` agent with its prompt template)
    carry ``is_side_channel = 0``, so every reload of a chat re-opened
    each finished update as a dead sub-agent tab.  The daemon calls this
    once at startup; it is idempotent.

    Returns:
        The number of rows newly stamped.
    """
    from kiss.agents.seas.task_update import task_update_sea
    from kiss.agents.sorcar.persistence import _mark_legacy_side_channel_rows

    return _mark_legacy_side_channel_rows(task_update_sea.PROMPT_TEMPLATE)


def charge_side_channel_usage(
    printer: Any,
    task_agent: Any,
    task_id: str,
    budget: float,
    tokens: int,
    steps: int,
    epoch: Any = None,
) -> None:
    """Charge a side channel's spend to the task it worked for.

    A side channel (a task-update run, an ``/ask`` answerer) may end
    before or after its task.  While the task runs, the spend is banked
    on *task_agent*'s live counters (bound to *epoch*), so the task's
    final save and, for a sub-agent task, its parent's fold include
    it.  Once the task has finished, the spend is added to its row and
    to each finished ancestor's row
    (:func:`~kiss.agents.sorcar.persistence._add_late_task_usage`),
    every updated task's tabs get its new totals as a persisted
    ``usage_info``, and a still-running ancestor, when there is one,
    banks the spend on its live agent.  A running task's new totals
    are broadcast as its own ``usage_info`` (recorded and persisted),
    so its replayed transcript ends with the cost its row will store.  A task
    with no row at all (a run that persists none) is banked on its
    live agent like an unfinished one.  A task that finishes in the
    moment between the row check and the bank, after its final save
    read its counters, loses the spend (the same narrow window every
    live-agent bank has).

    Args:
        printer: The printer that shows the task's tabs, or ``None``.
        task_agent: The live agent of the task *task_id*.
        task_id: The task's persisted ``task_history`` row id.
        budget: USD spent.
        tokens: Tokens spent.
        steps: Steps taken.
        epoch: *task_agent*'s usage epoch captured when the side channel
            started (see ``RelentlessAgent._attribute_usage``), or
            ``None`` for its current one.
    """
    from kiss.agents.sorcar.persistence import (
        _add_late_task_usage,
        _append_chat_event,
    )
    from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage, _live_agent_usage
    from kiss.server import agent_state

    if not task_id or (budget <= 0 and tokens <= 0 and steps <= 0):
        return
    with TASK_USAGE_LOCK:
        try:
            updated, running = _add_late_task_usage(task_id, tokens, budget, steps)
        except Exception:
            log.warning(
                "could not add side-channel spend to task %s", task_id,
                exc_info=True,
            )
            return
        _race_delay()  # test hook: widens the row-check-to-bank window
        if not updated and not running:
            # No row for the task (its run persists none, or has not
            # saved one yet): the live agent is the only place the
            # spend can go, exactly as for an unfinished row.
            running = task_id
        banked = False
        if running:
            # Banked first: the task can finish (saving its counters) at
            # any moment after the transaction above released it.
            if running == task_id:
                agent = task_agent
            else:
                state = agent_state.get(running)
                agent = state.agent if state is not None else None
                epoch = None
            if agent is None:
                log.warning(
                    "side-channel spend of task %s not charged to running "
                    "task %s: no live agent", task_id, running,
                )
            else:
                _attribute_sub_usage(agent, budget, tokens, steps, epoch=epoch)
                banked = True
                if printer is not None:
                    # Publish the new totals (banked plus the in-flight
                    # session's) as one of the running task's own
                    # events, recorded and persisted like its other
                    # usage events: the task may already be past its
                    # last one, and both its replayed transcript and a
                    # ``run_agent`` caller waiting on it
                    # (daemon_client.run) take the cost from the latest.
                    live_budget, live_tokens, live_steps = _live_agent_usage(agent)
                    printer.broadcast({
                        "type": "usage_info",
                        "text": "",
                        "taskId": running,
                        "total_tokens": live_tokens,
                        "cost": f"${live_budget:.4f}",
                        "total_steps": live_steps,
                    })
        for row_id, row_tokens, row_cost, row_steps in updated:
            event: dict[str, Any] = {
                "type": "usage_info",
                "text": "",
                "total_tokens": row_tokens,
                "cost": f"${row_cost:.4f}",
                "total_steps": row_steps,
            }
            if banked:
                # The spend was banked on the running ancestor above,
                # so a ``run_agent`` caller still waiting on this row's
                # task must not fold it again (daemon_client._net_totals
                # subtracts it).
                event["ancestor_charged"] = {
                    "cost": budget, "tokens": tokens, "steps": steps,
                }
            stamp_event_ts(event)
            if printer is not None:
                printer.broadcast_transient(event, task_id=row_id)
            _append_chat_event(dict(event), task_id=row_id)
        if updated and printer is not None:
            printer.broadcast({"type": "tasks_updated"})


def run_task_update_sea(parent_agent: Any, task_id: str) -> tuple[str, float]:
    """Ask the ``/ask`` agent :data:`UPDATE_QUESTION` about *task_id*, in-process.

    The child is a :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    stamped like a ``run_parallel`` child (``_tab_id`` /
    ``_subagent_info``) and resumed on *parent_agent*'s chat, so the
    frontend opens it as a nested tab of the task's tab and its history
    row nests under the task's row in the same chat.  The stamp marks it
    a side channel: its tab is open only while it runs — a reload never
    re-opens the finished tab, because its answer lives in the task-info
    panel, not in the tab.  It is configured exactly like a ``/ask``
    typed into the task's chat (:mod:`~kiss.agents.seas.ask.ask_sea`'s
    getters: its base system prompt and playbook suffix, the
    ``task_context`` tool and ``finish`` only, no memory, browser or
    sub-agents), capped at :data:`UPDATE_BUDGET_USD`, with the parent's
    model unless the script defines ``model()``, in the parent's work
    dir.

    Its spend is charged to the task whatever way the run ends.  While
    the task still runs it is banked on *parent_agent*'s live counters,
    bound to the usage epoch captured at the start (a parent that has
    since reset for its next task discards it instead of billing that
    task); the task's final save then writes those totals to its row.
    When the task finished while the update ran (its row already holds
    the final totals) the spend is added to the row and its finished
    ancestors instead (see :func:`charge_side_channel_usage`), so it is
    never counted twice.  Only an update that ends in the few
    milliseconds between the final save reading the counters and
    writing the row can lose its spend.

    Args:
        parent_agent: The agent of the running task to report on.
        task_id: That task's persisted ``task_history`` row id.

    Returns:
        ``(answer, cost)``: the child's ``finish`` summary and the USD
        it spent.

    Raises:
        Exception: Whatever the child's run raised (after the spend was
            attributed).
    """
    from kiss.agents.sorcar.chat_sorcar_agent import (
        ChatSorcarAgent,
        _extract_result_summary,
    )
    from kiss.agents.sorcar.sorcar_agent import (
        _live_agent_usage,
        _notify_subagent_done,
        _persisted_task_id,
        subagent_parent_tab_id_of,
    )

    printer = getattr(parent_agent, "printer", None)
    # The tab the webviews show the parent under — for a sub-agent
    # parent its ``{parent}__sub_{task}`` tab, never its synthetic id.
    parent_tab_id = subagent_parent_tab_id_of(parent_agent)
    sub_tab_id = f"task-{task_id}__update-{int(time.time() * 1000)}"
    ask = evaluate_sea([AskSea()], UPDATE_QUESTION, task_id)
    ask_settings = ask.settings
    model_name = str(ask_settings.get("model") or parent_agent.model_name)
    agent = ChatSorcarAgent("Task update")
    agent._tab_id = sub_tab_id
    # A side channel like the ``/ask`` answerer: its answer lands in the
    # task-info panel, so the finished child has no tab worth reopening.
    # The daemon replays a finished side channel as ``subagentDone``
    # instead of ``openSubagentTab`` (see ``server._is_side_channel_row``);
    # without the stamp every reload re-opened one finished tab per
    # periodic update.
    agent._subagent_info = {
        "parent_task_id": task_id,
        "parent_tab_id": parent_tab_id,
        "reviewer": False,
        "side_channel": True,
    }
    agent.resume_chat_by_id(str(getattr(parent_agent, "chat_id", "") or ""))
    epoch_getter = getattr(parent_agent, "_usage_epoch", None)
    epoch = epoch_getter() if callable(epoch_getter) else None
    result = ""
    try:
        result = agent.run(
            prompt_template=ask.prompt,
            model_name=model_name,
            work_dir=str(getattr(parent_agent, "work_dir", "") or "."),
            printer=printer,
            # The ask SEA's ``none`` tool profile: its ``tools()`` and
            # ``finish`` are the whole tool set, no built-in tools.
            tools_hook=ask.tools_hook,
            llm_call_hook=ask.llm_call_hook,
            tool_call_hook=ask.tool_call_hook,
            append_basic_tools=False,
            tool_profile=ask_settings["tool_profile"],
            max_budget=UPDATE_BUDGET_USD,
            model_config=(
                getattr(parent_agent, "model_config", None)
                if model_name == parent_agent.model_name else None
            ),
            system_prompt_hook=ask.system_prompt_hook,
            web_tools=ask_settings["use_web_tools"],
            use_memory=ask_settings["use_memory"],
        )
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        charge_side_channel_usage(
            printer, parent_agent, task_id, budget, tokens, steps, epoch=epoch,
        )
        if printer is not None:
            _notify_subagent_done(
                printer, _persisted_task_id(agent), sub_tab_id, model_name,
            )
    return _extract_result_summary(result), budget


def _task_started_at(task_id: str, now: float) -> float:
    """Return when the task *task_id* started (``time.time()``), or *now* if unknown.

    Read from the task's own ``task_history`` row (``start_ts``, or the
    row's insertion ``timestamp`` for legacy rows), which every run —
    each ``<task>`` block of a sequential submission, each
    ``run_parallel`` child — allocates with its own start; the agent's
    ``_task_start_ms`` would date a later block of a sequential
    submission by the first one.  Blocks on the history database's
    lock, so call it off the event loop.
    """
    from kiss.agents.sorcar import persistence

    with persistence._rw_lock.read_lock():
        row = persistence._get_db().execute(
            "SELECT start_ts, timestamp FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    if row is None:
        return now
    start_ms = persistence._safe_int(row["start_ts"], 0)
    if start_ms > 0:
        return start_ms / 1000.0
    return float(row["timestamp"]) or now


class TaskUpdateRunner:
    """Keeps one :class:`TaskUpdate` per task and schedules the agent runs.

    Args:
        run_sea: Runs the agent for ``(parent_agent, task_id)`` and
            returns ``(answer, cost)``; :func:`run_task_update_sea` by
            default.
    """

    def __init__(self, run_sea: TaskUpdateSeaRunner = run_task_update_sea) -> None:
        self._run_sea = run_sea
        self._updates: dict[str, TaskUpdate] = {}
        self._lock = threading.Lock()

    def poll(self, task_id: str, parent_agent: Any, force: bool = False) -> TaskUpdate:
        """Return the update for *task_id*, starting a run when one is due.

        A run is due once the task is :data:`FIRST_UPDATE_DELAY_S` old
        and no run has ended yet, when the last one ended
        :data:`UPDATE_INTERVAL_S` or more ago, or when *force* is set
        (the panel's refresh button); a run already in flight is never
        doubled.

        The first poll for a task reads the task's start from its
        history row, so call this off the event loop
        (``asyncio.to_thread``).

        Args:
            task_id: The running task's persisted id.
            parent_agent: The running task's agent (the run's parent).
            force: Start a run now even if the update is fresh.

        Returns:
            A snapshot of the task's update state.
        """
        now = time.time()
        upd = self._update_for(task_id, now)
        with self._lock:
            if (force or now >= upd.due_at) and not upd.running:
                # ``running`` is set only once the thread exists (a
                # failed ``start()`` under thread exhaustion would
                # otherwise pin the update on "running" for good);
                # ``_run``'s finishing write waits for this lock, so
                # it cannot be overtaken.
                threading.Thread(
                    target=self._run,
                    args=(task_id, parent_agent),
                    name=f"task-update-{task_id[:8]}",
                    daemon=True,
                ).start()
                upd.running = True
            return dataclasses.replace(upd)

    def _update_for(self, task_id: str, now: float) -> TaskUpdate:
        """Return the update of *task_id* polled at *now*, created on its first poll.

        Pruning runs first, then the polled update is touched so no
        concurrent poll prunes it; only a task without an update reads
        its start from the history row (outside the lock).
        """
        with self._lock:
            self._prune(now)
            upd = self._updates.get(task_id)
            if upd is not None:
                upd.polled_at = now
                return upd
        started_at = _task_started_at(task_id, now)
        with self._lock:
            # A concurrent first poll may have created it meanwhile.
            upd = self._updates.get(task_id)
            if upd is None:
                upd = TaskUpdate(started_at=started_at)
                self._updates[task_id] = upd
            upd.polled_at = now
            return upd

    def _prune(self, now: float) -> None:
        """Drop idle updates nobody polled for :data:`_PRUNE_AFTER_S`."""
        stale = [
            tid for tid, upd in self._updates.items()
            if not upd.running and now - upd.polled_at >= _PRUNE_AFTER_S
        ]
        for tid in stale:
            del self._updates[tid]

    def _run(self, task_id: str, parent_agent: Any) -> None:
        """Thread body: run the agent and record its answer."""
        text, cost, error = "", 0.0, ""
        try:
            text, cost = self._run_sea(parent_agent, task_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("task update for %s failed", task_id, exc_info=True)
        finally:
            # Also on a BaseException: an update pinned on ``running``
            # would never be restarted by ``poll`` nor pruned.
            with self._lock:
                upd = self._updates.setdefault(task_id, TaskUpdate())
                if text:
                    upd.text = text
                upd.error = error
                upd.cost = cost
                upd.running = False
                upd.finished_at = time.time()
