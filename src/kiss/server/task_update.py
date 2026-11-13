# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Periodic task updates for the task-info panel.

While a chat page shows a running task, its task-info panel asks the
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
:func:`~kiss.agents.sorcar.merge_conflict_resolver.run_merge_sea` runs the merge
agent: as a sub-agent of the task, in the task's own chat, so it opens as
a nested tab under the task's tab and its history row nests under the
task's; its spend is attributed to the task.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, cast

from kiss.agents.seas.ask.ask_sea import AskSea
from kiss.agents.sorcar.persistence import TASK_USAGE_LOCK, task_start_ms, usage_totals_event
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.core import stop_signal
from kiss.core.locked import _race_delay, locked_call
from kiss.server import agent_state
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
# How often a blocked ``_ChildStopEvent.wait`` re-checks the parent's event.
_PARENT_STOP_POLL_S = 0.1

# ``(parent_agent, task_id) -> (answer_html, cost_usd)``.
TaskUpdateSeaRunner = Callable[[Any, str], tuple[str, float]]


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
    tab_id: str = "",
    key: str | None = None,
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
    live agent like an unfinished one.  The row check and the bank are
    one section under ``TASK_USAGE_LOCK``, which the task's final save
    holds from reading its counters to writing its row: a save that
    comes first leaves the finished row for the charge to add to, one
    that comes second reads the bank.  A charge that fails (a row update, a usage
    broadcast, a chat-event write) is logged, never raised: it must not
    cost the caller what follows it, the tabs' ``subagentDone`` or the
    ``/ask`` answer panel.

    A charge made with a *key* counts once however often it is made: the
    live ledger counts the key once, the finished rows take it in one
    transaction that leaves the key as its receipt
    (:func:`~kiss.agents.sorcar.persistence._add_keyed_task_usage`), and
    what a running ancestor owes is recorded there and banked like a
    remote sub-agent's growth (:func:`_bank_remote_charge`).  A key the
    charge's ledger holds is charged nowhere again (:func:`_key_on_ledger`):
    its row may have finished since, from a snapshot of that ledger.

    Args:
        printer: The printer that shows the task's tabs, or ``None``.
        task_agent: The live agent of the task *task_id*, or ``None``
            when the spend is already on that agent's counters (the
            task's own agent charging spend made after its row was
            saved): an unfinished row's final save then includes it,
            and a finished row takes it without a retained charge,
            since every later snapshot of that ledger carries it
            (:func:`~kiss.agents.sorcar.persistence._add_late_task_usage`'s
            ``banked``).
        task_id: The task's persisted ``task_history`` row id.
        budget: USD spent.
        tokens: Tokens spent.
        steps: Steps taken.
        epoch: *task_agent*'s usage epoch captured when the side channel
            started (see ``RelentlessAgent._attribute_usage``), or
            ``None`` for its current one.
        tab_id: A tab that gets *task_id*'s new totals besides the tabs
            watching the task (whose subscriptions lapse a few minutes
            after it ends), or ``""``.
        key: The key of a charge its caller may make again (a gate
            review attempt a stop ended), or ``None``.
    """
    if not task_id or (budget <= 0 and tokens <= 0 and steps <= 0):
        return
    try:
        locked_call(
            TASK_USAGE_LOCK, _charge_locked,
            printer, task_agent, task_id, budget, tokens, steps, epoch, tab_id, key,
        )
    except Exception:
        log.warning("side-channel spend of task %s not charged", task_id, exc_info=True)


def _charge_locked(
    printer: Any,
    task_agent: Any,
    task_id: str,
    budget: float,
    tokens: int,
    steps: int,
    epoch: Any,
    tab_id: str,
    key: str | None,
) -> None:
    """Charge the spend to the row or the live agent (``TASK_USAGE_LOCK`` held).

    See :func:`charge_side_channel_usage` for the arguments.
    """
    from kiss.agents.sorcar.persistence import (
        _add_keyed_task_usage,
        _add_late_task_usage,
        _unpublished_keyed_charge,
    )

    charge = 0
    if key:
        # The caller's second charge of a key: the first is counted already, on
        # the ledger (this bridge banks there only with a live agent) or in the row.
        if task_agent is not None and _key_on_ledger(task_agent, epoch, key):
            return  # the ledger's snapshot is the row's totals, or will be
        charged = _add_keyed_task_usage(task_id, key, tokens, budget, steps)
        if charged is None:
            # The row kept the key as its receipt.  A Stop may have cut the
            # first charge between its commit and its publication: publish
            # the rows' totals their latest usage events still miss.
            stale, running, owed = _unpublished_keyed_charge(task_id, key)
            # As the first charge's events: they carry the share, under the
            # key, only when it went to the running ancestor's live agent.
            state = agent_state.get(running) if owed else None
            owed = state is not None and state.agent is not None
            if owed:
                stale = _unpublished_keyed_charge(task_id, key, by_key=True)[0]
            _publish_rows(
                printer, task_id, tab_id, stale,
                {"cost": budget, "tokens": tokens, "steps": steps} if owed else None,
                key=key,
            )
            return
        updated, running, charge = charged
    else:
        updated, running = _add_late_task_usage(
            task_id, tokens, budget, steps, banked=task_agent is None,
        )
    _publish_charge(
        printer, task_agent, task_id, budget, tokens, steps, epoch, tab_id, updated, running,
        charge=charge, key=key,
    )


def _key_on_ledger(task_agent: Any, epoch: Any, key: str) -> bool:
    """Whether the ledger a keyed charge of *task_agent* is banked on holds *key*.

    That ledger is *epoch*, or without one *task_agent*'s current epoch
    (``RelentlessAgent._attribute_usage``).  A bank on it leaves no
    receipt in the row, but the row's final save writes a snapshot of
    it under ``TASK_USAGE_LOCK``, which the caller holds: the totals of
    a row finished since hold the key, where a second add would double
    it, and those of a running row will.
    """
    from kiss.agents.sorcar.relentless_agent import _ledger_totals_and_banked, _UsageLedger

    ledger = epoch if isinstance(epoch, _UsageLedger) else task_agent._usage_epoch()
    return _ledger_totals_and_banked(ledger, key)[3]


def _usage_and_stamp(agent: Any) -> tuple[tuple[float, int, int], list[float | int] | None]:
    """Return *agent*'s live totals and the ledger they rest on (``None`` without one).

    Called under the offsets' publish lock (``_OFFSET_PUBLISH_LOCK``).
    """
    from kiss.agents.sorcar.sorcar_agent import _live_agent_usage

    snapshot = getattr(agent, "usage_snapshot", None)
    stamp = list(cast("Iterable[float | int]", snapshot())) if callable(snapshot) else None
    return _live_agent_usage(agent), stamp


def _publish_charge(
    printer: Any,
    task_agent: Any,
    task_id: str,
    budget: float,
    tokens: int,
    steps: int,
    epoch: Any,
    tab_id: str,
    updated: list[tuple[str, int, float, int]],
    running: str,
    charge: int = 0,
    key: str | None = None,
) -> None:
    """Bank on the live agent what the rows did not take, and publish the new totals.

    The second half of :func:`_charge_locked` (``TASK_USAGE_LOCK``
    held): *updated* and *running* are what the row charge returned
    (:func:`~kiss.agents.sorcar.persistence._add_late_task_usage`), the
    other arguments :func:`charge_side_channel_usage`'s.  *charge* is
    the ``remote_charges`` id of what the running row owes (a remote
    sub-agent's growth, a keyed charge's share of a running ancestor),
    banked through :func:`_bank_remote_charge`; ``0`` for a spend
    banked on the ledger directly, under *key* when it has one.
    """
    from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK
    from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

    _race_delay()  # test hook: widens the row-check-to-bank window
    if not updated and not running:
        # No row for the task (its run persists none, or has not
        # saved one yet): the live agent is the only place the
        # spend can go, exactly as for an unfinished row.
        running = task_id
    banked = False
    if running:
        # The running task's final save waits for ``TASK_USAGE_LOCK``,
        # held here: it reads its counters with this bank on them.
        if running == task_id:
            agent = task_agent
        else:
            state = agent_state.get(running)
            agent = state.agent if state is not None else None
            epoch = None
        if agent is None:
            # No agent for the task's own row: the spend is already on
            # the counters its final save reads (see ``task_agent``).
            if running != task_id:
                log.warning(
                    "side-channel spend of task %s not charged to running "
                    "task %s: no live agent", task_id, running,
                )
        else:
            if charge:
                _bank_remote_charge(agent, charge, budget, tokens, steps)
            else:
                _attribute_sub_usage(agent, budget, tokens, steps, epoch=epoch, key=key)
            banked = True
            if printer is not None:
                # Publish the new totals (banked plus the in-flight
                # session's) as one of the running task's own
                # events, recorded and persisted like its other
                # usage events: the task may already be past its
                # last one, and both its replayed transcript and a
                # ``run_agent`` caller waiting on it
                # (daemon_client.run) take the cost from the latest.
                (live_budget, live_tokens, live_steps), stamp = locked_call(
                    _OFFSET_PUBLISH_LOCK, _usage_and_stamp, agent,
                )
                event = usage_totals_event(running, live_tokens, live_budget, live_steps)
                # The ledger these totals rest on (the in-flight session's
                # counters excluded), as a per-step event's stamp: a killed
                # task's orphan recovery adopts this event's totals and reads
                # from the stamp which remote charges they hold
                # (persistence._usage_event_offsets).  Both are read under
                # the publish lock every bank holds, so no bank lands
                # between them.
                if stamp is not None:
                    event["banked"] = stamp
                printer.broadcast(event)
    _publish_rows(
        printer, task_id, tab_id, updated,
        # The spend was banked on the running ancestor above, so a
        # ``run_agent`` caller still waiting on this row's task must
        # not fold it again (daemon_client._net_totals subtracts it).
        {"cost": budget, "tokens": tokens, "steps": steps} if banked else None,
        key=key,
    )


def _publish_rows(
    printer: Any,
    task_id: str,
    tab_id: str,
    updated: list[tuple[str, int, float, int]],
    ancestor_charged: dict[str, Any] | None,
    key: str | None = None,
) -> None:
    """Broadcast and persist a ``usage_info`` with each updated row's new totals.

    Args:
        printer: The daemon's printer, or ``None``.
        task_id: The task the charge was for (its tab is *tab_id*).
        tab_id: The tab of *task_id*.
        updated: ``(id, tokens, cost, steps)`` of each finished row charged.
        ancestor_charged: The charge banked on a running ancestor, which
            each event carries, or ``None``.
        key: The charge's key, which an event carrying *ancestor_charged*
            carries as ``charge_key``: a retry that publishes the event
            again (:func:`_charge_locked`) is then subtracted once by a
            ``run_agent`` caller (``daemon_client._net_totals``).
    """
    from kiss.agents.sorcar.persistence import _append_chat_event

    for row_id, row_tokens, row_cost, row_steps in updated:
        # ``taskId``: the tab that showed this row may already run a
        # newer task, whose cost this late event must not overwrite
        # (the chat page drops usage events of another task).
        event = usage_totals_event(row_id, row_tokens, row_cost, row_steps)
        if ancestor_charged is not None:
            event["ancestor_charged"] = dict(ancestor_charged)
            if key:
                event["charge_key"] = key
        stamp_event_ts(event)
        if printer is not None:
            printer.broadcast_transient(
                event, task_id=row_id, tab_id=tab_id if row_id == task_id else "",
            )
        _append_chat_event(dict(event), task_id=row_id)
    if updated and printer is not None:
        printer.broadcast({"type": "tasks_updated"})


REMOTE_USAGE_WINDOW_S = 7 * 86400.0
"""How long after its fold a sub-agent run on another machine is still asked for late spend."""

REMOTE_USAGE_FULL_SWEEP_S = 3600.0
"""Least seconds between the syncs that ask for every recorded sub-agent, however old.

A sub-agent's row can be charged long after its dispatch (an ``/ask``
about an old task); the syncs between sweeps ask only for those of the
last :data:`REMOTE_USAGE_WINDOW_S` seconds, so a busy history costs the
machines one full pass an hour at most.
"""

_last_full_sweep: dict[str, float] = {}
"""When each machine (by name) last got a full sweep (epoch seconds)."""

REMOTE_USAGE_BATCH = 500
"""The most task ids one ``getTaskUsage`` request carries (the machine answers for 500)."""

REMOTE_USAGE_NESTED_INTERVAL_S = 60.0
"""Least seconds between the syncs another daemon's ``getTaskUsage`` starts here.

Such a request starts a sync of this daemon's own remote sub-agents (a
nested dispatch: the asking daemon's next request sees their late
spend), which asks the machines in turn; the interval keeps two daemons
registered with each other from asking each other without end.
"""

_remote_sync_lock = threading.Lock()
"""Held by the one remote-usage sync running (:func:`start_remote_usage_sync`)."""

_last_sync_started = 0.0
"""When the last remote-usage sync started (epoch seconds; set with the lock held)."""


def start_remote_usage_sync(printer: Any, machines: list[Any], min_interval: float = 0.0) -> None:
    """Run :func:`sync_remote_subtask_usage` for each of *machines* on a thread of its own.

    Called on every first-page refresh of the CHATS panel and every
    Spend refresh, which must not wait on the machines, and, with
    *min_interval* :data:`REMOTE_USAGE_NESTED_INTERVAL_S`, when another
    daemon asks for usage (``getTaskUsage``).  A call that comes while a
    sync still runs, or within *min_interval* seconds of the last one's
    start, starts none: a later one catches up.
    """
    global _last_sync_started
    if not machines or not _remote_sync_lock.acquire(blocking=False):
        return
    if min_interval > 0 and time.time() - _last_sync_started < min_interval:
        _remote_sync_lock.release()
        return
    _last_sync_started = time.time()
    try:
        threading.Thread(
            target=_sync_machines, args=(printer, machines), name="remote-usage-sync", daemon=True,
        ).start()
    except RuntimeError:  # pragma: no cover — thread exhaustion
        _remote_sync_lock.release()


def _sync_machines(printer: Any, machines: list[Any]) -> None:
    """Sync each of *machines* in turn, then let the next sync start."""
    try:
        for machine in machines:
            sync_remote_subtask_usage(printer, machine)
    finally:
        _remote_sync_lock.release()


def sync_remote_subtask_usage(printer: Any, machine: Any) -> None:
    """Charge local parents the spend their sub-agents on *machine* got after the fold.

    A ``run_agent(machine=...)`` sub-agent's row lives on the machine;
    the local parent folds the totals the dispatch returns with and
    records them (``remote_subtasks``,
    :func:`~kiss.agents.sorcar.persistence._record_remote_subtask`).
    Once the dispatch returned, nothing reaches this daemon from that
    run any more: its stream is unbound, and on the machine a late
    charge to the row (an ``/ask`` answer or task update finishing
    after it) stops at the parent, which has no row there.  So the
    machine is asked (``getTaskUsage``) for the current totals of the
    sub-agents recorded in the last :data:`REMOTE_USAGE_WINDOW_S`
    seconds (every recorded one, at most once per
    :data:`REMOTE_USAGE_FULL_SWEEP_S`), and each one's growth over its record is charged to its
    parent (:func:`charge_remote_subtask_usage`).  A machine that
    cannot be reached, or an older daemon that does not know the
    command, charges nothing now; a later sync catches up.
    """
    from kiss.agents.sorcar.machines import MachineError, link_for
    from kiss.agents.sorcar.persistence import _remote_subtask_ids

    now = time.time()
    full = now - _last_full_sweep.get(machine.name, 0.0) >= REMOTE_USAGE_FULL_SWEEP_S
    ids = _remote_subtask_ids(machine.name, 0.0 if full else now - REMOTE_USAGE_WINDOW_S)
    # In batches of the most task ids one getTaskUsage answers for, so no
    # record is left out however many there are.
    for start in range(0, len(ids), REMOTE_USAGE_BATCH):
        batch = ids[start:start + REMOTE_USAGE_BATCH]
        try:
            reply = link_for(machine).request(
                {"type": "getTaskUsage", "taskIds": batch}, "taskUsage", errors=True,
            )
        except MachineError as exc:
            log.info("late spend of the sub-agents on %s not synced: %s", machine.name, exc)
            return
        tasks = reply.get("tasks")
        if not isinstance(tasks, dict):
            return
        for task_id in batch:
            usage = tasks.get(task_id)
            if isinstance(usage, dict):
                charge_remote_subtask_usage(printer, task_id, usage)
    if full:  # only a sweep that reached the end counts: a failed one is retried
        _last_full_sweep[machine.name] = now


def charge_remote_subtask_usage(printer: Any, task_id: str, usage: dict[str, Any]) -> None:
    """Charge the parent of *task_id*, a sub-agent on another machine, its spend not yet folded.

    *usage* is the machine's report of the sub-agent's row: cumulative
    ``cost``, ``tokens`` and ``steps`` and their split by model
    (``models``).  Its growth over the recorded totals goes to the
    parent the way a side channel's spend does
    (:func:`charge_side_channel_usage`): added to its row and its
    finished ancestors' in the transaction that moves the record up
    (:func:`~kiss.agents.sorcar.persistence._charge_remote_subtask`,
    so a report seen twice charges once, one equal to the fold charges
    nothing, and a daemon that dies mid-way charges it next time), and
    banked on its live agent while it runs.  What a running row owes
    is recorded in that transaction too and banked through
    :func:`_bank_remote_charge`, so that the row's final save or orphan
    recovery settles it into the row exactly once, whatever the daemon
    lived through.  Failures are logged, never raised.
    """
    try:
        locked_call(TASK_USAGE_LOCK, _charge_remote_locked, printer, task_id, usage)
    except Exception:
        log.warning("late spend of remote sub-agent %s not charged", task_id, exc_info=True)


def _charge_remote_locked(printer: Any, task_id: str, usage: dict[str, Any]) -> None:
    """:func:`charge_remote_subtask_usage` with ``TASK_USAGE_LOCK`` held."""
    from kiss.agents.sorcar.persistence import _charge_remote_subtask, _safe_float, _safe_int

    charged = _charge_remote_subtask(
        task_id, _safe_float(usage.get("cost")), _safe_int(usage.get("tokens")),
        _safe_int(usage.get("steps")), usage.get("models"),
    )
    if charged is None:
        return
    parent_id, cost, tokens, steps, updated, running, charge = charged
    if cost <= 0 and tokens <= 0 and steps <= 0:
        return
    state = agent_state.get(parent_id)
    _publish_charge(
        printer, state.agent if state is not None else None, parent_id, cost, tokens, steps,
        None, "", updated, running, charge=charge,
    )


def _bank_remote_charge(agent: Any, charge: int, budget: float, tokens: int, steps: int) -> None:
    """Bank remote charge *charge* (its growth *budget*, *tokens*, *steps*) on *agent*'s ledger.

    The charge is marked banked first, with the ledger totals it is
    banked over (:func:`~kiss.agents.sorcar.persistence._mark_remote_charge_banked`),
    and the mark, the bank and its publication of the agent's offsets
    happen under the offsets' publish lock: no other publication of
    the task's totals lands in between, so every usage event the task
    persists carries offsets at most those totals (published before)
    or at least them plus the charge (published after).  That is what
    lets the row's orphan recovery tell, from its last persisted usage
    event alone, whether the event's totals hold the charge
    (:func:`~kiss.agents.sorcar.persistence._charges_held_by_usage_event`),
    whenever the process died: before the mark (unmarked, never
    banked), between the mark and the bank (marked, no event holds
    it) or after the bank.  The ledger counts the charge's key once,
    and the task's final save, a snapshot of the ledger, holds every
    marked charge and no other.  ``TASK_USAGE_LOCK`` is held, as for
    every bank of a late charge.  The publish lock is taken in slices:
    the caller may be a task thread (a keyed charge's share of a
    running ancestor), which a Stop must reach while it waits.

    Args:
        agent: The running row's live agent (a ``RelentlessAgent``).
        charge: The ``remote_charges`` id of the growth.
        budget: The growth's USD.
        tokens: The growth's tokens.
        steps: The growth's steps.
    """
    from kiss.agents.sorcar import sorcar_agent
    from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK

    locked_call(
        _OFFSET_PUBLISH_LOCK, sorcar_agent._bank_remote_charge,
        agent, charge, budget, tokens, steps,
    )


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

    The child runs on the update thread, which is bound first to a stop
    event of its own that observes the task's
    (:class:`_ChildStopEvent`): a Stop on the task's tab reaches the
    child the way it reaches a ``run_parallel`` child (every stop check
    of the child's loop polls that event), so a stopped task never
    leaves its update child calling the model until its budget cap;
    a Stop on the child's own nested tab sets only the child's event,
    so the task and its other children keep running.

    Its spend is charged to the task whatever way the run ends.  While
    the task still runs it is banked on *parent_agent*'s live counters,
    bound to the usage epoch captured at the start (a parent that has
    since reset for its next task discards it instead of billing that
    task); the task's final save then writes those totals to its row.
    When the task finished while the update ran (its row already holds
    the final totals) the spend is added to the row and its finished
    ancestors instead (see :func:`charge_side_channel_usage`), so it is
    counted once, whenever the update ends.

    Args:
        parent_agent: The agent of the running task to report on.
        task_id: That task's persisted ``task_history`` row id.

    Returns:
        ``(answer, cost)``: the child's ``finish`` summary and the USD
        it spent.

    Raises:
        Exception: Whatever the child's run raised (after the spend was
            attributed), with that spend in USD as its
            ``task_update_cost`` attribute.
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
    # Before the child is built: the agent captures its thread's event
    # at construction for its tools, and the printer bridge records it
    # as the child's own registry state's ``stop_event`` — the event a
    # Stop on the child's nested tab sets, so it must not be the
    # parent's own object.  Only the registry's event for the task (the
    # one ``_stop_task`` sets, renewed by the task runner for every run
    # of the agent) qualifies as the parent's: a task no longer
    # registered has nothing to stop, and its agent's current
    # ``_stop_event`` may already belong to the next run of a reused
    # agent (one kept for a pending worktree merge).
    state = agent_state.get(task_id)
    stop_signal.set_thread_stop_event(
        _ChildStopEvent(state.stop_event if state is not None else None)
    )
    # The tab the chat pages show the parent under — for a sub-agent
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
            llm_response_hook=ask.llm_response_hook,
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
            compaction=ask_settings.get("compaction", True),
        )
    except BaseException as exc:
        # A stopped or failed run spent too: the panel shows it.
        exc.task_update_cost = _live_agent_usage(agent)[0]  # type: ignore[attr-defined]
        raise
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        try:
            charge_side_channel_usage(
                printer, parent_agent, task_id, budget, tokens, steps, epoch=epoch,
            )
        finally:
            # Whatever the charge did: without ``subagentDone`` every
            # surface watching the parent keeps the child's nested tab
            # open, spinner and all.
            if printer is not None:
                _notify_subagent_done(
                    printer, _persisted_task_id(agent), sub_tab_id, model_name,
                    parent_tab_id,
                )
    return _extract_result_summary(result), budget


class _ChildStopEvent(threading.Event):
    """The stop event of an in-process child: its own, observing its parent's.

    ``set`` (a Stop on the child's nested tab) sets only this event, so
    the parent task and its other children keep running; ``is_set``
    and ``wait`` also report the *parent* event once set (a Stop on the
    task's tab), so the child ends with its task.  ``wait`` polls the
    parent in bounded slices of :data:`_PARENT_STOP_POLL_S`, since a
    ``set`` of the parent never wakes this event's own condition.

    Args:
        parent: The parent task's stop event, or ``None`` when the task
            is no longer registered (nothing of it to observe).
    """

    def __init__(self, parent: threading.Event | None) -> None:
        super().__init__()
        self._parent = parent

    def is_set(self) -> bool:
        """Return whether this event or the parent's is set."""
        return super().is_set() or (self._parent is not None and self._parent.is_set())

    def wait(self, timeout: float | None = None) -> bool:
        """Block until this event or the parent's is set, or *timeout* seconds pass.

        Args:
            timeout: Seconds to wait at most; ``None`` waits until set.

        Returns:
            Whether the event (own or parent's) is set.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            slice_s = _PARENT_STOP_POLL_S
            if deadline is not None:
                slice_s = min(slice_s, deadline - time.monotonic())
                if slice_s <= 0:
                    return False
            super().wait(slice_s)
        return True


class TaskUpdateRunner:
    """Keeps one :class:`TaskUpdate` per task and schedules the agent runs.

    Args:
        run_sea: Runs the agent for ``(parent_agent, task_id)`` and
            returns ``(answer, cost)``; :func:`run_task_update_sea` by
            default.  A run that raises reports its cost as the
            exception's ``task_update_cost`` (0 when it has none).
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
        # The task's start is its own ``task_history`` row's, which
        # every run allocates with its own start (the agent's
        # ``_task_start_ms`` would date a later block of a sequential
        # submission by the first one); unknown, it is now.  The read
        # blocks on the history database's lock, outside ``_lock``.
        start_ms = task_start_ms(task_id)
        started_at = start_ms / 1000.0 if start_ms > 0 else now
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
        except KeyboardInterrupt as exc:
            # The task (or this update alone) was stopped while the update
            # ran: the child's stop event observes the task's and ends with it.
            error = "The task was stopped before this update finished."
            cost = getattr(exc, "task_update_cost", 0.0)
        except BaseException as exc:
            # ``BaseException``: a cancelled asyncio call or a
            # ``SystemExit`` is a failed update too, not one recorded
            # below without an error; the thread ends here either way.
            error = f"{type(exc).__name__}: {exc}"
            cost = getattr(exc, "task_update_cost", 0.0)
            log.warning("task update for %s failed", task_id, exc_info=True)
        finally:
            # In a ``finally``: an update pinned on ``running`` would
            # never be restarted by ``poll`` nor pruned.
            with self._lock:
                upd = self._updates.setdefault(task_id, TaskUpdate())
                if text:
                    upd.text = text
                upd.error = error
                upd.cost = cost
                upd.running = False
                upd.finished_at = time.time()
