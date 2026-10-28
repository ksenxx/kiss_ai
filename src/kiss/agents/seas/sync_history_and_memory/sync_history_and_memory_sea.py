# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sync-history-and-memory agent — makes the task history and the memory the same on every machine.

Typing ``/sync_history_and_memory`` in the chat (or
``run_agent(agent="sync_history_and_memory", task="...")``) brings
``$KISS_HOME/history.db`` (tasks, their events, deletions) and
``$KISS_HOME/memories`` (pages and tombstones) of this machine and of every
machine registered with ``/machine add`` to the same state, in one run,
while tasks keep running everywhere: a task in flight is left for the
next run (:mod:`kiss.agents.seas.sync_history_and_memory.peer_sync`).

The agent is a thin worker around one deterministic tool,
:func:`sync_history_and_memory`, the way ``/sync`` wraps ``git_sync``:
it calls the tool once (``machines`` naming a subset when the message
does, ``dry_run`` when the message asks for a dry run or a preview),
and reports the tool's output.  The methods follow the SEA contract in
:mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync_history_and_memory.peer_sync import PeerReport, PeerSync
from kiss.agents.sorcar.machines import load_machines

MAX_BUDGET_USD = 1.0
"""Spending cap: a sync is one tool call and one short report."""

ADD_TO_PROMPT = (
    "Sync the task history and the memory of this machine with every registered machine."
)
"""Appended to every task: a bare ``/sync_history_and_memory`` runs on this text alone."""

SYSTEM_PROMPT = """\
You are the history-and-memory sync agent. You bring the task history (history.db)
and the agent memory (memories/) of this machine and of the machines registered with
`/machine add` to the same state. Nothing else.

1. Call `sync_history_and_memory` exactly once. Pass `machines` as a comma-separated
   list of machine names only when the message names specific machines; leave it
   empty for all of them. Pass `dry_run=True` only when the message asks for a dry run
   or a preview. Never run ssh, sqlite3, rsync, scp or any script yourself.
2. Call `finish` with a short HTML summary that repeats the tool's report: one line per
   machine saying what travelled each way (tasks, events, pages), the tasks held back
   because they are still running, the pages in conflict (the same page edited at the
   same moment on both sides, kept as it was on each), and any machine that failed with
   the reason given. Do not retry a failed machine and do not work around a failure.
"""
"""The whole base system prompt of the agent (replaces ``SYSTEM.md``)."""


def sync_history_and_memory(machines: str = "", dry_run: bool = False) -> str:
    """Bring the task history and the memory of this and the registered machines to the same state.

    Everything the other machines have is pulled here first, then this
    machine -- now holding the union -- is pushed to each of them, so one
    call leaves all machines equal.  Tasks still running anywhere are left
    for the next call.  Machines are worked on concurrently, each over one
    multiplexed ssh connection; only rows and pages a side lacks travel.

    Args:
        machines: Comma-separated registered machine names to sync with;
            empty for every registered machine.
        dry_run: Report what would travel without changing any machine.

    Returns:
        A report with one block per machine: history rows pulled and
        pushed, memory pages pulled and pushed, tasks held back, page
        conflicts, time taken, or the reason the machine was skipped.
    """
    registered = load_machines()
    if machines.strip():
        wanted = [name.strip() for name in machines.split(",") if name.strip()]
        known = {machine.name: machine for machine in registered}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            return "Not registered: " + ", ".join(unknown) + " (see `/machine list`)."
        registered = [known[name] for name in wanted]
    if not registered:
        return "No machine is registered (`/machine add user@host`); nothing to sync."
    return format_reports(PeerSync(registered, dry_run=dry_run).run(), dry_run)


def format_reports(reports: list[PeerReport], dry_run: bool) -> str:
    """Render the per-machine reports as the text the agent relays.

    Args:
        reports: One per machine, as :meth:`PeerSync.run` returns them.
        dry_run: Whether the counts are what would have travelled.
    """
    lines = [
        "DRY RUN: nothing was changed. The counts compare each machine with this one as it "
        "is now; a real run pushes each machine what the others bring as well."
        if dry_run
        else f"Synced {sum(1 for r in reports if not r.error)} of {len(reports)} machine(s)."
    ]
    for report in reports:
        lines.append("")
        lines.append(
            f"{report.name} ({report.ssh or 'no ssh'}): "
            + (f"FAILED: {report.error}" if report.error else f"ok in {report.seconds:.1f}s")
        )
        if report.pulled or report.pushed:
            lines.append(
                f"  history: pulled {_history(report.pulled)}; pushed {_history(report.pushed)}"
            )
        if report.memory_pulled or report.memory_pushed:
            lines.append(
                f"  memory: pulled {_memory(report.memory_pulled)}; "
                f"pushed {_memory(report.memory_pushed)}"
            )
        if report.held_back_there:
            there = ", ".join(report.held_back_there)
            lines.append(f"  still running there, left for the next sync: {there}")
        if report.held_back_here:
            lines.append(
                f"  still running here, left for the next sync: {', '.join(report.held_back_here)}"
            )
        if report.conflicts:
            lines.append(
                f"  conflicts (same page edited at the same moment on both sides, each side kept "
                f"its own): {', '.join(report.conflicts)}"
            )
        if abs(report.skew) >= 1.0:
            lines.append(f"  clock skew: {report.skew:+.1f}s (its clock minus ours)")
    return "\n".join(lines)


def _history(stats: dict[str, Any]) -> str:
    """One phrase for a ``sync_db`` statistics dict."""
    if not stats:
        return "nothing (not reached)"
    if "skipped" in stats:
        return f"nothing ({stats['skipped']})"
    return (
        f"{stats.get('tasks_inserted', 0)} task(s), {stats.get('events_inserted', 0)} event(s), "
        f"{stats.get('deletions_recorded', 0)} deletion(s), {stats.get('tasks_deleted', 0)} removed"
    )


def _memory(stats: dict[str, Any]) -> str:
    """One phrase for a memory transfer: entries sent, plus the merge counts when it merged."""
    if not stats:
        return "nothing (not reached)"
    counts = stats.get("counts")
    if counts is None:
        return f"{stats.get('entries', 0)} entry(ies)"
    return (
        f"{stats.get('entries', 0)} entry(ies): {counts['added']} added, "
        f"{counts['updated']} updated, {counts['removed']} removed, {counts['kept']} kept, "
        f"{counts['conflicts']} in conflict"
    )


class SyncHistoryAndMemorySea(WorkerSea):
    """The ``/sync_history_and_memory`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/sync_history_and_memory help``."""
        return (
            "Brings the task/chat history (history.db) and the agent memory (memories/) of this "
            "machine and of every registered machine (`/machine list`) to the same state in one "
            "run, while tasks keep running (a task in flight waits for the next run); use "
            "`/sync_history_and_memory` in the chat (`/sync_history_and_memory dry run` to "
            "preview, `/sync_history_and_memory <machine>` for one machine) or "
            'run_agent(agent="sync_history_and_memory", task="").'
        )

    def prompt(self, task: str) -> str:
        """Return the message followed by :data:`ADD_TO_PROMPT`."""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with no tools but its own, :data:`MAX_BUDGET_USD` for the run.

        ``tool_profile`` is ``locked`` at ``none``: the sync is one call
        of :func:`sync_history_and_memory` and the agent needs no shell,
        file or browser tool for it.
        """
        return settings | {
            "tool_profile": "none",
            "max_budget": MAX_BUDGET_USD,
            "locked": ["tool_profile"],
        }

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* plus :func:`sync_history_and_memory`."""
        return [*tools, sync_history_and_memory]
