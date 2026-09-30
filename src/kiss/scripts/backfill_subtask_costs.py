# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Re-add sub-task spend that was never folded into its parent in ``sorcar.db``.

Usage::

    uv run python -m kiss.scripts.backfill_subtask_costs [--db PATH] [--apply]
        [--min-shortfall USD]

A task's ``task_history.cost`` and ``tokens`` include the spend of its
sub-tasks (``run_parallel`` sub-agents, ``run_agent`` sub-tasks, ``/ask``
and status side channels), and the Spend panel counts top-level rows
only, scaling a task's sub-tasks down when they add up to more than the
task itself.  Before the fixes of 2026-09-29 a parent that was stopped
or interrupted while waiting on a ``run_agent`` sub-task was never
charged for it (task ``bc92cb57``: $0.52 on the row, $56.48 in its
sub-task), and a finished sub-task's classifier spend was left out.
Such rows are short, and the Spend panel under-reports the day.

For every finished task that has sub-tasks, children first::

    own      = sum of the task's own ``llm_call`` events (cost, tokens)
    expected = own + sum over its sub-tasks' (repaired) totals
    new      = max(current, expected)

``own`` is 0 for tasks that predate the ``llm_call`` event, and the
task classifier's spend has no event, so the repaired figure is a lower
bound: it never exceeds the true spend.  ``tokens`` are raised the same
way, but only on rows whose cost is short: a missing fold loses both,
while a token-only gap has another cause.  ``steps`` are left alone.

A task is left alone while it, or one of its sub-tasks, is still
running or ended less than ten minutes ago: a sub-task's row is saved
a moment before its spend is charged to the parent, and repairing the
parent in that gap would count the sub-task twice.  A row that never
ended but started over a day ago is treated as dead, not running: its
agent died before saving, and no fold will follow.  A task on a
``parent_task_id`` cycle is left alone too, since it lists an ancestor
among its sub-tasks.  Both are reported.

The script prints what it would change and writes nothing unless
``--apply`` is given.  With ``--apply`` each row is updated only if it
still holds the values that were read, so a task the daemon finishes
meanwhile is skipped and reported; running the script again is safe
and, once every row is repaired, changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

from kiss.core.config import kiss_home

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read", "cache_write")
"""``llm_call`` event fields whose sum is the call's share of ``tokens``."""

SETTLE_MS = 10 * 60 * 1000
"""How long after a row ends its spend may still be folded into its parent."""

STALE_MS = 24 * 60 * 60 * 1000
"""Age after which a row that never ended counts as dead rather than running."""


@dataclass
class Repair:
    """One ``task_history`` row whose totals fall short of its sub-tasks'."""

    task_id: str
    timestamp: float
    task: str
    children: int
    cost: float
    new_cost: float
    tokens: int
    new_tokens: int

    @property
    def added_cost(self) -> float:
        """USD the repair adds to the row."""
        return self.new_cost - self.cost


def _num(value: object) -> float:
    """Return *value* as a float, 0.0 when it is missing or not a number."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _own_spend(conn: sqlite3.Connection) -> dict[str, tuple[float, int]]:
    """Sum each task's own ``llm_call`` events into ``(cost, tokens)``.

    Args:
        conn: Connection to the ``sorcar.db`` database.

    Returns:
        Task id to the cost and tokens of the LLM calls the task made
        itself (its sub-tasks' calls are recorded under their own ids).
        Tasks without ``llm_call`` events are absent.
    """
    own: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    rows = conn.execute(
        "SELECT task_id, event_json FROM events WHERE event_json LIKE '%\"llm_call\"%'"
    )
    for task_id, event_json in rows:
        try:
            event = json.loads(event_json)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "llm_call":
            continue
        acc = own[str(task_id)]
        acc[0] += _num(event.get("cost"))
        acc[1] += sum(_num(event.get(field)) for field in TOKEN_FIELDS)
    return {task_id: (cost, int(tokens)) for task_id, (cost, tokens) in own.items()}


def _children_first(
    task_ids: list[str], children: dict[str, list[str]],
) -> list[str]:
    """Order *task_ids* so every task comes after all of its sub-tasks.

    Args:
        task_ids: Every ``task_history`` id.
        children: Task id to the ids of its sub-tasks.

    Returns:
        The ids in post-order; a ``parent_task_id`` cycle is cut at the
        first id seen twice on the current path.
    """
    order: list[str] = []
    done: set[str] = set()
    for start in task_ids:
        on_path: set[str] = set()
        stack = [(start, False)]
        while stack:
            task_id, expanded = stack.pop()
            if expanded:
                on_path.discard(task_id)
                done.add(task_id)
                order.append(task_id)
                continue
            if task_id in done or task_id in on_path:
                continue
            on_path.add(task_id)
            stack.append((task_id, True))
            stack.extend((kid, False) for kid in children.get(task_id, ()))
    return order


def _on_parent_cycle(parent_of: dict[str, str]) -> set[str]:
    """Return the task ids whose ``parent_task_id`` chain leads back to them.

    Such a task counts an ancestor among its sub-tasks, so folding the
    sub-tasks into it would grow both rows on every run.

    Args:
        parent_of: Task id to its ``parent_task_id`` ('' at the top).

    Returns:
        The ids that sit on a cycle.
    """
    cyclic: set[str] = set()
    for start in parent_of:
        seen: set[str] = set()
        task_id = parent_of[start]
        while task_id in parent_of and task_id not in seen:
            if task_id == start:
                cyclic.add(start)
                break
            seen.add(task_id)
            task_id = parent_of[task_id]
    return cyclic


def _settled(row: sqlite3.Row, now_ms: float) -> bool:
    """Whether *row* ended more than :data:`SETTLE_MS` ago.

    A sub-task's row is saved a moment before its spend is charged to
    the parent, and a task that has not ended is charged live, so
    rows younger than the window may still receive a legitimate fold.
    A row without ``end_ts`` that started over :data:`STALE_MS` ago is
    an orphan whose agent died before saving, not a running task.
    """
    end_ts = float(row["end_ts"])
    if not end_ts:
        return float(row["timestamp"] or 0) * 1000.0 <= now_ms - STALE_MS
    return end_ts <= now_ms - SETTLE_MS


def plan_repairs(
    conn: sqlite3.Connection, min_shortfall: float, now_ms: float | None = None,
) -> tuple[list[Repair], int, int]:
    """Find the settled tasks whose totals fall short of their sub-tasks'.

    Args:
        conn: Connection to the ``sorcar.db`` database.
        min_shortfall: Smallest missing cost, in USD, worth repairing;
            must be positive so a repair always raises the row.
        now_ms: The current time in epoch milliseconds (defaults to
            the wall clock); rows that ended within :data:`SETTLE_MS`
            before it are left alone.

    Returns:
        ``(repairs, unsettled, cyclic)``: the rows to update, children
        before parents; the number of short tasks skipped because they
        or one of their sub-tasks have not ended or ended within the
        settle window; and the number skipped because they sit on a
        ``parent_task_id`` cycle.
    """
    if now_ms is None:
        now_ms = time.time() * 1000.0
    rows = {
        str(row["id"]): row
        for row in conn.execute(
            "SELECT id, COALESCE(parent_task_id, '') AS parent, timestamp, task, "
            "COALESCE(cost, 0.0) AS cost, COALESCE(tokens, 0) AS tokens, "
            "COALESCE(end_ts, 0) AS end_ts FROM task_history"
        )
        if row["id"]
    }
    parent_of = {task_id: str(row["parent"]) for task_id, row in rows.items()}
    children: dict[str, list[str]] = defaultdict(list)
    for task_id, parent in parent_of.items():
        if parent:
            children[parent].append(task_id)
    cyclic_ids = _on_parent_cycle(parent_of)
    own = _own_spend(conn)
    cost_now = {task_id: max(float(row["cost"]), 0.0) for task_id, row in rows.items()}
    tokens_now = {task_id: max(int(row["tokens"]), 0) for task_id, row in rows.items()}
    repairs: list[Repair] = []
    unsettled = cyclic = 0
    for task_id in _children_first(list(rows), children):
        kids = children.get(task_id)
        if not kids:
            continue
        row = rows[task_id]
        own_cost, own_tokens = own.get(task_id, (0.0, 0))
        expected_cost = own_cost + sum(cost_now[kid] for kid in kids)
        expected_tokens = own_tokens + sum(tokens_now[kid] for kid in kids)
        new_cost = max(cost_now[task_id], round(expected_cost, 6))
        if new_cost - cost_now[task_id] < min_shortfall:
            continue
        if task_id in cyclic_ids:
            cyclic += 1
            continue
        if not all(_settled(rows[t], now_ms) for t in (task_id, *kids)):
            unsettled += 1
            continue
        repair = Repair(
            task_id=task_id,
            timestamp=float(row["timestamp"] or 0),
            task=str(row["task"] or ""),
            children=len(kids),
            cost=float(row["cost"]),
            new_cost=new_cost,
            tokens=int(row["tokens"]),
            new_tokens=max(tokens_now[task_id], expected_tokens),
        )
        repairs.append(repair)
        cost_now[task_id] = repair.new_cost
        tokens_now[task_id] = repair.new_tokens
    return repairs, unsettled, cyclic


def apply_repairs(conn: sqlite3.Connection, repairs: list[Repair]) -> list[Repair]:
    """Write *repairs* to ``task_history`` in one transaction.

    Args:
        conn: Read-write connection to the ``sorcar.db`` database.
        repairs: The rows to update, as returned by :func:`plan_repairs`.

    Returns:
        The repairs that were NOT written because the row no longer
        holds the ``cost`` and ``tokens`` that were read (the daemon
        updated it meanwhile); every other repair is committed.
    """
    skipped: list[Repair] = []
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        for repair in repairs:
            cursor = conn.execute(
                "UPDATE task_history SET cost = ?, tokens = ? "
                "WHERE id = ? AND COALESCE(cost, 0.0) = ? AND COALESCE(tokens, 0) = ?",
                (repair.new_cost, repair.new_tokens, repair.task_id, repair.cost, repair.tokens),
            )
            if (cursor.rowcount or 0) == 0:
                skipped.append(repair)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return skipped


def _format(repair: Repair) -> str:
    """One report line for *repair*."""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(repair.timestamp))
    title = " ".join(repair.task.split())[:60]
    return (
        f"{repair.task_id[:8]} {when}  ${repair.cost:.4f} -> ${repair.new_cost:.4f}"
        f"  (+${repair.added_cost:.4f}, {repair.children} sub-task"
        f"{'s' if repair.children != 1 else ''})  {title}"
    )


def main(argv: list[str] | None = None) -> int:
    """Report, and with ``--apply`` write, the missing sub-task spend.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when the database cannot be
        opened.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", default=str(kiss_home() / "sorcar.db"))
    parser.add_argument(
        "--apply", action="store_true",
        help="write the repaired totals; without it the script only reports them",
    )
    parser.add_argument(
        "--min-shortfall", type=float, default=0.0001, metavar="USD",
        help="ignore rows short by less than this much (default: 0.0001)",
    )
    args = parser.parse_args(argv)
    if not args.min_shortfall > 0:
        parser.error("--min-shortfall must be a positive number")
    mode = "rw" if args.apply else "ro"
    try:
        conn = sqlite3.connect(f"file:{args.db}?mode={mode}", uri=True, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM task_history LIMIT 1")
    except sqlite3.Error as exc:
        print(f"cannot open {args.db}: {exc}", file=sys.stderr)
        return 1
    repairs, unsettled, cyclic = plan_repairs(conn, args.min_shortfall)
    for repair in repairs:
        print(_format(repair))
    total = sum(repair.added_cost for repair in repairs)
    skipped_notes = []
    if unsettled:
        skipped_notes.append(f"{unsettled} skipped as running or ended in the last 10 minutes")
    if cyclic:
        skipped_notes.append(f"{cyclic} skipped as sitting on a parent_task_id cycle")
    print(
        f"{len(repairs)} task{'s' if len(repairs) != 1 else ''} short by "
        f"${total:,.4f} in total" + "".join(f"; {note}" for note in skipped_notes)
    )
    if not args.apply:
        if repairs:
            print("dry run: re-run with --apply to write these totals")
        conn.close()
        return 0
    skipped = apply_repairs(conn, repairs)
    conn.close()
    written = len(repairs) - len(skipped)
    added = total - sum(repair.added_cost for repair in skipped)
    print(f"updated {written} row{'s' if written != 1 else ''} (+${added:,.4f})")
    if skipped:
        print(
            f"skipped {len(skipped)} row{'s' if len(skipped) != 1 else ''} that changed "
            "meanwhile; run again to repair them: "
            + ", ".join(repair.task_id[:8] for repair in skipped)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
