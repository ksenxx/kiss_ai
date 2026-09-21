# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Token-cost KPIs of the tasks recorded in a ``sorcar.db``.

The 7-day efficiency audit of 2026-09-19 found that most avoidable spend
came from a handful of measurable patterns: a mandatory first-step
``Read("./SORCAR.md")``, files re-read within the same task, steps run
at very large contexts, sub-agents that start with a ~12k-token first
step, reviewer sub-trees that exceed half of a task's spend, LLM
sub-agents used as shell wrappers, and context hand-offs at 90 % of the
window.  This script measures each of them for a time window so the
levers in ``projects/cost-levers-implementation-plan.md`` can be checked
against real data before and after they land.

Usage::

    uv run python -m kiss.scripts.cost_report [--db PATH] [--hours 24] [--json]

Only the ``task_history`` and ``events`` tables are read; nothing is
written.  A missing or foreign event field contributes nothing to the
figure it belongs to.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.fanout_guard import is_review_task
from kiss.core.config import DEFAULT_CONFIG, kiss_home

CONTEXT_BUCKETS: tuple[tuple[str, int], ...] = (
    ("<50k", 50_000),
    ("50-100k", 100_000),
    ("100-200k", 200_000),
    ("200-300k", 300_000),
    (">300k", 1 << 62),
)
FANOUT_TOOLS = frozenset({"run_parallel", "run_agent"})
HANDOFF_CONTEXT_FRACTION = DEFAULT_CONFIG.context_limit_fraction * 0.95
"""A session restart whose last context was at or above this share of the
window counts as a context hand-off (as opposed to an is_continue restart
for another reason).  The agent hands off at
``DEFAULT_CONFIG.context_limit_fraction`` (0.7 by default); the 5 % margin
absorbs the round-off of the last reported context so a hand-off at exactly
the threshold is still counted."""

_CONTEXT_RE = re.compile(r"Context:\s*([\d,]+)\s*/\s*([\d,]+)")
_SHELL_WRAPPER_RE = re.compile(
    r"^\s*run\b.{0,120}\b(command|pytest|bash|shell|split)\b", re.IGNORECASE | re.DOTALL
)
SHELL_WRAPPER_MAX_STEPS = 8


@dataclass
class TaskRow:
    """One ``task_history`` row plus the per-task figures derived from its events."""

    id: str
    parent_id: str
    task: str
    model: str
    cost: float
    tokens: int
    steps: int
    timestamp: float
    in_window: bool = True
    """False for an ancestor loaded only to reconstruct a tree."""
    sorcar_md_reads: int = 0
    reads: int = 0
    repeat_reads: int = 0
    first_context: int | None = None
    prompt_events: int = 0
    handoffs: int = 0
    bucket_steps: dict[str, int] = field(default_factory=dict)
    bucket_cost: dict[str, float] = field(default_factory=dict)
    steps_with_cache_field: int = 0
    steps_with_cache_hit: int = 0


def _money(text: Any) -> float:
    """Parse the ``$1.2345`` cost string of a ``usage_info`` event."""
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def _bucket(context_tokens: int) -> str:
    for name, upper in CONTEXT_BUCKETS:
        if context_tokens < upper:
            return name
    return CONTEXT_BUCKETS[-1][0]  # pragma: no cover — sentinel bucket is unbounded


def _is_sorcar_md_read(event: dict[str, Any]) -> bool:
    path = str(event.get("path") or event.get("file_path") or "")
    return path.replace("\\", "/").rstrip("/").endswith("SORCAR.md") and "/.kiss/" not in path


def _scan_events(conn: sqlite3.Connection, rows: dict[str, TaskRow]) -> None:
    """Fill the per-task event figures of every task in *rows*."""
    if not rows:
        return
    placeholders = ",".join("?" * len(rows))
    cur = conn.execute(
        f"SELECT task_id, event_json FROM events WHERE task_id IN ({placeholders}) "
        "ORDER BY task_id, seq",
        list(rows),
    )
    seen_reads: dict[str, set[tuple[str, Any, Any]]] = defaultdict(set)
    last_cost: dict[str, float] = {}
    last_steps: dict[str, int] = {}
    # Peak (context, window) of the current session of each task.  A
    # hand-off runs a small trajectory-summarizer session before the next
    # ``prompt`` event, so the *last* context before a restart is not the
    # one that triggered it; the session's peak is.
    peak_context: dict[str, tuple[int, int]] = {}
    # The tool called last in each task: a usage_info right after a
    # run_parallel / run_agent carries the children's spend too, which
    # must not be attributed to the parent's context bucket.
    last_tool: dict[str, str] = {}
    for task_id, event_json in cur:
        try:
            event = json.loads(event_json)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        row = rows[task_id]
        kind = event.get("type")
        if kind == "tool_call":
            last_tool[task_id] = str(event.get("name", ""))
        if kind == "tool_call" and event.get("name") == "Read":
            row.reads += 1
            path = str(event.get("path") or event.get("file_path") or "")
            if _is_sorcar_md_read(event):
                row.sorcar_md_reads += 1
            key = (path, event.get("start_line"), event.get("max_lines"))
            if key in seen_reads[task_id]:
                row.repeat_reads += 1
            seen_reads[task_id].add(key)
        elif kind == "prompt":
            row.prompt_events += 1
            if row.prompt_events > 1:
                ctx, window = peak_context.get(task_id, (0, 0))
                if window and ctx >= HANDOFF_CONTEXT_FRACTION * window:
                    row.handoffs += 1
            peak_context.pop(task_id, None)
        elif kind == "usage_info":
            match = _CONTEXT_RE.search(str(event.get("text", "")))
            if match is None:
                continue
            context = int(match.group(1).replace(",", ""))
            window = int(match.group(2).replace(",", ""))
            if context >= peak_context.get(task_id, (0, 0))[0]:
                peak_context[task_id] = (context, window)
            if row.first_context is None:
                row.first_context = context
            cost = _money(event.get("cost"))
            steps = int(event.get("total_steps") or 0)
            delta = max(0.0, cost - last_cost.get(task_id, 0.0))
            last_cost[task_id] = cost
            # Only a genuine model step (step counter advanced by one)
            # whose previous tool was not a fan-out is one step's own
            # cost; other usage_info events (live fan-out totals, the
            # step after run_parallel/run_agent) fold children's spend.
            own_step = steps == last_steps.get(task_id, 0) + 1 and last_tool.get(
                task_id, ""
            ) not in FANOUT_TOOLS
            last_steps[task_id] = max(steps, last_steps.get(task_id, 0))
            if own_step:
                bucket = _bucket(context)
                row.bucket_steps[bucket] = row.bucket_steps.get(bucket, 0) + 1
                row.bucket_cost[bucket] = row.bucket_cost.get(bucket, 0.0) + delta
            cache_read = event.get("cache_read")
            model = str(event.get("model") or "")
            if (
                isinstance(cache_read, int) and steps >= 2 and "claude" in model.lower()
            ):
                row.steps_with_cache_field += 1
                if cache_read > 0:
                    row.steps_with_cache_hit += 1


def load_tasks(db_path: str, since: float) -> dict[str, TaskRow]:
    """Load every task that started after *since* with its event figures.

    Args:
        db_path: Path of the ``sorcar.db`` to read.
        since: Epoch seconds; tasks with an older ``timestamp`` are skipped.

    Returns:
        Task rows keyed by task id.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = _select_rows(conn, "timestamp >= ?", (since,))
        _scan_events(conn, rows)
        # Ancestors that started before the window, so a child in the
        # window is attributed to its real tree instead of becoming a
        # root of its own (their own figures are not counted).
        missing = {r.parent_id for r in rows.values() if r.parent_id} - set(rows)
        while missing:
            ancestors = _select_rows(
                conn, f"id IN ({','.join('?' * len(missing))})", tuple(missing),
            )
            for row in ancestors.values():
                row.in_window = False
            rows.update(ancestors)
            missing = {r.parent_id for r in ancestors.values() if r.parent_id} - set(rows)
    finally:
        conn.close()
    return rows


def _select_rows(
    conn: sqlite3.Connection, where: str, params: tuple[Any, ...],
) -> dict[str, TaskRow]:
    cur = conn.execute(
        "SELECT id, parent_task_id, task, model, cost, tokens, steps, timestamp "
        f"FROM task_history WHERE {where}",
        params,
    )
    return {
        r[0]: TaskRow(
            id=r[0], parent_id=r[1] or "", task=r[2] or "", model=r[3] or "",
            cost=float(r[4] or 0.0), tokens=int(r[5] or 0), steps=int(r[6] or 0),
            timestamp=float(r[7] or 0.0),
        )
        for r in cur
    }


def _root_of(row: TaskRow, rows: dict[str, TaskRow]) -> str:
    seen: set[str] = set()
    current = row
    while current.parent_id in rows and current.id not in seen:
        seen.add(current.id)
        current = rows[current.parent_id]
    return current.id


def _reviewer_depths(rows: dict[str, TaskRow]) -> dict[str, int]:
    """Return, per task, how many reviewer ancestors-or-self it has (0 = none)."""
    depths: dict[str, int] = {}

    def depth(row: TaskRow, guard: set[str]) -> int:
        if row.id in depths:
            return depths[row.id]
        if row.id in guard:
            return 0
        guard.add(row.id)
        parent = rows.get(row.parent_id)
        inherited = depth(parent, guard) if parent is not None else 0
        own = 1 if (row.parent_id and is_review_task(row.task)) else 0
        depths[row.id] = inherited + own
        return depths[row.id]

    for row in rows.values():
        depth(row, set())
    return depths


def compute_kpis(rows: dict[str, TaskRow]) -> dict[str, Any]:
    """Aggregate the per-task figures into the report's KPIs.

    Args:
        rows: Output of :func:`load_tasks`.

    Returns:
        A JSON-serialisable dict of KPIs (see the module docstring).
    """
    window = [r for r in rows.values() if r.in_window]
    subagents = [r for r in window if r.parent_id]
    # A persisted parent row already folds its descendants' spend, so
    # totals count each tree once: its root, or — when the root started
    # before the window — the topmost in-window rows of that tree.
    counted = [r for r in window if not (r.parent_id in rows and rows[r.parent_id].in_window)]
    total_cost = sum(r.cost for r in counted)
    reads = sum(r.reads for r in window)
    repeat_reads = sum(r.repeat_reads for r in window)
    bucket_steps: dict[str, int] = {name: 0 for name, _ in CONTEXT_BUCKETS}
    bucket_cost: dict[str, float] = {name: 0.0 for name, _ in CONTEXT_BUCKETS}
    for r in window:
        for name in bucket_steps:
            bucket_steps[name] += r.bucket_steps.get(name, 0)
            bucket_cost[name] += r.bucket_cost.get(name, 0.0)
    steps_total = sum(bucket_steps.values())
    steps_over_200k = bucket_steps["200-300k"] + bucket_steps[">300k"]

    first_contexts = [r.first_context for r in subagents if r.first_context]
    depths = _reviewer_depths(rows)
    tree_cost: dict[str, float] = defaultdict(float)
    tree_review_cost: dict[str, float] = defaultdict(float)
    for r in counted:
        tree_cost[_root_of(r, rows)] += r.cost
    for r in window:
        # Topmost reviewer rows only: their cost already includes the
        # reviewers (and helpers) nested under them.
        parent = rows.get(r.parent_id)
        if depths[r.id] > 0 and (parent is None or depths[parent.id] == 0):
            tree_review_cost[_root_of(r, rows)] += r.cost
    review_shares = {
        root: min(1.0, tree_review_cost[root] / tree_cost[root])
        for root in tree_cost if tree_cost[root] > 0 and tree_review_cost[root] > 0
    }
    wrappers = [
        r for r in subagents
        if _SHELL_WRAPPER_RE.search(r.task) and r.steps <= SHELL_WRAPPER_MAX_STEPS
    ]
    cache_steps = sum(r.steps_with_cache_field for r in window)
    cache_hits = sum(r.steps_with_cache_hit for r in window)
    return {
        "tasks": len(window),
        "top_level_tasks": len(window) - len(subagents),
        "subagents": len(subagents),
        "cost_usd": round(total_cost, 4),
        "tokens": sum(r.tokens for r in counted),
        "steps": sum(r.steps for r in counted),
        "sorcar_md_reads": sum(r.sorcar_md_reads for r in rows.values()),
        "reads": reads,
        "repeat_reads": repeat_reads,
        "repeat_read_ratio": round(repeat_reads / reads, 4) if reads else 0.0,
        "steps_by_context_bucket": bucket_steps,
        "cost_by_context_bucket": {k: round(v, 4) for k, v in bucket_cost.items()},
        "steps_over_200k_ratio": round(steps_over_200k / steps_total, 4) if steps_total else 0.0,
        "subagent_step1_context_avg": (
            round(sum(first_contexts) / len(first_contexts)) if first_contexts else 0
        ),
        "context_handoffs": sum(r.handoffs for r in window),
        "session_restarts": sum(max(0, r.prompt_events - 1) for r in window),
        "reviewer_trees": len(review_shares),
        "reviewer_share_max": round(max(review_shares.values()), 4) if review_shares else 0.0,
        "reviewer_trees_over_half": sum(1 for s in review_shares.values() if s > 0.5),
        "reviewer_nesting_max": max((depths[r.id] for r in window), default=0),
        "shell_wrapper_subagents": len(wrappers),
        "shell_wrapper_cost_usd": round(sum(r.cost for r in wrappers), 4),
        "cache_field_steps": cache_steps,
        "cache_hit_ratio": round(cache_hits / cache_steps, 4) if cache_steps else None,
    }


def format_report(kpis: dict[str, Any], hours: float) -> str:
    """Render *kpis* as an aligned plain-text table."""
    lines = [f"Cost report for the last {hours:g} h"]
    width = max(len(k) for k in kpis)
    for key, value in kpis.items():
        if isinstance(value, dict):
            lines.append(f"{key:<{width}}  " + "  ".join(f"{k}={v}" for k, v in value.items()))
        else:
            lines.append(f"{key:<{width}}  {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", default=str(kiss_home() / "sorcar.db"))
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    if not Path(args.db).is_file():
        parser.error(f"database not found: {args.db}")
    kpis = compute_kpis(load_tasks(args.db, time.time() - args.hours * 3600))
    print(json.dumps(kpis, indent=2) if args.json else format_report(kpis, args.hours))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
