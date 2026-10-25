# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Token-cost KPIs of the tasks recorded in a ``history.db``.

The 7-day efficiency audit of 2026-09-19 found that most avoidable spend
came from a handful of measurable patterns: a mandatory first-step
``Read("./AGENTS.md")``, files re-read within the same task, steps run
at very large contexts, sub-agents that start with a ~12k-token first
step, reviewer sub-trees that exceed half of a task's spend, LLM
sub-agents used as shell wrappers, and context hand-offs at 90 % of the
window.  This script measures each of them for a time window so the
levers in ``projects/cost-levers-implementation-plan.md`` can be checked
against real data before and after they land.

Usage::

    uv run python -m kiss.scripts.cost_report [--db PATH] [--hours 24] [--json]
    uv run python -m kiss.scripts.cost_report --since 2026-10-17T23:14 --until 2026-10-24T23:14
    uv run python -m kiss.scripts.cost_report --since ... --baseline baseline.json

``--since`` / ``--until`` (ISO 8601, UTC unless a zone is given, or epoch
seconds) select an absolute window instead of the last ``--hours``.
``--baseline`` takes the ``--json`` output of an earlier run and prints
every KPI of the current window next to it with the change, so the weeks
before and after a change can be compared.  Count KPIs are also reported
per 1,000 model calls (``*_per_1k_calls``) because two windows never
hold the same amount of work.

Only the ``task_history`` and ``events`` tables are read; nothing is
written.  A missing or foreign event field contributes nothing to the
figure it belongs to.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from kiss.agents.sorcar.fanout_guard import is_review_task
from kiss.core.config import DEFAULT_CONFIG, kiss_home
from kiss.core.tool_verdict import REFUSED_PREFIX

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
# ``REFUSED_PREFIX`` up to the tool name: every refused call's result opens
# with it since 2026-10-24.  Before that each guard wrote its own text; the
# three that the 2026-10-24 audit counted are matched by wording so a
# window before the change can serve as a baseline.
_REFUSED_START = REFUSED_PREFIX.split("{name}")[0]
_REFUSED_PREFIX_RE = re.compile(
    "^" + re.escape(REFUSED_PREFIX).replace(re.escape("{name}"), r"\S+")
)
# Each guard's own text, anchored at the start of the result (after the
# prefix when present) so a file whose content quotes the wording is not
# counted.
_REFUSAL_REASONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary_guard", re.compile(r"^Step \d+ is a multiple of 10: call summary\(")),
    ("unread_file", re.compile(r"^Error: \S+ ha[sd] not been read in this session")),
    ("worktree_path", re.compile(r"^Error: command references the parent-repo path")),
    ("edit_bypass", re.compile(r"^REFUSED: the (`sed -i` call|python heredoc) did NOT run")),
)
# A tool result at least this long counts as oversized.
OVERSIZED_OUTPUT_CHARS = 10_000


def _refusal_reason(content: str) -> str | None:
    """Return the refusal category of a tool result, ``None`` when it was not refused."""
    head = content[:400]
    text = _REFUSED_PREFIX_RE.sub("", head)
    for reason, pattern in _REFUSAL_REASONS:
        if pattern.search(text):
            return reason
    return "other" if head.startswith(_REFUSED_START) else None


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
    agents_md_reads: int = 0
    reads: int = 0
    repeat_reads: int = 0
    first_context: int | None = None
    prompt_events: int = 0
    handoffs: int = 0
    bucket_steps: dict[str, int] = field(default_factory=dict)
    bucket_cost: dict[str, float] = field(default_factory=dict)
    steps_with_cache_field: int = 0
    steps_with_cache_hit: int = 0
    refusals: dict[str, int] = field(default_factory=dict)
    """Tool calls refused by a hook or guard, per :func:`_refusal_reason` category."""
    model_summaries: int = 0
    framework_summaries: int = 0
    output_sizes: list[int] = field(default_factory=list)
    """Characters of every tool result: the output streamed as ``system_output``
    events (a Bash call's full output, whose persisted result is blank) plus the
    persisted result (other tools' copy, truncated at a few thousand characters,
    so their sizes are a floor)."""
    llm_calls: int = 0
    llm_cost: float = 0.0
    """Spend of this task's own model calls (children's calls are their own rows')."""
    llm_output_tokens: int = 0
    routine_calls: int = 0
    """Model calls marked routine (the turn after a refusal, summary, memory tool, wait or
    unchanged Read)."""
    routine_output_tokens: int = 0
    routine_lowered: int = 0
    """Routine calls that ran at the lowered ``routine_turn_effort``."""


def _money(text: Any) -> float:
    """Parse the ``$1.2345`` cost string of a ``usage_info`` event."""
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def _event_cost(event: dict[str, Any]) -> float:
    """The USD cost of a ``usage_info`` event: exact ``cost_usd``, else the string."""
    exact = event.get("cost_usd")
    if isinstance(exact, (int, float)) and not isinstance(exact, bool):
        return float(exact)
    return _money(event.get("cost"))


def _bucket(context_tokens: int) -> str:
    for name, upper in CONTEXT_BUCKETS:
        if context_tokens < upper:
            return name
    return CONTEXT_BUCKETS[-1][0]  # pragma: no cover — sentinel bucket is unbounded


def _is_agents_md_read(event: dict[str, Any]) -> bool:
    """Whether *event* read the repository's ``AGENTS.md`` (``SORCAR.md`` before 2026.10.2)."""
    path = str(event.get("path") or event.get("file_path") or "")
    name = path.replace("\\", "/").rstrip("/")
    return name.endswith(("AGENTS.md", "SORCAR.md")) and "/.kiss/" not in path


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
    # Characters streamed as system_output since the last tool_call.
    streamed: dict[str, int] = defaultdict(int)
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
            streamed[task_id] = 0
            if event.get("name") == "summary":
                extras = event.get("extras")
                if isinstance(extras, dict) and extras.get("source") == "framework":
                    row.framework_summaries += 1
                else:
                    row.model_summaries += 1
        elif kind == "system_output":
            streamed[task_id] += len(str(event.get("text") or ""))
        elif kind == "tool_result":
            content = str(event.get("content") or "")
            row.output_sizes.append(len(content) + streamed.pop(task_id, 0))
            reason = _refusal_reason(content)
            if reason is not None:
                row.refusals[reason] = row.refusals.get(reason, 0) + 1
        elif kind == "llm_call":
            output_tokens = int(event.get("output_tokens") or 0)
            row.llm_calls += 1
            row.llm_cost += _money(event.get("cost"))
            row.llm_output_tokens += output_tokens
            if event.get("routine"):
                row.routine_calls += 1
                row.routine_output_tokens += output_tokens
                row.routine_lowered += 1 if event.get("effort") else 0
        if kind == "tool_call" and event.get("name") == "Read":
            row.reads += 1
            path = str(event.get("path") or event.get("file_path") or "")
            if _is_agents_md_read(event):
                row.agents_md_reads += 1
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
            cost = _event_cost(event)
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


def load_tasks(db_path: str, since: float, until: float | None = None) -> dict[str, TaskRow]:
    """Load every task that started in ``[since, until)`` with its event figures.

    Args:
        db_path: Path of the ``history.db`` to read.
        since: Epoch seconds; tasks with an older ``timestamp`` are skipped.
        until: Epoch seconds; tasks starting at or after it are skipped
            (``None`` = no upper bound).

    Returns:
        Task rows keyed by task id.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = _select_rows(
            conn, "timestamp >= ? AND timestamp < ?",
            (since, until if until is not None else 1e18),
        )
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
    output_sizes = sorted(size for r in window for size in r.output_sizes)
    llm_calls = sum(r.llm_calls for r in window)
    llm_output = sum(r.llm_output_tokens for r in window)
    routine_calls = sum(r.routine_calls for r in window)
    routine_output = sum(r.routine_output_tokens for r in window)
    tree_costs = sorted(tree_cost.values())
    llm_cost = sum(r.llm_cost for r in window)
    refused_by_reason: dict[str, int] = {reason: 0 for reason, _ in _REFUSAL_REASONS}
    refused_by_reason["other"] = 0
    for r in window:
        for reason, count in r.refusals.items():
            refused_by_reason[reason] += count
    refusals = sum(refused_by_reason.values())
    model_summaries = sum(r.model_summaries for r in window)
    oversized = sum(1 for s in output_sizes if s >= OVERSIZED_OUTPUT_CHARS)
    per_1k = 1000 / llm_calls if llm_calls else 0.0
    return {
        "tasks": len(window),
        "top_level_tasks": len(window) - len(subagents),
        "subagents": len(subagents),
        "cost_usd": round(total_cost, 4),
        "tokens": sum(r.tokens for r in counted),
        "steps": sum(r.steps for r in counted),
        "tree_cost_median_usd": round(_percentile(tree_costs, 0.5), 4),
        "tree_cost_p90_usd": round(_percentile(tree_costs, 0.9), 4),
        "cost_per_llm_call_usd": round(llm_cost / llm_calls, 4) if llm_calls else 0.0,
        "agents_md_reads": sum(r.agents_md_reads for r in rows.values()),
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
        "refused_tool_calls": refusals,
        "refused_by_reason": refused_by_reason,
        "refused_per_1k_calls": round(refusals * per_1k, 2),
        "summaries_by_model": model_summaries,
        "summaries_by_model_per_1k_calls": round(model_summaries * per_1k, 2),
        "summaries_by_framework": sum(r.framework_summaries for r in window),
        "tool_outputs": len(output_sizes),
        "tool_output_p99_chars": _percentile(output_sizes, 0.99),
        "tool_output_chars_total": sum(output_sizes),
        "tool_outputs_oversized": oversized,
        "oversized_per_1k_calls": round(oversized * per_1k, 2),
        "llm_calls": llm_calls,
        "routine_calls": routine_calls,
        "routine_calls_lowered": sum(r.routine_lowered for r in window),
        "routine_output_tokens_mean": (
            round(routine_output / routine_calls) if routine_calls else 0
        ),
        "other_output_tokens_mean": (
            round((llm_output - routine_output) / (llm_calls - routine_calls))
            if llm_calls > routine_calls else 0
        ),
    }


def _percentile[N: (int, float)](sorted_values: list[N], fraction: float) -> N | int:
    """Return the nearest-rank percentile of *sorted_values* (0 when empty)."""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _flatten(kpis: dict[str, Any]) -> dict[str, Any]:
    """Return *kpis* with nested dicts flattened to ``outer.inner`` keys."""
    flat: dict[str, Any] = {}
    for key, value in kpis.items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    return flat


def compare_kpis(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Set every KPI of *current* against *baseline*.

    Args:
        current: KPIs of the window under study (:func:`compute_kpis`).
        baseline: KPIs of the reference window (an earlier ``--json`` output).

    Returns:
        Per flattened KPI name: ``baseline``, ``current``, ``delta`` and
        ``pct`` (change relative to the baseline; ``None`` when either side
        is missing or not a number, or the baseline is zero).
    """
    flat_now, flat_base = _flatten(current), _flatten(baseline)
    table: dict[str, dict[str, Any]] = {}
    for key in list(flat_now) + [k for k in flat_base if k not in flat_now]:
        now, base = flat_now.get(key), flat_base.get(key)
        delta: float | None = None
        pct: float | None = None
        if _is_number(now) and _is_number(base):
            delta = round(now - base, 4)
            pct = round(100.0 * (now - base) / base, 1) if base else None
        table[key] = {"baseline": base, "current": now, "delta": delta, "pct": pct}
    return table


def _is_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_report(kpis: dict[str, Any], window: str) -> str:
    """Render *kpis* as an aligned plain-text table headed by *window*."""
    lines = [f"Cost report for {window}"]
    width = max(len(k) for k in kpis)
    for key, value in kpis.items():
        if isinstance(value, dict):
            lines.append(f"{key:<{width}}  " + "  ".join(f"{k}={v}" for k, v in value.items()))
        else:
            lines.append(f"{key:<{width}}  {value}")
    return "\n".join(lines)


def format_comparison(table: dict[str, dict[str, Any]], window: str) -> str:
    """Render :func:`compare_kpis`'s *table* as an aligned plain-text table."""
    width = max(len(k) for k in table)
    lines = [
        f"Cost report for {window} against the baseline",
        f"{'kpi':<{width}}  {'baseline':>14}  {'current':>14}  {'delta':>14}  {'change':>8}",
    ]
    for key, cells in table.items():
        pct = "" if cells["pct"] is None else f"{cells['pct']:+.1f}%"
        delta = "" if cells["delta"] is None else f"{cells['delta']:+g}"
        lines.append(
            f"{key:<{width}}  {str(cells['baseline']):>14}  {str(cells['current']):>14}  "
            f"{delta:>14}  {pct:>8}"
        )
    return "\n".join(lines)


def _parse_time(text: str) -> float:
    """Return epoch seconds for an ISO 8601 timestamp (UTC by default) or a number."""
    try:
        return float(text)
    except ValueError:
        pass
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%MZ")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", default=str(kiss_home() / "history.db"))
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--since", help="window start (ISO 8601 or epoch); overrides --hours")
    parser.add_argument("--until", help="window end (ISO 8601 or epoch); default: now")
    parser.add_argument("--baseline", help="--json output of an earlier run to compare with")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    if not Path(args.db).is_file():
        parser.error(f"database not found: {args.db}")
    until = _parse_time(args.until) if args.until else None
    if args.since:
        since = _parse_time(args.since)
        window = f"{_utc(since)} to {_utc(until) if until is not None else 'now'}"
    else:
        since = (time.time() if until is None else until) - args.hours * 3600
        window = (
            f"the last {args.hours:g} h" if until is None
            else f"the {args.hours:g} h before {_utc(until)}"
        )
    kpis = compute_kpis(load_tasks(args.db, since, until))
    if not args.baseline:
        print(json.dumps(kpis, indent=2) if args.json else format_report(kpis, window))
        return 0
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    table = compare_kpis(kpis, baseline)
    print(json.dumps(table, indent=2) if args.json else format_comparison(table, window))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
