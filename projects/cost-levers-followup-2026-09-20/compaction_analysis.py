#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
"""Why has tool-output compaction not reduced the share of steps above 100k context?

Walks the ``events`` of every task in a window of a ``sorcar.db`` snapshot
and reconstructs, per session (one ``prompt`` event = one session) and per
step (one ``usage_info`` event = one model call):

* the reported context size,
* the *observed* compaction drops (context falling between two consecutive
  steps of the same session — compaction runs right before a model call),
* the composition of what was added to the conversation since the previous
  step: tool-result chars by tool, tool-call argument chars (Write/Edit
  content lives here, on the assistant side), assistant text and thinking.

Usage::

    .venv/bin/python projects/cost-levers-followup-2026-09-20/compaction_analysis.py \
        --db /tmp/cost72.db --since 1789804587 [--exclude-workdir kiss_ablation] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

CONTEXT_RE = re.compile(r"Context:\s*([\d,]+)\s*/\s*([\d,]+)")
CHARS_PER_TOKEN = 3.6  # calibrated below against the first steps' context deltas
COMPACTION_START = 100_000
MIN_COMPACT_CHARS = 2_000
KEEP_RECENT = 20
FANOUT_TOOLS = frozenset({"run_parallel", "run_agent"})


@dataclass
class Step:
    n: int
    context: int
    cost: float
    cache_read: int  # -1 when the event carries no cache_read field (pre-lever code)
    ts: float = 0.0
    added: Counter = field(default_factory=Counter)  # chars by category
    fanout: bool = False  # a run_parallel/run_agent preceded this call: cost folds children
    tool_results: list[tuple[str, int]] = field(default_factory=list)  # (tool, chars)


@dataclass
class Session:
    task_id: str
    model: str
    index: int
    prompt_chars: int = 0
    steps: list[Step] = field(default_factory=list)


def _money(text: Any) -> float:
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def load_sessions(
    conn: sqlite3.Connection, since: float, exclude_workdir: str | None
) -> tuple[dict[str, dict[str, Any]], list[Session]]:
    where = "timestamp >= ?"
    params: list[Any] = [since]
    if exclude_workdir:
        where += " AND work_dir NOT LIKE ?"
        params.append(f"%{exclude_workdir}%")
    tasks = {
        r[0]: {"model": r[1], "parent": r[2] or "", "cost": r[3], "steps": r[4], "task": r[5][:80]}
        for r in conn.execute(
            f"SELECT id, model, parent_task_id, cost, steps, task FROM task_history WHERE {where}",
            params,
        )
    }
    sessions: list[Session] = []
    if not tasks:
        return tasks, sessions
    ids = list(tasks)
    chunk = 500
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        cur = conn.execute(
            "SELECT task_id, event_json FROM events WHERE task_id IN "
            f"({','.join('?' * len(part))}) ORDER BY task_id, seq",
            part,
        )
        current: dict[str, Session] = {}
        pending: dict[str, Step] = {}
        last_tool: dict[str, str] = {}
        last_cost: dict[str, float] = {}
        bash_cap: dict[str, int] = {}
        for task_id, event_json in cur:
            try:
                ev = json.loads(event_json)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            kind = ev.get("type")
            sess = current.get(task_id)
            step = pending.setdefault(task_id, Step(0, 0, 0.0, 0))
            if kind == "prompt":
                sess = Session(task_id, tasks[task_id]["model"], (sess.index + 1) if sess else 0)
                sess.prompt_chars = len(str(ev.get("text", "")))
                current[task_id] = sess
                sessions.append(sess)
                pending[task_id] = Step(0, 0, 0.0, 0)
            elif kind == "system_prompt":
                step.added["system_prompt"] += len(str(ev.get("text", "")))
            elif kind == "tool_call":
                name = str(ev.get("name", ""))
                last_tool[task_id] = name
                step.fanout = step.fanout or name in FANOUT_TOOLS
                if name == "Bash":
                    extras = ev.get("extras") or {}
                    try:
                        bash_cap[task_id] = int(extras.get("max_output_chars") or 50_000)
                    except (TypeError, ValueError):
                        bash_cap[task_id] = 50_000
                size = len(event_json)
                step.added["tool_call_args"] += size
                step.added[f"args:{name}"] += size
            elif kind == "tool_result":
                name = str(ev.get("tool_name") or last_tool.get(task_id, "?"))
                size = len(str(ev.get("content", "")))
                if (
                    name == "Bash"
                    and size == 0
                    and step.tool_results
                    and step.tool_results[-1][0] == "Bash(stream)"
                ):
                    # Bash output was streamed as system_output events (folded
                    # above); the model sees at most max_output_chars of it.
                    cap = bash_cap.get(task_id, 50_000)
                    streamed = step.tool_results[-1][1]
                    if streamed > cap:
                        step.tool_results[-1] = ("Bash(stream)", cap)
                        step.added["tool_results"] -= streamed - cap
                        step.added["result:Bash(stream)"] -= streamed - cap
                    continue
                step.added["tool_results"] += size
                step.added[f"result:{name}"] += size
                step.tool_results.append((name, size))
            elif kind == "system_output":
                # Bash output is streamed as system_output chunks; the
                # matching tool_result has empty content.
                size = len(str(ev.get("text", "")))
                step.added["tool_results"] += size
                step.added["result:Bash(stream)"] += size
                if step.tool_results and step.tool_results[-1][0] == "Bash(stream)":
                    step.tool_results[-1] = ("Bash(stream)", step.tool_results[-1][1] + size)
                else:
                    step.tool_results.append(("Bash(stream)", size))
            elif kind == "text_delta":
                step.added["assistant_text"] += len(str(ev.get("text", "")))
            elif kind == "thinking_delta":
                step.added["thinking"] += len(str(ev.get("text", "")))
            elif kind == "usage_info":
                m = CONTEXT_RE.search(str(ev.get("text", "")))
                if m is None or sess is None:
                    continue
                total_steps = int(ev.get("total_steps") or 0)
                if sess.steps and total_steps == sess.steps[-1].n:
                    # live fan-out total or a duplicate; not a model call
                    continue
                cost = _money(ev.get("cost"))
                step.n = total_steps
                step.context = int(m.group(1).replace(",", ""))
                # The trajectory summarizer's usage events (a hand-off) are
                # persisted out of order with a lower cumulative cost, so
                # keep the running maximum as the reference.
                cost = max(cost, last_cost.get(task_id, 0.0))
                step.cost = max(0.0, cost - last_cost.get(task_id, 0.0))
                step.cache_read = int(ev["cache_read"]) if ev.get("cache_read") is not None else -1
                step.ts = float(ev.get("ts") or 0) / 1000.0
                last_cost[task_id] = cost
                sess.steps.append(step)
                pending[task_id] = Step(0, 0, 0.0, 0)
    return tasks, sessions


def analyse(tasks: dict[str, dict[str, Any]], sessions: list[Session]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    steps_all = [(s, st) for s in sessions for st in s.steps]
    out["tasks"] = len(tasks)
    out["sessions"] = len(sessions)
    out["steps"] = len(steps_all)

    # ---- 1. observed compaction drops ------------------------------------
    drops: list[dict[str, Any]] = []
    for s in sessions:
        for prev, cur in zip(s.steps, s.steps[1:]):
            # what was appended since prev (chars -> tokens) must be netted out
            added_tokens = (
                sum(
                    v
                    for k, v in cur.added.items()
                    if k in ("tool_results", "tool_call_args", "assistant_text")
                )
                / CHARS_PER_TOKEN
            )
            saved = prev.context + added_tokens - cur.context
            if cur.context < prev.context - 2_000:
                drops.append(
                    {
                        "task": s.task_id[:8],
                        "step": cur.n,
                        "before": prev.context,
                        "after": cur.context,
                        "raw_drop": prev.context - cur.context,
                        "est_saved": round(saved),
                    }
                )
    out["compactions_observed"] = len(drops)
    out["compaction_raw_drop_total"] = sum(d["raw_drop"] for d in drops)
    out["compaction_est_saved_total"] = sum(d["est_saved"] for d in drops)
    out["compaction_median_raw_drop"] = (
        sorted(d["raw_drop"] for d in drops)[len(drops) // 2] if drops else 0
    )
    out["compaction_after_below_100k"] = sum(1 for d in drops if d["after"] < COMPACTION_START)
    out["compaction_examples"] = drops[:12]

    # ---- 2. share of steps >= 100k by session length ---------------------
    total_ctx = sum(st.context for _, st in steps_all)
    over = [(s, st) for s, st in steps_all if st.context >= COMPACTION_START]
    out["steps_over_100k"] = len(over)
    out["steps_over_100k_share"] = round(len(over) / len(steps_all), 4) if steps_all else 0
    out["context_tokens_total"] = total_ctx
    out["context_tokens_over_100k_share"] = (
        round(sum(st.context for _, st in over) / total_ctx, 4) if total_ctx else 0
    )
    by_len: dict[str, dict[str, Any]] = {}
    bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10_000)]
    for lo, hi in bins:
        ss = [s for s in sessions if lo <= len(s.steps) < hi]
        n = sum(len(s.steps) for s in ss)
        o = sum(1 for s in ss for st in s.steps if st.context >= COMPACTION_START)
        c = sum(st.cost for s in ss for st in s.steps if not st.fanout)
        by_len[f"{lo}-{hi if hi < 10_000 else 'inf'} steps"] = {
            "sessions": len(ss),
            "steps": n,
            "steps_over_100k": o,
            "share": round(o / n, 3) if n else 0,
            "cost": round(c, 2),
            "peak_ctx_avg": round(
                sum(max((st.context for st in s.steps), default=0) for s in ss) / len(ss)
            )
            if ss
            else 0,
        }
    out["by_session_length"] = by_len
    # cost concentration: sessions with >=100 steps
    long = [s for s in sessions if len(s.steps) >= 100]
    out["long_sessions"] = [
        {
            "task": s.task_id[:8],
            "model": s.model,
            "session": s.index,
            "steps": len(s.steps),
            "peak_ctx": max(st.context for st in s.steps),
            "steps_over_100k": sum(1 for st in s.steps if st.context >= COMPACTION_START),
            "cost": round(sum(st.cost for st in s.steps if not st.fanout), 2),
            "first_step_over_100k": next(
                (st.n for st in s.steps if st.context >= COMPACTION_START), None
            ),
            "task_text": tasks[s.task_id]["task"][:60],
        }
        for s in sorted(long, key=lambda s: -sum(st.cost for st in s.steps if not st.fanout))
    ]

    # ---- 3. what the context is made of (chars added per session) --------
    comp: Counter[str] = Counter()
    for s in sessions:
        comp["prompt"] += s.prompt_chars
        for st in s.steps:
            for k, v in st.added.items():
                if not k.startswith(("args:", "result:")):
                    comp[k] += v
    out["chars_added_by_category"] = dict(comp.most_common())
    res_by_tool: Counter[str] = Counter()
    args_by_tool: Counter[str] = Counter()
    for s in sessions:
        for st in s.steps:
            for k, v in st.added.items():
                if k.startswith("result:"):
                    res_by_tool[k[7:]] += v
                elif k.startswith("args:"):
                    args_by_tool[k[5:]] += v
    out["tool_result_chars_by_tool"] = dict(res_by_tool.most_common(12))
    out["tool_call_arg_chars_by_tool"] = dict(args_by_tool.most_common(12))

    # ---- 4. compaction eligibility of tool results -----------------------
    sizes = [size for _, st in steps_all for _, size in st.tool_results]
    small = sum(v for v in sizes if v <= MIN_COMPACT_CHARS)
    out["tool_results"] = len(sizes)
    out["tool_result_chars"] = sum(sizes)
    out["tool_result_chars_below_min_compact"] = small
    out["tool_result_chars_below_min_compact_share"] = round(small / sum(sizes), 3) if sizes else 0
    protected = sum(
        size
        for _, st in steps_all
        for t, size in st.tool_results
        if t in ("Edit", "Write", "finish")
    )
    out["tool_result_chars_protected_tools"] = protected
    # how much of the tool-result volume sits in the 20 most recent results
    # at the time a session is at >=100k?  Approximate by sessions' tails.
    tail = 0
    total_in_long = 0
    for s in sessions:
        results = [(size) for st in s.steps for _, size in st.tool_results]
        if not results or max((st.context for st in s.steps), default=0) < COMPACTION_START:
            continue
        total_in_long += sum(results)
        tail += sum(results[-KEEP_RECENT:])
    out["tool_result_chars_in_sessions_reaching_100k"] = total_in_long
    out["tool_result_chars_in_last_20_results_of_those"] = tail

    # ---- 5. calibration: chars added vs context delta on early steps -----
    ratios = []
    for s in sessions:
        for prev, cur in zip(s.steps, s.steps[1:]):
            if cur.context <= prev.context or cur.context >= 90_000:
                continue
            added = sum(
                v
                for k, v in cur.added.items()
                if k in ("tool_results", "tool_call_args", "assistant_text")
            )
            if added > 4_000:
                ratios.append(added / (cur.context - prev.context))
    ratios.sort()
    out["chars_per_context_token_median"] = round(ratios[len(ratios) // 2], 2) if ratios else None

    # ---- 6. per-model $/step and >=100k share ----------------------------
    per_model: dict[str, dict[str, Any]] = {}
    for s, st in steps_all:
        d = per_model.setdefault(
            s.model, {"steps": 0, "cost": 0.0, "over_100k": 0, "ctx_sum": 0, "own_steps": 0}
        )
        d["steps"] += 1
        if not st.fanout:
            d["cost"] += st.cost
            d["own_steps"] += 1
        d["over_100k"] += st.context >= COMPACTION_START
        d["ctx_sum"] += st.context
    for m, d in per_model.items():
        d["cost"] = round(d["cost"], 2)
        d["cost_per_step"] = round(d["cost"] / d["own_steps"], 4) if d["own_steps"] else 0
        d["over_100k_share"] = round(d["over_100k"] / d["steps"], 3) if d["steps"] else 0
        d["ctx_avg"] = round(d["ctx_sum"] / d["steps"]) if d["steps"] else 0
        del d["ctx_sum"]
    out["per_model"] = dict(sorted(per_model.items(), key=lambda kv: -kv[1]["cost"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True)
    p.add_argument("--since", type=float, required=True, help="epoch seconds")
    p.add_argument("--until", type=float, default=None)
    p.add_argument("--exclude-workdir", default="kiss_ablation")
    p.add_argument("--json", default=None, help="write full result JSON here")
    a = p.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    tasks, sessions = load_sessions(conn, a.since, a.exclude_workdir or None)
    if a.until is not None:
        keep = {
            r[0]
            for r in conn.execute("SELECT id FROM task_history WHERE timestamp < ?", (a.until,))
        }
        tasks = {k: v for k, v in tasks.items() if k in keep}
        sessions = [s for s in sessions if s.task_id in tasks]
    res = analyse(tasks, sessions)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    for k, v in res.items():
        if isinstance(v, (dict, list)):
            print(f"{k}:")
            print(json.dumps(v, indent=2)[:6000])
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
