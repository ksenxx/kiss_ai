#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
"""Replay recorded sessions through ``compact_tool_results`` with different settings.

For every session found by :mod:`compaction_analysis` the conversation is
rebuilt in the Anthropic shape from the persisted events (prompt, per step:
assistant thinking/text/tool_use, then tool_result blocks — Bash output
comes from the streamed ``system_output`` events capped at the call's
``max_output_chars``).  The real :func:`kiss.core.context_compaction.
compact_tool_results` is then applied with the production trigger logic
(first at ``start`` tokens, then every ``step`` tokens of growth) under
several parameter sets, and the resulting context-token series is compared:

* total context tokens summed over steps (what the provider bills as input),
* the number of steps at or above 100k,
* the peak context (what drives hand-offs).

Context tokens are estimated as ``prefix + chars / CHARS_PER_TOKEN +
PER_STEP_OVERHEAD * steps``; the constants are fitted so that the
*recorded* context series (which already includes the production
compaction) is reproduced, and the fit quality is printed.

Usage::

    .venv/bin/python projects/cost-levers-followup-2026-09-20/compaction_simulation.py \
        --db /tmp/cost72.db --since 1789804587 [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compaction_analysis as ca  # noqa: E402

from kiss.core.context_compaction import (  # noqa: E402
    KEEP_RECENT_TOOL_RESULTS,
    MIN_COMPACT_CHARS,
    PROTECTED_TOOLS,
    compact_tool_results,
)

CHARS_PER_TOKEN = 3.3
PER_STEP_OVERHEAD = 700  # tool_use/tool_result framing + usage string + thinking carry-over


@dataclass(frozen=True)
class Policy:
    name: str
    enabled: bool = True
    start: int = 100_000
    step: int = 50_000
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS
    min_chars: int = MIN_COMPACT_CHARS
    compact_args: bool = (
        False  # also stub large Bash commands / Write content on the assistant side
    )
    args_min_chars: int = 2_000


POLICIES = [
    Policy("A: no compaction", enabled=False),
    Policy("B: production (100k/50k, keep 20, min 2000)"),
    Policy("C: keep 5", keep_recent=5),
    Policy("D: keep 5, min 500", keep_recent=5, min_chars=500),
    Policy(
        "E: start 50k, keep 5, min 500", start=50_000, step=25_000, keep_recent=5, min_chars=500
    ),
    Policy("F: D + compact large tool-call args", keep_recent=5, min_chars=500, compact_args=True),
    Policy(
        "G: E + compact large tool-call args",
        start=50_000,
        step=25_000,
        keep_recent=5,
        min_chars=500,
        compact_args=True,
    ),
]


def _conversation_chars(conv: list[dict[str, Any]]) -> int:
    total = 0
    for m in conv:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
            continue
        for block in content or []:
            if block.get("type") == "text":
                total += len(block.get("text", ""))
            elif block.get("type") == "tool_use":
                total += len(json.dumps(block.get("input", {})))
            elif block.get("type") == "tool_result":
                c = block.get("content")
                total += len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c)
            elif block.get("type") == "thinking":
                total += len(block.get("thinking", ""))
    return total


def _compact_args(conv: list[dict[str, Any]], keep_recent: int, min_chars: int) -> int:
    """Stub large tool_use inputs of old assistant turns (Bash heredocs, Write content)."""
    turns = [m for m in conv if m.get("role") == "assistant"]
    n = 0
    for m in turns[: max(0, len(turns) - keep_recent)]:
        for block in m.get("content") or []:
            if block.get("type") != "tool_use":
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            for key in ("command", "content", "new_string", "old_string", "description"):
                val = inp.get(key)
                if (
                    isinstance(val, str)
                    and len(val) > min_chars
                    and not val.startswith("[compacted")
                ):
                    inp[key] = f"[compacted tool argument: {len(val):,} chars]\n" + val[:200]
                    n += 1
    return n


def replay(session: ca.Session, policy: Policy, prefix_tokens: int) -> list[int]:
    """Return the estimated context-token series of *session* under *policy*."""
    conv: list[dict[str, Any]] = [{"role": "user", "content": "x" * session.prompt_chars}]
    series: list[int] = []
    next_at = policy.start
    ctx = 0
    call_ids = 0
    for step in session.steps:
        # --- before the model call: compaction ---
        if policy.enabled and ctx >= next_at:
            next_at = ctx + policy.step
            compact_tool_results(
                conv,
                keep_recent=policy.keep_recent,
                min_chars=policy.min_chars,
                protected_tools=PROTECTED_TOOLS,
            )
            if policy.compact_args:
                _compact_args(conv, policy.keep_recent, policy.args_min_chars)
        # --- the assistant turn produced by this call ---
        # Prior-turn thinking is stripped by the provider; only this step's
        # thinking counts (as output).  Model it as a text block replaced
        # each step.
        blocks: list[dict[str, Any]] = []
        if step.added.get("assistant_text"):
            blocks.append({"type": "text", "text": "t" * step.added["assistant_text"]})
        results: list[dict[str, Any]] = []
        arg_chars = {k[5:]: v for k, v in step.added.items() if k.startswith("args:")}
        for tool, size in step.tool_results:
            call_ids += 1
            cid = f"c{call_ids}"
            name = "Bash" if tool == "Bash(stream)" else tool
            # spread this step's argument chars over its calls of that tool
            n_calls = sum(1 for t, _ in step.tool_results if t == tool) or 1
            a = arg_chars.get(name, 0) // n_calls
            key = "command" if name == "Bash" else "content"
            blocks.append({"type": "tool_use", "id": cid, "name": name, "input": {key: "a" * a}})
            results.append({"type": "tool_result", "tool_use_id": cid, "content": "r" * size})
        if not step.tool_results:
            blocks.append(
                {"type": "text", "text": "t" * max(1, step.added.get("assistant_text", 0))}
            )
        conv.append({"role": "assistant", "content": blocks})
        if results:
            conv.append({"role": "user", "content": results})
        chars = _conversation_chars(conv)
        thinking_out = (
            step.added.get("thinking", 0) * 0.9
        )  # summarized thinking text -> billed tokens
        ctx = int(
            prefix_tokens + chars / CHARS_PER_TOKEN + PER_STEP_OVERHEAD * len(series) + thinking_out
        )
        series.append(ctx)
    return series


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True)
    p.add_argument("--since", type=float, required=True)
    p.add_argument("--exclude-workdir", default="kiss_ablation")
    p.add_argument("--min-steps", type=int, default=20, help="only replay sessions this long")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    tasks, sessions = ca.load_sessions(conn, a.since, a.exclude_workdir or None)
    sessions = [s for s in sessions if len(s.steps) >= a.min_steps and "claude" in s.model]

    # ---- calibration of the production policy against the recorded series
    prod = POLICIES[1]
    rec_total = sim_total = 0
    abs_err = []
    per_session_fit = []
    for s in sessions:
        prefix = s.steps[0].context - int(
            (
                s.prompt_chars
                + sum(
                    v
                    for k, v in s.steps[0].added.items()
                    if k in ("tool_results", "tool_call_args", "assistant_text")
                )
            )
            / CHARS_PER_TOKEN
        )
        prefix = max(5_000, prefix)
        sim = replay(s, prod, prefix)
        rec = [st.context for st in s.steps]
        rec_total += sum(rec)
        sim_total += sum(sim)
        abs_err.append(abs(sum(sim) - sum(rec)) / sum(rec))
        per_session_fit.append(
            (s.task_id[:8], s.index, len(rec), max(rec), max(sim), round(sum(sim) / sum(rec), 2))
        )
    print(
        f"sessions replayed: {len(sessions)}; recorded ctx-token total {rec_total:,}; "
        f"simulated (production policy) {sim_total:,} (ratio {sim_total / rec_total:.3f}); "
        f"median per-session |err| {sorted(abs_err)[len(abs_err) // 2]:.2%}"
    )
    print("worst fits (task, session, steps, rec_peak, sim_peak, sim/rec):")
    for row in sorted(per_session_fit, key=lambda r: -abs(r[5] - 1))[:6]:
        print("  ", row)

    # ---- counterfactuals
    out: dict[str, Any] = {
        "sessions": len(sessions),
        "recorded_ctx_total": rec_total,
        "policies": {},
    }
    for pol in POLICIES:
        tot = over = peaks = 0
        n_steps = 0
        handoff_sessions = 0
        for s in sessions:
            prefix = max(
                5_000,
                s.steps[0].context
                - int(
                    (
                        s.prompt_chars
                        + sum(
                            v
                            for k, v in s.steps[0].added.items()
                            if k in ("tool_results", "tool_call_args", "assistant_text")
                        )
                    )
                    / CHARS_PER_TOKEN
                ),
            )
            sim = replay(s, pol, prefix)
            tot += sum(sim)
            over += sum(1 for c in sim if c >= 100_000)
            peaks += max(sim)
            n_steps += len(sim)
            handoff_sessions += max(sim) >= 350_000
        out["policies"][pol.name] = {
            "ctx_tokens_total": tot,
            "vs_production": round(tot / out["policies"][prod.name]["ctx_tokens_total"], 3)
            if prod.name in out["policies"]
            else 1.0,
            "steps_over_100k": over,
            "steps_over_100k_share": round(over / n_steps, 3),
            "peak_ctx_avg": round(peaks / len(sessions)),
            "sessions_reaching_350k": handoff_sessions,
        }
    print(json.dumps(out, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
