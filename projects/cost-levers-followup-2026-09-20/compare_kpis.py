#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
"""Re-run the cost-lever KPI comparison against ``tmp/baseline_24h.json``.

Wraps :mod:`kiss.scripts.cost_report` (``--hours N``) and adds the two
filters the first 24 h check needed by hand:

* ``--strip-workdir kiss_ablation`` drops the orphan sub-agents of the
  ``/tmp/kiss_ablation/*`` experiments (their parents were never persisted,
  so every orphan counts as its own review-shaped tree);
* ``--post-lever-only`` keeps only trees whose *root* task started after the
  lever commit (``756ddf4c0``, 2026-09-19 07:56:27 UTC) **and** whose steps
  carry the ``cache_read`` field that only post-lever code emits.

It then prints, per model, the $/step split by step type (normal cached
step, compaction cache-miss, TTL-expiry cache-miss, fan-out step, first
step of a session) so a $/step change can be attributed.

Usage::

    .venv/bin/python projects/cost-levers-followup-2026-09-20/compare_kpis.py \
        --hours 72 [--db ~/.kiss/sorcar.db] [--baseline tmp/baseline_24h.json] \
        [--out tmp/compare_72h] [--snapshot]

``--snapshot`` first copies the live DB with ``VACUUM INTO`` (the live DB is
several GB and busy) and works on the copy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compaction_analysis as ca  # noqa: E402

from kiss.core.config import kiss_home  # noqa: E402
from kiss.scripts import cost_report as cr  # noqa: E402

LEVER_EPOCH = 1789804587  # commit 756ddf4c0, 2026-09-19 07:56:27 UTC
CACHE_TTL_S = 300
KPI_ORDER = [
    "tasks",
    "top_level_tasks",
    "subagents",
    "cost_usd",
    "steps",
    "sorcar_md_reads",
    "shell_wrapper_subagents",
    "shell_wrapper_cost_usd",
    "subagent_step1_context_avg",
    "reviewer_trees",
    "reviewer_trees_over_half",
    "reviewer_share_max",
    "reviewer_nesting_max",
    "cache_field_steps",
    "cache_hit_ratio",
    "repeat_read_ratio",
    "steps_over_200k_ratio",
    "context_handoffs",
    "session_restarts",
]


def _excluded_ids(db: str, pattern: str, since: float) -> set[str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT id FROM task_history WHERE timestamp >= ? AND work_dir LIKE ?",
                (since, f"%{pattern}%"),
            )
        }
    finally:
        conn.close()


def _post_lever_ids(rows: dict[str, cr.TaskRow]) -> set[str]:
    keep: set[str] = set()
    for row in rows.values():
        root = rows[cr._root_of(row, rows)]
        if root.timestamp >= LEVER_EPOCH and (row.steps_with_cache_field or row.steps == 0):
            keep.add(row.id)
    return keep


def step_type_breakdown(
    db: str, since: float, ids: set[str], exclude_workdir: str | None
) -> dict[str, Any]:
    """Per model: $/step split by step type over the sessions of *ids*."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        _tasks, sessions = ca.load_sessions(conn, since, exclude_workdir)
    finally:
        conn.close()
    per_model: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0.0, 0])
    )
    for s in sessions:
        if s.task_id not in ids:
            continue
        for i, st in enumerate(s.steps):
            if st.fanout:
                kind = "fan-out step (children's cost folded in)"
            elif i == 0:
                kind = "first step of a session"
            elif st.cache_read < 0:
                kind = "no cache_read field (pre-lever code)"
            elif st.cache_read == 0 and st.context >= 60_000:
                prev = s.steps[i - 1]
                if prev.context - st.context > 2_000 and prev.context >= 95_000:
                    kind = "cache miss: compaction"
                elif st.ts and prev.ts and st.ts - prev.ts >= CACHE_TTL_S:
                    kind = "cache miss: >=5 min gap (TTL expiry)"
                else:
                    kind = "cache miss: other"
            else:
                kind = "normal step"
            a = per_model[s.model][kind]
            a[0] += 1
            a[1] += st.cost
            a[2] += st.context
    out: dict[str, Any] = {}
    for model, kinds in per_model.items():
        total_cost = sum(v[1] for v in kinds.values())
        total_steps = sum(v[0] for v in kinds.values())
        own = {k: v for k, v in kinds.items() if not k.startswith("fan-out")}
        own_steps = sum(v[0] for v in own.values())
        own_cost = sum(v[1] for v in own.values())
        normal = kinds.get("normal step", [0, 0.0, 0])
        out[model] = {
            "steps": total_steps,
            "cost_usd": round(total_cost, 2),
            "own_cost_per_step": round(own_cost / own_steps, 4) if own_steps else None,
            "normal_step_cost": round(normal[1] / normal[0], 4) if normal[0] else None,
            "normal_step_ctx_avg": round(normal[2] / normal[0]) if normal[0] else None,
            "by_type": {
                k: {
                    "steps": v[0],
                    "cost_usd": round(v[1], 2),
                    "cost_share": round(v[1] / total_cost, 3) if total_cost else 0,
                    "cost_per_step": round(v[1] / v[0], 4),
                    "ctx_avg": round(v[2] / v[0]),
                }
                for k, v in sorted(kinds.items(), key=lambda kv: -kv[1][1])
            },
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["cost_usd"]))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def render_markdown(
    baseline: dict[str, Any] | None,
    raw: dict[str, Any],
    filt: dict[str, Any],
    breakdown: dict[str, Any],
    hours: float,
    ran_at: str,
) -> str:
    lines = [f"# cost_report --hours {hours:g} vs tmp/baseline_24h.json (run {ran_at})", ""]
    lines.append(
        "| KPI | baseline (pre-lever 24 h) | live raw | "
        "live filtered (no ablation orphans, post-lever trees only) |"
    )
    lines.append("|---|---|---|---|")
    for k in KPI_ORDER:
        b = baseline.get(k, "n/a") if baseline else "n/a"
        lines.append(
            f"| {k} | {_fmt(b)} | {_fmt(raw.get(k, 'n/a'))} | {_fmt(filt.get(k, 'n/a'))} |"
        )
    for bucket in ["<50k", "50-100k", "100-200k", "200-300k", ">300k"]:
        b = (baseline or {}).get("steps_by_context_bucket", {}).get(bucket, "n/a")
        lines.append(
            f"| steps_by_context_bucket[{bucket}] | {_fmt(b)} | "
            f"{_fmt(raw.get('steps_by_context_bucket', {}).get(bucket, 'n/a'))} | "
            f"{_fmt(filt.get('steps_by_context_bucket', {}).get(bucket, 'n/a'))} |"
        )

    def _over100(k: dict[str, Any]) -> str:
        s = k.get("steps_by_context_bucket", {})
        tot = sum(s.values()) or 1
        over = sum(v for b, v in s.items() if b in ("100-200k", "200-300k", ">300k"))
        return f"{over} ({over / tot:.1%})"

    lines.append(
        f"| steps >=100k (share) | {_over100(baseline) if baseline else 'n/a'} | "
        f"{_over100(raw)} | {_over100(filt)} |"
    )

    def _cps(k: dict[str, Any] | None) -> str:
        if not k or not k.get("steps"):
            return "n/a"
        return _fmt(k["cost_usd"] / k["steps"])

    lines.append(
        f"| cost per step (all models, derived) | {_cps(baseline)} | {_cps(raw)} | {_cps(filt)} |"
    )
    lines += ["", "## Per-model $/step by step type (live filtered)", ""]
    for model, d in breakdown.items():
        lines.append(
            f"### {model}: {d['steps']} steps, ${d['cost_usd']}, "
            f"own $/step {d['own_cost_per_step']}, "
            f"normal-step $/step {d['normal_step_cost']} at avg ctx {d['normal_step_ctx_avg']:,}"
            if d["normal_step_ctx_avg"] is not None
            else f"### {model}: {d['steps']} steps, ${d['cost_usd']}"
        )
        lines.append("")
        lines.append("| step type | steps | cost | share | $/step | avg ctx |")
        lines.append("|---|---|---|---|---|---|")
        for k, v in d["by_type"].items():
            lines.append(
                f"| {k} | {v['steps']} | ${v['cost_usd']} | {v['cost_share']:.1%} | "
                f"{v['cost_per_step']} | {v['ctx_avg']:,} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--db", default=str(kiss_home() / "sorcar.db"))
    p.add_argument("--hours", type=float, default=72.0)
    p.add_argument("--baseline", default="tmp/baseline_24h.json")
    p.add_argument("--strip-workdir", default="kiss_ablation")
    p.add_argument("--no-post-lever-filter", action="store_true")
    p.add_argument("--out", default=None, help="write <out>.json and <out>.md")
    p.add_argument("--snapshot", action="store_true", help="VACUUM INTO a temp copy first")
    a = p.parse_args()

    db = a.db
    if a.snapshot:
        tmp = Path(tempfile.mkdtemp()) / "sorcar_snapshot.db"
        live = sqlite3.connect(a.db)
        try:
            live.execute("VACUUM INTO ?", (str(tmp),))
        finally:
            live.close()
        db = str(tmp)

    since = time.time() - a.hours * 3600
    rows = cr.load_tasks(db, since)
    raw = cr.compute_kpis(rows)
    excluded = _excluded_ids(db, a.strip_workdir, since) if a.strip_workdir else set()
    kept = {k: v for k, v in rows.items() if k not in excluded}
    if not a.no_post_lever_filter:
        keep_ids = _post_lever_ids(kept)
        kept = {k: v for k, v in kept.items() if k in keep_ids}
    filt = cr.compute_kpis(kept)
    breakdown = step_type_breakdown(db, since, set(kept), a.strip_workdir or None)

    baseline = None
    if Path(a.baseline).exists():
        baseline = json.loads(Path(a.baseline).read_text())
    ran_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    md = render_markdown(baseline, raw, filt, breakdown, a.hours, ran_at)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out + ".json").write_text(
            json.dumps(
                {
                    "ran_at": ran_at,
                    "hours": a.hours,
                    "baseline": baseline,
                    "raw": raw,
                    "filtered": filt,
                    "step_type_breakdown": breakdown,
                    "excluded_ablation_tasks": len(excluded),
                },
                indent=2,
            )
        )
        Path(a.out + ".md").write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
