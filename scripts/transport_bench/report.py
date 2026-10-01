# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Render the transport benchmark results as a self-contained HTML report.

Reads the JSON written by :mod:`run_bench`, aggregates the rounds
(median across rounds of every per-round statistic; peak RSS is the
max), and writes one HTML file with inline SVG bar charts and tables.

Usage::

    uv run python scripts/transport_bench/report.py \
        tmp/transport_bench/results.json reports/transport_bench_uds_vs_wss.html \
        --findings tmp/transport_bench/findings.html
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import string
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
COLORS = {"uds": "#1f6f8b", "wss": "#c9622b", "wss-nocomp": "#8a8a3a"}
LABELS = {"uds": "UDS", "wss": "WSS (deflate)", "wss-nocomp": "WSS (no compression)"}


def _latency_key(workload: str) -> str:
    if workload in ("ping", "models"):
        return "rtt"
    if workload == "connect":
        return "first_pong"
    if workload.startswith("burst"):
        return "latency"
    return "text_delta_latency"


def _ops(workload: str, r: dict[str, Any]) -> int:
    """Measured operations in the phase: replies, or delivered stream events.

    Lifecycle events (status, result, ...) are not counted, so "CPU per
    op" is CPU per round trip / per delivered ``bench_event`` or
    ``text_delta`` copy.
    """
    if workload.startswith(("ping", "models", "connect")):
        return int(r[_latency_key(workload)]["n"])
    if workload.startswith("burst"):
        return int(r["latency"]["n"])
    delta = int(r["text_delta_latency"]["n"])
    viewer = int(r["viewer_text_delta_latency"].get("n", 0))
    return delta + viewer


def aggregate(results: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{workload: {transport: {metric: value}}}`` medians over rounds."""
    per: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rnd in results["rounds"]:
        for tr, wls in rnd.items():
            for wl, r in wls.items():
                per.setdefault(wl, {}).setdefault(tr, []).append(r)
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for wl, by_tr in per.items():
        for tr, rounds in by_tr.items():
            m: dict[str, float] = {}
            procs = [r["proc"] for r in rounds]
            m["wall_s"] = statistics.median(p["wall_s"] for p in procs)
            m["daemon_cpu_s"] = statistics.median(p["daemon_cpu_s"] for p in procs)
            m["daemon_cpu_pct"] = statistics.median(p["daemon_cpu_pct"] for p in procs)
            m["client_cpu_pct"] = statistics.median(p["client_cpu_pct"] for p in procs)
            m["rss_peak_mb"] = max(p["daemon_rss_peak_kb"] for p in procs) / 1024
            m["rss_after_mb"] = statistics.median(p["daemon_rss_after_kb"] for p in procs) / 1024
            m["rss_growth_mb"] = statistics.median(
                (p["daemon_rss_after_kb"] - p["daemon_rss_before_kb"]) for p in procs) / 1024
            if wl != "idle":
                key = _latency_key(wl)
                for stat in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
                    m[stat] = statistics.median(r[key][stat] for r in rounds)
                ops = statistics.median(_ops(wl, r) for r in rounds)
                m["daemon_us_per_msg"] = 1e6 * m["daemon_cpu_s"] / ops if ops else 0
                m["client_us_per_msg"] = 1e6 * statistics.median(
                    p["client_cpu_s"] for p in procs) / ops if ops else 0
            if wl.startswith("burst"):
                m["events_per_s"] = statistics.median(r["events_per_s"] for r in rounds)
                m["bytes_in_mb"] = statistics.median(r["bytes_in_total"] for r in rounds) / 1e6
            if wl.startswith("task"):
                m["task_wall_s"] = statistics.median(r["task_wall_s"] for r in rounds)
                m["stream_wall_s"] = statistics.median(r["stream_wall_s"] for r in rounds)
                m["events"] = statistics.median(sum(r["events"].values()) for r in rounds)
                m["runner_bytes_mb"] = statistics.median(r["runner_bytes_in"] for r in rounds) / 1e6
                if all(r["viewer_text_delta_latency"].get("n") for r in rounds):
                    m["viewer_p95_ms"] = statistics.median(
                        r["viewer_text_delta_latency"]["p95_ms"] for r in rounds)
            agg.setdefault(wl, {})[tr] = m
    return agg


# ----------------------------------------------------------------- SVG


def bar_chart(
    title: str, groups: list[str], series: dict[str, list[float]], unit: str,
    width: int = 900, log: bool = False,
) -> str:
    """Grouped bar chart: one group per workload, one bar per transport."""
    import math

    n_groups, n_series = len(groups), len(series)
    left, right, top, bottom = 70, 20, 40, 110
    height = 340
    plot_w, plot_h = width - left - right, height - top - bottom
    vals = [v for vs in series.values() for v in vs if v > 0]
    if not vals:
        return ""
    vmax = max(vals)
    vmin = min(vals) if log else 0.0

    def y_of(v: float) -> float:
        if log:
            lo, hi = math.log10(vmin / 2), math.log10(vmax * 1.15)
            return top + plot_h - (math.log10(max(v, vmin / 2)) - lo) / (hi - lo) * plot_h
        return top + plot_h - v / (vmax * 1.15) * plot_h

    group_w = plot_w / n_groups
    bar_w = group_w * 0.8 / n_series
    esc_title = html.escape(title)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="{esc_title}" style="font-family:system-ui,sans-serif;font-size:12px">',
           f'<text x="{left}" y="22" font-size="15" font-weight="600">{esc_title}</text>']
    # axis + gridlines
    if log:
        lo_dec, hi_dec = math.floor(math.log10(vmin / 2)), math.ceil(math.log10(vmax * 1.15))
        ticks = [10.0 ** k for k in range(lo_dec, hi_dec + 1)]
    else:
        ticks = [vmax * 1.15 * k / 5 for k in range(6)]
    for t in ticks:
        if t > vmax * 1.2:
            continue
        y = y_of(t)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
                   'stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#555">'
                   f'{_fmt(t)}</text>')
    mid_y = f"{top + plot_h / 2:.0f}"
    out.append(f'<text x="14" y="{mid_y}" transform="rotate(-90 14 {mid_y})" '
               f'text-anchor="middle" fill="#555">{html.escape(unit)}</text>')
    for gi, g in enumerate(groups):
        gx = left + gi * group_w + group_w * 0.1
        for si, (tr, vs) in enumerate(series.items()):
            v = vs[gi]
            if v <= 0:
                continue
            x = gx + si * bar_w
            y = y_of(v)
            tip = (f"{html.escape(LABELS.get(tr, tr))} {html.escape(g)}: "
                   f"{_fmt(v)} {html.escape(unit)}")
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" '
                       f'height="{top + plot_h - y:.1f}" fill="{COLORS.get(tr, "#999")}">'
                       f'<title>{tip}</title></rect>')
            out.append(f'<text x="{x + bar_w / 2 - 1:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
                       f'font-size="10" fill="#333">{_fmt(v)}</text>')
        label_x, label_y = f"{gx + group_w * 0.4:.1f}", top + plot_h + 14
        out.append(f'<text x="{label_x}" y="{label_y}" text-anchor="end" '
                   f'transform="rotate(-30 {label_x} {label_y})" fill="#333">'
                   f'{html.escape(g)}</text>')
    lx = left
    for tr in series:
        out.append(f'<rect x="{lx}" y="{height - 16}" width="12" height="12" '
                   f'fill="{COLORS.get(tr, "#999")}"/>')
        out.append(f'<text x="{lx + 16}" y="{height - 6}" fill="#333">'
                   f'{html.escape(LABELS.get(tr, tr))}</text>')
        lx += 170
    out.append("</svg>")
    return "\n".join(out)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# ---------------------------------------------------------------- HTML


def _table(agg: dict[str, dict[str, dict[str, float]]], workloads: list[str],
           cols: list[tuple[str, str]], transports: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for _, c in cols)
    rows = [f"<table><thead><tr><th>workload</th><th>transport</th>{head}</tr></thead><tbody>"]
    for wl in workloads:
        for i, tr in enumerate(transports):
            m = agg.get(wl, {}).get(tr)
            if not m:
                continue
            cells = "".join(f"<td>{_fmt(m[k]) if k in m else '–'}</td>" for k, _ in cols)
            first = ""
            if i == 0:
                first = f'<td rowspan="{len(transports)}"><code>{html.escape(wl)}</code></td>'
            rows.append(f"<tr>{first}<td>{html.escape(LABELS.get(tr, tr))}</td>{cells}</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def render(results: dict[str, Any], findings_html: str, results_path: str) -> str:
    """Build the whole HTML document."""
    agg = aggregate(results)
    transports = list(results["config"]["transports"])
    cfg = results["config"]
    order = [wl for wl in agg if wl != "idle"]
    rr = [w for w in order if w in ("connect", "ping", "models")]
    bursts_flood = [w for w in order if w.startswith("burst") and w.endswith("_g0")]
    bursts_paced = [w for w in order if w.startswith("burst") and not w.endswith("_g0")]
    tasks = [w for w in order if w.startswith("task")]

    def chart(title: str, wls: list[str], metric: str, unit: str, log: bool = False) -> str:
        series = {tr: [agg[w].get(tr, {}).get(metric, 0.0) for w in wls] for tr in transports}
        return bar_chart(title, wls, series, unit, log=log)

    charts = [
        chart("Command round trips: p50 latency", rr, "p50_ms", "ms", log=True),
        chart("Command round trips: p95 latency", rr, "p95_ms", "ms", log=True),
        chart("Command round trips: daemon CPU per round trip", rr, "daemon_us_per_msg",
              "µs CPU / round trip", log=True),
        chart("Paced broadcast (500 events/s): one-way p95 latency", bursts_paced, "p95_ms", "ms"),
        chart("Paced broadcast: daemon CPU per delivered event", bursts_paced, "daemon_us_per_msg",
              "µs CPU / event"),
        chart("Flood broadcast: delivered events per second (all clients)", bursts_flood,
              "events_per_s", "events / s"),
        chart("Flood broadcast: daemon CPU per delivered event", bursts_flood, "daemon_us_per_msg",
              "µs CPU / event"),
        chart("Real agent task: text_delta one-way p95 latency", tasks, "p95_ms", "ms", log=True),
        chart("Real agent task: daemon CPU during the task", tasks, "daemon_cpu_s", "CPU seconds"),
        chart("Real agent task: wall time", tasks, "task_wall_s", "s"),
        chart("Daemon peak RSS during each workload", order, "rss_peak_mb", "MB"),
    ]
    idle = agg.get("idle", {})
    idle_rows = "".join(
        f"<tr><td>{html.escape(LABELS.get(tr, tr))}</td><td>{_fmt(m['daemon_cpu_pct'])} %</td>"
        f"<td>{_fmt(m['rss_after_mb'])} MB</td></tr>" for tr, m in idle.items())
    lat_cols = [("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms"),
                ("max_ms", "max ms"), ("daemon_cpu_pct", "daemon CPU %"),
                ("daemon_us_per_msg", "daemon µs/round trip"), ("client_cpu_pct", "client CPU %"),
                ("rss_peak_mb", "daemon peak RSS MB")]
    burst_cols = lat_cols[:4] + [
        ("events_per_s", "events/s"), ("daemon_cpu_pct", "daemon CPU %"),
        ("daemon_us_per_msg", "daemon µs/event"), ("client_cpu_pct", "client CPU %"),
        ("bytes_in_mb", "MB received"), ("rss_peak_mb", "peak RSS MB")]
    task_cols = lat_cols[:3] + [
        ("viewer_p95_ms", "viewers p95 ms"), ("task_wall_s", "task wall s"),
        ("stream_wall_s", "stream wall s"), ("events", "events to runner"),
        ("daemon_cpu_s", "daemon CPU s"), ("daemon_cpu_pct", "daemon CPU %"),
        ("client_cpu_pct", "client CPU %"), ("runner_bytes_mb", "MB to runner"),
        ("rss_peak_mb", "peak RSS MB")]
    figs = "\n".join(f"<figure>{c}</figure>" for c in charts if c)
    template = string.Template((HERE / "report_template.html").read_text())
    return template.substitute(
        rounds=cfg["rounds"], python=html.escape(results["python"].split()[0]),
        findings=findings_html, connects=cfg["connects"], pings=cfg["pings"],
        models_reqs=cfg["models_reqs"], burst_events=cfg["burst_events"],
        chunk_chars=cfg["chunk_chars"], idle_rows=idle_rows, figs=figs,
        rr_table=_table(agg, rr, lat_cols, transports),
        burst_table=_table(agg, bursts_paced + bursts_flood, burst_cols, transports),
        task_table=_table(agg, tasks, task_cols, transports),
        results_path=html.escape(results_path),
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--findings", default="", help="HTML fragment inserted after the intro")
    ns = ap.parse_args()
    results = json.loads(Path(ns.results).read_text())
    findings = Path(ns.findings).read_text() if ns.findings else ""
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(render(results, findings, ns.results))
    print(ns.out)


if __name__ == "__main__":
    main()
