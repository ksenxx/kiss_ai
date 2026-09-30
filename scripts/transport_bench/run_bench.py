# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Benchmark the daemon's Unix-domain socket against its local WSS.

Drives ONE real ``kiss-web`` daemon (:mod:`bench_daemon`) through the
same JSON command API over each transport in turn and records, per
workload and transport, latency percentiles plus the daemon's CPU time
and resident memory read from ``/proc``:

``connect``     connection setup until the first ``pong`` (WSS: TCP + TLS
                + upgrade + ``auth`` handshake; UDS: socket connect).
``ping``        sequential ``ping`` -> ``pong`` round trips on one
                persistent connection.
``models``      sequential ``getModels`` -> ``models`` request/reply
                (a multi-KB catalog payload).
``burst``       the daemon broadcasts a burst of padded events from a
                worker thread to C connected clients; one-way delivery
                latency from the daemon's ``t0`` stamp.
``task``        a real agent task whose stand-in model streams N text
                deltas (:mod:`fake_model`); one-way latency of every
                ``text_delta`` and the task's wall time.

Transports: ``uds``; ``wss`` (permessage-deflate negotiated, as browsers
do); ``wss-nocomp`` (the client declines compression).

Usage (from the repo root)::

    uv run python scripts/transport_bench/run_bench.py --rounds 3 \
        --out tmp/transport_bench/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import queue
import resource
import shutil
import ssl
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect as ws_connect

HERE = Path(__file__).resolve().parent
MAX_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------- stats


def percentiles(samples_ns: list[int]) -> dict[str, float]:
    """Summarise nanosecond samples in milliseconds.

    Percentiles use the nearest-rank definition: the p-quantile is the
    ``ceil(p * n)``-th smallest sample.
    """
    if not samples_ns:
        return {"n": 0}
    s = sorted(samples_ns)

    def q(p: float) -> float:
        return s[max(0, math.ceil(p * len(s)) - 1)] / 1e6

    return {
        "n": len(s), "p50_ms": q(0.5), "p95_ms": q(0.95), "p99_ms": q(0.99),
        "mean_ms": statistics.fmean(s) / 1e6, "max_ms": s[-1] / 1e6,
        "min_ms": s[0] / 1e6,
    }


# ---------------------------------------------------------- /proc probes


class ProcStats:
    """CPU seconds and resident memory of the benchmark daemon.

    CPU time comes from the daemon itself (``cpu`` control command ->
    ``time.process_time_ns()``, all threads, ns resolution); memory from
    ``/proc/<pid>/status``.
    """

    def __init__(self, pid: int, daemon: Child) -> None:
        self.pid = pid
        self.daemon = daemon

    def cpu_seconds(self) -> float:
        """User + system CPU time consumed so far."""
        self.daemon.write("cpu")
        return int(self.daemon.expect("CPU")[1]) / 1e9

    def memory_kb(self) -> dict[str, int]:
        """``VmRSS`` and ``VmHWM`` (peak RSS since the last reset) in KiB."""
        out: dict[str, int] = {}
        for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0]] = int(line.split()[1])
        return out

    def reset_peak(self) -> None:
        """Reset ``VmHWM`` so the next reading is this phase's peak."""
        try:
            Path(f"/proc/{self.pid}/clear_refs").write_text("5")
        except OSError:
            pass


class Phase:
    """Measure daemon + client CPU and daemon memory over a steady-state window.

    Workloads call :meth:`start` once their connections are open and
    :meth:`stop` before closing them, so connection setup and teardown
    never land in the per-message CPU figures.  ``daemon_rss_peak_kb``
    is the largest of the ``VmHWM`` reading (reset at :meth:`start`) and
    the before/after ``VmRSS`` readings: procfs RSS counters are
    per-thread cached, so ``VmHWM`` alone can lag behind ``VmRSS``.
    """

    def __init__(self, daemon: ProcStats) -> None:
        self.daemon = daemon
        self.result: dict[str, float] = {}

    def start(self) -> None:
        """Snapshot CPU and memory at the start of the measured window."""
        self.daemon.reset_peak()
        self.mem0 = self.daemon.memory_kb()
        self.cpu0 = self.daemon.cpu_seconds()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self.client0 = ru.ru_utime + ru.ru_stime
        self.t0 = time.monotonic()

    def stop(self) -> dict[str, float]:
        """Close the window and return its resource deltas."""
        wall = time.monotonic() - self.t0
        ru = resource.getrusage(resource.RUSAGE_SELF)
        mem1 = self.daemon.memory_kb()
        before, after = self.mem0.get("VmRSS", 0), mem1.get("VmRSS", 0)
        self.result = {
            "wall_s": wall,
            "daemon_cpu_s": self.daemon.cpu_seconds() - self.cpu0,
            "client_cpu_s": ru.ru_utime + ru.ru_stime - self.client0,
            "daemon_rss_before_kb": before,
            "daemon_rss_after_kb": after,
            "daemon_rss_peak_kb": max(mem1.get("VmHWM", 0), before, after),
        }
        self.result["daemon_cpu_pct"] = 100 * self.result["daemon_cpu_s"] / wall if wall else 0
        self.result["client_cpu_pct"] = 100 * self.result["client_cpu_s"] / wall if wall else 0
        return self.result


# ----------------------------------------------------------- transports


class UdsConn:
    """One newline-delimited JSON connection to the daemon's Unix socket."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer
        self.bytes_in = 0

    async def send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode() + b"\n")
        await self.writer.drain()

    async def recv(self) -> dict[str, Any]:
        line = await self.reader.readline()
        if not line:
            raise ConnectionError("UDS closed")
        self.bytes_in += len(line)
        return dict(json.loads(line))

    async def close(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass


class WssConn:
    """One authenticated WebSocket connection to the daemon's WSS port."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.bytes_in = 0

    async def send(self, cmd: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(cmd))

    async def recv(self) -> dict[str, Any]:
        raw = await self.ws.recv()
        self.bytes_in += len(raw)
        return dict(json.loads(raw))

    async def close(self) -> None:
        await self.ws.close()


class Transport:
    """Factory for connections of one transport flavour."""

    def __init__(self, name: str, uds_path: str, port: int) -> None:
        self.name, self.uds_path, self.port = name, uds_path, port
        self.ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl.check_hostname = False
        self.ssl.verify_mode = ssl.CERT_NONE

    async def connect(self) -> UdsConn | WssConn:
        """Open one connection ready to accept commands."""
        if self.name == "uds":
            reader, writer = await asyncio.open_unix_connection(self.uds_path, limit=MAX_BYTES)
            return UdsConn(reader, writer)
        ws = await ws_connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=self.ssl, max_size=MAX_BYTES,
            compression="deflate" if self.name == "wss" else None,
            ping_interval=None,
        )
        conn = WssConn(ws)
        await conn.send({"type": "auth", "password": ""})
        first = await conn.recv()
        if first.get("type") != "auth_ok":
            raise RuntimeError(f"auth failed: {first}")
        return conn


async def wait_for(conn: UdsConn | WssConn, wanted: str, timeout: float = 30) -> dict[str, Any]:
    """Return the first event of type *wanted*, skipping others."""
    deadline = time.monotonic() + timeout
    while True:
        ev = await asyncio.wait_for(conn.recv(), timeout=max(0.01, deadline - time.monotonic()))
        if ev.get("type") == wanted:
            return ev


# ------------------------------------------------------------ workloads


async def bench_connect(tr: Transport, n: int, phase: Phase) -> dict[str, Any]:
    """Connection setup cost: to handshake done and to first pong.

    Here setup IS the workload, so the whole loop is measured.
    """
    setup: list[int] = []
    first_pong: list[int] = []
    phase.start()
    for _ in range(n):
        t0 = time.monotonic_ns()
        conn = await tr.connect()
        t1 = time.monotonic_ns()
        await conn.send({"type": "ping"})
        await wait_for(conn, "pong")
        t2 = time.monotonic_ns()
        await conn.close()
        setup.append(t1 - t0)
        first_pong.append(t2 - t0)
    return {"setup": percentiles(setup), "first_pong": percentiles(first_pong),
            "proc": phase.stop()}


async def bench_request_reply(
    tr: Transport, n: int, cmd: dict[str, Any], reply: str, phase: Phase,
) -> dict[str, Any]:
    """Sequential request/reply round trips on one persistent connection."""
    conn = await tr.connect()
    rtt: list[int] = []
    conn.bytes_in = 0
    phase.start()
    for _ in range(n):
        t0 = time.monotonic_ns()
        await conn.send(cmd)
        await wait_for(conn, reply)
        rtt.append(time.monotonic_ns() - t0)
    out = {"rtt": percentiles(rtt), "reply_bytes_avg": conn.bytes_in / n, "proc": phase.stop()}
    await conn.close()
    return out


async def _drain_burst(conn: UdsConn | WssConn, n: int) -> tuple[list[int], int, int]:
    """Receive *n* ``bench_event`` events; return one-way latencies."""
    lat: list[int] = []
    seen = 0
    while seen < n:
        ev = await asyncio.wait_for(conn.recv(), timeout=120)
        if ev.get("type") != "bench_event":
            continue
        lat.append(time.monotonic_ns() - ev["t0"])
        seen += 1
    return lat, time.monotonic_ns(), conn.bytes_in


async def bench_burst(
    tr: Transport, daemon: Child, clients: int, n: int, payload: int, gap_us: int,
    phase: Phase,
) -> dict[str, Any]:
    """Daemon-side broadcast burst to *clients* connections of *tr*."""
    conns = [await tr.connect() for _ in range(clients)]
    for c in conns:
        c.bytes_in = 0
    drains = [asyncio.create_task(_drain_burst(c, n)) for c in conns]
    await asyncio.sleep(0.2)
    phase.start()
    started = time.monotonic_ns()
    daemon.write(f"burst {n} {payload} {gap_us}")
    results = await asyncio.gather(*drains)
    ack = await asyncio.to_thread(daemon.expect, "BURST_DONE")
    proc = phase.stop()
    emit_ns = int(ack[1])
    for c in conns:
        await c.close()
    all_lat = [x for lat, _, _ in results for x in lat]
    last_recv = max(t for _, t, _ in results)
    total_bytes = sum(b for _, _, b in results)
    return {
        "latency": percentiles(all_lat),
        "emit_wall_s": emit_ns / 1e9,
        "deliver_wall_s": (last_recv - started) / 1e9,
        "events_per_s": clients * n / ((last_recv - started) / 1e9),
        "bytes_in_total": total_bytes,
        "per_client_p95_ms": [percentiles(lat)["p95_ms"] for lat, _, _ in results],
        "proc": proc,
    }


async def bench_task(
    tr: Transport, viewers: int, chunks: int, gap_us: int, model_url: str,
    work_dir: str, tab: str, phase: Phase,
) -> dict[str, Any]:
    """Run one real streamed agent task; time every ``text_delta``.

    The stand-in model streams *chunks* text deltas *gap_us* apart, so
    ``gap_us=0`` floods the pipeline and ``gap_us=5000`` mimics a fast
    model at 200 tokens/s.  *viewers* extra connections of the same
    transport watch the stream, like additional open surfaces.
    """
    runner = await tr.connect()
    extra = [await tr.connect() for _ in range(viewers)]
    runner.bytes_in = 0

    async def drain_viewer(c: UdsConn | WssConn) -> list[int]:
        lat: list[int] = []
        while True:
            ev = await asyncio.wait_for(c.recv(), timeout=180)
            if ev.get("type") == "text_delta":
                lat.append(time.monotonic_ns() - ev["t0"])
            elif ev.get("type") == "status" and ev.get("running") is False:
                return lat

    viewer_tasks = [asyncio.create_task(drain_viewer(c)) for c in extra]
    phase.start()
    t_start = time.monotonic_ns()
    await runner.send({
        "type": "run", "tabId": tab,
        "prompt": f"benchmark: stream text chunks={chunks} gap_us={gap_us}",
        "model": "gpt-4o-mini",
        "modelConfig": {"base_url": model_url, "api_key": "bench"},
        "workDir": work_dir, "useWorktree": False, "autoCommit": False,
        "classifyTasks": False, "webTools": False, "useMemory": False,
    })
    deltas: list[int] = []
    counts: dict[str, int] = {}
    first_delta_ns = 0
    while True:
        ev = await asyncio.wait_for(runner.recv(), timeout=180)
        typ = ev.get("type", "")
        counts[typ] = counts.get(typ, 0) + 1
        if typ == "text_delta":
            if not first_delta_ns:
                first_delta_ns = time.monotonic_ns()
            deltas.append(time.monotonic_ns() - ev["t0"])
        elif typ == "status" and ev.get("running") is False and ev.get("tabId") == tab:
            break
    t_end = time.monotonic_ns()
    viewer_lat = await asyncio.gather(*viewer_tasks) if extra else []
    proc = phase.stop()
    for c in [runner, *extra]:
        await c.close()
    return {
        "text_delta_latency": percentiles(deltas),
        "viewer_text_delta_latency": percentiles([x for lat in viewer_lat for x in lat]),
        "task_wall_s": (t_end - t_start) / 1e9,
        "stream_wall_s": (t_end - first_delta_ns) / 1e9 if first_delta_ns else 0,
        "events": counts,
        "runner_bytes_in": runner.bytes_in,
        "proc": proc,
    }


# -------------------------------------------------------------- driver


class Child:
    """A child process whose stdout lines are read by a thread into a queue.

    stderr goes to *log* so a chatty child can never block on a full pipe.
    """

    def __init__(self, cmd: list[str], env: dict[str, str], log: Path) -> None:
        self.name = Path(cmd[1]).name
        self.log = log
        self.log_file = log.open("w")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
            text=True, bufsize=1, env=env, cwd=str(HERE.parents[1]),
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")

    def expect(self, prefix: str, timeout: float = 120) -> list[str]:
        """Return the fields of the next stdout line starting with *prefix*."""
        deadline = time.monotonic() + timeout
        while True:
            line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            if not line:
                raise RuntimeError(f"{self.name} exited; see {self.log}")
            if line.startswith(prefix):
                return line.split()

    def write(self, line: str) -> None:
        """Send one control line to the child's stdin."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def stop(self, quit_line: str | None) -> None:
        """Ask the child to quit (or kill it) and close the log."""
        try:
            if quit_line is not None:
                self.write(quit_line)
                self.proc.wait(timeout=30)
            else:
                self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        self.log_file.close()


async def _one_pass(
    tr: Transport, daemon: Child, stats: ProcStats, model_url: str, work_dir: str,
    label: str, scale: float, ns: argparse.Namespace,
) -> dict[str, Any]:
    """Run every workload once on *tr*; *scale* shrinks the counts (warm-up)."""

    def count(n: int) -> int:
        return max(2, int(n * scale))

    out: dict[str, Any] = {}
    idle = Phase(stats)
    idle.start()
    await asyncio.sleep(2 * scale)  # daemon idle reference (no client connected)
    out["idle"] = {"proc": idle.stop()}
    out["connect"] = await bench_connect(tr, count(ns.connects), Phase(stats))
    out["ping"] = await bench_request_reply(
        tr, count(ns.pings), {"type": "ping"}, "pong", Phase(stats))
    out["models"] = await bench_request_reply(
        tr, count(ns.models_reqs), {"type": "getModels"}, "models", Phase(stats))
    for gap_us in ns.burst_gaps_us:
        for clients in ns.burst_clients:
            for payload in ns.burst_payloads:
                key = f"burst_c{clients}_b{payload}_g{gap_us}"
                out[key] = await bench_burst(
                    tr, daemon, clients, count(ns.burst_events), payload, gap_us, Phase(stats))
    for spec in ns.task_configs:
        chunks, gap_us, viewers = (int(x) for x in spec.split(":"))
        key = f"task_n{chunks}_g{gap_us}_v{viewers}"
        out[key] = await bench_task(
            tr, viewers, count(chunks), gap_us, model_url, work_dir,
            f"bench-{tr.name}-{label}-{key}", Phase(stats))
    return out


async def run_all(ns: argparse.Namespace) -> dict[str, Any]:
    """Start the processes, run every workload per transport, collect."""
    tmp = Path(tempfile.mkdtemp(prefix="kiss-transport-bench-"))
    kiss_home = tmp / "kiss_home"
    work_dir = tmp / "work"
    kiss_home.mkdir()
    work_dir.mkdir()
    (kiss_home / "config.json").write_text(json.dumps({
        "remote_password": "", "classify_tasks": False, "is_worktree": False,
        "auto_commit": False, "max_budget": 100,
    }))
    env = {**os.environ, "KISS_HOME": str(kiss_home), "PYTHONUNBUFFERED": "1"}
    env.pop("KISS_SORCAR_SOCK", None)

    logs = Path(ns.out).parent
    logs.mkdir(parents=True, exist_ok=True)
    model: Child | None = None
    daemon: Child | None = None
    try:
        model = Child([sys.executable, str(HERE / "fake_model.py"),
                       "--chunk-chars", str(ns.chunk_chars)], env, logs / "fake_model.log")
        model_port = int(model.expect("READY")[1])
        model_url = f"http://127.0.0.1:{model_port}/v1"

        port = ns.port or _free_port()
        daemon = Child([sys.executable, str(HERE / "bench_daemon.py"),
                        "--kiss-home", str(kiss_home), "--work-dir", str(work_dir),
                        "--port", str(port)], env, logs / "daemon.log")
        ready = daemon.expect("READY")
        daemon_pid, uds_path = int(ready[1]), ready[3]
        stats = ProcStats(daemon_pid, daemon)
        baseline_mem = stats.memory_kb()

        transports = [Transport(n, uds_path, port) for n in ns.transports]
        results: dict[str, Any] = {
            "config": vars(ns), "daemon_pid": daemon_pid,
            "daemon_baseline_rss_kb": baseline_mem.get("VmRSS", 0),
            "python": sys.version, "rounds": [],
        }
        # Warm-up: one full pass per transport with small counts, discarded,
        # so lazy imports (agent, model catalog) and caches settle before
        # any measured phase.
        for tr in transports:
            await _one_pass(tr, daemon, stats, model_url, str(work_dir), "warm", scale=0.05, ns=ns)
        for rnd in range(ns.rounds):
            # Rotate so every transport takes every position once per
            # len(transports) rounds (drift and thermal effects spread evenly).
            shift = rnd % len(transports)
            order = transports[shift:] + transports[:shift]
            round_out: dict[str, Any] = {}
            for tr in order:
                round_out[tr.name] = await _one_pass(
                    tr, daemon, stats, model_url, str(work_dir), str(rnd), 1.0, ns)
                print(f"round {rnd} {tr.name}: done", file=sys.stderr, flush=True)
            results["rounds"].append(round_out)
    finally:
        if daemon is not None:
            daemon.stop("quit")
        if model is not None:
            model.stop(None)
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def pooled(results: dict[str, Any]) -> dict[str, Any]:
    """Merge the rounds: pool latency samples' summaries by median of rounds."""
    out: dict[str, Any] = {}
    for rnd in results["rounds"]:
        for tr, wl in rnd.items():
            for key, val in wl.items():
                out.setdefault(tr, {}).setdefault(key, []).append(val)
    return out


def _fmt_table(results: dict[str, Any]) -> str:
    """Render one Markdown table row per (workload, transport)."""
    agg = pooled(results)
    lines = ["| workload | transport | p50 ms | p95 ms | p99 ms | daemon CPU % | client CPU % "
             "| daemon peak RSS MB |", "|---|---|---|---|---|---|---|---|"]
    for tr, wls in agg.items():
        for key, rounds in wls.items():
            lat_key = {"connect": "first_pong", "ping": "rtt", "models": "rtt"}.get(
                key, "latency" if key.startswith("burst") else "text_delta_latency")
            if not all(lat_key in r and "p50_ms" in r[lat_key] for r in rounds):
                continue
            p50 = statistics.median(r[lat_key]["p50_ms"] for r in rounds)
            p95 = statistics.median(r[lat_key]["p95_ms"] for r in rounds)
            p99 = statistics.median(r[lat_key]["p99_ms"] for r in rounds)
            dcpu = statistics.median(r["proc"]["daemon_cpu_pct"] for r in rounds)
            ccpu = statistics.median(r["proc"]["client_cpu_pct"] for r in rounds)
            rss = max(r["proc"]["daemon_rss_peak_kb"] for r in rounds) / 1024
            lines.append(f"| {key} | {tr} | {p50:.3f} | {p95:.3f} | {p99:.3f} | {dcpu:.1f} "
                         f"| {ccpu:.1f} | {rss:.0f} |")
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, run the benchmark, write JSON and print a table."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--transports", nargs="+", default=["uds", "wss", "wss-nocomp"])
    ap.add_argument("--connects", type=int, default=50)
    ap.add_argument("--pings", type=int, default=2000)
    ap.add_argument("--models-reqs", type=int, default=300)
    ap.add_argument("--burst-events", type=int, default=5000)
    ap.add_argument("--burst-payloads", nargs="+", type=int, default=[200, 4096])
    ap.add_argument("--burst-clients", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--burst-gaps-us", nargs="+", type=int, default=[0, 2000])
    ap.add_argument("--chunk-chars", type=int, default=40)
    ap.add_argument(
        "--task-configs", nargs="+", default=["1000:5000:0", "1000:5000:3", "2000:0:0"],
        help="real-task workloads as chunks:gap_us:viewers",
    )
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--out", default="tmp/transport_bench/results.json")
    ns = ap.parse_args()
    results = asyncio.run(run_all(ns))
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(_fmt_table(results))
    print(f"\nraw results: {out}")


if __name__ == "__main__":
    main()
