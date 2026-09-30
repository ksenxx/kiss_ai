# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real ``kiss-web`` daemon under benchmark control.

Starts a :class:`kiss.server.web_server.RemoteAccessServer` inside an
isolated ``KISS_HOME`` (so the developer's ``~/.kiss`` is never touched)
serving BOTH transports — the Unix-domain socket and the TLS WebSocket —
exactly as production does, then obeys one-line commands on stdin:

``burst <n> <payload_bytes> <gap_us>``
    Broadcast *n* global ``bench_event`` events of about *payload_bytes*
    bytes from a worker thread (the way an agent thread emits streamed
    tokens), sleeping *gap_us* microseconds between events.  Answers
    ``BURST_DONE <elapsed_ns>`` once the last broadcast was queued.
``cpu``
    Answer ``CPU <process_time_ns>`` — the daemon's CPU time over all
    threads at nanosecond resolution (``/proc/<pid>/stat`` only has
    10 ms ticks).
``quit``
    Stop the daemon.

Every broadcast event additionally carries ``t0`` — the daemon's
``time.monotonic_ns()`` at emission — so a client on the same machine
can compute one-way delivery latency (``CLOCK_MONOTONIC`` is shared by
all processes on a Linux host).

Prints ``READY <pid> <port> <uds_path>`` on stdout once both transports
are listening.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kiss-home", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    return ap.parse_args()


def _install_t0_stamp(printer: Any) -> None:
    """Stamp ``t0`` (monotonic ns) on every event the printer broadcasts."""
    original = printer.broadcast

    def stamped_broadcast(event: dict[str, Any]) -> None:
        event.setdefault("t0", time.monotonic_ns())
        original(event)

    printer.broadcast = stamped_broadcast


def _run_burst(printer: Any, n: int, payload_bytes: int, gap_us: int) -> int:
    """Broadcast *n* padded events from the calling (worker) thread."""
    pad = "x" * max(0, payload_bytes - 60)
    started = time.monotonic_ns()
    for seq in range(n):
        printer.broadcast({"type": "bench_event", "seq": seq, "n": n, "pad": pad})
        if gap_us:
            time.sleep(gap_us / 1e6)
    return time.monotonic_ns() - started


async def _serve(ns: argparse.Namespace) -> None:
    from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

    home = Path(ns.kiss_home)
    certfile, keyfile = home / "cert.pem", home / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)
    uds_path = home / "sorcar.sock"
    server = RemoteAccessServer(
        host="127.0.0.1",
        port=ns.port,
        work_dir=ns.work_dir,
        certfile=str(certfile),
        keyfile=str(keyfile),
        url_file=home / "remote-url.json",
        uds_path=uds_path,
        uds_owner_wait_s=1.0,
    )
    await server.start_async()
    printer = server._printer
    _install_t0_stamp(printer)
    print(f"READY {os.getpid()} {ns.port} {uds_path}", flush=True)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin,
    )
    try:
        while True:
            line = (await reader.readline()).decode().strip()
            if not line or line == "quit":
                break
            parts = line.split()
            if parts[0] == "burst" and len(parts) == 4:
                elapsed = await asyncio.to_thread(
                    _run_burst, printer, int(parts[1]), int(parts[2]), int(parts[3]),
                )
                print(f"BURST_DONE {elapsed}", flush=True)
            elif parts[0] == "cpu":
                print(f"CPU {time.process_time_ns()}", flush=True)
            else:
                print(f"ERROR unknown command {line!r}", flush=True)
    finally:
        await server.stop_async()


def main() -> None:
    """Run the benchmark daemon until ``quit`` or EOF on stdin."""
    ns = _parse()
    os.environ["KISS_HOME"] = ns.kiss_home
    os.environ.pop("KISS_SORCAR_SOCK", None)
    asyncio.run(_serve(ns))


if __name__ == "__main__":
    main()
