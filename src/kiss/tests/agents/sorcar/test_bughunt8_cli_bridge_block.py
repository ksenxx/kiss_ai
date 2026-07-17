# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8: the CLI→daemon bridge must never block forever.

:mod:`kiss.ui.cli.cli_daemon_bridge` documents itself as a
*best-effort* one-way bridge: "Connection failures are absorbed
silently (CLI must still work when no daemon is running)".  The
pre-fix transport used a plain blocking ``socket.sendall`` with no
timeout, so a daemon that accepted the UDS connection but stopped
reading (wedged event loop, paused process, stuck under a debugger)
made ``send_event`` block indefinitely as soon as the kernel socket
buffer filled — freezing the whole CLI agent loop, which emits every
display event through this bridge.

This test spins a real AF_UNIX server that accepts the connection and
then never reads, pushes several megabytes of events through
:func:`send_event` on a worker thread, and asserts the worker finishes
within a generous deadline instead of hanging forever.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path

from kiss.ui.cli import cli_daemon_bridge


class TestBridgeNeverBlocksForever:
    def test_send_event_returns_when_daemon_stops_reading(self) -> None:
        # AF_UNIX paths are capped at ~104 bytes on macOS, so use a
        # short /tmp directory instead of pytest's deep tmp_path.
        tmp_dir = Path(tempfile.mkdtemp(prefix="kiss-bh8-", dir="/tmp"))
        sock_path = tmp_dir / "wedged.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(4)
        accepted: list[socket.socket] = []
        stop = threading.Event()

        def accept_loop() -> None:
            # Accept every connection but NEVER read from it — the
            # wedged-daemon scenario.
            server.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                accepted.append(conn)

        acceptor = threading.Thread(target=accept_loop, daemon=True)
        acceptor.start()

        old_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = str(sock_path)
        done = threading.Event()

        def sender() -> None:
            # 8 x 1 MiB events vastly exceed any UDS kernel buffer, so
            # a blocking sendall with no timeout wedges here forever.
            payload = "A" * (1024 * 1024)
            for _ in range(8):
                cli_daemon_bridge.send_event(
                    {"type": "text_delta", "text": payload},
                )
            done.set()

        worker = threading.Thread(target=sender, daemon=True)
        try:
            worker.start()
            assert done.wait(timeout=30.0), (
                "send_event blocked forever on a daemon that accepted "
                "the connection but never reads"
            )
        finally:
            if old_env is None:
                os.environ.pop("KISS_SORCAR_SOCK", None)
            else:
                os.environ["KISS_SORCAR_SOCK"] = old_env
            stop.set()
            acceptor.join(timeout=2)
            for conn in accepted:
                try:
                    conn.close()
                except OSError:
                    pass
            server.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
