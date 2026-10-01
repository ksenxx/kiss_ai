# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real Sorcar daemon for the extension host's end-to-end JS tests.

Started by ``test/_realDaemon.js`` as ``uv run python _real_daemon.py
<work_dir>``: serves a real :class:`RemoteAccessServer` on a loopback
WSS port and publishes it in the local endpoint file under
``$KISS_HOME`` (the temp home the JS test set up, which the compiled
extension host reads to connect), prints ``READY`` once it listens, and
exits when its stdin closes.

Every non-empty stdin line is a JSON event the test wants broadcast to
the connected clients through the daemon's own printer — a stand-in for
the events a running agent task would emit (``worktree_created``, ...),
so a test can put a tab into a state (a pending worktree) that only a
real task would otherwise reach.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys

from kiss.agents.sorcar.local_endpoint import default_endpoint_path
from kiss.server.web_server import RemoteAccessServer


def _free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def main(work_dir: str) -> None:
    """Serve until stdin closes, broadcasting each stdin JSON line."""
    server = RemoteAccessServer(
        host="127.0.0.1",
        port=_free_port(),
        work_dir=work_dir,
        local_endpoint_file=default_endpoint_path(),
        use_tunnel=False,
    )
    await server.start_async()
    print("READY", flush=True)
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        if line.strip():
            server._printer.broadcast(json.loads(line))
    await server.stop_async()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
