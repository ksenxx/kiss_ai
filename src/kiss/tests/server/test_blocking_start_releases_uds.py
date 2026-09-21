"""A blocking ``start()`` stopped in-process must not stall the next server.

Regression test for the 30-second-per-test stall in the server suite:
``TestStartMethodLifecycle`` ran ``RemoteAccessServer.start()`` on a
thread, stopped its loop, and left the UDS listener open in the test
process.  Every later test that bound the shared
``$KISS_HOME/sorcar.sock`` then waited ``uds_owner_wait_s`` for a
"predecessor" that was only a leaked socket.  ``close_leaked_listeners``
is the shared clean-up those tests now call; this test checks that it
actually frees the pathname for a successor.

Both servers here share a test-local ``uds_path`` and ``url_file`` under
``tmp_path`` so the test never touches the session-wide socket or URL
marker other tests rely on.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

from kiss.core.vscode_config import save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server._blocking_start import close_leaked_listeners


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_server(
    tmp_path: Path, name: str, uds_owner_wait_s: float = 30.0,
) -> RemoteAccessServer:
    """Build a tunnel-less server whose files all live under ``tmp_path``."""
    work_dir = tmp_path / name
    work_dir.mkdir()
    return RemoteAccessServer(
        host="127.0.0.1", port=_free_port(), use_tunnel=False,
        work_dir=str(work_dir), url_file=tmp_path / f"{name}-url.json",
        uds_path=tmp_path / "sorcar.sock", uds_owner_wait_s=uds_owner_wait_s,
    )


def _run_start(server: RemoteAccessServer) -> None:
    """Thread body: ``loop.stop()`` makes ``asyncio.run`` raise RuntimeError."""
    try:
        server.start()
    except RuntimeError:
        pass


def _start_on_thread(server: RemoteAccessServer) -> threading.Thread:
    """Run ``server.start()`` on a daemon thread and wait until it accepts TCP."""
    thread = threading.Thread(target=_run_start, args=(server,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            with socket.create_connection(("127.0.0.1", server.port), timeout=1):
                return thread
        except OSError:
            continue
    raise AssertionError("server.start() never opened its TCP port")


def _stop_thread(server: RemoteAccessServer, thread: threading.Thread) -> None:
    """Stop the loop the way the lifecycle tests do and join the thread."""
    assert server._loop is not None
    server._loop.call_soon_threadsafe(server._loop.stop)
    thread.join(timeout=10)
    assert not thread.is_alive()


async def _bind_successor_uds(successor: RemoteAccessServer) -> bool:
    """Start ``successor`` and report whether it got the UDS listener."""
    await successor.start_async()
    try:
        return successor._uds_server is not None
    finally:
        await successor.stop_async()


@requires_unix_sockets
def test_stopped_blocking_start_releases_uds_for_successor(tmp_path: Path) -> None:
    save_config({"remote_password": ""})
    server = _make_server(tmp_path, "first")
    thread = _start_on_thread(server)
    assert server._uds_server is not None, "start() did not bind the UDS listener"
    uds_path = server._uds_path
    _stop_thread(server, thread)

    close_leaked_listeners(server)

    assert server._uds_server is None
    assert server._ws_server is None
    assert not uds_path.exists(), "stopped server left its socket pathname behind"
    # ``server`` is still referenced here, so without the clean-up the
    # leaked listener would still be accepting and the successor would
    # give up on its UDS after ``uds_owner_wait_s`` (kept short so a
    # regression fails in seconds rather than the 30 s default).
    successor = _make_server(tmp_path, "second", uds_owner_wait_s=3.0)
    started = time.monotonic()
    assert asyncio.run(_bind_successor_uds(successor))
    assert time.monotonic() - started < 3.0, "successor waited on a dead predecessor"
    del server
