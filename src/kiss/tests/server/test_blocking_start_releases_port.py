"""A blocking ``start()`` stopped in-process must not stall the next server.

Regression test for the per-test stall in the server suite:
``TestStartMethodLifecycle`` ran ``RemoteAccessServer.start()`` on a
thread, stopped its loop, and left the WSS listener open in the test
process.  Every later test that bound the same port then saw
``EADDRINUSE`` and walked the whole bind-retry backoff, and the leaked
endpoint file kept advertising a daemon no loop served.
``close_leaked_listeners`` is the shared clean-up those tests now call;
this test checks that it actually frees the port and the endpoint file
for a successor.

Both servers here share a test-local port, ``local_endpoint_file`` and
``url_file`` under ``tmp_path`` so the test never touches the
session-wide endpoint file or URL marker other tests rely on.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import local_endpoint
from kiss.core.vscode_config import save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server._blocking_start import close_leaked_listeners


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_server(tmp_path: Path, name: str, port: int) -> RemoteAccessServer:
    """Build a tunnel-less server on *port* whose files all live under ``tmp_path``."""
    work_dir = tmp_path / name
    work_dir.mkdir()
    return RemoteAccessServer(
        host="127.0.0.1", port=port, use_tunnel=False,
        work_dir=str(work_dir), url_file=tmp_path / f"{name}-url.json",
        local_endpoint_file=tmp_path / "sorcar-local.json",
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


async def _bind_successor(successor: RemoteAccessServer) -> tuple[str, float]:
    """Start ``successor``; report the token its endpoint file advertises and
    how long ``start_async`` took (its own teardown is not part of the
    measurement)."""
    started = time.monotonic()
    await successor.start_async()
    elapsed = time.monotonic() - started
    try:
        endpoint = local_endpoint.read_endpoint(successor._local_endpoint_file)
        assert endpoint is not None, "successor published no endpoint file"
        return endpoint.token, elapsed
    finally:
        await successor.stop_async()


def test_stopped_blocking_start_releases_port_for_successor(tmp_path: Path) -> None:
    save_config({"remote_password": ""})
    port = _free_port()
    server = _make_server(tmp_path, "first", port)
    thread = _start_on_thread(server)
    try:
        assert server._ws_server is not None, "start() did not bind the WSS listener"
        endpoint_file = server._local_endpoint_file
        assert endpoint_file.exists(), "start() did not publish the endpoint file"
    finally:
        # Stop the server even when the assertion fails: a server left
        # running keeps its cron scheduler thread alive and breaks the
        # cron tests that run later in the same pytest process.
        _stop_thread(server, thread)

    close_leaked_listeners(server)

    assert server._ws_server is None
    assert server._ws_loopback_server is None
    assert not endpoint_file.exists(), "stopped server left its endpoint file behind"
    # ``server`` is still referenced here, so without the clean-up the
    # leaked listener would still hold the port and the successor would
    # walk the bind-retry backoff (0.5 s + 1 s + 2 s + ...) before giving up.
    successor = _make_server(tmp_path, "second", port)
    token, elapsed = asyncio.run(_bind_successor(successor))
    assert token == successor._local_token, "endpoint file does not name the successor"
    assert token != server._local_token
    assert elapsed < 3.0, "successor waited on a dead predecessor's port"
    del server
