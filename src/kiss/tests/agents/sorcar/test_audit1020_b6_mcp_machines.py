# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, slice B6 (mcp_servers / mcp_oauth / machines).

F1: a connection whose handshake lands in the instant its connect
deadline passes was orphaned under a live lease (the session closed
under another task's tool call, the healthy child killed).
F2: ``FileTokenStorage.clear`` unlinked outside the inter-process lock
``put`` holds, so a concurrent refresh re-created the stale file.
F3: ``MCPLoginSession.start`` ignored the cooperative tool interrupt
while waiting for the authorization URL.
F4: ``_cached_tunnel`` closed the superseded ssh tunnel while holding
``_tunnels_lock``, stalling every other machine's tunnel lookup for
as long as the old forward's login took.

Real manager, real FastMCP stdio child, real file locks, real sockets
and a fake ``ssh`` on PATH; no mocks of the code under test.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_oauth import MCPLoginSession
from kiss.agents.sorcar.mcp_servers import FileTokenStorage, MCPManager, MCPServerConfig
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import locked_call
from kiss.core.processes import pid_alive
from kiss.tests.agents.sorcar.test_audit1015_sorcar_tools_mcp_oauth_wait_sliced import (
    _run_interrupted,
)
from kiss.tests.agents.sorcar.test_machines import fake_ssh_only
from kiss.tests.agents.sorcar.test_mcp_connection_lifecycle import (
    _config,
    _wait_for_pid_file,
    real_stdin,  # noqa: F401  (fixture)
    server_script,  # noqa: F401  (fixture)
)


@pytest.fixture
def manager() -> Iterator[MCPManager]:
    """A private manager, always shut down."""
    mgr = MCPManager(idle_timeout_s=60.0, max_connections=4, health_interval_s=5.0)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def test_handshake_landing_at_the_deadline_keeps_the_leased_connection(
    manager: MCPManager,
    server_script: Path,  # noqa: F811
    tmp_path: Path,
    real_stdin: None,  # noqa: F811
) -> None:
    """F1: ``_await_ready`` past its deadline must keep a connection that just became ready."""
    pid_file = tmp_path / "pid"
    key, conn = manager._start(_config(server_script, pid_file))
    waiter_error: list[BaseException] = []

    def late_waiter() -> None:
        # The deadline is already over: the wait returns at once and the
        # orphaning step queues on the manager lock the test holds.
        try:
            manager._await_ready(key, conn, time.monotonic())
        except BaseException as exc:  # noqa: BLE001
            waiter_error.append(exc)

    with manager._lock:
        waiter = threading.Thread(target=late_waiter, name="late-waiter")
        waiter.start()
        # The handshake lands while the waiter is queued on the lock ...
        assert conn.ready.wait(30.0) and conn.error == ""
        # ... and another task leases the connection before it gets in.
        leased, session = manager._lease(key)
        assert leased is conn and session is not None
    waiter.join(30.0)
    assert not waiter.is_alive() and waiter_error == []
    try:
        assert manager._connections.get(key) is conn
        assert conn.error == ""
        assert not conn.stop.is_set()
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool("add", {"a": 2, "b": 3}, read_timeout_seconds=timedelta(seconds=20)),
            manager._loop,
        )
        assert mcp_servers._result_text(future.result(30.0)) == "5"
    finally:
        locked_call(manager._lock, mcp_servers._release_lease, conn)
    assert pid_alive(_wait_for_pid_file(pid_file))
    assert manager.call_tool(key, "add", {"a": 1, "b": 1}) == "2"
    assert manager._orphans == []


def test_clear_waits_for_a_concurrent_token_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2: ``clear`` takes the store lock, so a paused read-modify-write cannot undo it."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    storage = FileTokenStorage("srv")
    storage.put("client_info", {"client_id": "stale"})
    tokens = {"access_token": "t", "token_type": "bearer"}
    cleared: list[bool] = []
    clear_done = threading.Event()
    written = threading.Event()

    def clearer() -> None:
        cleared.append(storage.clear())
        clear_done.set()

    def refresher() -> None:
        # The real writer: a token refresh queued behind the lock holder.
        storage.put("tokens", tokens)
        written.set()

    # A refresh in another kiss process holds the store lock: neither
    # the clear nor a further write may land until it lets go, however
    # long it takes, so there is no amount of time after which an
    # early ``clear`` would be excused.
    with exclusive_file_lock(storage._lock_path):
        threads = [
            threading.Thread(target=clearer, name="clearer"),
            threading.Thread(target=refresher, name="refresher"),
        ]
        for thread in threads:
            thread.start()
        assert not clear_done.wait(0.5), "clear did not wait for the lock holder"
        assert not written.is_set()
        assert storage.path.exists()
    for thread in threads:
        thread.join(10.0)
        assert not thread.is_alive(), thread.name
    assert cleared == [True]
    # The two land in either order once the holder is gone; the write
    # starts from the stored data of its turn, so the stale client
    # registration never survives the clear by riding on a refresh.
    if storage.path.exists():
        assert storage._read() == {"tokens": tokens, "expires_at": None}
        assert storage.clear() is True
    assert not storage.path.exists()
    assert storage.clear() is False


def test_start_url_wait_stops_within_a_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mcp_redirect_port: int,
) -> None:
    """F3: a Stop during ``start``'s URL wait is honoured in the next 0.1 s slice."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KISS_MCP_CLIENT_METADATA_URL", raising=False)
    # Accepts the connection and never answers: no URL, no finish.
    mute = socket.socket()
    mute.bind(("127.0.0.1", 0))
    mute.listen()
    cfg = MCPServerConfig(
        name="mute",
        transport="http",
        url=f"http://127.0.0.1:{mute.getsockname()[1]}/mcp",
    )
    previous = MCPLoginSession._active
    try:
        outcome, elapsed = _run_interrupted(
            lambda: MCPLoginSession.start(cfg, wait_seconds=30.0),
            "connect_mcp_server",
        )
        assert outcome == {"interrupted": True}
        # The cooperative check lands within one slice; the watchdog's
        # injected interrupt (the only exit before the fix) takes 1 s.
        assert elapsed < 0.5, elapsed
        session = MCPLoginSession.active("mute")
        assert session is not None and not session.ready and not session.done
    finally:
        session = MCPLoginSession._active
        if session is not None and session is not previous:
            session.cancel()
            session.wait(5.0)
        MCPLoginSession._active = previous
        mute.close()


_SLOW_FAKE_SSH = '''#!__PYTHON__
"""Stand-in for ssh: ``-N -L`` logs in once $KISS_TEST_SSH_RELEASE exists, then binds the port."""
import os
import socket
import sys
import time

args = sys.argv[1:]
_host, local, _rhost, _remote = args[args.index("-L") + 1].split(":")
while not os.path.exists(os.environ["KISS_TEST_SSH_RELEASE"]):
    time.sleep(0.02)
server = socket.socket()
server.bind(("127.0.0.1", int(local)))
server.listen()
while True:
    time.sleep(1)
'''


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    """Poll *condition* for up to 10 s; fail with *what* when it never holds."""
    deadline = time.monotonic() + 10.0
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


@fake_ssh_only
def test_superseded_tunnel_is_closed_outside_the_tunnels_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F4: replacing a tunnel mid-login must not hold ``_tunnels_lock`` for the login."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ssh"
    script.write_text(_SLOW_FAKE_SSH.replace("__PYTHON__", sys.executable), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(tmp_path / "no-such-config"))
    # The fake ssh's login lasts until this file exists: the old
    # forward is mid-login for exactly as long as the test wants.
    release = tmp_path / "ssh-login-done"
    monkeypatch.setenv("KISS_TEST_SSH_RELEASE", str(release))
    login = "ksen@far.example.org"
    outcomes: dict[int, object] = {}

    def opener(remote_port: int) -> None:
        try:
            outcomes[remote_port] = mach.tunnel_port(login, remote_port)
        except BaseException as exc:  # noqa: BLE001
            outcomes[remote_port] = exc

    def first_ssh_started() -> bool:
        return login in mach._tunnels and mach._tunnels[login].process is not None

    def tunnel_replaced() -> bool:
        return mach._tunnels.get(login) is not stale

    mach.close_tunnels()
    first = threading.Thread(target=opener, args=(8787,), name="opener-8787")
    second = threading.Thread(target=opener, args=(9797,), name="opener-9797")
    closer = threading.Thread(
        target=mach.close_tunnels, kwargs={"keep": {login}}, name="close-tunnels",
    )
    first.start()
    try:
        _wait_for(first_ssh_started, "the first forward never started its ssh")
        stale = mach._tunnels[login]
        # The daemon moved: the second opener replaces the tunnel while
        # the first is still inside its login (which holds the stale
        # tunnel's own lock, so the replaced tunnel's close must wait).
        second.start()
        _wait_for(tunnel_replaced, "the tunnel was not replaced")
        # Another registry write: it must not queue behind that close.
        # With the close under ``_tunnels_lock`` this returns only once
        # the login ends, which it never does before the release below.
        closer.start()
        closer.join(10.0)
        assert not closer.is_alive(), "the tunnel cache was held for the old login"
        release.touch()
        first.join(30.0)
        second.join(30.0)
        assert not first.is_alive() and not second.is_alive()
        assert isinstance(outcomes[8787], mach.MachineError)
        assert "the machine registry changed, try again" in str(outcomes[8787])
        assert isinstance(outcomes[9797], int)
        assert stale.process is None and not stale.alive()
        assert mach._tunnels[login].remote_port == 9797 and mach._tunnels[login].alive()
    finally:
        release.touch()
        first.join(30.0)
        second.join(30.0)
        closer.join(30.0)
        mach.close_tunnels()
        stale_left = mach._tunnels.get(login)
        if stale_left is not None:
            stale_left.close()
