# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the atomic in-flight lease in ``MCPManager.call_tool``.

``call_tool`` used to look up the connection under the manager lock,
RELEASE the lock, and only then increment ``in_flight`` under a second
acquisition.  In that window a concurrent :meth:`MCPManager.connect`'s
``_evict_surplus`` saw ``in_flight == 0``, evicted the connection, and
unwound the session underneath the call about to run on it.  The lease
is now taken atomically with the lookup (``MCPManager._lease``).

These tests spawn REAL FastMCP stdio servers and speak the real
protocol over real pipes; no mocks, patches or doubles of the system
under test.
"""

from __future__ import annotations

import os
import pty
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    MCPManager,
    MCPServerConfig,
    _connection_key,
)

_SERVER_SCRIPT = '''
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("leasesrv")


@mcp.tool()
def slow(seconds: float) -> str:
    """Sleep for the given number of seconds, then return "done"."""
    time.sleep(seconds)
    return "done"


@mcp.tool()
def ping() -> str:
    """Return the string "pong"."""
    return "pong"


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    Same plumbing as ``test_sorcar_mcp.py``: under pytest the std
    streams are in-memory capture objects whose ``.fileno()`` raises,
    so the stdio transport cannot spawn its child.  A pty gives stdin a
    real descriptor and a plain file serves as the child's stderr.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    from mcp.client.stdio import stdio_client

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


def _stdio_config(tmp_path: Path, name: str) -> MCPServerConfig:
    """Return a stdio config running the FastMCP lease-test server."""
    script = tmp_path / f"{name}.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(script),),
    )


class _EvictionGateLock:
    """A drop-in for the manager lock that runs eviction after releases.

    Every acquire/release delegates to a real lock.  While *armed*, each
    release performed by the armed thread triggers a hook that runs the
    manager's own :meth:`MCPManager._evict_surplus` — the exact routine
    a concurrent ``connect()`` runs.  The first trigger therefore lands
    deterministically in the window the old ``call_tool`` left open
    between its connection lookup (first lock release) and its
    ``in_flight`` increment (second acquisition); the fixed ``_lease``
    leaves no such window, because by its first release the lease is
    already taken.

    The hook also records the first release after which the connection
    was observed leased (``in_flight > 0``): it backdates ``last_used``
    so the connection is idle-expired, runs another eviction pass, and
    stores whether the leased connection survived it.
    """

    def __init__(self, manager: MCPManager, key: str) -> None:
        self._inner = threading.Lock()
        self._manager = manager
        self._key = key
        self._armed_ident: int | None = None
        self._in_hook = False
        self._first_attack_done = False
        #: ``(survived_eviction, in_flight_seen)`` from the eviction
        #: pass run while the connection was observed leased.
        self.leased_attack: tuple[bool, int] | None = None

    def arm(self) -> None:
        """Fire the hook on this thread's lock releases from now on."""
        self._armed_ident = threading.get_ident()

    def disarm(self) -> None:
        """Stop firing the hook."""
        self._armed_ident = None

    def acquire(self, *args: object, **kwargs: object) -> bool:
        """Acquire the underlying lock."""
        return self._inner.acquire(*args, **kwargs)  # type: ignore[arg-type]

    def release(self) -> None:
        """Release the underlying lock, then maybe run the hook."""
        self._inner.release()
        self._after_release()

    def __enter__(self) -> _EvictionGateLock:
        self._inner.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.release()
        self._after_release()

    def _after_release(self) -> None:
        if self._in_hook or threading.get_ident() != self._armed_ident:
            return
        self._in_hook = True
        try:
            manager = self._manager
            conn = manager._connections.get(self._key)
            if not self._first_attack_done:
                self._first_attack_done = True
                # Keep the manager loop busy for a moment so the (old,
                # racy) caller captures the still-live session before
                # the eviction's teardown lands — the exact production
                # interleaving that stranded the call.
                manager._loop.call_soon_threadsafe(time.sleep, 0.5)
                manager._evict_surplus("")
            elif (
                conn is not None
                and conn.in_flight > 0
                and self.leased_attack is None
            ):
                # The lease is held: even an idle-expired connection
                # must survive a full eviction pass.
                conn.last_used = time.monotonic() - 1000.0
                manager._evict_surplus("")
                survived = manager._connections.get(self._key) is conn
                self.leased_attack = (survived, conn.in_flight)
        finally:
            self._in_hook = False


def test_lease_is_atomic_with_lookup_under_forced_eviction(
    real_stdin: None, tmp_path: Path,
) -> None:
    """Eviction forced into the old lookup-to-increment window is safe.

    The gate lock runs a real eviction pass immediately after the
    calling thread's first lock release inside ``call_tool``.  Against
    the old split-lock implementation that release happened *after* the
    connection lookup but *before* the ``in_flight`` increment, so the
    eviction tore down the session the call was about to run on and the
    call failed.  With the atomic ``_lease`` the same interleaving is
    harmless — the call must return the real tool result — and a later
    eviction pass must have actually observed the lease (``in_flight >
    0``) and left the connection alone.
    """
    manager = MCPManager(idle_timeout_s=0.05, max_connections=1)
    try:
        config_a = _stdio_config(tmp_path, "srv_a")
        conn_a = manager.connect(config_a)
        assert conn_a.error == "", conn_a.error
        key_a = _connection_key(config_a)
        # Let the idle timeout expire so the forced pass really evicts.
        time.sleep(0.1)

        gate = _EvictionGateLock(manager, key_a)
        manager._lock = gate  # type: ignore[assignment]
        gate.arm()
        try:
            result = manager.call_tool(key_a, "ping", {})
        finally:
            gate.disarm()

        assert result == "pong", result
        assert gate.leased_attack is not None, (
            "the eviction pass never observed the in-flight lease"
        )
        survived, in_flight_seen = gate.leased_attack
        assert survived, "a leased connection was evicted"
        assert in_flight_seen > 0, in_flight_seen
        with manager._lock:
            conn = manager._connections.get(key_a)
            assert conn is not None
            assert conn.in_flight == 0, "the lease was not released"
    finally:
        manager.shutdown()


def test_inflight_call_survives_eviction_pressure(
    real_stdin: None, tmp_path: Path,
) -> None:
    """A leased connection is never evicted, even when idle-expired.

    The pool is capped at one connection and the idle timeout is tiny,
    so the second ``connect`` puts maximal eviction pressure on the
    first server — which by then is BOTH over the cap and idle-expired,
    but has a slow tool call in flight.  The call must complete.
    """
    manager = MCPManager(idle_timeout_s=0.05, max_connections=1)
    try:
        config_a = _stdio_config(tmp_path, "srv_a")
        conn_a = manager.connect(config_a)
        assert conn_a.error == "", conn_a.error
        key_a = _connection_key(config_a)

        results: list[str] = []

        def _slow_call() -> None:
            results.append(manager.call_tool(key_a, "slow", {"seconds": 2}))

        worker = threading.Thread(target=_slow_call)
        worker.start()
        # Let the call get in flight AND the idle timeout expire.
        observed_lease = False
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with manager._lock:
                conn = manager._connections.get(key_a)
                if conn is not None and conn.in_flight > 0:
                    observed_lease = True
                    break
            time.sleep(0.01)
        assert observed_lease, "never saw the call's lease on the connection"

        config_b = _stdio_config(tmp_path, "srv_b")
        conn_b = manager.connect(config_b)
        assert conn_b.error == "", conn_b.error

        worker.join(timeout=30)
        assert not worker.is_alive(), "slow tool call never returned"
        assert results == ["done"], results
    finally:
        manager.shutdown()


def test_call_after_eviction_reconnects_and_leases(
    real_stdin: None, tmp_path: Path,
) -> None:
    """A call on an evicted connection transparently reconnects."""
    manager = MCPManager(idle_timeout_s=0.05, max_connections=1)
    try:
        config_a = _stdio_config(tmp_path, "srv_a")
        conn_a = manager.connect(config_a)
        assert conn_a.error == "", conn_a.error
        key_a = _connection_key(config_a)

        time.sleep(0.1)
        config_b = _stdio_config(tmp_path, "srv_b")
        conn_b = manager.connect(config_b)
        assert conn_b.error == "", conn_b.error
        # srv_a was idle beyond the timeout with nothing in flight: the
        # second connect must have evicted it.
        with manager._lock:
            assert key_a not in manager._connections

        # The next call rebuilds the connection and runs for real.
        assert manager.call_tool(key_a, "ping", {}) == "pong"
        with manager._lock:
            releases = manager._connections[key_a].in_flight
        assert releases == 0, "the lease was not released"
    finally:
        manager.shutdown()
