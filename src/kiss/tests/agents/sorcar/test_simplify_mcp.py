# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the MCP race fixes (PART=mcp).

* M1 — ``MCPManager._evict_surplus`` never evicts a connection whose
  handshake is still running: the ``connect()`` waiting on it would
  otherwise be told "evicted" instead of getting its session.
* M2 — ``MCPManager.disconnect_all`` bounds its waits with one shared
  deadline per phase, so a shutdown with several hung servers takes
  ``2 × _DISCONNECT_WAIT_S`` at most instead of that much per server.
* M3 — ``MCPLoginSession`` opens the sign-in page on its own daemon
  thread, so a flow that finishes while the opener still lingers is
  reported ``done`` at once instead of once the opener returns.
* M4 — ``MCPLoginSession.start`` cancels and waits for the previous
  session outside the class lock, so ``active()`` never stalls behind
  that wait.

Real subprocess MCP servers, a real OAuth-protected MCP server and real
sockets throughout; no mocks.  The harness for the OAuth server is the
one of ``test_mcp_oauth_login.py``.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_oauth import MCPLoginSession, _session_answer
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig, _connection_key
from kiss.tests.agents.sorcar.test_mcp_oauth_login import (
    _approve,
    _AuthMCPServer,
    _cfg,
    _own_redirect_port,
    home,
    server,
)
from kiss.tests.agents.sorcar.test_wave2_mcp_bugs import (
    _SERVER_SCRIPT,
    _stdio_config,
    real_stdin,
)

__all__ = ["_own_redirect_port", "home", "real_stdin", "server"]

_SLOW_SERVER_SCRIPT = '''
import time

time.sleep(3)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slowsrv")


@mcp.tool()
def ping() -> str:
    """Return pong."""
    return "pong"


if __name__ == "__main__":
    mcp.run()
'''


# ---------------------------------------------------------------------------
# M1 — a connection mid-handshake is never evicted
# ---------------------------------------------------------------------------


def test_handshake_in_progress_is_not_evicted(tmp_path: Path, real_stdin: None) -> None:
    """A fast connect during a slow one must not evict the slow one.

    Pre-fix, with the pool cap reached, ``_evict_surplus`` treated the
    not-yet-ready connection as an ordinary idle candidate and tore it
    down, so its ``connect()`` returned ``error="evicted: ..."``.
    """
    manager = MCPManager(max_connections=1)
    slow_cfg = _stdio_config(tmp_path, "slowsrv", _SLOW_SERVER_SCRIPT)
    fast_cfg = _stdio_config(tmp_path, "testsrv", _SERVER_SCRIPT)
    slow_conns: list[mcp_servers._Connection] = []

    def connect_slow() -> None:
        slow_conns.append(manager.connect(slow_cfg))

    try:
        waiter = threading.Thread(target=connect_slow, daemon=True)
        waiter.start()
        time.sleep(1.0)
        assert waiter.is_alive(), "the slow server must still be in its handshake"
        fast = manager.connect(fast_cfg)
        assert fast.error == "" and fast.session is not None
        waiter.join(timeout=30)
        assert not waiter.is_alive()
        slow = slow_conns[0]
        assert slow.error == "", "the handshake in progress was evicted"
        assert slow.session is not None
        assert manager.call_tool(_connection_key(slow_cfg), "ping", {}) == "pong"
        assert manager.call_tool(_connection_key(fast_cfg), "add", {"a": 1, "b": 2}) == "3"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# M2 — disconnect_all's waits share one deadline
# ---------------------------------------------------------------------------


def _silent_config(name: str) -> MCPServerConfig:
    """A real child that never speaks MCP, so its handshake never ends."""
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=("-c", "import time; time.sleep(120)"),
    )


@pytest.mark.slow
def test_disconnect_all_bounded_by_one_deadline_for_all_hung_servers() -> None:
    """Three hung handshakes cost one wait period, not three.

    Pre-fix, ``disconnect_all`` waited ``_DISCONNECT_WAIT_S`` for each
    connection in turn (about 30 s here); now the first phase shares
    one deadline across all of them.
    """
    manager = MCPManager()
    results: list[mcp_servers._Connection] = []

    def do_connect(cfg: MCPServerConfig) -> None:
        results.append(manager.connect(cfg))

    waiters = [
        threading.Thread(target=do_connect, args=(_silent_config(f"silent{i}"),), daemon=True)
        for i in range(3)
    ]
    try:
        for waiter in waiters:
            waiter.start()
        time.sleep(2.0)
        assert all(waiter.is_alive() for waiter in waiters)
        started = time.monotonic()
        manager.disconnect_all()
        elapsed = time.monotonic() - started
        assert elapsed < 2 * mcp_servers._DISCONNECT_WAIT_S, (
            f"disconnect_all took {elapsed:.1f}s for three hung servers: "
            "the waits were not sharing one deadline"
        )
        for waiter in waiters:
            waiter.join(timeout=10)
        assert not any(waiter.is_alive() for waiter in waiters)
        assert len(results) == 3
        assert all(conn.session is None and conn.error for conn in results)
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# M3 — a lingering opener does not hold ``done`` back
# ---------------------------------------------------------------------------


def test_flow_is_done_while_the_opener_still_lingers(server: _AuthMCPServer, home: Path) -> None:
    """The opener approves the sign-in and then lingers: ``done`` must not wait for it.

    Pre-fix the opener ran under ``asyncio.to_thread`` and the flow's
    ``asyncio.run`` joined that executor on exit, so the session only
    counted as finished once the opener returned.
    """
    from kiss.core.browser_handoff import set_browser_tab_opener

    release = threading.Event()

    def approve_then_linger(url: str) -> bool:
        assert "Sign-in received" in _approve(url).text
        assert release.wait(30)
        return True

    set_browser_tab_opener(approve_then_linger)
    try:
        session = MCPLoginSession.start(_cfg(server), wait_seconds=1.0)
        assert session.wait(10), "the finished flow waited for the lingering opener"
        assert session.done, session.error
        assert not release.is_set()
        assert _session_answer(session)["ok"] is True
        release.set()
        deadline = time.monotonic() + 10
        while not session.ready and time.monotonic() < deadline:
            time.sleep(0.05)
        assert session.ready and session.opened_in == "browser_tab"
        assert _session_answer(session)["ok"] is True
    finally:
        release.set()
        set_browser_tab_opener(None)


# ---------------------------------------------------------------------------
# M4 — active() does not stall behind the previous session's wait
# ---------------------------------------------------------------------------


class _HungHTTPServer:
    """Accepts TCP connections and never answers, so a client's request hangs."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._accepted: list[socket.socket] = []
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}/mcp"
        self._thread = threading.Thread(target=self._accept_forever, daemon=True)
        self._thread.start()

    def _accept_forever(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self._accepted.append(conn)

    def close(self) -> None:
        """Close the listener and every hung connection (clients then fail at once)."""
        self._listener.close()
        for conn in self._accepted:
            conn.close()
        self._thread.join(timeout=5)


@pytest.fixture
def hung_server() -> Iterator[_HungHTTPServer]:
    srv = _HungHTTPServer()
    try:
        yield srv
    finally:
        srv.close()


def test_active_is_not_blocked_while_start_waits_for_the_previous_session(
    hung_server: _HungHTTPServer, home: Path
) -> None:
    """``active()`` answers at once while ``start`` waits for a stuck predecessor.

    Pre-fix ``start`` held the class lock across the five-second wait
    for the previous session, so ``active()`` (and every tool answer
    built from it) blocked for as long.
    """
    cfg = MCPServerConfig(name="hung", transport="http", url=hung_server.url)
    first = MCPLoginSession.start(cfg, wait_seconds=0.5)
    assert not first.done and not first.error, "the first session must still be stuck"
    second: list[MCPLoginSession] = []

    def start_second() -> None:
        second.append(MCPLoginSession.start(cfg, wait_seconds=0.5))

    starter = threading.Thread(target=start_second, daemon=True)
    try:
        starter.start()
        time.sleep(0.5)
        assert starter.is_alive(), "start() should still be waiting for the first session"
        began = time.monotonic()
        active = MCPLoginSession.active("hung")
        elapsed = time.monotonic() - began
        assert elapsed < 2.0, f"active() blocked for {elapsed:.1f}s behind start()"
        assert active is first
        starter.join(timeout=30)
        assert not starter.is_alive()
        assert MCPLoginSession.active("hung") is second[0]
    finally:
        hung_server.close()  # the hung requests fail at once; both threads exit
        for session in [first, *second]:
            session.cancel()
            assert session.wait(10)
    assert first.error == "sign-in cancelled"


def test_concurrent_starts_replace_each_other_instead_of_failing_on_the_port(
    hung_server: _HungHTTPServer, home: Path
) -> None:
    """Two ``start`` calls at once both succeed; the later one is the active session.

    With ``_active`` swapped outside the critical section that read it,
    both callers saw the same predecessor and the second to bind the
    redirect port failed with ``Address already in use``.
    """
    cfg = MCPServerConfig(name="hung", transport="http", url=hung_server.url)
    first = MCPLoginSession.start(cfg, wait_seconds=0.5)
    assert not first.done and not first.error, "the first session must still be stuck"
    started: list[MCPLoginSession] = []
    errors: list[BaseException] = []
    go = threading.Event()

    def start_one() -> None:
        go.wait(5)
        try:
            started.append(MCPLoginSession.start(cfg, wait_seconds=0.5))
        except BaseException as exc:  # noqa: BLE001 — the failure is the finding
            errors.append(exc)

    starters = [threading.Thread(target=start_one, daemon=True) for _ in range(2)]
    try:
        for thread in starters:
            thread.start()
        go.set()
        for thread in starters:
            thread.join(timeout=60)
            assert not thread.is_alive()
        assert errors == [], f"a concurrent start() failed: {errors!r}"
        assert len(started) == 2
        assert MCPLoginSession.active("hung") in started
    finally:
        hung_server.close()
        for session in [first, *started]:
            session.cancel()
            assert session.wait(10)
