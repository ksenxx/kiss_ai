# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 9 (SORCAR-EXT): shutdown must unblock stuck connect() waiters.

``MCPManager.connect`` blocks on ``conn.ready.wait(CONNECT_TIMEOUT)``
(60 s) and relies on ``_maintain_connection``'s ``finally`` clause to
set ``ready``.  But when ``disconnect_all``/``shutdown`` tears the
manager down while a connection task is still stuck mid-handshake
(e.g. a stdio child that never speaks MCP, so ``session.initialize()``
never returns), setting the ``stop`` event does nothing — the task is
not parked on ``stop.wait()`` — and once ``shutdown`` stops the event
loop the frozen task's ``finally`` can never run.  ``conn.ready`` is
therefore never set and every thread blocked in ``connect()`` burns
the full 60-second CONNECT_TIMEOUT even though the manager already
knows the connection is dead.

The fix makes ``disconnect_all`` cancel a task that did not unwind in
time and always stamp ``conn.error`` + set ``conn.ready`` itself, so
waiters return promptly with a proper error.

Real end-to-end test: a real subprocess child is spawned through the
real MCP stdio transport; no mocks.
"""

from __future__ import annotations

import sys
import threading
import time

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig

# disconnect_all waits up to 10 s for each task to unwind gracefully;
# anything comfortably below CONNECT_TIMEOUT (60 s) proves the waiter
# was unblocked by the teardown rather than by its own timeout.
_PROMPT_DEADLINE = 30.0


def test_shutdown_unblocks_stuck_connect_waiter() -> None:
    """A connect() blocked on a silent server returns promptly on shutdown."""
    config = MCPServerConfig(
        name="silent",
        transport="stdio",
        # A real child process that never speaks the MCP protocol, so
        # the session handshake hangs until the transport dies.
        command=sys.executable,
        args=("-c", "import time; time.sleep(120)"),
    )
    manager = MCPManager()
    elapsed: list[float] = []
    errors: list[str] = []
    sessions: list[object] = []

    def do_connect() -> None:
        started = time.monotonic()
        conn = manager.connect(config)
        elapsed.append(time.monotonic() - started)
        errors.append(conn.error)
        sessions.append(conn.session)

    waiter = threading.Thread(target=do_connect, daemon=True)
    waiter.start()
    # Let the connection task reach the hanging initialize() await.
    time.sleep(2.0)
    assert waiter.is_alive(), "connect() should still be blocked mid-handshake"

    manager.shutdown()
    waiter.join(timeout=_PROMPT_DEADLINE)
    assert not waiter.is_alive(), (
        "connect() stayed blocked after shutdown — the waiter was left to "
        "burn the full CONNECT_TIMEOUT"
    )
    assert sessions == [None]
    assert errors[0], "a torn-down connection must carry an error"
    assert elapsed[0] < _PROMPT_DEADLINE
