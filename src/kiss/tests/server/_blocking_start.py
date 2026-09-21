"""Helpers for tests that drive the blocking ``RemoteAccessServer.start()``.

``start()`` is the daemon's main-thread entry point: it relies on process
exit to free its listening sockets.  A test that runs it on a thread and
then stops the loop leaves the Unix-domain-socket listener open in the
test process.  The kernel keeps accepting connections on that listener
(they queue in the backlog even though no loop ever calls ``accept``), so
the next server to bind the session-wide ``$KISS_HOME/sorcar.sock``
sees a "live" predecessor and waits the full ``uds_owner_wait_s`` (30 s
by default) in ``_wait_for_uds_release`` before giving up on its UDS.
That repeats for every later test in the process until the garbage
collector happens to reclaim the stopped server — 2 minutes per run
under normal load, 20 minutes under heavy load.
"""

from __future__ import annotations

from kiss.server.web_server import RemoteAccessServer


def close_leaked_listeners(server: RemoteAccessServer) -> None:
    """Close the listeners a stopped blocking ``start()`` left open.

    Call after the thread running ``start()`` has been joined.  The event
    loop is closed by then, so the websockets server cannot be closed
    through its own ``close()`` (it schedules a task on the loop); its
    underlying plain ``asyncio.Server`` is closed instead.  Closing the
    UDS listener before ``_unlink_own_uds_socket`` is what that method's
    ownership check requires.

    Args:
        server: The server whose ``start()`` was stopped in-process.
    """
    if server._uds_server is not None:
        server._uds_server.close()
        server._uds_server = None
        server._unlink_own_uds_socket()
    if server._ws_server is not None:
        server._ws_server.server.close()
        server._ws_server = None
