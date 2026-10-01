"""Helpers for tests that drive the blocking ``RemoteAccessServer.start()``.

``start()`` is the daemon's main-thread entry point: it relies on process
exit to free its listening sockets.  A test that runs it on a thread and
then stops the loop leaves the WSS listeners open in the test process,
so the next server that binds the same port sees ``EADDRINUSE`` and
walks the whole bind-retry backoff before giving up; and the endpoint
file the stopped server published keeps advertising a daemon that no
loop serves any more.  That repeats for every later test in the process
until the garbage collector happens to reclaim the stopped server.
"""

from __future__ import annotations

from kiss.agents.sorcar import local_endpoint
from kiss.server.web_server import RemoteAccessServer


def close_leaked_listeners(server: RemoteAccessServer) -> None:
    """Close the listeners a stopped blocking ``start()`` left open.

    Call after the thread running ``start()`` has been joined.  The event
    loop is closed by then, so the websockets servers cannot be closed
    through their own ``close()`` (it schedules a task on the loop); their
    underlying plain ``asyncio.Server`` is closed instead.  The endpoint
    file is removed while it still carries this server's token, the same
    ownership check ``stop_async`` applies.

    Args:
        server: The server whose ``start()`` was stopped in-process.
    """
    for attr in ("_ws_server", "_ws_loopback_server"):
        ws_server = getattr(server, attr)
        if ws_server is not None:
            ws_server.server.close()
            setattr(server, attr, None)
    local_endpoint.remove_endpoint_if_owned(
        server._local_endpoint_file, server._local_token,
    )
