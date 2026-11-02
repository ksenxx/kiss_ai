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
    through their own ``close()`` (it schedules a task on the loop), and
    neither can their underlying ``asyncio.Server``: its ``close()`` hands
    every socket to the loop's ``_stop_serving``, which the proactor loop
    (Windows) no longer has once closed (``AttributeError``; the selector
    loop happens to tolerate it).  The listening sockets are therefore
    closed directly.  They are still open whenever the stop landed before
    ``_serve_async`` reached its serving ``try`` (the endpoint file is
    published first, the cron store adopted next, and that adoption is
    slow on Windows), so this is the usual case there and a no-op where
    ``_close_ws_listeners`` already ran.  The endpoint file is removed
    while it still carries this server's token, the same ownership check
    ``stop_async`` applies.

    Args:
        server: The server whose ``start()`` was stopped in-process.
    """
    for attr in ("_ws_server", "_ws_loopback_server"):
        ws_server = getattr(server, attr)
        if ws_server is not None:
            # ``_sockets`` is the list ``asyncio.Server.close()`` walks
            # (``None`` once it ran); the public ``sockets`` property
            # wraps them read-only.
            for sock in getattr(ws_server.server, "_sockets", None) or ():
                sock.close()
            setattr(server, attr, None)
    local_endpoint.remove_endpoint_if_owned(
        server._local_endpoint_file, server._local_token,
    )
