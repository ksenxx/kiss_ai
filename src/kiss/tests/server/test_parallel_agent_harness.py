# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The stand-in model server of the parallel-agent harness stops whatever its accept finds.

Linux full run 2026-11-22 (load average 85): ``ClosedTabKeepsApiClient`` hung in its
cleanup until the 300 s timeout of its module killed the whole split.  The serving loop
had been told by ``select`` that a connection waited and then blocked in ``accept``: the
connection was gone by then (accept(2): "the connection might have been removed by an
asynchronous network error or another thread before accept() is called").  ``shutdown``
waits for the serving loop, so :meth:`StandInModelServer.stop` never returned.
"""

from __future__ import annotations

import functools
import socket
import socketserver
import threading
import time
from typing import Any

from kiss.tests.server.parallel_agent_harness import StandInModelServer
from kiss.tests.server.pause import Pause


def _take_it_first(taken: list[socket.socket], entered: dict[str, Any]) -> bool:
    """At the entry of ``get_request``: take the connection ``select`` announced.

    Run on the serving thread right before its own ``accept``, which then finds no
    connection, as after the removal accept(2) describes.  Never pauses the thread.
    """
    if not taken:
        taken.append(entered["self"].socket.accept()[0])
    return False


def test_stop_returns_when_the_announced_connection_is_gone_at_accept() -> None:
    """``stop`` ends a serving loop whose ``accept`` found the announced connection gone."""
    server = StandInModelServer(dict)
    taken: list[socket.socket] = []
    stopper = threading.Thread(target=server.stop, daemon=True)
    get_request = socketserver.TCPServer.get_request.__code__
    with Pause(get_request, functools.partial(_take_it_first, taken)):
        with socket.create_connection(("127.0.0.1", server._server.server_port)):
            deadline = time.monotonic() + 10
            while not taken and time.monotonic() < deadline:
                time.sleep(0.01)
            assert taken, "the serving loop never reached its accept"
            taken[0].close()
            stopper.start()
            stopper.join(10)
    assert not stopper.is_alive(), "stop() hangs on a serving loop blocked in accept"
