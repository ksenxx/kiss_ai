# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A local-WSS daemon stand-in that records every ``run`` command it receives.

Tests of the dispatchers (``run_agent``, cron delivery, SEA rollouts)
need to see exactly what ``daemon_client.run`` sends to the daemon.
Instead of replacing ``daemon_client.run`` with a stub, this module
serves the daemon's local endpoint through
:func:`kiss.tests.local_ws.fake_daemon`, so the real client connects,
authenticates and sends its ``run`` command over TLS; the command is
recorded and answered with a scripted ``result`` event.

A test reaches the stand-in either through ``KISS_SORCAR_LOCAL`` (the
endpoint file a standalone client reads) or through
:data:`kiss.agents.sorcar.cron_agent._daemon_endpoint_file` (the endpoint a
channel launcher started by the daemon records), e.g.::

    daemon = RecordingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        tool(task="hi")
        assert daemon.run_commands[0]["workDir"] == ...
    finally:
        daemon.close()
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.tests.local_ws import fake_daemon


class RecordingDaemon:
    """Serve a daemon stand-in; record each client's ``run`` command.

    Every authenticated connection's first frame (the ``run`` command) is
    appended to :attr:`commands`; the client's tab then receives
    ``status running=true``, a ``clear`` event carrying *chat_id*, the
    scripted ``result`` event and ``status running=false``, so
    ``daemon_client.run`` returns a successful ``TaskResult``.  Later
    frames on the same connection (``stop`` and the like) are appended
    to :attr:`commands` as well.

    Attributes:
        endpoint_file: The endpoint file the clients read.
        commands: Every command received, in arrival order.
    """

    def __init__(
        self,
        *,
        text: str = "ok",
        success: bool = True,
        cost: float = 0.0,
        tokens: int = 0,
        steps: int = 0,
        chat_id: str = "",
        task_id: str = "task-recorded-1",
    ) -> None:
        """Start serving in a thread of its own; return once the endpoint file exists."""
        self.result = {
            "type": "result",
            "taskId": task_id,
            "success": success,
            "summary": text,
            "text": text,
            "cost": f"{cost:.6f}",
            "total_tokens": tokens,
            "step_count": steps,
        }
        self.chat_id = chat_id
        self.commands: list[dict[str, Any]] = []
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_recording_daemon_"))
        self.endpoint_file = self.tmp / "sorcar-local.json"
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete,
            args=(self._serve(),),
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(10) or self._startup_error is not None:
            self.close()
            raise RuntimeError("the recording daemon never came up") from self._startup_error

    async def _serve(self) -> None:
        try:
            async with fake_daemon(self.tmp, self._handle, endpoint_file=self.endpoint_file):
                self._ready.set()
                await self._closed.wait()
        except BaseException as exc:  # surfaced to the constructor
            self._startup_error = exc
            self._ready.set()

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            command = json.loads(await ws.recv())
        except (ConnectionClosed, ValueError):
            return
        self.commands.append(command)
        tab_id = command.get("tabId", "")
        stream: list[dict[str, Any]] = [{"type": "status", "running": True}]
        if self.chat_id:
            stream.append({"type": "clear", "chat_id": self.chat_id})
        stream += [self.result, {"type": "status", "running": False}]
        try:
            for event in stream:
                await ws.send(json.dumps({**event, "tabId": tab_id}))
            async for line in ws:
                self.commands.append(json.loads(line))
        except ConnectionClosed:
            pass

    @property
    def run_commands(self) -> list[dict[str, Any]]:
        """The recorded ``run`` commands only (no ``stop`` or other frames)."""
        return [c for c in self.commands if c.get("type") == "run"]

    def close(self) -> None:
        """Stop serving, close every connection, join the server thread, drop the temp dir.

        Bounded: a peer that never answers the close handshake keeps the
        ``websockets`` server busy for its 10 s close timeout, so the join
        waits longer than that and the loop is closed only once its
        thread has really exited (closing a running loop would raise and
        skip the cleanup below).
        """
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=30)
        if not self._thread.is_alive():
            self._loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
