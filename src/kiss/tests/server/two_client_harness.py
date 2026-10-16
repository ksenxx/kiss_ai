# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Several real local WSS clients of one real daemon.

The ``*_two_clients.py`` suites check that what the daemon tells one
connected chat page it tells every other one: sub-agent tabs, browser
tabs and a child's question are open (and closed) on every surface.
Each :class:`Client` is one real connection to the harness daemon's
token-authenticated local endpoint (``kiss.tests.local_ws``), drained
into its own event list -- exactly one connection per chat page, as in
production.  A client can be disconnected and a new one opened in its
place, which is what a page reload or a dropped WebSocket does.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness


class Client:
    """One chat-page-like local connection and everything it received."""

    def __init__(self, harness: TwoClientHarness, name: str) -> None:
        self.name = name
        self.harness = harness
        self.events: list[dict[str, Any]] = []
        self.tab_id = f"{name}-tab-{uuid.uuid4().hex[:8]}"
        self.closed = False

        async def _open() -> tuple[Any, Any]:
            return await open_local_connection(
                harness.server, limit=16 * 1024 * 1024,
            )

        self.reader, self.writer = asyncio.run_coroutine_threadsafe(
            _open(), harness.loop,
        ).result(timeout=10)

        async def _drain() -> None:
            while True:
                line = await self.reader.readline()
                if not line:
                    return
                try:
                    self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        asyncio.run_coroutine_threadsafe(_drain(), harness.loop)

    def send(self, cmd: dict[str, Any]) -> None:
        """Write one JSON command on this connection."""

        async def _send() -> None:
            self.writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
            await self.writer.drain()

        asyncio.run_coroutine_threadsafe(_send(), self.harness.loop).result(
            timeout=10,
        )

    def ready(self, restored_tabs: list[dict[str, Any]] | None = None) -> None:
        """Announce this page to the daemon, as ``main.js`` does at boot."""
        self.send({
            "type": "ready",
            "tabId": self.tab_id,
            "restoredTabs": restored_tabs or [],
        })

    def of_type(self, event_type: str, **fields: Any) -> list[dict[str, Any]]:
        """Every event of *event_type* whose *fields* all match."""
        return [
            e for e in list(self.events)
            if e.get("type") == event_type
            and all(e.get(k) == v for k, v in fields.items())
        ]

    def close(self) -> None:
        """Drop the connection (a closed page or a lost WebSocket)."""
        if self.closed:
            return
        self.closed = True

        async def _close() -> None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(
                _close(), self.harness.loop,
            ).result(timeout=5)
        except Exception:
            pass


class TwoClientHarness(DaemonLocalHarness):
    """:class:`DaemonLocalHarness` plus named, independently drained clients."""

    def setUp(self) -> None:
        super().setUp()
        self.clients: list[Client] = []

    def tearDown(self) -> None:
        for client in self.clients:
            client.close()
        super().tearDown()

    def open_client(self, name: str) -> Client:
        """Connect a new page *name* and wait for the daemon to register it."""
        before = self._local_client_count()
        client = Client(self, name)
        self.clients.append(client)
        self.wait_for(
            lambda: self._local_client_count() > before,
            what=f"{name}: local connection registered",
        )
        return client

    def _local_client_count(self) -> int:
        with self.server._printer._ws_lock:
            return len(self.server._printer._local_clients)

    @staticmethod
    def wait_for(
        predicate: Callable[[], Any], timeout: float = 30.0, what: str = "",
    ) -> Any:
        """Poll *predicate* until truthy; fail with *what* on timeout."""
        deadline = time.monotonic() + timeout
        last: Any = None
        while time.monotonic() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}; last={last!r}")

    def wait_event(
        self, client: Client, event_type: str, timeout: float = 30.0, **fields: Any,
    ) -> dict[str, Any]:
        """Wait until *client* has an *event_type* event matching *fields*."""
        found: list[dict[str, Any]] = self.wait_for(
            lambda: client.of_type(event_type, **fields),
            timeout=timeout,
            what=f"{client.name}: {event_type} {fields or ''}",
        )
        return found[0]
