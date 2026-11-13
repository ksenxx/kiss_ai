# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon's SEA slash-command registry subscription lifecycle.

End-to-end over a real :class:`RemoteAccessServer` and a real local
WebSocket client:

* while the server runs, a registry change fans a ``seaCommands``
  event out to every connected client (the subscription installed by
  ``_start_sea_command_watcher``);
* the subscription is installed exactly once even when the start hook
  runs again (a rebound listener);
* ``stop_async`` removes the subscription and stops the poller, so a
  server torn down in-process leaves no subscriber bound to its dead
  printer behind.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer, WebPrinter
from kiss.tests.agents.sorcar.conftest import reset_sea_registry
from kiss.tests.local_ws import LocalReader, open_local_connection
from kiss.tests.server.pause import Pause


def _watcher_threads() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate() if t.name == "kiss-sea-registry-watcher" and t.is_alive()
    ]


def _on_broadcast_thread(_locals: dict[str, Any]) -> bool:
    """Pause predicate: the entering thread is the detached SEA-command broadcaster."""
    return threading.current_thread().name == "kiss-sea-commands-broadcast"


class SeaWatcherLifecycleTest(IsolatedAsyncioTestCase):
    """The server subscribes once, broadcasts changes, and unsubscribes on stop."""

    async def asyncSetUp(self) -> None:
        reset_sea_registry()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        # A private KISS home: the test writes its own ``SEAS.md`` there
        # and never touches the suite-wide one.
        self.saved_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(kiss_dir)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        reset_sea_registry()
        if self.saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self.saved_home
        th._DB_PATH, th._KISS_DIR = self.saved
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _recv_until(
        self,
        reader: LocalReader,
        event_type: str,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(f"no {event_type!r} event within {timeout}s")
            line = await asyncio.wait_for(reader.readline(), remaining)
            if not line:
                raise AssertionError("connection closed before the event arrived")
            event: dict[str, Any] = json.loads(line)
            if event.get("type") == event_type:
                return event

    async def test_registry_change_reaches_clients_and_stop_unsubscribes(
        self,
    ) -> None:
        """One subscription while running; broadcast on change; none after stop."""
        with sea_commands._lock:
            subscribers = list(sea_commands._subscribers)
        self.assertEqual(len(subscribers), 1)
        self.assertTrue(_watcher_threads(), "registry poller not started")

        # A second start hook (a rebound listener) must not stack a
        # duplicate subscriber.
        self.server._start_sea_command_watcher()
        with sea_commands._lock:
            self.assertEqual(len(sea_commands._subscribers), 1)

        reader, writer = await open_local_connection(self.server)
        try:
            folder = Path(self.tmpdir) / "seas"
            sea = folder / "auditfcmd" / "auditfcmd_sea.py"
            sea.parent.mkdir(parents=True)
            sea.write_text("# stub SEA\n", encoding="utf-8")
            sea_commands.seas_md_path().write_text(f"{folder}\n", encoding="utf-8")
            # The poller would notice within its interval; refresh now
            # so the test exercises the subscriber -> broadcast path
            # without waiting on it.
            await asyncio.to_thread(sea_commands.refresh_registry)
            # The startup rescan's broadcast, carrying the list from
            # before the write, may reach this client first.
            event = await self._recv_until(reader, "seaCommands")
            while "auditfcmd" not in event["commands"]:
                event = await self._recv_until(reader, "seaCommands")
        finally:
            writer.close()
            await writer.wait_closed()

        await self.server.stop_async()
        with sea_commands._lock:
            self.assertEqual(sea_commands._subscribers, [])
        self.assertFalse(self.server._sea_command_subscribed)
        self.assertEqual(_watcher_threads(), [])
        # Stopping twice is harmless (the blocking start() cleanup and
        # stop_async may both run in one process).
        self.server._stop_sea_command_watcher()
        with sea_commands._lock:
            self.assertEqual(sea_commands._subscribers, [])

    async def test_detached_broadcast_never_leaves_clients_on_a_stale_list(
        self,
    ) -> None:
        """A rescan published between the detached broadcaster's last registry
        check and its send must not be taken back by that (older) send: the
        last ``seaCommands`` event a client receives is the current registry.
        """
        folder = Path(self.tmpdir) / "seas"
        for name in ("stalecmdone", "stalecmdtwo"):
            sea = folder / name / f"{name}_sea.py"
            sea.parent.mkdir(parents=True)
        (folder / "stalecmdone" / "stalecmdone_sea.py").write_text("# stub SEA\n", encoding="utf-8")
        sea_commands.seas_md_path().write_text(f"{folder}\n", encoding="utf-8")
        await asyncio.to_thread(sea_commands.refresh_registry)
        reader, writer = await open_local_connection(self.server)
        try:
            with Pause(WebPrinter.broadcast.__code__, _on_broadcast_thread) as pause:
                # A notification on the loop (the startup rescan's path)
                # starts the detached broadcaster; it is held at its send,
                # after its last registry check.
                self.server._on_sea_commands_changed(sea_commands.list_commands())
                self.assertTrue(await asyncio.to_thread(pause.paused.wait, 30))
                (folder / "stalecmdtwo" / "stalecmdtwo_sea.py").write_text(
                    "# stub SEA\n", encoding="utf-8",
                )
                await asyncio.to_thread(sea_commands.refresh_registry)
                newer = await self._recv_until(reader, "seaCommands")
                self.assertIn("stalecmdtwo", newer["commands"])
                pause.release()
            for thread in threading.enumerate():
                if thread.name == "kiss-sea-commands-broadcast":
                    await asyncio.to_thread(thread.join, 30)
            # Everything the broadcaster sent precedes this probe's reply.
            writer.write(json.dumps({"type": "activeTasksQuery"}).encode() + b"\n")
            await writer.drain()
            last = newer
            while True:
                line = await asyncio.wait_for(reader.readline(), 10)
                event = json.loads(line)
                if event.get("type") == "activeTasksResponse":
                    break
                if event.get("type") == "seaCommands":
                    last = event
            self.assertIn(
                "stalecmdtwo", last["commands"],
                "regression: the detached broadcaster's older list was delivered last",
            )
        finally:
            writer.close()
            await writer.wait_closed()
