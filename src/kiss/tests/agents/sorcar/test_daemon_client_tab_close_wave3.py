# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""When ``daemon_client.run`` closes its tab, and how it reads a lost machine.

* ``run()``'s ``finally`` used to send ``closeTab`` on every exit path,
  including a plain timeout, an unconfirmed stop and a lost connection,
  where the task keeps running: the daemon then removed a running
  task's tab from every surface.  A top-level run now closes its tab
  only once the task is known to be over (it returned, its ``result``
  arrived, its stop was confirmed, or the run was refused); a sub-agent
  run, whose tab no surface shows, still closes it on every exit.
* The daemon marks the terminal status of a run whose machine stream
  was lost with ``lost`` (``RemoteTabs.dropped``) and follows it with a
  ``prompt_refused`` error.  The client read that status as the task's
  end (or, before ``running=true``, the error as a refusal); the task
  may still be running on the machine, so it is now a
  ``ConnectionLostError``.

The daemon is a real local-WSS stand-in (:func:`fake_daemon`) that
answers the ``run`` with a scripted event list, optionally answers a
``stop`` with another, and records every later client command.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import daemon_client
from kiss.tests.local_ws import fake_daemon

RUNNING = {"type": "status", "running": True}
LOST = {"type": "status", "running": False, "lost": "gpu-box"}
LOST_REFUSAL = {
    "type": "error", "code": "prompt_refused", "text": "lost the connection to gpu-box",
}
SPEND = {"type": "usage_info", "cost": "$0.0200", "total_tokens": 7, "step_count": 2}


class _ScriptedDaemon:
    """A daemon stand-in that answers the ``run`` with *script* and records commands.

    Every scripted event gets the run's ``tabId``.  A ``stop`` command
    is answered with *on_stop* (a confirmation, when the test wants
    one).  :attr:`disconnected` is set once the client closed, when
    every command it sent is recorded.
    """

    def __init__(
        self, script: list[dict[str, Any]], on_stop: list[dict[str, Any]] | None = None,
    ) -> None:
        """Start serving; returns once the endpoint file is written."""
        self.script = script
        self.on_stop = on_stop or []
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_tab_close_"))
        self.endpoint_file = self._dir / "sorcar-local.json"
        self.commands: list[dict[str, Any]] = []
        self.disconnected = threading.Event()
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(10), "the daemon stand-in never came up"

    async def _serve(self) -> None:
        """Keep the stand-in up until :meth:`close`."""
        async with fake_daemon(self._dir, self._handle, endpoint_file=self.endpoint_file):
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        """Serve one authenticated client connection."""
        try:
            tab_id = json.loads(await ws.recv())["tabId"]
            for event in self.script:
                await ws.send(json.dumps({**event, "tabId": tab_id}))
            async for message in ws:
                cmd = json.loads(message)
                self.commands.append(cmd)
                if cmd.get("type") == "stop":
                    for event in self.on_stop:
                        await ws.send(json.dumps({**event, "tabId": tab_id}))
        except ConnectionClosed:
            pass
        finally:
            self.disconnected.set()

    def sent(self) -> list[str]:
        """The types of the commands the client sent after its ``run``, once it left."""
        assert self.disconnected.wait(10), "the client never disconnected"
        return [str(c.get("type")) for c in self.commands]

    def close(self) -> None:
        """Stop serving, close every connection and remove the temp dir."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        self._loop.close()
        shutil.rmtree(self._dir, ignore_errors=True)


@pytest.mark.parametrize("parent_task_id", ["", "parent-task-1"])
def test_lost_machine_mid_run_is_a_lost_connection(parent_task_id: str) -> None:
    """A ``lost`` terminal status raises ``ConnectionLostError``, not a result.

    The task may still be running on the machine: the stop is cascaded,
    the spend so far rides on ``task_result``, and a top-level run keeps
    its tab (a sub-agent run's tab is on no surface and is closed).
    """
    daemon = _ScriptedDaemon([RUNNING, SPEND, LOST, LOST_REFUSAL])
    try:
        with pytest.raises(daemon_client.ConnectionLostError) as caught:
            daemon_client.run(
                "remote task", endpoint_file=daemon.endpoint_file, timeout=30.0,
                parent_task_id=parent_task_id,
            )
        assert str(caught.value) == "lost the connection to machine gpu-box"
        spent = caught.value.task_result  # type: ignore[attr-defined]
        assert (spent.cost, spent.tokens, spent.steps) == (0.02, 7, 2)
        assert daemon.sent() == (["stop", "closeTab"] if parent_task_id else ["stop"])
    finally:
        daemon.close()


def test_lost_machine_before_start_is_not_a_refusal() -> None:
    """Before ``running=true`` the ``lost`` status still wins over the refusal that follows."""
    daemon = _ScriptedDaemon([LOST, LOST_REFUSAL])
    try:
        with pytest.raises(daemon_client.ConnectionLostError, match="machine gpu-box"):
            daemon_client.run("remote task", endpoint_file=daemon.endpoint_file, timeout=30.0)
        assert daemon.sent() == ["stop"]
    finally:
        daemon.close()


def test_timeout_after_the_result_closes_the_tab() -> None:
    """A ``result`` proves the task is over even when its terminal status is late."""
    result = {"type": "result", "taskId": "t-1", "success": True, "text": "done"}
    daemon = _ScriptedDaemon([RUNNING, result])
    try:
        with pytest.raises(TimeoutError):
            daemon_client.run("quick task", endpoint_file=daemon.endpoint_file, timeout=0.5)
        assert daemon.sent() == ["closeTab"]
    finally:
        daemon.close()


@pytest.mark.parametrize("confirmed", [True, False])
def test_cancel_closes_the_tab_only_once_confirmed(confirmed: bool) -> None:
    """A confirmed cancel ends the task, so its tab closes; an unconfirmed one keeps it."""
    on_stop = [{"type": "status", "running": False}] if confirmed else []
    daemon = _ScriptedDaemon([RUNNING], on_stop=on_stop)
    cancel = threading.Event()
    cancel.set()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.5)
            with pytest.raises(daemon_client.CancelledError) as caught:
                daemon_client.run(
                    "cancelled task", endpoint_file=daemon.endpoint_file, timeout=None,
                    cancel=cancel,
                )
        assert caught.value.confirmed is confirmed
        assert daemon.sent() == (["stop", "closeTab"] if confirmed else ["stop"])
    finally:
        daemon.close()
