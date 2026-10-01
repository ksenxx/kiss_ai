# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``daemon_client.run`` returns when a task fails before it starts.

``task_runner._run_task`` resolves the chat and work-dir overrides
BEFORE it broadcasts the initial ``status running=true``.  When that
setup raises (a stop injected from the sub-agent tab, a failed state
resolution), the ``except`` broadcasts ``result {success: false}`` and
the ``finally`` broadcasts the terminal ``status running=false`` — and
``running=true`` is never sent.

Before the fix the client's terminal-status branch required
``started`` (set only by ``running=true``), so this terminal status was
ignored: the read loop waited out the whole timeout (forever with
``timeout=None``) for a task that had already finished.  The fix treats
the terminal status as final whenever a ``result`` was already received.

The daemon stand-in below is a real local-WSS server (served by
:func:`kiss.tests.local_ws.fake_daemon`) that answers the ``run``
command with exactly that failure sequence.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import daemon_client
from kiss.tests.local_ws import fake_daemon


class _FailBeforeStartDaemon:
    """A daemon stand-in whose task fails before its ``running=true``.

    Reads the ``run`` command, sends ``result {success: false}`` and
    then ``status running=false`` (never ``running=true``), then keeps
    the connection open and records every further client command.
    """

    def __init__(self) -> None:
        """Start serving; returns once the endpoint file is written."""
        self.endpoint_file = Path(tempfile.mkdtemp(prefix="kiss_prestart_")) / "sorcar-local.json"
        self.commands: list[dict[str, Any]] = []
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(10), "the fake daemon never came up"

    async def _serve(self) -> None:
        """Keep the fake daemon up until :meth:`close`."""
        async with fake_daemon(
            self.endpoint_file.parent, self._handle, endpoint_file=self.endpoint_file,
        ):
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        """Serve one authenticated client connection."""
        try:
            run_cmd: dict[str, Any] = json.loads(await ws.recv())
        except (ConnectionClosed, json.JSONDecodeError):
            return
        tab_id = run_cmd.get("tabId", "")
        try:
            await ws.send(json.dumps({
                "type": "result",
                "tabId": tab_id,
                "taskId": "task-prestart-1",
                "success": False,
                "text": "Failed to resolve the chat state",
                "cost": "$0.0000",
                "total_tokens": 0,
                "step_count": 0,
            }))
            await ws.send(json.dumps({"type": "status", "running": False, "tabId": tab_id}))
            async for message in ws:
                self.commands.append(json.loads(message))
        except ConnectionClosed:
            pass

    def close(self) -> None:
        """Stop serving and close every connection."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        self._loop.close()


def test_run_returns_failure_when_task_dies_before_running_true() -> None:
    """A ``result`` + terminal status without ``running=true`` ends ``run`` promptly.

    ``timeout=None`` makes the pre-fix behaviour an unbounded hang (the
    loop only wakes every 10 s to re-check for an abort), so the test
    runs the client on a helper thread and bounds the wait itself.
    """
    daemon = _FailBeforeStartDaemon()
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            outcome["result"] = daemon_client.run(
                "doomed child task", endpoint_file=daemon.endpoint_file, timeout=None,
            )
        except BaseException as exc:  # noqa: BLE001 — capture for assert
            outcome["exc"] = exc

    worker = threading.Thread(target=call, daemon=True)
    started_at = time.monotonic()
    worker.start()
    try:
        worker.join(timeout=5)
        elapsed = time.monotonic() - started_at
        assert not worker.is_alive(), "run() hung after the task's terminal status"
        assert "exc" not in outcome, f"run() raised: {outcome.get('exc')!r}"
        assert elapsed < 5, f"run() took {elapsed:.1f}s to return"
        result = outcome["result"]
        assert result.success is False
        assert result.text == "Failed to resolve the chat state"
        assert result.task_id == "task-prestart-1"
        # The normal exit path still releases the client's synthetic
        # tab; the stop cascade is for aborted waits only.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            c.get("type") == "closeTab" for c in daemon.commands
        ):
            time.sleep(0.02)
        assert [c["type"] for c in daemon.commands] == ["closeTab"]
    finally:
        daemon.close()
