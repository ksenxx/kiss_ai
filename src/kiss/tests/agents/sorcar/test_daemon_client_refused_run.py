# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A refused ``run`` must end ``daemon_client.run`` at once, not after its timeout.

The daemon turns a ``run`` down (``commands._refuse_run``, e.g. while a
self-update is being installed) with exactly two frames on the client's
tab: ``status running=false`` and ``error code="prompt_refused"``.
Before the fix the receive loop ignored both — the status because
nothing had started and no result existed, the error because the loop
never looked at ``error`` frames — so the client waited out its whole
``timeout`` and, with ``timeout=None`` (the ``run_agent`` job thread,
cron jobs), hung until the daemon closed the socket.  The fix raises
:class:`daemon_client.RunRefusedError` with the daemon's reason.

End-to-end against a real websocket server (``kiss.tests.local_ws.fake_daemon``)
that answers the ``run`` with the refusal and keeps the socket open, as
the daemon does when the install failed and the barrier was lowered.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import daemon_client
from kiss.tests.local_ws import fake_daemon

_REASON = "An update is being installed; wait for the daemon to restart."


class _RefusingDaemon:
    """A daemon stand-in that refuses every ``run`` and stays up.

    Answers the ``run`` with ``status running=false`` and
    ``error code=prompt_refused`` (what ``commands._refuse_run``
    broadcasts), then keeps the connection open and records every
    further client command.
    """

    def __init__(self) -> None:
        """Start serving; returns once the endpoint file is written."""
        self.endpoint_file = Path(tempfile.mkdtemp(prefix="kiss_refused_")) / "sorcar-local.json"
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
        assert run_cmd.get("type") == "run"
        tab_id = run_cmd.get("tabId", "")
        try:
            await ws.send(json.dumps({"type": "status", "running": False, "tabId": tab_id}))
            await ws.send(json.dumps({
                "type": "error", "code": "prompt_refused", "text": _REASON, "tabId": tab_id,
            }))
            async for message in ws:
                self.commands.append(json.loads(message))
        except ConnectionClosed:
            pass

    def close(self) -> None:
        """Stop serving and close every connection."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        self._loop.close()


@pytest.mark.parametrize("timeout", [None, 30.0])
def test_refused_run_raises_at_once(timeout: float | None) -> None:
    """``run`` raises ``RunRefusedError`` with the daemon's reason within a second.

    With ``timeout=None`` the pre-fix behaviour is an unbounded hang
    and with ``timeout=30`` a 30 s wait ending in ``TimeoutError``, so
    the client runs on a helper thread and the test bounds the wait.
    """
    daemon = _RefusingDaemon()
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            outcome["result"] = daemon_client.run(
                "refused task", endpoint_file=daemon.endpoint_file, timeout=timeout,
            )
        except BaseException as exc:  # noqa: BLE001 — capture for assert
            outcome["exc"] = exc

    worker = threading.Thread(target=call, daemon=True)
    started_at = time.monotonic()
    worker.start()
    try:
        worker.join(timeout=5)
        elapsed = time.monotonic() - started_at
        assert not worker.is_alive(), "run() did not notice the refusal"
        assert "result" not in outcome, f"run() returned {outcome.get('result')!r}"
        exc = outcome["exc"]
        assert isinstance(exc, daemon_client.RunRefusedError), repr(exc)
        assert not isinstance(exc, (TimeoutError, ConnectionError))
        assert _REASON in str(exc)
        assert elapsed < 5, f"run() took {elapsed:.1f}s to notice the refusal"
        # No task was started, so nothing is stopped: only the client's
        # synthetic tab is released.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            c.get("type") == "closeTab" for c in daemon.commands
        ):
            time.sleep(0.02)
        assert [c["type"] for c in daemon.commands] == ["closeTab"]
    finally:
        daemon.close()
