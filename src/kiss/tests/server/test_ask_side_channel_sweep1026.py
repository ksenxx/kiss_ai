# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an unconfirmed ``stop_on_timeout`` stop still charges the spend seen.

``daemon_client.run(..., stop_on_timeout=True)`` sends ``stop`` at the
deadline and waits :data:`daemon_client._STOP_CONFIRM_GRACE_SECONDS`
for the task's terminal ``status running=false``.  When the grace
expires it raises :class:`daemon_client.StopUnconfirmedTimeoutError`;
every sibling abort (the confirmed stop, the cancel, the generic
abort) hands the caller the spend the task reported before the abort,
but the unconfirmed-stop error used to carry nothing, so the ``/ask``
side channel (``commands._dispatch_ask_side_channel``) charged the
owner task nothing for an answerer whose ``result`` had already
reported its cost.

The tests below run against a real local-``wss://`` daemon stand-in
(:func:`kiss.tests.local_ws.fake_daemon`) that answers the ``stop``
with the failure ``result`` a real daemon's cancel path broadcasts and
then never sends the terminal status.

The ``/ask`` dispatch's other fix of this sweep — ``owner.agent`` is
read once under ``_state_lock`` in ``_cmd_append_user_message`` and
that value is handed to the side channel, instead of a second,
unlocked read after the lock is released that task teardown (which
clears ``state.agent`` under the same lock) could turn into ``None`` —
cannot be reproduced end to end: the window lies between the
``with self._state_lock`` block and the dispatch call inside one
method, with no hook a test could use to clear the agent exactly
there.  :func:`test_ask_through_the_command_charges_the_live_owner`
drives the production path and pins that the agent registered on
the state is the one charged.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from kiss.agents.sorcar import cron_agent, daemon_client
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.server import agent_state, commands
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.local_ws import fake_daemon

pytestmark = pytest.mark.usefixtures("clean_registry")


class _UnconfirmedStopDaemon:
    """A daemon stand-in whose task dies on ``stop`` but never confirms it.

    Sends ``status running=true``; on the client's ``stop`` it sends
    the failure ``result`` (``success: false``, carrying the spend)
    and nothing else, so the client's confirmation grace expires.
    """

    def __init__(self) -> None:
        """Serve the stand-in from a thread with its own event loop."""
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_ask_unconfirmed_"))
        self.endpoint_file = self._dir / "sorcar-local.json"
        self._loop = asyncio.new_event_loop()
        self._stop = asyncio.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._main(),), daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("stand-in daemon did not start")

    async def _main(self) -> None:
        async with fake_daemon(self._dir, self._serve, endpoint_file=self.endpoint_file):
            self._ready.set()
            await self._stop.wait()

    async def _serve(self, ws: ServerConnection) -> None:
        tab_id = json.loads(await ws.recv()).get("tabId", "")
        await ws.send(json.dumps({"type": "status", "running": True, "tabId": tab_id}))
        try:
            while json.loads(await ws.recv()).get("type") != "stop":
                pass
            await ws.send(json.dumps({
                "type": "result", "tabId": tab_id, "taskId": "ask-child",
                "success": False, "text": "Task stopped by user",
                "cost": "$0.0100", "total_tokens": 3, "step_count": 1,
            }))
            while True:
                await ws.recv()
        except Exception:
            return

    def close(self) -> None:
        """Shut down the listener and remove the temp endpoint dir."""
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)
        self._loop.close()
        shutil.rmtree(self._dir, ignore_errors=True)


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A running :class:`_UnconfirmedStopDaemon` with short client deadlines."""
    stand_in = _UnconfirmedStopDaemon()
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(stand_in.endpoint_file))
    monkeypatch.setattr(commands, "_ASK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", 0.5)
    yield stand_in
    stand_in.close()


def test_unconfirmed_stop_timeout_carries_the_spend_seen_so_far(daemon: Any) -> None:
    """The error of a never-confirmed stop reports the task's spend like its siblings."""
    with pytest.raises(daemon_client.StopUnconfirmedTimeoutError) as excinfo:
        daemon_client.run(
            "never finishes", endpoint_file=daemon.endpoint_file,
            timeout=0.3, stop_on_timeout=True,
        )
    result = excinfo.value.result
    assert (result.cost, result.tokens, result.steps) == (pytest.approx(0.01), 3, 1)
    assert result.success is False and result.task_id == "ask-child"


def _server() -> tuple[AgentServer, list[dict[str, Any]]]:
    """A real server whose printer's broadcasts land in a list."""
    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = _capture  # type: ignore[assignment]
    return server, events


def _wait_for_answer(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the ``ask_answer`` broadcast once it lands (bounded)."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for event in list(events):
            if event.get("type") == "ask_answer":
                return event
        time.sleep(0.02)
    raise AssertionError("no ask_answer broadcast")


def test_ask_through_the_command_charges_the_live_owner(daemon: Any) -> None:
    """``/ask`` typed on a running tab charges the owner even when the stop is unconfirmed.

    Drives the production path from the typed command: the owner's
    live agent is read under the state lock in
    ``_cmd_append_user_message`` and receives the answerer's reported
    spend (banked on the live agent while the owner row is running).
    """
    owner_task, _ = _add_task("owner of the /ask", "", {"cost": 1.0, "tokens": 10, "steps": 1})
    owner_agent = ChatSorcarAgent("ask-owner-live")
    owner_agent._last_task_id = owner_task
    state = AgentState(
        owner_task, agent=owner_agent, tab_id="tab-ask", chat_id="chat-ask", server_owned=True,
    )
    state.is_task_active = True
    agent_state.register(state)
    server, events = _server()

    server._cmd_append_user_message({"tabId": "tab-ask", "prompt": "/ask why?"})

    answer = _wait_for_answer(events)
    assert answer["success"] is False and answer["taskId"] == owner_task
    assert "did not finish" in answer["text"]
    assert state.pending_user_messages == []
    assert _agent_usage(owner_agent) == (pytest.approx(0.01), 3, 1)
