# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: an ``/ask`` answer reaches the transcript even when its spend charge fails.

``_dispatch_ask_side_channel``'s worker charges the answering run's
spend to the owner task (``charge_side_channel_usage``) and then
broadcasts the ``ask_answer`` panel.  A raise inside the charge (its
chat-event write, a usage broadcast) escaped the worker thread, so the
user's ``/ask`` line, already echoed, never got its answer.

The charge fails here for real: the owner task has finished, so the
charge writes a ``usage_info`` chat event, and the event writer thread
cannot be started because ``RLIMIT_NPROC`` is lowered to 1 for exactly
that step.  :class:`Pause` hooks (instrumentation, nothing replaced)
lower the limit at the charge's entry and restore it at the answer
broadcast's entry.  The answering daemon is a protocol-level stand-in
on a real ``wss://`` socket: the real ``/ask`` agent would need a live
model, and the code under test is the dispatching worker, not the
answering run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_events_by_task_id,
    _stop_event_writer,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.commands import _CommandsMixin
from kiss.server.server import AgentServer
from kiss.server.task_update import charge_side_channel_usage
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.local_ws import fake_daemon
from kiss.tests.server.pause import Pause, Pauses

ANSWER = "<p>because</p>"


class _AnswerDaemon:
    """Serves one ``/ask`` run over a real ``wss://`` socket: a result costing $0.25."""

    def __init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="kiss_ask_charge_"))
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
        for event in (
            {"type": "status", "running": True},
            {
                "type": "result", "taskId": "ask-child", "success": True,
                "text": ANSWER, "cost": "$0.2500", "total_tokens": 300, "step_count": 2,
            },
            {"type": "status", "running": False},
        ):
            await ws.send(json.dumps({**event, "tabId": tab_id}))
        with contextlib.suppress(Exception):
            await ws.recv()

    def close(self) -> None:
        """Stop serving and remove the endpoint directory."""
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)
        self._loop.close()
        shutil.rmtree(self._dir, ignore_errors=True)


@pytest.mark.skipif(
    not thread_start_can_be_starved(),
    reason="needs a host where RLIMIT_NPROC makes Thread.start fail (non-root Linux)",
)
@pytest.mark.usefixtures("clean_registry")
def test_ask_answer_is_delivered_when_the_charge_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The charge's chat-event write fails; the ``ask_answer`` is still persisted."""
    owner_task, _ = _add_task(
        "owner task", "", {"cost": 1.0, "tokens": 100, "steps": 2,
                           "endTs": int(time.time() * 1000)},
    )
    daemon = _AnswerDaemon()
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    server = AgentServer()
    starved = contextlib.ExitStack()
    charged: list[bool] = []

    def starve_at_charge(_locals: dict[str, Any]) -> bool:
        _stop_event_writer()
        starved.enter_context(nproc_limit_lowered_to_one())
        charged.append(True)
        return False

    def restore_at_answer(_locals: dict[str, Any]) -> bool:
        starved.close()
        return False

    try:
        with Pauses(
            Pause(charge_side_channel_usage.__code__, starve_at_charge),
            Pause(_CommandsMixin._broadcast_ask_answer.__code__, restore_at_answer),
        ):
            server._dispatch_ask_side_channel(
                tab_id="tab-ask", owner_task_id=owner_task,
                owner_agent=SorcarAgent("ask-owner"), chat_id="", question="why?",
            )
            for worker in threading.enumerate():
                if worker.name == "kiss-ask-sidechannel":
                    worker.join(timeout=30)
                    assert not worker.is_alive(), "the /ask worker did not finish"
    finally:
        starved.close()
        daemon.close()

    assert charged, "the charge never ran"
    _flush_chat_events(owner_task)
    session = _load_chat_events_by_task_id(owner_task)
    assert session is not None
    answers = [e for e in session["events"] if e.get("type") == "ask_answer"]  # type: ignore[attr-defined]
    assert [a["text"] for a in answers] == [ANSWER]
