# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``_subscribe_chat_viewers`` must not subscribe a tab whose own run lands mid-call.

``_subscribe_chat_viewers`` (task A's worker, once A's row id exists)
decided "viewer tab V is idle" under ``_state_lock``, released the lock
and only then subscribed V to A's stream and broadcast ``clear`` +
``status running=True`` to it.  If the dispatch thread installed V's
OWN run (``_cmd_run``: state with ``task_thread`` registered under the
lock, ``clear`` broadcast) in between, A's ``clear`` wiped V's fresh
transcript and every later event of A fanned out into V on top of its
own run's events.

The fix re-checks ``busy()`` and subscribes + broadcasts under ONE
lock hold per viewer, exactly as ``_broadcast_status_end_to_viewers``
already does for the end event.

The window is made deterministic through the ``_race_delay()`` test
hook between the viewer scan and the per-viewer work: the test's stand-in
holds A's worker there until V's run is installed, so the interleaving
is fixed whatever the load (the ``KISS_RACE_DELAY`` sleep is capped at
0.1 s, which a busy machine overshoots).  The printer is a real
``JsonPrinter`` subclass that records what it broadcasts.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.server import agent_state, task_runner
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        with self.events_lock:
            self.events.append(dict(event))
        super().broadcast(event)


def _noop() -> None:
    """Target for the never-started placeholder worker thread."""


def _install_own_run(server: AgentServer, tab_id: str, chat_id: str) -> AgentState:
    """Register *tab_id*'s own run exactly as ``_cmd_run`` does before starting it."""
    with server._state_lock:
        state = AgentState(
            f"own-run-{tab_id}",
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
            task_thread=threading.Thread(target=_noop, daemon=True),
        )
        agent_state.register(state)
        server.printer.broadcast({"type": "clear", "chat_id": chat_id, "tabId": tab_id})
    return state


def test_viewer_whose_run_lands_mid_subscribe_is_not_hijacked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id = "chat-subscribe-race"
    launcher, viewer = "launcher-tab", "viewer-tab"
    task_a = "task-a-row"
    printer = _CapturePrinter()
    server = AgentServer(printer=printer)
    with server._state_lock:
        server._tab_chat_views[launcher] = chat_id
        server._tab_chat_views[viewer] = chat_id
    scanned, own_run_installed = threading.Event(), threading.Event()

    def hold_scan_to_subscribe_window() -> None:
        """A's worker, past the viewer scan and off the lock: wait for V's run."""
        scanned.set()
        own_run_installed.wait(timeout=10)

    monkeypatch.setattr(task_runner, "_race_delay", hold_scan_to_subscribe_window)
    try:
        worker = threading.Thread(
            target=server._subscribe_chat_viewers,
            args=(task_a, chat_id),
            kwargs={"source_tab_id": launcher, "start_ms": 123},
            daemon=True,
        )
        worker.start()
        # Land the viewer's own run inside A's worker's race window.
        assert scanned.wait(timeout=10), "A's worker never reached the race window"
        _install_own_run(server, viewer, chat_id)
        own_run_installed.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        own_run_installed.set()
        agent_state.agent_states.clear()

    with printer._lock:
        subscribers = set(printer._subscribers.get(task_a, set()))
    assert viewer not in subscribers, subscribers
    with printer.events_lock:
        viewer_events = [e for e in printer.events if e.get("tabId") == viewer]
    assert viewer_events == [{"type": "clear", "chat_id": chat_id, "tabId": viewer}], viewer_events


def test_idle_viewer_is_still_subscribed() -> None:
    chat_id = "chat-subscribe-idle"
    launcher, viewer = "launcher-tab-2", "idle-viewer-tab"
    task_a = "task-a-row-2"
    printer = _CapturePrinter()
    server = AgentServer(printer=printer)
    with server._state_lock:
        server._tab_chat_views[launcher] = chat_id
        server._tab_chat_views[viewer] = chat_id
    try:
        server._subscribe_chat_viewers(
            task_a,
            chat_id,
            source_tab_id=launcher,
            start_ms=456,
            client_task_id="c1",
        )
    finally:
        agent_state.agent_states.clear()

    with printer._lock:
        subscribers = set(printer._subscribers.get(task_a, set()))
    assert viewer in subscribers and launcher not in subscribers, subscribers
    with printer.events_lock:
        viewer_events = [e for e in printer.events if e.get("tabId") == viewer]
    assert viewer_events == [
        {"type": "clear", "chat_id": chat_id, "tabId": viewer},
        {"type": "status", "running": True, "tabId": viewer, "startTs": 456, "taskId": "c1"},
    ], viewer_events
