# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the run-token guard of ``_stop_task``.

Companion to ``tests/agents/sorcar/test_dispatch_stop_cascade.py``.
The abort cascade of ``daemon_client.run`` sends
``{"type": "stop", "tabId": …, "taskId": <run token>}`` when the
caller's wait is interrupted.  A synthetic ``api-…`` tab is global
state: after the original run finishes, another client can start a NEW
run on the same (now idle) tab.  A late, tab-only cascade stop would
then kill that innocent newer run (gpt-5.6-sol review finding).  The
guard: a ``stop`` carrying a ``taskId`` only applies when the resolved
owner state was created by the ``run`` command that minted the same
token (``AgentState.client_run_token``); UI stops send no ``taskId``
and keep the tab-only behavior.
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    """Point the persistence layer at a scratch DB inside *tmpdir*."""
    from pathlib import Path

    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    """Undo :func:`_redirect_db`."""
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestStopRunTokenGuard:
    """A token-qualified stop must only hit the run that minted it."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)
        self.server = VSCodeServer()

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _register(self, task_id: str, tab_id: str, token: str) -> agent_state.AgentState:
        """Register a running-looking state on *tab_id* with *token*."""
        state = agent_state.AgentState(
            task_id,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.client_run_token = token
        agent_state.register(state)
        return state

    def test_matching_token_stops_the_run(self) -> None:
        """The cascade stop reaches its own still-running task."""
        state = self._register("run-a", "api-guard-1", "tok-a")
        try:
            self.server._stop_task("api-guard-1", run_token="tok-a")
            assert state.stop_event is not None
            assert state.stop_event.is_set()
        finally:
            agent_state.unregister(state.task_id, state)

    def test_stale_token_never_stops_a_reused_tab(self) -> None:
        """The incident from the review: tab reused by a newer run.

        Run A finished and its state was replaced by run B on the same
        synthetic tab.  A's late cascade stop (token ``tok-a``) must
        NOT set B's stop event — and must not leak onto B through the
        viewer-subscription fallback either, so B is subscribed to the
        tab in the printer first.
        """
        state_b = self._register("run-b", "api-guard-2", "tok-b")
        self.server.printer.subscribe_tab("run-b", "api-guard-2")
        try:
            self.server._cmd_stop(
                {"tabId": "api-guard-2", "taskId": "tok-a"},
            )
            assert state_b.stop_event is not None
            assert not state_b.stop_event.is_set(), (
                "a stale run token stopped a newer run that reused "
                "the synthetic api tab"
            )
        finally:
            agent_state.unregister(state_b.task_id, state_b)

    def test_ui_stop_without_token_keeps_tab_only_behavior(self) -> None:
        """A token-less stop (UI button) stops whatever owns the tab."""
        state = self._register("run-c", "api-guard-3", "tok-c")
        try:
            self.server._cmd_stop({"tabId": "api-guard-3"})
            assert state.stop_event is not None
            assert state.stop_event.is_set()
        finally:
            agent_state.unregister(state.task_id, state)
