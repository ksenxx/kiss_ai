# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``_replay_session`` must subscribe the new tab to
a running sub-agent's live event stream EVEN WHEN the sub-agent has
not persisted any events yet.

The bug
-------
When ``ChatSorcarAgent._run_tasks_parallel`` spawns a sub-agent, the
sub-agent's ``run`` broadcasts a ``new_tab`` event carrying its own
``task_id``.  The frontend then:

1. Calls ``createNewTab`` to allocate a fresh frontend tab id.
2. Posts ``resumeSession`` back to the backend with
   ``{taskId: <sub_task_id>, tabId: <new_tab_id>}``.

The backend's ``_cmd_resume_session`` forwards to
``_replay_session(chat_id, new_tab_id, task_id=sub_task_id)`` which
loads events for the task and — under the old behaviour — returns
early when the events table is empty.  But sub-agent events are
persisted asynchronously by a background writer thread; at the moment
the round-trip ``resumeSession`` arrives, the events table for the
freshly-allocated sub-agent row is almost always empty.

Returning early meant ``_reattach_running_chat`` was never called and
``printer.subscribe_tab(sub_task_id, new_tab_id)`` was never
invoked — so subsequent live broadcasts from the sub-agent (tagged
with ``taskId=sub_task_id``) had **no fan-out target** for the new
tab.  Result: the sub-agent's events were never streamed to the
freshly-opened tab.  They were only visible after the user reloaded
the page (history replay) or never visible at all when the sub-agent
completed before any flush hit the DB.

This test exercises the production code path with a real
:class:`VSCodeServer` and a real :class:`JsonPrinter`, no
mocks/patches/fakes.  It asserts that the new tab is subscribed to
the sub-agent's ``task_history_id`` regardless of whether any events
have been persisted yet.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class _SubscribeCapturingPrinter(JsonPrinter):
    """Records ``broadcast`` events and ``subscribe_tab`` calls.

    Uses the real ``subscribe_tab`` so the printer's internal
    ``_subscribers`` map is populated end-to-end; we just also keep a
    side log so the test can assert the call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.subscribe_calls: list[tuple[Any, str]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        super().broadcast(event)

    def subscribe_tab(self, task_id: Any, tab_id: str) -> None:
        self.subscribe_calls.append((task_id, tab_id))
        super().subscribe_tab(task_id, tab_id)


class TestReplaySessionRunningSubagentNoEvents:
    """End-to-end: a still-running sub-agent with no persisted events
    must still get its new tab subscribed to its live event stream."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_tab_subscribes_to_live_subagent_with_no_persisted_events(
        self,
    ) -> None:
        """Reproduces the bug: sub-agent broadcasts ``new_tab`` → frontend
        opens a new tab and posts ``resumeSession`` → backend's
        ``_replay_session`` must subscribe the new tab to the live
        sub-agent's ``task_history_id`` even when the events table is
        empty (the sub-agent has just started and hasn't flushed any
        events yet).
        """
        chat_id = "chat-parallel-A"
        parent_id, _ = th._add_task("parent", chat_id=chat_id)
        sub_id, _ = th._add_task("sub task body", chat_id=chat_id)
        # Mark as sub-agent in ``extra`` so ``_replay_session`` sees
        # the ``subagent`` payload and converts the new tab into a
        # sub-agent tab via ``openSubagentTab``.
        th._save_task_extra(
            {
                "model": "m",
                "work_dir": "/tmp",
                "version": "v",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": False,
                "is_worktree": False,
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )
        # IMPORTANT: do NOT append any events for ``sub_id`` — this
        # is the precise condition that triggered the original bug
        # (sub-agent just started, events not flushed yet).

        # NOTE: ``VSCodeServer.__init__`` clears
        # ``_RunningAgentState.running_agent_states`` for test
        # isolation, so the server must be built BEFORE the live
        # running-state entries are inserted.
        printer = _SubscribeCapturingPrinter()
        server = VSCodeServer(printer=printer)

        # Register the parent and the sub-agent as live running
        # tasks (the printer disambiguates by ``task_history_id``).
        parent_state = _RunningAgentState("tab-parent", "test-model")
        parent_state.chat_id = chat_id
        parent_state.task_history_id = parent_id
        parent_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        sub_state = _RunningAgentState("tab-parent__sub_0", "test-model")
        sub_state.chat_id = chat_id
        sub_state.task_history_id = sub_id
        sub_state.is_subagent = True
        sub_state.parent_task_id = parent_id
        sub_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent__sub_0"] = (
            sub_state
        )

        # Simulate the frontend → backend round trip after the
        # ``new_tab`` broadcast: the freshly allocated tab posts
        # ``resumeSession`` with the sub-agent's task_id.
        new_tab_id = "tab-fresh-from-new_tab"
        server._replay_session(
            chat_id=chat_id, tab_id=new_tab_id, task_id=sub_id,
        )

        # 1) The new tab MUST be subscribed to the sub-agent's live
        #    event stream (keyed by ``task_history_id``), not the
        #    parent's.  Before the fix this assertion failed because
        #    ``_replay_session`` returned early on empty events.
        assert (sub_id, new_tab_id) in printer.subscribe_calls, (
            f"subscribe_tab was not called for the live sub-agent; "
            f"got {printer.subscribe_calls!r}"
        )
        assert (parent_id, new_tab_id) not in printer.subscribe_calls, (
            "new tab must not be subscribed to the parent's stream"
        )

        # 2) The printer's internal subscriber map must reflect the
        #    subscription — this is what ``WebPrinter.broadcast``
        #    consults via ``_fanout_targets`` to fan live events out
        #    to the new tab.
        assert new_tab_id in printer._fanout_targets(sub_id)
        assert new_tab_id not in printer._fanout_targets(parent_id)

        # 3) ``openSubagentTab`` must fire so the new tab gets the
        #    purple sub-agent styling.
        opens = [e for e in printer.events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, (
            f"expected one openSubagentTab broadcast, got {opens!r}"
        )
        assert opens[0].get("tab_id") == new_tab_id

        # 4) ``status running=true`` must fire BEFORE ``task_events``
        #    so the webview's ``isRunning`` flag is set before
        #    ``replayTaskEvents`` runs (mirrors the live-task ordering).
        types = [e.get("type") for e in printer.events]
        assert "status" in types
        assert "task_events" in types
        assert types.index("status") < types.index("task_events")

        # 5) Now simulate a live broadcast from the sub-agent's
        #    worker thread (its own ``thread_local.task_id`` is set
        #    to ``sub_id``).  The printer's ``_inject_task_id``
        #    stamps the event with ``taskId=sub_id``, and
        #    ``_fanout_targets(sub_id)`` returns ``[new_tab_id]`` →
        #    the WebPrinter would fan the event out to the new tab.
        printer._thread_local.task_id = str(sub_id)
        targets = printer._fanout_targets(sub_id)
        assert new_tab_id in targets, (
            f"live sub-agent event would have no fan-out target for "
            f"the new tab; got {targets!r}"
        )
