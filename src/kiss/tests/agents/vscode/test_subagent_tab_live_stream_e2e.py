# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a spawned sub-agent's freshly opened tab must receive
the sub-agent's events — the whole transcript, head and tail.

Production sequence under test (reported bug: "when an agent creates a
subtask and opens a tab, the tab does not start showing the events
from the subtask"):

1. The parent agent's ``run_parallel`` tool spawns a sub-agent through
   the real fan-out engine (:func:`run_tasks_parallel`).
2. The sub-agent's REAL ``ChatSorcarAgent.run`` allocates its task row,
   registers its agent state and broadcasts ``new_tab``.
3. The webview reacts by opening a background sub-agent tab and posting
   ``resumeSession {taskId, tabId}`` — simulated here by calling the
   server's real command handler the moment ``new_tab`` is observed.
4. The daemon replays the sub-agent's transcript to the new tab and
   subscribes it to the live stream.
5. Every event the sub-agent broadcast — BEFORE the round trip
   (persisted asynchronously!) and after — must reach the tab: the
   early ones via the ``task_events`` replay, the later ones via the
   printer's per-task fan-out.

The agent stack is real (``ChatSorcarAgent.run`` → task-id allocation →
``new_tab`` broadcast → printer recording/persistence → fan-out); only
the innermost LLM-driven ``run`` (the grandparent of ``SorcarAgent``)
is replaced so no model call happens.  The captured ``broadcast``
mirrors :meth:`WebPrinter.broadcast` fan-out exactly.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.server import VSCodeServer

_HEAD_TEXT = "sub-head-before-resume"
_TAIL_TEXT = "sub-tail-after-resume"
_SUB_TAB_ID = "tab-client-sub-view"


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    """A ``VSCodeServer`` whose broadcasts mirror ``WebPrinter`` fan-out.

    Events with an explicit ``tabId`` are captured verbatim; events
    with a (thread-local) task id are recorded, persisted, and fanned
    out once per subscribed tab with the viewer's ``tabId`` stamped.
    Events with neither are global system events, captured verbatim.
    """
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        if "tabId" in event:
            with lock:
                events.append(event)
            return
        ev = printer._inject_task_id(event)
        if not ev.get("taskId"):
            with lock:
                events.append(ev)
            return
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        for tab_id in printer._fanout_targets(ev.get("taskId")):
            with lock:
                events.append({**ev, "tabId": tab_id})

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class TestSubagentTabLiveStreamE2E(unittest.TestCase):
    """The freshly opened sub-agent tab sees the whole transcript."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        agent_state.agent_states.clear()
        self.server, self.events, self.lock = _make_server()
        self.sub_head_sent = threading.Event()
        self.resume_done = threading.Event()
        parent_cls = cast(Any, SorcarAgent.__mro__[1])
        self.original_run = parent_cls.run
        test = self

        def _run_proxy(self_agent: Any, **kwargs: Any) -> str:
            printer = (
                kwargs.get("printer") or getattr(self_agent, "printer", None)
            )
            if getattr(self_agent, "_subagent_info", None) is not None:
                # The sub-agent's own LLM loop: one event before the
                # webview's resumeSession round-trip lands, one after.
                assert printer is not None
                printer.broadcast(
                    {"type": "text_delta", "text": _HEAD_TEXT},
                )
                printer.broadcast({"type": "text_end", "text": _HEAD_TEXT})
                test.sub_head_sent.set()
                assert test.resume_done.wait(timeout=30), (
                    "test driver never completed the resumeSession"
                )
                printer.broadcast(
                    {"type": "text_delta", "text": _TAIL_TEXT},
                )
                printer.broadcast({"type": "text_end", "text": _TAIL_TEXT})
                return str(yaml.dump({"success": True, "summary": "sub done"}))
            # The parent's LLM loop: fan out one sub-task, exactly what
            # the run_parallel tool does.
            self_agent._run_tasks_parallel(["sub task body"])
            return str(yaml.dump({"success": True, "summary": "parent done"}))

        parent_cls.run = _run_proxy
        self._parent_cls = parent_cls

    def tearDown(self) -> None:
        self._parent_cls.run = self.original_run
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _events_snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)

    def test_new_subagent_tab_receives_head_and_tail(self) -> None:
        """The tab opened for a spawned sub-agent must show the whole
        sub-agent transcript: events broadcast before the webview's
        resumeSession round-trip (via the replay) AND events broadcast
        after it (via the live fan-out)."""
        parent_tab = "tab-parent"

        self.server._handle_command({
            "type": "run",
            "prompt": "parent task",
            "model": self.model,
            "workDir": self.tmpdir,
            "tabId": parent_tab,
            "autoCommit": False,
            "useWorktree": False,
        })

        # The webview's part: new_tab arrives → open a tab → resume it.
        assert self.sub_head_sent.wait(timeout=60), (
            "the sub-agent never started (no head event broadcast)"
        )
        new_tabs = [
            e for e in self._events_snapshot() if e.get("type") == "new_tab"
        ]
        self.assertTrue(new_tabs, "the sub-agent must broadcast new_tab")
        sub_task_id = str(new_tabs[0].get("task_id") or "")
        self.assertTrue(sub_task_id, "new_tab must carry the sub task id")
        self.assertEqual(
            str(new_tabs[0].get("parent_tab_id") or ""),
            parent_tab,
            "new_tab must name the parent's frontend tab",
        )

        self.server._handle_command({
            "type": "resumeSession",
            "taskId": sub_task_id,
            "tabId": _SUB_TAB_ID,
        })
        self.resume_done.set()

        # The run command only LAUNCHES the task thread; join it.
        state = agent_state.find_by_tab(parent_tab)
        assert state is not None, "no agent state for the parent tab"
        task_thread = state.task_thread
        assert task_thread is not None
        task_thread.join(timeout=60)
        self.assertFalse(task_thread.is_alive(), "parent run never finished")

        events = self._events_snapshot()

        # The tab must have been bound as a sub-agent tab.
        opens = [
            e for e in events
            if e.get("type") == "openSubagentTab"
            and e.get("tab_id") == _SUB_TAB_ID
        ]
        self.assertTrue(
            opens,
            f"no openSubagentTab for the resumed tab; got "
            f"{[e.get('type') for e in events]}",
        )

        # Tail: live events after the resume must be fanned out to the tab.
        tail = [
            e for e in events
            if e.get("type") == "text_delta"
            and e.get("text") == _TAIL_TEXT
            and e.get("tabId") == _SUB_TAB_ID
        ]
        self.assertTrue(
            tail,
            "the sub-agent's live events after the resume never reached "
            "the freshly opened tab (subscription missing)",
        )

        # Head: events broadcast BEFORE the resume must reach the tab
        # through the task_events replay (they were persisted
        # asynchronously — the replay must not lose them).
        replays = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") == _SUB_TAB_ID
            and str(e.get("task_id") or "") == sub_task_id
        ]
        self.assertTrue(replays, "no task_events replay reached the new tab")
        replayed_texts = [
            str(ev.get("text") or "")
            for replay in replays
            for ev in cast(list, replay.get("events") or [])
            if ev.get("type") in ("text_delta", "text_end")
        ]
        self.assertTrue(
            any(_HEAD_TEXT in t for t in replayed_texts),
            "the sub-agent's events from before the resumeSession round "
            "trip are missing from the replay — the tab never shows the "
            f"start of the sub-task's transcript; replayed: "
            f"{replayed_texts!r}",
        )


if __name__ == "__main__":
    unittest.main()
