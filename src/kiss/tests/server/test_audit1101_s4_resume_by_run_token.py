"""A resume that names a run with no history row yet replays THAT run.

A run allocates its ``task_history`` row after its setup, and its
events are stamped with the row id only from then on.  A client that
loses its connection before that (``daemon_client.run``'s reconnect, a
page's ``ready``) can name the run only by the token it minted for the
``run`` command (``AgentState.client_run_token``) or by the provisional
key the state is registered under.  ``_replay_session`` used to look the
named id up in ``task_history`` only, and on a miss fell back to the
chat's latest row: the client then received ANOTHER task's transcript
(the parent's, or the previous task's for an in-chat cron job) and
absorbed its ``result`` and task id as its own.

Everything here is real: a real ``AgentServer``, a run submitted through
``_cmd_run`` and parked in a real SEA getter before its row exists, a
finished row of the same chat, and the real ``resumeSession`` handler.

Round 2: the worker allocates the row (and re-keys the state to its id)
after setup, possibly between the resume's lookup of the named run and
its attach; the resume carries the state it resolved, not a key that
may be obsolete by then, and still subscribes the tab to the run.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    finish_body,
    serve,
)
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause, Pauses

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("released")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)

CHILD_TAB = "api-child-tab"
CHILD_TOKEN = "tok-child-run"
CHILD_PROMPT = "child parked in setup"


def _wait(predicate: Any, timeout: float) -> bool:
    """Poll *predicate* every 20 ms until it holds or *timeout* passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestResumeNamingAPreRowRun(TestCase):
    """``resumeSession`` with the token or provisional key of a pre-row run."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit1101-s4-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)
        self.chat_id = f"chat-{self.tmp.name}"
        self.parent_id, _ = persistence._add_task("parent done", self.chat_id)
        persistence._append_chat_event(
            {"type": "result", "success": True, "summary": "<p>parent done</p>"},
            self.parent_id,
        )
        persistence._flush_chat_events(self.parent_id)

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _park_child(self, **extra: Any) -> str:
        """Start the child run on ``CHILD_TAB`` and return its provisional state key."""
        self.server._cmd_run({
            "type": "run",
            "prompt": CHILD_PROMPT,
            "tabId": CHILD_TAB,
            "taskId": CHILD_TOKEN,
            "chatId": self.chat_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
            "seaPath": str(self.script),
            **extra,
        })
        self.assertTrue(_wait((self.tmp / "entered").exists, 30.0), "the run never reached setup")
        state = agent_state.find_by_tab(CHILD_TAB)
        assert state is not None
        self.assertEqual(state.client_run_token, CHILD_TOKEN)
        return state.task_id

    def _resume(self, task_id: str) -> dict[str, Any]:
        """Resume ``CHILD_TAB`` naming *task_id*; return the transcript frame it got."""
        first = len(self.printer.emitted)
        self.server._cmd_resume_session({
            "type": "resumeSession",
            "tabId": CHILD_TAB,
            "chatId": self.chat_id,
            "taskId": task_id,
            "scoped": True,
            "connId": "conn-child",
            "replayId": "replay-child",
        })
        frames = [
            ev for ev in list(self.printer.emitted)[first:]
            if ev.get("type") == "task_events" and ev.get("tabId") == CHILD_TAB
        ]
        self.assertEqual(len(frames), 1, frames)
        self.assertEqual(frames[0].get("replayId"), "replay-child")
        return frames[0]

    def _assert_replays_the_child(self, frame: dict[str, Any], key: str) -> None:
        self.assertEqual(frame["task_id"], key)
        self.assertEqual(frame["task"], CHILD_PROMPT)
        self.assertNotIn("result", [ev.get("type") for ev in frame["events"]])

    def test_the_run_token_names_the_parked_run(self) -> None:
        key = self._park_child()
        self._assert_replays_the_child(self._resume(CHILD_TOKEN), key)

    def test_the_provisional_key_names_the_parked_run(self) -> None:
        key = self._park_child()
        self._assert_replays_the_child(self._resume(key), key)

    def test_a_dispatched_child_is_not_mistaken_for_its_parent(self) -> None:
        """A sub-agent's tab (``parentTaskId`` set) attaches to the child, unpublished."""
        key = self._park_child(parentTaskId=self.parent_id)
        first = len(self.printer.emitted)
        self._assert_replays_the_child(self._resume(CHILD_TOKEN), key)
        published = [
            ev for ev in list(self.printer.emitted)[first:]
            if ev.get("type") == "tabs_state"
            and any(t.get("tabId") == CHILD_TAB for t in ev.get("tabs", []))
        ]
        self.assertEqual(published, [], "a sub-agent's tab must not reach the shared registry")

    def test_an_unknown_task_still_loads_the_chats_latest_row(self) -> None:
        """A pinned task whose row is gone keeps resolving to the chat (the ready path)."""
        frame = self._resume("no-such-task")
        self.assertEqual(frame["task_id"], self.parent_id)
        self.assertEqual(frame["task"], "parent done")

    def test_the_row_allocated_between_lookup_and_attach_keeps_the_run(self) -> None:
        """A run re-keyed to its row id during the resume is still the one the tab gets."""
        with serve([finish_body("<p>child finished</p>")]) as (url, _requests):
            settings = {
                "model": MODEL,
                "model_config": {"base_url": url, "api_key": "local"},
                "auto_classify": False,
                "use_memory": False,
                "use_web_tools": False,
            }
            # Released, the SEA lets the run go on: the worker allocates
            # the row, re-keys the state and calls the (scripted) model.
            self.script.write_text(
                _BLOCKING_SCRIPT.replace(
                    'raise RuntimeError("released")', f"return settings | {settings!r}",
                ),
                encoding="utf-8",
            )
            key = self._park_child(parentTaskId=self.parent_id)
            state = agent_state.find_by_tab(CHILD_TAB)
            assert state is not None
            attach = Pause(
                AgentServer._attach_viewer_to_running_chat.__code__,
                lambda loc: loc.get("new_tab_id") == CHILD_TAB,
            )
            model = Pause(ScriptedHandler.do_POST.__code__, lambda loc: True)
            first = len(self.printer.emitted)
            resume = threading.Thread(
                target=self.server._cmd_resume_session,
                args=({
                    "type": "resumeSession",
                    "tabId": CHILD_TAB,
                    "chatId": self.chat_id,
                    "taskId": CHILD_TOKEN,
                    "scoped": True,
                    "connId": "conn-child",
                    "replayId": "replay-rekey",
                },),
            )
            with Pauses(attach, model):
                resume.start()
                self.assertTrue(attach.paused.wait(15), "the resume never reached the attach")
                (self.tmp / "release").write_text("1", encoding="utf-8")
                self.assertTrue(model.paused.wait(30), "the run never reached the model")
                # The row is allocated and the state keyed by it; the run goes on.
                new_key = state.task_id
                self.assertNotEqual(new_key, key)
                self.assertIsNone(agent_state.get(key))
                self.assertTrue(state.running())
                attach.release()
                resume.join(15)
                self.assertFalse(resume.is_alive(), "the resume never finished")
                frames = [
                    ev for ev in list(self.printer.emitted)[first:]
                    if ev.get("type") == "task_events" and ev.get("tabId") == CHILD_TAB
                ]
                self.assertEqual(len(frames), 1, frames)
                self.assertEqual(frames[0].get("replayId"), "replay-rekey")
                self.assertEqual(frames[0]["task_id"], new_key)
                self.assertEqual(frames[0]["task"], CHILD_PROMPT)
                with self.printer._lock:
                    subscribed = self.printer._subscribed_keys(CHILD_TAB)
                self.assertIn(new_key, subscribed)
                model.release()
