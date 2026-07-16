# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the chat-viewer live-stream invariant.

Invariant
---------
When a task is running in a tab, then ANY tab — in any remote browser
window or VS Code window — that has opened the chat id of that task
must see the events streaming from the running task.

Tabs that open the chat WHILE the task is already running are covered
by ``_replay_session`` → ``_reattach_running_chat`` (tested in
``test_detach_tab_and_reattach.py``).  These tests cover the tabs that
opened the chat BEFORE the task started:

1. A viewer tab that resumed the chat from history while it was idle
   must receive the live stream of a follow-up task launched from a
   different tab (``clear`` + ``status running`` + every task event).
2. The same must hold for a viewer tab that has NO
   ``_RunningAgentState`` registry entry (e.g. a tab restored by
   ``ready``/``resumeSession`` after a daemon restart, where
   ``_replay_session`` deliberately does not create registry state).
3. A tab displaying a SUB-AGENT row of the chat must NOT be subscribed
   to the parent chat's follow-up stream (it shows a different task's
   stream entirely).
4. A tab that navigated away (``newChat``) or closed (``closeTab``)
   must STOP receiving streams for chats it no longer displays.

The agent stack is real (``WorktreeSorcarAgent.run`` →
``ChatSorcarAgent.run`` → task-id allocation → printer subscription);
only the innermost LLM-driven ``run`` (the grandparent of
``SorcarAgent``) is replaced so no model call happens.  The captured
``broadcast`` mirrors :meth:`WebPrinter.broadcast` fan-out exactly:
explicit-``tabId`` events pass verbatim and task events are duplicated
once per subscribed tab, each copy stamped with the viewer's tab id.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server.server import VSCodeServer

_LIVE_TEXT = "live-follow-up-delta"


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


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    """Create a ``VSCodeServer`` whose broadcasts mirror ``WebPrinter``.

    Events with an explicit ``tabId`` are captured verbatim; events
    with a (thread-local) task id are recorded, persisted, and fanned
    out once per subscribed tab with the viewer's ``tabId`` stamped —
    exactly the dispatch that decides which frontend tabs (across all
    connected windows) see a task's stream.
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


def _patch_grandparent_run() -> Any:
    """Replace the LLM-driven grandparent ``run`` with a stub.

    ``WorktreeSorcarAgent.run`` and ``ChatSorcarAgent.run`` (which owns
    task-id allocation and printer subscription — the code under test)
    stay REAL; only the innermost agent loop is stubbed.  The stub
    broadcasts one ``text_delta`` so the tests can observe exactly
    which tabs the live stream fans out to.
    """
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: Any, **kwargs: Any) -> str:
        printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
        if printer is not None:
            printer.broadcast({"type": "text_delta", "text": _LIVE_TEXT})
        return str(yaml.dump({"success": True, "summary": "done"}))

    parent.run = _run_proxy
    return original


def _unpatch_grandparent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


class TestChatViewerLiveStream(unittest.TestCase):
    """Tabs that opened a chat must receive later tasks' live streams."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.original_run = _patch_grandparent_run()

    def tearDown(self) -> None:
        _unpatch_grandparent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_and_wait(self, tab_id: str, prompt: str) -> None:
        self.server._handle_command({
            "type": "run", "prompt": prompt, "model": self.model,
            "workDir": self.tmpdir, "tabId": tab_id, "autoCommit": True,
        })
        t = self.server._get_tab(tab_id).task_thread
        assert t is not None
        t.join(timeout=60)
        assert not t.is_alive()

    def _events_since(self, idx: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events[idx:])

    def _live_delta_tab_ids(self, since: int) -> set[str]:
        return {
            str(e.get("tabId") or "")
            for e in self._events_since(since)
            if e.get("type") == "text_delta" and e.get("text") == _LIVE_TEXT
        }

    def _open_chat_in_tab(self, tab_id: str, chat_id: str,
                          with_new_chat: bool = True) -> None:
        """Open *chat_id* in *tab_id* the way a frontend window does."""
        if with_new_chat:
            self.server._handle_command({"type": "newChat", "tabId": tab_id})
        self.server._handle_command({
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_id,
        })

    def test_viewer_tab_streams_followup_task_from_other_tab(self) -> None:
        """Core invariant: a viewer that opened the chat while idle sees
        the live stream of a follow-up task launched in another tab."""
        tab_a, tab_b = "tab-A", "tab-B"
        self._run_and_wait(tab_a, "first task")
        chat_id = self.server._get_tab(tab_a).chat_id
        assert chat_id

        # Tab B (simulating a tab in ANOTHER window) opens the chat
        # from history while no task is running.
        self._open_chat_in_tab(tab_b, chat_id)
        replays = [e for e in self._events_since(0)
                   if e.get("type") == "task_events"
                   and e.get("tabId") == tab_b]
        assert replays, "viewer tab should have received the replay"

        mark = len(self.events)
        self._run_and_wait(tab_a, "follow-up task")

        post = self._events_since(mark)
        # 1. The live task event reached BOTH the launcher and the viewer.
        delta_tabs = self._live_delta_tab_ids(mark)
        assert tab_a in delta_tabs, (
            f"launcher tab lost its own live stream: {post}"
        )
        assert tab_b in delta_tabs, (
            "viewer tab that opened the chat BEFORE the task started "
            f"did not receive the live stream: {post}"
        )
        # 2. The viewer got 'clear' (reset replayed content) and a
        #    running status (spinner / stop button), mirroring the
        #    launcher's start sequence.
        clears_b = [e for e in post if e.get("type") == "clear"
                    and e.get("tabId") == tab_b]
        assert clears_b and clears_b[0].get("chat_id") == chat_id, (
            f"viewer tab missing 'clear' for the new task: {post}"
        )
        running_b = [e for e in post if e.get("type") == "status"
                     and e.get("running") is True and e.get("tabId") == tab_b]
        assert running_b, f"viewer tab missing status running=True: {post}"
        assert int(running_b[0].get("startTs") or 0) > 0
        # 3. The launcher's start sequence is not duplicated: exactly
        #    one running status for tab A for this run.
        running_a = [e for e in post if e.get("type") == "status"
                     and e.get("running") is True and e.get("tabId") == tab_a]
        assert len(running_a) == 1, f"launcher status duplicated: {running_a}"

    def test_viewer_without_registry_entry_still_streams(self) -> None:
        """A restored tab (resumeSession only, no newChat, no registry
        entry — the post-daemon-restart shape) must also be subscribed."""
        tab_a, tab_c = "tab-A", "tab-C"
        self._run_and_wait(tab_a, "first task")
        chat_id = self.server._get_tab(tab_a).chat_id

        # ``ready``-style restore: resumeSession without prior newChat.
        # ``_replay_session`` deliberately does not associate the chat
        # with registry state it did not find (C2/C3); the entry that
        # ``_emit_pending_worktree`` → ``_get_tab`` lazily creates at
        # the tail of the replay carries an EMPTY ``chat_id`` — so the
        # registry alone cannot route this viewer, only the
        # ``_tab_chat_views`` map can.
        self._open_chat_in_tab(tab_c, chat_id, with_new_chat=False)
        state = _RunningAgentState.running_agent_states.get(tab_c)
        assert state is None or state.chat_id == "", (
            "precondition: viewer tab's registry entry must be chat-less"
        )

        mark = len(self.events)
        self._run_and_wait(tab_a, "follow-up task")
        assert tab_c in self._live_delta_tab_ids(mark), (
            "registry-less viewer tab did not receive the live stream"
        )

    def test_subagent_row_viewer_not_subscribed_to_parent_stream(self) -> None:
        """A tab showing a sub-agent row of the chat must NOT receive the
        parent chat's follow-up stream."""
        tab_a, tab_d = "tab-A", "tab-D"
        self._run_and_wait(tab_a, "first task")
        chat_id = self.server._get_tab(tab_a).chat_id
        parent_task_id = self.server._get_tab(tab_a).last_task_id

        # Persist a sub-agent row in the same chat and open it in tab D.
        sub_task_id, _ = th._add_task(
            "sub task", chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_task_id,
                                "parent_tab_id": tab_a}},
        )
        self.server._handle_command({"type": "newChat", "tabId": tab_d})
        self.server._handle_command({
            "type": "resumeSession", "chatId": chat_id,
            "taskId": sub_task_id, "tabId": tab_d,
        })

        mark = len(self.events)
        self._run_and_wait(tab_a, "follow-up task")
        assert tab_d not in self._live_delta_tab_ids(mark), (
            "sub-agent row viewer must not be fed the parent's stream"
        )

    def test_new_chat_and_close_tab_unsubscribe_viewer(self) -> None:
        """Navigating away (newChat) or closing the tab stops the feed."""
        tab_a, tab_b, tab_c = "tab-A", "tab-B", "tab-C"
        self._run_and_wait(tab_a, "first task")
        chat_id = self.server._get_tab(tab_a).chat_id

        self._open_chat_in_tab(tab_b, chat_id)
        self._open_chat_in_tab(tab_c, chat_id)
        # Tab B navigates to a fresh chat; tab C is closed.
        self.server._handle_command({"type": "newChat", "tabId": tab_b})
        self.server._handle_command({"type": "closeTab", "tabId": tab_c})

        mark = len(self.events)
        self._run_and_wait(tab_a, "follow-up task")
        delta_tabs = self._live_delta_tab_ids(mark)
        assert tab_a in delta_tabs
        assert tab_b not in delta_tabs, (
            "tab that navigated to a new chat must not keep streaming "
            "the old chat's tasks"
        )
        assert tab_c not in delta_tabs, (
            "closed tab must not keep streaming the chat's tasks"
        )


if __name__ == "__main__":
    unittest.main()
