# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the chat live-stream + one-tab-per-chat rules.

Invariants
----------
For a given chat id, AT MOST ONE tab is open (tabs are mirrored from
the daemon's canonical registry, so this holds on every client):
opening a chat in another tab MOVES the chat there — the previously
bound tab is displaced (closed everywhere).  And when a task is
running in a chat, the ONE tab currently showing that chat must see
the events streaming from the running task.

Tabs that open the chat WHILE the task is already running are covered
by ``_replay_session`` → ``_reattach_running_chat`` (tested in
``test_detach_tab_and_reattach.py``).  These tests cover tabs that
opened the chat BEFORE the task started:

1. A tab that resumed the chat from history while it was idle takes
   the chat over (the launcher tab is displaced) and must receive the
   live stream of a follow-up task launched from it (``clear`` +
   ``status running`` + every task event); the displaced tab must
   NOT receive that stream, and a later run from the displaced tab
   starts a NEW chat.
2. The same must hold for a takeover tab that has NO
   ``kiss.server.agent_state`` registry entry (e.g. a tab restored by
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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
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
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tab_state(self, tab_id: str) -> agent_state.AgentState:
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"no agent state registered for tab {tab_id}"
        return state

    def _run_and_wait(self, tab_id: str, prompt: str) -> None:
        self.server._handle_command({
            "type": "run", "prompt": prompt, "model": self.model,
            "workDir": self.tmpdir, "tabId": tab_id, "autoCommit": True,
        })
        t = self._tab_state(tab_id).task_thread
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

    def test_takeover_tab_owns_chat_and_streams_followup(self) -> None:
        """Core invariant: opening a chat in another tab MOVES the chat
        there (one tab per chat).  The takeover tab receives follow-up
        streams; the displaced tab does not, and a later run from it
        starts a NEW chat."""
        tab_a, tab_b = "tab-A", "tab-B"
        self._run_and_wait(tab_a, "first task")
        chat_id = self._tab_state(tab_a).chat_id
        assert chat_id

        self._open_chat_in_tab(tab_b, chat_id)
        replays = [e for e in self._events_since(0)
                   if e.get("type") == "task_events"
                   and e.get("tabId") == tab_b]
        assert replays, "takeover tab should have received the replay"
        # The chat now lives in tab-B only: the registry never binds
        # one chat to two tabs.
        bound = [t["tabId"] for t in self.server.tab_registry.snapshot()
                 if t["chatId"] == chat_id]
        assert bound == [tab_b], (
            f"one chat must be bound to exactly one tab, got {bound}"
        )

        mark = len(self.events)
        self._run_and_wait(tab_b, "follow-up task")

        post = self._events_since(mark)
        delta_tabs = self._live_delta_tab_ids(mark)
        assert tab_b in delta_tabs, (
            f"the tab owning the chat lost the live stream: {post}"
        )
        assert tab_a not in delta_tabs, (
            "the displaced tab must not keep streaming the chat it no "
            f"longer shows: {post}"
        )
        assert self._tab_state(tab_b).chat_id == chat_id, (
            "the follow-up run must continue the moved chat"
        )
        clears_b = [e for e in post if e.get("type") == "clear"
                    and e.get("tabId") == tab_b]
        assert clears_b and clears_b[0].get("chat_id") == chat_id, (
            f"takeover tab missing 'clear' for the new task: {post}"
        )
        running_b = [e for e in post if e.get("type") == "status"
                     and e.get("running") is True and e.get("tabId") == tab_b]
        assert len(running_b) == 1, f"owner status duplicated: {running_b}"
        assert int(running_b[0].get("startTs") or 0) > 0

        # A run from the DISPLACED tab starts a fresh chat — it never
        # rebinds (and never duplicates) the moved chat.
        mark = len(self.events)
        self._run_and_wait(tab_a, "unrelated task")
        new_chat = self._tab_state(tab_a).chat_id
        assert new_chat and new_chat != chat_id, (
            "a displaced tab must start a NEW chat, not re-enter the "
            "moved one"
        )
        chats = [t["chatId"] for t in self.server.tab_registry.snapshot()
                 if t["chatId"]]
        assert len(chats) == len(set(chats)), (
            f"registry bound one chat to two tabs: {chats}"
        )

    def test_takeover_without_registry_entry_still_streams(self) -> None:
        """A restored tab (resumeSession only, no newChat, no registry
        entry — the post-daemon-restart shape) also takes the chat over
        and is subscribed to its follow-up streams."""
        tab_a, tab_c = "tab-A", "tab-C"
        self._run_and_wait(tab_a, "first task")
        chat_id = self._tab_state(tab_a).chat_id

        self._open_chat_in_tab(tab_c, chat_id, with_new_chat=False)
        state = agent_state.find_by_tab(tab_c)
        assert state is None or state.chat_id == "", (
            "precondition: takeover tab's registry entry must be chat-less"
        )
        bound = [t["tabId"] for t in self.server.tab_registry.snapshot()
                 if t["chatId"] == chat_id]
        assert bound == [tab_c], (
            f"one chat must be bound to exactly one tab, got {bound}"
        )

        mark = len(self.events)
        self._run_and_wait(tab_c, "follow-up task")
        delta_tabs = self._live_delta_tab_ids(mark)
        assert tab_c in delta_tabs, (
            "registry-less takeover tab did not receive the live stream"
        )
        assert tab_a not in delta_tabs, (
            "the displaced tab must not receive the moved chat's stream"
        )
        assert self._tab_state(tab_c).chat_id == chat_id, (
            "the follow-up run must continue the moved chat"
        )

    def test_subagent_row_viewer_not_subscribed_to_parent_stream(self) -> None:
        """A tab showing a sub-agent row of the chat must NOT receive the
        parent chat's follow-up stream."""
        tab_a, tab_d = "tab-A", "tab-D"
        self._run_and_wait(tab_a, "first task")
        chat_id = self._tab_state(tab_a).chat_id
        parent_task_id = self._tab_state(tab_a).task_id

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
        chat_id = self._tab_state(tab_a).chat_id

        self._open_chat_in_tab(tab_b, chat_id)
        self._open_chat_in_tab(tab_c, chat_id)
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
