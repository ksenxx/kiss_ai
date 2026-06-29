# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: restoring a parent agent that fanned out
sub-agents (``run_parallel``) after a VS Code restart.

After a VS Code restart the webview restores its tabs from persisted
state and sends ``resumeSession {chatId, tabId}`` — chat id only, no
task id — for each restored tab (see ``init()`` in media/main.js and
the ``ready`` handler in SorcarSidebarView.ts).  For a parent agent
that spawned sub-agents the restored parent tab must:

1. load the PARENT's own chat events into the parent tab (NOT the
   events of the most recently persisted sub-agent row, which shares
   the parent's chat_id and was inserted later),
2. NOT be converted into a sub-agent tab, and
3. reopen every persisted sub-agent row in its own sub-agent tab
   (``openSubagentTab`` + ``task_events``) anchored to the parent tab
   so the restored layout mirrors the live execution layout.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _seed_parent_with_subagents(
    chat_id: str,
) -> tuple[str, list[str]]:
    """Persist a finished parent task plus two finished sub-agent rows.

    Mirrors exactly what ``ChatSorcarAgent`` writes during a
    ``run_parallel`` fan-out: the parent ``task_history`` row is
    created first, then one row per sub-agent (sharing the parent's
    ``chat_id``) whose ``extra.subagent.parent_task_id`` points back
    at the parent row.

    Returns:
        Tuple of (parent task id, list of sub-agent task ids).
    """
    parent_id, _ = th._add_task("parent task with fanout", chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": "parent-event"}, task_id=parent_id,
    )
    th._save_task_extra(
        {
            "model": "test-model",
            "work_dir": "/tmp",
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": True,
            "is_worktree": False,
        },
        task_id=parent_id,
    )
    sub_ids: list[str] = []
    for idx in range(2):
        sub_id, _ = th._add_task(f"sub task {idx}", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": f"sub-event-{idx}"},
            task_id=sub_id,
        )
        th._save_task_extra(
            {
                "model": "test-model",
                "work_dir": "/tmp",
                "version": "test",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": True,
                "is_worktree": False,
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )
        sub_ids.append(sub_id)
    return parent_id, sub_ids


class TestRestoreParentTabWithSubagents:
    """Simulates the post-restart ``resumeSession`` (chat id only, no
    task id) that the webview sends for a restored parent tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _resume_restored_parent_tab(
        self,
    ) -> tuple[VSCodeServer, list[dict], str, list[str], str]:
        chat_id = "chat-restart-1"
        parent_id, sub_ids = _seed_parent_with_subagents(chat_id)
        server, events = _make_server()
        parent_tab_id = "tab-restored-parent"
        # This is exactly what ``_cmd_resume_session`` does for the
        # webview's post-restart ``resumeSession {chatId, tabId}``.
        server._cmd_resume_session({
            "type": "resumeSession",
            "chatId": chat_id,
            "tabId": parent_tab_id,
        })
        return server, events, parent_id, sub_ids, parent_tab_id

    def test_parent_tab_loads_its_own_events(self) -> None:
        """The restored parent tab must replay the PARENT's events,
        not the events of the latest sub-agent row in the chat."""
        _, events, parent_id, _, parent_tab_id = (
            self._resume_restored_parent_tab()
        )
        parent_replays = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        ]
        assert len(parent_replays) == 1, f"events={events}"
        replay = parent_replays[0]
        assert replay["task_id"] == parent_id
        assert replay["task"] == "parent task with fanout"
        assert any(
            ev.get("type") == "text_delta" and ev.get("text") == "parent-event"
            for ev in replay["events"]
        ), f"parent tab replayed wrong events: {replay['events']}"

    def test_parent_tab_is_not_converted_into_subagent_tab(self) -> None:
        """No ``openSubagentTab`` may target the restored parent tab id."""
        _, events, _, _, parent_tab_id = self._resume_restored_parent_tab()
        assert not any(
            e.get("type") == "openSubagentTab"
            and e.get("tab_id") == parent_tab_id
            for e in events
        ), f"parent tab was converted into a sub-agent tab: {events}"

    def test_subagent_tabs_reopen_right_of_parent_with_own_events(
        self,
    ) -> None:
        """Each persisted sub-agent row reopens in its own sub-agent
        tab anchored to the parent tab, replaying its own events."""
        _, events, _, sub_ids, parent_tab_id = (
            self._resume_restored_parent_tab()
        )
        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == len(sub_ids), f"events={events}"
        for idx, (op, sub_id) in enumerate(zip(opens, sub_ids)):
            assert op["parent_tab_id"] == parent_tab_id
            assert op["isSubagentTab"] is True
            assert op["isDone"] is True
            assert op["description"] == f"sub task {idx}"
            sub_tab_id = op["tab_id"]
            assert sub_tab_id != parent_tab_id
            sub_replays = [
                e for e in events
                if e.get("type") == "task_events"
                and e.get("tabId") == sub_tab_id
            ]
            assert len(sub_replays) == 1
            assert sub_replays[0]["task_id"] == sub_id
            assert any(
                ev.get("type") == "text_delta"
                and ev.get("text") == f"sub-event-{idx}"
                for ev in sub_replays[0]["events"]
            )

    def test_parent_events_replay_before_subagent_tabs_open(self) -> None:
        """The parent tab's own events must be replayed first; the
        sub-agent tabs open after (to the right of) the parent."""
        _, events, _, _, parent_tab_id = self._resume_restored_parent_tab()
        parent_idx = next(
            i for i, e in enumerate(events)
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        )
        open_idxs = [
            i for i, e in enumerate(events)
            if e.get("type") == "openSubagentTab"
        ]
        assert open_idxs, f"no sub-agent tabs opened: {events}"
        assert all(parent_idx < i for i in open_idxs)


class TestLatestChatEventsSkipSubagentRows:
    """``_load_latest_chat_events_by_chat_id`` must return the latest
    NON-sub-agent row: chat-id-only resumes always target the parent
    session, while sub-agent rows are loaded explicitly by task id."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_latest_skips_trailing_subagent_rows(self) -> None:
        chat_id = "chat-restart-2"
        parent_id, _ = _seed_parent_with_subagents(chat_id)
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        assert result["task_id"] == parent_id
        assert result["task"] == "parent task with fanout"

    def test_latest_returns_newer_followup_parent_row(self) -> None:
        """A follow-up (non-sub-agent) task persisted after the fan-out
        is the new session tail and must win."""
        chat_id = "chat-restart-3"
        _seed_parent_with_subagents(chat_id)
        followup_id, _ = th._add_task("follow-up task", chat_id=chat_id)
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        assert result["task_id"] == followup_id

    def test_chat_with_only_subagent_rows_returns_none(self) -> None:
        """Degenerate case: no parent row at all (e.g. parent row was
        deleted) — there is nothing chat-level to resume."""
        chat_id = "chat-restart-4"
        sub_id, _ = th._add_task("orphan sub task", chat_id=chat_id)
        th._save_task_extra(
            {"subagent": {
                "parent_task_id":
                    "ffffffffffffffffffffffffffffffff"
            }}, task_id=sub_id,
        )
        assert th._load_latest_chat_events_by_chat_id(chat_id) is None
