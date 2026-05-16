"""Regression: clicking a running task in history must STREAM live
events to the freshly opened tab — not just load persisted history.

Two scenarios:

1. **Synchronous ``clear`` emission** (Python): ``_cmd_run`` MUST
   broadcast the initial ``clear`` event (with ``chat_id``) *before*
   it returns.  The extension layer's chat-id → tab-id index is
   populated from that event; if the worker thread emits ``clear``
   *after* ``_cmd_run`` returns, a fast follow-up ``resumeSession``
   command racing the worker thread cannot find the running task
   process and routes the resume to the service process, which has
   no ``_TabState`` for the chat — live streaming is lost.  This
   test pins the synchronous ordering at the Python level.

2. **TS-side fallback to ``_taskProcesses[chatId]``**: even when
   ``_chatIdToTabId`` is missing the entry (e.g. the very first
   command in a brand new extension session), the convention that a
   task's initial chat id equals the tab id it was first submitted
   under lets ``_reattachRunningChat`` resolve via
   ``_taskProcesses.get(chatId)``.  The TS code path is exercised by
   reading the source file and asserting the fallback is present —
   we cannot import TypeScript directly here.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _make_server() -> tuple[VSCodeServer, list[dict], threading.Lock]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_tab_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)
        for viewer in server.printer._fanout_targets(ev.get("tabId")):
            with lock:
                events.append({**ev, "tabId": viewer})

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class TestSynchronousClearEmission:
    """``_cmd_run`` must emit ``clear`` before returning so the chat-id
    is visible to the caller without waiting on the worker thread."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_clear_emitted_before_cmd_run_returns(self) -> None:
        server, events, _lock = _make_server()
        # Use a model that doesn't exist so the inner loop returns
        # fast.  We only assert on the synchronous ``clear`` event,
        # not on actual model execution.
        cmd = {
            "type": "run",
            "tabId": "T1",
            "model": "no-such-model",
            "prompt": "hi",
        }
        server._handle_command(cmd)

        # Find the very first clear event.  It MUST have been
        # broadcast before any worker-thread output (no agent thread
        # output can appear before its caller returns from a
        # synchronous emit).
        clears = [e for e in events if e.get("type") == "clear"]
        assert clears, f"no 'clear' event emitted; got {events}"
        first_clear = clears[0]
        assert first_clear.get("tabId") == "T1"
        # The chat id of the first task in a fresh chat equals the tab id.
        assert first_clear.get("chat_id") == "T1", first_clear

        # Idempotency check: there must NOT be a duplicate ``clear``
        # broadcast (i.e. ``_run_task_inner`` must not also emit one).
        assert len(clears) == 1, (
            "Duplicate clear broadcasts — _run_task_inner should no "
            f"longer emit ``clear``.  Got: {clears}"
        )

        # Drain the worker thread that ``_cmd_run`` started so we
        # don't leave a runaway thread for teardown.
        with server._state_lock:
            tab = server._tab_states.get("T1")
        if tab and tab.task_thread is not None:
            tab.task_thread.join(timeout=10)


class TestResumeRaceWithSyncClear:
    """``resumeSession`` for the chat of a still-running task MUST set
    up the multi-viewer subscription so live events also reach the new
    tab.  With synchronous ``clear``, this is race-free."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_live_events_reach_new_tab_after_history_click(self) -> None:
        # Seed one persisted event so the resume has history to load.
        task_id, chat_id = th._add_task("running task")
        th._append_chat_event(
            {"type": "text_delta", "text": "history-event"},
            task_id=task_id,
        )

        server, events, _lock = _make_server()

        # Manually plant a running task in tab_a (the convention is
        # that chat_id == tab_id for first-task chats).
        tab_a, tab_b = chat_id, "T2"
        tab1 = server._get_tab(tab_a)
        tab1.agent._chat_id = chat_id
        tab1.is_task_active = True
        tab1.stop_event = threading.Event()

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def fake_run() -> None:
            server.printer._thread_local.tab_id = tab_a
            server.printer.broadcast(
                {"type": "text_delta", "text": "pre-release"},
            )
            started.set()
            release.wait(10)
            server.printer.broadcast(
                {"type": "text_delta", "text": "post-release"},
            )
            with server._state_lock:
                tab1.is_task_active = False
                tab1.task_thread = None
            finished.set()

        thread = threading.Thread(target=fake_run, daemon=True)
        tab1.task_thread = thread
        thread.start()
        started.wait(5)

        # The user clicks the running task in history.  Note we do
        # NOT close tab_a first — the original tab stays open, the
        # new tab tab_b must ALSO see the stream (multi-viewer).
        server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_b,
        })

        replay_idx = len(events)

        # task_events broadcast tagged with tab_b.
        replays = [
            e for e in events[:replay_idx] if e.get("type") == "task_events"
        ]
        assert len(replays) == 1
        assert replays[0]["tabId"] == tab_b

        # Running status broadcast for tab_b.
        running = [
            e for e in events[:replay_idx]
            if e.get("type") == "status"
            and e.get("running") is True
            and e.get("tabId") == tab_b
        ]
        assert running, "Missing status:running=True for tab_b"

        # Release the fake task.  Its post-release ``text_delta`` is
        # tagged with tab_a (thread-local), but the printer's fan-out
        # MUST emit an additional copy tagged with tab_b.
        release.set()
        finished.wait(5)
        thread.join(timeout=5)

        post = [
            e for e in events[replay_idx:]
            if e.get("type") == "text_delta"
            and e.get("text") == "post-release"
        ]
        post_tab_ids = sorted({str(e["tabId"]) for e in post})
        assert tab_a in post_tab_ids, post_tab_ids
        assert tab_b in post_tab_ids, (
            f"Multi-viewer broken: tab_b did not receive the fan-out "
            f"copy of the live event.  Got: {post}"
        )


class TestExtensionFallbackForStaleChatIndex:
    """The extension layer's ``_reattachRunningChat`` must fall back to
    ``_taskProcesses.get(chatId)`` when the chat-id → tab-id index is
    missing — the convention that initial chat id equals the original
    tab id is what makes the lookup work.

    This is a source-level guard: the TypeScript code path is
    exercised in the extension build, but we pin the implementation
    here so that future refactors cannot silently regress.
    """

    def test_source_contains_fallback(self) -> None:
        src = Path(
            "src/kiss/agents/vscode/src/SorcarSidebarView.ts",
        ).read_text()
        # The fallback must look up ``_taskProcesses`` by chatId when
        # ``_chatIdToTabId.get(chatId)`` returned undefined.
        assert "this._taskProcesses.get(chatId)" in src, (
            "Missing _taskProcesses.get(chatId) fallback in "
            "_reattachRunningChat — a fast history click during the "
            "very first task of an extension session would fail to "
            "route resumeSession to the live task proc and live "
            "streaming would be lost."
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
