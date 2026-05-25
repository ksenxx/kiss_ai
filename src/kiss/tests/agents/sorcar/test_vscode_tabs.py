"""Integration tests for multi-tab tabId routing in the VS Code backend.

Verifies that userAnswer, error, stop, askUser, and merge events are
correctly routed to the right tab when multiple tabs are active
concurrently.  No mocks — uses real VSCodeServer instances with captured
broadcast output.
"""

import queue
import threading
import time
import unittest

from kiss.agents.vscode.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture.

    Returns:
        (server, events) — the events list collects all broadcast calls.
    """
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestUserAnswerRouting(unittest.TestCase):
    """userAnswer commands are delivered to the correct tab's queue."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_answer_reaches_correct_tab_queue(self) -> None:
        """Answer with tabId=2 goes to tab 2's queue, not tab 1's."""
        q1: queue.Queue[str] = queue.Queue(maxsize=1)
        q2: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("1").user_answer_queue = q1
        self.server._get_tab("2").user_answer_queue = q2

        self.server._handle_command({"type": "userAnswer", "answer": "hello", "tabId": "2"})

        assert q2.get_nowait() == "hello"
        assert q1.empty()

    def test_answer_without_tabid_is_dropped(self) -> None:
        """Answer with no tabId is dropped (no default queue)."""
        q1: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("1").user_answer_queue = q1

        self.server._handle_command({"type": "userAnswer", "answer": "hi"})

        assert q1.empty()

    def test_answer_for_unknown_tab_is_dropped(self) -> None:
        """Answer for a tab with no queue is dropped."""
        self.server._handle_command({"type": "userAnswer", "answer": "x", "tabId": "99"})


    def test_stale_answer_drained_before_new_one(self) -> None:
        """A stale answer in the queue is drained before the new answer is put."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        q.put("stale")
        self.server._get_tab("3").user_answer_queue = q

        self.server._handle_command({"type": "userAnswer", "answer": "fresh", "tabId": "3"})

        assert q.get_nowait() == "fresh"


class TestAwaitUserResponse(unittest.TestCase):
    """_await_user_response blocks on the correct per-tab queue."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_reads_from_tab_queue(self) -> None:
        """_await_user_response reads from the tab-specific queue."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("5").user_answer_queue = q
        self.server.printer._thread_local.task_id = "5"
        self.server.printer.subscribe_tab("5", "5")
        stop = threading.Event()
        self.server.printer._thread_local.stop_event = stop

        def answer_later() -> None:
            time.sleep(0.1)
            q.put("the answer")

        threading.Thread(target=answer_later, daemon=True).start()
        result = self.server._await_user_response()
        assert result == "the answer"

    def test_raises_on_stop_event(self) -> None:
        """_await_user_response raises KeyboardInterrupt when stop is set."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("6").user_answer_queue = q
        self.server.printer._thread_local.task_id = "6"
        self.server.printer.subscribe_tab("6", "6")
        stop = threading.Event()
        self.server.printer._thread_local.stop_event = stop

        def set_stop_later() -> None:
            time.sleep(0.1)
            stop.set()

        threading.Thread(target=set_stop_later, daemon=True).start()
        with self.assertRaises(KeyboardInterrupt):
            self.server._await_user_response()

    def test_no_tab_id_waits_until_stop(self) -> None:
        """Without tab_id, _await_user_response waits until stop is set."""
        self.server.printer._thread_local.task_id = None
        stop = threading.Event()
        self.server.printer._thread_local.stop_event = stop

        def set_stop_later() -> None:
            time.sleep(0.1)
            stop.set()

        threading.Thread(target=set_stop_later, daemon=True).start()
        with self.assertRaises(KeyboardInterrupt):
            self.server._await_user_response()


class TestTabIdInjection(unittest.TestCase):
    """Events broadcast from a task thread get tabId auto-injected."""

    def test_broadcast_injects_tabid_from_thread_local(self) -> None:
        """When thread-local tab_id is set, broadcast adds tabId to events."""
        from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        printer._thread_local.task_id = "7"
        printer.subscribe_tab("7", "7")

        printer.broadcast({"type": "askUser", "question": "What?"})

        assert len(printer.emitted) == 1
        assert printer.emitted[0]["tabId"] == "7"
        assert printer.emitted[0]["type"] == "askUser"

    def test_broadcast_does_not_overwrite_explicit_tabid(self) -> None:
        """If event already has tabId, broadcast does not overwrite it."""
        from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        printer._thread_local.task_id = "7"

        printer.broadcast({"type": "error", "text": "oops", "tabId": "3"})

        assert len(printer.emitted) == 1
        assert printer.emitted[0]["tabId"] == "3"


class TestStopRouting(unittest.TestCase):
    """Stop commands target the correct tab(s)."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_stop_with_tabid_only_stops_that_tab(self) -> None:
        """Stop with tabId=1 sets only tab 1's stop event."""
        ev1, ev2 = threading.Event(), threading.Event()
        tab1 = self.server._get_tab("1")
        tab2 = self.server._get_tab("2")
        tab1.stop_event = ev1
        tab2.stop_event = ev2
        t1 = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        t2 = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        t1.start()
        t2.start()
        tab1.task_thread = t1
        tab2.task_thread = t2

        self.server._handle_command({"type": "stop", "tabId": "1"})
        time.sleep(0.2)

        assert ev1.is_set()
        assert not ev2.is_set()

    def test_stop_without_tabid_is_noop(self) -> None:
        """Stop with no tabId is a no-op (C4 fix).

        Previously, ``_stop_task(None)`` stopped every tab's task,
        violating per-tab state isolation.  A missing tabId from the
        frontend now silently does nothing.
        """
        ev1, ev2 = threading.Event(), threading.Event()
        tab1 = self.server._get_tab("1")
        tab2 = self.server._get_tab("2")
        tab1.stop_event = ev1
        tab2.stop_event = ev2
        t1 = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
        t2 = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
        t1.start()
        t2.start()
        tab1.task_thread = t1
        tab2.task_thread = t2

        self.server._handle_command({"type": "stop"})
        time.sleep(0.2)

        assert not ev1.is_set()
        assert not ev2.is_set()


class TestConcurrentTabs(unittest.TestCase):
    """Two tasks on different tabs run concurrently without interference."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_two_tabs_run_concurrently(self) -> None:
        """Tasks on tab 1 and tab 2 run simultaneously."""
        barrier = threading.Barrier(2, timeout=5)
        done = [False, False]

        def slow_run(cmd: dict) -> None:
            tab = cmd.get("tabId", "")
            idx = 0 if tab == "1" else 1
            barrier.wait()
            done[idx] = True

        self.server._run_task_inner = slow_run  # type: ignore[assignment]

        self.server._handle_command({
            "type": "run", "prompt": "task1", "model": "m", "tabId": "1",
        })
        self.server._handle_command({
            "type": "run", "prompt": "task2", "model": "m", "tabId": "2",
        })

        time.sleep(2)
        threads = [
            t.task_thread for t in self.server._running_agent_states.values()
            if t.task_thread is not None
        ]
        for t in threads:
            t.join(timeout=5)

        assert done[0] and done[1], f"Both tasks should have run: {done}"

    def test_duplicate_run_on_same_tab_dropped_silently(self) -> None:
        """A second run on the same tab is silently dropped while first is running.

        The user-visible behaviour is that nothing happens: no error
        broadcast, no new task thread, no status flap.
        """
        started = threading.Event()
        release = threading.Event()
        call_count = [0]

        def slow_run(cmd: dict) -> None:
            call_count[0] += 1
            started.set()
            release.wait(timeout=5)

        self.server._run_task_inner = slow_run  # type: ignore[assignment]

        self.server._handle_command({
            "type": "run", "prompt": "task1", "model": "m", "tabId": "1",
        })
        started.wait(timeout=3)

        events_before = len(self.events)
        self.server._handle_command({
            "type": "run", "prompt": "task2", "model": "m", "tabId": "1",
        })

        # No error broadcast, no status broadcast, no new run.
        new_events = self.events[events_before:]
        assert all(e.get("type") != "error" for e in new_events)
        assert call_count[0] == 1

        release.set()
        time.sleep(0.5)


class TestMergeTabIsolation(unittest.TestCase):
    """Merge ownership is per-tab — other tabs are unaffected."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_finish_merge_broadcasts_with_tabid(self) -> None:
        """_finish_merge includes the tab's tabId in merge_ended."""
        self.server._get_tab("42").is_merging = True
        self.server._finish_merge("42")
        ended = [e for e in self.events if e["type"] == "merge_ended"]
        assert len(ended) == 1
        assert ended[0]["tabId"] == "42"
        assert self.server._get_tab("42").is_merging is False

    def test_finish_merge_no_tab_is_noop(self) -> None:
        """_finish_merge with no tab_id is a no-op (B8 fix).

        Previously it cleared every tab's ``is_merging`` flag and
        emitted an untagged ``merge_ended``, violating per-tab state
        isolation.
        """
        self.server._get_tab("10").is_merging = True
        self.server._finish_merge()
        ended = [e for e in self.events if e["type"] == "merge_ended"]
        assert ended == []
        assert self.server._get_tab("10").is_merging is True

    def test_merging_tabs_are_independent(self) -> None:
        """Multiple tabs can be in merge state simultaneously."""
        self.server._get_tab("1").is_merging = True
        self.server._get_tab("2").is_merging = True
        assert self.server._get_tab("1").is_merging is True
        assert self.server._get_tab("2").is_merging is True
        self.server._finish_merge("1")
        assert self.server._get_tab("1").is_merging is False
        assert self.server._get_tab("2").is_merging is True

    def test_merge_guard_blocks_only_merging_tab(self) -> None:
        """_run_task_inner rejects runs on merging tabs but allows others."""
        self.server._get_tab("1").is_merging = True
        assert self.server._get_tab("1").is_merging is True
        assert self.server._get_tab("2").is_merging is False


class TestRunTaskStatusBroadcast(unittest.TestCase):
    """_run_task always brackets execution with status events."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_status_running_true_then_false(self) -> None:
        """_run_task broadcasts running=true then running=false."""
        def noop_inner(cmd: dict) -> None:
            pass

        self.server._run_task_inner = noop_inner  # type: ignore[assignment]

        self.server._run_task({"tabId": "1", "prompt": "x", "model": "m"})

        status_events = [e for e in self.events if e["type"] == "status"]
        assert len(status_events) >= 2
        assert status_events[0]["running"] is True
        assert status_events[-1]["running"] is False

    def test_status_false_even_on_exception(self) -> None:
        """_run_task broadcasts running=false even when inner raises."""
        def failing_inner(cmd: dict) -> None:
            raise RuntimeError("boom")

        self.server._run_task_inner = failing_inner  # type: ignore[assignment]

        t = threading.Thread(
            target=self.server._run_task,
            args=({"tabId": "2", "prompt": "x", "model": "m"},),
            daemon=True,
        )
        t.start()
        t.join(timeout=5)

        status_events = [e for e in self.events if e["type"] == "status"]
        assert status_events[-1]["running"] is False

    def test_run_task_cleans_up_thread_state(self) -> None:
        """After _run_task, the tab's thread/stop/queue are cleared."""
        def noop_inner(cmd: dict) -> None:
            pass

        self.server._run_task_inner = noop_inner  # type: ignore[assignment]
        self.server._run_task({"tabId": "3", "prompt": "x", "model": "m"})

        tab = self.server._get_tab("3")
        assert tab.task_thread is None
        assert tab.stop_event is None
        assert tab.user_answer_queue is None


class TestAskUserQuestion(unittest.TestCase):
    """_ask_user_question broadcasts askUser and blocks for answer."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_ask_user_broadcasts_question(self) -> None:
        """_ask_user_question broadcasts the question."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        q.put("yes")
        self.server._get_tab("8").user_answer_queue = q
        self.server.printer._thread_local.task_id = "8"
        self.server.printer.subscribe_tab("8", "8")
        self.server.printer._thread_local.stop_event = threading.Event()

        result = self.server._ask_user_question("Continue?")

        asks = [e for e in self.events if e["type"] == "askUser"]
        assert len(asks) == 1
        assert asks[0]["question"] == "Continue?"
        assert result == "yes"






class TestBashFlushTimerTabId(unittest.TestCase):
    """The 0.1s bash flush timer propagates the owning thread's tab_id."""

    def test_timer_flush_injects_tab_id(self) -> None:
        """Bash output flushed by the timer includes the correct tabId."""
        from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        printer._thread_local.task_id = "99"

        with printer._bash_lock:
            printer._bash_state.last_flush = time.monotonic()
        printer.print("line1\n", type="bash_stream")
        time.sleep(0.5)

        for event in printer.emitted:
            if event.get("type") == "system_output":
                assert event.get("tabId") == "99", (
                    f"Expected tabId='99', got {event.get('tabId')}"
                )


class TestRecordingIsolation(unittest.TestCase):
    """Recording captures all broadcast events (no owner filtering needed
    with per-task processes — each process has its own printer)."""

    def test_recording_captures_own_tab_events(self) -> None:
        """Recording captures events for the current tab (per-tab isolation)."""
        from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        printer._thread_local.task_id = "1"

        printer.start_recording()

        printer.broadcast({"type": "tool_call", "name": "Read"})
        printer.broadcast({"type": "tool_result", "content": "ok"})
        printer.broadcast({"type": "prompt", "text": "global event"})

        events = printer.stop_recording()

        types = [e["type"] for e in events]
        assert types == ["tool_call", "tool_result", "prompt"]

    def test_stop_recording_clears_state(self) -> None:
        """stop_recording removes the tab's recording entry."""
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        printer._thread_local.task_id = "rec-1"
        printer.start_recording()
        key = printer._task_key()
        assert key in printer._recordings
        printer.stop_recording()
        assert key not in printer._recordings


if __name__ == "__main__":
    unittest.main()


class TestPerTabAgentIsolation(unittest.TestCase):
    """Each tab gets its own agent instances — no cross-tab state leakage."""

    def test_different_tabs_have_independent_chat_ids(self) -> None:
        """Tabs hold their own chat_id (no shared per-tab agent)."""
        server, _ = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        assert tab1 is not tab2
        assert tab1.chat_id == ""
        assert tab2.chat_id == ""

    def test_new_chat_on_one_tab_does_not_affect_other(self) -> None:
        """Resetting tab 1's chat_id does not change tab 2's chat_id."""
        server, _ = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        tab2.chat_id = "42"
        tab1.chat_id = ""
        assert tab2.chat_id == "42"

    def test_use_worktree_is_per_tab(self) -> None:
        """Setting use_worktree on one tab does not affect others."""
        server, _ = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        tab1.use_worktree = True
        assert tab2.use_worktree is False

    def test_use_parallel_is_per_tab(self) -> None:
        """Setting use_parallel on one tab does not affect others."""
        server, _ = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        tab1.use_parallel = False
        assert tab2.use_parallel is True

    def test_task_history_id_is_per_tab(self) -> None:
        """task_history_id is independent per tab."""
        server, _ = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        tab1.task_history_id = 42
        assert tab2.task_history_id is None

    def test_get_tab_creates_on_demand(self) -> None:
        """_get_tab creates a new _RunningAgentState if one doesn't exist."""
        server, _ = _make_server()
        assert "99" not in server._running_agent_states
        tab = server._get_tab("99")
        assert "99" in server._running_agent_states
        assert tab is server._get_tab("99")

    def test_agent_is_transient_per_task(self) -> None:
        """tab.agent is replaced fresh per task — no state persists across tasks."""
        server, _ = _make_server()
        tab = server._get_tab("1")
        # ``_get_tab`` lazily allocates a slot agent so tests and
        # out-of-task callers (worktree merge / discard, conflict
        # check, history click) can read agent state directly.  The
        # slot is wiped to ``None`` at the end of every ``_run_task``
        # so no agent state survives a task boundary; a fresh agent
        # is built in ``_cmd_run`` immediately before each worker
        # thread starts.
        first = tab.agent
        assert first is not None
        tab.agent = None
        again = tab  # same _RunningAgentState
        again = server._get_tab("1")  # _get_tab repopulates the slot
        assert again.agent is not None
        assert again.agent is not first  # fresh per allocation


class TestSelectedModelIsolation(unittest.TestCase):
    """S7 fix: selected_model is per-tab, not global."""

    def test_select_model_on_one_tab_does_not_affect_other(self) -> None:
        """Changing model on tab 1 leaves tab 2's model unchanged."""
        server, events = _make_server()
        tab1 = server._get_tab("1")
        tab2 = server._get_tab("2")
        original = tab2.selected_model

        server._handle_command({
            "type": "selectModel",
            "model": "gpt-4o",
            "tabId": "1",
        })
        assert tab1.selected_model == "gpt-4o"
        assert tab2.selected_model == original

    def test_select_model_updates_default_for_new_tabs(self) -> None:
        """selectModel also updates the default so new tabs inherit it."""
        server, _ = _make_server()
        server._handle_command({
            "type": "selectModel",
            "model": "gpt-4o",
            "tabId": "1",
        })
        tab99 = server._get_tab("99")
        assert tab99.selected_model == "gpt-4o"



class TestBashBufferIsolation(unittest.TestCase):
    """S11 fix: bash buffer is per-tab, not shared."""

    def test_bash_state_exists(self) -> None:
        """Printer has a single _bash_state instance."""
        server, _ = _make_server()
        printer = server.printer
        bs = printer._bash_state
        assert bs.buffer == []
        assert bs.timer is None
        assert bs.generation == 0

    def test_offsets_default_to_zero(self) -> None:
        """tokens_offset, budget_offset, steps_offset default to 0."""
        server, _ = _make_server()
        printer = server.printer
        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0


class TestClearChatDedup(unittest.TestCase):
    """When the secondary panel is closed and re-opened, clicking the KS
    button fires newConversation which sends clearChat.  The webview
    already has a fresh empty tab from initialization, so clearChat
    must NOT create a second one.

    The fix: the clearChat handler in main.js checks whether the active
    tab is already an empty new-chat tab (no backendChatId, welcome
    visible) and skips createNewTab() in that case.
    """

    js_src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        js_path = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "main.js"
        )
        cls.js_src = js_path.read_text()

    def _get_clear_chat_block(self) -> str:
        idx = self.js_src.index("case 'clearChat':")
        end = self.js_src.index("case 'ensureChat':", idx)
        return self.js_src[idx:end]

    def test_clear_chat_checks_backend_chat_id(self) -> None:
        """clearChat handler guards against creating a duplicate empty tab
        by checking that the active tab has no backendChatId."""
        block = self._get_clear_chat_block()
        assert "backendChatId" in block, (
            "clearChat handler must check backendChatId to avoid "
            "creating a duplicate empty tab when the panel is freshly opened"
        )

    def test_clear_chat_checks_welcome_visible(self) -> None:
        """clearChat handler checks that the welcome screen is still visible
        (i.e. the tab has no output content) before skipping tab creation."""
        block = self._get_clear_chat_block()
        assert "welcome" in block.lower(), (
            "clearChat handler must check welcome visibility to detect "
            "that the tab is still empty"
        )
