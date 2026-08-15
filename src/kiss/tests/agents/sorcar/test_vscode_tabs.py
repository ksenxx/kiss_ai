# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for multi-tab tabId routing in the VS Code backend.

Verifies that userAnswer, error, stop, askUser, and merge events are
correctly routed to the right tab when multiple tabs are active
concurrently.  No mocks — uses real VSCodeServer instances with captured
broadcast output.  Task state lives in the task-id-keyed
:mod:`kiss.server.agent_state` registry.
"""

import queue
import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _sleep_swallowing_kbi(seconds: float) -> None:
    """Sleep for *seconds*, swallowing ``KeyboardInterrupt``.

    Used as a daemon thread target where a stop test injects a
    ``KeyboardInterrupt`` via ``PyThreadState_SetAsyncExc``.  Without
    this guard, the injected exception propagates out of the thread and
    is captured by pytest's threading hook as
    :class:`pytest.PytestUnhandledThreadExceptionWarning`, polluting the
    test output (and, due to delayed delivery while the thread is
    sleeping, attributing the warning to whichever test happens to run
    next).
    """
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass


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


def _register_state(
    task_id: str,
    tab_id: str = "",
    *,
    with_queue: bool = False,
) -> agent_state.AgentState:
    """Register a server-owned AgentState for a test tab/task."""
    state = agent_state.AgentState(
        task_id,
        tab_id=tab_id,
        server_owned=True,
    )
    if with_queue:
        state.user_answer_queue = queue.Queue(maxsize=1)
    agent_state.register(state)
    return state


class _RegistryCleanupTestCase(unittest.TestCase):
    """Base that guarantees the agent-state registry is left empty."""

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()


class TestUserAnswerRouting(_RegistryCleanupTestCase):
    """userAnswer commands are delivered to the correct task's queue."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_answer_reaches_correct_tab_queue(self) -> None:
        """Answer with tabId=2 goes to tab 2's task queue, not tab 1's."""
        s1 = _register_state("task-1", "1", with_queue=True)
        s2 = _register_state("task-2", "2", with_queue=True)
        q1, q2 = s1.user_answer_queue, s2.user_answer_queue
        assert q1 is not None and q2 is not None

        self.server._handle_command({"type": "userAnswer", "answer": "hello", "tabId": "2"})

        assert q2.get_nowait() == "hello"
        assert q1.empty()

    def test_answer_without_tabid_is_dropped(self) -> None:
        """Answer with no tabId is dropped (no default queue)."""
        s1 = _register_state("task-1", "1", with_queue=True)
        q1 = s1.user_answer_queue
        assert q1 is not None

        self.server._handle_command({"type": "userAnswer", "answer": "hi"})

        assert q1.empty()

    def test_answer_for_unknown_tab_is_dropped(self) -> None:
        """Answer for a tab with no live task/queue is dropped."""
        self.server._handle_command({"type": "userAnswer", "answer": "x", "tabId": "99"})

    def test_stale_answer_drained_before_new_one(self) -> None:
        """A stale answer in the queue is drained before the new answer is put."""
        s3 = _register_state("task-3", "3", with_queue=True)
        q = s3.user_answer_queue
        assert q is not None
        q.put("stale")

        self.server._handle_command({"type": "userAnswer", "answer": "fresh", "tabId": "3"})

        assert q.get_nowait() == "fresh"


class TestAwaitUserResponse(_RegistryCleanupTestCase):
    """_await_user_response blocks on the current task's queue."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_reads_from_task_queue(self) -> None:
        """_await_user_response reads from the task-specific queue."""
        state = _register_state("5", "5", with_queue=True)
        q = state.user_answer_queue
        assert q is not None
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
        _register_state("6", "6", with_queue=True)
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

    def test_no_task_id_raises_keyboard_interrupt(self) -> None:
        """Without a task_id there is no queue — KeyboardInterrupt."""
        self.server.printer._thread_local.task_id = None
        stop = threading.Event()
        self.server.printer._thread_local.stop_event = stop

        with self.assertRaises(KeyboardInterrupt):
            self.server._await_user_response()


class TestTabIdInjection(unittest.TestCase):
    """Events broadcast from a task thread get tabId auto-injected."""

    def test_broadcast_injects_tabid_from_thread_local(self) -> None:
        """When thread-local task_id is set, broadcast adds tabId to events."""
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


class TestStopRouting(_RegistryCleanupTestCase):
    """Stop commands target the correct tab(s)."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_stop_with_tabid_only_stops_that_tab(self) -> None:
        """Stop with tabId=1 sets only tab 1's task stop event."""
        ev1, ev2 = threading.Event(), threading.Event()
        s1 = _register_state("task-1", "1")
        s2 = _register_state("task-2", "2")
        s1.stop_event = ev1
        s2.stop_event = ev2
        t1 = threading.Thread(
            target=_sleep_swallowing_kbi, args=(5,), daemon=True,
        )
        t2 = threading.Thread(
            target=_sleep_swallowing_kbi, args=(5,), daemon=True,
        )
        t1.start()
        t2.start()
        s1.task_thread = t1
        s2.task_thread = t2

        self.server._handle_command({"type": "stop", "tabId": "1"})
        time.sleep(0.2)

        assert ev1.is_set()
        assert not ev2.is_set()

    def test_stop_without_tabid_is_noop(self) -> None:
        """Stop with no tabId is a no-op (C4 fix).

        A missing tabId from the frontend silently does nothing rather
        than stopping every task.
        """
        ev1, ev2 = threading.Event(), threading.Event()
        s1 = _register_state("task-1", "1")
        s2 = _register_state("task-2", "2")
        s1.stop_event = ev1
        s2.stop_event = ev2
        t1 = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
        t2 = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
        t1.start()
        t2.start()
        s1.task_thread = t1
        s2.task_thread = t2

        self.server._handle_command({"type": "stop"})
        time.sleep(0.2)

        assert not ev1.is_set()
        assert not ev2.is_set()


class TestConcurrentTabs(_RegistryCleanupTestCase):
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

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (done[0] and done[1]):
            time.sleep(0.05)

        assert done[0] and done[1], f"Both tasks should have run: {done}"

    def test_duplicate_run_on_same_tab_injects_instead_of_dropping(
        self,
    ) -> None:
        """A second run on the same tab while the first is still running
        must NOT start a second task — but it must NOT silently discard
        the user's text either.  The prompt is injected into the running
        task's ``pending_user_messages`` (and echoed back as a ``prompt``
        event), exactly like an ``appendUserMessage``.

        This is the fix for "input ignored during the task" after a
        close+reopen: a re-opened webview can momentarily still believe
        the task is idle and send the typed text as a ``submit`` (→
        ``run``); the daemon — the source of truth for whether a task is
        live — injects it rather than dropping it.  There must still be
        no error broadcast and no second task thread.
        """
        started = threading.Event()
        release = threading.Event()
        call_count = [0]

        def slow_run(cmd: dict) -> None:
            call_count[0] += 1
            state = agent_state.get(cmd.get("_state_key", ""))
            assert state is not None
            state.is_task_active = True
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

        new_events = self.events[events_before:]
        assert all(e.get("type") != "error" for e in new_events)
        assert call_count[0] == 1
        state = agent_state.find_by_tab("1")
        assert state is not None
        assert state.pending_user_messages == ["task2"]
        echoes = [
            e
            for e in new_events
            if e.get("type") == "prompt"
            and e.get("tabId") == "1"
            and e.get("text") == "task2"
        ]
        assert len(echoes) == 1

        release.set()
        time.sleep(0.5)


class TestRunTaskStatusBroadcast(_RegistryCleanupTestCase):
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
        """After _run_task, the task state's thread/stop/queue are cleared."""
        def noop_inner(cmd: dict) -> None:
            pass

        self.server._run_task_inner = noop_inner  # type: ignore[assignment]
        state = _register_state("task-3", "3", with_queue=True)
        state.stop_event = threading.Event()
        state.task_thread = threading.current_thread()
        state.is_task_active = True

        self.server._run_task({
            "tabId": "3",
            "prompt": "x",
            "model": "m",
            "_state_key": "task-3",
        })

        assert state.task_thread is None
        assert state.stop_event is None
        assert state.user_answer_queue is None
        assert state.is_task_active is False


class TestAskUserQuestion(_RegistryCleanupTestCase):
    """_ask_user_question broadcasts askUser and blocks for answer."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def test_ask_user_broadcasts_question(self) -> None:
        """_ask_user_question broadcasts the question."""
        state = _register_state("8", "8", with_queue=True)
        q = state.user_answer_queue
        assert q is not None
        self.server.printer._thread_local.task_id = "8"
        self.server.printer.subscribe_tab("8", "8")
        self.server.printer._thread_local.stop_event = threading.Event()

        def _answer_when_asked() -> None:
            for _ in range(500):
                if any(e["type"] == "askUser" for e in list(self.events)):
                    q.put("yes")
                    return
                time.sleep(0.01)

        answerer = threading.Thread(target=_answer_when_asked, daemon=True)
        answerer.start()
        result = self.server._ask_user_question("Continue?")
        answerer.join(timeout=10)

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
        from kiss.server.json_printer import JsonPrinter

        printer = JsonPrinter()
        printer._thread_local.task_id = "rec-1"
        printer.start_recording()
        key = printer._task_key()
        assert key in printer._recordings
        printer.stop_recording()
        assert key not in printer._recordings


class TestSelectedModelIsolation(_RegistryCleanupTestCase):
    """S7 fix: selected model is per-tab, not global."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def tearDown(self) -> None:
        super().tearDown()
        self.server._tab_models.pop("1", None)
        self.server._tab_models.pop("2", None)

    def test_select_model_on_one_tab_does_not_affect_other(self) -> None:
        """Changing model on tab 1 leaves tab 2's pinned model unchanged."""
        self.server._handle_command({
            "type": "selectModel",
            "model": "model-b",
            "tabId": "2",
        })
        self.server._handle_command({
            "type": "selectModel",
            "model": "gpt-4o",
            "tabId": "1",
        })
        assert self.server._tab_model("1") == "gpt-4o"
        assert self.server._tab_model("2") == "model-b"

    def test_select_model_updates_default_for_new_tabs(self) -> None:
        """selectModel also updates the default so new tabs inherit it."""
        self.server._handle_command({
            "type": "selectModel",
            "model": "gpt-4o",
            "tabId": "1",
        })
        assert self.server._tab_model("99") == "gpt-4o"


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
        end = self.js_src.index("case 'showWelcome':", idx)
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


if __name__ == "__main__":
    unittest.main()
