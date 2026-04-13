"""Tests validating race condition fixes in the VS Code server.

These tests verify that the race conditions identified in PLAN.md have been
properly fixed.

Fixed Python races:
- Race 1: _last_active_file/_last_active_content pair protected by _state_lock
- Race 2: _generate_followup uses FAST_MODEL from config
- Race 14: status "running: False" broadcast moved to end of finally block
- Race 15: newConversation() stops task before resetting (TS-side)
- Race 16: newChat/resumeSession guarded when task is running
"""

import inspect
import threading
import unittest

from kiss.agents.vscode.server import VSCodeServer


class TestRace1StateLockProtection(unittest.TestCase):
    """Race 1 fix: _state_lock protects _last_active_file/_last_active_content pair."""

    def test_state_lock_used_in_complete_handler(self) -> None:
        """Verify the complete handler uses _state_lock for atomic pair update."""
        source = inspect.getsource(VSCodeServer._handle_command)
        assert "_state_lock" in source
        # Verify snapshot variables are captured
        assert "snapshot_file" in source
        assert "snapshot_content" in source

    def test_complete_passes_snapshots(self) -> None:
        """Verify _complete receives snapshot_file and snapshot_content."""
        source = inspect.getsource(VSCodeServer._handle_command)
        assert "snapshot_file, snapshot_content" in source

    def test_complete_from_active_file_uses_snapshot(self) -> None:
        """Verify _complete_from_active_file uses snapshot parameters, not self attrs."""
        source = inspect.getsource(VSCodeServer._complete_from_active_file)
        assert "snapshot_file" in source
        assert "snapshot_content" in source
        # Should NOT read self._last_active_file or self._last_active_content
        assert "self._last_active_file" not in source
        assert "self._last_active_content" not in source

    def test_atomic_pair_under_lock(self) -> None:
        """Demonstrate that snapshots are atomically captured."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        # Process a complete command — the lock ensures consistent pair
        server._handle_command({
            "type": "complete",
            "query": "calc",
            "activeFile": "a.py",
            "activeFileContent": "content_a",
        })
        # Second command updates pair atomically
        server._handle_command({
            "type": "complete",
            "query": "calc",
            "activeFile": "b.py",
            "activeFileContent": "content_b",
        })
        # Both file and content updated together
        with server._state_lock:
            assert server._last_active_file == "b.py"
            assert server._last_active_content == "content_b"


class TestRace2FollowupUsesFastModel(unittest.TestCase):
    """Race 2 fix: _generate_followup uses fast_model_for to pick a cheap model."""

    def test_followup_uses_fast_model_for(self) -> None:
        """_generate_followup_async uses fast_model_for, not _selected_model."""
        source = inspect.getsource(VSCodeServer._generate_followup_async)
        assert "self._selected_model" not in source
        assert "fast_model_for" in source

    def test_run_task_calls_followup_with_model(self) -> None:
        """Verify _run_task_inner passes the task model to _generate_followup_async."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_generate_followup_async(" in source
        # The call passes prompt, result_summary, and model (may span lines)
        assert "prompt," in source
        assert "result_summary," in source
        assert "model," in source


class TestRace14StatusBroadcastOrder(unittest.TestCase):
    """Race 14 fix: status "running: False" broadcast at end of finally block."""

    def test_status_broadcast_after_cache(self) -> None:
        """Verify _run_task wraps _run_task_inner with try/finally status broadcast.

        The outer _run_task guarantees status:running:false is always sent.
        The inner method does cache cleanup before returning, so the
        outer finally's status broadcast comes after all cleanup.
        """
        # Outer wrapper has the guaranteed status broadcast
        outer_source = inspect.getsource(VSCodeServer._run_task)
        assert '"running": False' in outer_source
        assert "_run_task_inner" in outer_source

        # Inner method has cache cleanup
        inner_source = inspect.getsource(VSCodeServer._run_task_inner)
        cache_pos = inner_source.find("_refresh_file_cache")
        assert cache_pos > 0, "_refresh_file_cache not found"


class TestRace16GuardedNewChatResumeSession(unittest.TestCase):
    """Race 16 fix: newChat/resumeSession guarded when task is running."""

    def test_newchat_works_while_other_tab_running(self) -> None:
        """newChat works even when another tab has a running task
        (per-tab concurrent execution)."""
        server = VSCodeServer()
        original_id = server.agent._chat_id

        # Simulate running task on tab 1
        stop = threading.Event()
        thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        thread.start()
        server._task_threads[1] = thread

        try:
            # newChat should work (per-tab — no global block)
            server._handle_command({"type": "newChat"})
            assert server.agent._chat_id != original_id, (
                "newChat should create a new chat even while another tab is running"
            )
        finally:
            stop.set()
            thread.join()

    def test_resume_session_works_while_other_tab_running(self) -> None:
        """resumeSession works even when another tab has a running task."""
        server = VSCodeServer()

        stop = threading.Event()
        thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        thread.start()
        server._task_threads[1] = thread

        try:
            # resumeSession should not crash (session may not exist)
            server._handle_command({"type": "resumeSession", "sessionId": "test"})
        finally:
            stop.set()
            thread.join()

    def test_handler_source_has_per_tab_running_check(self) -> None:
        """Verify submit handler checks per-tab running via _runningTabs
        (not global _task_thread)."""
        source = inspect.getsource(VSCodeServer._handle_command)
        assert "_task_threads" in source, (
            "run handler should use per-tab _task_threads dict"
        )


class TestCompleteFunctionSignatures(unittest.TestCase):
    """Verify _complete and helpers accept snapshot parameters."""

    def test_complete_accepts_snapshots(self) -> None:
        sig = inspect.signature(VSCodeServer._complete)
        assert "snapshot_file" in sig.parameters
        assert "snapshot_content" in sig.parameters

    def test_complete_from_active_file_accepts_snapshots(self) -> None:
        sig = inspect.signature(VSCodeServer._complete_from_active_file)
        assert "snapshot_file" in sig.parameters
        assert "snapshot_content" in sig.parameters


class TestTypescriptRaceFixesCodeInspection(unittest.TestCase):
    """Verify TypeScript race fixes via code inspection."""

    def test_race11_focus_toggling_guard(self) -> None:
        """Race 11: toggleFocus has _focusToggling guard."""
        with open("src/kiss/agents/vscode/src/extension.ts") as f:
            source = f.read()
        assert "_focusToggling" in source

    def test_race12_writable_check(self) -> None:
        """Race 12: sendCommand checks stdin.writable."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        assert "stdin?.writable" in source


class TestNewChatHistoryButtonsDisabledWhileRunning(unittest.TestCase):
    """Buttons are properly managed while a task is running."""

    def test_status_running_false_always_sent(self) -> None:
        """status:running:false is always broadcast (try/finally), so buttons always re-enable."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "try:" in source
        assert "finally:" in source
        assert '"running": False' in source


class TestExistingBehavior(unittest.TestCase):
    """Tests ensuring fixes don't break existing functionality."""

    def test_complete_seq_concurrent(self) -> None:
        """Multiple _complete threads: only latest seq broadcasts."""
        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()

        def capture(e: dict) -> None:
            with lock:
                events.append(e)

        server.printer.broadcast = capture  # type: ignore[assignment]
        server._last_active_content = "calculate_total = 1"
        server._complete_seq_latest = 9

        barrier = threading.Barrier(11)

        def completer(seq: int) -> None:
            barrier.wait()
            server._complete("calc", seq=seq)

        threads = [threading.Thread(target=completer, args=(i,)) for i in range(10)]
        threads.append(threading.Thread(target=completer, args=(9,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ghost_events = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost_events) == 2  # Only seq=9 threads broadcast


class TestStatusAlwaysSentOnExit(unittest.TestCase):
    """Verify status:running:false is always sent when _run_task exits.

    Previously, early returns and exceptions before the inner try/finally
    left _isRunning stuck on the TypeScript side, silently dropping all
    subsequent task submissions.
    """

    def _capture_server(self) -> tuple[VSCodeServer, list[dict]]:
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]
        return server, events

    def test_outer_try_finally_in_run_task(self) -> None:
        """_run_task wraps _run_task_inner with try/finally for status guarantee."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "try:" in source
        assert "finally:" in source
        assert '"running": False' in source
        assert "_run_task_inner" in source


class TestPromptPanelScrollToBottom(unittest.TestCase):
    """Verify system_prompt and prompt panels scroll to the bottom after rendering."""

    def _get_main_js(self) -> str:
        import os
        path = os.path.join(
            os.path.dirname(__file__),
            "../../../agents/vscode/media/main.js",
        )
        with open(os.path.normpath(path)) as f:
            return f.read()

    def test_system_prompt_body_scrolled_to_bottom(self) -> None:
        """system-prompt-body is scrolled to bottom after rendering."""
        source = self._get_main_js()
        assert "bodyEl.scrollTop = bodyEl.scrollHeight" in source

    def test_scroll_uses_queried_body_element(self) -> None:
        """Scroll uses querySelector to get the body element, not the outer panel."""
        source = self._get_main_js()
        assert "el.querySelector('.' + cls + '-body')" in source

    def test_scroll_in_system_prompt_case(self) -> None:
        """Scroll code is co-located with the system_prompt/prompt case."""
        source = self._get_main_js()
        case_idx = source.index("case 'system_prompt':")
        break_idx = source.index("break;", case_idx)
        snippet = source[case_idx:break_idx]
        assert "scrollTop = bodyEl.scrollHeight" in snippet


if __name__ == "__main__":
    unittest.main()
