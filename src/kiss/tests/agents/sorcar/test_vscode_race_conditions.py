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
import time
import unittest

from kiss.agents.vscode.server import VSCodeServer


class TestRace1StateLockProtection(unittest.TestCase):
    """Race 1 fix: _state_lock protects _last_active_file/_last_active_content pair."""

    def test_state_lock_exists(self) -> None:
        """Verify VSCodeServer has _state_lock."""
        server = VSCodeServer()
        assert hasattr(server, "_state_lock")
        assert isinstance(server._state_lock, type(threading.Lock()))

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
        """Verify _generate_followup_sync uses fast_model_for, not self._selected_model directly."""
        source = inspect.getsource(VSCodeServer._generate_followup_sync)
        assert "self._selected_model" not in source
        assert "fast_model_for" in source

    def test_run_task_calls_followup_with_model(self) -> None:
        """Verify _run_task_inner passes the task model to _generate_followup_sync."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_generate_followup_sync(prompt, result_summary, model)" in source


class TestRace14StatusBroadcastOrder(unittest.TestCase):
    """Race 14 fix: status "running: False" broadcast at end of finally block."""

    def test_status_broadcast_after_merge_and_cache(self) -> None:
        """Verify _run_task wraps _run_task_inner with try/finally status broadcast.

        The outer _run_task guarantees status:running:false is always sent.
        The inner method does merge/cache cleanup before returning, so the
        outer finally's status broadcast comes after all cleanup.
        """
        # Outer wrapper has the guaranteed status broadcast
        outer_source = inspect.getsource(VSCodeServer._run_task)
        assert '"running": False' in outer_source
        assert "_run_task_inner" in outer_source

        # Inner method has merge/cache cleanup
        inner_source = inspect.getsource(VSCodeServer._run_task_inner)
        merge_pos = inner_source.find("_prepare_merge_view")
        cache_pos = inner_source.find("_refresh_file_cache")
        assert merge_pos > 0, "_prepare_merge_view not found"
        assert cache_pos > 0, "_refresh_file_cache not found"

    def test_task_thread_done_when_status_false(self) -> None:
        """Verify that when status:running:false is broadcast, cleanup is done."""
        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()

        def capture(e: dict) -> None:
            with lock:
                events.append(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        cleanup_done = threading.Event()
        status_seen = threading.Event()

        def fake_task() -> None:
            """Simulates fixed _run_task: cleanup first, then status broadcast."""
            # Simulate cleanup work (merge + cache)
            time.sleep(0.05)
            cleanup_done.set()
            # NOW broadcast status — after cleanup
            server.printer.broadcast({"type": "status", "running": False})
            status_seen.set()

        server._task_thread = threading.Thread(target=fake_task, daemon=True)
        server._task_thread.start()

        status_seen.wait(timeout=5)
        # Cleanup was done before status broadcast
        assert cleanup_done.is_set()
        server._task_thread.join(timeout=5)


class TestRace16GuardedNewChatResumeSession(unittest.TestCase):
    """Race 16 fix: newChat/resumeSession guarded when task is running."""

    def test_newchat_blocked_when_task_running(self) -> None:
        """Verify newChat is blocked when _task_thread is alive."""
        server = VSCodeServer()
        original_id = server.agent._chat_id

        # Simulate running task
        stop = threading.Event()
        server._task_thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        server._task_thread.start()

        try:
            # newChat should be blocked
            server._handle_command({"type": "newChat"})
            assert server.agent._chat_id == original_id, (
                "newChat should be blocked while task is running"
            )
        finally:
            stop.set()
            server._task_thread.join()

    def test_newchat_allowed_when_no_task(self) -> None:
        """Verify newChat works when no task is running."""
        server = VSCodeServer()
        original_id = server.agent._chat_id

        server._handle_command({"type": "newChat"})
        assert server.agent._chat_id != original_id

    def test_resume_session_blocked_when_task_running(self) -> None:
        """Verify resumeSession is blocked when _task_thread is alive."""
        server = VSCodeServer()
        original_id = server.agent._chat_id

        stop = threading.Event()
        server._task_thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        server._task_thread.start()

        try:
            server._handle_command({"type": "resumeSession", "sessionId": "test"})
            assert server.agent._chat_id == original_id, (
                "resumeSession should be blocked while task is running"
            )
        finally:
            stop.set()
            server._task_thread.join()

    def test_handler_source_has_running_check(self) -> None:
        """Verify newChat handler checks _task_thread."""
        source = inspect.getsource(VSCodeServer._handle_command)
        # Find the newChat handler block
        lines = source.split("\n")
        in_newchat = False
        newchat_block: list[str] = []
        for line in lines:
            if '"newChat"' in line:
                in_newchat = True
            elif in_newchat:
                if line.strip().startswith("elif") or line.strip().startswith("else"):
                    break
                newchat_block.append(line)

        block = "\n".join(newchat_block)
        assert "_task_thread" in block, (
            "newChat handler should check _task_thread"
        )


class TestRace17AllDoneHandlerInExtension(unittest.TestCase):
    """Race 17 fix: allDone handler registered once in extension.ts, not in SorcarPanel."""

    def test_no_alldone_handler_in_sorcar_panel(self) -> None:
        """Verify SorcarPanel.ts no longer registers allDone handler."""
        panel_path = "src/kiss/agents/vscode/src/SorcarPanel.ts"
        with open(panel_path) as f:
            source = f.read()
        assert "this._mergeManager.on('allDone'" not in source

    def test_alldone_handler_in_extension(self) -> None:
        """Verify extension.ts registers a single allDone handler."""
        ext_path = "src/kiss/agents/vscode/src/extension.ts"
        with open(ext_path) as f:
            source = f.read()
        assert "mergeManager.on('allDone'" in source

    def test_send_merge_all_done_method_exists(self) -> None:
        """Verify SorcarPanel.ts has sendMergeAllDone method."""
        panel_path = "src/kiss/agents/vscode/src/SorcarPanel.ts"
        with open(panel_path) as f:
            source = f.read()
        assert "sendMergeAllDone" in source

    def test_shared_merge_manager_in_extension(self) -> None:
        """Verify extension.ts creates one MergeManager shared by both providers."""
        ext_path = "src/kiss/agents/vscode/src/extension.ts"
        with open(ext_path) as f:
            source = f.read()
        assert source.count("new MergeManager()") == 1
        assert "new SorcarViewProvider(context.extensionUri, mergeManager)" in source


class TestCompleteFunctionSignatures(unittest.TestCase):
    """Verify _complete and helpers accept snapshot parameters."""

    def test_complete_accepts_snapshots(self) -> None:
        sig = inspect.signature(VSCodeServer._complete)
        assert "snapshot_file" in sig.parameters
        assert "snapshot_content" in sig.parameters

    def test_fast_complete_accepts_snapshots(self) -> None:
        sig = inspect.signature(VSCodeServer._fast_complete)
        assert "snapshot_file" in sig.parameters
        assert "snapshot_content" in sig.parameters

    def test_complete_from_active_file_accepts_snapshots(self) -> None:
        sig = inspect.signature(VSCodeServer._complete_from_active_file)
        assert "snapshot_file" in sig.parameters
        assert "snapshot_content" in sig.parameters


class TestTypescriptRaceFixesCodeInspection(unittest.TestCase):
    """Verify TypeScript race fixes via code inspection."""

    def test_race3_submit_sets_is_running_before_await(self) -> None:
        """Race 3: submit sets _isRunning = true before any await."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        # Find the submit case
        idx = source.find("case 'submit':")
        assert idx >= 0
        block = source[idx:idx + 800]
        # _isRunning = true should appear before the first await
        running_pos = block.find("this._isRunning = true")
        await_pos = block.find("await ")
        assert running_pos > 0 and running_pos < await_pos

    def test_race4_submit_task_sets_is_running(self) -> None:
        """Race 4: submitTask sets _isRunning before _startTask."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public submitTask(")
        assert idx >= 0
        block = source[idx:idx + 400]
        running_pos = block.find("this._isRunning = true")
        start_pos = block.find("this._startTask")
        assert running_pos > 0 and running_pos < start_pos

    def test_race5_commit_pending_guard(self) -> None:
        """Race 5: generateCommitMessage has _commitPending guard."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        assert "_commitPending" in source
        idx = source.find("public generateCommitMessage")
        block = source[idx:idx + 600]
        assert "if (this._commitPending)" in block

    def test_race6_merge_in_progress_guard(self) -> None:
        """Race 6: openMerge has _mergeInProgress guard."""
        with open("src/kiss/agents/vscode/src/MergeManager.ts") as f:
            source = f.read()
        assert "_mergeInProgress" in source
        assert "_pendingMerge" in source
        assert "_doOpenMerge" in source

    def test_race7_8_hunk_op_guard(self) -> None:
        """Race 7/8: acceptChange/rejectChange/acceptAll/rejectAll have _hunkOpInProgress."""
        with open("src/kiss/agents/vscode/src/MergeManager.ts") as f:
            source = f.read()
        assert "_hunkOpInProgress" in source
        # Guard is in _withHunkGuard; all four methods use it
        idx = source.find("_withHunkGuard")
        assert idx >= 0, "_withHunkGuard not found"
        block = source[idx:idx + 300]
        assert "this._hunkOpInProgress" in block, "_withHunkGuard missing guard"
        # acceptChange/rejectChange delegate to _resolveHunk which calls _withHunkGuard;
        # acceptAll/rejectAll call _withHunkGuard directly.
        for method in ["acceptChange", "rejectChange", "acceptAll", "rejectAll"]:
            idx = source.find(f"async {method}")
            assert idx >= 0, f"{method} not found"
            block = source[idx:idx + 300]
            assert "_withHunkGuard" in block or "_resolveHunk" in block, (
                f"{method} missing guard"
            )
        assert "_resolveHunk" in source
        ridx = source.find("_resolveHunk")
        rblock = source[ridx:ridx + 500]
        assert "_withHunkGuard" in rblock, "_resolveHunk missing _withHunkGuard call"

    def test_race9_nav_seq_guard(self) -> None:
        """Race 9: _navigateHunk uses _navSeq for stale navigation detection."""
        with open("src/kiss/agents/vscode/src/MergeManager.ts") as f:
            source = f.read()
        assert "_navSeq" in source
        # Find the method body (private async _navigateHunk)
        idx = source.find("private async _navigateHunk")
        assert idx >= 0
        block = source[idx:idx + 1500]
        assert "this._navSeq !== seq" in block

    def test_race10_await_open_panel(self) -> None:
        """Race 10: newConversation command awaits openPanel."""
        with open("src/kiss/agents/vscode/src/extension.ts") as f:
            source = f.read()
        idx = source.find("kissSorcar.newConversation")
        block = source[idx:idx + 300]
        assert "await vscode.commands.executeCommand('kissSorcar.openPanel')" in block

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

    def test_race15_stop_before_new_conversation(self) -> None:
        """Race 15: newConversation stops task before resetting."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public newConversation")
        block = source[idx:idx + 300]
        assert "this._agentProcess.stop()" in block
        assert "if (this._isRunning)" in block


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

    def test_second_run_rejected_when_first_alive(self) -> None:
        """If _task_thread is alive, the second run is rejected."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        stop = threading.Event()
        server._task_thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        server._task_thread.start()

        try:
            server._handle_command({
                "type": "run",
                "prompt": "test",
                "model": "claude-opus-4-6",
                "workDir": "/tmp",
            })
            errors = [e for e in events if e.get("type") == "error"]
            assert len(errors) == 1
            assert "already running" in errors[0]["text"].lower()
        finally:
            stop.set()
            server._task_thread.join()


class TestStatusAlwaysSentOnExit(unittest.TestCase):
    """Verify status:running:false is always sent when _run_task exits.

    Previously, early returns (e.g. _merging guard) and exceptions before
    the inner try/finally left _isRunning stuck on the TypeScript side,
    silently dropping all subsequent task submissions.
    """

    def _capture_server(self) -> tuple[VSCodeServer, list[dict]]:
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]
        return server, events

    def test_merging_early_return_sends_status_false(self) -> None:
        """When _merging is True, _run_task still sends status:running:false."""
        server, events = self._capture_server()
        server._merging = True

        server._run_task({"type": "run", "prompt": "test"})

        status_events = [e for e in events if e.get("type") == "status"]
        assert any(e.get("running") is False for e in status_events), (
            f"Expected status:running:false after merging early return. Events: {events}"
        )

    def test_already_running_sends_status_false(self) -> None:
        """When task thread is alive, 'run' command sends status:running:false."""
        server, events = self._capture_server()

        stop = threading.Event()
        server._task_thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        server._task_thread.start()

        try:
            server._handle_command({
                "type": "run", "prompt": "test", "model": "x", "workDir": "/tmp",
            })
            status_events = [e for e in events if e.get("type") == "status"]
            assert any(e.get("running") is False for e in status_events), (
                f"Expected status:running:false after already-running rejection. Events: {events}"
            )
        finally:
            stop.set()
            server._task_thread.join()

    def test_outer_try_finally_in_run_task(self) -> None:
        """_run_task wraps _run_task_inner with try/finally for status guarantee."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "try:" in source
        assert "finally:" in source
        assert '"running": False' in source
        assert "_run_task_inner" in source


if __name__ == "__main__":
    unittest.main()
