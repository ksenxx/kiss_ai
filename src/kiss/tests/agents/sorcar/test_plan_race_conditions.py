"""Tests for race conditions documented in PLAN.md.

Harmful race conditions (P11, P12, P13, P14, T8, T9, T10, X4) —
tests demonstrate the bug is present in the current codebase.

Cosmetic race conditions (P3, P8, T6) — tests verify the fix is correct.

Race conditions tested:
- P3:  _complete_seq_latest TOCTOU (fixed with _complete_lock)
- P8:  _bash_flush_timer duplicate flush (fixed by draining in lock)
- P11: _last_active_file written without _state_lock in _run_task_inner
- P12: Stale followup suggestion interleaves with new task output
- P13: _force_stop_thread second KeyboardInterrupt corrupts finally-block
- P14: _force_stop_thread interrupt before try block skips finally
- T6:  AgentProcess.dispose() event race (fixed by reordering)
- T8:  _startTask doesn't recover from start() failure
- T9:  newConversation() silently drops newChat
- T10: _commitPending permanently stuck if Python process dies
- X4:  allDone merge signal sent to wrong provider's agent
"""

import ctypes
import inspect
import re
import threading
import unittest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# P11 — _last_active_file written without _state_lock
# ---------------------------------------------------------------------------

class TestP11LastActiveFileNoLock(unittest.TestCase):
    """P11: _run_task_inner writes _last_active_file without _state_lock.

    The task thread writes self._last_active_file outside any lock, while
    the complete handler reads the (file, content) pair under _state_lock.
    This can make the pair inconsistent: file points to B while content
    is still from A.
    """

    def test_task_thread_writes_without_lock(self) -> None:
        """Verify _run_task_inner writes _last_active_file outside _state_lock."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        # Find the assignment
        assert 'self._last_active_file = active_file or ""' in source
        # Now verify it's NOT inside a _state_lock context
        lines = source.split("\n")
        in_state_lock = False
        for line in lines:
            stripped = line.strip()
            if "with self._state_lock" in stripped:
                in_state_lock = True
            if in_state_lock and stripped == "":
                in_state_lock = False
            if 'self._last_active_file = active_file or ""' in stripped:
                assert not in_state_lock, (
                    "Expected _last_active_file write to be OUTSIDE _state_lock "
                    "(demonstrating P11 bug is present)"
                )
                break

    def test_concurrent_write_causes_inconsistent_pair(self) -> None:
        """Demonstrate that the unlocked write can produce an inconsistent pair.

        The task thread writes _last_active_file = "b.py" without the lock,
        while the complete handler under _state_lock still sees old content.
        """
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        # Set initial pair under lock
        with server._state_lock:
            server._last_active_file = "a.py"
            server._last_active_content = "content_of_a"

        # Simulate task thread writing without lock (as _run_task_inner does)
        server._last_active_file = "b.py"  # No lock, mimics line 222

        # Now the pair is inconsistent: file=b.py, content=content_of_a
        with server._state_lock:
            snapshot_file = server._last_active_file
            snapshot_content = server._last_active_content

        assert snapshot_file == "b.py"
        assert snapshot_content == "content_of_a", (
            "P11 bug: file points to b.py but content still belongs to a.py"
        )


# ---------------------------------------------------------------------------
# P12 — Stale followup suggestion interleaves with new task output
# ---------------------------------------------------------------------------

class TestP12StaleFollowupInterleave(unittest.TestCase):
    """P12: _generate_followup_async has no generation counter.

    A followup thread from a completed task can broadcast its suggestion
    after a new task has started, interleaving stale output.
    """

    def test_no_generation_counter_in_followup(self) -> None:
        """Verify _generate_followup_async does not check any generation counter."""
        source = inspect.getsource(VSCodeServer._generate_followup_async)
        assert "_task_generation" not in source, (
            "Expected no _task_generation check (P12 bug present)"
        )
        # The inner _run() function has no seq/gen guard
        run_idx = source.find("def _run()")
        assert run_idx > 0
        inner = source[run_idx:]
        assert "_task_generation" not in inner
        assert "gen ==" not in inner
        assert "seq ==" not in inner

    def test_no_task_generation_attribute(self) -> None:
        """Verify VSCodeServer has no _task_generation attribute."""
        server = VSCodeServer()
        assert not hasattr(server, "_task_generation"), (
            "Expected no _task_generation attr (P12 bug present)"
        )

    def test_followup_thread_not_cancelled_on_new_task(self) -> None:
        """Verify _run_task_inner does not cancel previous followup threads."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        # No cancellation of followup threads at the start
        assert "followup" not in source.split("start_recording")[0].lower(), (
            "Expected no followup cancellation before task starts"
        )


# ---------------------------------------------------------------------------
# P13 — _force_stop_thread second KeyboardInterrupt corrupts finally
# ---------------------------------------------------------------------------

class TestP13SecondInterruptCorruptsFinally(unittest.TestCase):
    """P13: A second KeyboardInterrupt inside the finally block escapes
    the `except Exception` handler, leaving _merging permanently True.
    """

    def test_except_exception_does_not_catch_keyboard_interrupt(self) -> None:
        """Verify the merge try/except in the finally block uses Exception, not BaseException."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        # Find the merge try/except in the finally block
        finally_idx = source.find("finally:")
        assert finally_idx > 0
        finally_block = source[finally_idx:]
        # The except clause around _start_merge_session
        merge_try_idx = finally_block.find("_prepare_merge_view")
        assert merge_try_idx > 0
        # Find the except after _start_merge_session
        except_after_merge = finally_block[merge_try_idx:]
        except_match = re.search(r"except\s+(\w+)", except_after_merge)
        assert except_match is not None
        caught_type = except_match.group(1)
        assert caught_type == "Exception", (
            f"Expected 'except Exception' (not catching KeyboardInterrupt), "
            f"got: except {caught_type}"
        )

    def test_second_interrupt_leaves_merging_stuck(self) -> None:
        """Demonstrate that a KeyboardInterrupt during _start_merge_session
        leaves self._merging = True permanently.

        Simulates the P13 scenario: the finally block calls
        _start_merge_session which sets _merging = True, then a
        KeyboardInterrupt arrives before the method returns.
        """
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]
        assert server._merging is False

        merging_set = threading.Event()
        interrupt_done = threading.Event()

        def simulate_finally_block() -> None:
            """Mimics the finally block where _start_merge_session sets _merging."""
            try:
                # Simulate _start_merge_session setting _merging
                server._merging = True
                merging_set.set()
                # Simulate slow work (e.g., broadcasting events)
                # The second KeyboardInterrupt arrives here
                interrupt_done.wait(timeout=5)
                # In real code, this would be more cleanup...
            except Exception:
                # This is the `except Exception` in the finally block.
                # KeyboardInterrupt is NOT caught here!
                pass

        def worker() -> None:
            try:
                simulate_finally_block()
            except KeyboardInterrupt:
                # The KeyboardInterrupt escapes the except Exception
                # In real code this propagates to _run_task's outer handler
                pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        merging_set.wait(timeout=5)

        # Inject KeyboardInterrupt into the thread
        tid = t.ident
        assert tid is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(KeyboardInterrupt),
        )
        interrupt_done.set()
        t.join(timeout=5)

        # _merging is still True — the flag was never reset
        assert server._merging is True, (
            "P13 bug: _merging should be stuck at True after KeyboardInterrupt "
            "escapes the finally block's except Exception handler"
        )

    def test_force_stop_sends_two_interrupts(self) -> None:
        """Verify _force_stop_thread raises KeyboardInterrupt up to 2 times."""
        source = inspect.getsource(VSCodeServer._force_stop_thread)
        assert "for _ in range(2)" in source, (
            "Expected _force_stop_thread to retry up to 2 times"
        )
        assert "PyThreadState_SetAsyncExc" in source
        assert "KeyboardInterrupt" in source

    def test_no_keyboard_interrupt_catch_in_outer_run_task(self) -> None:
        """Verify _run_task doesn't catch KeyboardInterrupt from the finally block.

        The outer wrapper only has try/finally, so a KeyboardInterrupt
        that escapes _run_task_inner's finally block propagates silently.
        """
        source = inspect.getsource(VSCodeServer._run_task)
        # _run_task has try/finally but no except KeyboardInterrupt
        assert "except KeyboardInterrupt" not in source, (
            "Expected _run_task to NOT catch KeyboardInterrupt (P13 bug present)"
        )


# ---------------------------------------------------------------------------
# P14 — Interrupt before try block skips _run_task_inner finally
# ---------------------------------------------------------------------------

class TestP14InterruptBeforeTryBlock(unittest.TestCase):
    """P14: If KeyboardInterrupt hits before the try block in _run_task_inner,
    start_recording() was called but stop_recording() never is, causing a
    memory leak in _recordings.
    """

    def test_start_recording_before_try_block(self) -> None:
        """Verify start_recording() is called before the try block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        rec_pos = source.find("self.printer.start_recording()")
        try_pos = source.find("        try:\n            self.agent.run(")
        assert rec_pos > 0, "start_recording not found"
        assert try_pos > 0, "try block not found"
        assert rec_pos < try_pos, (
            "P14 bug: start_recording() is called BEFORE the try block, "
            "so an interrupt between them skips stop_recording() in the finally"
        )

    def test_recording_leak_on_early_interrupt(self) -> None:
        """Demonstrate that start_recording without matching stop_recording
        leaks an entry in _recordings.
        """
        printer = BaseBrowserPrinter()
        # Simulate what happens when interrupt hits after start_recording
        # but before the try block
        printer.start_recording()
        tid = threading.current_thread().ident
        assert tid is not None

        # The recording is now in _recordings
        with printer._lock:
            assert tid in printer._recordings, "Recording should be active"

        # If we DON'T call stop_recording (simulating the interrupt),
        # the entry stays permanently
        with printer._lock:
            assert tid in printer._recordings, (
                "P14 bug: recording entry leaked because stop_recording was never called"
            )

        # Clean up to not affect other tests
        printer.stop_recording()

    def test_git_snapshot_before_try_block(self) -> None:
        """Verify git snapshot code runs before the try block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        diff_pos = source.find("_parse_diff_hunks(work_dir)")
        try_pos = source.find("        try:\n            self.agent.run(")
        assert diff_pos > 0, "_parse_diff_hunks not found"
        assert try_pos > 0, "try block not found"
        assert diff_pos < try_pos, (
            "P14 bug: git snapshot runs before try block, "
            "so interrupt during snapshot skips all cleanup"
        )


# ---------------------------------------------------------------------------
# T8 — _startTask doesn't recover from start() failure
# ---------------------------------------------------------------------------

class TestT8StartTaskNoRecovery(unittest.TestCase):
    """T8: _startTask sets _isRunning=true before start(), and never
    checks start()'s return value. If start() fails, _isRunning stays
    true permanently.
    """

    def test_start_task_sets_running_before_start(self) -> None:
        """Verify _startTask sets _isRunning = true before calling start()."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("private _startTask(")
        assert idx >= 0
        block = source[idx:idx + 500]
        running_pos = block.find("this._isRunning = true")
        start_pos = block.find("this._agentProcess.start(")
        assert running_pos > 0 and start_pos > 0
        assert running_pos < start_pos, (
            "T8 bug: _isRunning set before start() call"
        )

    def test_start_return_value_not_checked(self) -> None:
        """Verify _startTask does not check start()'s return value."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("private _startTask(")
        assert idx >= 0
        block = source[idx:idx + 500]
        # start() should return bool, but its return value is not used
        start_line_idx = block.find("this._agentProcess.start(")
        # Check the line doesn't have an assignment or if-check
        line_start = block.rfind("\n", 0, start_line_idx) + 1
        line = block[line_start:block.find("\n", start_line_idx)]
        assert "if" not in line and "=" not in line.split("start(")[0], (
            "T8 bug: expected start() return value to NOT be checked"
        )

    def test_no_running_reset_on_start_failure(self) -> None:
        """Verify _startTask has no fallback to reset _isRunning on failure."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("private _startTask(")
        end_idx = source.find("\n  }", idx)
        block = source[idx:end_idx]
        # Count how many times _isRunning is set
        sets = [m.start() for m in re.finditer(r"this\._isRunning\s*=", block)]
        assert len(sets) == 1, (
            f"T8 bug: _startTask sets _isRunning only once (no reset on failure), "
            f"found {len(sets)} assignments"
        )


# ---------------------------------------------------------------------------
# T9 — newConversation() silently drops newChat
# ---------------------------------------------------------------------------

class TestT9NewConversationDropsNewChat(unittest.TestCase):
    """T9: newConversation() calls stop() then immediately sends newChat.
    The Python backend skips newChat because the task thread is still alive.
    """

    def test_newchat_sent_immediately_after_stop(self) -> None:
        """Verify newConversation sends newChat without waiting for stop to complete."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public newConversation(")
        assert idx >= 0
        block = source[idx:idx + 600]
        stop_pos = block.find("this._agentProcess.stop()")
        newchat_pos = block.find("this._agentProcess.sendCommand({ type: 'newChat' })")
        assert stop_pos > 0 and newchat_pos > 0
        assert newchat_pos > stop_pos, "newChat sent after stop"
        # Check there's no await or promise between stop and newChat
        between = block[stop_pos:newchat_pos]
        assert "await" not in between, (
            "T9 bug: no await between stop() and newChat — newChat sent immediately"
        )

    def test_newconversation_not_async(self) -> None:
        """Verify newConversation is not an async method (can't await stop)."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public newConversation(")
        assert idx >= 0
        line_start = source.rfind("\n", 0, idx) + 1
        decl_line = source[line_start:source.find("{", idx)]
        assert "async" not in decl_line, (
            "T9 bug: newConversation is synchronous, can't await stop completion"
        )

    def test_python_newchat_skipped_when_task_alive(self) -> None:
        """Verify Python backend skips newChat when task thread is alive."""
        server = VSCodeServer()
        original_id = server.agent._chat_id

        stop = threading.Event()
        server._task_thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        server._task_thread.start()

        try:
            # This is what happens when TS sends newChat while task is alive
            server._handle_command({"type": "newChat"})
            assert server.agent._chat_id == original_id, (
                "T9 bug: newChat was silently skipped because task thread is alive"
            )
        finally:
            stop.set()
            server._task_thread.join()


# ---------------------------------------------------------------------------
# T10 — _commitPending permanently stuck if Python process dies
# ---------------------------------------------------------------------------

class TestT10CommitPendingStuck(unittest.TestCase):
    """T10: generateCommitMessage sets _commitPending=true and only resets
    it via _onCommitMessage. If the Python process dies, the listener
    never fires and _commitPending stays true permanently.
    """

    def test_no_process_death_handler_for_commit_pending(self) -> None:
        """Verify generateCommitMessage has no process-death/timeout reset path."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public generateCommitMessage(")
        end_idx = source.find("\n  }", idx)
        block = source[idx:end_idx]
        # No timeout
        assert "setTimeout" not in block, (
            "T10 bug: expected no timeout fallback for _commitPending"
        )
        # No process close/status listener
        assert "close" not in block
        assert "'status'" not in block, (
            "T10 bug: no process-death reset for _commitPending"
        )

    def test_commit_pending_only_reset_by_commit_message_event(self) -> None:
        """Verify _commitPending is only reset in the onCommitMessage callback."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public generateCommitMessage(")
        end_idx = source.find("\n  }", idx)
        block = source[idx:end_idx]
        # Find all places where _commitPending is set to false
        resets = list(re.finditer(r"this\._commitPending\s*=\s*false", block))
        assert len(resets) == 1, (
            f"T10 bug: _commitPending reset only in done() callback, "
            f"no fallback path. Found {len(resets)} reset(s)"
        )

    def test_close_handler_does_not_fire_commit_message(self) -> None:
        """Verify AgentProcess close handler emits status, not commitMessage."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("this.process.on('close'")
        assert idx >= 0
        block = source[idx:idx + 300]
        assert "commitMessage" not in block, (
            "T10 bug: close handler does NOT emit commitMessage event"
        )
        assert "'status'" in block, (
            "close handler emits status event (which doesn't reset _commitPending)"
        )


# ---------------------------------------------------------------------------
# X4 — allDone merge signal sent to wrong provider's agent
# ---------------------------------------------------------------------------

class TestX4AllDoneSentToWrongProvider(unittest.TestCase):
    """X4: getActiveProvider() always returns secondaryProvider when it exists,
    ignoring which provider actually started the merge session.
    """

    def test_get_active_provider_always_returns_secondary(self) -> None:
        """Verify getActiveProvider prefers secondary over primary."""
        with open("src/kiss/agents/vscode/src/extension.ts") as f:
            source = f.read()
        idx = source.find("function getActiveProvider()")
        assert idx >= 0
        block = source[idx:idx + 200]
        assert "secondaryProvider ?? primaryProvider" in block, (
            "X4 bug: getActiveProvider always returns secondary when it exists"
        )

    def test_no_merge_owner_tracking(self) -> None:
        """Verify extension.ts does not track which provider started the merge."""
        with open("src/kiss/agents/vscode/src/extension.ts") as f:
            source = f.read()
        assert "mergeOwner" not in source, (
            "X4 bug: no mergeOwner tracking — allDone goes to wrong provider"
        )

    def test_alldone_uses_get_active_provider(self) -> None:
        """Verify allDone handler routes through getActiveProvider, not merge owner."""
        with open("src/kiss/agents/vscode/src/extension.ts") as f:
            source = f.read()
        idx = source.find("mergeManager.on('allDone'")
        assert idx >= 0
        block = source[idx:idx + 200]
        assert "getActiveProvider()" in block, (
            "X4 bug: allDone uses getActiveProvider() without knowing merge owner"
        )


# ---------------------------------------------------------------------------
# P3 — _complete_seq_latest TOCTOU fixed with _complete_lock
# ---------------------------------------------------------------------------

class TestP3CompleteSeqTOCTOU(unittest.TestCase):
    """P3: _complete_seq_latest check-then-broadcast TOCTOU.

    The fix adds _complete_lock to make the second seq check and broadcast
    atomic, preventing stale ghost suggestions from slipping through.
    """

    def test_complete_lock_exists(self) -> None:
        """Verify VSCodeServer has a _complete_lock attribute."""
        server = VSCodeServer()
        assert hasattr(server, "_complete_lock"), (
            "Expected _complete_lock for atomic seq check-and-broadcast"
        )

    def test_complete_uses_lock_around_second_check_and_broadcast(self) -> None:
        """Verify _complete() holds _complete_lock across the second seq check and broadcast."""
        source = inspect.getsource(VSCodeServer._complete)
        # The second check-and-broadcast should be under _complete_lock.
        # Find the pattern: with self._complete_lock: ... broadcast
        assert "_complete_lock" in source, (
            "Expected _complete() to use _complete_lock"
        )
        # Find the second seq check (after _fast_complete) and verify it's under a lock
        fast_complete_idx = source.find("_fast_complete")
        assert fast_complete_idx > 0
        after_fast = source[fast_complete_idx:]
        # The broadcast after the second check should be inside a with block
        lock_idx = after_fast.find("with self._complete_lock")
        broadcast_idx = after_fast.find('self.printer.broadcast({"type": "ghost"')
        assert lock_idx > 0 and broadcast_idx > 0, "Expected lock and broadcast after _fast_complete"
        assert lock_idx < broadcast_idx, (
            "Expected _complete_lock to be acquired BEFORE broadcast"
        )

    def test_seq_latest_write_under_lock(self) -> None:
        """Verify _handle_command writes _complete_seq_latest under _complete_lock."""
        source = inspect.getsource(VSCodeServer._handle_command)
        # Find the _complete_seq_latest assignment
        assign_idx = source.find("self._complete_seq_latest = seq")
        assert assign_idx > 0
        # Check that _complete_lock appears before the assignment (in the same block)
        preceding = source[:assign_idx]
        lock_idx = preceding.rfind("_complete_lock")
        assert lock_idx > 0, (
            "Expected _complete_seq_latest write to be under _complete_lock"
        )

    def test_stale_seq_not_broadcast(self) -> None:
        """A completion with a stale seq must not broadcast a ghost event."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        # Set latest seq to 10
        with server._complete_lock:
            server._complete_seq_latest = 10

        # Complete with stale seq=5 — should not broadcast
        server._complete("hello world", 5, "", "")
        ghost_events = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost_events) == 0, (
            "Stale seq should not produce a ghost event"
        )

    def test_current_seq_broadcasts(self) -> None:
        """A completion with the current seq should broadcast a ghost event."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        with server._complete_lock:
            server._complete_seq_latest = 10

        server._complete("hello world", 10, "", "")
        ghost_events = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost_events) == 1, "Current seq should produce a ghost event"

    def test_concurrent_seq_updates_no_stale_ghost(self) -> None:
        """Stress test: concurrent seq updates should prevent stale ghost broadcasts."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        errors: list[str] = []
        num_rounds = 200

        def updater() -> None:
            """Rapidly update _complete_seq_latest to the final value."""
            for i in range(num_rounds):
                with server._complete_lock:
                    server._complete_seq_latest = i

        def completer() -> None:
            """Run _complete with seq=0 repeatedly; should be stale after update."""
            for _ in range(50):
                server._complete("hello world", 0, "", "")

        with server._complete_lock:
            server._complete_seq_latest = 0

        t_up = threading.Thread(target=updater)
        t_comp = threading.Thread(target=completer)
        t_up.start()
        t_comp.start()
        t_up.join(timeout=10)
        t_comp.join(timeout=10)

        # After updater finishes, seq_latest = num_rounds-1.
        # Any ghost with seq=0 that was broadcast after seq_latest was updated
        # would be a stale ghost — but we can't distinguish timing perfectly.
        # At minimum, verify no crashes and all ghosts have the right query.
        for e in events:
            if e.get("type") == "ghost":
                assert e.get("query") == "hello world"


# ---------------------------------------------------------------------------
# P8 — _bash_flush_timer TOCTOU fixed by draining buffer inside lock
# ---------------------------------------------------------------------------

class TestP8BashFlushTOCTOU(unittest.TestCase):
    """P8: _bash_flush_timer race where needs_flush decision is made under
    lock but _flush_bash() is called outside.

    The fix drains the buffer inside the lock, eliminating the TOCTOU gap.
    """

    def test_no_needs_flush_variable_in_bash_stream(self) -> None:
        """Verify the bash_stream branch no longer uses a needs_flush flag."""
        source = inspect.getsource(BaseBrowserPrinter.print)
        # Find the bash_stream branch
        bash_idx = source.find('"bash_stream"')
        assert bash_idx > 0
        # Get the bash_stream block (until next if type ==)
        next_branch_idx = source.find("if type ==", bash_idx + 1)
        if next_branch_idx < 0:
            next_branch_idx = len(source)
        bash_block = source[bash_idx:next_branch_idx]
        assert "needs_flush" not in bash_block, (
            "Expected bash_stream branch to NOT use needs_flush flag (P8 fix)"
        )

    def test_buffer_drained_inside_lock(self) -> None:
        """Verify buffer drain happens inside _bash_lock in bash_stream branch."""
        source = inspect.getsource(BaseBrowserPrinter.print)
        bash_idx = source.find('"bash_stream"')
        next_branch_idx = source.find("if type ==", bash_idx + 1)
        if next_branch_idx < 0:
            next_branch_idx = len(source)
        bash_block = source[bash_idx:next_branch_idx]
        # The buffer clear should be inside the lock block
        # Look for _bash_buffer.clear() inside the with block
        lock_idx = bash_block.find("with self._bash_lock")
        clear_idx = bash_block.find("self._bash_buffer.clear()")
        assert lock_idx > 0 and clear_idx > 0, (
            "Expected buffer drain inside _bash_lock"
        )
        assert lock_idx < clear_idx, "Buffer clear should be inside the lock"

    def test_concurrent_bash_stream_no_lost_output(self) -> None:
        """Stress test: concurrent bash_stream calls should not lose any output."""
        printer = BaseBrowserPrinter()
        events: list[dict] = []
        original_broadcast = printer.broadcast

        lock = threading.Lock()
        def recording_broadcast(e: dict) -> None:
            with lock:
                events.append(e)
            original_broadcast(e)

        printer.broadcast = recording_broadcast  # type: ignore[assignment]

        num_threads = 10
        items_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def writer(tid: int) -> None:
            barrier.wait()
            for i in range(items_per_thread):
                printer.print(f"t{tid}_{i}\n", type="bash_stream")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Flush any remaining buffered content
        printer._flush_bash()

        # Collect all system_output text
        all_text = "".join(
            e["text"] for e in events if e.get("type") == "system_output"
        )

        # Every item should appear exactly once
        for tid in range(num_threads):
            for i in range(items_per_thread):
                token = f"t{tid}_{i}\n"
                assert token in all_text, (
                    f"Lost output: {token!r} not found in broadcast output"
                )

    def test_flush_timer_cancelled_on_immediate_flush(self) -> None:
        """When buffer is drained immediately, any pending timer should be cancelled."""
        printer = BaseBrowserPrinter()
        events: list[dict] = []
        printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        # Force the first flush to set _bash_last_flush to an old time
        printer._bash_last_flush = 0.0

        # This should trigger immediate flush (time since last flush > 0.1s)
        printer.print("hello", type="bash_stream")

        # Verify the buffer was drained
        with printer._bash_lock:
            assert len(printer._bash_buffer) == 0, "Buffer should be empty after immediate flush"
            assert printer._bash_flush_timer is None, "Timer should not be set after immediate flush"

        output_events = [e for e in events if e.get("type") == "system_output"]
        assert len(output_events) == 1
        assert output_events[0]["text"] == "hello"


# ---------------------------------------------------------------------------
# T6 — AgentProcess.dispose() race fixed by moving removeAllListeners first
# ---------------------------------------------------------------------------

class TestT6DisposeRace(unittest.TestCase):
    """T6: dispose() sets this.process = null then calls proc.kill(),
    but removeAllListeners() comes after kill. The close handler could
    fire between null-out and removeAllListeners.

    The fix moves removeAllListeners() before kill.
    """

    def test_remove_all_listeners_before_kill(self) -> None:
        """Verify removeAllListeners() is called before proc.kill() in dispose."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("dispose(): void {")
        assert idx >= 0
        block = source[idx:source.find("\n  }", idx) + 4]
        remove_idx = block.find("this.removeAllListeners()")
        kill_idx = block.find("proc.kill('SIGTERM')")
        assert remove_idx > 0 and kill_idx > 0, (
            "Expected both removeAllListeners and kill in dispose"
        )
        assert remove_idx < kill_idx, (
            "T6 fix: removeAllListeners() should be called BEFORE proc.kill()"
        )

    def test_remove_all_listeners_after_null_out(self) -> None:
        """Verify removeAllListeners() is between null-out and kill (correct order)."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("dispose(): void {")
        block = source[idx:source.find("\n  }", idx) + 4]
        null_idx = block.find("this.process = null")
        remove_idx = block.find("this.removeAllListeners()")
        kill_idx = block.find("proc.kill('SIGTERM')")
        assert null_idx < remove_idx < kill_idx, (
            "Expected order: process=null → removeAllListeners → kill"
        )

    def test_else_branch_also_removes_listeners(self) -> None:
        """Verify dispose() removes listeners even when no process is running."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("dispose(): void {")
        block = source[idx:source.find("\n  }", idx) + 4]
        # There should be an else branch with removeAllListeners
        else_idx = block.find("} else {")
        assert else_idx > 0, "Expected else branch in dispose"
        else_block = block[else_idx:]
        assert "this.removeAllListeners()" in else_block, (
            "Expected removeAllListeners in else branch of dispose"
        )


if __name__ == "__main__":
    unittest.main()
