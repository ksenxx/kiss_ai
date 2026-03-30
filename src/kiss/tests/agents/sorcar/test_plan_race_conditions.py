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

import inspect
import re
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
    """T8: Callers set _isRunning=true before _startTask, which calls
    start() without checking its return value. If start() fails,
    _isRunning stays true permanently because _startTask has no
    error handling.
    """

    def test_start_task_sets_running_before_start(self) -> None:
        """Verify _isRunning is set to true in the caller before _startTask."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        # _isRunning is now set in 'submit' handler before _startTask is called
        submit_idx = source.find("case 'submit':")
        assert submit_idx >= 0
        start_task_idx = source.find("this._startTask(", submit_idx)
        assert start_task_idx > 0
        between = source[submit_idx:start_task_idx]
        assert "this._isRunning = true" in between, (
            "T8 bug: _isRunning set in caller before _startTask"
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
        """Verify _startTask has no error handling to reset _isRunning on failure."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("private _startTask(")
        end_idx = source.find("\n  }", idx)
        block = source[idx:end_idx]
        # _startTask does not set _isRunning at all (callers set it),
        # so there is no try/catch to reset it if start() fails
        assert "try" not in block, (
            "T8 bug: _startTask has no error handling for start() failure"
        )
        sets = [m.start() for m in re.finditer(r"this\._isRunning\s*=", block)]
        assert len(sets) == 0, (
            f"T8 bug: _startTask does not manage _isRunning (callers do), "
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


# ---------------------------------------------------------------------------
# T10 — _commitPending permanently stuck if Python process dies
# ---------------------------------------------------------------------------

class TestT10CommitPendingStuck(unittest.TestCase):
    """T10: generateCommitMessage sets _commitPending=true.

    Originally only reset via _onCommitMessage (if Python dies, stuck forever).
    Now has a 30s setTimeout fallback, but still no explicit process-death handler.
    """

    def test_commit_pending_has_timeout_fallback(self) -> None:
        """Verify generateCommitMessage has a timeout fallback to reset _commitPending."""
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        idx = source.find("public generateCommitMessage(")
        end_idx = source.find("\n  }", idx)
        block = source[idx:end_idx]
        # T10 fix: setTimeout now provides a fallback to reset _commitPending
        assert "setTimeout" in block, (
            "T10 fix: expected setTimeout fallback for _commitPending"
        )
        assert "clearTimeout" in block, (
            "T10 fix: expected clearTimeout to cancel timer on success"
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

    def test_complete_uses_lock_around_second_check_and_broadcast(self) -> None:
        """Verify _complete() holds _complete_lock across the second seq check and broadcast."""
        source = inspect.getsource(VSCodeServer._complete)
        # The second check-and-broadcast should be under _complete_lock.
        assert "_complete_lock" in source, (
            "Expected _complete() to use _complete_lock"
        )
        # Find the completion logic (prefix match / active file) and verify
        # the broadcast after it is under a lock
        match_idx = source.find("_prefix_match_task")
        assert match_idx > 0, "Expected _prefix_match_task in _complete"
        after_match = source[match_idx:]
        # The broadcast after the completion logic should be inside a with block
        lock_idx = after_match.find("with self._complete_lock")
        broadcast_idx = after_match.find('self.printer.broadcast({"type": "ghost"')
        assert lock_idx > 0 and broadcast_idx > 0, (
            "Expected lock and broadcast after completion logic"
        )
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
