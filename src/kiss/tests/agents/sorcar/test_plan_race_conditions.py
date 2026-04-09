"""Tests for race conditions documented in PLAN.md.

Tests verify fixes for:
- P3:  _complete_seq_latest TOCTOU (fixed with _complete_lock)
- P8:  _bash_flush_timer duplicate flush (fixed by draining in lock)
- P11: _last_active_file written under _state_lock in _run_task_inner
- P12: Stale followup suppressed by _task_generation counter
- P13: Entire finally block wrapped in try/except BaseException
- P14: start_recording inside try block (stop_recording always runs)
- T6:  AgentProcess.dispose() event race (fixed by reordering)
- T8:  _startTask checks start() return value and resets _isRunning
- T9:  newConversation() queues newChat for status:running:false
- T10: _commitPending reset on status:running:false
- X4:  allDone merge signal uses mergeOwner tracking
"""

import inspect
import re
import unittest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# P11 — _last_active_file written under _state_lock
# ---------------------------------------------------------------------------

class TestP11LastActiveFileWithLock(unittest.TestCase):
    """P11 fix: _run_task_inner writes _last_active_file under _state_lock."""

    def test_task_thread_writes_with_lock(self) -> None:
        """Verify _run_task_inner writes _last_active_file inside _state_lock."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert 'self._last_active_file = active_file or ""' in source
        # Find the assignment and verify it's inside a _state_lock context
        lines = source.split("\n")
        in_state_lock = False
        state_lock_indent = 0
        found_inside_lock = False
        for line in lines:
            stripped = line.strip()
            if "with self._state_lock" in stripped:
                in_state_lock = True
                state_lock_indent = len(line) - len(line.lstrip())
                continue
            if in_state_lock:
                current_indent = (
                    len(line) - len(line.lstrip()) if stripped else state_lock_indent + 1
                )
                if current_indent <= state_lock_indent and stripped:
                    in_state_lock = False
            if 'self._last_active_file = active_file or ""' in stripped and in_state_lock:
                found_inside_lock = True
                break
        assert found_inside_lock, (
            "P11 fix: _last_active_file write should be INSIDE _state_lock"
        )


# ---------------------------------------------------------------------------
# P12 — Stale followup suppressed by _task_generation counter
# ---------------------------------------------------------------------------

class TestP12StaleFollowupSuppressed(unittest.TestCase):
    """P12 fix: _generate_followup_async checks _task_generation counter."""

    def test_generation_counter_in_followup(self) -> None:
        """Verify _generate_followup_async checks _task_generation."""
        source = inspect.getsource(VSCodeServer._generate_followup_async)
        assert "_task_generation" in source, (
            "P12 fix: _generate_followup_async should check _task_generation"
        )
        assert "gen" in source, "P12 fix: should accept gen parameter"

    def test_generation_counter_incremented_in_run_task_inner(self) -> None:
        """Verify _run_task_inner increments _task_generation."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_task_generation" in source, (
            "P12 fix: _run_task_inner should increment _task_generation"
        )

    def test_followup_receives_gen_parameter(self) -> None:
        """Verify _generate_followup_async accepts gen parameter."""
        sig = inspect.signature(VSCodeServer._generate_followup_async)
        assert "gen" in sig.parameters, (
            "P12 fix: _generate_followup_async should accept gen parameter"
        )


# ---------------------------------------------------------------------------
# P13 — Entire finally block wrapped in try/except BaseException
# ---------------------------------------------------------------------------

class TestP13FinallyBlockProtected(unittest.TestCase):
    """P13 fix: The finally block catches BaseException, not just Exception."""

    def test_except_base_exception_in_finally(self) -> None:
        """Verify the finally block uses except BaseException for merge try."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        finally_idx = source.find("finally:")
        assert finally_idx > 0
        finally_block = source[finally_idx:]
        # Should have except BaseException somewhere in the finally block
        assert "except BaseException" in finally_block, (
            "P13 fix: finally block should catch BaseException"
        )

    def test_force_stop_sends_two_interrupts(self) -> None:
        """Verify _force_stop_thread raises KeyboardInterrupt up to 2 times."""
        source = inspect.getsource(VSCodeServer._force_stop_thread)
        assert "for _ in range(2)" in source
        assert "PyThreadState_SetAsyncExc" in source
        assert "KeyboardInterrupt" in source

    def test_outer_cleanup_catches_base_exception(self) -> None:
        """Verify the outermost cleanup try/except in finally catches BaseException."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        finally_idx = source.find("finally:")
        assert finally_idx > 0
        finally_block = source[finally_idx:]
        # Count occurrences of 'except BaseException'
        base_exception_count = finally_block.count("except BaseException")
        assert base_exception_count >= 2, (
            f"P13 fix: expected at least 2 'except BaseException' in finally "
            f"(one for merge, one for outer cleanup), got {base_exception_count}"
        )


# ---------------------------------------------------------------------------
# P14 — start_recording inside try block
# ---------------------------------------------------------------------------

class TestP14StartRecordingInsideTry(unittest.TestCase):
    """P14 fix: start_recording() is inside the try block so stop_recording()
    is guaranteed to run in the finally block.
    """

    def test_start_recording_after_try(self) -> None:
        """Verify start_recording() is called after the outer try statement."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        rec_pos = source.find("self.printer.start_recording(")
        # Find the outer try that has the finally with stop_recording
        try_positions = [m.start() for m in re.finditer(r"\btry\b:", source)]
        # start_recording should be AFTER a try: statement
        try_before = [tp for tp in try_positions if tp < rec_pos]
        assert try_before, (
            "P14 fix: start_recording() should be inside a try block"
        )

    def test_stop_recording_in_finally(self) -> None:
        """Verify stop_recording() is in the finally block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        finally_idx = source.find("finally:")
        assert finally_idx > 0
        finally_block = source[finally_idx:]
        assert "self.printer.stop_recording(" in finally_block, (
            "P14 fix: stop_recording should be in the finally block"
        )


# ---------------------------------------------------------------------------
# T8 — _startTask checks start() return value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P3 — _complete_seq_latest TOCTOU fixed with _complete_lock
# ---------------------------------------------------------------------------

class TestP3CompleteSeqTOCTOU(unittest.TestCase):
    """P3: _complete_seq_latest stale-request check.

    The fix uses _complete_lock to check seq before doing work,
    preventing stale requests from wasting computation.
    """

    def test_complete_checks_seq_under_lock(self) -> None:
        """Verify _complete() checks seq under _complete_lock before processing."""
        source = inspect.getsource(VSCodeServer._complete)
        assert "_complete_lock" in source
        lock_idx = source.find("with self._complete_lock")
        seq_check_idx = source.find("_complete_seq_latest", lock_idx)
        assert lock_idx > 0 and seq_check_idx > 0
        assert lock_idx < seq_check_idx

    def test_seq_latest_write_under_lock(self) -> None:
        """Verify _handle_command writes _complete_seq_latest under _complete_lock."""
        source = inspect.getsource(VSCodeServer._handle_command)
        assign_idx = source.find("self._complete_seq_latest = seq")
        assert assign_idx > 0
        preceding = source[:assign_idx]
        lock_idx = preceding.rfind("_complete_lock")
        assert lock_idx > 0


# ---------------------------------------------------------------------------
# P8 — _bash_flush_timer TOCTOU fixed by draining buffer inside lock
# ---------------------------------------------------------------------------

class TestP8BashFlushTOCTOU(unittest.TestCase):
    """P8: _bash_flush_timer race fixed by draining buffer inside lock."""

    def test_no_needs_flush_variable_in_bash_stream(self) -> None:
        """Verify the bash_stream branch no longer uses a needs_flush flag."""
        source = inspect.getsource(BaseBrowserPrinter.print)
        bash_idx = source.find('"bash_stream"')
        assert bash_idx > 0
        next_branch_idx = source.find("if type ==", bash_idx + 1)
        if next_branch_idx < 0:
            next_branch_idx = len(source)
        bash_block = source[bash_idx:next_branch_idx]
        assert "needs_flush" not in bash_block


# ---------------------------------------------------------------------------
# T6 — AgentProcess.dispose() race fixed by reordering
# ---------------------------------------------------------------------------

class TestT6DisposeRace(unittest.TestCase):
    """T6: dispose() calls removeAllListeners before kill."""

    def test_remove_all_listeners_before_kill(self) -> None:
        """Verify removeAllListeners() is called before proc.kill()."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("dispose(): void {")
        assert idx >= 0
        block = source[idx:source.find("\n  }", idx) + 4]
        remove_idx = block.find("this.removeAllListeners()")
        kill_idx = block.find("proc.kill('SIGTERM')")
        assert remove_idx > 0 and kill_idx > 0
        assert remove_idx < kill_idx

    def test_else_branch_also_removes_listeners(self) -> None:
        """Verify dispose() removes listeners even when no process is running."""
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            source = f.read()
        idx = source.find("dispose(): void {")
        block = source[idx:source.find("\n  }", idx) + 4]
        else_idx = block.find("} else {")
        assert else_idx > 0
        else_block = block[else_idx:]
        assert "this.removeAllListeners()" in else_block


if __name__ == "__main__":
    unittest.main()
