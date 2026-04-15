"""Tests for race conditions documented in PLAN.md.

Tests verify fixes for:
- P3:  _complete_seq_latest stale-request guard (lockless — single writer/reader)
- P8:  _bash_flush_timer duplicate flush (fixed by draining in lock)
- P11: _last_active_file written under _state_lock in _run_task_inner
- P12: Stale followup: task_generation removed (per-task processes)
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
# P11 → S8 — _last_active_file no longer written in _run_task_inner
# ---------------------------------------------------------------------------

class TestP11LastActiveFileWithLock(unittest.TestCase):
    """S8 fix: _run_task_inner no longer writes _last_active_file.

    The active file cache is now only updated via the ``complete``
    command handler (on the main thread) so concurrent task threads
    never stomp on each other's autocomplete context.
    """

    def test_run_task_inner_does_not_write_last_active_file(self) -> None:
        """Verify _run_task_inner does NOT write _last_active_file."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert 'self._last_active_file' not in source, (
            "S8 fix: _run_task_inner must not write _last_active_file"
        )


# ---------------------------------------------------------------------------
# P12 — Stale followup: no longer needed with per-task processes
# ---------------------------------------------------------------------------

class TestP12StaleFollowupNotNeeded(unittest.TestCase):
    """P12: task_generation removed — per-task processes prevent stale followups."""

    def test_task_generation_removed_from_followup(self) -> None:
        """Verify task_generation is no longer in _generate_followup_async."""
        source = inspect.getsource(VSCodeServer._generate_followup_async)
        assert "task_generation" not in source, (
            "task_generation should be removed (per-task processes prevent stale followups)"
        )

    def test_task_generation_removed_from_run_task_inner(self) -> None:
        """Verify task_generation is no longer in _run_task_inner."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "task_generation" not in source, (
            "task_generation should be removed from _run_task_inner"
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
# P3 — _complete_seq_latest stale-request guard (lockless)
# ---------------------------------------------------------------------------

class TestP3CompleteSeqTOCTOU(unittest.TestCase):
    """P3: _complete_seq_latest stale-request check.

    Single writer (main thread) and single reader (worker thread) make a
    lock redundant.  The seq guard itself still prevents stale requests.
    """

    def test_complete_checks_seq_before_processing(self) -> None:
        """Verify _complete() checks seq against _complete_seq_latest."""
        source = inspect.getsource(VSCodeServer._complete)
        assert "_complete_seq_latest" in source

    def test_seq_latest_written_in_handle_command(self) -> None:
        """Verify _handle_command writes _complete_seq_latest."""
        source = inspect.getsource(VSCodeServer._handle_command)
        assert "self._complete_seq_latest = seq" in source


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
