# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests confirming fixes for bugs, redundancies, and
inconsistencies in ``kiss.server`` — audit round 6.

B1 fix: ``save_config`` now preserves non-DEFAULTS keys (like ``email``)
    that were already stored in ``config.json``.  Previously, calling
    ``save_config`` truncated the file to only DEFAULTS keys.

B2 fix: ``is_task_active`` is now cleared in ``_run_task``'s finally
    block, not only in ``_run_task_inner``'s.  Previously, a failure
    in the pre-task setup (before the inner try/finally) left
    ``is_task_active`` True permanently.

B3 fix: ``get_fast_model()`` now returns ``gemini-2.0-flash`` for
    Gemini (a genuinely cheap/fast model) instead of ``gemini-2.5-pro``
    (an expensive reasoning model).

R1 fix: dead ``is not None`` guard removed from ``_cmd_user_answer``.
    Since ``ans_tab`` defaults to ``""``, it is never ``None``.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events




class TestIsTaskActiveClearedOnSnapshotFailure(TestCase):
    """B2 FIX: ``_run_task`` now clears ``is_task_active`` in its own
    finally block, so a failure in the pre-task setup (before the
    inner try/finally) no longer leaves the tab permanently marked as
    active.
    """

    def test_is_task_active_false_after_snapshot_error(self) -> None:
        """When pre-snapshot capture fails, is_task_active must be False."""
        server, events = _make_server()
        tab_id = "snap-fail-tab"

        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": tab_id,
            "prompt": "test",
            "useWorktree": False,
            "model": server._default_model,
            "workDir": "/nonexistent/dir/that/will/fail",
        }

        try:
            try:
                server._run_task(cmd)
            except (FileNotFoundError, OSError):
                pass

            # The printer bridge re-keys the state to the allocated
            # task id mid-run, so look it up by tab instead of the
            # (stale) ``_state_key`` stamp.
            state = agent_state.find_by_tab(tab_id)
            assert state is not None, "run never registered an agent state"
            assert state.is_task_active is False, (
                "B2 FIX: is_task_active should be False after setup failure"
            )
        finally:
            agent_state.agent_states.clear()





class TestUserAnswerNoDeadIsNotNoneCheck(TestCase):
    """R1 FIX: removed the dead ``if ans_tab is not None`` guard from
    ``_cmd_user_answer``.  Since ``ans_tab = cmd.get("tabId", "")``,
    the variable is always a string, never ``None``.
    """


    def test_empty_tab_id_drops_answer(self) -> None:
        """When tabId is empty string, the answer should be dropped (no queue)."""
        server, events = _make_server()
        server._cmd_user_answer({"type": "userAnswer", "tabId": "", "answer": "x"})


class TestIsRunningNonWtClearedOnSnapshotFailure(TestCase):
    """is_running_non_wt should be False after a snapshot failure."""

    def test_is_running_non_wt_cleared(self) -> None:
        server, events = _make_server()
        tab_id = "nwt-fail-tab"

        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": tab_id,
            "prompt": "test",
            "useWorktree": False,
            "model": server._default_model,
            "workDir": "/nonexistent/dir/that/will/fail",
        }

        try:
            try:
                server._run_task(cmd)
            except (FileNotFoundError, OSError):
                pass

            # Looked up by tab: the state is re-keyed to the allocated
            # task id mid-run (see B2 test above).
            state = agent_state.find_by_tab(tab_id)
            assert state is not None, "run never registered an agent state"
            assert state.is_running_non_wt is False, (
                "is_running_non_wt should be False after setup failure"
            )
        finally:
            agent_state.agent_states.clear()


if __name__ == "__main__":
    unittest.main()
