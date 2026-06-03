# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduction + fix for "Agent Failed Abruptly" sentinel sticking.

Symptom: occasional task_history rows survive with
``result == "Agent Failed Abruptly"`` (the initial sentinel written by
``_add_task``) even though the agent thread reached the LLM streaming
loop and emitted many events.  Inspection of three failing rows
(2138, 2136, 2116) showed:

* the ``extra`` column held only the 5-key dict that
  ``ChatSorcarAgent.run`` writes to ``_add_task`` at task-creation time
  (no ``tokens``/``cost``/``auto_commit_mode``), and
* the ``events`` table held no ``task_done`` / ``task_stopped`` /
  ``task_error`` event,

i.e. neither :func:`_save_task_result` nor :func:`_save_task_extra`
from ``_TaskRunnerMixin._run_task_inner``'s cleanup finally ever ran
to overwrite the sentinel.

Root cause: when ``tab.agent.run`` raises a ``BaseException`` that is
NOT ``Exception`` and NOT ``KeyboardInterrupt`` (e.g. ``SystemExit``,
``GeneratorExit``, or a custom ``BaseException`` subclass propagated
from the Anthropic streaming context manager / asyncio cancellation),
the inner per-subtask ``try/except`` blocks do not match, the
exception falls through to the outer ``except BaseException`` which
silently sets a default ``task_end_event`` BUT leaves
``result_summary`` at the initial ``"Agent Failed Abruptly"``
sentinel.  The cleanup finally then writes that sentinel back into
the row.

The fix is twofold:

1. The outer ``except BaseException`` updates ``result_summary`` to a
   meaningful "Task failed: <type>: <msg>" string when it is still the
   initial sentinel, so the cleanup finally persists a real result.
2. The cleanup finally calls ``_save_task_result`` /
   ``_save_task_extra`` BEFORE any other broadcast / merge-view / etc.
   work, so any further BaseException during cleanup does not skip
   the result write.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


class _CustomBaseException(BaseException):
    """A BaseException subclass that is NOT ``Exception``.

    Mirrors the runtime behaviour of ``SystemExit`` /
    ``asyncio.CancelledError`` (Python 3.11+) / ``GeneratorExit``.
    Used here instead of those so the test does not also abort the
    interpreter or the active asyncio loop.
    """


def _run_task_with_agent_raising(
    exc_to_raise: BaseException,
    *,
    tab_id: str,
) -> tuple[int, dict[str, Any]]:
    """Drive ``_run_task`` end-to-end with a stub agent that raises *exc*.

    The stub mimics ``ChatSorcarAgent.run`` to the extent the bug
    requires: it writes the task row via ``_add_task`` (so a row
    exists with the sentinel result and the 5-key early_extra), sets
    ``_last_task_id``, accrues fake usage counters, then raises.

    Returns ``(task_id, row_dict)`` so the test can assert on what
    survived in ``task_history`` AFTER ``_run_task`` completes.
    """
    server = _make_server()
    tab = server._get_tab(tab_id)
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    tab.agent = agent
    tab.chat_id = ""

    captured_task_id: dict[str, int] = {}

    def fake_run(**kwargs: Any) -> None:
        agent.total_tokens_used = 4242
        agent.budget_used = 0.1234
        agent.step_count = 17
        agent._chat_id = agent._chat_id or "test-chat-id"
        task_id, _ = _persistence._add_task(
            kwargs.get("prompt_template", ""),
            chat_id=agent._chat_id,
            extra={
                "model": kwargs.get("model_name", ""),
                "work_dir": kwargs.get("work_dir", ""),
                "version": "test",
                "is_parallel": bool(kwargs.get("is_parallel", False)),
                "is_worktree": bool(kwargs.get("use_worktree", False)),
            },
        )
        agent._last_task_id = task_id
        captured_task_id["id"] = task_id
        raise exc_to_raise

    agent.run = fake_run  # type: ignore[assignment]

    tab.stop_event = threading.Event()
    tab.user_answer_queue = queue.Queue()

    task_thread = threading.Thread(
        target=server._run_task,
        args=({
            "type": "run",
            "prompt": "reproduce-agent-failed-abruptly",
            "tabId": tab_id,
            "workDir": "/tmp",
            "useParallel": False,
            "useWorktree": False,
            "autoCommit": False,
        },),
        daemon=True,
    )
    tab.task_thread = task_thread
    task_thread.start()
    task_thread.join(timeout=15)
    assert not task_thread.is_alive(), "task thread did not finish"

    task_id = captured_task_id["id"]
    _persistence._flush_chat_events()
    db = _persistence._get_db()
    row = db.execute(
        "SELECT id, result, extra FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row is not None
    return task_id, {"id": row["id"], "result": row["result"], "extra": row["extra"]}


class TestAgentFailedAbruptlyReproduction(TestCase):
    """Reproduce and verify the fix for the sentinel-result bug."""

    def test_base_exception_subclass_does_not_leave_sentinel(self) -> None:
        """A ``BaseException`` (not ``Exception``, not ``KeyboardInterrupt``)
        propagated from ``tab.agent.run`` must NOT leave the row with the
        ``"Agent Failed Abruptly"`` sentinel.  Before the fix, this was the
        observed production failure mode (see tasks 2138, 2136, 2116 in
        the bundled ``sorcar.db``).
        """
        exc = _CustomBaseException("simulated async interrupt")
        _task_id, row = _run_task_with_agent_raising(exc, tab_id="abrupt-1")
        assert row["result"] != "Agent Failed Abruptly", (
            "regression: task row still carries the initial sentinel; "
            f"got {row['result']!r}"
        )
        # The recovered message should mention the exception type so the
        # user gets diagnostic context in the history sidebar.
        assert "_CustomBaseException" in row["result"], (
            f"expected exception type in result, got {row['result']!r}"
        )

    def test_system_exit_does_not_leave_sentinel(self) -> None:
        """``SystemExit`` from the agent (e.g. a stray ``sys.exit()``) is
        also a ``BaseException``.  Same expectation as above.
        """
        _task_id, row = _run_task_with_agent_raising(
            SystemExit("simulated"), tab_id="abrupt-2",
        )
        assert row["result"] != "Agent Failed Abruptly", (
            f"regression: SystemExit left sentinel; got {row['result']!r}"
        )

    def test_extra_is_saved_with_usage_fields(self) -> None:
        """The post-cleanup ``extra`` must include ``tokens`` / ``cost`` /
        ``auto_commit_mode`` (the 8-key payload written by the task_runner
        finally), not just the 5-key ``early_extra`` from ``_add_task``.
        This was the second smoking-gun signal in the failing rows.
        """
        import json

        _task_id, row = _run_task_with_agent_raising(
            _CustomBaseException("simulated"), tab_id="abrupt-3",
        )
        extra = json.loads(row["extra"]) if row["extra"] else {}
        assert "tokens" in extra and extra["tokens"] == 4242, (
            f"task_runner finally did not write usage extra; got {extra!r}"
        )
        assert "cost" in extra, f"missing cost in extra: {extra!r}"
        assert "auto_commit_mode" in extra, (
            f"missing auto_commit_mode in extra: {extra!r}"
        )

    def test_regular_exception_still_works(self) -> None:
        """Regression guard: a plain ``Exception`` from the agent should
        continue to land in the inner ``except Exception`` and produce
        ``"Task failed: <msg>"``.  The fix must not alter this path.
        """
        _task_id, row = _run_task_with_agent_raising(
            RuntimeError("model down"), tab_id="abrupt-4",
        )
        assert row["result"].startswith("Task failed:"), (
            f"Exception path regressed; got {row['result']!r}"
        )
        assert "model down" in row["result"]
