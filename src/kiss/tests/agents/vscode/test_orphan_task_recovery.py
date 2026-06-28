# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Startup-time recovery of orphan ``Agent Failed Abruptly`` rows.

Production failure signature (rows 2143, 2140, 2139, 2136 in the
shipped ``sorcar.db``):

* ``result == "Agent Failed Abruptly"`` (the sentinel written by
  :func:`_add_task` at task-creation time), AND
* ``extra`` is the 5-key ``early_extra`` dict only — no ``tokens`` /
  ``cost`` / ``auto_commit_mode``, AND
* the ``events`` table has no terminal
  ``task_done`` / ``task_stopped`` / ``task_error`` event.

The combination means neither :func:`_save_task_result` nor
:func:`_save_task_extra` from ``_TaskRunnerMixin._run_task_inner``'s
cleanup ``finally`` ever ran.  That can happen if a ``BaseException``
subclass propagates out of the inner ``try`` (covered by
``test_agent_failed_abruptly.py`` and the outer ``except``), OR if
the host process is killed externally — SIGKILL, VS Code extension
reload, OOM — in which case no Python code runs at all.

Because no Python code runs in the external-kill variant, the only
viable fix is a startup-time recovery sweep:
:func:`_recover_orphaned_tasks` rewrites any surviving sentinel row
to a diagnostic message, and ``VSCodeServer.__init__`` invokes it on
every fresh server boot (with an empty active set, since at
construction time no task in THIS process is running yet).
"""

from __future__ import annotations

import os
import threading
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


def _insert_sentinel_row(task: str, chat_id: str = "orphan-chat") -> str:
    """Insert a task_history row with the abrupt-failure sentinel.

    Mirrors the production ``early_extra`` shape (5 keys) so the
    recovery sweep is exercised against the exact column contents
    observed in the failing rows.
    """
    task_id, _ = _persistence._add_task(
        task,
        chat_id=chat_id,
        extra={
            "model": "anthropic/claude-3-5-sonnet",
            "work_dir": "/tmp",
            "version": "test",
            "is_parallel": False,
            "is_worktree": False,
        },
    )
    return task_id


def _row_result(task_id: str) -> str:
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None
    return str(row["result"])


class TestOrphanTaskRecovery(TestCase):
    """Verify the startup sweep replaces stale sentinels."""

    def test_sweep_idempotent_on_repeat_boot(self) -> None:
        """Booting the server twice must not corrupt the recovered
        text — the second sweep sees no rows still carrying the
        sentinel and is a no-op.
        """
        orphan_id = _insert_sentinel_row(
            "double-boot orphan",
            chat_id="recovery-test-chat-2",
        )
        _make_server()
        first_result = _row_result(orphan_id)
        _make_server()
        second_result = _row_result(orphan_id)
        assert first_result == second_result, (
            "second sweep must not modify already-recovered rows"
        )

    def test_sweep_preserves_non_sentinel_rows(self) -> None:
        """Rows with a real (non-sentinel) result — including
        explicit "Task stopped by user", "Task failed: ...",
        and completed task summaries — must NOT be touched.
        """
        ok_id, _ = _persistence._add_task(
            "completed task",
            chat_id="recovery-test-chat-3",
            extra={"model": "m", "work_dir": "/tmp", "version": "test",
                   "is_parallel": False, "is_worktree": False},
        )
        # Overwrite the sentinel to simulate a normally-completed
        # task.
        _persistence._save_task_result(
            "Task completed successfully", task_id=ok_id,
        )
        # And a deliberately-stopped task — the user clicked Stop.
        stopped_id, _ = _persistence._add_task(
            "user-stopped task",
            chat_id="recovery-test-chat-3",
            extra={"model": "m", "work_dir": "/tmp", "version": "test",
                   "is_parallel": False, "is_worktree": False},
        )
        _persistence._save_task_result(
            "Task stopped by user", task_id=stopped_id,
        )

        _make_server()

        assert _row_result(ok_id) == "Task completed successfully"
        assert _row_result(stopped_id) == "Task stopped by user"

    def test_active_task_ids_are_excluded(self) -> None:
        """Calling ``_recover_orphaned_tasks`` with an explicit
        ``active_task_ids`` set must leave those rows untouched
        even if they still carry the sentinel.

        This protects an in-flight task whose ``_run_task_inner``
        finally has not yet had a chance to run from being
        clobbered by a concurrent recovery sweep (e.g. a second
        server instance constructed for tests, or a future
        re-arming of the sweep from somewhere other than init).
        """
        active_id = _insert_sentinel_row(
            "currently running",
            chat_id="recovery-test-chat-4-active",
        )
        orphan_id = _insert_sentinel_row(
            "actually dead",
            chat_id="recovery-test-chat-4-orphan",
        )
        n = _persistence._recover_orphaned_tasks({active_id})
        assert n >= 1, "at least the orphan row must be rewritten"
        assert _row_result(active_id) == "Agent Failed Abruptly", (
            "active row was clobbered by the sweep"
        )
        assert _row_result(orphan_id) != "Agent Failed Abruptly", (
            "orphan row was not swept"
        )

    def test_sweep_with_no_orphans_returns_zero(self) -> None:
        """Boot-time call when the table has no sentinel rows must
        be a no-op (rowcount 0) and not raise.
        """
        # Construct a server once to flush any pre-existing orphans
        # from prior tests in the same process.
        _make_server()
        n = _persistence._recover_orphaned_tasks(set())
        assert n == 0, f"expected zero updates, got {n}"

    def test_concurrent_boot_does_not_corrupt(self) -> None:
        """Two ``VSCodeServer`` constructions on different threads
        must both complete without raising; the orphan row ends up
        with the recovered text exactly once.
        """
        orphan_id = _insert_sentinel_row(
            "concurrent-boot orphan",
            chat_id="recovery-test-chat-5",
        )

        errors: list[BaseException] = []

        def boot() -> None:
            try:
                _make_server()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=boot) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
            assert not t.is_alive(), "boot thread did not finish"
        assert not errors, f"concurrent boot raised: {errors!r}"
        assert _row_result(orphan_id) != "Agent Failed Abruptly"
