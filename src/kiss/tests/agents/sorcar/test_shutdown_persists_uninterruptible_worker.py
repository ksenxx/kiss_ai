# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Scoping rules of the pre-emptive shutdown persistence helper.

Moved from
``kiss.tests.agents.vscode.test_shutdown_persists_uninterruptible_worker``
because these tests drive
:func:`persistence._shutdown_persist_in_flight_results` directly and
depend only on ``kiss.core`` and ``kiss.agents.sorcar``.  The
end-to-end wedged-worker test that constructs a real
``RemoteAccessServer`` (and thus depends on ``kiss.server``) remains in
the original module, which imports the row helpers below back from
here.

Background: :meth:`ChatSorcarAgent.run` inserts every ``task_history``
row with the sentinel ``result = "Agent Failed Abruptly"``; the worker's
cleanup ``finally`` normally overwrites it.  When a shutdown finds a
worker wedged in an uninterruptible call, the pre-emptive safety net
:func:`_persistence._shutdown_persist_in_flight_results` rewrites the
sentinel to ``"Task interrupted by server restart/shutdown"`` BEFORE the
join timeout can expire — but it must be tightly scoped, which is what
the tests here pin down.
"""

from __future__ import annotations

from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence


def _insert_sentinel_row(chat_id: str) -> str:
    """Insert a fresh ``task_history`` row carrying the sentinel.

    Mirrors what :meth:`ChatSorcarAgent.run` does at the start of every
    task — the same shape :func:`_recover_orphaned_tasks` looks at when
    rewriting orphans.
    """
    task_id, _ = _persistence._add_task(
        "shutdown-while-wedged",
        chat_id=chat_id,
        extra={
            "model": "test/model",
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
    assert row is not None, f"row {task_id} disappeared"
    return str(row["result"])


class TestShutdownPersistsUninterruptibleWorker(TestCase):
    """The pre-emptive rewrite must touch exactly the supplied sentinel rows."""

    def test_pre_emptive_rewrite_only_touches_active_ids(self) -> None:
        """The safety net must be tightly scoped: an unrelated row
        that also still carries the sentinel (e.g. a true orphan from
        a previous crash that has not yet been swept) must NOT be
        clobbered to "restart/shutdown" — that is the
        ``_recover_orphaned_tasks`` sweep's job, with its own,
        different, message.
        """
        active_id = _insert_sentinel_row("shutdown-uninterruptible-chat-2a")
        bystander_id = _insert_sentinel_row("shutdown-uninterruptible-chat-2b")
        assert _row_result(active_id) == "Agent Failed Abruptly"
        assert _row_result(bystander_id) == "Agent Failed Abruptly"

        affected = _persistence._shutdown_persist_in_flight_results(
            {active_id},
        )
        assert affected == 1, f"expected exactly 1 rewrite, got {affected}"
        assert _row_result(active_id) == (
            "Task interrupted by server restart/shutdown"
        )
        assert _row_result(bystander_id) == "Agent Failed Abruptly", (
            "bystander row was clobbered; helper must be scoped to "
            "the supplied id set"
        )

    def test_pre_emptive_rewrite_does_not_clobber_completed_row(self) -> None:
        """A row that the worker *did* manage to overwrite with a
        real result before the shutdown call must NEVER be
        downgraded to "restart/shutdown" — the helper conditions on
        ``result = 'Agent Failed Abruptly'`` exactly to avoid this.
        """
        completed_id = _insert_sentinel_row("shutdown-uninterruptible-chat-3")
        _persistence._save_task_result(
            "Task completed successfully", task_id=completed_id,
        )
        assert _row_result(completed_id) == "Task completed successfully"

        affected = _persistence._shutdown_persist_in_flight_results(
            {completed_id},
        )
        assert affected == 0, (
            "helper must not rewrite rows that already have a real "
            f"result; got affected={affected}"
        )
        assert _row_result(completed_id) == "Task completed successfully"

    def test_empty_id_set_is_noop(self) -> None:
        """Calling the helper with no ids must be a safe no-op."""
        affected = _persistence._shutdown_persist_in_flight_results(set())
        assert affected == 0
