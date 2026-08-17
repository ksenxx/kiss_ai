# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep-level recovery of orphan ``Agent Failed Abruptly`` rows.

Moved from ``kiss.tests.server.test_orphan_task_recovery``
because these tests drive :func:`persistence._recover_orphaned_tasks`
directly and depend only on ``kiss.core`` and ``kiss.agents.sorcar``.
The boot-time tests that construct a real ``VSCodeServer`` (and thus
depend on ``kiss.server``) remain in the original module, which imports
the row helpers below back from here.

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
cleanup ``finally`` ever ran.  :func:`_recover_orphaned_tasks` rewrites
any surviving sentinel row to a diagnostic message; the tests here pin
down the protections that scope that sweep (active-id set, owner
liveness, creation cut-off).
"""

from __future__ import annotations

import time
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence


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


def _disown(task_id: str) -> None:
    """Clear the ``owner`` token of *task_id*.

    ``_add_task`` stamps every row with :func:`persistence.
    _process_owner_token`, published as an flock-held marker under
    ``<KISS_HOME>/task-owners/``, and
    :func:`persistence._recover_orphaned_tasks` deliberately never
    rewrites a row whose owning process is still alive (a booting
    daemon must not paint another LIVE process's running task as
    "process killed").

    These tests necessarily seed their "orphan" with ``_add_task``
    from the live pytest process, so that row would be correctly
    exempted.  Clearing the token is what a row genuinely left behind
    by a prior, now-DEAD process looks like: its marker's flock was
    released by the kernel when the process died.  Only rows that are
    meant to be swept are disowned; every row a test expects the
    sweep to PROTECT keeps its owner unless the test explicitly
    isolates a different protection mechanism.
    """
    _persistence._get_db().execute(
        "UPDATE task_history SET owner = '' WHERE id = ?", (task_id,),
    )


def _owner_of(task_id: str) -> str:
    db = _persistence._get_db()
    row = db.execute(
        "SELECT owner FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None
    return str(row["owner"] or "")


def _set_owner(task_id: str, owner: str) -> None:
    _persistence._get_db().execute(
        "UPDATE task_history SET owner = ? WHERE id = ?", (owner, task_id),
    )


def _row_result(task_id: str) -> str:
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None
    return str(row["result"])


class TestOrphanTaskRecovery(TestCase):
    """Verify the sweep's row-protection rules without a server boot."""

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
        # Only the row that is playing the part of a task abandoned by
        # a prior, DEAD process is disowned (see :func:`_disown`).
        # ``active_id`` keeps the live owner token ``_add_task`` gave
        # it, exactly like the in-flight task it models.
        _disown(orphan_id)
        n = _persistence._recover_orphaned_tasks({active_id})
        assert n >= 1, "at least the orphan row must be rewritten"
        assert _row_result(active_id) == "Agent Failed Abruptly", (
            "active row was clobbered by the sweep"
        )
        assert _row_result(orphan_id) != "Agent Failed Abruptly", (
            "orphan row was not swept"
        )

    def test_active_set_alone_protects_a_disowned_row(self) -> None:
        """``active_task_ids`` must protect a row on its own.

        Companion to ``test_active_task_ids_are_excluded``, which
        keeps the live owner token on the active row.  Ownership is a
        second, independent shield there, so that test alone cannot
        prove the active set is still honoured.  Here BOTH rows are
        disowned, leaving ``active_task_ids`` as the only thing that
        can save the active row — the exact invariant the original
        test was written for.
        """
        active_id = _insert_sentinel_row(
            "currently running, owner already reaped",
            chat_id="recovery-test-chat-4b-active",
        )
        orphan_id = _insert_sentinel_row(
            "actually dead",
            chat_id="recovery-test-chat-4b-orphan",
        )
        owner = _owner_of(active_id)
        _disown(active_id)
        _disown(orphan_id)
        try:
            n = _persistence._recover_orphaned_tasks({active_id})
            assert n >= 1, "at least the orphan row must be rewritten"
            assert _row_result(active_id) == "Agent Failed Abruptly", (
                "regression: a row named in active_task_ids was swept "
                "once its owner token was gone — an in-flight task "
                "whose cleanup finally has not run yet would be "
                "mislabeled as 'process killed'"
            )
            assert _row_result(orphan_id) != "Agent Failed Abruptly", (
                "orphan row was not swept"
            )
        finally:
            # Leave no disowned sentinel row behind: the database is
            # shared by every test in this module.
            _set_owner(active_id, owner)

    def test_live_owner_row_is_exempt_from_sweep(self) -> None:
        """A row owned by a LIVE process is never rewritten.

        The sweep decides liveness from the database, not from process
        memory: a second Sorcar process (a ``kiss`` CLI run, a VS Code
        reload, a restarted daemon) used to rewrite rows for tasks
        still RUNNING in the first process, painting a red failure dot
        on a live task and destroying the sentinel that both the
        shutdown safety net and any later sweep condition on.  Here
        the row is NOT in ``active_task_ids`` and no cut-off is
        given, so its owner token is the only thing protecting it.
        """
        live_id = _insert_sentinel_row(
            "running in a live process",
            chat_id="recovery-test-chat-8",
        )
        owner = _owner_of(live_id)
        assert owner, "_add_task must stamp the creating process's token"
        assert _persistence._owner_is_alive(owner), (
            "this pytest process still holds its liveness marker"
        )
        _persistence._recover_orphaned_tasks(set())
        assert _row_result(live_id) == "Agent Failed Abruptly", (
            "regression: the sweep rewrote a row whose owning process "
            "is still alive"
        )

    def test_sweep_ignores_rows_created_after_cutoff(self) -> None:
        """``created_before`` scopes the sweep to pre-boot rows.

        A sentinel row inserted AFTER the cut-off models a task that
        legitimately started while the background sweep was still
        pending (e.g. delayed by SQLite lock contention).  It belongs
        to the live process and must NOT be rewritten — only the
        pre-cut-off row is a true orphan of a prior, dead process.
        """
        orphan_id = _insert_sentinel_row(
            "pre-boot orphan",
            chat_id="recovery-test-chat-6-orphan",
        )
        # Only the pre-cut-off row models a task abandoned by a prior,
        # DEAD process, so only it is disowned (see :func:`_disown`).
        # ``fresh_id`` keeps the live owner token ``_add_task`` gave
        # it, exactly like the just-started task it models;
        # ``test_created_before_alone_protects_a_disowned_fresh_row``
        # covers the cut-off in isolation from ownership.
        _disown(orphan_id)
        cutoff = time.time()
        fresh_id = _insert_sentinel_row(
            "task started after boot",
            chat_id="recovery-test-chat-6-fresh",
        )
        _persistence._recover_orphaned_tasks(set(), created_before=cutoff)
        assert _row_result(orphan_id) == (
            "Task terminated unexpectedly (process killed)"
        ), "pre-boot orphan row must still be swept"
        assert _row_result(fresh_id) == "Agent Failed Abruptly", (
            "regression: the sweep clobbered a sentinel row created "
            "after the boot cut-off — a live task would be mislabeled "
            "as 'process killed' and the pre-emptive shutdown "
            "persistence (which conditions on the sentinel) defeated"
        )

    def test_created_before_alone_protects_a_disowned_fresh_row(
        self,
    ) -> None:
        """``created_before`` must protect a fresh row on its own.

        Companion to ``test_sweep_ignores_rows_created_after_cutoff``,
        where the fresh row keeps its live owner token and is thus
        shielded twice over.  Disowning it too models the real race
        the cut-off exists for: a task started just after boot by a
        process that the sweep cannot see as alive.  Only the
        timestamp filter can save it here.
        """
        orphan_id = _insert_sentinel_row(
            "pre-boot orphan",
            chat_id="recovery-test-chat-6b-orphan",
        )
        _disown(orphan_id)
        cutoff = time.time()
        fresh_id = _insert_sentinel_row(
            "task started after boot",
            chat_id="recovery-test-chat-6b-fresh",
        )
        owner = _owner_of(fresh_id)
        _disown(fresh_id)
        try:
            _persistence._recover_orphaned_tasks(set(), created_before=cutoff)
            assert _row_result(orphan_id) == (
                "Task terminated unexpectedly (process killed)"
            ), "pre-boot orphan row must still be swept"
            assert _row_result(fresh_id) == "Agent Failed Abruptly", (
                "regression: the sweep clobbered a post-cut-off "
                "sentinel row once ownership stopped shielding it"
            )
        finally:
            # Leave no disowned sentinel row behind: the database is
            # shared by every test in this module.
            _set_owner(fresh_id, owner)
