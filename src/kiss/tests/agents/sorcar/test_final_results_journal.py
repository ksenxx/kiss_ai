# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Final-results sidecar journal: lost DB results must be restorable.

Incident 2026-09-12: task ``f1649eb1…`` finished successfully and
``_save_task_result`` committed the summary at 16:25:24, but the
database had entered a ``disk I/O error`` state around 16:20:44 and
every WAL frame committed after that point was silently discarded when
the next server process opened the file.  The startup orphan sweep
(:func:`persistence._recover_orphaned_tasks`) then found the creation
sentinel ``"Agent Failed Abruptly"`` where the result used to be and
mislabeled the finished task as
``"Task terminated unexpectedly (process killed)"``.

The fix journals every saved final result to a plain JSON-lines
sidecar (``<db>.final_results.jsonl``) that does not share SQLite's
failure modes, and the sweep restores a journalled result instead of
writing the kill message.  These tests exercise the journal append,
load, restore, prune, and failure-tolerance paths end-to-end against
real files and a real temporary database — no mocks.

Branch-coverage notes: the ``OSError`` fallbacks in
``_journal_final_result``, ``_load_final_results``, and
``_prune_final_results_journal`` are reached without test doubles by
making the sidecar path a *directory* (``open()`` then raises
``IsADirectoryError``, an ``OSError`` subclass).  The
``fcntl is None`` fallbacks inside ``_journal_file_lock`` and
``_prune_final_results_journal`` are platform-specific (Windows) and
unreachable on this platform, as already documented in
``persistence.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_SENTINEL = "Agent Failed Abruptly"
_KILLED = "Task terminated unexpectedly (process killed)"


class TestFinalResultsJournal:
    """End-to-end coverage of the final-results sidecar journal."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sidecar(self) -> str:
        return th._final_results_path(str(th._DB_PATH))

    def _add_sentinel_task(self, task: str) -> str:
        task_id, _ = th._add_task(
            task,
            chat_id="journal-test-chat",
            extra={"model": "m", "work_dir": "/tmp", "version": "t",
                   "is_parallel": False, "is_worktree": False},
        )
        return task_id

    def _row_result(self, task_id: str) -> str:
        row = th._get_db().execute(
            "SELECT result FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        assert row is not None
        return str(row["result"])

    def _simulate_wal_loss(self, task_id: str) -> None:
        """Reset the row to its creation state, as if the committed
        result UPDATE had been discarded with the WAL, and clear the
        owner token, as the kernel does when the owning process dies."""
        th._get_db().execute(
            "UPDATE task_history SET result = ?, owner = '' WHERE id = ?",
            (_SENTINEL, task_id),
        )

    def test_save_task_result_journals_to_sidecar(self) -> None:
        """A saved result lands in both the DB row and the sidecar."""
        task_id = self._add_sentinel_task("journalled task")
        th._save_task_result("<p>done</p>", task_id=task_id)
        assert self._row_result(task_id) == "<p>done</p>"
        assert th._load_final_results(self._sidecar()) == {
            task_id: "<p>done</p>",
        }

    def test_legacy_save_without_task_id_does_not_journal(self) -> None:
        """The legacy task-text fallback path skips the journal."""
        self._add_sentinel_task("legacy task")
        th._save_task_result("<p>legacy</p>", task=("legacy task"))
        assert not os.path.exists(self._sidecar())

    def test_sweep_restores_journalled_result_after_wal_loss(self) -> None:
        """The exact incident: finished task, lost UPDATE, dead owner.

        The sweep must restore the journalled result instead of
        writing the "process killed" message, while a genuinely killed
        row swept in the same pass still gets the kill message.
        """
        finished_id = self._add_sentinel_task("finished, then WAL lost")
        killed_id = self._add_sentinel_task("killed mid-task")
        th._save_task_result("<h3>real summary</h3>", task_id=finished_id)
        self._simulate_wal_loss(finished_id)
        self._simulate_wal_loss(killed_id)  # never had a saved result

        n = th._recover_orphaned_tasks(set())

        assert n == 2
        assert self._row_result(finished_id) == "<h3>real summary</h3>"
        assert self._row_result(killed_id) == _KILLED
        # The journal survives the sweep (entries are fresh).
        assert finished_id in th._load_final_results(self._sidecar())

    def test_last_journal_entry_wins(self) -> None:
        """A re-saved result supersedes the earlier journal entry."""
        task_id = self._add_sentinel_task("saved twice")
        th._save_task_result("<p>first</p>", task_id=task_id)
        th._save_task_result("<p>second</p>", task_id=task_id)
        self._simulate_wal_loss(task_id)
        th._recover_orphaned_tasks(set())
        assert self._row_result(task_id) == "<p>second</p>"

    def test_load_skips_corrupt_and_mistyped_lines(self) -> None:
        """Torn writes and mistyped entries never break the load."""
        sidecar = self._sidecar()
        with open(sidecar, "w", encoding="utf-8") as stream:
            stream.write("{torn json\n")
            stream.write(json.dumps(["not", "a", "dict"]) + "\n")
            stream.write(json.dumps({"task_id": 7, "result": "x"}) + "\n")
            stream.write(json.dumps({"task_id": "a"}) + "\n")
            stream.write(
                json.dumps({"task_id": "ok", "result": "<p>ok</p>"}) + "\n"
            )
        assert th._load_final_results(sidecar) == {"ok": "<p>ok</p>"}

    def test_load_missing_file_returns_empty(self) -> None:
        assert th._load_final_results(self._sidecar()) == {}

    def test_append_after_torn_tail_preserves_new_record(self) -> None:
        """A torn last record (no trailing newline, from a process
        killed mid-append) must not swallow the NEXT record: the
        append terminates the torn fragment first, so only the torn
        record is discarded."""
        sidecar = self._sidecar()
        with open(sidecar, "w", encoding="utf-8") as stream:
            stream.write('{"task_id":"torn')  # no trailing newline
        task_id = self._add_sentinel_task("after torn tail")
        th._save_task_result("<p>intact</p>", task_id=task_id)
        assert th._load_final_results(sidecar) == {task_id: "<p>intact</p>"}

    def test_journal_append_failure_is_swallowed(self) -> None:
        """An unwritable sidecar must not break result persistence."""
        os.makedirs(self._sidecar())  # open(sidecar, "a") -> OSError
        task_id = self._add_sentinel_task("unwritable sidecar")
        th._save_task_result("<p>still saved</p>", task_id=task_id)
        assert self._row_result(task_id) == "<p>still saved</p>"

    def test_sweep_prunes_stale_journal_entries(self) -> None:
        """Entries older than the cap (and corrupt lines) are pruned;
        fresh entries are kept."""
        sidecar = self._sidecar()
        stale_ts = time.time() - th._FINAL_RESULTS_MAX_AGE_S - 60
        with open(sidecar, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(
                {"task_id": "old", "result": "x", "ts": stale_ts}) + "\n")
            stream.write("{torn json\n")
            stream.write(json.dumps(
                {"task_id": "new", "result": "y", "ts": time.time()}) + "\n")
        th._recover_orphaned_tasks(set())
        assert th._load_final_results(sidecar) == {"new": "y"}

    def test_prune_without_journal_or_changes_is_a_no_op(self) -> None:
        """No sidecar, then an all-fresh sidecar: both leave the file
        system untouched (early-return branches)."""
        sidecar = self._sidecar()
        th._prune_final_results_journal(sidecar)
        assert not os.path.exists(sidecar)
        task_id = self._add_sentinel_task("fresh entry")
        th._save_task_result("<p>fresh</p>", task_id=task_id)
        before = os.stat(sidecar).st_mtime_ns
        th._prune_final_results_journal(sidecar)
        assert os.stat(sidecar).st_mtime_ns == before

    def test_prune_failure_is_swallowed(self) -> None:
        """A sidecar that cannot be read (a directory) is left alone."""
        os.makedirs(self._sidecar())
        th._prune_final_results_journal(self._sidecar())  # must not raise

    def test_sweep_with_no_dead_rows_touches_nothing(self) -> None:
        """A live-owner sentinel row is neither restored nor killed."""
        task_id = self._add_sentinel_task("still running")
        assert th._recover_orphaned_tasks(set()) == 0
        assert self._row_result(task_id) == _SENTINEL
