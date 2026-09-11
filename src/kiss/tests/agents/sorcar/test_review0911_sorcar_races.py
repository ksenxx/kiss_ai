# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the 2026-09-11 sorcar concurrency review.

Five findings from a read-only audit of ``src/kiss/agents/sorcar``:

1. Normal worktree merges ran the multi-command stash → checkout →
   squash → commit → pop transaction on the shared main worktree under
   only the process-local ``repo_lock``; a second PROCESS (whose
   failure paths run ``git reset --hard HEAD``) could wipe the staged
   merge.  The cross-process ``kiss-reclaim.lock`` flock must cover
   normal merges too.
2. ``persistence._RWLock`` stranded ``_readers`` / ``_pending_writers``
   / ``_writer`` when a ``KeyboardInterrupt`` (the server injects one
   at arbitrary bytecode boundaries to stop a task) landed after the
   state was published but before the cleanup ``try/finally`` was
   entered, permanently wedging all persistence operations.
3. ``_AbandonedSubagent.unbanked_usage()`` subtracted already-banked
   spend from the parent when the child's live usage snapshot
   momentarily regressed (RelentlessAgent detaches
   ``_current_executor`` BEFORE folding it into the cumulative
   counters at every session handoff).
4. Journal replay was not exactly-once: a crash after the event batch
   committed but before the claimed snapshot file was unlinked made
   the next replayer insert the same events again under fresh seqs.
5. Multiple claimed journal snapshots were replayed in
   ``.consumed-<pid>-<uuid>`` filename order, not chronological order,
   inverting event order in the transcript.

Everything here is real: real threads with real injected exceptions,
real git repos and OS processes holding a real ``flock``, a real
SQLite database and real journal snapshot files.  Nothing is mocked
or patched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar.git_worktree import _git
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _AbandonedSubagent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

# ---------------------------------------------------------------------------
# Finding 2: _RWLock vs. asynchronously injected KeyboardInterrupt
# ---------------------------------------------------------------------------

_ACQUIRE_TIMEOUT_S = 5.0


def _lock_usable(lock: persistence._RWLock) -> bool:
    """Return True when a fresh writer AND reader can still get the lock.

    A stranded ``_readers`` count blocks writers forever; a stranded
    ``_writer`` / ``_pending_writers`` blocks everyone.  Each probe
    runs in its own real thread with a hard timeout.
    """
    results: list[bool] = []

    def writer() -> None:
        with lock.write_lock():
            results.append(True)

    def reader() -> None:
        with lock.read_lock():
            results.append(True)

    for target in (writer, reader):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=_ACQUIRE_TIMEOUT_S)
        if thread.is_alive():
            return False
    return len(results) == 2


def _run_victim_with_injection(
    lock: persistence._RWLock, acquire: Any, nth_line_event: int,
) -> bool:
    """Acquire *lock* in a real thread, raising KeyboardInterrupt at the
    *nth* traced line event of the ACQUISITION phase in persistence.py.

    This delivers a real exception at an exact Python line boundary —
    the same arbitrary-boundary delivery the server's
    ``PyThreadState_SetAsyncExc`` stop performs, made deterministic.
    Counting stops once the lock body has been entered, so only the
    acquisition windows (the ones the audit demonstrated) are swept.

    Args:
        lock: The lock under test.
        acquire: ``lock.read_lock`` or ``lock.write_lock``.
        nth_line_event: 1-based acquisition line event to interrupt at.

    Returns:
        Whether the counter was reached (False: the acquisition ran to
        completion first, i.e. every window has been swept).
    """
    module_file = persistence.__file__
    counter = {"n": 0}
    entered = {"done": False}

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if entered["done"]:
            return None
        if event == "line" and frame.f_code.co_filename == module_file:
            counter["n"] += 1
            if counter["n"] == nth_line_event:
                raise KeyboardInterrupt("injected stop")
        return tracer

    def run() -> None:
        sys.settrace(tracer)
        try:
            with acquire():
                entered["done"] = True
        except KeyboardInterrupt:
            pass
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=_ACQUIRE_TIMEOUT_S * 4)
    assert not thread.is_alive(), (
        f"victim wedged with injection at line event {nth_line_event}"
    )
    return counter["n"] >= nth_line_event


def _sweep_injection_points(acquire_name: str) -> None:
    """Inject KeyboardInterrupt at EVERY acquisition line boundary.

    For each injection point a fresh ``_RWLock`` acquisition runs in a
    real thread; afterwards the lock must still be fully usable by new
    readers and writers.  The sweep stops at the first injection point
    the acquisition no longer reaches (it completed uninterrupted).
    """
    for nth in range(1, 200):
        lock = persistence._RWLock()
        injected = _run_victim_with_injection(
            lock, getattr(lock, acquire_name), nth,
        )
        assert _lock_usable(lock), (
            f"_RWLock permanently wedged by a KeyboardInterrupt "
            f"delivered at line event {nth}: readers={lock._readers} "
            f"writer={lock._writer} pending={lock._pending_writers}"
        )
        if not injected:
            return
    pytest.fail("sweep never reached an uninterrupted acquisition")


class TestRWLockAsyncInterrupt:
    """Finding 2: no interrupt boundary may strand _RWLock state."""

    def test_read_lock_survives_interrupt_at_every_boundary(self) -> None:
        """A reader stopped at any line boundary leaks no reader count."""
        _sweep_injection_points("read_lock")

    def test_write_lock_survives_interrupt_at_every_boundary(self) -> None:
        """A writer stopped at any line boundary leaks no writer state."""
        _sweep_injection_points("write_lock")

    def test_contended_write_lock_survives_interrupt(self) -> None:
        """A writer interrupted while WAITING behind a live reader leaks
        no ``_pending_writers`` count (covers the wait-loop windows)."""
        for nth in range(1, 200):
            lock = persistence._RWLock()
            release_reader = threading.Event()
            reader_in = threading.Event()

            def hold_read() -> None:
                with lock.read_lock():
                    reader_in.set()
                    release_reader.wait(timeout=_ACQUIRE_TIMEOUT_S * 4)

            holder = threading.Thread(target=hold_read, daemon=True)
            holder.start()
            assert reader_in.wait(timeout=_ACQUIRE_TIMEOUT_S)

            release_timer = threading.Timer(0.2, release_reader.set)
            release_timer.start()
            try:
                injected = _run_victim_with_injection(
                    lock, lock.write_lock, nth,
                )
            finally:
                release_reader.set()
                release_timer.cancel()
                holder.join(timeout=_ACQUIRE_TIMEOUT_S)
            assert _lock_usable(lock), (
                f"contended write_lock wedged at line event {nth}: "
                f"pending={lock._pending_writers} writer={lock._writer}"
            )
            if not injected:
                return
        pytest.fail("sweep never reached an uninterrupted acquisition")


# ---------------------------------------------------------------------------
# Finding 3: abandoned-child usage must never regress the parent's totals
# ---------------------------------------------------------------------------


def _child_mid_handoff(
    budget: float, tokens: int, steps: int,
) -> RelentlessAgent:
    """Return a real RelentlessAgent in the mid-session-handoff state.

    ``relentless_agent.py`` sets ``_current_executor = None`` BEFORE
    ``_accumulate_usage()`` folds that executor's spend into the
    cumulative counters, on the success path and on every exception
    path.  A reader in that window sees the given cumulative counters
    and no live executor — this constructs exactly that real state.
    """
    child = RelentlessAgent("review0911-child")
    child.budget_used = budget
    child.total_tokens_used = tokens
    child.total_steps = steps
    child._current_executor = None
    return child


class TestAbandonedUsageRegression:
    """Finding 3: a torn (regressing) live snapshot must be clamped."""

    def test_reclaim_never_subtracts_banked_spend(self) -> None:
        """A mid-handoff snapshot below ``counted`` banks zero, not a
        negative delta, and must not reset ``counted`` downward."""
        parent = SorcarAgent("review0911-parent")
        parent.budget_used = 1.0
        parent.total_tokens_used = 100
        parent.total_steps = 1
        parent.printer = None

        child = _child_mid_handoff(0.0, 0, 0)
        item = _AbandonedSubagent(Future(), child, (1.0, 100, 1))
        parent._abandoned_subagents.append(item)

        assert not parent.reclaim_abandoned_subagents()

        assert (
            parent.budget_used,
            parent.total_tokens_used,
            parent.total_steps,
        ) == (1.0, 100, 1), (
            "reclaim subtracted already-banked spend from the parent "
            "after reading a torn mid-handoff child snapshot"
        )
        assert item.counted == (1.0, 100, 1), (
            "``counted`` regressed; the next reclaim would double-count"
        )

    def test_later_growth_banks_only_the_new_delta(self) -> None:
        """Once the fold lands and spend grows, only the excess over the
        already-counted figure is attributed — never twice."""
        parent = SorcarAgent("review0911-parent2")
        parent.budget_used = 1.0
        parent.total_tokens_used = 100
        parent.total_steps = 1
        parent.printer = None

        child = _child_mid_handoff(0.0, 0, 0)
        item = _AbandonedSubagent(Future(), child, (1.0, 100, 1))
        parent._abandoned_subagents.append(item)
        parent.reclaim_abandoned_subagents()

        # The handoff fold lands, plus fresh post-abandon spend.
        child.budget_used = 1.5
        child.total_tokens_used = 150
        child.total_steps = 2
        parent.reclaim_abandoned_subagents()

        assert (
            parent.budget_used,
            parent.total_tokens_used,
            parent.total_steps,
        ) == (1.5, 150, 2), "delta after a torn read was double-counted"


# ---------------------------------------------------------------------------
# Findings 4 & 5: journal replay exactly-once + chronological ordering
# ---------------------------------------------------------------------------


@pytest.fixture
def kiss_home() -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME for this test."""
    home = Path(tempfile.mkdtemp(prefix="kiss-review0911-"))
    saved_env = os.environ.get("KISS_HOME")
    saved = (persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR)
    os.environ["KISS_HOME"] = str(home)
    persistence._KISS_DIR = home
    persistence._DB_PATH = home / "sorcar.db"
    persistence._db_conn = None
    try:
        yield home
    finally:
        if persistence._db_conn is not None:
            try:
                persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            persistence._DB_PATH,
            persistence._db_conn,
            persistence._KISS_DIR,
        ) = saved
        if saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved_env
        shutil.rmtree(home, ignore_errors=True)


def _journal_row(task_id: str, text: str, timestamp: float) -> str:
    """Return one serialized journal line for *task_id*."""
    return json.dumps({
        "task_id": task_id,
        "event_json": json.dumps({"type": "text", "text": text}),
        "timestamp": timestamp,
        "origin_db_path": str(persistence._DB_PATH),
    }) + "\n"


def _event_texts(task_id: str) -> list[str]:
    """Return the ``text`` of every persisted event of *task_id*, by seq."""
    db = persistence._get_db()
    rows = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [json.loads(row[0]).get("text", "") for row in rows]


_CRASHED_REPLAYER_SCRIPT = """
import json
import os
import sys

from kiss.agents.sorcar import persistence

snapshot = sys.argv[1]
batch = []
with open(snapshot, encoding="utf-8") as stream:
    for line in stream.read().splitlines():
        record = json.loads(line)
        batch.append((
            str(record["task_id"]),
            str(record["event_json"]),
            float(record["timestamp"]),
            str(record["origin_db_path"]),
        ))
# The first half of _replay_journal_snapshot: commit the batch (with
# its exactly-once marker when the parameter exists), then die before
# the os.unlink — the exact crash window under test.
try:
    persistence._write_event_batch(
        batch, replay_marker=os.path.basename(snapshot),
    )
except TypeError:  # pre-fix signature without the marker
    persistence._write_event_batch(batch)
os._exit(1)
"""


class TestJournalReplayExactlyOnce:
    """Finding 4: a committed-then-unremoved snapshot must not replay."""

    def test_snapshot_surviving_its_commit_is_not_replayed_again(
        self, kiss_home: Path,
    ) -> None:
        """A real second process replays the claimed snapshot up to and
        including the database commit, then dies (``os._exit``) before
        the snapshot unlink — the exact crash window.  The next replay
        in this process must remove the file WITHOUT inserting the
        events again."""
        task_id, _chat = persistence._add_task("exactly-once task")
        sidecar = Path(
            persistence._failed_events_path(str(persistence._DB_PATH)),
        )
        snapshot = sidecar.with_name(sidecar.name + ".consumed-4242-cafe")
        snapshot.write_text(_journal_row(task_id, "only once", 10.0), "utf-8")

        crashed = subprocess.run(
            [sys.executable, "-c", _CRASHED_REPLAYER_SCRIPT, str(snapshot)],
            env={**os.environ, "KISS_HOME": str(kiss_home)},
            capture_output=True, text=True, timeout=120,
        )
        assert crashed.returncode == 1, crashed.stderr
        assert snapshot.exists(), "the crashed replayer must leave the file"
        assert _event_texts(task_id) == ["only once"]

        persistence._replay_failed_events()

        assert _event_texts(task_id) == ["only once"], (
            "a snapshot that had already committed was replayed again: "
            "journal replay is not exactly-once across a crash between "
            "the database commit and the snapshot unlink"
        )
        assert not snapshot.exists(), "the recovered snapshot must be removed"

    def test_marker_is_pruned_and_replay_still_works_later(
        self, kiss_home: Path,
    ) -> None:
        """After a fully successful replay the marker row is pruned, and
        a NEW snapshot that reuses the same claimed name (PID + random
        UUID collision across daemon generations) still replays."""
        task_id, _chat = persistence._add_task("marker prune task")
        sidecar = Path(
            persistence._failed_events_path(str(persistence._DB_PATH)),
        )
        snapshot = sidecar.with_name(sidecar.name + ".consumed-7-beef")
        snapshot.write_text(_journal_row(task_id, "first", 1.0), "utf-8")
        persistence._replay_failed_events()

        db = persistence._get_db()
        left = db.execute("SELECT COUNT(*) FROM replayed_journals").fetchone()
        assert left[0] == 0, "replay marker not pruned after the unlink"

        snapshot.write_text(_journal_row(task_id, "second", 2.0), "utf-8")
        persistence._replay_failed_events()
        assert _event_texts(task_id) == ["first", "second"]


class TestJournalSnapshotOrdering:
    """Finding 5: snapshots must replay oldest-first by age, not name."""

    def test_snapshots_replay_in_chronological_order(
        self, kiss_home: Path,
    ) -> None:
        """An older snapshot whose ``<pid>-<uuid>`` name sorts AFTER a
        newer one must still replay first, so its events get the lower
        seqs.  The names below lexicographically invert the real ages
        (``1000-a…`` < ``9000-f…``), exactly the inversion a dead
        high-PID replayer plus a later low-PID replayer produces."""
        task_id, _chat = persistence._add_task("ordering task")
        sidecar = Path(
            persistence._failed_events_path(str(persistence._DB_PATH)),
        )
        older = sidecar.with_name(
            sidecar.name + ".consumed-9000-ffffffffffff",
        )
        newer = sidecar.with_name(
            sidecar.name + ".consumed-1000-aaaaaaaaaaaa",
        )
        older.write_text(_journal_row(task_id, "older event", 100.0), "utf-8")
        time.sleep(0.05)  # distinct mtimes even on coarse filesystems
        newer.write_text(_journal_row(task_id, "newer event", 200.0), "utf-8")

        persistence._replay_failed_events()

        assert _event_texts(task_id) == ["older event", "newer event"], (
            "journal snapshots were replayed in filename order, not "
            "chronological order: newer events received lower seqs"
        )


# ---------------------------------------------------------------------------
# Finding 1: normal merges must hold the cross-process repository flock
# ---------------------------------------------------------------------------


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with one committed file."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True, check=True,
        )
    (path / "f.txt").write_text("f0\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


_FLOCK_HOLDER_SCRIPT = """
import fcntl
import sys
import time

lock_path, held_marker, release_marker = sys.argv[1], sys.argv[2], sys.argv[3]
handle = open(lock_path, "a+")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
open(held_marker, "w").close()
deadline = time.monotonic() + 60
import os
while time.monotonic() < deadline:
    if os.path.exists(release_marker):
        break
    time.sleep(0.02)
fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
handle.close()
"""


class TestMergeHoldsCrossProcessLock:
    """Finding 1: ``_do_merge`` must serialize against other processes."""

    def test_merge_blocks_while_another_process_holds_the_repo_flock(
        self,
    ) -> None:
        """A second OS process holding ``kiss-reclaim.lock`` — as a
        reclaimer or another Sorcar merge does for its whole staged
        transaction — must block this process's merge until released.
        Without the flock the merge runs concurrently and the peer's
        failure-path ``git reset --hard`` can wipe its staged work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            agent = WorktreeSorcarAgent("review0911-merge-lock")
            wt_work = agent._try_setup_worktree(repo, None)
            assert wt_work is not None
            assert agent._wt is not None
            (agent._wt.wt_dir / "agent.txt").write_text("agent change\n")
            _git("add", "-A", cwd=agent._wt.wt_dir)
            result = _git("commit", "-m", "agent work", cwd=agent._wt.wt_dir)
            assert result.returncode == 0, result.stderr

            common = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            common_dir = Path(common)
            if not common_dir.is_absolute():
                common_dir = (repo / common_dir).resolve()
            lock_path = common_dir / "kiss-reclaim.lock"
            held = Path(tmp) / "held"
            release = Path(tmp) / "release"

            holder = subprocess.Popen(
                [
                    sys.executable, "-c", _FLOCK_HOLDER_SCRIPT,
                    str(lock_path), str(held), str(release),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 30
                while not held.exists():
                    assert time.monotonic() < deadline, "holder never locked"
                    time.sleep(0.02)

                merged = threading.Event()
                message: list[str] = []

                def do_merge() -> None:
                    message.append(agent.merge())
                    merged.set()

                merger = threading.Thread(target=do_merge, daemon=True)
                merger.start()
                assert not merged.wait(timeout=2.0), (
                    "merge() completed while another process held the "
                    "cross-process repository lock: normal merges do "
                    "not take the flock that serializes multi-command "
                    "main-worktree transactions between processes"
                )

                release.write_text("go")
                assert merged.wait(timeout=60), "merge never finished"
                merger.join(timeout=10)
            finally:
                release.write_text("go")
                out, err = holder.communicate(timeout=60)
                assert holder.returncode == 0, f"holder failed: {out}\n{err}"

            assert (repo / "agent.txt").exists(), message
