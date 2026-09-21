# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the 2026-09-11 round-2 sorcar review.

A read-only re-review of the round-1 concurrency fixes found three
remaining issues (see ``tmp/review2-sorcar.md``):

1. ``persistence._RWLock`` was still vulnerable to a REAL
   ``PyThreadState_SetAsyncExc``-injected exception: delivery between
   the C-level ``Condition.acquire()`` returning and the next bytecode
   (``STORE_FAST locked``) left the condition's underlying mutex
   permanently locked (read path) or self-deadlocked the victim in its
   own ``finally`` (write path, non-reentrant mutex).  The round-1
   line-tracing sweep can not reach these call/store windows.
2. Journal snapshot replay ordered snapshots by ``st_mtime_ns``, which
   is wrong once a claimed snapshot is RESTORED to the live sidecar
   after a failed replay and then receives later appends (replay order
   became middle, old, new), and undefined when mtimes tie (random
   ``<pid>-<uuid>`` pathname order).
3. ``WorktreeSorcarAgent._try_setup_worktree`` captured the main
   tree's branch and dirty state WITHOUT the cross-process
   ``kiss-reclaim.lock`` flock; on the pooled-spare path it completed
   while another process held the flock, so a peer's merge could
   mutate the shared main tree mid-capture.

Everything here is real: real threads receiving real
``PyThreadState_SetAsyncExc`` injections, real journal files and a
real SQLite database (made to refuse writes by really replacing the
database path with a directory), real git repos, a real pooled spare
worktree, and a real second OS process holding a real ``flock``.
Nothing is mocked or patched.
"""

from __future__ import annotations

import ctypes
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.conftest import posix_only

# ---------------------------------------------------------------------------
# Finding 1: _RWLock vs. a REAL PyThreadState_SetAsyncExc injection
# ---------------------------------------------------------------------------

_ACQUIRE_TIMEOUT_S = 5.0

_set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
_set_async_exc.argtypes = [ctypes.c_ulong, ctypes.py_object]


def _inject_keyboard_interrupt(tid: int) -> int:
    """Queue a real KeyboardInterrupt in thread *tid*.

    Same call ``server.task_runner.inject_keyboard_interrupt`` makes to
    stop a task; duplicated here so a sorcar test does not import the
    server layer.
    """
    rc = int(_set_async_exc(ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)))
    if rc > 1:  # pragma: no cover — rare: exception set in multiple states
        _set_async_exc(ctypes.c_ulong(tid), None)
    return rc


def _lock_usable(lock: persistence._RWLock) -> bool:
    """Return True when a fresh writer AND reader can still get the lock."""
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


def _inject_between_c_acquire_and_next_bytecode(
    lock: persistence._RWLock, acquire_name: str,
) -> threading.Thread:
    """Deliver a real injected stop in the exact call/store window.

    Reproduction from the round-2 review, on the real lock with the
    real injection mechanism (no tracing, no overridden setters):

    1. This thread holds the condition's underlying mutex, so the
       victim blocks inside its first C-level ``Condition.acquire()``.
    2. A real ``PyThreadState_SetAsyncExc`` queues KeyboardInterrupt
       while the victim is still blocked in C (async exceptions are
       only delivered by the eval loop, never inside the C call).
    3. The mutex is released: the victim's C acquire succeeds, and the
       pending exception is delivered BEFORE the next bytecode — the
       same-line boundary no ``sys.settrace`` line event can reach.

    Returns:
        The victim thread, already asked to stop (caller joins it).
    """
    mutex = lock._cond._lock  # type: ignore[attr-defined] # the condition's underlying mutex
    started = threading.Event()

    def victim() -> None:
        started.set()
        try:
            with getattr(lock, acquire_name)():
                pass
        except KeyboardInterrupt:
            pass

    mutex.acquire()
    try:
        thread = threading.Thread(target=victim, daemon=True)
        thread.start()
        assert started.wait(timeout=_ACQUIRE_TIMEOUT_S)
        # Wait until the victim's top Python frame is the acquisition
        # generator itself — from there its only blocking point is the
        # C-level mutex acquire we are holding.  (The generator body is
        # named `_<acquire_name>_gen` since the round-3 traceback-
        # retention fix moved it out of the public wrapper.)
        gen_names = {acquire_name, f"_{acquire_name}_gen"}
        deadline = time.monotonic() + 10.0
        while True:
            frame = sys._current_frames().get(thread.ident or -1)
            if frame is not None and frame.f_code.co_name in gen_names:
                break
            assert time.monotonic() < deadline, "victim never reached acquire"
            time.sleep(0.005)
        time.sleep(0.2)  # let it enter the blocking C acquire
        assert _inject_keyboard_interrupt(thread.ident or -1) == 1
    finally:
        mutex.release()
    return thread


class TestRWLockRealAsyncInjection:
    """Finding 1: the real injection window must not wedge the lock."""

    def test_read_lock_survives_injection_after_c_acquire(self) -> None:
        """A reader stopped between ``Condition.acquire`` returning and
        the next bytecode must not exit with the raw mutex locked."""
        lock = persistence._RWLock()
        thread = _inject_between_c_acquire_and_next_bytecode(lock, "read_lock")
        thread.join(timeout=_ACQUIRE_TIMEOUT_S)
        assert not thread.is_alive(), "read victim wedged in its own teardown"
        assert _lock_usable(lock), (
            "the injected stop stranded the condition's underlying mutex: "
            "every subsequent persistence operation would deadlock"
        )
        assert lock._readers == 0 and not lock._writer
        assert lock._pending_writers == 0

    def test_write_lock_survives_injection_after_c_acquire(self) -> None:
        """A writer stopped in the same window must neither strand the
        mutex nor self-deadlock reacquiring it in its ``finally``."""
        lock = persistence._RWLock()
        thread = _inject_between_c_acquire_and_next_bytecode(lock, "write_lock")
        thread.join(timeout=_ACQUIRE_TIMEOUT_S)
        assert not thread.is_alive(), (
            "write victim self-deadlocked reacquiring the non-reentrant "
            "condition mutex in its own finally"
        )
        assert _lock_usable(lock), "injected stop wedged the write path"
        assert lock._readers == 0 and not lock._writer
        assert lock._pending_writers == 0

    def test_stress_random_time_injections_never_deadlock(self) -> None:
        """Thousands of real acquisitions under hundreds of real
        ``PyThreadState_SetAsyncExc`` injections at random times: the
        lock must stay fully usable with no stranded state.  This is
        the generalization check for the documented residual windows
        (they require multiple injections landing on consecutive
        single-bytecode boundaries)."""
        lock = persistence._RWLock()
        stop = threading.Event()
        counts = [0, 0]
        errors: list[BaseException] = []
        unraisable: list[str] = []
        thread_errors: list[str] = []
        saved_hook = sys.unraisablehook
        saved_thread_hook = threading.excepthook

        def tolerant_thread_hook(args: threading.ExceptHookArgs) -> None:
            """Tolerate a worker killed by an injection at an unguarded
            loop boundary (outside its ``try``); the lock must still be
            healthy afterwards, which the assertions below verify.  Any
            OTHER escaping exception is a real bug and recorded.
            """
            if not (args.exc_type and issubclass(args.exc_type, KeyboardInterrupt)):
                thread_errors.append(repr(args))

        def tolerant_hook(args: sys.UnraisableHookArgs) -> None:
            """Tolerate injected stops landing in dealloc-time code.

            An injection can be delivered inside a weakref callback or
            other dealloc-time code, where CPython routes it to the
            unraisable hook instead of raising — precisely the
            lost-callback case the lock's liveness predicates and
            bounded wait timeout must absorb.  Anything OTHER than the
            injected KeyboardInterrupt is a real bug and recorded.
            """
            if not (args.exc_type and issubclass(args.exc_type, KeyboardInterrupt)):
                unraisable.append(repr(args))

        def worker(idx: int) -> None:
            while True:
                try:
                    if stop.is_set():
                        return
                    with lock.read_lock():
                        counts[idx] += 1
                    with lock.write_lock():
                        counts[idx] += 1
                except KeyboardInterrupt:
                    pass
                except BaseException as exc:  # noqa: BLE001 — must record
                    errors.append(exc)
                    return

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in (0, 1)
        ]
        sys.unraisablehook = tolerant_hook
        threading.excepthook = tolerant_thread_hook
        try:
            for thread in threads:
                thread.start()
            rng = random.Random(20260911)
            injections = 0
            deadline = time.monotonic() + 10.0
            while injections < 400 and time.monotonic() < deadline:
                time.sleep(rng.uniform(0.0, 0.003))
                target = threads[injections % 2]
                if target.is_alive() and target.ident is not None:
                    _inject_keyboard_interrupt(target.ident)
                    injections += 1
            stop.set()
            for thread in threads:
                thread.join(timeout=15.0)
                assert not thread.is_alive(), (
                    f"worker wedged after {injections} injections and "
                    f"{sum(counts)} acquisitions"
                )
        finally:
            sys.unraisablehook = saved_hook
            threading.excepthook = saved_thread_hook
        assert not errors, f"lock raised under injection: {errors!r}"
        assert not unraisable, f"unexpected unraisable errors: {unraisable!r}"
        assert not thread_errors, f"unexpected thread errors: {thread_errors!r}"
        assert _lock_usable(lock), "lock unusable after random injections"
        assert lock._readers == 0 and not lock._writer
        assert lock._pending_writers == 0
        assert sum(counts) > 1000, "stress made too little progress"


# ---------------------------------------------------------------------------
# Finding 2: journal replay order after a failed replay + later appends
# ---------------------------------------------------------------------------


@pytest.fixture
def journal_home() -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME for this test."""
    home = Path(tempfile.mkdtemp(prefix="kiss-review0911r2-"))
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


def _journal(task_id: str, text: str, timestamp: float) -> None:
    """Append one event for *task_id* to the live journal sidecar."""
    persistence._journal_failed_events(
        [(
            task_id,
            json.dumps({"type": "text", "text": text}),
            timestamp,
            str(persistence._DB_PATH),
        )],
        1,
    )


def _claim_all() -> list[str]:
    """Claim the live sidecar exactly as a real replayer does."""
    live = persistence._failed_events_path(str(persistence._DB_PATH))
    with persistence._journal_lock, persistence._journal_file_lock(live):
        return persistence._claim_journal_snapshots(live)


def _event_texts(task_id: str) -> list[str]:
    """Return the ``text`` of every persisted event of *task_id*, by seq."""
    db = persistence._get_db()
    rows = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [json.loads(row[0]).get("text", "") for row in rows]


@contextmanager
def _database_refusing_writes() -> Iterator[None]:
    """Make the real database genuinely unopenable for the duration.

    The database file is moved aside and a DIRECTORY is created at its
    path: every fresh ``sqlite3.connect`` then really fails ("unable
    to open database file"), which is exactly how a replay attempt
    fails while the database refuses writes.  No mocking: real files,
    real sqlite errors.
    """
    db_path = Path(str(persistence._DB_PATH))
    backup = db_path.with_name(db_path.name + ".moved-aside")
    os.replace(db_path, backup)
    db_path.mkdir()
    try:
        yield
    finally:
        db_path.rmdir()
        os.replace(backup, db_path)


class TestJournalOrderAcrossFailedReplays:
    """Finding 2: chronology must survive failed replays + appends."""

    @posix_only("replacing a SQLite file another connection holds open")
    def test_failed_replay_then_later_appends_keep_chronology(
        self, journal_home: Path,
    ) -> None:
        """The review's reproduction: snapshot A (oldest rows) fails
        replay while snapshot B (middle rows) stays claimed; later rows
        are then journalled and everything replays once the database
        recovers.  Replay produced ``middle, old, new`` — because the
        failed snapshot was RESTORED to the live sidecar, received the
        later append, and its last-append mtime then postdated B's.
        The order must be ``old, middle, new``."""
        task_id, _chat = persistence._add_task("round2 ordering task")

        _journal(task_id, "old", 1.0)
        _claim_all()  # snapshot A — a replayer claimed it, then failed
        _journal(task_id, "middle", 2.0)
        snapshots = _claim_all()  # snapshot B joins A
        assert len(snapshots) == 2
        # Deterministic, strictly ordered mtimes (A older than B), so
        # the pre-fix (mtime, path) sort is reproducible rather than
        # dependent on filesystem timestamp granularity.
        now = time.time()
        for index, snap in enumerate(sorted(snapshots, key=os.path.getmtime)):
            os.utime(snap, (now - 10 + index, now - 10 + index))

        with _database_refusing_writes():
            persistence._replay_failed_events()  # every replay fails

        _journal(task_id, "new", 3.0)  # later rows, after the failure

        persistence._replay_failed_events()

        assert _event_texts(task_id) == ["old", "middle", "new"], (
            "journal snapshots replayed out of order after a failed "
            "replay followed by later appends: newer events got lower "
            "seqs than older ones"
        )

    def test_equal_mtime_snapshots_replay_in_claim_order(
        self, journal_home: Path,
    ) -> None:
        """Eight snapshots whose mtimes all TIE (coarse-resolution or
        network filesystems) must still replay in the order they were
        claimed — the pre-fix ``(mtime, path)`` key degenerated to
        random ``<pid>-<uuid>`` pathname order."""
        task_id, _chat = persistence._add_task("round2 tie ordering task")
        texts = [f"evt-{index}" for index in range(8)]
        claimed: list[str] = []
        for index, text in enumerate(texts):
            _journal(task_id, text, float(index))
            new_claims = set(_claim_all()) - set(claimed)
            assert len(new_claims) == 1
            claimed.append(new_claims.pop())
        stamp = time.time()
        for snap in claimed:
            os.utime(snap, (stamp, stamp))

        persistence._replay_failed_events()

        assert _event_texts(task_id) == texts, (
            "equal-mtime snapshots replayed in random pathname order "
            "instead of claim order"
        )

    def test_claim_keys_stay_monotonic_past_a_clock_step(
        self, journal_home: Path,
    ) -> None:
        """A snapshot claimed under a wall clock that then stepped
        BACKWARDS carries a claim key in the future; later claims must
        still receive larger keys, or the older snapshot would replay
        after them."""
        task_id, _chat = persistence._add_task("round2 clock step task")
        sidecar = Path(
            persistence._failed_events_path(str(persistence._DB_PATH)),
        )
        future_key = time.time_ns() + 10**14  # ~28 hours ahead
        older = sidecar.with_name(
            f"{sidecar.name}.consumed-{future_key:020d}-77-abcd",
        )
        older.write_text(
            json.dumps({
                "task_id": task_id,
                "event_json": json.dumps({"type": "text", "text": "first"}),
                "timestamp": 1.0,
                "origin_db_path": str(persistence._DB_PATH),
            }) + "\n",
            encoding="utf-8",
        )
        _journal(task_id, "second", 2.0)

        persistence._replay_failed_events()

        assert _event_texts(task_id) == ["first", "second"], (
            "a claim key taken from the stepped-back wall clock sorted "
            "the newer snapshot before the older one"
        )


# ---------------------------------------------------------------------------
# Finding 3: worktree setup's baseline capture vs. the cross-process flock
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


def _reclaim_lock_path(repo: Path) -> Path:
    """Return the repo's ``kiss-reclaim.lock`` path."""
    common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    common_dir = Path(common)
    if not common_dir.is_absolute():
        common_dir = (repo / common_dir).resolve()
    return common_dir / "kiss-reclaim.lock"


_FLOCK_HOLDER_SCRIPT = """
import os
import sys
import time

from kiss.core.file_lock import lock_exclusive, unlock

lock_path, held_marker, release_marker = sys.argv[1], sys.argv[2], sys.argv[3]
handle = open(lock_path, "a+")
lock_exclusive(handle)
open(held_marker, "w").close()
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    if os.path.exists(release_marker):
        break
    time.sleep(0.02)
unlock(handle)
handle.close()
"""


class TestSetupWorktreeHoldsCrossProcessLock:
    """Finding 3: baseline capture must hold ``kiss-reclaim.lock``."""

    def test_pooled_spare_setup_blocks_while_flock_held(self) -> None:
        """With a real pooled spare available (the fast path, which
        pre-fix took NO flock at all), ``_try_setup_worktree`` must
        block while a second OS process holds ``kiss-reclaim.lock`` —
        that peer may be mid stash/checkout/merge on the shared main
        tree, and capturing the branch tip and dirty state meanwhile
        reads a torn snapshot."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            try:
                assert worktree_pool.prewarm(repo, None), "no pooled spare"
                (repo / "dirty.txt").write_text("uncommitted user edit\n")
                held = Path(tmp) / "held"
                release = Path(tmp) / "release"
                holder = subprocess.Popen(
                    [
                        sys.executable, "-c", _FLOCK_HOLDER_SCRIPT,
                        str(_reclaim_lock_path(repo)), str(held), str(release),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    deadline = time.monotonic() + 30
                    while not held.exists():
                        assert time.monotonic() < deadline, "holder never locked"
                        time.sleep(0.02)

                    agent = WorktreeSorcarAgent("review0911r2-setup-lock")
                    done = threading.Event()
                    result: list[Path | None] = []

                    def setup() -> None:
                        result.append(agent._try_setup_worktree(repo, None))
                        done.set()

                    runner = threading.Thread(target=setup, daemon=True)
                    runner.start()
                    assert not done.wait(timeout=2.0), (
                        "_try_setup_worktree captured the main tree's "
                        "branch and dirty state while another process "
                        "held the cross-process kiss-reclaim.lock"
                    )

                    release.write_text("go")
                    assert done.wait(timeout=120), "setup never finished"
                    runner.join(timeout=10)
                finally:
                    release.write_text("go")
                    out, err = holder.communicate(timeout=120)
                    assert holder.returncode == 0, f"holder failed: {out}\n{err}"

                assert result and result[0] is not None
                assert agent._wt is not None
                assert (agent._wt.wt_dir / "dirty.txt").exists(), (
                    "dirty state was not captured into the worktree"
                )
            finally:
                worktree_pool.discard_all()

    def test_uncontended_setup_does_not_self_deadlock(self) -> None:
        """The slow path (empty pool) calls reclaim / sweep / create,
        which each take the flock themselves; with the capture now
        wrapping them in the same flock, the nested takes must
        re-enter, not deadlock on a second file descriptor."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            try:
                (repo / "dirty.txt").write_text("uncommitted user edit\n")
                agent = WorktreeSorcarAgent("review0911r2-setup-reentry")
                done = threading.Event()
                result: list[Path | None] = []

                def setup() -> None:
                    result.append(agent._try_setup_worktree(repo, None))
                    done.set()

                runner = threading.Thread(target=setup, daemon=True)
                runner.start()
                assert done.wait(timeout=120), (
                    "_try_setup_worktree deadlocked taking the "
                    "kiss-reclaim.lock flock it already holds"
                )
                runner.join(timeout=10)
                assert result and result[0] is not None
                assert agent._wt is not None
                assert (agent._wt.wt_dir / "dirty.txt").exists()
            finally:
                worktree_pool.discard_all()
