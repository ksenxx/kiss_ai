# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SQLite persistence for task history, chat events, model and file usage.

All data is stored in a single SQLite database, ``history.db`` in the
KISS home directory (``$KISS_HOME`` or the brand's home, e.g.
``~/.s10s`` for Seamless Loop), using WAL mode for concurrent access.
Its tables hold task history, per-chat summaries, chat events, model
and file usage counters, frequent tasks, text typed into running tasks,
tombstones of deleted tasks and chats, and the replayed event journals.

Thread safety is achieved with:
- **Per-thread connections** via ``threading.local()`` so concurrent
  threads never share a Python ``sqlite3.Connection`` object (which
  avoids cursor-state interference).
- A **read-write lock** (``_rw_lock``) that allows concurrent readers
  but gives writers exclusive access, matching SQLite's own WAL
  constraint of at most one writer at a time.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, TypeGuard, cast

from kiss.agents.sorcar._concurrency import _race_delay, locked, locked_call
from kiss.agents.sorcar.chat_summary import NOT_SUBAGENT, launch_ms, upsert_chat_summary
from kiss.agents.sorcar.history_columns import (
    ADDED_SUMMARY_COLUMNS,
    ADDED_TASK_COLUMNS,
    CHAT_TABLE,
    TASK_TABLE,
    add_missing_columns,
    complete_rows,
    missing_columns,
)
from kiss.agents.sorcar.task_metadata import trajectory_has_secret, write_task_tags
from kiss.core.config import adopt_legacy_file, is_pre_2026_04_db, kiss_home
from kiss.core.file_lock import lock_exclusive, unlock
from kiss.core.locked import LockTimeoutError
from kiss.core.models.model_info import _strip_provider_prefix
from kiss.core.printer import event_cost

logger = logging.getLogger(__name__)


@contextmanager
def _immediate_txn(db: sqlite3.Connection) -> Iterator[None]:
    """Run a read-modify-write sequence as ONE atomic transaction.

    Connections are opened with ``isolation_level=None`` (autocommit),
    so without an explicit transaction every statement of a
    read-modify-write sequence commits on its own and another PROCESS
    can interleave between them — ``_rw_lock`` is a ``threading``
    primitive and provides no cross-process exclusion.  ``BEGIN
    IMMEDIATE`` takes SQLite's write lock for the whole block, which
    every process on the database respects.

    The ``BEGIN`` runs inside the ``try``: a Stop injected on the task
    thread lands at the bytecode boundary right after the statement
    returned, and with the BEGIN outside the handler the interrupt
    escaped with the transaction still open on this thread's
    connection — SQLite's write lock then stayed held for the
    connection's life and every other thread's next write sat in the
    30 s ``busy_timeout`` ("a peer hung on a leaked lock" in the
    persistence hammer).  ``in_transaction`` tells whether the BEGIN
    took effect before the interrupt, so nothing is rolled back that
    was never begun.  The ``COMMIT`` is inside too: SQLite keeps the
    transaction open after a ``COMMIT`` that failed with ``BUSY``
    (possible in rollback-journal mode while another connection
    reads), and the connection would otherwise answer every later
    ``BEGIN`` with "cannot start a transaction within a transaction".

    Args:
        db: The connection to run the transaction on.

    Yields:
        ``None`` — run the statements inside the ``with`` block.
    """
    try:
        db.execute("BEGIN IMMEDIATE")
        yield
        db.execute("COMMIT")
    except BaseException:
        try:
            if db.in_transaction:  # raises too once the connection is closed
                db.execute("ROLLBACK")
        except sqlite3.Error:  # a closed connection: nothing left to roll back
            pass
        raise


class _Token:
    """Per-acquisition ownership token for :class:`_RWLock`.

    A plain ``object()`` cannot be weak-referenced; this can.  The
    acquisition's generator frame holds the primary strong reference,
    so in the common case the token dies with the frame and the lock's
    weakref machinery observes the release.

    Frames, however, can outlive their acquisition: an exception that
    escapes the acquisition (e.g. an injected stop whose unwind skipped
    the ``finally``) carries a traceback that retains the generator's
    frame — and with it the token — for as long as any handler or
    stored exception keeps the traceback alive.  ``gen`` therefore
    records a WEAK reference to the acquisition's own generator: once
    that generator is collected, or is CLOSED (``gi_frame is None`` no
    matter who still retains the frame object),
    :meth:`_RWLock._token_active` pronounces the acquisition over.  The
    reference must be weak, and the frame must not hold its own
    generator either (the generator pops itself out of the ``box`` it
    arrives in): either would close a cycle through the frame, and an
    acquisition leaked WITHOUT ``__exit__`` (an injected stop can land
    between ``__enter__`` returning and the ``with`` block installing,
    orphaning the context manager) would then block other threads until
    a garbage collection, instead of dying by refcount, on its own
    thread, the moment its context manager is released.

    ``owner`` records the acquiring thread.  A dead exception↔frame
    reference CYCLE (the stop's traceback in a killed thread) can keep
    even the weakly-referenced generator alive until a gc pass, so
    liveness additionally requires the owning thread to still be alive
    — the one fact about an abandoned acquisition that neither
    refcounting nor gc timing can misreport.
    """

    __slots__ = ("gen", "owner", "__weakref__")

    def __init__(self) -> None:
        self.gen: weakref.ref[Any] | None = None
        self.owner: threading.Thread = threading.current_thread()


class _TokenRef(weakref.ref):  # type: ignore[type-arg]
    """Weak reference to a :class:`_Token`, hashed and compared by identity.

    A plain ``weakref.ref`` derives its hash from the referent and caches
    it on first use; ``set.discard`` on a DEAD ref whose hash was never
    computed raises ``TypeError: weak object has gone away``.  An
    injected stop landing between the ref's creation and its ``set.add``
    produces exactly that ref, and its callback — the lock's healing
    path — would then die as an unraisable exception.  Identity hashing
    never touches the referent, so the callback is safe at every
    boundary.  ``__ne__`` is overridden too: ``weakref.ref.__ne__``
    compares referents, so without it two distinct refs to one live
    token would be neither ``==`` nor ``!=``.
    """

    __slots__ = ()
    __hash__ = object.__hash__
    __eq__ = object.__eq__
    __ne__ = object.__ne__


class _RWLock:
    """Writer-preferring read-write lock, safe against injected stops.

    Multiple readers can hold the lock concurrently.  A writer gets
    exclusive access — no readers or other writers may proceed while a
    write lock is held.  Pending writers block new readers to prevent
    writer starvation.

    Not reentrant, in either mode.  A thread holding a read token that
    takes a second one while a writer is pending waits for that writer,
    which waits for the first token: deadlock.  Never call a locking
    helper from inside a ``read_lock()`` / ``write_lock()`` block: a
    helper that runs under its caller's lock takes the connection as
    an argument and says so in its docstring.

    Interrupt safety.  The server stops a task by injecting
    ``KeyboardInterrupt`` at an ARBITRARY bytecode boundary
    (``PyThreadState_SetAsyncExc``) — including between a C-level
    ``acquire()`` returning and the very next bytecode, and including
    the boundary at a ``finally`` block's entry, where the injection
    makes the interpreter skip the block's body entirely.  No
    Python-level bookkeeping that must EXECUTE after the stop can
    therefore be trusted; this lock is instead built so that every
    stranded acquisition heals without its own code running:

    * Ownership lives in per-acquisition :class:`_Token` objects held
      by the acquisition's frame and tracked here through weak
      references.  The wait predicates ignore tokens that are dead OR
      whose acquisition generator has been closed
      (:meth:`_token_active`): an escaped exception's traceback may
      retain the acquisition frame — and so the token — indefinitely,
      but the closed generator's ``gi_frame`` is ``None`` from the
      moment the exception left it, so a stranded acquisition stops
      blocking others within one bounded ``wait`` tick even when the
      token object itself never dies.  Each weakref's callback also
      re-checks waiters on token death, and the bounded ``wait``
      timeout below covers a lost callback or a still-referenced
      closed frame.
    * Registration and withdrawal are SINGLE atomic C operations
      (``set.add`` / ``set.discard`` / one ``STORE_ATTR``), each
      idempotently reversible without knowing whether it executed, so
      no state/flag pairing exists to tear.
    * The condition's mutex is an ``RLock``: its C ``acquire`` records
      the owning thread and recursion level atomically, so
      :meth:`_teardown` consults ``RLock._recursion_count()`` — ground
      truth, not a flag a torn store could miss — and releases every
      level this thread still holds, except those of a critical
      section a garbage collection interrupted to finalize an orphaned
      acquisition.  Being reentrant, teardown can always re-enter the
      mutex even when the injection landed while it was already held
      (a plain ``Lock`` self-deadlocks there).  Teardown retries a
      bounded number of times, re-raising a newly injected exception
      only AFTER the repair committed.

    Residual window, deliberate and documented: the mutex itself can
    only be stranded if one injection interrupts an acquisition WHILE
    the mutex is held and a SECOND, separately timed injection then
    lands exactly on the unwinding ``finally``'s entry boundary —
    skipping the teardown that would release it.  A single stop
    (``task_runner.inject_keyboard_interrupt`` performs one injection
    per attempt) can never wedge the lock, and the random-time stress
    regression (hundreds of real injections over ~10⁵ acquisitions)
    shows no deadlock.
    """

    _REPAIR_ATTEMPTS = 8
    # Upper bound on how long a waiter can oversleep a wakeup whose
    # notify was lost (possible only when an injected stop lands inside
    # a weakref callback); predicates re-check liveness on every tick.
    _WAIT_RECHECK_S = 1.0

    def __init__(self) -> None:
        self._mutex = threading.RLock()
        self._cond = threading.Condition(self._mutex)
        self._reader_refs: set[weakref.ref[_Token]] = set()
        self._pending_refs: set[weakref.ref[_Token]] = set()
        self._writer_ref: weakref.ref[_Token] | None = None

    @staticmethod
    def _token_active(ref: weakref.ref[_Token]) -> bool:
        """Return True while *ref*'s acquisition is genuinely in progress.

        Three conditions must hold: the token object is still alive,
        its acquisition generator is still alive, and that generator
        has not been closed.  The ``gi_frame`` check heals the
        traceback-retention case — an escaped exception keeps the
        acquisition frame (and token) reachable, but the generator's
        ``gi_frame`` drops to ``None`` the moment the exception left
        it, independent of garbage collection.  The generator weakref
        heals the orphaned-context-manager case — a stop landing
        between ``__enter__`` and the ``with`` block installation
        leaks a SUSPENDED generator, which then dies by refcount with
        its context manager.  A token without a recorded generator is
        treated as active (fail-closed).

        Args:
            ref: The acquisition's weak reference.

        Returns:
            Whether the acquisition still owns or awaits the lock.
        """
        token = ref()
        if token is None:
            return False
        if not token.owner.is_alive():
            # The acquiring thread is gone; whatever retains the token
            # (a dead thread's exception↔frame cycle awaiting gc) no
            # longer represents a live acquisition.
            return False
        gen_ref = token.gen
        if gen_ref is None:
            return True
        gen = gen_ref()
        return gen is not None and gen.gi_frame is not None

    def _writer_alive(self) -> bool:
        """Return True when a live writer token holds exclusive access."""
        ref = self._writer_ref
        return ref is not None and self._token_active(ref)

    @classmethod
    def _any_alive(cls, refs: set[weakref.ref[_Token]]) -> bool:
        """Return True when any weakref in *refs* has an active token.

        Walks a snapshot: a garbage collection during the walk can run a
        token's weakref callback, or an orphaned acquisition's teardown,
        on this thread, and both discard from *refs*.
        """
        return any(cls._token_active(ref) for ref in tuple(refs))

    def _drop_reader_ref(self, ref: weakref.ref[_Token]) -> None:
        """Withdraw a reader registration and wake waiters.

        Doubles as the token's weakref callback: it runs even when an
        injected stop skipped the acquisition's own teardown, as soon
        as the acquisition frame — the only strong token holder — is
        destroyed.

        Args:
            ref: The reader acquisition's weak reference.
        """
        with self._cond:
            self._reader_refs.discard(ref)
            self._cond.notify_all()

    def _drop_writer_ref(self, ref: weakref.ref[_Token]) -> None:
        """Withdraw a writer registration (pending or holding) and wake
        waiters; also the writer token's weakref callback.

        Args:
            ref: The writer acquisition's weak reference.
        """
        with self._cond:
            self._pending_refs.discard(ref)
            if self._writer_ref is ref:
                self._writer_ref = None
            self._cond.notify_all()

    def read_lock(self) -> AbstractContextManager[None]:
        """Acquire shared read access (use as ``with lock.read_lock():``).

        See the class docstring for why an injected stop landing at
        any bytecode boundary here cannot wedge the lock.  The box
        dance hands the acquisition generator to its own token BEFORE
        any registration, so :meth:`_token_active` can pronounce the
        acquisition over (``gi_frame is None``) even when an escaped
        exception's traceback retains the frame and token forever.
        The generator pops itself out of the box, so its frame keeps no
        strong reference to it and an orphaned acquisition dies with
        its context manager.

        Returns:
            A context manager holding read access for its block.
        """
        box: list[Any] = []
        cm = self._read_lock_gen(box)
        box.append(cm.gen)
        return cm

    @contextmanager
    def _read_lock_gen(self, box: list[Any]) -> Iterator[None]:
        """Generator body of :meth:`read_lock`; *box* carries its generator."""
        token = _Token()
        token.gen = weakref.ref(box.pop()) if box else None
        ref = _TokenRef(token, self._drop_reader_ref)
        try:
            self._cond.acquire()
            while self._writer_alive() or self._any_alive(self._pending_refs):
                self._cond.wait(timeout=self._WAIT_RECHECK_S)
            self._reader_refs.add(ref)
            self._cond.release()
            yield
        finally:
            self._teardown(ref, token, writer=False)

    def write_lock(self) -> AbstractContextManager[None]:
        """Acquire exclusive write access (use as ``with lock.write_lock():``).

        See the class docstring for the interrupt-safety design.  The
        writer publication (``_writer_ref = ref``) happens before the
        pending-ref withdrawal; an injection between the two leaves a
        state (writer + still pending) that only blocks others
        conservatively until :meth:`_teardown` — idempotent for both
        pieces — or the token's deactivation repairs it.  As in
        :meth:`read_lock`, the token records its own acquisition
        generator so traceback retention cannot prolong ownership.

        Returns:
            A context manager holding write access for its block.
        """
        box: list[Any] = []
        cm = self._write_lock_gen(box)
        box.append(cm.gen)
        return cm

    @contextmanager
    def _write_lock_gen(self, box: list[Any]) -> Iterator[None]:
        """Generator body of :meth:`write_lock`; *box* carries its generator."""
        token = _Token()
        token.gen = weakref.ref(box.pop()) if box else None
        ref = _TokenRef(token, self._drop_writer_ref)
        try:
            self._cond.acquire()
            self._pending_refs.add(ref)
            while self._writer_alive() or self._any_alive(self._reader_refs):
                self._cond.wait(timeout=self._WAIT_RECHECK_S)
            self._writer_ref = ref
            self._pending_refs.discard(ref)
            self._cond.release()
            yield
        finally:
            self._teardown(ref, token, writer=True)

    def _teardown(
        self, ref: weakref.ref[_Token], token: _Token, writer: bool,
    ) -> None:
        """Withdraw *ref* from every piece of lock state and wake waiters.

        Every operation is idempotent, so it is safe regardless of how
        far the acquisition got before an injected stop unwound it.
        The bounded retry loop absorbs FURTHER exceptions injected
        while the repair itself runs; the last one is re-raised once
        the repair has committed (never swallowed: a stop aimed at the
        surrounding task must still reach it).

        A ``GeneratorExit`` in flight means ``close()`` is finalizing an
        orphaned acquisition suspended at its ``yield``, which holds no
        mutex level of its own.  That finalizer can run inside a garbage
        collection on any thread, in the middle of another critical
        section of this lock, so the levels found on entry stay held.
        Every other teardown runs on the acquiring thread outside any
        critical section, and every level it finds is released.

        Args:
            ref: The acquisition's weak reference to withdraw.
            token: The acquisition's token; the explicit parameter
                keeps it alive so its weakref callback — the redundant
                healing path — cannot run concurrently with this one.
            writer: Whether the acquisition was a write acquisition.
        """
        keep = (
            self._mutex._recursion_count()  # type: ignore[attr-defined]
            if isinstance(sys.exception(), GeneratorExit)
            else 0
        )
        caught: BaseException | None = None
        for _ in range(self._REPAIR_ATTEMPTS):
            try:
                try:
                    if writer:
                        self._drop_writer_ref(ref)
                    else:
                        self._drop_reader_ref(ref)
                    break
                finally:
                    # Ground truth from the C RLock: releases every
                    # level above ``keep`` this thread still holds —
                    # whether the injection landed before, inside, or
                    # after the acquisition's own release.
                    while self._mutex._recursion_count() > keep:  # type: ignore[attr-defined]
                        self._cond.release()
            except BaseException as exc:  # second injected stop mid-repair
                caught = exc
        if caught is not None:
            raise caught


_rw_lock = _RWLock()

_init_tables_lock = threading.RLock()

# Serializes a task's usage finalization against spend charged to it
# from another thread: a run's final "read the ledger, save the row with
# its end stamp" and its last ``usage_info`` on one side, a late charge's
# "is the row finished? then add to the row, else bank on the live
# ledger and publish" on the other (``charge_side_channel_usage``).
# Without it a charge could check the row (unfinished), lose the CPU to
# the final save, then bank on a ledger nothing reads again — or publish
# newer totals that the run's stale last event then overwrites.  Held
# only around those short DB/ledger/printer sections, never across a
# join or a model call.  Task threads take it through
# :func:`kiss.agents.sorcar._concurrency.locked_call` (hence the
# reentrant type: it is never taken reentrantly, but a Stop interrupt
# landing right after the acquisition must be able to release it).
TASK_USAGE_LOCK = threading.RLock()


_chat_context_text_cache: dict[str, str] = {}
_chat_context_cache_lock = threading.RLock()
_chat_context_cache_gen: int = 0


@locked(_chat_context_cache_lock)
def _invalidate_chat_context_cache(chat_id: str = "") -> None:
    """Drop the cached chat-context text for *chat_id*.

    When *chat_id* is empty, the entire cache is cleared (used by test
    fixtures that swap the underlying database file).
    """
    global _chat_context_cache_gen
    if chat_id:
        _chat_context_text_cache.pop(chat_id, None)
    else:
        _chat_context_text_cache.clear()
    _chat_context_cache_gen += 1


_KISS_DIR = kiss_home()
_DB_PATH = _KISS_DIR / "history.db"


def _current_db_path() -> str:
    """Return the active database path as a string.

    Used by asynchronous producers (the background event writer's
    enqueue path, the daemon's fire-and-forget follow-up
    thread) to stamp each pending write with the database it was
    produced against, so a late write can never land in a *different*
    database after ``_DB_PATH`` has been reassigned (test fixtures,
    daemon restarts pointed at another home dir).  Numeric
    ``task_history`` ids are only unique within one database file —
    AUTOINCREMENT prevents reuse inside a database, but a swapped
    database restarts the counter, so a stale id would otherwise
    resolve to an unrelated task's row.
    """
    return str(_DB_PATH)

_MAX_FILE_USAGE_ENTRIES = 10000

_MAX_FREQUENT_TASKS = 100

# Cap on the ``steer_inputs`` table (messages typed into a running
# task's composer, kept only so autocomplete can offer them again).
_MAX_STEER_INPUTS = 1000

# Cap on the composer's ArrowUp history sent to every (re)connecting
# client.  Unbounded, a long-lived install ships every prompt ever
# typed (tens of thousands of rows, megabytes) on each connect.
_MAX_INPUT_HISTORY = 500

_OWNER_DIR_NAME = "task-owners"

_owner_state: tuple[str, str, IO[Any]] | None = None
_owner_state_lock = threading.RLock()


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


def _owner_dir() -> Path:
    """Return the directory holding one liveness marker per process."""
    return _KISS_DIR / _OWNER_DIR_NAME


@locked(_owner_state_lock)
def _process_owner_token() -> str:
    """Return this process's owner token, publishing its liveness marker.

    The database is shared by every Sorcar process on the machine (the
    ``kiss-web`` daemon, a ``kiss`` CLI run, a daemon restart), so
    "is the task that wrote this row still running?" cannot be answered
    from Python memory.  Each process therefore creates
    ``<KISS_HOME>/task-owners/<token>.lock`` and holds an exclusive
    ``flock`` on it for its whole lifetime: the kernel releases that
    lock when the process dies, however violently, so any other
    process can test liveness by trying to take the lock.

    The token is re-minted when ``_KISS_DIR`` is redirected (test
    fixtures, a daemon pointed at another home) so the marker always
    lives next to the database the rows are written to; the marker of
    the home being left is deleted, and so is this process's marker
    when the interpreter exits normally (:func:`_release_owner_marker`).
    Without that, every daemon and CLI lifecycle would leave one file
    behind forever, since a marker is otherwise only unlinked when
    some *other* process happens to test a sentinel row of the dead
    owner.

    Returns:
        The token to store in ``task_history.owner``, or ``""`` when
        the marker could not be created (liveness then degrades to the
        previous timestamp-only heuristic).
    """
    global _owner_state
    current_dir = str(_owner_dir())
    if _owner_state is not None and _owner_state[0] == current_dir:
        return _owner_state[1]
    _discard_owner_state()
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    marker = Path(current_dir) / f"{token}.lock"
    handle: IO[Any] | None = None
    try:
        _ensure_kiss_dir()
        Path(current_dir).mkdir(parents=True, exist_ok=True)
        handle = open(marker, "w", encoding="utf-8")
        if not lock_exclusive(handle, blocking=False):  # pragma: no cover
            raise BlockingIOError(f"{token}.lock is already held")
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        logger.warning(
            "could not publish task-owner liveness marker", exc_info=True,
        )
        if handle is not None:
            # No row will ever name this token, so nothing else would
            # delete the file.
            _delete_owner_marker(handle, marker)
        return ""
    _owner_state = (current_dir, token, handle)
    return token


def _delete_owner_marker(handle: IO[Any], marker: Path) -> None:
    """Close *handle*, releasing its kernel lock, and delete *marker*."""
    try:
        handle.close()
    except OSError:  # a publish whose flush failed fails again on close
        pass
    try:
        marker.unlink()
    except OSError:  # pragma: no cover — already swept by another process
        pass


def _discard_owner_state() -> None:
    """Close and delete this process's current liveness marker.

    Caller holds :data:`_owner_state_lock` (or is the ``atexit`` hook,
    which takes it).  Deleting the file is what keeps
    ``task-owners/`` from growing by one entry per process lifetime;
    the kernel lock is released by the close either way.
    """
    global _owner_state
    if _owner_state is None:
        return
    directory, token, handle = _owner_state
    _owner_state = None
    _delete_owner_marker(handle, Path(directory) / f"{token}.lock")


def _release_owner_marker() -> None:
    """``atexit`` hook: drop this process's liveness marker on exit."""
    locked_call(_owner_state_lock, _discard_owner_state)


atexit.register(_release_owner_marker)


def _owner_is_alive(token: str) -> bool:
    """Return True when the process that recorded *token* is still running.

    A missing marker file, or one whose ``flock`` can be taken, means
    the owning process is gone.  Stale markers are unlinked as they are
    discovered so ``task-owners/`` cannot grow without bound.

    Args:
        token: The value of a row's ``owner`` column.  ``""`` (legacy
            rows written before owner tracking) reports not alive.

    Returns:
        True when the owning process still holds its marker.
    """
    if not token:
        return False
    state = _owner_state  # one read: another thread may reset the global
    if state is not None and state[1] == token:
        return True
    marker = _owner_dir() / f"{token}.lock"
    try:
        with open(marker, "r+", encoding="utf-8") as handle:
            if not lock_exclusive(handle, blocking=False):
                return True
            unlock(handle)
    except OSError:
        return False
    try:
        marker.unlink()
    except OSError:  # pragma: no cover — concurrent sweep won the unlink
        pass
    return False


_HistoryEntry = dict[str, object]


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce *value* to ``int``, returning *default* on failure.

    Non-finite floats (NaN/Inf) yield *default* rather than raising
    ``OverflowError``.  Any object whose ``__eq__`` raises an
    arbitrary exception is treated as the default rather than
    propagating — this keeps the task-completion finally robust
    against caller-supplied misbehaving objects.
    """
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)  # type: ignore[arg-type, call-overload, no-any-return]
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to ``float``, returning *default* on failure.

    Non-finite floats (NaN/Inf) yield *default* so the value never
    leaks into a JSON-serialised payload that would break SQLite's
    ``json_valid``.  Any object whose ``__eq__`` raises an arbitrary
    exception is treated as the default rather than propagating.
    """
    try:
        if value is None or value == "":
            return default
        result = float(value)  # type: ignore[arg-type]
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _safe_str(value: object, default: str = "") -> str:
    """Coerce *value* to ``str`` for JSON-bound payloads.

    bughunt8: SQLite's dynamic typing lets a hand-edited /
    3rd-party-source DB store a BLOB in a TEXT column, and
    ``json.dumps`` raises ``TypeError`` on ``bytes`` — uncaught by
    ``_dumps_extra``'s ``ValueError`` handler — which blanked the
    whole history sidebar over one corrupt row.  BLOBs are decoded as
    UTF-8 with replacement so the result is always JSON-serialisable;
    other non-string scalars go through ``str()``.  A non-finite float
    or a ``__str__`` that raises yields *default* so one bad field in a
    ``_save_task_extra`` payload never aborts the whole UPDATE.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return default
    try:
        return str(value)
    except Exception:
        return default


def _sanitize_non_finite(value: object) -> object:
    """Recursively replace non-finite floats (NaN/±Inf) with ``None``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(v) for v in value]
    return value


def _dumps_extra(extra: dict[str, object]) -> str:
    """JSON-encode *extra* guaranteeing valid RFC 8259 output.

    Non-finite floats anywhere in the payload are replaced with
    ``None`` — plain ``json.dumps`` would serialise them as the bare
    ``NaN``/``Infinity`` tokens, which SQLite's ``json_valid`` (used by
    ``NOT_SUBAGENT``) rejects.

    Args:
        extra: Metadata dict to serialise.

    Returns:
        A valid JSON object string.
    """
    try:
        return json.dumps(extra, allow_nan=False)
    except ValueError:
        sanitized = _sanitize_non_finite(extra)
        return json.dumps(sanitized, allow_nan=False)


_db_conn: sqlite3.Connection | None = None

_thread_local = threading.local()
_db_generation: int = 0


def _bump_db_generation() -> None:
    """Retire every thread's cached connection lazily (caller holds ``_init_tables_lock``).

    The two bumpers, sidecar recovery and :func:`_close_db`, both run
    this under the same lock, so no bump is ever lost.  ``_get_db``
    reads the counter without the lock (a stale read only costs one
    extra reconnect).
    """
    global _db_generation
    _db_generation += 1


class _LockedCursor(sqlite3.Cursor):
    """Cursor of a :class:`_LockedConnection`.

    Every call that steps the statement (``execute``, the ``fetch*``
    family, iteration) runs under the connection's lock, so the
    connection is never closed from another thread while this cursor
    is inside ``sqlite3_step``.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.lock = cast(_LockedConnection, connection).lock

    def execute(self, sql: str, parameters: Any = (), /) -> _LockedCursor:
        """Run *sql* with *parameters* under the connection's lock."""
        locked_call(self.lock, super().execute, sql, parameters)
        return self

    def executemany(self, sql: str, seq_of_parameters: Any, /) -> _LockedCursor:
        """Run *sql* once per parameter set under the connection's lock."""
        locked_call(self.lock, super().executemany, sql, seq_of_parameters)
        return self

    def fetchone(self) -> Any:
        """Return the next row, or ``None``, stepping under the lock."""
        return locked_call(self.lock, super().fetchone)

    def fetchmany(self, size: int | None = None) -> list[Any]:
        """Return up to *size* (default ``arraysize``) rows, stepping under the lock."""
        return locked_call(
            self.lock, super().fetchmany, self.arraysize if size is None else size,
        )

    def fetchall(self) -> list[Any]:
        """Return every remaining row, stepping under the lock."""
        return locked_call(self.lock, super().fetchall)

    def executescript(self, sql_script: str, /) -> _LockedCursor:
        """Run *sql_script* under the connection's lock."""
        locked_call(self.lock, super().executescript, sql_script)
        return self

    def __next__(self) -> Any:
        return locked_call(self.lock, super().__next__)


class _LockedConnection(sqlite3.Connection):
    """``sqlite3.Connection`` whose statement steps run under ``self.lock``.

    Connections are opened with ``check_same_thread=False`` and cached
    per thread, and :func:`_recover_orphaned_sidecars` has to close
    handles that belong to OTHER threads (SQLite drops a dead ``-shm``
    mapping only when the last connection using it closes).  Closing a
    ``sqlite3.Connection`` while its owner is inside ``sqlite3_step``
    is a use-after-free in CPython, so every step on the connection —
    and :meth:`close` — holds a reentrant lock: a handle whose lock can
    be taken is idle and safe to close from any thread (its owner's
    next statement raises ``ProgrammingError`` instead of crashing);
    one whose lock is held is left to its owner.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def cursor(self, factory: Any = None) -> Any:
        """Return a :class:`_LockedCursor` (or a *factory* cursor)."""
        return super().cursor(_LockedCursor if factory is None else factory)

    def execute(self, sql: str, parameters: Any = (), /) -> _LockedCursor:
        """Run *sql* on a new locked cursor and return it."""
        cursor: _LockedCursor = self.cursor()
        return cursor.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any, /) -> _LockedCursor:
        """Run *sql* once per parameter set on a new locked cursor."""
        cursor: _LockedCursor = self.cursor()
        return cursor.executemany(sql, seq_of_parameters)

    def executescript(self, sql_script: str, /) -> _LockedCursor:
        """Run *sql_script* on a new locked cursor."""
        cursor: _LockedCursor = self.cursor()
        return cursor.executescript(sql_script)

    def commit(self) -> None:
        """Commit the open transaction under the lock."""
        locked_call(self.lock, super().commit)

    def rollback(self) -> None:
        """Roll back the open transaction under the lock."""
        locked_call(self.lock, super().rollback)

    def close(self) -> None:
        """Close the connection; waits for a statement in flight on it."""
        locked_call(self.lock, super().close)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
        /,
    ) -> Literal[False]:
        # ``with conn:`` commits or rolls back from C, bypassing the
        # Python overrides above, so the lock is taken here.
        return locked_call(self.lock, super().__exit__, exc_type, exc, tb)


#: Every connection :func:`_get_db` has opened in this process and not
#: yet closed, keyed by ``id(conn)`` -> ``(conn, owning thread, db
#: path)``.  ``sqlite3.Connection`` cannot be weakly referenced, so a
#: connection whose owning thread died without ``_close_thread_db()``
#: is closed and dropped by :func:`_prune_dead_thread_conns` instead of
#: being pinned here forever.  Guarded by ``_init_tables_lock``.  Exists
#: for one purpose: the orphaned-sidecar recovery in
#: :func:`_recover_orphaned_sidecars` has to close ALL connections to a
#: database — see :func:`_sidecars_orphaned`.
_open_conns: dict[int, tuple[_LockedConnection, threading.Thread, str]] = {}

#: How long :func:`_recover_orphaned_sidecars` waits, in total, for
#: other threads' statements in flight to finish.  They are interrupted
#: first, so this is normally instant; the bound only matters for a
#: handle queued in SQLite's (uninterruptible) 30 s ``busy_timeout``
#: wait, and it exceeds that wait so the recovery outlasts it rather
#: than failing the task that ran into the deleted sidecars.
_RECOVERY_IDLE_WAIT_S = 35.0

#: Per database path: ``(db_file_id, shm_file_id)`` of the ``-shm``
#: sidecar the connections in this process are mapped to.  An entry
#: exists exactly while at least one registered connection to that
#: path is open (SQLite deletes the sidecars on the last close, so the
#: next connection records a fresh identity).  Guarded by
#: ``_init_tables_lock`` for writes; read without the lock (a stale read
#: only delays detection by one call).
_attached_shm: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}


def _path_has_open_conns(path: str) -> bool:
    """Return whether any registered connection to *path* is open."""
    return any(p == path for _c, _o, p in _open_conns.values())


def _drop_open_conn_locked(key: int) -> None:
    """Forget registry entry *key*; drop the path's ``-shm`` identity if
    it was the last connection to that path.  Caller holds the lock."""
    entry = _open_conns.pop(key, None)
    if entry is not None and not _path_has_open_conns(entry[2]):
        _attached_shm.pop(entry[2], None)


def _prune_dead_thread_conns() -> None:
    """Close and forget connections whose owning thread has exited.

    Caller holds ``_init_tables_lock``.  A thread that ended without
    calling ``_close_thread_db()`` used to leave its connection to the
    garbage collector; the registry pins it, so it is closed here.
    """
    for key, (conn, owner, _path) in list(_open_conns.items()):
        if not owner.is_alive():
            _drop_open_conn_locked(key)
            try:
                conn.close()
            except sqlite3.Error:
                pass


#: One ``_attached_shm`` entry: ``(db_file_id, shm_file_id)``.
_ShmMapping = tuple[tuple[int, int], tuple[int, int]]


def _sidecars_orphaned(
    current_path: str, current_id: tuple[int, int] | None, attached: _ShmMapping,
) -> bool:
    """Return ``True`` when this process is mapped to a dead ``-shm``.

    SQLite maps exactly ONE ``-shm`` per process per database inode —
    shared by every connection in the process — but opens the ``-wal``
    by *name* for each connection.  When something outside this process
    unlinks ``history.db-wal``/``history.db-shm`` (the daemon was found
    holding sixteen descriptors to ``history.db-wal (deleted)``), the
    connections that are already open keep working against a deleted
    inode (so every frame they commit is lost on a hard kill), other
    processes read a database missing those frames, and every NEW
    connection in this process inherits the dead ``-shm`` mapping while
    opening a fresh ``-wal`` — the two disagree and the connection fails
    with ``sqlite3.OperationalError: disk I/O error``
    (``SQLITE_IOERR_SHORT_READ``).  That is the "every new task fails
    within 100 ms" failure.

    The condition is detected by identity: the ``-shm`` on disk (missing,
    or a different inode because another process already recreated it)
    is not the one this process attached to.  A transient ``os.stat``
    failure (:data:`_FILE_ID_UNKNOWN`) proves nothing and is ignored.

    Args:
        current_path: The active database path.
        current_id: Identity of the database file itself (already
            computed by the caller); the recorded ``-shm`` identity only
            applies to connections opened against this same file.
        attached: The :data:`_attached_shm` entry of *current_path* as
            the caller read it; the caller hands the same object to
            :func:`_recover_orphaned_sidecars`.
    """
    db_id, shm_id = attached
    if db_id != current_id:
        return False
    on_disk = _db_file_identity(current_path + "-shm")
    if on_disk is _FILE_ID_UNKNOWN or on_disk == shm_id:
        return False
    # The -shm differs.  Re-stat the database file AFTER the -shm so a
    # concurrent deletion of the whole database (db, then -wal, then
    # -shm — the redirect/delete pattern of the test fixtures) is not
    # mistaken for an orphaned mapping: a deleted or replaced database
    # is handled per thread by the identity check in ``_get_db``.
    return _db_file_identity(current_path) == db_id


def _recover_orphaned_sidecars(current_path: str, observed: _ShmMapping) -> None:
    """Drop the dead ``-shm`` mapping *observed*: checkpoint, close every connection.

    Called by :func:`_get_db` when :func:`_sidecars_orphaned` is true or
    when a brand-new connection failed with ``SQLITE_IOERR``.  While the
    old mapping is still valid, the frames this process committed into
    the deleted ``-wal`` are folded into the main file with a passive
    checkpoint (so they survive), then every open connection to
    *current_path* in the process is interrupted and closed — SQLite
    releases the shared ``-shm`` mapping only when the last connection
    using it closes, so closing just the calling thread's connection
    would leave every new connection failing.  A handle is closed only
    once its :class:`_LockedConnection` lock is held, i.e. while no
    other thread is inside a statement on it (the interrupt makes a
    statement in flight abort promptly): closing it under its owner's
    feet would crash the process.  A handle whose owner keeps it busy
    beyond :data:`_RECOVERY_IDLE_WAIT_S` stays open and the owner
    retires it on its next ``_get_db()`` — the generation counter is
    bumped so every thread's next ``_get_db()`` reconnects — and this
    call raises ``OperationalError`` without reopening anything, since
    a connection opened while an old handle lives joins the dead
    mapping for good; the ``_attached_shm`` entry stays so every
    ``_get_db()`` retries the recovery until the last old handle is
    retired.  A thread whose idle handle was closed here fails once
    with ``ProgrammingError`` on a statement it runs on the old handle.

    Idempotent under ``_init_tables_lock``: a racing thread that finds
    *observed* no longer recorded returns without doing anything — a
    recovery that completed while it queued for the lock dropped that
    mapping, and the entry recorded now (if any) is the live mapping
    the reconnects since then joined, whose handles must stay open.  A
    recovery that gave up on a busy handle, and ``_close_db``, keep the
    entry, so a queued observer of a still-dead mapping does recover it.

    Args:
        current_path: The active database path.
        observed: The :data:`_attached_shm` entry the caller found dead
            (or was mapped to when its open failed).
    """
    if locked_call(_init_tables_lock, _recover_orphaned_sidecars_locked, current_path, observed):
        _invalidate_chat_context_cache("")


def _close_idle_conn(conn: _LockedConnection, checkpointed: bool) -> bool:
    """Close *conn*, a handle on a dead mapping, checkpointing first unless one was.

    The caller holds ``conn.lock``, so no statement runs on the handle.

    Args:
        conn: The handle to close.
        checkpointed: Whether an earlier handle already checkpointed.

    Returns:
        Whether a checkpoint has been taken, by now or before.
    """
    if not checkpointed:
        # Fails with "interrupted" on a handle whose paused statement
        # the caller's interrupt hit; the next idle handle checkpoints
        # instead.
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            checkpointed = True
        except sqlite3.Error:
            logger.debug("Exception caught", exc_info=True)
    try:
        conn.close()
    except sqlite3.Error:
        logger.debug("Exception caught", exc_info=True)
    return checkpointed


def _recover_orphaned_sidecars_locked(current_path: str, observed: _ShmMapping) -> bool:
    """Body of :func:`_recover_orphaned_sidecars`; the caller holds ``_init_tables_lock``.

    Returns:
        True when connections were recovered, False when nothing was to do.
    """
    global _db_conn
    # Identity, not equality: a recreated ``-shm`` may get the deleted
    # one's inode number back, while every re-record stores a new tuple.
    if _attached_shm.get(current_path) is not observed:
        return False
    _prune_dead_thread_conns()
    doomed = [
        (key, conn) for key, (conn, _owner, path) in _open_conns.items()
        if path == current_path
    ]
    logger.error(
        "%s-wal/-shm were deleted or replaced under this process while "
        "%d connection(s) were open; checkpointing and reconnecting "
        "every thread (new connections were failing with 'disk I/O "
        "error' until now)",
        current_path,
        len(doomed),
    )
    deadline = time.monotonic() + _RECOVERY_IDLE_WAIT_S
    checkpointed = False
    for key, conn in doomed:
        # Thread-safe by SQLite's contract; a statement another
        # thread is stepping on this handle aborts with
        # "interrupted", which releases the handle's lock.  (A
        # busy-timeout wait is not interruptible, so a thread
        # queued behind another process's write transaction can
        # keep its lock past the deadline.)
        try:
            conn.interrupt()
        except sqlite3.Error:  # already closed behind the registry's back
            logger.debug("Exception caught", exc_info=True)
        try:
            checkpointed = locked_call(
                conn.lock,
                _close_idle_conn,
                conn,
                checkpointed,
                acquire_timeout=max(0.0, deadline - time.monotonic()),
            )
        except LockTimeoutError:
            continue  # busy: its owner closes it on its next _get_db()
        _drop_open_conn_locked(key)
    _db_conn = None
    _bump_db_generation()
    busy = sum(1 for _c, _o, path in _open_conns.values() if path == current_path)
    if busy:
        # A connection opened now would join the dead mapping for
        # good (SQLite keeps one shared-memory node per database
        # per process while any handle uses it), so nothing is
        # reopened: the ``_attached_shm`` entry stays, every
        # ``_get_db()`` re-runs this recovery until the owners
        # retire their handles, and this call fails instead.
        raise sqlite3.OperationalError(
            f"disk I/O error: {current_path}-wal/-shm were deleted while "
            f"{busy} connection(s) are still busy; retry once their "
            "statements finish"
        )
    _attached_shm.pop(current_path, None)
    return True


def _forget_open_conn(conn: sqlite3.Connection) -> None:
    """Remove *conn* from :data:`_open_conns` (no-op if absent)."""
    locked_call(_init_tables_lock, _drop_open_conn_locked, id(conn))


def _retire_conn(conn: sqlite3.Connection) -> None:
    """Forget and close *conn*, the calling thread's own handle (best effort).

    The registry entry goes BEFORE the close so a close that raises
    cannot leave a dead handle registered.
    """
    _forget_open_conn(conn)
    try:
        conn.close()
    except Exception:
        logger.debug("Exception caught", exc_info=True)


def _close_cached_thread_conn() -> sqlite3.Connection | None:
    """Close and forget the CALLING thread's cached connection.

    Single teardown used by :func:`_close_db` and
    :func:`_close_thread_db` (previously duplicated in both, where the
    two copies could drift): closes the thread-local connection
    (best-effort) and resets every thread-local cache field so the
    thread's next ``_get_db()`` reconnects from scratch.

    Returns:
        The connection that was cached (now closed), or ``None`` when
        the thread had none.
    """
    tl_conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if tl_conn is not None:
        _retire_conn(tl_conn)
    _thread_local.conn = None
    _thread_local.gen = -1
    _thread_local.path = None
    _thread_local.file_id = None
    return tl_conn


def _close_db() -> None:
    """Close the calling thread's connection and invalidate cached handles.

    The current thread's connection, and those of threads that have
    already exited, are closed eagerly.  Bumping the generation counter
    invalidates every live thread's cached connection *lazily*: each is
    detected as stale on that thread's next ``_get_db()`` call and
    replaced (and closed) there.  A dead thread never makes that call,
    and its handle would otherwise keep the database file open (and,
    on Windows, undeletable) until some thread opens a new connection.
    """
    global _db_conn
    _stop_event_writer()
    _close_cached_thread_conn()
    locked_call(_init_tables_lock, _retire_other_threads_conns)
    _db_conn = None


def _retire_other_threads_conns() -> None:
    """Bump the generation and close dead threads' connections.

    Caller holds ``_init_tables_lock``.
    """
    _bump_db_generation()
    _prune_dead_thread_conns()


def _close_thread_db() -> None:
    """Close and forget the CALLING thread's cached connection only.

    Unlike :func:`_close_db`, the global generation counter is left
    untouched so every other thread's cached connection stays valid.
    Used by short-lived background threads (e.g. the startup
    orphan-task sweep in ``AgentServer.__init__``) to release their
    per-thread SQLite connection when they finish, instead of leaking
    an open connection for the life of the process.
    """
    global _db_conn
    tl_conn = _close_cached_thread_conn()
    if tl_conn is not None and _db_conn is tl_conn:
        _db_conn = None


_HISTORY_SELECT = (
    "SELECT id, timestamp, task, has_events, result, chat_id, "
    "model, work_dir, version, tokens, cost, steps, "
    "is_parallel, "
    "start_ts, end_ts, is_favorite, parent_task_id, max_budget, "
    "is_side_channel, tags, sea, success, is_cron "
    "FROM task_history "
)

# Row shape shared by the composer-history queries
# (``_prefix_match_tasks``, ``_load_input_history``): task_history rows
# are UNIONed with ``steer_inputs`` rows, whose ``rid`` is 0 so a
# timestamp tie is broken in favour of the task_history row.
_INPUT_TEXTS_SELECT = (
    "SELECT task, timestamp, rowid AS rid FROM task_history "
)

_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_task_history_id(value: object) -> TypeGuard[str]:
    """Return True when *value* is shaped like a ``task_history.id``.

    The canonical id format is the un-hyphenated 32-character
    lowercase hex string produced by ``uuid.uuid4().hex``.  Callers
    use this guard at IPC / SQL boundaries to reject malformed or
    legacy-int payloads before they propagate.
    """
    return isinstance(value, str) and _TASK_ID_RE.fullmatch(value) is not None


def _coerce_parent_task_id(value: object) -> str:
    """Return a canonical ``parent_task_id`` column value.

    Accepts only a 32-char lowercase-hex UUID string.  Any other
    shape (None, empty, int, list, dict, non-UUID string) maps to the
    empty-string sentinel that ``NOT_SUBAGENT`` treats as
    "not a sub-agent" — preventing garbage parent ids from being
    silently persisted as text that never matches any real UUID.
    """
    return value if is_task_history_id(value) else ""


def _extract_parent_task_id(payload: dict[str, object]) -> str:
    """Extract the canonical parent task id from an extra payload.

    Shared by :func:`_add_task` and :func:`_save_task_extra`, which
    both accept the parent link in two shapes: a flat
    ``parent_task_id`` value, or the legacy nested
    ``{"subagent": {"parent_task_id": <uuid>}}`` object (a bare
    ``"subagent": <uuid>`` string is tolerated too).  The two shapes
    are mutually exclusive — passing both keys is always rejected, so
    a caller can never smuggle two conflicting parents into one write.

    Args:
        payload: The extra dict a caller handed to the task writer.

    Returns:
        The coerced parent task id (``""`` when absent or malformed —
        see :func:`_coerce_parent_task_id`).

    Raises:
        ValueError: If *payload* contains both ``'subagent'`` and
            ``'parent_task_id'`` keys.
    """
    if "subagent" in payload and "parent_task_id" in payload:
        raise ValueError(
            "Cannot pass both 'parent_task_id' and 'subagent' "
            "in a task extra payload"
        )
    sub = payload.get("subagent")
    if isinstance(sub, dict):
        return _coerce_parent_task_id(sub.get("parent_task_id"))
    if sub is not None:
        return _coerce_parent_task_id(sub)
    return _coerce_parent_task_id(payload.get("parent_task_id"))


def _row_to_extra_json(row: sqlite3.Row) -> str:
    """Build the legacy-compat ``extra`` JSON string from typed columns.

    Many consumers (history sidebar, replay) read ``entry["extra"]`` as
    a JSON-encoded string.  This helper synthesizes the same shape from
    the new flat columns so those consumers continue to work unchanged.
    """
    payload: dict[str, object] = {}
    try:
        payload["model"] = _safe_str(row["model"])
        payload["work_dir"] = _safe_str(row["work_dir"])
        payload["version"] = _safe_str(row["version"])
        payload["tokens"] = _safe_int(row["tokens"], 0)
        payload["cost"] = _safe_float(row["cost"], 0.0)
        payload["steps"] = _safe_int(row["steps"], 0)
        payload["is_parallel"] = bool(row["is_parallel"])
        # The same fallback to the insertion timestamp the history
        # sidebar applies, so replay synthesis (task_settings events)
        # and the sidebar agree on when a legacy task started.
        payload["startTs"] = launch_ms(row["start_ts"], row["timestamp"])
        payload["endTs"] = _safe_int(row["end_ts"], 0)
        payload["max_budget"] = _safe_float(row["max_budget"], 0.0)
        payload["is_favorite"] = bool(row["is_favorite"])
        # Only when set, so a row without them reads exactly as before
        # the columns existed.
        for key in ("tags", "sea"):
            if row[key]:
                payload[key] = _safe_str(row[key])
        if row["is_cron"]:
            payload["is_cron"] = True
        if row["parent_task_id"]:
            sub: dict[str, object] = {
                "parent_task_id": _safe_str(row["parent_task_id"]),
            }
            if row["is_side_channel"]:
                sub["side_channel"] = True
            payload["subagent"] = sub
    except (KeyError, IndexError):
        return ""
    return _dumps_extra(payload) if payload else ""


def _history_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    """Convert a ``_HISTORY_SELECT`` row into a consumer-friendly dict.

    Exposes every selected typed column (``model``, ``cost``, etc.) so
    callers that switched to the new flat schema can read them
    directly, AND synthesises the legacy ``extra`` JSON string so
    callers that still parse ``entry["extra"]`` continue to work
    without any migration on their end.
    """
    out: dict[str, object] = {col: row[col] for col in row.keys()}
    out["extra"] = _row_to_extra_json(row)
    return out


def _is_failed_result(result: str, success: object = None) -> bool:
    """Return True when a ``task_history`` row represents a failed task
    that should be flagged with a red cross in the history sidebar
    (``.sidebar-item-failed``).

    A row fails either by the agent's own verdict or by how it ended:

    * ``success == 0`` — the row's ``success`` column, written by
      :func:`_save_task_result` from the ``success`` key of the
      agent's ``finish()`` result.  A ``finish(success=False)`` run
      keeps its HTML summary as the result text, so the text alone
      reads like a success; the chat tab strip already shows such a
      run as failed (``success === false`` on its terminal event) and
      the history must agree.

    Recognized failure markers in *result*:

    * ``Task failed*`` — the standard in-process failure prefix
      written by ``_save_task_result`` for ``task_error`` events.
    * ``Agent Failed Abruptly`` — the sentinel inserted by
      ``_add_task`` that survives only when the host process was
      SIGKILL'd / OOM-killed mid-task before any
      Python ``finally`` could run.
    * ``Task terminated unexpectedly (process killed)`` — the rewrite
      that ``_recover_orphaned_tasks`` applies to surviving sentinel
      rows on fresh-server boot.
    * ``Task stopped by user`` — an explicit user cancellation.  It is
      not a successful completion, so the history sidebar should mark
      it with the same red status dot used for failed runs.
    * ``Task interrupted*`` — an interrupted run.  Covers both the
      bare ``Task interrupted`` marker persisted by
      ``ChatSorcarAgent.run``'s ``except BaseException`` handler (a
      ``KeyboardInterrupt`` reaching a sub-agent / channel-agent
      run) and ``Task interrupted by server restart/shutdown`` (a
      graceful daemon/server shutdown cancellation).  Both are
      incomplete task outcomes rather than successes.

    Args:
        result: The row's ``result`` text.
        success: The row's ``success`` column (``0``, ``1`` or ``None``).
    """
    return (
        success == 0
        or result.startswith("Task failed")
        or result == "Agent Failed Abruptly"
        or result == "Task terminated unexpectedly (process killed)"
        or result == "Task stopped by user"
        or result.startswith("Task interrupted")
    )


_INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_th_timestamp ON task_history(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_th_task ON task_history(task)",
    "CREATE INDEX IF NOT EXISTS idx_th_chat_id ON task_history(chat_id)",
    "CREATE INDEX IF NOT EXISTS idx_th_parent_task_id "
    "ON task_history(parent_task_id)",
    "CREATE INDEX IF NOT EXISTS idx_ev_task_id ON events(task_id)",
    # UNIQUE: the database is shared by several processes whose
    # in-memory ``_next_seq_cache`` counters cannot see each other, so
    # a stale counter must FAIL the insert (and force a re-read of
    # ``MAX(seq)``) instead of silently persisting a duplicate
    # ``(task_id, seq)`` pair that misorders replay.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_ev_task_seq "
    "ON events(task_id, seq)",
)


def _dedupe_event_seqs(conn: sqlite3.Connection) -> None:
    """Resequence events of tasks holding duplicate ``(task_id, seq)`` rows.

    Databases written before ``idx_ev_task_seq`` existed can already
    contain duplicate sequence numbers (two processes with stale
    ``_next_seq_cache`` counters).  Creating the unique index over
    them would fail forever, so this migration renumbers every
    affected task's events to ``0..n-1`` in their existing order
    (``seq`` first, insertion ``id`` as the tie-break, so the first
    occurrence keeps its position).  No row is deleted — resequencing
    preserves every recorded event, which is the safest repair.

    Args:
        conn: Connection to migrate (caller manages the transaction).
    """
    dup_tasks = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT task_id FROM ("
            "SELECT task_id FROM events "
            "GROUP BY task_id, seq HAVING COUNT(*) > 1)"
        ).fetchall()
    ]
    for tid in dup_tasks:
        rows = conn.execute(
            "SELECT id FROM events WHERE task_id = ? ORDER BY seq, id",
            (tid,),
        ).fetchall()
        for new_seq, row in enumerate(rows):
            conn.execute(
                "UPDATE events SET seq = ? WHERE id = ?", (new_seq, row[0]),
            )
    if dup_tasks:
        logger.warning(
            "resequenced duplicate event seqs for %d task(s) before "
            "creating idx_ev_task_seq", len(dup_tasks),
        )


def _repair_and_create_index(conn: sqlite3.Connection, ddl: str) -> None:
    """Atomically repair duplicate event seqs and create *ddl*'s index.

    ``_init_tables_lock`` is process-LOCAL, so a concurrent old
    process can insert another duplicate ``(task_id, seq)`` row
    between an autocommit repair and the index retry — leaving
    duplicate rows AND no ``idx_ev_task_seq``.  Repair and creation
    therefore run together inside ONE cross-process SQLite write
    transaction (``BEGIN IMMEDIATE``), which every process on the
    database respects, and the combined step is retried a bounded
    number of times so a writer that sneaks a duplicate in between
    attempts cannot strand the database without the index.

    Args:
        conn: Connection to repair and create the index on.  When it
            already owns a transaction (the legacy-schema migration
            runs inside its own ``BEGIN IMMEDIATE``), the repair joins
            that transaction — the caller's write lock already
            excludes other processes — instead of nesting a ``BEGIN``.

    Raises:
        sqlite3.IntegrityError: When every attempt was refused (only
            possible if duplicates keep reappearing, which a held
            write transaction prevents; kept as a bounded safety
            valve rather than an infinite loop).
    """
    if conn.in_transaction:
        _dedupe_event_seqs(conn)
        conn.execute(ddl)
        return
    attempts = 3
    for attempt in range(attempts):
        try:
            with _immediate_txn(conn):
                _dedupe_event_seqs(conn)
                conn.execute(ddl)
            return
        except sqlite3.IntegrityError:
            if attempt == attempts - 1:
                raise
            _race_delay()


def _apply_index_ddl(conn: sqlite3.Connection) -> None:
    """Create every index in :data:`_INDEX_DDL` on *conn*.

    The unique ``idx_ev_task_seq`` index can be refused by a
    pre-existing database that already holds duplicate
    ``(task_id, seq)`` rows; those are repaired and the creation
    retried atomically via :func:`_repair_and_create_index`.

    Args:
        conn: Connection to create the indexes on.
    """
    for ddl in _INDEX_DDL:
        try:
            conn.execute(ddl)
        except sqlite3.IntegrityError:
            _repair_and_create_index(conn, ddl)


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_history (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            has_events INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            chat_id CHAR(32) DEFAULT '',
            model TEXT DEFAULT '',
            work_dir TEXT DEFAULT '',
            version TEXT DEFAULT '',
            tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            steps INTEGER DEFAULT 0,
            -- The framework (AgentState, sorcar.run()) defaults the
            -- toggle to ON, so a row whose producer did not state a
            -- value must read as ON too; DEFAULT 0 would label it
            -- "no parallel" in the history sidebar.
            is_parallel INTEGER DEFAULT 1,
            start_ts INTEGER DEFAULT 0,
            end_ts INTEGER DEFAULT 0,
            is_favorite INTEGER DEFAULT 0,
            parent_task_id TEXT DEFAULT '',
            -- The run's budget cap in USD (0.0 = unknown / legacy
            -- row); shown by the static task panel's settings info.
            max_budget REAL DEFAULT 0.0,
            -- Token of the process that created the row; see
            -- _process_owner_token / _recover_orphaned_tasks.
            owner TEXT DEFAULT '',
            -- 1 for a side-channel sub-agent (the /ask answerer): its
            -- result lands in the parent's transcript, so replays close
            -- its nested tab instead of re-opening it.
            is_side_channel INTEGER DEFAULT 0,
            -- Comma-separated classification of the task ("work,coding");
            -- written when the task finishes (task_metadata.classify_task_tags).
            tags TEXT DEFAULT '',
            -- File stem of the SEA that ran the task
            -- ("write_paper_sea", "cron_agent"); '' for a plain run.
            sea TEXT DEFAULT '',
            -- The agent's own finish() verdict: 0 for
            -- finish(success=False), 1 for success, NULL when the run
            -- returned no verdict (a crash, a stop, a legacy row).
            success INTEGER,
            -- 1 when the cron scheduler launched the task (a prompt,
            -- slash-command or in-chat job) or a sub-task of such a
            -- run; the CHATS list hides these unless its Cron chip is on.
            is_cron INTEGER DEFAULT 0,
            -- JSON {model: [cost, tokens]} of the task's own LLM calls
            -- (its llm_call events), written with the final cost, so the
            -- Spend panel files a set_model switch under the model that
            -- ran; '' = not recorded (legacy row).
            model_costs TEXT DEFAULT ''
        );
        -- One row per chat: a 6-8 word summary of what the chat's tasks
        -- did, the launch instant (epoch ms) of its latest task and
        -- that task's completion instant (0 while it runs).
        -- Maintained by _save_task_extra when a task finishes.
        -- user_summary is the title the user typed over it in the
        -- CHATS panel ('' = none); it survives every recompute.
        CREATE TABLE IF NOT EXISTS chat_summaries (
            chat_id TEXT PRIMARY KEY,
            summary TEXT DEFAULT '',
            last_launched INTEGER DEFAULT 0,
            last_finished INTEGER DEFAULT 0,
            user_summary TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES task_history(id),
            seq INTEGER NOT NULL,
            event_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL UNIQUE,
            count INTEGER DEFAULT 0,
            -- ``is_last`` is retained in the schema for backward
            -- compatibility with existing databases, but is no longer
            -- read or written: the last-selected model is now a user
            -- preference stored in ``config.json`` (see _load_last_model
            -- / _save_last_model).  Keeping the column here ensures the
            -- table schema does not change for new or existing databases.
            is_last INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS file_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            count INTEGER DEFAULT 0,
            last_used REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS frequent_tasks (
            task TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL DEFAULT 0
        );
        -- Text the user typed into a RUNNING task's composer (steer
        -- mode).  Such a message never gets a task_history row, so it
        -- is remembered here for the composer's autocomplete (prefix
        -- completions, ghost text, ArrowUp history).
        CREATE TABLE IF NOT EXISTS steer_inputs (
            text TEXT PRIMARY KEY,
            timestamp REAL NOT NULL DEFAULT 0
        );
        -- Claimed failed-event journal snapshots whose rows have been
        -- committed to ``events``.  The marker is inserted in the SAME
        -- transaction as the rows, so a replayer that dies between the
        -- commit and the snapshot file's unlink cannot cause the next
        -- replayer to insert the rows again under fresh seqs (the
        -- snapshot's unique ``.consumed-<pid>-<uuid>`` claim name is
        -- the key).  Rows are pruned once the file is really gone.
        CREATE TABLE IF NOT EXISTS replayed_journals (
            snapshot TEXT PRIMARY KEY,
            timestamp REAL NOT NULL DEFAULT 0
        );
        -- The id of every task the user deleted (_delete_task_rows), with
        -- the moment of the deletion.  The task database is synced between
        -- machines by adding the rows one of them lacks
        -- (kiss.scripts.sync_db), and without this record a task deleted
        -- here would come back from the other machine on the next sync;
        -- with it the deletion travels instead.  Rows are never pruned:
        -- a tombstone is a few dozen bytes.
        CREATE TABLE IF NOT EXISTS deleted_tasks (
            id TEXT PRIMARY KEY,
            deleted_ts REAL NOT NULL
        );
        -- The same for a deleted chat: its tasks are in deleted_tasks, and
        -- this says that its chat_summaries row (the title the user gave
        -- it, among other things) went too -- deleting a task on its own
        -- keeps the chat's row, so the task tombstones alone cannot tell.
        CREATE TABLE IF NOT EXISTS deleted_chats (
            chat_id TEXT PRIMARY KEY,
            deleted_ts REAL NOT NULL
        );
    """)
    # Columns first: an index over a column this version added to an
    # existing table can only be created once the column is there.
    _add_missing_columns(conn)
    _apply_index_ddl(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    ``CREATE TABLE IF NOT EXISTS`` cannot extend an existing table, so
    every column added to :func:`_init_tables` after the first release
    needs a matching ``ALTER TABLE`` here.  The columns and their
    backfills live in :mod:`kiss.agents.sorcar.history_columns`, shared
    with ``kiss.scripts.sync_db`` so that a database synced across two
    releases comes out the same as one migrated here.

    The columns and their backfill run as ONE transaction: the
    connection is autocommit, and a column that landed without its
    backfill (a Stop injected on the opening task thread, a crash)
    would never be backfilled, since the next open finds it present.
    SQLite's DDL is transactional, so the rollback drops the column
    again and the next open redoes both.  A caller that already owns a
    transaction (``backfill_task_metadata``'s ``with conn:``) covers
    the atomicity itself, as in :func:`_repair_and_create_index`.

    The transaction is opened only when a column is absent.  This runs
    for every connection a thread opens (every reader and writer calls
    ``_get_db()`` before taking ``_rw_lock``, so no token is held for
    it), and ``BEGIN IMMEDIATE`` waits out SQLite's ``busy_timeout``
    behind any other process's write transaction; a reader must not
    pay that when there is nothing to migrate.  The checks run again inside the
    transaction, so a peer that migrated in between adds nothing twice.

    Args:
        conn: Connection whose ``task_history`` and ``chat_summaries``
            tables to extend.
    """
    if not (
        missing_columns(conn, TASK_TABLE, ADDED_TASK_COLUMNS)
        or missing_columns(conn, CHAT_TABLE, ADDED_SUMMARY_COLUMNS)
    ):
        return
    txn = nullcontext() if conn.in_transaction else _immediate_txn(conn)
    with txn:
        complete_rows(conn, add_missing_columns(conn, TASK_TABLE, ADDED_TASK_COLUMNS))
        add_missing_columns(conn, CHAT_TABLE, ADDED_SUMMARY_COLUMNS)


def _migrate_old_schema_if_needed(conn: sqlite3.Connection) -> bool:
    """Port a pre-UUID task_history DB to the new schema in-place.

    Detects the old schema (``task_history.id`` is ``INTEGER``), creates
    new-shaped tables under temporary names, assigns each row a fresh
    ``uuid.uuid4().hex``, copies row data into the typed columns, remaps
    every ``events.task_id`` to the new UUID, then atomically replaces
    the old tables.  The JSON ``extra`` column holds the per-row
    metadata; a database from before 2026-04-13 has no such column,
    and its rows migrate with the metadata defaults.

    Returns ``True`` when migration was performed, ``False`` when the
    DB already has the new schema or no ``task_history`` table yet.
    """
    cols = {
        r[1]: (r[2] or "").upper()
        for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
    }
    if not cols or cols.get("id") == "TEXT":
        return False
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        with _immediate_txn(conn):
            return _migrate_old_schema_locked(conn)
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def _legacy_flag(v: object, missing: int = 0) -> int:
    """Coerce a legacy ``extra`` value to a 0/1 column value.

    *missing* is returned for a key the old row never recorded —
    the framework toggles predate their own persistence, so
    mapping their absence to 0 would assert that a legacy run had
    worktree/parallel/auto-commit explicitly DISABLED, the exact
    opposite of the framework default at the time.
    """
    if v is None:
        return missing
    if isinstance(v, str):
        return 0 if v.strip().lower() in {"", "0", "false", "no"} else 1
    return 1 if bool(v) else 0


def _migrate_old_schema_locked(conn: sqlite3.Connection) -> bool:
    """Body of :func:`_migrate_old_schema_if_needed` inside its write transaction.

    Re-checks the schema under the lock (another process may have
    migrated first) and returns ``False`` without writing when there
    is nothing to port.
    """
    cols_locked = {
        r[1]: (r[2] or "").upper()
        for r in conn.execute(
            "PRAGMA table_info(task_history)"
        ).fetchall()
    }
    if not cols_locked or cols_locked.get("id") == "TEXT":
        return False
    extra_column = "extra" if "extra" in cols_locked else "''"
    conn.execute("DROP TABLE IF EXISTS task_history__new")
    conn.execute("DROP TABLE IF EXISTS events__new")
    conn.execute(
        "CREATE TABLE task_history__new ("
        "id TEXT PRIMARY KEY, "
        "timestamp REAL NOT NULL, "
        "task TEXT NOT NULL, "
        "has_events INTEGER DEFAULT 0, "
        "result TEXT DEFAULT '', "
        "chat_id CHAR(32) DEFAULT '', "
        "model TEXT DEFAULT '', "
        "work_dir TEXT DEFAULT '', "
        "version TEXT DEFAULT '', "
        "tokens INTEGER DEFAULT 0, "
        "cost REAL DEFAULT 0.0, "
        "steps INTEGER DEFAULT 0, "
        "is_parallel INTEGER DEFAULT 1, "
        "start_ts INTEGER DEFAULT 0, "
        "end_ts INTEGER DEFAULT 0, "
        "is_favorite INTEGER DEFAULT 0, "
        "parent_task_id TEXT DEFAULT '', "
        "owner TEXT DEFAULT ''"
        ")"
    )
    conn.execute(
        "CREATE TABLE events__new ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "task_id TEXT NOT NULL REFERENCES task_history__new(id), "
        "seq INTEGER NOT NULL, "
        "event_json TEXT NOT NULL, "
        "timestamp REAL NOT NULL"
        ")"
    )
    rows = conn.execute(
        "SELECT id, timestamp, task, has_events, result, chat_id, "
        f"{extra_column} FROM task_history ORDER BY id ASC"
    ).fetchall()
    id_map: dict[int, str] = {int(r[0]): uuid.uuid4().hex for r in rows}
    dropped_unknown_keys = 0
    known_extra_keys = {
        "model", "work_dir", "version", "tokens", "cost", "steps",
        "is_parallel", "startTs", "endTs", "is_favorite", "subagent",
    }
    for r in rows:
        old_id = int(r[0])
        extra_raw = r[6] or ""
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
        except (json.JSONDecodeError, TypeError):
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        parent_task_id = ""
        sub = extra.get("subagent")
        if isinstance(sub, dict):
            old_parent = sub.get("parent_task_id")
            if isinstance(old_parent, int):
                parent_task_id = id_map.get(old_parent, "")
            elif (
                isinstance(old_parent, str)
                and _TASK_ID_RE.fullmatch(old_parent)
            ):
                parent_task_id = old_parent
        for k in extra:
            if k not in known_extra_keys:
                dropped_unknown_keys += 1
        conn.execute(
            "INSERT INTO task_history__new (id, timestamp, task, "
            "has_events, result, chat_id, model, work_dir, version, "
            "tokens, cost, steps, is_parallel, "
            "start_ts, end_ts, is_favorite, "
            "parent_task_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                id_map[old_id], r[1], r[2],
                r[3] or 0, r[4] or "", r[5] or "",
                _safe_str(extra.get("model")), _safe_str(extra.get("work_dir")),
                _safe_str(extra.get("version")), _safe_int(extra.get("tokens")),
                _safe_float(extra.get("cost")),
                _safe_int(extra.get("steps")),
                _legacy_flag(extra.get("is_parallel"), missing=1),
                _safe_int(extra.get("startTs")),
                _safe_int(extra.get("endTs")),
                _legacy_flag(extra.get("is_favorite")), parent_task_id,
            ),
        )
    has_events_table = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='events'"
    ).fetchone() is not None
    dropped_events = 0
    if has_events_table:
        ev_rows = conn.execute(
            "SELECT task_id, seq, event_json, timestamp FROM events"
        ).fetchall()
        for er in ev_rows:
            try:
                new_tid = id_map.get(int(er[0]))
            except (TypeError, ValueError):
                new_tid = None
            if new_tid is None:
                dropped_events += 1
                continue
            conn.execute(
                "INSERT INTO events__new "
                "(task_id, seq, event_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (new_tid, er[1], er[2], er[3]),
            )
        conn.execute("DROP TABLE events")
    conn.execute("DROP TABLE task_history")
    conn.execute(
        "ALTER TABLE task_history__new RENAME TO task_history"
    )
    conn.execute(
        "ALTER TABLE events__new RENAME TO events"
    )
    _apply_index_ddl(conn)
    if dropped_unknown_keys:
        logger.warning(
            "task_history migration dropped %d unknown extra key(s)",
            dropped_unknown_keys,
        )
    if dropped_events:
        logger.warning(
            "task_history migration dropped %d orphan event row(s) "
            "whose task_id had no surviving parent",
            dropped_events,
        )
    return True


#: Sentinel returned by :func:`_db_file_identity` when ``os.stat``
#: failed for a reason OTHER than the file being absent (EACCES, EIO,
#: EMFILE, ...).  Impossible as a real ``(st_dev, st_ino)`` pair.
_FILE_ID_UNKNOWN: tuple[int, int] = (-1, -1)


def _db_file_identity(path: str) -> tuple[int, int] | None:
    """Return the ``(st_dev, st_ino)`` identity of *path*.

    ``None`` means the file is CONFIRMED absent — ``os.stat`` raised
    ``FileNotFoundError`` (ENOENT/ENOTDIR).  Any other ``OSError`` is a
    transient or environmental failure that says nothing about whether
    the file exists, so the distinct sentinel :data:`_FILE_ID_UNKNOWN`
    is returned instead; callers must treat it as "identity unknown,
    assume unchanged" — never as "file deleted or replaced".

    Conflating the two was catastrophic: one transient stat failure
    under heavy load made every thread treat its healthy connection as
    stale, and the reconnect path then deleted the LIVE ``-wal``/
    ``-shm`` sidecars out from under the remaining connections,
    corrupting the database and losing every commit still in the WAL.

    The identity distinguishes a file that was deleted and recreated
    at the same pathname from the original file — a plain existence
    check cannot.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "transient stat failure on %s; assuming file unchanged",
            path,
            exc_info=True,
        )
        return _FILE_ID_UNKNOWN
    return (st.st_dev, st.st_ino)


def _get_db() -> sqlite3.Connection:
    """Return a per-thread database connection, creating one if needed.

    Each calling thread gets its own ``sqlite3.Connection`` so that
    concurrent threads never share cursor state.  Connections are
    cached in ``threading.local()`` and invalidated when:

    * ``_db_generation`` is bumped (via ``_close_db()``),
    * ``_DB_PATH`` changes (test redirects), or
    * the file at ``_DB_PATH`` is deleted or replaced on disk (its
      ``(st_dev, st_ino)`` identity no longer matches the one the
      cached connection was opened against), or
    * the ``-shm`` sidecar this process is mapped to was deleted or
      replaced on disk (:func:`_sidecars_orphaned`) — then EVERY
      connection in the process is closed first
      (:func:`_recover_orphaned_sidecars`), because a new connection
      cannot work while any old one keeps the dead mapping alive.

    The process-global ``_db_conn`` is deliberately NOT part of that
    validity test: it names whichever connection was created last by
    ANY thread, so a short-lived thread calling ``_close_thread_db()``
    used to force every other thread to close a healthy connection and
    re-run the WAL pragma and the migration check.
    """
    global _db_conn
    tl = _thread_local
    tl_conn: sqlite3.Connection | None = getattr(tl, "conn", None)
    tl_gen: int = getattr(tl, "gen", -1)
    tl_path: str | None = getattr(tl, "path", None)
    current_path = str(_DB_PATH)
    current_id = _db_file_identity(current_path)
    _maybe_reset_caches(current_path, current_id)
    attached = _attached_shm.get(current_path)
    if attached is not None and _sidecars_orphaned(current_path, current_id, attached):
        _recover_orphaned_sidecars(current_path, attached)
    # Read AFTER the recovery above: it bumps the generation, and this
    # thread's cached handle (closed by it) must be seen as stale.
    gen_snapshot = _db_generation

    if (
        tl_conn is not None
        and tl_gen == gen_snapshot
        and tl_path == current_path
        and (
            # A transient stat failure proves nothing about the file;
            # keep using the healthy cached connection rather than
            # tearing it down and reconnecting to a database that is
            # momentarily unreadable.
            current_id is _FILE_ID_UNKNOWN
            or (
                current_id is not None
                and getattr(tl, "file_id", None) == current_id
            )
        )
    ):
        return tl_conn

    if tl_conn is not None:
        # Forget the handle BEFORE closing it: when the reconnect
        # below raises (the database file is unreadable right now) the
        # cache must not keep pointing at a closed connection that a
        # later call would hand out as valid.
        tl.conn = None
        tl.file_id = None
        _retire_conn(tl_conn)

    _ensure_kiss_dir()
    _adopt_legacy_db_name(current_path)
    attached = _attached_shm.get(current_path)
    try:
        conn = _open_db_connection(current_path)
    except sqlite3.OperationalError as exc:
        code = getattr(exc, "sqlite_errorcode", 0) or 0
        # A fresh connection failing with SQLITE_IOERR while older ones
        # are open is the dead -shm mapping (see ``_sidecars_orphaned``)
        # caught before the identity check could see it — e.g. the
        # sidecars were unlinked between the stat above and the open.
        # The mapping the open joined is the one recorded before it; a
        # recovery completed since then (the entry is a new object, or
        # gone) stands down.  When nothing was recorded before the open,
        # the first connection of the process was registered during it
        # and the entry it recorded is the one the open joined.
        if attached is None:
            attached = _attached_shm.get(current_path)
        if code & 0xFF != sqlite3.SQLITE_IOERR or attached is None:
            raise
        _recover_orphaned_sidecars(current_path, attached)
        # The recovery bumped the generation; snapshot again BEFORE the
        # reopen (never after: a ``_close_db`` racing the open must
        # still retire this connection on the next call).
        gen_snapshot = _db_generation
        conn = _open_db_connection(current_path)

    _adopt_legacy_journal_snapshots(conn, current_path)
    tl.conn = conn
    tl.gen = gen_snapshot
    tl.path = current_path
    tl.file_id = _db_file_identity(current_path)
    _db_conn = conn
    return conn


_LEGACY_DB_NAME = "sorcar.db"  # the database's name before version 2026.10.2


def _adopt_legacy_db_name(current_path: str) -> None:
    """Rename a ``sorcar.db`` left by a pre-2026.10.2 install to *current_path*.

    The database was ALSO called ``history.db`` from March 2026 until
    2026-04-24, when it became ``sorcar.db`` and the old file was left
    where it was.  An install from those weeks has both names when it
    upgrades, and the leftover under the new name is not the database:
    its schema cannot even be opened (``no such column:
    parent_task_id``), while every task since lives in ``sorcar.db``.
    :func:`kiss.core.config.is_pre_2026_04_db` recognises
    the leftover by its schema and :func:`adopt_legacy_file` sets it
    aside (as ``history.db.stale-<UTC time>``) before the rename.

    The journals named after the database (:func:`_failed_events_path`,
    :func:`_final_results_path`) are adopted on their own, so that they
    follow even when the database itself was renamed by the deploy
    scripts (``rsorcar``, ``scripts/sync-task-db.sh``), which only
    know about the database and its ``-wal``/``-shm`` sidecars.
    """
    db_file = Path(current_path)
    adopt_legacy_file(db_file, _LEGACY_DB_NAME, ("-wal", "-shm", ""), stale=is_pre_2026_04_db)
    for journal in (".failed_events.jsonl", ".final_results.jsonl"):
        adopt_legacy_file(db_file.with_name(db_file.name + journal), _LEGACY_DB_NAME + journal)


def _adopt_legacy_journal_snapshots(conn: sqlite3.Connection, current_path: str) -> None:
    """Rename claimed journal snapshots of the legacy database after *current_path*.

    A ``.consumed-*`` snapshot (:func:`_claim_journal_snapshots`) left
    by a replay that crashed or was refused is only discovered under
    the active database's name.  Its basename is also its exactly-once
    marker in ``replayed_journals``, so the marker is re-keyed too and
    a snapshot whose rows were committed before the crash is still
    never inserted twice.

    The marker is copied to the new name BEFORE the rename and the old
    one deleted after it.  This runs outside :func:`_journal_file_lock`
    (the replayer holds that lock while it calls :func:`_get_db`), so a
    replay in another thread may claim the renamed snapshot the instant
    it appears; it must find the marker already there, or it would
    insert the committed rows a second time.  A copy left behind by a
    failed rename is a stale row, which is harmless.

    A refused write (the busy timeout expired behind another process's
    transaction, a ``disk I/O error``) is logged and ends the adoption
    for now; the snapshot stays under its legacy name until the next
    new connection retries.  It must not propagate: *conn* is already
    registered in :data:`_open_conns` and is only cached on the thread
    once this returns, so an exception here leaked one connection per
    :func:`_get_db` call and handed the caller none.
    """
    legacy_prefix = _failed_events_path(_LEGACY_DB_NAME) + _JOURNAL_CONSUMED_SUFFIX
    directory = os.path.dirname(current_path) or "."
    new_base = os.path.basename(_failed_events_path(current_path))
    try:
        names = [name for name in os.listdir(directory) if name.startswith(legacy_prefix)]
    except OSError:  # pragma: no cover — unreadable journal directory
        return
    for name in names:
        new_name = new_base + name[len(_failed_events_path(_LEGACY_DB_NAME)):]
        try:
            conn.execute(
                "INSERT OR IGNORE INTO replayed_journals (snapshot, timestamp) "
                "SELECT ?, timestamp FROM replayed_journals WHERE snapshot = ?",
                (new_name, name),
            )
            try:
                os.replace(os.path.join(directory, name), os.path.join(directory, new_name))
            except OSError:  # pragma: no cover — another process adopted it first
                continue
            _race_delay()
            conn.execute("DELETE FROM replayed_journals WHERE snapshot = ?", (name,))
        except sqlite3.Error:
            logger.warning(
                "legacy journal snapshot %s not adopted; retrying on the next connection",
                name, exc_info=True,
            )
            return


def _register_new_conn(conn: _LockedConnection, current_path: str) -> None:
    """Switch *conn* to WAL, create the schema and enter it in :data:`_open_conns`.

    The caller holds ``_init_tables_lock`` (see :func:`_open_db_connection`).
    """
    wal_deadline = time.monotonic() + 30.0
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            busy = (
                code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                if code is not None
                else "locked" in str(exc).lower()
                or "busy" in str(exc).lower()
            )
            if not busy or time.monotonic() >= wal_deadline:
                raise
            time.sleep(0.05)
    _migrate_old_schema_if_needed(conn)
    _init_tables(conn)
    # The first live connection is the one that created (or
    # re-created) the process's -shm mapping; a later one only
    # joins it.  Re-record for the first so an identity left
    # over from a fully closed earlier generation (SQLite
    # deletes the sidecars on the last close) is not compared
    # against the new mapping.
    _prune_dead_thread_conns()
    first_live = not _path_has_open_conns(current_path)
    _open_conns[id(conn)] = (
        conn, threading.current_thread(), current_path,
    )
    db_id = _db_file_identity(current_path)
    shm_id = _db_file_identity(current_path + "-shm")
    attached = _attached_shm.get(current_path)
    if (
        db_id is not None
        and db_id is not _FILE_ID_UNKNOWN
        and shm_id is not None
        and shm_id is not _FILE_ID_UNKNOWN
        and (first_live or attached is None or attached[0] != db_id)
    ):
        _attached_shm[current_path] = (db_id, shm_id)


def _open_db_connection(current_path: str) -> _LockedConnection:
    """Open, configure and register one new connection to *current_path*.

    Runs the WAL pragma (retried while another connection holds the
    database busy), the schema migration and table initialisation under
    ``_init_tables_lock``, then records the connection in
    :data:`_open_conns` and — for the first connection to this database
    file — the identity of the ``-shm`` sidecar the process is now
    mapped to (:data:`_attached_shm`), which is what
    :func:`_sidecars_orphaned` later compares against.

    Args:
        current_path: The database file to open.

    Returns:
        The configured connection.  On any failure the half-open
        connection is closed before the exception propagates.
    """
    # Deliberately NO manual cleanup of stale ``-wal``/``-shm``
    # sidecars here.  Application-level unlink of SQLite's sidecar
    # files is impossible to make safe: any check-then-unlink is racy
    # against other threads and OTHER PROCESSES concurrently opening
    # or holding the same database, and unlinking a live WAL destroys
    # committed-but-uncheckpointed pages (the root cause of the
    # 2026-08-15 history.db corruption).  SQLite itself handles a
    # leftover sidecar of a deleted-and-recreated database safely: a
    # WAL whose salt/checksums do not match is ignored and reset on
    # the first write, so no cleanup is needed for correctness.
    conn = _LockedConnection(current_path, check_same_thread=False, isolation_level=None)
    try:
        # The one busy handler; ``sqlite3.connect(timeout=...)`` would
        # install another that this pragma immediately replaces.
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        locked_call(_init_tables_lock, _register_new_conn, conn, current_path)
    except BaseException:
        _forget_open_conn(conn)
        try:
            conn.close()
        except sqlite3.Error:
            pass
        raise
    return conn


def _add_task(
    task: str,
    chat_id: str = "",
    extra: dict[str, object] | None = None,
) -> tuple[str, str]:
    """Append a task to the history and return ``(task_id, chat_id)``.

    When *chat_id* is ``""`` (new session), a new UUID-style string
    is generated as the chat session identifier.
    Otherwise the given *chat_id* is stored directly (continuation task).

    When *extra* is provided, its known keys (see ``_EXTRA_COL_MAP``)
    are written into their typed columns in the same INSERT so that
    values known at task creation time (model, work_dir, version,
    toggles) are immediately visible in the history sidebar — even
    before the task completes.  Callers that need to add
    post-completion values (tokens, cost) later call
    :func:`_save_task_extra`, which updates only the columns it is
    given (so an ``is_favorite`` flag set in the meantime survives).

    Thread-safe: all writes are protected by ``_rw_lock.write_lock()``.

    Args:
        task: The task description string.
        chat_id: Chat session identifier.  ``""`` starts a new session.
        extra: Optional dict of metadata to store immediately.

    Returns:
        ``(task_id, chat_id)`` — the inserted row id and the
        chat session identifier.
    """
    db = _get_db()
    payload = dict(extra) if extra else {}
    parent_task_id = _extract_parent_task_id(payload)
    sub = payload.get("subagent")
    is_side_channel = (
        1 if parent_task_id and isinstance(sub, dict)
        and sub.get("side_channel") else 0
    )
    # Minting the marker is file I/O (first call per home); keep it out
    # of the write lock.
    owner = _process_owner_token()
    with _rw_lock.write_lock():
        if chat_id == "":
            chat_id = _allocate_chat_id()
        task_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, chat_id, result, "
            "model, work_dir, version, tokens, cost, steps, is_parallel, "
            "start_ts, end_ts, is_favorite, "
            "parent_task_id, max_budget, owner, is_side_channel, sea, is_cron) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, time.time(), task, chat_id,
                "Agent Failed Abruptly",
                _safe_str(payload.get("model", "") or ""),
                _safe_str(payload.get("work_dir", "") or ""),
                _safe_str(payload.get("version", "") or ""),
                _safe_int(payload.get("tokens"), 0),
                _safe_float(payload.get("cost"), 0.0),
                _safe_int(payload.get("steps"), 0),
                1 if payload.get("is_parallel") else 0,
                _safe_int(payload.get("startTs"), 0),
                _safe_int(payload.get("endTs"), 0),
                1 if payload.get("is_favorite") else 0,
                parent_task_id,
                _safe_float(payload.get("max_budget"), 0.0),
                owner,
                is_side_channel,
                _safe_str(payload.get("sea", "") or ""),
                1 if payload.get("is_cron") else 0,
            ),
        )
    _invalidate_chat_context_cache(chat_id)
    return task_id, chat_id


def _allocate_chat_id() -> str:
    """Pre-allocate a chat session id without keeping a task row.

    Generates a new UUID-style string that can be used as a unique
    chat session identifier.

    Used by ``ChatSorcarAgent.run`` to fix the session id *before* the
    first task row is persisted, so every early consumer (worktree
    setup, printers) sees the same id the row will carry.

    Returns:
        A unique 32-character string suitable for use as a ``chat_id``.
    """
    return uuid.uuid4().hex


def _mark_legacy_side_channel_rows(task_template: str) -> int:
    """Stamp ``is_side_channel`` on sub-agent rows persisted before their
    runner set the flag.

    A side-channel child (see ``_add_task``) is recognised by its prompt:
    *task_template* is that prompt with a ``{task_id}`` placeholder
    standing for the child's ``parent_task_id``.  Rows already stamped
    are left alone, so the call is idempotent and cheap to repeat.

    Args:
        task_template: The child's prompt template, e.g.
            ``"What have the task with {task_id} done so far ..."``.

    Returns:
        The number of rows newly stamped.
    """
    prefix, _, suffix = task_template.partition("{task_id}")
    db = _get_db()
    with _rw_lock.write_lock():
        cursor = db.execute(
            "UPDATE task_history SET is_side_channel = 1 "
            "WHERE is_side_channel = 0 AND parent_task_id IS NOT NULL "
            "AND parent_task_id != '' AND task = ? || parent_task_id || ?",
            (prefix, suffix),
        )
    return int(cursor.rowcount or 0)


_TAG_FILTER_RE = re.compile(r"^[a-z]+$")


def _tag_filter_sql(tag: str) -> tuple[str, tuple[str, ...]]:
    """Build the ``AND`` clause that keeps only rows tagged ``tag``.

    ``tags`` is a comma-joined string (``"work,coding,testing"``), so the
    row matches when ``,tag,`` occurs in ``,tags,``.  A tag that is not a
    plain lowercase word (the whole vocabulary is) cannot come from the
    history panel's dropdown and is ignored: no clause, no filtering.

    Args:
        tag: The tag to keep, or ``""`` for no tag filter.

    Returns:
        The SQL fragment (empty or ``"AND ... "``) and its parameters.
    """
    if not _TAG_FILTER_RE.match(tag):
        return "", ()
    return "AND (',' || COALESCE(tags, '') || ',') LIKE ? ", (f"%,{tag},%",)


# The instant (epoch seconds) the Chats panel sorts, dates and filters a
# task by: its completion ``end_ts`` (epoch ms); a row without one (still
# running, or a legacy row) falls back to its launch ``timestamp``.
_HISTORY_DONE_SQL = "CASE WHEN end_ts > 0 THEN end_ts / 1000.0 ELSE timestamp END"


def _history_order_sql(running_task_ids: set[str] | None) -> tuple[str, tuple[str, ...]]:
    """Order running tasks first, then the most recently completed first."""
    ids = tuple(sorted(running_task_ids or ()))
    priority = f"id IN ({','.join('?' for _ in ids)}) DESC, " if ids else ""
    return priority + f"{_HISTORY_DONE_SQL} DESC, rowid DESC", ids


def _load_history(
    limit: int = 0, offset: int = 0, tag: str = "", *,
    query: str = "", work_dir: str = "", running_task_ids: set[str] | None = None,
) -> list[_HistoryEntry]:
    """Load task history entries (most-recent-first). Thread-safe.

    Args:
        limit: Maximum number of entries to return.
            0 returns all entries (no cap).
        offset: Number of entries to skip before returning results.
        tag: When set, only entries carrying this tag are returned.
        query: When set, a case-insensitive substring the task text
            must contain.
        work_dir: When set, only entries whose ``work_dir`` equals it
            (the history panel's WORKDIR dropdown).
        running_task_ids: Tasks to put before completed history, including
            tasks older than the first chronological page.

    Returns:
        List of history entry dicts with ``id``, ``timestamp``,
        ``task``, ``has_events``, ``result``, and ``chat_id`` keys.
    """
    query_sql = ""
    query_params: tuple[str, ...] = ()
    if query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query_sql = "AND task LIKE ? ESCAPE '\\' "
        query_params = (f"%{escaped}%",)
    work_dir_sql = "AND work_dir = ? " if work_dir else ""
    work_dir_params: tuple[str, ...] = (work_dir,) if work_dir else ()
    db = _get_db()
    with _rw_lock.read_lock():
        effective_limit = limit if limit > 0 else -1
        tag_sql, tag_params = _tag_filter_sql(tag)
        order_sql, order_params = _history_order_sql(running_task_ids)
        sql = (
            _HISTORY_SELECT
            + f"WHERE {NOT_SUBAGENT} "
            + query_sql
            + tag_sql
            + work_dir_sql
            + f"ORDER BY {order_sql} LIMIT ? OFFSET ?"
        )
        rows = db.execute(
            sql,
            (
                *query_params, *tag_params, *work_dir_params, *order_params,
                effective_limit, offset,
            ),
        ).fetchall()
        return [_history_row_to_dict(r) for r in rows]


def _chat_first_tasks(chat_ids: list[str]) -> dict[str, str]:
    """Return each chat's first (oldest) listable task text. Thread-safe.

    The History sidebar shows one collapsible panel per chat whose
    header is the chat's FIRST task — which may lie beyond the rows the
    current page carries, so it is looked up here over the same row set
    the sidebar lists (sub-agent rows excluded).

    Args:
        chat_ids: Chat session ids to look up; empty ids are skipped.

    Returns:
        Mapping of chat id to the task text of its earliest listable
        row — ties on ``timestamp`` break on ``rowid``, the project's
        chronological order (coarse clocks and imported databases do
        produce ties, and a bare column beside ``MIN()`` picks among
        them plan-dependently). The text is truncated to 1000
        characters: the header clamps to 3 lines and its tooltip needs
        no more, while the text is stamped on EVERY row of the chat —
        an unbounded prompt would multiply itself across the whole
        page's JSON. Chats with no listable row are absent.
    """
    ids = sorted({c for c in chat_ids if c})
    if not ids:
        return {}
    db = _get_db()
    with _rw_lock.read_lock():
        placeholders = ",".join("?" * len(ids))
        # NOT_SUBAGENT is unqualified, so it binds to t2
        # inside the subquery (innermost scope) and to t outside.
        rows = db.execute(
            "SELECT chat_id, substr(task, 1, 1000) AS task "
            "FROM task_history t "
            f"WHERE chat_id IN ({placeholders}) AND {NOT_SUBAGENT} "
            "AND rowid = (SELECT t2.rowid FROM task_history t2 "
            "WHERE t2.chat_id = t.chat_id "
            f"AND {NOT_SUBAGENT} "
            "ORDER BY t2.timestamp ASC, t2.rowid ASC LIMIT 1)",
            ids,
        ).fetchall()
    return {str(r["chat_id"]): str(r["task"] or "") for r in rows}


def _history_date_range() -> tuple[float | None, float | None]:
    """Return the first and last task completion times in the history.

    Computes the min and max of each task's completion time (launch
    time when it has none, see ``_HISTORY_DONE_SQL``) over the same row
    set the History sidebar lists (i.e. excluding sub-agent rows) so
    the sidebar's From/To date inputs can be pre-filled with the
    first and last task dates the date filter compares.  Thread-safe.

    Returns:
        ``(min_ts, max_ts)`` in epoch seconds, or ``(None, None)``
        when no listable rows exist.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        row = db.execute(
            f"SELECT MIN({_HISTORY_DONE_SQL}) AS mn, MAX({_HISTORY_DONE_SQL}) AS mx "
            f"FROM task_history WHERE {NOT_SUBAGENT}"
        ).fetchone()
    if row is None or row["mn"] is None or row["mx"] is None:
        return (None, None)
    return (float(row["mn"]), float(row["mx"]))


def _split_tree_spend(
    root: dict[str, Any],
    children: dict[str, list[dict[str, Any]]],
    field: str,
) -> dict[str, float]:
    """Split a top-level task's *field* (``cost`` or ``tokens``) by model.

    A task's persisted total already includes the spend of the
    sub-agents folded into it, which may run on other models.  Each
    task on the tree is credited with its own share, its total minus
    its sub-agents' totals, split by :func:`_split_own_share` among the
    models its own calls ran on, so the shares add up to the top-level
    task's total exactly.  When a task's sub-agents add up to more than
    the task itself (spend that was never folded into it, e.g. rows
    written before a fold was added), they are scaled down to fit and
    the task's own share is 0.

    Args:
        root: The top-level ``task_history`` row (``id``, ``m``, ``mc``,
            *field*).
        children: Sub-agent rows keyed by their ``parent_task_id``.
        field: The column to split, ``"cost"`` or ``"tokens"``.

    Returns:
        Model name to its share of *root*'s *field*.
    """
    shares: dict[str, float] = {}
    stack = [(root, 1.0)]
    seen: set[str] = set()
    while stack:
        row, scale = stack.pop()
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        total = max(float(row[field] or 0), 0.0)
        kids = children.get(row["id"], [])
        kids_total = sum(max(float(k[field] or 0), 0.0) for k in kids)
        kid_scale = scale
        if kids_total > total:
            kid_scale = scale * total / kids_total
        own = _split_own_share(row, max(total - kids_total, 0.0), field)
        for model, amount in own.items():
            shares[model] = shares.get(model, 0.0) + amount * scale
        stack.extend((k, kid_scale) for k in kids)
    return shares


def _split_own_share(row: dict[str, Any], own: float, field: str) -> dict[str, float]:
    """Split a task's *own* spend among the models its own calls ran on.

    The row's ``mc`` (``model_costs``) records the cost and tokens of
    the task's own LLM calls per model (see :func:`_own_model_costs`).
    Each model is credited with what its calls spent; spend no call
    accounts for (a ``decide`` call, a late side-channel charge) stays
    with the row's model.  Should the calls add up to more than *own*,
    they are scaled down to fit.  A row with no record (a legacy row)
    credits everything to its model.

    Args:
        row: The ``task_history`` row (``m``, ``mc``).
        own: The task's own share of *field*, sub-agents excluded.
        field: ``"cost"`` or ``"tokens"``.

    Returns:
        Model name to its part of *own*; the parts add up to *own*.
    """
    try:
        recorded = json.loads(row["mc"] or "{}")
    except (TypeError, ValueError):
        recorded = {}
    index = 0 if field == "cost" else 1
    calls: dict[str, float] = {}
    if isinstance(recorded, dict):
        for model, sums in recorded.items():
            if isinstance(sums, list) and len(sums) == 2:
                calls[model] = max(_safe_float(sums[index]), 0.0)
    called = sum(calls.values())
    fit = min(own / called, 1.0) if called > 0 else 0.0
    split = {model: amount * fit for model, amount in calls.items()}
    split[row["m"]] = split.get(row["m"], 0.0) + own - called * fit
    return split


def _round_shares(shares: dict[str, float], total: int) -> dict[str, int]:
    """Round *shares* to integers that add up to *total* (largest remainder).

    Args:
        shares: Non-negative fractional shares summing to *total*.
        total: The integer the rounded shares must add up to.

    Returns:
        The same keys with integer shares; ties go to the key that sorts
        first, so the result is deterministic.
    """
    rounded = {key: math.floor(value) for key, value in shares.items()}
    by_remainder = sorted(shares, key=lambda key: (rounded[key] - shares[key], key))
    for key in by_remainder[: max(total - sum(rounded.values()), 0)]:
        rounded[key] += 1
    return rounded


def _spend_by_day_and_model() -> list[dict[str, object]]:
    """Aggregate cost, tokens and task count per local calendar day and model.

    Covers the same top-level tasks the History sidebar lists, on the
    daemon's local date each task started, over the whole history, so
    the sidebar's Spend subpanel can draw its daily heatmap and
    cost-by-model bars.  A top-level task's cost and tokens (which
    include its sub-agents') are split among the models its task tree
    ran on (see :func:`_split_tree_spend`), so sub-agents on another
    model are not filed under the top-level task's model while the
    per-day sums still equal the listed tasks' totals.  ``tasks``
    counts top-level tasks under the model they were started with.  An
    empty or NULL model is reported as ``"unknown"``.  Thread-safe.

    Returns:
        ``{"date": "YYYY-MM-DD", "model": str, "cost": float,
        "tokens": int, "tasks": int}`` dicts in ascending date order;
        a day's models in descending cost order.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            "SELECT id, COALESCE(parent_task_id, '') AS p, "
            "date(timestamp, 'unixepoch', 'localtime') AS d, "
            "COALESCE(NULLIF(model, ''), 'unknown') AS m, "
            "COALESCE(cost, 0) AS cost, COALESCE(tokens, 0) AS tokens, "
            "COALESCE(model_costs, '') AS mc "
            "FROM task_history"
        ).fetchall()
    # A launch name such as "anthropic/claude-sonnet-4-6" is served (and
    # its llm_call events named) as "claude-sonnet-4-6": file both alike.
    rows = [dict(row, m=_strip_provider_prefix(row["m"])) for row in rows]
    children: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["p"]:
            children.setdefault(row["p"], []).append(row)
    sums: dict[tuple[str, str], list[float]] = {}  # [cost, tokens, tasks]
    for root in rows:
        if root["p"] or not root["d"]:
            continue
        day = str(root["d"])
        costs = _split_tree_spend(root, children, "cost")
        tokens = _round_shares(
            _split_tree_spend(root, children, "tokens"),
            max(int(root["tokens"] or 0), 0),
        )
        for model in costs.keys() | tokens.keys() | {root["m"]}:
            acc = sums.setdefault((day, model), [0.0, 0.0, 0])
            acc[0] += costs.get(model, 0.0)
            acc[1] += tokens.get(model, 0)
        sums[(day, root["m"])][2] += 1
    result: list[dict[str, object]] = [
        {
            "date": day,
            "model": model,
            "cost": acc[0],
            "tokens": int(acc[1]),
            "tasks": int(acc[2]),
        }
        for (day, model), acc in sums.items()
    ]
    result.sort(key=lambda r: (r["date"], -float(r["cost"]), r["model"]))  # type: ignore[arg-type]
    return result


def user_tasks_since(since: float) -> list[dict[str, object]]:
    """List the tasks the user started on or after *since* (epoch seconds).

    Only top-level tasks the user asked for: sub-agent rows
    (``parent_task_id`` set) and side-channel rows (``/ask``) are left
    out.  Rows carry the ``sea`` the task ran under so a caller can drop
    automations (cron runs, the recommendations agent itself); there is
    no row limit, so such a caller never loses a user task to a burst
    of automation rows.  Thread-safe.

    Args:
        since: Epoch seconds; rows with ``timestamp >= since`` qualify.

    Returns:
        ``{"task", "workDir", "timestamp", "model", "sea", "success",
        "result"}`` dicts, newest first; ``success`` is ``None`` for a
        task that never finished (stopped, or still running).
    """
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            "SELECT task, work_dir, timestamp, model, sea, success, result "
            f"FROM task_history WHERE timestamp >= ? AND {NOT_SUBAGENT} "
            "AND is_side_channel = 0 ORDER BY timestamp DESC",
            (float(since),),
        ).fetchall()
    return [
        {
            "task": str(row["task"] or ""),
            "workDir": str(row["work_dir"] or ""),
            "timestamp": _safe_float(row["timestamp"], 0.0),
            "model": str(row["model"] or ""),
            "sea": str(row["sea"] or ""),
            "success": None if row["success"] is None else bool(row["success"]),
            "result": str(row["result"] or ""),
        }
        for row in rows
    ]


# Collapse the union to one row per distinct text, most recent first.
_INPUT_TEXTS_RECENT_TAIL = (
    "GROUP BY task ORDER BY MAX(timestamp) DESC, MAX(rid) DESC LIMIT ?"
)


def _input_texts_sql(
    columns: str, task_where: str, steer_where: str, tail: str,
) -> str:
    """Build the query over every text the user typed into the composer.

    Unions the listable ``task_history`` rows with the ``steer_inputs``
    table (both exposed as ``task, timestamp, rid``) so the callers
    only differ in their filters and in what they do with the rows.

    Args:
        columns: Column list of the outer ``SELECT``.
        task_where: ``WHERE`` predicate on the ``task_history`` side.
        steer_where: ``WHERE`` predicate on the ``steer_inputs`` side
            (``""`` for none).
        tail: Clauses appended after the union (``GROUP BY`` ...).
    """
    steer = "SELECT text AS task, timestamp, 0 AS rid FROM steer_inputs"
    if steer_where:
        steer += f" WHERE {steer_where}"
    return (
        f"SELECT {columns} FROM ({_INPUT_TEXTS_SELECT}WHERE {task_where} "
        f"UNION ALL {steer}) {tail}"
    )


def _prefix_match_tasks(query: str, limit: int = 8) -> list[str]:
    """Find recent unique tasks starting with *query* (case-sensitive).

    The SQL ``GLOB`` filter does case-sensitive prefix matching, and the
    ``GROUP BY task`` tail collapses identical task strings to one row
    ordered by their latest use, so the dropdown menu never shows the
    same suggestion twice.

    Args:
        query: The prefix string to match against task text.
        limit: Maximum number of distinct matches to return.

    Returns:
        Up to *limit* full task strings, most recent first.  Empty when
        *query* is empty or no task matches.
    """
    if not query or limit <= 0:
        return []
    db = _get_db()
    with _rw_lock.read_lock():
        escaped = query.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")
        pattern = escaped + "*"
        rows = db.execute(
            _input_texts_sql(
                "task",
                f"CAST(task AS TEXT) GLOB ? AND LENGTH(task) > ? AND {NOT_SUBAGENT}",
                "text GLOB ? AND LENGTH(text) > ?",
                _INPUT_TEXTS_RECENT_TAIL,
            ),
            (pattern, len(query), pattern, len(query), limit),
        ).fetchall()
    return [_safe_str(row["task"]) for row in rows]


def _load_input_history(limit: int = _MAX_INPUT_HISTORY) -> list[str]:
    """Return the most recent distinct texts the user typed into the composer.

    Combines the listable ``task_history`` rows (sub-agent rows
    excluded) with the ``steer_inputs`` table, most recent first —
    a text's position is that of its most recent use in either
    table.  Feeds the composer's ArrowUp history.  Thread-safe.

    The texts go through :func:`_safe_str`, as every other JSON-bound
    column read does: a BLOB in the TEXT ``task`` column (a hand-edited
    or third-party database) made ``json.dumps`` of the
    ``inputHistory`` reply raise, so the composer's history request
    was never answered.  The readers that filter on the text
    (:func:`_prefix_match_tasks`, :func:`_load_slash_command_usage`)
    match ``CAST(task AS TEXT)``: whether ``GLOB`` matches a BLOB at all
    depends on how SQLite was built (``LIKE_DOESNT_MATCH_BLOBS``).

    Args:
        limit: Maximum number of texts returned (the most recent ones);
            defaults to :data:`_MAX_INPUT_HISTORY`.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            _input_texts_sql("task", NOT_SUBAGENT, "", _INPUT_TEXTS_RECENT_TAIL),
            (int(limit),),
        ).fetchall()
    return [_safe_str(row["task"]) for row in rows]


# The ``/name`` head of a slash-command prompt: the same characters
# ``sea_commands._split_slash_command`` accepts, followed by whitespace
# or the end of the prompt (``/xxxfoo`` is not a command).
_SLASH_COMMAND_HEAD_RE = re.compile(r"/([A-Za-z0-9_-]+)(?:\s|$)")


def _load_slash_command_usage() -> dict[str, tuple[int, float]]:
    """Return how often and how recently each slash command was run.

    Reads the texts the user typed that are a ``/name ...`` slash
    command and aggregates them by command name: the listable
    ``task_history`` rows (sub-agent rows excluded; a slash run keeps
    the raw ``/xxx text`` as its task text) and the ``steer_inputs``
    rows (a ``/ask`` typed into a running chat lands only there; that
    table holds one row per distinct text, so repeating the very same
    message counts once).  Feeds the ordering of the composer's ``/``
    command popup.  Thread-safe.

    Returns:
        ``{command_name: (run_count, last_run_timestamp)}``.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            _input_texts_sql(
                "task, timestamp",
                f"CAST(task AS TEXT) GLOB '/*' AND {NOT_SUBAGENT}",
                "text GLOB '/*'",
                "",
            ),
        ).fetchall()
    usage: dict[str, tuple[int, float]] = {}
    for row in rows:
        m = _SLASH_COMMAND_HEAD_RE.match(_safe_str(row["task"]))
        if m is None:
            continue
        count, last = usage.get(m.group(1), (0, 0.0))
        usage[m.group(1)] = (count + 1, max(last, float(row["timestamp"] or 0.0)))
    return usage


def _record_steer_input(text: str) -> None:
    """Remember *text* typed into the composer where no local task row will hold it.

    That is a message typed into a running task (an ``appendUserMessage``,
    a ``/ask`` line, a ``run`` sent to a busy tab; see
    ``_echo_injected_prompt``) and any prompt or message typed into a tab
    bound to a remote machine, whose task row and steer echo are the
    machine's (``RemoteTabs.route``).  Without the row the composer
    history (ArrowUp, ghost completions, ``/`` popup order) built by
    :func:`_load_input_history` would never offer the text again.

    Upserts the ``steer_inputs`` row so a repeated message only
    refreshes its ``timestamp``.  The table is capped at
    :data:`_MAX_STEER_INPUTS` rows: inserting a new text beyond the cap
    first evicts the oldest row, inside the same ``BEGIN IMMEDIATE``
    transaction (two processes could otherwise both see "under the
    cap" and both insert).

    Args:
        text: The message as typed.  Blank strings are ignored.
    """
    if not text.strip():
        return
    now = time.time()
    _capped_upsert(
        "steer_inputs", "text", text, _MAX_STEER_INPUTS, "timestamp ASC",
        "INSERT INTO steer_inputs (text, timestamp) VALUES (?, ?) "
        "ON CONFLICT(text) DO UPDATE SET timestamp = ?",
        (text, now, now),
    )


def _capped_upsert(
    table: str, key_col: str, key: str, cap: int, evict_order: str,
    upsert_sql: str, params: tuple[object, ...],
) -> None:
    """Run *upsert_sql* against the capped *table* in one write transaction.

    Evicts rows first when *key* is new and the table is at *cap* (see
    :func:`_evict_for_new_key`), inside the same ``BEGIN IMMEDIATE``
    transaction as the upsert.

    Args:
        table: Table name (a literal from this module).
        key_col: The table's unique text column.
        key: The value about to be upserted.
        cap: Maximum number of rows the table may hold.
        evict_order: ``ORDER BY`` clause selecting the rows to evict first.
        upsert_sql: The ``INSERT ... ON CONFLICT`` statement.
        params: Its bound parameters.
    """
    db = _get_db()
    with _rw_lock.write_lock(), _immediate_txn(db):
        _evict_for_new_key(db, table, key_col, key, cap, evict_order)
        db.execute(upsert_sql, params)


def _evict_for_new_key(
    db: sqlite3.Connection, table: str, key_col: str, key: str, cap: int, evict_order: str,
) -> None:
    """Make room in a capped table for an upsert of *key*.

    No-op when *key* already has a row (the upsert only refreshes it)
    or the table is under *cap*; otherwise deletes the first rows in
    *evict_order* until the insert will leave exactly *cap* rows — a
    table an older daemon let grow past the cap shrinks back to it
    too.  Runs inside the caller's ``BEGIN IMMEDIATE`` transaction: as
    separate autocommit statements, two PROCESSES could both observe
    "cap not reached" and both insert, pushing the table permanently
    over the cap.

    Args:
        db: The connection holding the write transaction.
        table: Table name (a literal from this module).
        key_col: The table's unique text column.
        key: The value about to be upserted.
        cap: Maximum number of rows the table may hold.
        evict_order: ``ORDER BY`` clause selecting the rows to evict first.
    """
    existing = db.execute(
        f"SELECT 1 FROM {table} WHERE {key_col} = ?", (key,),
    ).fetchone()
    if existing is not None:
        return
    row = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    _race_delay()
    if row[0] >= cap:
        db.execute(
            f"DELETE FROM {table} WHERE {key_col} IN "
            f"(SELECT {key_col} FROM {table} ORDER BY {evict_order} LIMIT ?)",
            (row[0] - cap + 1,),
        )


def _resolve_task_id(
    db: sqlite3.Connection,
    task_id: str | None,
) -> str | None:
    """Return *task_id* when it names an existing ``task_history`` row.

    Args:
        db: Active database connection.
        task_id: Row id to check.

    Returns:
        The row id, or ``None`` when *task_id* is not a history id or
        names no row.
    """
    if not is_task_history_id(task_id):
        return None
    row = db.execute(
        "SELECT id FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    return str(row["id"]) if row is not None else None


def _log_orphaned_task_forensics(
    db: sqlite3.Connection,
    rowids: list[int],
) -> None:
    """Log diagnostic info for each row still carrying the orphan sentinel.

    Called by :func:`_recover_orphaned_tasks` (under the write lock)
    before it rewrites the sentinel rows, so the startup log captures
    exactly which tasks were interrupted and their last recorded
    state.  This is the primary forensic evidence when a kill
    (SIGKILL / OOM) prevents the normal
    ``_save_task_result`` → ``_append_chat_event`` finally block from
    running.

    Args:
        db: Active database connection.
        rowids: The ``rowid``s about to be rewritten.  Empty is a
            no-op.
    """
    if not rowids:
        return
    placeholders = ",".join(["?"] * len(rowids))
    diag_rows = db.execute(
        "SELECT id, task, chat_id, model, start_ts, steps, cost "
        f"FROM task_history WHERE rowid IN ({placeholders})",
        list(rowids),
    ).fetchall()
    for row in diag_rows:
        task_id_val = row["id"]
        last_events = db.execute(
            "SELECT seq, event_json, timestamp FROM events "
            "WHERE task_id = ? ORDER BY seq DESC LIMIT 3",
            (task_id_val,),
        ).fetchall()
        last_event_summaries = []
        for ev in last_events:
            try:
                ev_data = json.loads(ev["event_json"])
                ev_type = ev_data.get("type", "unknown")
                ev_ts = _safe_float(ev["timestamp"], 0.0)
            except Exception:
                ev_type = "parse_error"
                ev_ts = 0
            last_event_summaries.append(
                f"seq={ev['seq']} type={ev_type} ts={ev_ts:.1f}"
            )
        model_name = row["model"] or "unknown"
        start_ts = row["start_ts"] or 0
        steps = row["steps"] if row["steps"] is not None else "?"
        cost = row["cost"] if row["cost"] is not None else "?"
        task_preview = (row["task"] or "")[:120]
        logger.warning(
            "Orphaned task recovered: id=%s chat_id=%s model=%s "
            "startTs=%s steps=%s cost=%s task=%r last_events=[%s]",
            task_id_val,
            row["chat_id"] or "",
            model_name,
            start_ts,
            steps,
            cost,
            task_preview,
            "; ".join(last_event_summaries),
        )


_USAGE_STEPS_RE = re.compile(r"Steps:\s*(\d+)")
_USAGE_TOKENS_RE = re.compile(r"Total tokens:\s*([\d,]+)")
_USAGE_COST_RE = re.compile(r"Budget:\s*\$([0-9][\d,]*\.?\d*)")


def _recovered_progress_from_events(
    db: sqlite3.Connection, task_id: str,
) -> dict[str, int | float]:
    """Reconstruct a killed task's last known progress from its events.

    When the owning process dies before the task runner's cleanup can
    call ``_save_task_extra``, the ``steps``/``tokens``/``cost``/
    ``end_ts`` columns keep their creation-time zeros and the history
    UI shows "0 steps, 0 tok" for a task that may have run for hours.
    The surviving ``events`` rows record how far the task actually
    got: every ``usage_info`` event carries the per-task step, token,
    and budget counters in its ``text`` field, and the newest event of
    any type dates the last observed activity.

    Args:
        db: Active database connection (caller holds the write lock).
        task_id: Task whose events should be inspected.

    Only the per-task counter events emitted once per agent step
    (text ``"Steps: 175/10000, ... Total tokens: 33,641,687, Budget:
    $56.4682/$1000.00"``) are trusted; their structured
    ``total_steps``/``total_tokens``/``cost`` fields (task totals across
    continuation sessions) win over the session-relative text.  The live-usage monitor's
    ``usage_info`` events carry a different text form and structured
    ``total_tokens``/``total_steps``/``cost`` fields, but those are
    CROSS-TASK aggregates ("incl. parallel sub-agents") — writing them
    into the per-task columns would overstate the task, so such events
    are skipped and the scan continues to the newest per-step event.

    Returns:
        Mapping of the subset of ``steps``/``tokens``/``cost``/
        ``end_ts`` columns that could be recovered — empty when the
        task has no events.  Values are parsed defensively: a
        malformed or foreign-format ``usage_info`` event contributes
        nothing and never raises.
    """
    progress: dict[str, int | float] = {}
    last = db.execute(
        "SELECT MAX(timestamp) AS ts FROM events WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    last_ts = _safe_float(last["ts"] if last is not None else None, 0.0)
    if last_ts <= 0:
        return progress
    progress["end_ts"] = int(last_ts * 1000)
    usage_rows = db.execute(
        "SELECT event_json FROM events "
        "WHERE task_id = ? AND event_json LIKE '%\"usage_info\"%' "
        "ORDER BY seq DESC",
        (task_id,),
    )
    for usage_row in usage_rows:
        try:
            event = json.loads(usage_row["event_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict) or event.get("type") != "usage_info":
            continue
        text = str(event.get("text", ""))
        steps_m = _USAGE_STEPS_RE.search(text)
        if steps_m is None:
            # Not the per-step counter form.  In particular the live
            # monitor's "Tokens: N, Budget: $C (live, incl. parallel
            # sub-agents)" text has no "Steps:" marker, and its Budget
            # figure is a cross-task aggregate that must not be
            # mistaken for this task's cost.
            continue
        found: dict[str, int | float] = {}
        tokens_m = _USAGE_TOKENS_RE.search(text)
        cost_m = _USAGE_COST_RE.search(text)
        try:
            found["steps"] = int(steps_m.group(1))
            if tokens_m:
                found["tokens"] = int(tokens_m.group(1).replace(",", ""))
            if cost_m:
                found["cost"] = float(cost_m.group(1).replace(",", ""))
        except ValueError:
            # The token/cost regexes admit comma-only or otherwise
            # unconvertible digit groups; a corrupt event must not
            # abort the recovery sweep.
            continue
        # The text is the executor's own session-relative counter, but
        # the printer rewrites the structured fields of the same event
        # to task totals (sessions banked by earlier continuations
        # added), which is what the task actually spent.
        found.update(_structured_usage_totals(event))
        progress.update(found)
        break
    return progress


def _structured_usage_totals(event: dict[str, Any]) -> dict[str, int | float]:
    """Return the task-total counters carried by a per-step ``usage_info`` event.

    Args:
        event: A decoded ``usage_info`` event whose text is the per-step
            counter form.

    Returns:
        The subset of ``steps``/``tokens``/``cost`` present as valid
        ``total_steps``/``total_tokens``/``cost`` fields (events written
        by versions that predate these fields contribute nothing).  The
        cost is the exact ``cost_usd`` when the event carries it, else
        the four-decimal ``cost`` string.
    """
    totals: dict[str, int | float] = {}
    steps = event.get("total_steps")
    tokens = event.get("total_tokens")
    if isinstance(steps, int) and not isinstance(steps, bool):
        totals["steps"] = steps
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        totals["tokens"] = tokens
    cost = event_cost(event)
    if cost is not None:
        totals["cost"] = cost
    return totals


def _backfill_orphan_progress(
    db: sqlite3.Connection, rowids: list[int],
) -> None:
    """Backfill progress columns of just-recovered orphan rows.

    Called by :func:`_recover_orphaned_tasks` (inside its write
    transaction) for the rows whose ``result`` it rewrites.  Each
    still-zero ``steps``/``tokens``/``cost``/``end_ts`` column is
    filled from the evidence in the task's surviving events, so the
    history sidebar shows the task's real last-known progress instead
    of "0 steps, 0 tok".  A column that already holds a non-zero value
    (written by a partial cleanup before the kill) is never
    overwritten — the recorded value is more authoritative than a
    reconstruction.

    The backfill is best-effort by construction: any per-row failure
    is logged and skipped, because an exception escaping here would
    roll back the enclosing transaction and undo the sentinel rewrite
    itself — losing the primary purpose of the sweep over a cosmetic
    reconstruction.

    Args:
        db: Active database connection (caller holds the write lock).
        rowids: The ``rowid``s whose sentinel result was rewritten.
    """
    for rid in rowids:
        try:
            row = db.execute(
                "SELECT id FROM task_history WHERE rowid = ?", (rid,),
            ).fetchone()
            if row is None or not row["id"]:
                continue
            # The row's result was just rewritten to its terminal text:
            # derive its tags (``failed`` included) and refresh its
            # chat's summary row, as _save_task_extra does for a task
            # that ends normally.
            _finalize_task_metadata(db, str(row["id"]))
            progress = _recovered_progress_from_events(db, str(row["id"]))
            if not progress:
                continue
            sets = [
                f"{col} = CASE WHEN {col} IS NULL OR {col} = 0 "
                f"THEN ? ELSE {col} END"
                for col in progress
            ]
            db.execute(
                f"UPDATE task_history SET {', '.join(sets)} WHERE rowid = ?",
                [*progress.values(), rid],
            )
        except Exception:
            logger.warning(
                "orphan progress backfill failed for rowid %s", rid,
                exc_info=True,
            )


def _recover_orphaned_tasks(
    active_task_ids: set[str],
    created_before: float | None = None,
) -> int:
    """Replace the ``"Agent Failed Abruptly"`` sentinel on dead rows.

    The sentinel is written by :func:`_add_task` at task-creation
    time and is supposed to be overwritten by
    :func:`_save_task_result` from ``_TaskRunnerMixin._run_task_inner``'s
    cleanup ``finally``.  When the host process is killed externally
    (SIGKILL, OOM) mid-task the Python
    ``finally`` never runs and the sentinel survives in the history
    sidebar verbatim as "Agent Failed Abruptly".  Catching the
    in-process ``BaseException`` variants in the task runner cannot
    cover this case because no Python code runs to catch anything.

    The remedy is a startup-time sweep: on every fresh ``AgentServer``
    instantiation (one per Python process) we scan ``task_history``
    for any row that still carries the sentinel AND whose id is not
    in *active_task_ids* (the currently-running tasks in THIS
    process), and rewrite ``result`` to a diagnostic message that
    truthfully describes what happened.  A sentinel row whose final
    result survives in the plain-file sidecar journal (see
    :func:`_journal_final_result`) actually FINISHED — its committed
    result was lost by the database afterwards (WAL discarded after a
    ``disk I/O error``) — so the journalled result is restored instead
    of the "process killed" message.  The still-zero progress
    columns of each rewritten row are then backfilled from the task's
    surviving events (see :func:`_backfill_orphan_progress`) so the
    history sidebar shows the real last-known step/token/cost state
    instead of "0 steps, 0 tok".

    Liveness is decided from the DATABASE, not from process memory:
    every row records the ``owner`` token of the process that created
    it (see :func:`_process_owner_token`), and a row whose owner still
    holds its liveness marker is skipped.  Without that check a second
    Sorcar process — a ``kiss`` CLI run, a restarted daemon — rewrote the
    rows of tasks that were still RUNNING in the first process, painting
    a red failure dot on a live task; and because the sentinel was gone,
    neither the shutdown safety net nor a later sweep could ever record
    the real outcome.

    Args:
        active_task_ids: Row ids that are still being processed in
            the current process and must therefore NOT be rewritten.
            Pass an empty set at fresh-server startup — by then any
            row carrying the sentinel must belong to a prior process,
            live or dead.
        created_before: Optional epoch-seconds cut-off.  When given,
            only sentinel rows whose ``timestamp`` column is strictly
            older are rewritten.  ``AgentServer`` passes its boot
            timestamp here because its sweep runs on a BACKGROUND
            thread: a task legitimately started after boot (inserting
            a fresh sentinel row) must never be mistaken for an
            orphan of a prior process by a sweep whose UPDATE races
            past the insertion — that mislabels a live task as
            "process killed" and defeats the pre-emptive shutdown
            persistence in ``_stop_active_agent_tasks`` (which only
            rewrites rows still carrying the sentinel).  ``None``
            applies no time filter.

    Returns:
        The number of rows whose ``result`` column was rewritten.
    """
    db = _get_db()
    active_ids = {str(t) for t in active_task_ids}
    # ``rowid`` rather than ``id``: ``id`` is a TEXT primary key, so
    # SQLite accepts NULL there (a partial INSERT from an older
    # release or an external tool), and such a row could then never be
    # targeted by an id-based UPDATE.  Every non-WITHOUT-ROWID table
    # has a unique, non-NULL rowid.
    select_sql = (
        "SELECT rowid AS rid, id, owner FROM task_history WHERE result = ? "
    )
    params: list[object] = ["Agent Failed Abruptly"]
    if created_before is not None:
        select_sql += "AND timestamp < ? "
        params.append(float(created_before))
    sidecar = _final_results_path(str(_DB_PATH))
    journalled = _load_final_results(sidecar)
    with _rw_lock.read_lock():
        candidates = db.execute(select_sql, params).fetchall()
    # The liveness probes touch the filesystem (flock per marker), so
    # they run outside the write transaction; the transaction below
    # re-checks that each row still carries the sentinel.
    probed = [
        (int(row["rid"]), str(row["id"] or ""))
        for row in candidates
        if str(row["id"] or "") not in active_ids
        and not _owner_is_alive(row["owner"] or "")
    ]
    if not probed:
        _prune_final_results_journal(sidecar)
        return 0
    with _rw_lock.write_lock(), _immediate_txn(db):
        placeholders = ",".join(["?"] * len(probed))
        still_sentinel = {
            int(row["rid"])
            for row in db.execute(
                "SELECT rowid AS rid FROM task_history "
                f"WHERE rowid IN ({placeholders}) AND result = ?",
                [*[rid for rid, _tid in probed], "Agent Failed Abruptly"],
            ).fetchall()
        }
        dead_rows = [(rid, tid) for rid, tid in probed if rid in still_sentinel]
        # A dead row whose final result survives in the sidecar journal
        # was NOT killed mid-task: its result was saved to the database
        # but the committed pages were later lost (e.g. the 2026-09-12
        # WAL loss after a ``disk I/O error``).  Restore the journalled
        # result instead of mislabeling the task as "process killed".
        restored = [(rid, tid) for rid, tid in dead_rows if tid in journalled]
        killed_rowids = [rid for rid, tid in dead_rows if tid not in journalled]
        _log_orphaned_task_forensics(db, killed_rowids)
        rowcount = 0
        for rid, tid in restored:
            cursor = db.execute(
                "UPDATE task_history SET result = ?, success = ? "
                "WHERE rowid = ? AND result = ?",
                [
                    journalled[tid]["result"],
                    journalled[tid]["success"],
                    rid,
                    "Agent Failed Abruptly",
                ],
            )
            rowcount += cursor.rowcount or 0
            logger.warning(
                "Task result lost by the database was restored from "
                "the final-results journal: id=%s", tid,
            )
        if killed_rowids:
            placeholders = ",".join(["?"] * len(killed_rowids))
            cursor = db.execute(
                "UPDATE task_history SET result = ? "
                f"WHERE rowid IN ({placeholders}) AND result = ?",
                [
                    "Task terminated unexpectedly (process killed)",
                    *killed_rowids,
                    "Agent Failed Abruptly",
                ],
            )
            rowcount += cursor.rowcount or 0
        if dead_rows:
            _backfill_orphan_progress(db, [rid for rid, _tid in dead_rows])
    if rowcount:
        logger.warning(
            "Recovered %d orphaned task(s) from prior process kill",
            rowcount,
        )
        _invalidate_chat_context_cache("")
    _prune_final_results_journal(sidecar)
    return rowcount


def _shutdown_persist_in_flight_results(task_ids: set[str]) -> int:
    """Pre-emptive sentinel rewrite for in-flight tasks during shutdown.

    Called by :meth:`RemoteAccessServer._stop_active_agent_tasks` BEFORE
    the worker threads are signalled to stop.  For each row in
    *task_ids* that still carries the ``"Agent Failed Abruptly"``
    sentinel (set by :func:`_add_task` at task creation time and
    normally overwritten by :func:`_save_task_result` from
    ``_TaskRunnerMixin._run_task_inner``'s cleanup ``finally``), the
    column is rewritten to ``"Task interrupted by server
    restart/shutdown"``.

    This is a safety net for the failure mode where the worker thread
    cannot reach ``_save_task_result`` before the process exits — e.g.
    because it is wedged in C code (a blocking LLM API call ignoring
    ``KeyboardInterrupt``) or its cleanup ``finally`` exceeds the
    shutdown timeout.  Without the pre-emptive rewrite, the row stays
    at the sentinel and the next startup's orphan sweep
    (:func:`_recover_orphaned_tasks`) rewrites it to ``"Task
    terminated unexpectedly (process killed)"`` — the silent failure
    mode users report as "the agent was killed mid-task".

    Workers that *do* manage to finish their cleanup will overwrite
    this placeholder with a more detailed message (e.g. the per-task
    summary or "Task interrupted by server restart/shutdown" from the
    same cleanup path) — that ordering is fine because we set the
    placeholder BEFORE signalling the workers.

    Only rows still at the sentinel are touched, so a task that
    already completed cleanly (its row carrying a real result) is
    never clobbered.

    Args:
        task_ids: Row ids whose still-pending sentinel rows should be
            pre-emptively rewritten.

    Returns:
        The number of rows whose ``result`` column was rewritten.
    """
    if not task_ids:
        return 0
    db = _get_db()
    id_list = [str(t) for t in task_ids]
    placeholders = ",".join(["?"] * len(id_list))
    sql = (
        f"UPDATE task_history SET result = ? "
        f"WHERE id IN ({placeholders}) AND result = ?"
    )
    with _rw_lock.write_lock(), _immediate_txn(db):
        cursor = db.execute(
            sql,
            ["Task interrupted by server restart/shutdown",
             *id_list, "Agent Failed Abruptly"],
        )
        rowcount = cursor.rowcount or 0
    if rowcount:
        logger.warning(
            "Pre-emptively persisted shutdown result for %d in-flight task(s)",
            rowcount,
        )
        # The process is shutting down: clearing the whole chat-context
        # cache is as good as finding the affected chats.
        _invalidate_chat_context_cache("")
    return rowcount


_UPDATABLE_COLUMNS = frozenset({"result", "success"})


def _update_task_columns(
    values: dict[str, object],
    task_id: str | None,
) -> str | None:
    """Write *values* into the resolved ``task_history`` row in one UPDATE.

    Drains pending queued events first so the column update is ordered
    after every event the task has emitted so far, then performs the
    UPDATE under the process-wide write lock.

    Args:
        values: Column name to new value.  The names are interpolated
            into the SQL, so each must be one of
            :data:`_UPDATABLE_COLUMNS`; anything else raises
            ``ValueError``.
        task_id: Stable row id to update.

    Returns:
        The updated row's ``chat_id`` (possibly ``""``), or ``None``
        when no row could be resolved.

    Raises:
        ValueError: When a column is not an allowed column name.
    """
    refused = set(values) - _UPDATABLE_COLUMNS
    if refused:
        raise ValueError(
            f"_update_task_columns refuses columns {sorted(refused)}; "
            f"allowed: {sorted(_UPDATABLE_COLUMNS)}"
        )
    _flush_chat_events(task_id if is_task_history_id(task_id) else None)
    db = _get_db()
    with _rw_lock.write_lock(), _immediate_txn(db):
        resolved = _resolve_task_id(db, task_id)
        if resolved is None:
            return None
        sets = ", ".join(f"{column} = ?" for column in values)
        db.execute(
            f"UPDATE task_history SET {sets} WHERE id = ?",
            (*values.values(), resolved),
        )
        row = db.execute(
            "SELECT chat_id FROM task_history WHERE id = ?", (resolved,),
        ).fetchone()
        return (row["chat_id"] or "") if row is not None else ""


def _save_task_result(
    result: str,
    task_id: str | None = None,
    *,
    success: bool | None = None,
) -> None:
    """Save the result summary (and the agent's verdict) for a task.

    No event table changes.  The result is additionally journalled to
    a plain-file sidecar (see :func:`_journal_final_result`) so the
    startup orphan sweep can restore it if the database write is later
    lost to an I/O failure — the sweep would otherwise mislabel the
    finished task as "process killed".

    Args:
        result: The task result text to store in the history entry.
        task_id: Stable row id to update.
        success: The ``success`` key of the agent's ``finish()`` result
            when the run returned one; ``False`` makes the history
            sidebar show the task as failed (see
            :func:`_is_failed_result`).  ``None`` (a crash, a stop, a
            caller without a verdict) leaves the column untouched.
    """
    values: dict[str, object] = {"result": result}
    if success is not None:
        values["success"] = int(success)
    affected_chat_id = _update_task_columns(values, task_id)
    if affected_chat_id is None:
        return
    if is_task_history_id(task_id):
        _journal_final_result(
            str(task_id), result, None if success is None else int(success),
        )
    _invalidate_chat_context_cache(affected_chat_id)


def _set_task_favorite(task_id: str, is_favorite: bool) -> bool:
    """Toggle the ``is_favorite`` column for a task row.

    Thread-safe: drains the background event queue first so the
    favourite flag write is ordered after any in-flight event inserts
    for the same task, then acquires the process-wide write lock.

    Args:
        task_id: Primary key of the ``task_history`` row to update.
        is_favorite: New value for the ``is_favorite`` flag.

    Returns:
        True when the row existed and was updated, False otherwise.
    """
    _flush_chat_events(task_id)
    db = _get_db()
    with _rw_lock.write_lock():
        cursor = db.execute(
            "UPDATE task_history SET is_favorite = ? WHERE id = ?",
            (1 if is_favorite else 0, task_id),
        )
        return (cursor.rowcount or 0) > 0


def _add_task_usage(
    task_id: str, tokens: int, cost: float, steps: int,
) -> tuple[int, float, int] | None:
    """Add post-run spend to a task row's ``tokens`` / ``cost`` / ``steps``.

    Used when work done on the task's behalf AFTER its row was persisted
    (the merge agent that resolves its auto-merge conflicts) must count
    towards the task.  The add is one atomic ``UPDATE`` on the stored
    values, so it is correct whether or not the live agent counters
    still hold the task's own totals.

    Args:
        task_id: Primary key of the ``task_history`` row to update.
        tokens: Tokens to add.
        cost: USD to add.
        steps: Steps to add.

    Returns:
        The row's new ``(tokens, cost, steps)``, or ``None`` when no
        such row exists.
    """
    _flush_chat_events(task_id)
    db = _get_db()
    with _rw_lock.write_lock(), _immediate_txn(db):
        return _add_usage_locked(db, task_id, tokens, cost, steps)


def _add_usage_locked(
    db: sqlite3.Connection, task_id: str, tokens: int, cost: float, steps: int,
) -> tuple[int, float, int] | None:
    """Add spend to a row's ``tokens`` / ``cost`` / ``steps`` and return the new totals.

    Args:
        db: The connection holding the write transaction.
        task_id: Primary key of the ``task_history`` row to update.
        tokens: Tokens to add.
        cost: USD to add.
        steps: Steps to add.

    Returns:
        The row's new ``(tokens, cost, steps)``, or ``None`` when no
        such row exists.
    """
    cursor = db.execute(
        "UPDATE task_history SET tokens = COALESCE(tokens, 0) + ?, "
        "cost = COALESCE(cost, 0.0) + ?, steps = COALESCE(steps, 0) + ? "
        "WHERE id = ?",
        (int(tokens), float(cost), int(steps), task_id),
    )
    if (cursor.rowcount or 0) == 0:
        return None
    row = db.execute(
        "SELECT tokens, cost, steps FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    return (_safe_int(row[0]), _safe_float(row[1]), _safe_int(row[2]))


def _add_late_task_usage(
    task_id: str, tokens: int, cost: float, steps: int,
) -> tuple[list[tuple[str, int, float, int]], str]:
    """Charge side-channel spend to *task_id* and its finished ancestors.

    A side channel (an ``/ask`` answerer, a task-update run) can end
    after the task it works for has finished, when that task's
    ``tokens`` / ``cost`` / ``steps`` are final and folded, where it
    has one, into its parent's.  The spend is therefore added to the
    row itself and to every finished ancestor up the
    ``parent_task_id`` chain, stopping at the first unfinished row (no
    ``end_ts``): that task is still running, so the caller banks the
    spend on its live agent instead, from which it reaches the rows
    above through the normal sub-agent fold.  The walk and the updates
    are one transaction, so a row cannot finish between the check and
    the add.

    Args:
        task_id: Primary key of the ``task_history`` row the spend is for.
        tokens: Tokens to add.
        cost: USD to add.
        steps: Steps to add.

    Returns:
        ``(updated, running)``: the finished rows updated, as ``(id,
        tokens, cost, steps)`` with their new totals, bottom-up; and the
        id of the first unfinished row on the chain (*task_id* itself
        when it is still running), or ``""`` when every row on the
        chain is finished or the chain reaches a missing row.
    """
    _flush_chat_events(task_id)
    db = _get_db()
    updated: list[tuple[str, int, float, int]] = []
    seen: set[str] = set()
    current = task_id
    with _rw_lock.write_lock(), _immediate_txn(db):
        while current and current not in seen:
            seen.add(current)
            row = db.execute(
                "SELECT end_ts, parent_task_id FROM task_history WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                return updated, ""
            if not _safe_int(row["end_ts"], 0):
                return updated, current
            totals = _add_usage_locked(db, current, tokens, cost, steps)
            if totals is not None:  # pragma: no branch — row read just above
                updated.append((current, *totals))
            current = _safe_str(row["parent_task_id"])
    return updated, ""


def _flag_int(value: object) -> int:
    """Coerce a truthy flag to ``1``/``0`` for an INTEGER column.

    Non-finite floats (NaN/Inf) are truthy but store ``0``, matching
    the numeric coercers.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    try:
        return 1 if value else 0
    except Exception:
        return 0


# ``extra`` key -> (task_history column, coercion to the column's type).
_EXTRA_COL_MAP: dict[str, tuple[str, Callable[[object], object]]] = {
    "model": ("model", _safe_str),
    "work_dir": ("work_dir", _safe_str),
    "version": ("version", _safe_str),
    "tokens": ("tokens", _safe_int),
    "cost": ("cost", _safe_float),
    "steps": ("steps", _safe_int),
    "is_parallel": ("is_parallel", _flag_int),
    "startTs": ("start_ts", _safe_int),
    "endTs": ("end_ts", _safe_int),
    "max_budget": ("max_budget", _safe_float),
    "sea": ("sea", _safe_str),
    "is_cron": ("is_cron", _flag_int),
}


def _save_task_extra(
    extra: dict[str, object],
    task_id: str | None = None,
) -> None:
    """Save extra metadata for a task into typed columns.

    Writes each known key from *extra* to its column in
    ``task_history``.  Unknown keys are silently ignored.  Keys absent
    from *extra* are NOT included in the UPDATE — so the
    ``is_favorite`` flag (set independently by
    :func:`_set_task_favorite`) is automatically preserved.

    The legacy nested ``{"subagent": {"parent_task_id": <uuid>}}``
    payload is translated to a write of the ``parent_task_id`` column
    (only when the payload contains the dotted shape).

    Args:
        extra: Dictionary of metadata to persist.
        task_id: Stable row id to update.
    """
    _flush_chat_events(task_id if is_task_history_id(task_id) else None)
    db = _get_db()
    finished = _safe_int(extra.get("endTs"), 0) > 0
    has_secret = _scan_finished_trajectory(db, task_id) if finished else None
    with _rw_lock.write_lock(), _immediate_txn(db):
        resolved = _resolve_task_id(db, task_id)
        if resolved is None:
            return
        pairs: list[tuple[str, object]] = []
        parent = _extract_parent_task_id(extra)
        if parent:
            pairs.append(("parent_task_id = ?", parent))
        for k, v in extra.items():
            if k == "is_favorite":
                raise ValueError(
                    "_save_task_extra does not write 'is_favorite'; "
                    "use _set_task_favorite() instead"
                )
            if k in ("parent_task_id", "subagent"):
                continue
            mapping = _EXTRA_COL_MAP.get(k)
            if mapping is None:
                continue
            col, coerce = mapping
            pairs.append((f"{col} = ?", coerce(v)))
        if "cost" in extra:
            pairs.append(("model_costs = ?", _own_model_costs(db, resolved)))
        if not pairs:
            return
        sets = [s for s, _ in pairs]
        vals = [v for _, v in pairs]
        vals.append(resolved)
        db.execute(
            f"UPDATE task_history SET {', '.join(sets)} WHERE id = ?", vals
        )
        if finished:
            _finalize_task_metadata(db, resolved, has_secret)


def _scan_finished_trajectory(db: sqlite3.Connection, task_id: str | None) -> bool | None:
    """Scan the trajectory of finished task *task_id* for a secret, holding no lock.

    :func:`_save_task_extra` calls this before it takes the write lock
    and its ``BEGIN IMMEDIATE``: the scan reads every persisted event of
    the task (seconds for a long one), and under those locks it stalled
    every reader and writer of the daemon and every other process
    writing ``history.db``.  A WAL reader needs neither lock.

    Args:
        db: The calling thread's connection.
        task_id: The finished row's id.

    Returns:
        The :func:`trajectory_has_secret` verdict, or ``None`` when no
        row has the id (the save then finds none either, or scans under
        its lock should the row appear in between).
    """
    if not is_task_history_id(task_id):
        return None
    row = db.execute(
        "SELECT task, result FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None
    return trajectory_has_secret(db, task_id, row["task"], row["result"])


def _own_model_costs(db: sqlite3.Connection, task_id: str) -> str:
    """Sum the task's own ``llm_call`` events per model, as JSON.

    Every billed LLM call of a task persists one ``llm_call`` event under
    the task's own id (a sub-agent's calls sit under the sub-agent's id),
    carrying the model that served it, its USD cost and its token
    counts.  The Spend panel splits a task's own spend by this record, so
    a task that switched models with ``set_model`` is not filed entirely
    under the model it was launched with.

    Args:
        db: The connection (inside the caller's transaction).
        task_id: The ``task_history`` row id.

    Returns:
        ``{"model": [cost, tokens], ...}`` as JSON, or ``""`` when the
        task has no ``llm_call`` event.
    """
    rows = db.execute(
        "SELECT json_extract(event_json, '$.model') AS m, "
        "SUM(json_extract(event_json, '$.cost')) AS c, "
        "SUM(COALESCE(json_extract(event_json, '$.input_tokens'), 0)"
        " + COALESCE(json_extract(event_json, '$.output_tokens'), 0)"
        " + COALESCE(json_extract(event_json, '$.cache_read'), 0)"
        " + COALESCE(json_extract(event_json, '$.cache_write'), 0)) AS t "
        "FROM events WHERE task_id = ? AND event_json LIKE '%\"llm_call\"%' "
        "AND json_extract(event_json, '$.type') = 'llm_call' GROUP BY m",
        (task_id,),
    ).fetchall()
    if not rows:
        return ""
    return json.dumps(
        {str(r[0] or "unknown"): [float(r[1] or 0.0), int(r[2] or 0)] for r in rows}
    )


def _finalize_task_metadata(
    db: sqlite3.Connection, task_id: str, has_secret: bool | None = None
) -> None:
    """Derive the finished task's ``tags`` and refresh its chat's summary row.

    Called inside :func:`_save_task_extra`'s transaction once the row
    carries an ``endTs``: the result (``_save_task_result`` runs first)
    decides the ``failed`` tag, the persisted events decide whether the
    trajectory holds a secret (``secret`` tag), and a listable
    (non-sub-agent) task refreshes the ``chat_summaries`` row of its chat.

    Args:
        db: The connection holding the write transaction.
        task_id: The finished row's id.
        has_secret: The secret verdict the caller computed before it
            took the write lock; ``None`` scans the events here.
    """
    row = db.execute(
        "SELECT task, result, success, chat_id, parent_task_id "
        "FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return
    write_task_tags(db, task_id, row, has_secret)
    if not row["parent_task_id"] and row["chat_id"]:
        upsert_chat_summary(db, _safe_str(row["chat_id"]))


def _chat_summaries(chat_ids: list[str]) -> dict[str, dict[str, object]]:
    """Return the ``chat_summaries`` rows of *chat_ids*. Thread-safe.

    Args:
        chat_ids: Chat session ids to look up; empty ids are skipped.

    Returns:
        Mapping of chat id to ``{"summary": str, "last_launched": int,
        "last_finished": int}`` (instants in epoch ms; ``last_finished``
        is 0 while the newest task runs); ``summary`` is the title the
        user typed in the CHATS panel when there is one, else the
        computed one.  Chats without a row are absent.
    """
    ids = sorted({c for c in chat_ids if c})
    if not ids:
        return {}
    db = _get_db()
    with _rw_lock.read_lock():
        placeholders = ",".join("?" * len(ids))
        rows = db.execute(
            "SELECT chat_id, summary, last_launched, last_finished, user_summary "
            f"FROM chat_summaries WHERE chat_id IN ({placeholders})",
            ids,
        ).fetchall()
    return {
        _safe_str(r["chat_id"]): {
            "summary": _safe_str(r["user_summary"]) or _safe_str(r["summary"]),
            "last_launched": _safe_int(r["last_launched"], 0),
            "last_finished": _safe_int(r["last_finished"], 0),
        }
        for r in rows
    }


def task_start_ms(task_id: str) -> int:
    """Return when task *task_id* started, in epoch milliseconds. Thread-safe.

    The row's ``start_ts``, or its insertion ``timestamp`` for a legacy
    row (:func:`launch_ms`).  Every run — each ``<task>`` block of a
    sequential submission, each ``run_parallel`` child — allocates its
    own row with its own start, so this dates the run itself.

    Args:
        task_id: Primary key of the ``task_history`` row.

    Returns:
        The launch instant, or 0 when the row is missing or carries no
        usable time.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        row = db.execute(
            "SELECT start_ts, timestamp FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    if row is None:
        return 0
    return launch_ms(row["start_ts"], row["timestamp"])


def _set_chat_summary(chat_id: str, summary: str) -> bool:
    """Store *summary* as the title the user gave chat *chat_id*.

    The text replaces the computed summary in the CHATS panel from now
    on, whatever tasks the chat runs later; an empty *summary* drops the
    user's title and the computed one shows again.  A chat that has no
    ``chat_summaries`` row yet (no finished task) gets one.

    Args:
        chat_id: The chat session id.
        summary: The user's title, whitespace-trimmed by the caller.

    Returns:
        True when a row was written, False for an empty *chat_id*.
    """
    if not chat_id:
        return False
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute(
            "INSERT INTO chat_summaries (chat_id, user_summary) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET user_summary = excluded.user_summary",
            (chat_id, summary),
        )
    return True


def _task_tree_ids(db: sqlite3.Connection, root_ids: list[str]) -> list[str]:
    """The ids of *root_ids* and every sub-task under them, by ``parent_task_id``."""
    found: list[str] = []
    seen: set[str] = set()
    frontier = [tid for tid in root_ids if tid]
    while frontier:
        batch = [tid for tid in frontier if tid not in seen]
        if not batch:
            break
        seen.update(batch)
        found.extend(batch)
        marks = ",".join("?" * len(batch))
        rows = db.execute(
            f"SELECT id FROM task_history WHERE parent_task_id IN ({marks})", batch
        ).fetchall()
        frontier = [str(row["id"]) for row in rows]
    return found


def _chat_tree_ids(db: sqlite3.Connection, chat_id: str) -> list[str]:
    """The ids of chat *chat_id*'s tasks and every sub-task under them."""
    rows = db.execute("SELECT id FROM task_history WHERE chat_id = ?", (chat_id,)).fetchall()
    return _task_tree_ids(db, [str(row["id"]) for row in rows])


def _delete_task_rows(db: sqlite3.Connection, task_ids: list[str], chat_id: str = "") -> int:
    """Under the write lock: drop *task_ids* and their events in one transaction.

    When *chat_id* is set, the chat's ``chat_summaries`` row goes in
    the same transaction.  Each deleted id is recorded in
    ``deleted_tasks`` in that transaction too (and the chat in
    ``deleted_chats``), so that the deletion travels to the other
    machines the database is synced with instead of the task coming
    back from them.

    Returns:
        The number of ``task_history`` rows removed.
    """
    removed = 0
    marks = ",".join("?" * len(task_ids))
    with _immediate_txn(db):
        if task_ids:
            db.execute(f"DELETE FROM events WHERE task_id IN ({marks})", task_ids)
            cursor = db.execute(f"DELETE FROM task_history WHERE id IN ({marks})", task_ids)
            removed = cursor.rowcount or 0
            now = time.time()
            db.executemany(
                "INSERT OR REPLACE INTO deleted_tasks (id, deleted_ts) VALUES (?, ?)",
                [(task_id, now) for task_id in task_ids],
            )
        if chat_id:
            db.execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))
            db.execute(
                "INSERT OR REPLACE INTO deleted_chats (chat_id, deleted_ts) VALUES (?, ?)",
                (chat_id, time.time()),
            )
    locked_call(_caches_lock, _drop_event_caches, set(task_ids))
    return removed


def _task_tree(task_id: str) -> list[str]:
    """The ids :func:`_delete_task` would remove: *task_id* and its sub-tasks.

    Args:
        task_id: A ``task_history`` row id (``""`` yields ``[]``).
    """
    if not task_id:
        return []
    db = _get_db()
    with _rw_lock.read_lock():
        return _task_tree_ids(db, [task_id])


def _chat_task_tree(chat_id: str) -> list[str]:
    """The ids :func:`_delete_chat` would remove: the chat's tasks and their sub-tasks.

    Args:
        chat_id: The chat session id (``""`` yields ``[]``).
    """
    if not chat_id:
        return []
    db = _get_db()
    with _rw_lock.read_lock():
        return _chat_tree_ids(db, chat_id)


def _delete_task(task_id: str) -> int:
    """Delete task *task_id*, its sub-tasks and all their events from the database.

    Sub-tasks are the rows whose ``parent_task_id`` chain leads to
    *task_id* (a ``run_parallel`` child's trajectory is meaningless
    without its parent).  The chat's ``chat_summaries`` row is kept:
    the chat may still hold other tasks.

    Args:
        task_id: Primary key of the ``task_history`` row to delete.

    Returns:
        The number of ``task_history`` rows removed (0 when no such task).
    """
    if not task_id:
        return 0
    _flush_chat_events(task_id)
    db = _get_db()
    with _rw_lock.write_lock():
        removed = _delete_task_rows(db, _task_tree_ids(db, [task_id]))
    _invalidate_chat_context_cache("")
    return removed


def _delete_chat(chat_id: str) -> int:
    """Delete chat *chat_id*: every task of the chat, their events and its summary.

    Args:
        chat_id: The chat session id.

    Returns:
        The number of ``task_history`` rows removed (0 for an unknown chat).
    """
    if not chat_id:
        return 0
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        removed = _delete_task_rows(db, _chat_tree_ids(db, chat_id), chat_id)
    _invalidate_chat_context_cache(chat_id)
    return removed


_event_queue: queue.Queue = queue.Queue()
_event_writer_thread: threading.Thread | None = None
_event_writer_lock = threading.RLock()
_event_writer_stop = threading.Event()
_journal_lock = threading.RLock()
_pending_cond = threading.Condition()
_pending_by_task: dict[str, int] = {}
_next_seq_cache: dict[str, int] = {}
_marked_has_events: set[str] = set()
_caches_db_key: tuple[str, tuple[int, int] | None] | None = None
_caches_lock = threading.RLock()


def _maybe_reset_caches(
    current_path: str, file_id: tuple[int, int] | None = None,
) -> None:
    """Clear seq/has_events caches when the database is swapped out.

    Triggered both by a ``_DB_PATH`` reassignment (test fixtures) and
    by an on-disk replacement of the SAME pathname (*file_id* — the
    file's ``(st_dev, st_ino)`` — changes): the cached sequence
    counters and has-events marks were seeded from the previous
    database and are meaningless for the new one.  A *file_id* of
    :data:`_FILE_ID_UNKNOWN` (a transient ``stat`` failure) proves
    nothing and leaves the caches alone, exactly as :func:`_get_db`
    leaves its connection alone.
    """
    if file_id is _FILE_ID_UNKNOWN:
        return
    key = (current_path, file_id)
    if _caches_db_key == key:
        return
    if locked_call(_caches_lock, _reset_caches_locked, key):
        _invalidate_chat_context_cache("")


def _reset_caches_locked(key: tuple[str, tuple[int, int] | None]) -> bool:
    """Clear the seq/has_events caches unless *key* is already current.

    The caller holds ``_caches_lock``.

    Returns:
        True when the caches were cleared, False when *key* was current.
    """
    global _caches_db_key
    if _caches_db_key == key:
        return False
    _next_seq_cache.clear()
    _marked_has_events.clear()
    _caches_db_key = key
    return True

_BATCH_MAX = 256
_BATCH_WINDOW_S = 0.020


def _start_event_writer() -> None:
    """Lazily spawn the background event writer thread (idempotent)."""
    if _event_writer_thread is not None and _event_writer_thread.is_alive():
        return
    locked_call(_event_writer_lock, _start_event_writer_locked)


def _start_event_writer_locked() -> None:
    """Spawn the writer thread unless one is alive; caller holds ``_event_writer_lock``."""
    global _event_writer_thread, _event_writer_stop
    if _event_writer_thread is not None and _event_writer_thread.is_alive():
        return
    stop = threading.Event()
    t = threading.Thread(
        target=_event_writer_loop,
        args=(stop,),
        name="kiss-event-writer",
        daemon=True,
    )
    _event_writer_stop = stop
    _event_writer_thread = t
    t.start()


def _event_writer_loop(stop: threading.Event) -> None:
    """Drain the event queue in batches and persist them.

    Args:
        stop: This writer's private stop event (set by
            ``_stop_event_writer``); private so a concurrent
            ``_start_event_writer`` can never un-stop it.
    """
    while not stop.is_set():
        try:
            first = _event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if first is None:
            _event_queue.task_done()
            if stop.is_set():
                return
            continue
        batch: list[tuple[str, str, float, str]] = [first]
        deadline = time.monotonic() + _BATCH_WINDOW_S
        shutdown_pending = False
        while len(batch) < _BATCH_MAX:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = _event_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                _event_queue.task_done()
                shutdown_pending = stop.is_set()
                break
            batch.append(item)
        try:
            _persist_batch_with_retry(batch)
        finally:
            _release_pending(batch)
        if shutdown_pending:
            return


def _persist_batch_with_retry(batch: list[tuple[str, str, float, str]]) -> None:
    """Persist *batch*, retrying a few times before giving up.

    Transient failures (e.g. external lock contention beyond the busy
    timeout) must not silently drop events; a bounded retry keeps the
    writer from stalling forever while still recovering the common
    case.  Events are only abandoned — with an ``error`` log — after
    every attempt fails.
    """
    attempts = 4
    for attempt in range(attempts):
        try:
            _write_event_batch(batch)
            return
        except Exception:
            if attempt < attempts - 1:
                logger.warning("event writer batch failed; retrying", exc_info=True)
                time.sleep(0.05 * (attempt + 1))
            else:
                _journal_failed_events(batch, attempts)


def _failed_events_path(db_path: str) -> str:
    """Return the journal path holding unwritable events for *db_path*."""
    return db_path + ".failed_events.jsonl"


def _final_results_path(db_path: str) -> str:
    """Return the sidecar path journalling final task results for *db_path*."""
    return db_path + ".final_results.jsonl"


#: Journalled final results older than this are pruned at sweep time.
#: By then the row either kept its committed result (the journal entry
#: was never needed) or an earlier sweep already restored it.
_FINAL_RESULTS_MAX_AGE_S = 30 * 24 * 3600.0


def _journal_final_result(
    task_id: str, result: str, success: int | None = None,
) -> None:
    """Best-effort append of a task's final result to a sidecar file.

    SQLite alone is not a sufficient store for the terminal ``result``
    of a task: in the 2026-09-12 incident the database entered a
    ``disk I/O error`` state, every WAL frame committed during the
    final minutes of a task was silently discarded when the next
    process opened the file, and the startup sweep
    (:func:`_recover_orphaned_tasks`) — seeing the creation sentinel
    where the successful result used to be — mislabeled the finished
    task as "process killed".

    The remedy is a plain append-only JSON-lines sidecar next to the
    database: a medium that does not share SQLite's failure modes.  The
    sweep consults it before declaring a sentinel row killed and
    restores the journalled result instead (see
    :func:`_load_final_results`).

    The append is serialised against concurrent appends and the
    sweep-time prune in other processes by
    :func:`_journal_file_lock`; a record that could not take the lock
    in time is spilled to its own file (see :data:`_SPILL_SUFFIX`).
    Failures are logged and swallowed: the journal is a safety net,
    and the primary database write has already succeeded when this
    runs.

    Args:
        task_id: History row id whose result was just saved.
        result: The result text that was written to the database.
        success: The row's ``success`` column value saved with it
            (``0``, ``1`` or ``None``), restored alongside the text.
    """
    sidecar = _final_results_path(str(_DB_PATH))
    line = json.dumps({
        "task_id": task_id, "result": result, "success": success, "ts": time.time(),
    })
    locked_call(_journal_lock, _append_final_result, sidecar, task_id, line)


def _append_final_result(sidecar: str, task_id: str, line: str) -> None:
    """Append *line* to the final-results journal; caller holds ``_journal_lock``."""
    try:
        with _journal_file_lock(sidecar) as locked:
            if not locked:
                _spill_journal_lines(sidecar, [line + "\n"])
                return
            # A process killed mid-append can leave a torn last
            # record with no trailing newline; appending directly
            # onto that fragment would fuse THIS record into one
            # malformed line and lose it too.  Terminate any torn
            # tail first so only the torn record is discarded.
            prefix = ""
            if os.path.exists(sidecar) and os.path.getsize(sidecar):
                with open(sidecar, "rb") as tail:
                    tail.seek(-1, os.SEEK_END)
                    if tail.read(1) != b"\n":
                        prefix = "\n"
            with open(sidecar, "a", encoding="utf-8") as stream:
                stream.write(prefix + line + "\n")
    except OSError:
        logger.warning(
            "final result for task %s could not be journalled in %s",
            task_id, sidecar, exc_info=True,
        )


#: Suffix of a journal record written while a peer process kept the
#: journal lock past its timeout (:data:`_JOURNAL_LOCK_TIMEOUT_S` for a
#: final result, :data:`_JOURNAL_APPEND_TIMEOUT_S` for a failed-events
#: batch).  The record must not be appended to the sidecar itself
#: without the lock: the holder may be the prune, which replaces the
#: final-results sidecar's inode, or a replayer renaming or unlinking
#: the failed-events sidecar, and a record landing in the old inode
#: would be erased with it.  Nor may a failed-events batch be
#: published as a claimed snapshot: a claim key taken now sorts BEFORE
#: the key the live sidecar gets when it is claimed later, which would
#: replay the spilled rows ahead of the rows already waiting there.
#: So the record goes to its own ``<sidecar>.spill-<time ns>-<pid>-
#: <uuid>`` file — written under a private name and renamed into
#: place, so no reader ever sees a partial record — which
#: :func:`_load_final_results` reads alongside its sidecar and the next
#: locked prune folds back into it, and which the next holder of the
#: failed-events lock claims right behind the live sidecar
#: (:func:`_claim_journal_rows`).
_SPILL_SUFFIX = ".spill-"


def _journal_spills(sidecar: str) -> list[str]:
    """Return the spill files of *sidecar*, oldest first (see :data:`_SPILL_SUFFIX`).

    Args:
        sidecar: Path returned by :func:`_final_results_path` or
            :func:`_failed_events_path`.
    """
    directory = os.path.dirname(sidecar) or "."
    prefix = os.path.basename(sidecar) + _SPILL_SUFFIX
    try:
        names = os.listdir(directory)
    except OSError:  # pragma: no cover — unreadable journal directory
        return []
    return sorted(os.path.join(directory, name) for name in names if name.startswith(prefix))


def _spill_journal_lines(sidecar: str, lines: list[str]) -> str:
    """Write *lines* to a fresh spill file of *sidecar*, bypassing its lock.

    The file is renamed into place complete, so a claim never moves a
    half-written one, and its zero-padded ``time_ns`` orders this
    process's successive spills.

    Args:
        sidecar: The live sidecar the records belong to.
        lines: The journal records, each newline-terminated.

    Returns:
        The spill path.
    """
    unique = f"{os.getpid()}-{uuid.uuid4().hex}"
    writing = f"{sidecar}.writing-{unique}"
    with open(writing, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    spill = f"{sidecar}{_SPILL_SUFFIX}{time.time_ns():020d}-{unique}"
    os.replace(writing, spill)
    return spill


def _journal_lines(path: str) -> Iterator[str]:
    """Yield the lines of the journal file at *path*; a missing or unreadable file has none."""
    try:
        with open(path, encoding="utf-8") as stream:
            yield from stream
    except OSError:
        return


def _load_final_results(sidecar: str) -> dict[str, dict[str, object]]:
    """Load the journalled final results from *sidecar* and its spill files.

    Args:
        sidecar: Path returned by :func:`_final_results_path`.

    Returns:
        Mapping of task id to its most recently journalled
        ``{"result": text, "success": 0 | 1 | None}`` (a record from
        before the verdict was journalled loads with ``None``).  The
        record with the latest ``ts`` wins, whichever file holds it —
        the sidecar and a spill file do not order each other.  Missing
        or unreadable files and corrupt lines (a torn write from a
        killed process) yield/skip to an empty or partial map.
    """
    results: dict[str, dict[str, object]] = {}
    journalled_at: dict[str, float] = {}
    for path in (sidecar, *_journal_spills(sidecar)):
        for raw in _journal_lines(path):
            try:
                entry = json.loads(raw)
                task_id = entry["task_id"]
                result = entry["result"]
                success = entry.get("success")
                ts = _safe_float(entry.get("ts"), 0.0)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if not (isinstance(task_id, str) and isinstance(result, str)):
                continue
            if ts < journalled_at.get(task_id, ts):
                continue
            journalled_at[task_id] = ts
            results[task_id] = {
                "result": result,
                "success": success if success in (0, 1) else None,
            }
    return results


def _prune_final_results_journal(sidecar: str) -> None:
    """Drop journal entries older than :data:`_FINAL_RESULTS_MAX_AGE_S`.

    Called after each orphan sweep so the sidecar stays a bounded
    incident log (one small line per finished task) instead of growing
    forever.  Spill files are folded into the sidecar at the same
    time.  The rewrite happens under :func:`_journal_file_lock` and
    lands via an atomic rename, so a concurrent append in another
    process is either retained or ordered after the prune — never
    torn.  All failures are logged and swallowed.

    Args:
        sidecar: Path returned by :func:`_final_results_path`.
    """
    cutoff = time.time() - _FINAL_RESULTS_MAX_AGE_S
    with _journal_lock:
        try:
            with _journal_file_lock(sidecar) as locked:
                # A rewrite without the lock could drop a peer's
                # concurrent append; the next sweep prunes instead.
                if not locked:
                    return
                spills = _journal_spills(sidecar)
                if not spills and not os.path.exists(sidecar):
                    return
                kept: list[tuple[float, str]] = []
                dropped = False
                for path in (sidecar, *spills):
                    for raw in _journal_lines(path):
                        try:
                            entry_ts = float(json.loads(raw)["ts"])
                        except (ValueError, TypeError, KeyError):
                            dropped = True
                            continue
                        if entry_ts >= cutoff:
                            kept.append((entry_ts, raw.rstrip("\n") + "\n"))
                        else:
                            dropped = True
                if not dropped and not spills:
                    return
                kept.sort()
                tmp_path = sidecar + ".pruning"
                with open(tmp_path, "w", encoding="utf-8") as stream:
                    stream.writelines(raw for _ts, raw in kept)
                os.replace(tmp_path, sidecar)
                for spill in spills:
                    os.unlink(spill)
        except OSError:
            logger.warning(
                "final-results journal %s could not be pruned",
                sidecar, exc_info=True,
            )


#: Suffix of a journal file a replayer has taken ownership of.  The
#: live sidecar is renamed to ``<sidecar>.consumed-<pid>-<uuid>``
#: before it is replayed, so the file that is finally deleted is
#: exactly the one that was written to the database — never a file
#: another process has appended to since.
_JOURNAL_CONSUMED_SUFFIX = ".consumed-"

# Zero-padded width of the monotonic claim key embedded in a claimed
# snapshot's filename (see _claim_journal_snapshots).  Wide enough for
# any time.time_ns() value; also what distinguishes a claim key from
# the pid in a legacy ``.consumed-<pid>-<uuid>`` name.
_CLAIM_KEY_WIDTH = 20


#: Longest wait for a peer process's :func:`_journal_file_lock`.  The
#: lock is taken on task threads (every final-result save flushes and
#: replays), so a peer wedged under it must cost them a bounded time,
#: not the peer's lifetime.
_JOURNAL_LOCK_TIMEOUT_S = 5.0

#: Longest wait of the event-writer thread for the journal lock before
#: a failed batch is spilled instead of appended
#: (:func:`_journal_failed_events`).  Longer than the task-thread wait
#: because an append out of place costs ordering, and long enough to
#: outlast a healthy replayer's single database write attempt.
_JOURNAL_APPEND_TIMEOUT_S = _JOURNAL_LOCK_TIMEOUT_S * 6


@contextmanager
def _journal_file_lock(
    sidecar: str, timeout: float = _JOURNAL_LOCK_TIMEOUT_S,
) -> Iterator[bool]:
    """Hold an inter-process lock on *sidecar* for the whole block.

    The database and its journal are shared by every Sorcar process on
    the machine (the ``kiss-web`` daemon, a ``kiss`` CLI run, a daemon
    restart), so the module-level :data:`_journal_lock` — a plain
    thread lock — cannot order an append in one process against a
    replay in another.  Without that ordering a replayer deletes
    batches a peer appended after the replayer read the file, and two
    replayers write the same batch twice.

    The lock lives in a sibling ``<sidecar>.lock`` file rather than in
    the journal itself, because the journal is renamed and deleted
    while the lock is held.  Closing the handle releases the kernel
    lock, and the kernel releases it anyway if the process dies.

    Args:
        sidecar: Path of the journal file being appended or replayed.
        timeout: Longest wait for a peer's lock, in seconds.

    Yields:
        ``True`` with the lock held for the duration of the block.
        ``False`` when the lock file cannot be opened or a peer kept
        the lock for *timeout*.  The block must then leave the shared
        journal file alone: the peer may be replacing or unlinking its
        inode, so an unlocked append could land in a file nobody reads
        again.  A replay or prune skips and is retried by the next
        flush or sweep; an append writes its record elsewhere.
    """
    try:
        handle = open(sidecar + ".lock", "a+", encoding="utf-8")
    except OSError:  # pragma: no cover — unwritable journal directory
        yield False
        return
    try:
        try:
            lock_exclusive(handle, timeout=timeout)
        except TimeoutError:
            logger.warning(
                "journal lock %s.lock held by another process for over %.0fs; "
                "an append goes to a spill file, a replay or prune waits for "
                "the next round", sidecar, timeout,
            )
            yield False
            return
        yield True
    finally:
        handle.close()


def _journal_failed_events(
    batch: list[tuple[str, str, float, str]], attempts: int,
) -> None:
    """Preserve a permanently unwritable batch in a durable sidecar file.

    A batch that failed every write attempt must not be silently
    acknowledged and lost — ``_flush_chat_events`` would then report
    completion for events that were never persisted.  The rows are
    appended as JSON lines to ``<db>.failed_events.jsonl`` and replayed
    by :func:`_replay_failed_events` as soon as the database accepts
    writes again, so the transcript is recoverable instead of merely
    inspectable.

    Each row is journalled next to the database it was PRODUCED
    against (its own ``origin_db_path``), not next to whichever
    database happens to be active now: after a database swap the two
    differ, and a journal written next to the new database would be
    replayed into it and dropped.

    The append is serialised against every other process's append and
    replay by :func:`_journal_file_lock`, so a batch can never be
    written into a file another process is in the middle of consuming.
    It waits for that lock up to :data:`_JOURNAL_APPEND_TIMEOUT_S`:
    this runs on the event-writer thread, which every
    ``_flush_chat_events`` (so every final-result save on a task
    thread) waits for, and a peer wedged under the lock must not stall
    those for its lifetime.  A holder that is a healthy replayer lets
    go within its one database write attempt (SQLite's busy timeout),
    which the wait covers; past it the rows go to a private spill file
    (:func:`_spill_journal_lines`) that the next locked append or
    replay claims as a snapshot behind the live sidecar
    (:func:`_claim_journal_rows`), so they stay durable and in order.
    An append that finds such spills claims them first, so its own
    rows land in a fresh sidecar that sorts after them.
    """
    by_origin: dict[str, list[tuple[str, str, float, str]]] = {}
    for row in batch:
        by_origin.setdefault(row[3], []).append(row)
    with _journal_lock:
        for origin, rows in by_origin.items():
            sidecar = _failed_events_path(origin)
            lines = [
                json.dumps({
                    "task_id": task_id,
                    "event_json": event_json,
                    "timestamp": timestamp,
                    "origin_db_path": origin_path,
                }) + "\n"
                for task_id, event_json, timestamp, origin_path in rows
            ]
            try:
                with _journal_file_lock(sidecar, timeout=_JOURNAL_APPEND_TIMEOUT_S) as locked:
                    if locked:
                        if _journal_spills(sidecar):
                            _claim_journal_rows(sidecar)
                        with open(sidecar, "a", encoding="utf-8") as stream:
                            stream.writelines(lines)
                        target = sidecar
                    else:
                        target = _spill_journal_lines(sidecar, lines)
                logger.error(
                    "%d chat events could not be written after %d attempts; "
                    "journalled in %s for replay",
                    len(rows), attempts, target, exc_info=True,
                )
            except OSError:
                logger.error(
                    "dropping %d chat events after %d failed write attempts "
                    "(sidecar %s also unwritable)",
                    len(rows), attempts, sidecar, exc_info=True,
                )


def _replay_failed_events() -> None:
    """Re-insert events journalled while the database was unwritable.

    Called from :func:`_flush_chat_events`, so the recovery happens on
    the very next write-ordering barrier after the database becomes
    writable again — before ``_task_has_transcript_events`` decides that
    a task produced no transcript and a stub stream must be synthesized in
    its place.

    A snapshot is only removed once every row it holds has landed in
    the database; a still-failing replay leaves it on disk for the next
    attempt.

    Concurrency: the journal is shared by every Sorcar process, so the
    whole claim → write → delete sequence runs under
    :func:`_journal_file_lock`, and the rows are claimed by *renaming*
    the sidecar aside first.  Deleting the renamed snapshot can
    therefore never destroy a batch a peer appended in the meantime —
    that batch goes to a fresh sidecar — and no batch is ever replayed
    by two processes at once.

    A snapshot whose replay fails stays under its claimed name — it is
    deliberately NOT renamed back to the live sidecar.  A restored
    file would receive LATER appends, mixing the directory's oldest
    and newest rows into one file whose last-append mtime postdates
    snapshots holding middle-aged rows; no per-snapshot ordering can
    replay such a mixture chronologically.  Claimed names are still
    discovered by :func:`_journal_has_pending_rows` and
    :func:`_claim_journal_snapshots`, so nothing is lost.  The loop
    also stops at the first failure: replay fails only when the
    database refuses the write, so every later (newer) snapshot would
    fail too — and committing a newer snapshot before an older one
    retries would hand newer rows the lower seqs.
    """
    path = _failed_events_path(_current_db_path())
    if not _journal_has_pending_rows(path):
        # The overwhelmingly common case: nothing ever failed to write.
        # Checked before locking so a healthy database never pays for
        # the lock file or the lock itself.
        return
    locked_call(_journal_lock, _replay_journal_files, path)


def _replay_journal_files(path: str) -> None:
    """Replay every snapshot of the journal at *path*; caller holds ``_journal_lock``.

    Without the cross-process lock nothing is replayed: a peer may be
    replaying the same snapshots, and the next flush tries again.
    """
    with _journal_file_lock(path) as locked:
        if not locked:
            return
        for snapshot in _claim_journal_snapshots(path):
            if not _replay_journal_snapshot(snapshot):
                break


def _journal_has_pending_rows(path: str) -> bool:
    """Return True when a journal, a spill or an unfinished snapshot exists.

    Args:
        path: The live sidecar path for the active database.
    """
    if os.path.exists(path):
        return True
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path)
    prefixes = (base + _JOURNAL_CONSUMED_SUFFIX, base + _SPILL_SUFFIX)
    try:
        return any(name.startswith(prefixes) for name in os.listdir(directory))
    except OSError:  # pragma: no cover — unreadable journal directory
        return False


def _claim_journal_snapshots(path: str) -> list[str]:
    """Take ownership of *path* and return every snapshot to replay.

    The live sidecar is renamed to a unique
    ``.consumed-<claim key>-<pid>-<uuid>`` sibling, which is what
    makes the later delete safe.  Snapshots a previous replayer left
    behind — it crashed, or the database was still refusing writes —
    are picked up too, so a rename is never a way to lose events.

    Replay assigns fresh monotonically increasing seqs in list order,
    so the order must be CHRONOLOGICAL.  The claim key — a wall-clock
    nanosecond timestamp forced past every claim key already in the
    directory (claims are serialised by :func:`_journal_file_lock`, so
    the maximum is race-free) — records exactly that: every row of an
    earlier-claimed snapshot predates every row of a later-claimed one
    because the earlier sidecar was renamed aside before the later
    sidecar received its first append.  Unlike ``st_mtime_ns``, the
    key is immutable once assigned, total (never a tie), and immune to
    filesystem timestamp granularity.

    Args:
        path: The live sidecar path for the active database.

    Returns:
        Snapshot paths to replay, oldest snapshot first.  Caller holds
        :func:`_journal_file_lock`.
    """
    directory = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + _JOURNAL_CONSUMED_SUFFIX
    _claim_journal_rows(path)
    try:
        names = os.listdir(directory)
    except OSError:  # pragma: no cover — unreadable journal directory
        return []
    return sorted(
        (
            os.path.join(directory, name)
            for name in names
            if name.startswith(prefix)
        ),
        key=lambda snapshot: _snapshot_order_key(snapshot, prefix),
    )


def _claim_journal_rows(path: str) -> None:
    """Rename the live sidecar at *path*, then its spill files, into claimed snapshots.

    The spills (:func:`_spill_journal_lines`) were written while a peer
    held the lock, after every row then in the live sidecar, so they
    get the claim keys right behind it, oldest spill first; rows this
    process appends to the fresh sidecar afterwards sort after them
    all.  Caller holds :func:`_journal_file_lock`.

    Args:
        path: The live sidecar path for the active database.
    """
    directory = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + _JOURNAL_CONSUMED_SUFFIX
    pending = [path] if os.path.exists(path) else []
    for source in pending + _journal_spills(path):
        try:
            os.replace(source, _new_snapshot_path(directory, prefix))
        except OSError:  # pragma: no cover — unrenamable journal
            logger.warning("could not claim journal %s", source, exc_info=True)


def _parse_claim_key(name: str, prefix: str) -> int | None:
    """Extract the monotonic claim key from a snapshot file *name*.

    Args:
        name: Basename of a ``.consumed-*`` snapshot file.
        prefix: The live sidecar basename plus the consumed suffix.

    Returns:
        The claim key, or ``None`` for a legacy
        ``.consumed-<pid>-<uuid>`` name that predates claim keys (a
        pid never has the key's fixed 20-digit width).
    """
    head = name[len(prefix):].split("-", 1)[0]
    if len(head) == _CLAIM_KEY_WIDTH and head.isdigit():
        return int(head)
    return None


def _new_snapshot_path(directory: str, prefix: str) -> str:
    """Return a fresh ``<prefix><claim key>-<pid>-<uuid>`` snapshot path in *directory*."""
    return os.path.join(directory, (
        f"{prefix}{_next_claim_key(directory, prefix):020d}"
        f"-{os.getpid()}-{uuid.uuid4().hex}"
    ))


def _next_claim_key(directory: str, prefix: str) -> int:
    """Return a claim key greater than every key already in *directory*.

    Starts from ``time.time_ns()`` and bumps past any existing key, so
    the sequence stays strictly increasing even across a wall-clock
    step backwards.  Caller holds :func:`_journal_file_lock`.

    Args:
        directory: The journal directory.
        prefix: The live sidecar basename plus the consumed suffix.
    """
    key = time.time_ns()
    try:
        names = os.listdir(directory)
    except OSError:  # pragma: no cover — unreadable journal directory
        return key
    for name in names:
        if not name.startswith(prefix):
            continue
        existing = _parse_claim_key(name, prefix)
        if existing is not None and existing >= key:
            key = existing + 1
    return key


def _snapshot_order_key(snapshot: str, prefix: str) -> tuple[int, int, str]:
    """Sort key ordering claimed journal snapshots oldest-first.

    Args:
        snapshot: Path of a ``.consumed-*`` snapshot file.
        prefix: The live sidecar basename plus the consumed suffix.

    Returns:
        ``(claim key, mtime ns, path)``.  A legacy snapshot without a
        claim key falls back to its last-append mtime, which compares
        correctly against real claim keys: a snapshot's claim always
        happens after its own last append and before any younger
        sidecar's first append.  The path tie-breaks (only reachable
        between legacy names on a coarse-timestamp filesystem, where
        no chronology survives at all).  A vanished file sorts first;
        replay tolerates it (an unreadable snapshot is skipped).
    """
    try:
        mtime = os.stat(snapshot).st_mtime_ns
    except OSError:  # pragma: no cover — concurrently removed snapshot
        mtime = 0
    claim = _parse_claim_key(os.path.basename(snapshot), prefix)
    return (mtime if claim is None else claim, mtime, snapshot)


def _replay_journal_snapshot(snapshot: str) -> bool:
    """Write one claimed journal *snapshot* to the database and delete it.

    Args:
        snapshot: Path of a ``.consumed-*`` file this process owns.

    Returns:
        True when the snapshot was replayed (or held nothing usable)
        and removed; False when the database still refuses the write,
        in which case the snapshot is kept for the next attempt.

    Exactly-once: the SQLite commit and the file unlink cannot be
    atomic, so the batch is committed together with a marker row in
    ``replayed_journals`` keyed by the snapshot's unique claim name.
    A crash (or unlink failure) between the two steps leaves the file
    behind, but the next replayer finds the marker and only removes
    the file — the events are never inserted twice.
    """
    try:
        with open(snapshot, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError:  # pragma: no cover — unreadable snapshot
        return True
    batch: list[tuple[str, str, float, str]] = []
    for line in lines:
        try:
            record = json.loads(line)
            batch.append((
                str(record["task_id"]),
                str(record["event_json"]),
                _safe_float(record["timestamp"], 0.0),
                str(record["origin_db_path"]),
            ))
        except (ValueError, TypeError, KeyError):
            logger.warning("skipping malformed journal line", exc_info=True)
    marker = os.path.basename(snapshot)
    if batch:
        try:
            _write_event_batch(batch, replay_marker=marker)
        except Exception:
            logger.warning(
                "replay of %d journalled events failed; keeping %s",
                len(batch), snapshot, exc_info=True,
            )
            return False
        logger.info("replayed %d journalled chat events", len(batch))
    try:
        os.unlink(snapshot)
    except OSError:  # pragma: no cover — concurrent unlink
        # The marker row stays: with the file still on disk, a later
        # replay must keep skipping the already-committed rows.
        return True
    _delete_replay_marker(marker)
    return True


def _delete_replay_marker(marker: str) -> None:
    """Prune *marker* from ``replayed_journals`` (best-effort).

    Called only AFTER the snapshot file it names was unlinked: from
    that point no replayer can rediscover the snapshot, so the marker
    has done its job.  A failure here merely leaves a stale row.

    Args:
        marker: The snapshot's basename, as stored by
            :func:`_write_event_batch_locked`.
    """
    try:
        db = _get_db()
        with _rw_lock.write_lock(), _immediate_txn(db):
            db.execute(
                "DELETE FROM replayed_journals WHERE snapshot = ?",
                (marker,),
            )
    except Exception:  # pragma: no cover — pruning is best-effort
        logger.warning(
            "could not prune replay marker %s", marker, exc_info=True,
        )


def _is_active_db(origin_db_path: str) -> bool:
    """Whether *origin_db_path* names the active database.

    A row journalled before version 2026.10.2 carries the database's
    former path (``sorcar.db``): the active database's own name before
    the rename (:func:`_adopt_legacy_db_name`).  The symlink the rename
    leaves behind resolves it to ``history.db``; where there is none
    (Windows, or a user deleted it) the old name is accepted as long as
    no file exists under it -- an actual ``sorcar.db`` beside the active
    database is an unrelated database, not its former name.
    """
    current_path = _current_db_path()
    if origin_db_path == current_path:
        return True
    origin = os.path.realpath(origin_db_path)
    if origin == os.path.realpath(current_path):
        return True
    legacy_path = os.path.join(os.path.dirname(current_path), _LEGACY_DB_NAME)
    return origin == os.path.realpath(legacy_path) and not os.path.exists(legacy_path)


def _write_event_batch(
    batch: list[tuple[str, str, float, str]],
    replay_marker: str | None = None,
) -> None:
    """Persist a batch of (task_id, event_json, timestamp, origin_db_path) rows.

    Rows whose ``origin_db_path`` no longer matches the active
    ``_DB_PATH`` are dropped: their numeric ``task_id`` belongs to the
    database that was active when they were enqueued, so writing them
    into the current database would attach them to an unrelated task
    that merely shares the same row id.

    Args:
        batch: The rows to insert.
        replay_marker: For journal replays, the claimed snapshot's
            basename; committed with the rows so a replay is
            exactly-once (see :func:`_replay_journal_snapshot`).
            ``None`` on the ordinary event-writer path.
    """
    if not batch:
        return
    batch = [row for row in batch if _is_active_db(row[3])]
    if not batch:
        return
    db = _get_db()
    task_ids = {tid for (tid, _ej, _ts, _op) in batch}
    with _rw_lock.write_lock():
        locked_call(_caches_lock, _write_event_batch_retrying, db, batch, task_ids, replay_marker)


def _write_event_batch_retrying(
    db: sqlite3.Connection,
    batch: list[tuple[str, str, float, str]],
    task_ids: set[str],
    replay_marker: str | None,
) -> None:
    """Write *batch*, retrying once after a cross-process seq collision.

    The caller holds the write lock and ``_caches_lock``.  The
    transaction itself is rolled back by :func:`_immediate_txn` on
    ANY exception — including the runner's Stop, which lands on the
    task thread when the batch is a journal replay run by
    ``_flush_chat_events`` — so this only drops the caches.
    """
    try:
        try:
            _write_event_batch_locked(db, batch, task_ids, replay_marker)
        except sqlite3.IntegrityError:
            # ``idx_ev_task_seq`` refused a duplicate ``(task_id,
            # seq)``: another PROCESS (journal replay from a CLI run,
            # a second daemon) inserted seqs this process's
            # ``_next_seq_cache`` never saw.  Re-read ``MAX(seq)``
            # from the database and retry ONCE; a second refusal is a
            # real error and propagates like any other failure.
            _drop_event_caches(task_ids)
            _write_event_batch_locked(db, batch, task_ids, replay_marker)
    except BaseException:
        _drop_event_caches(task_ids)
        raise


def _drop_event_caches(task_ids: set[str]) -> None:
    """Drop the seq/has_events cache entries of a rolled-back batch.

    The seq cache may have advanced past rows that were rolled back,
    and another process may have moved the on-disk ``MAX(seq)`` in
    the meantime; dropping the entries forces the next attempt to
    recompute them from the database.

    Args:
        task_ids: Tasks whose cache entries must be invalidated.
    """
    for tid in task_ids:
        _next_seq_cache.pop(tid, None)
        _marked_has_events.discard(tid)


def _write_event_batch_locked(
    db: sqlite3.Connection,
    batch: list[tuple[str, str, float, str]],
    task_ids: set[str],
    replay_marker: str | None = None,
) -> None:
    """Insert *batch* inside one explicit transaction.

    Caller holds ``_rw_lock.write_lock()`` and ``_caches_lock`` and
    invalidates the seq caches on any failure; the transaction is
    :func:`_immediate_txn`, which rolls back on any exception
    (including an injected Stop), so a mid-batch error can never
    diverge the cache from the database or leave SQLite's write lock
    held on this thread's connection.

    When *replay_marker* is given and already present in
    ``replayed_journals``, a previous replayer committed this very
    snapshot and died before removing its file: nothing is inserted,
    so the transcript never gets duplicate events.
    """
    with _immediate_txn(db):
        if replay_marker is not None:
            seen = db.execute(
                "SELECT 1 FROM replayed_journals WHERE snapshot = ?",
                (replay_marker,),
            ).fetchone()
            if seen is not None:
                return
            db.execute(
                "INSERT INTO replayed_journals (snapshot, timestamp) "
                "VALUES (?, ?)",
                (replay_marker, time.time()),
            )
        for tid in task_ids:
            if tid not in _next_seq_cache:
                exists = db.execute(
                    "SELECT 1 FROM task_history WHERE id = ?", (tid,),
                ).fetchone()
                if exists is None:
                    continue
                row = db.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq "
                    "FROM events WHERE task_id = ?",
                    (tid,),
                ).fetchone()
                _next_seq_cache[tid] = row["next_seq"] if row else 0
        rows: list[tuple[str, int, str, float]] = []
        for tid, ev_json, ts, _op in batch:
            seq = _next_seq_cache.get(tid)
            if seq is None:
                continue
            _next_seq_cache[tid] = seq + 1
            rows.append((tid, seq, ev_json, ts))
        db.executemany(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        to_mark = [
            tid for tid in task_ids
            if tid in _next_seq_cache and tid not in _marked_has_events
        ]
        if to_mark:
            placeholders = ",".join("?" * len(to_mark))
            db.execute(
                f"UPDATE task_history SET has_events = 1 "
                f"WHERE id IN ({placeholders})",
                to_mark,
            )
            _marked_has_events.update(to_mark)


def _queue_chat_event(
    event: dict[str, object] | str,
    task_id: str,
    origin_db_path: str | None = None,
) -> None:
    """Asynchronously persist an event for a known task_id.

    Sub-microsecond enqueue from the producer's perspective.  A
    background writer thread (started lazily) batches enqueued events
    and persists them with one transaction per batch.

    Callers that need ordering relative to a subsequent synchronous
    write to ``task_history`` (``_save_task_result``, ``_save_task_extra``)
    must call ``_flush_chat_events()`` first.

    Args:
        event: The event dict to persist, or its ``json.dumps``
            encoding (lets a caller encode before taking a lock).
        task_id: Stable ``task_history`` row id.  Must be non-None.
        origin_db_path: Database path *task_id* was resolved against.
            Defaults to the active ``_DB_PATH``.  The background
            writer drops the event if the active database has changed
            since, because the numeric id would then point at an
            unrelated task in the new database (see
            :func:`_current_db_path`).
    """
    # Build the complete queue item BEFORE reserving: ``json.dumps``
    # raises TypeError on a non-serialisable value, and a reservation
    # with no matching queue item is never released by the writer, so
    # ``_flush_chat_events(task_id)`` would then spin forever.
    item = (
        task_id,
        event if isinstance(event, str) else json.dumps(event),
        time.time(),
        origin_db_path or _current_db_path(),
    )
    _reserve_pending(task_id)
    try:
        # The queue is unbounded, so this never blocks; the guard only
        # rolls the reservation back if an injected stop (or any other
        # BaseException) lands between the reserve and the publication.
        _event_queue.put_nowait(item)
    except BaseException:
        _unreserve_pending(task_id)
        raise
    _start_event_writer()


@locked(_pending_cond)
def _reserve_pending(task_id: str) -> None:
    """Record one queued-but-unwritten event for *task_id*."""
    _pending_by_task[task_id] = _pending_by_task.get(task_id, 0) + 1


def _decrement_pending_locked(task_id: str) -> None:
    """Drop one pending reservation of *task_id*; caller holds ``_pending_cond``."""
    remaining = _pending_by_task.get(task_id, 0) - 1
    if remaining > 0:
        _pending_by_task[task_id] = remaining
    else:
        _pending_by_task.pop(task_id, None)


def _unreserve_pending(task_id: str) -> None:
    """Roll back one :func:`_reserve_pending` whose item never reached the queue.

    Unlike :func:`_release_pending` this must NOT call
    ``_event_queue.task_done()`` — nothing was enqueued.
    """
    locked_call(_pending_cond, _unreserve_pending_locked, task_id)


def _unreserve_pending_locked(task_id: str) -> None:
    """Drop one reservation of *task_id* and wake waiters; caller holds ``_pending_cond``."""
    _decrement_pending_locked(task_id)
    _pending_cond.notify_all()


def _release_pending(batch: list[tuple[str, str, float, str]]) -> None:
    """Mark every event in *batch* as no longer pending and wake waiters."""
    locked_call(_pending_cond, _release_batch_locked, batch)
    for _ in batch:
        _event_queue.task_done()


def _release_batch_locked(batch: list[tuple[str, str, float, str]]) -> None:
    """Drop the reservations of *batch* and wake waiters; caller holds ``_pending_cond``."""
    for task_id, _ev, _ts, _origin in batch:
        _decrement_pending_locked(task_id)
    _pending_cond.notify_all()


def _pending_count(task_id: str | None) -> int:
    """Return the number of unwritten events, optionally for one task."""
    if task_id is None:
        return int(_event_queue.unfinished_tasks)
    return _pending_by_task.get(task_id, 0)


def _flush_chat_events(task_id: str | None = None) -> None:
    """Block until queued events have been persisted.

    Waits on the writer's condition variable rather than polling, and
    re-plays any events that were journalled while the database was
    unwritable so the caller's "everything is persisted" assumption
    actually holds.

    Safe to call when no events are queued (returns immediately).  MUST
    be called BEFORE acquiring ``_rw_lock.write_lock()`` from the same
    thread — the writer thread also takes that lock per batch, so
    calling this while holding the write lock would deadlock.

    Args:
        task_id: When given, only that task's events are waited for.
            Write ordering only matters within one task, so a caller
            updating task A must not be delayed by a high-volume run
            in task B.  ``None`` waits for every queued event.
    """
    while not locked_call(_pending_cond, _pending_drained, task_id):
        _start_event_writer()
    _replay_failed_events()


def _pending_drained(task_id: str | None) -> bool:
    """Return True when no event of *task_id* is pending; else wait one tick.

    The caller holds ``_pending_cond``.  The wait is skipped when no
    writer is alive, so the caller starts one before retrying.
    """
    if _pending_count(task_id) == 0:
        return True
    writer = _event_writer_thread
    if writer is not None and writer.is_alive():
        _pending_cond.wait(0.05)
    return False


def _stop_event_writer() -> None:
    """Drain and stop the writer thread.  Used by ``_close_db``/tests.

    A producer can enqueue an event *after* the drain below but
    *before* the old writer observes its stop flag, in which case the
    stopped writer exits without consuming it.  The loop re-checks the
    queue after each stopped writer and drains again (starting a fresh
    writer if needed) until nothing is left unfinished, so no queued
    event is ever stranded.
    """
    while True:
        _flush_chat_events()
        t, stop = locked_call(_event_writer_lock, _writer_and_stop)
        if t is not None:
            stop.set()
            _event_queue.put_nowait(None)  # unbounded queue: never raises queue.Full
            t.join(timeout=5)
            if t.is_alive():
                logger.warning(
                    "event writer thread did not stop within 5s; "
                    "deferring writer cleanup until it exits"
                )
                return
            locked_call(_event_writer_lock, _forget_writer, t)
        if not _event_queue.unfinished_tasks:
            break
    locked_call(_caches_lock, _clear_seq_caches)


def _clear_seq_caches() -> None:
    """Drop the seq/has_events caches and their database key; caller holds ``_caches_lock``."""
    global _caches_db_key
    _next_seq_cache.clear()
    _marked_has_events.clear()
    _caches_db_key = None


def _writer_and_stop() -> tuple[threading.Thread | None, threading.Event]:
    """Return the current writer thread and its stop flag; caller holds ``_event_writer_lock``."""
    return _event_writer_thread, _event_writer_stop


def _forget_writer(t: threading.Thread) -> None:
    """Clear ``_event_writer_thread`` if it is still *t*; caller holds ``_event_writer_lock``."""
    global _event_writer_thread
    if _event_writer_thread is t:
        _event_writer_thread = None


def _drain_events_at_exit() -> None:
    """``atexit`` hook: flush queued events before interpreter teardown.

    The writer is a daemon thread, so a process exiting right after
    ``JsonPrinter`` enqueued events would otherwise lose them.  Thread
    creation can fail during interpreter shutdown, hence best-effort.
    """
    try:
        _stop_event_writer()
    except Exception:  # pragma: no cover — interpreter shutdown edge
        logger.debug("event drain at exit failed", exc_info=True)


atexit.register(_drain_events_at_exit)


def _append_chat_event(
    event: dict[str, object],
    task_id: str | None = None,
    origin_db_path: str | None = None,
) -> None:
    """Append a single event to the saved chat events for a task.

    Synchronous: completes the write before returning.  Callers that
    can tolerate asynchronous persistence should prefer
    ``_queue_chat_event`` instead.

    Args:
        event: The event dict to append.
        task_id: Stable row id to update.
        origin_db_path: Database path *task_id* was resolved against
            (see :func:`_queue_chat_event`).  Late asynchronous
            callers pass the path captured when the task completed
            so the event is
            dropped — instead of attached to an unrelated task with
            the same row id — if the active database has changed.
    """
    if origin_db_path is not None and origin_db_path != _current_db_path():
        return
    db = _get_db()
    with _rw_lock.read_lock():
        resolved = _resolve_task_id(db, task_id)
    if resolved is None:
        return
    _queue_chat_event(event, resolved, origin_db_path)
    # Only this task's events: a side-channel charge calls this under
    # TASK_USAGE_LOCK, and waiting for every task's queue there would
    # park another task's finishing write behind a high-volume run.
    _flush_chat_events(resolved)


def _task_has_transcript_events(task_id: str) -> bool:
    """Return whether any TRANSCRIPT events are persisted for *task_id*.

    Flushes the asynchronous event queue first so events enqueued by a
    recording printer (which land on the events table via the background
    writer) are visible before the check.  The run's single
    ``task_settings`` metadata event, which is broadcast (and
    persisted) BEFORE the agent produces any output, is ignored.  Used
    by :meth:`ChatSorcarAgent._persist_replay_events_if_missing` to
    decide whether it must synthesize a minimal replayable event stream
    (prompt + result) for a run that happened outside a chat page —
    without a recording printer that would have persisted the live
    event stream; a run that failed right after allocation must still
    get its synthesized events even though the metadata event already
    exists.

    Args:
        task_id: Stable ``task_history`` row id.

    Returns:
        ``True`` if at least one persisted event is not a
        ``task_settings`` event.
    """
    _flush_chat_events(task_id)
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            "SELECT event_json FROM events WHERE task_id = ? "
            "ORDER BY seq ASC LIMIT 3",
            (task_id,),
        ).fetchall()
    for row in rows:
        try:
            event = json.loads(row["event_json"])
        except (json.JSONDecodeError, TypeError):
            return True
        if not isinstance(event, dict):
            return True
        if event.get("type") != "task_settings":
            return True
    # A run broadcasts task_settings once, so three metadata-only rows
    # cannot occur; anything beyond the scanned window is transcript.
    return len(rows) >= 3


def _select_event_rows(db: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    """Return the raw event rows of *task_id* in seq order, oversized rows clipped.

    Callers must hold ``_rw_lock.read_lock()`` (or ``write_lock()``);
    every transcript loader (this module's and
    :func:`task_digest.load_task`) runs only this SELECT under the lock
    and calls :func:`_decode_event_rows` outside it, because the lock is
    writer-preferring and a decode held under it would stall every
    write in the process for its duration.
    Rows longer than :data:`MAX_EVENT_ROW_CHARS` (tool output recorded
    before the per-call cap, up to 100 MB each) are never transferred
    to Python: SQLite returns only their ``type`` and the first
    :data:`MAX_EVENT_TEXT_CHARS` characters of their ``text``, and
    :func:`_decode_event_rows` replaces the row by a clipped event (or
    a ``system_output`` note naming the event type and size when it
    carried no ``text``).

    Args:
        db: Active database connection.
        task_id: Primary key of the ``task_history`` row.
    """
    return db.execute(
        "SELECT timestamp, length(event_json) AS size, "
        "CASE WHEN length(event_json) <= ? THEN event_json END AS event_json, "
        "CASE WHEN length(event_json) > ? AND json_valid(event_json) "
        "THEN json_extract(event_json, '$.type') END AS big_type, "
        "CASE WHEN length(event_json) > ? AND json_valid(event_json) "
        "THEN substr(json_extract(event_json, '$.text'), 1, ?) END AS big_text, "
        "CASE WHEN length(event_json) > ? AND json_valid(event_json) "
        "AND length(json_remove(event_json, '$.text')) <= ? "
        "THEN json_remove(event_json, '$.text') END AS big_rest "
        "FROM events WHERE task_id = ? ORDER BY seq",
        (
            MAX_EVENT_ROW_CHARS,
            MAX_EVENT_ROW_CHARS,
            MAX_EVENT_ROW_CHARS,
            MAX_EVENT_TEXT_CHARS,
            MAX_EVENT_ROW_CHARS,
            MAX_EVENT_REST_CHARS,
            task_id,
        ),
    ).fetchall()


def _decode_event_rows(event_rows: list[sqlite3.Row]) -> list[dict[str, object]]:
    """Decode the rows of :func:`_select_event_rows`; needs no lock.

    Each surviving event dict has its ``_timestamp`` field injected
    from the matching ``events.timestamp`` column.  Rows whose
    ``event_json`` fails to decode are silently dropped (logged at
    DEBUG level).

    Args:
        event_rows: The rows, already fetched.

    Returns:
        List of event dicts with ``_timestamp`` injected.
    """
    events: list[dict[str, object]] = []
    for r in event_rows:
        if r["event_json"] is None:
            ev = _clipped_event_row(
                r["big_type"], r["big_text"], r["big_rest"], r["size"],
            )
            if ev is None:
                continue
            ev["_timestamp"] = r["timestamp"]
            events.append(ev)
            continue
        try:
            ev = json.loads(r["event_json"])
            ev["_timestamp"] = r["timestamp"]
            events.append(ev)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Exception caught", exc_info=True)
    return events


#: An event row longer than this is clipped by SQLite instead of being
#: loaded (see :func:`_select_event_rows`).  Generous enough
#: for a ``tool_result`` carrying a few embedded screenshots.
MAX_EVENT_ROW_CHARS = 4 * 1024 * 1024
#: How much of an oversized row's ``text`` survives.
MAX_EVENT_TEXT_CHARS = 200_000
#: An oversized row whose fields other than ``text`` fit in this many
#: characters keeps them all (its bulk was the text); otherwise only
#: the type is known and the row becomes a note.
MAX_EVENT_REST_CHARS = 64 * 1024


def _clipped_event_row(
    big_type: object, big_text: object, big_rest: object, size: int,
) -> dict[str, object] | None:
    """Build the stand-in for an event row too large to load.

    Args:
        big_type: The row's ``type`` (``None`` for malformed JSON).
        big_text: The first :data:`MAX_EVENT_TEXT_CHARS` characters of
            its ``text`` (``None`` when the event has no text).
        big_rest: The event without its ``text``, when that fits in
            :data:`MAX_EVENT_REST_CHARS` (``None`` otherwise).
        size: The row's length in characters.

    Returns:
        A clipped copy of a text-bearing event (its other fields kept
        when *big_rest* is available), a ``system_output`` note for
        any other, or ``None`` for a malformed row.
    """
    if not isinstance(big_type, str):
        return None
    if isinstance(big_text, str):
        rest: dict[str, object] = (
            json.loads(big_rest) if isinstance(big_rest, str) else {"type": big_type}
        )
        rest["text"] = big_text + (
            f"\n... [truncated: this {big_type} event was {size // 1_000_000} MB] ...\n"
        )
        return rest
    return {
        "type": "system_output",
        "text": f"[a {big_type} event of {size // 1_000_000} MB was not replayed]",
    }


def _events_session_dict(
    row: sqlite3.Row, event_rows: list[sqlite3.Row],
) -> dict[str, object]:
    """Build the replay-session dict shared by every chat-events loader.

    Decodes the event rows of :func:`_select_event_rows` and needs
    no lock: the loaders call it after leaving ``_rw_lock.read_lock()``
    so the JSON work of a large transcript never blocks the writer.

    Args:
        row: A ``task_history`` row from ``_HISTORY_SELECT``.
        event_rows: Its raw event rows (``[]`` for a deferred replay).

    Returns:
        Dict with ``task``, ``task_id``, ``events``, ``chat_id``, and
        ``extra`` keys (``task`` through :func:`_safe_str`: the dict is
        JSON-bound, and a hand-edited database can hold a BLOB there).
    """
    return {
        "task": _safe_str(row["task"]),
        "task_id": str(row["id"]),
        "events": _decode_event_rows(event_rows),
        "chat_id": str(row["chat_id"] or ""),
        "extra": _row_to_extra_json(row) or "",
    }


def _load_events_session_row(
    where_sql: str,
    params: tuple[object, ...],
    with_events: bool = True,
) -> dict[str, object] | None:
    """Load one ``task_history`` row and its events as a session dict.

    Shared engine of :func:`_load_latest_chat_events_by_chat_id` and
    :func:`_load_chat_events_by_task_id` — runs ``_HISTORY_SELECT``
    plus *where_sql* under the read lock and converts the first
    matching row via :func:`_events_session_dict`.

    Args:
        where_sql: SQL appended to ``_HISTORY_SELECT`` (the
            WHERE/ORDER BY/LIMIT clauses).
        params: Bind parameters for *where_sql*.
        with_events: ``False`` skips the events table (``events`` is
            then ``[]``).

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` when no row matches.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        row = db.execute(_HISTORY_SELECT + where_sql, params).fetchone()
        if row is None:
            return None
        event_rows = _select_event_rows(db, str(row["id"])) if with_events else []
    return _events_session_dict(row, event_rows)


def _load_latest_chat_events_by_chat_id(
    chat_id: str,
    with_events: bool = True,
) -> dict[str, object] | None:
    """Load the latest task and its events for a chat session.

    Finds the most recent NON-sub-agent task in the given chat session
    and returns its task description string and recorded events.
    Sub-agent rows (``extra.subagent`` present, identified by a
    non-NULL ``parent_task_id`` column) share the parent's ``chat_id``
    and are persisted AFTER the parent row, so a chat-id-only lookup
    (e.g. the chat page's post-restart ``resumeSession``) must skip them —
    otherwise a restored parent tab would replay the last sub-agent's
    events and be styled as a sub-agent tab.  Sub-agent rows are only
    ever loaded explicitly by task id
    (:func:`_load_chat_events_by_task_id`).

    Args:
        chat_id: The string chat session identifier.
        with_events: ``False`` skips the events table (``events`` is
            then ``[]``).

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if chat_id is ``""`` or has no
        non-sub-agent tasks.
    """
    if not chat_id:
        return None
    return _load_events_session_row(
        f"WHERE chat_id = ? AND {NOT_SUBAGENT} "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (chat_id,),
        with_events,
    )


def _load_chat_events_by_task_id(
    task_id: str,
) -> dict[str, object] | None:
    """Load a specific task and its events by the task row ID.

    Unlike ``_load_latest_chat_events_by_chat_id`` which always picks
    the most recent task in a chat session, this loads the exact task
    identified by *task_id*.

    Args:
        task_id: The primary key of the ``task_history`` row.

    Returns:
        A dict with ``task`` (str), ``task_id`` (str), ``events``
        (list of event dicts), ``chat_id`` (str), and ``extra`` (str,
        JSON metadata), or ``None`` if no such row exists.
    """
    return _load_events_session_row("WHERE id = ?", (task_id,))


def _load_all_chat_events_by_chat_id(
    chat_id: str,
    max_json_bytes: int | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Load every task of a chat session and its events, oldest first.

    Powers the daemon's ``shareChatTasks`` command: the chat page
    exports a chat as one standalone HTML page and needs the
    transcripts of ALL of the chat's tasks, not just the latest one
    (:func:`_load_latest_chat_events_by_chat_id`) — earlier tasks are
    no longer in the page's DOM after a reload, because a session
    replay repaints only one task.  Sub-agent rows are skipped exactly
    like the latest-task lookup skips them: they replay inside their
    parent's transcript, not as chat tasks of their own.

    The rows are walked newest first so a *max_json_bytes* budget can
    stop BEFORE fetching an oversized chat's older transcripts — the
    newest ones are the ones the page cannot redraw from its own
    DOM — instead of materializing every event and discarding the
    surplus afterwards.  Each task's rows are fetched under their own
    read lock and decoded (and measured) outside it, so a long export
    never holds the writer-preferring lock for more than one SELECT.

    Args:
        chat_id: The string chat session identifier.
        max_json_bytes: Optional byte budget; each task is charged the
            UTF-8 length of its JSON encoding, and loading stops at
            the first task that does not fit.  ``None`` loads all.

    Returns:
        ``(tasks, truncated)``.  *tasks* is ordered by ``timestamp
        ASC, rowid ASC`` (the order the tasks ran in), each dict with
        ``task`` (str), ``task_id`` (str), ``events`` (list of event
        dicts), ``chat_id`` (str), and ``extra`` (str, JSON metadata);
        empty when *chat_id* is ``""`` or names no non-sub-agent
        tasks.  *truncated* is True when the budget dropped at least
        one (oldest) task.
    """
    if not chat_id:
        return [], False
    out: list[dict[str, object]] = []
    truncated = False
    budget = max_json_bytes
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            _HISTORY_SELECT
            + f"WHERE chat_id = ? AND {NOT_SUBAGENT} "
            "ORDER BY timestamp DESC, rowid DESC",
            (chat_id,),
        ).fetchall()
    for row in rows:
        with _rw_lock.read_lock():
            event_rows = _select_event_rows(db, str(row["id"]))
        entry = _events_session_dict(row, event_rows)
        if budget is not None:
            budget -= len(json.dumps(entry).encode("utf-8"))
            if budget < 0:
                truncated = True
                break
        out.append(entry)
    out.reverse()
    return out, truncated


def _load_subagent_rows_by_parent_task_id(
    parent_task_id: str,
    with_events: bool = True,
) -> list[dict[str, object]]:
    """Return persisted sub-agent rows whose parent is *parent_task_id*.

    Used by :meth:`AgentServer._replay_session` when the user clicks
    a parent task in the history sidebar: every sub-agent fanned out
    by the parent's ``run_parallel`` tool call is reopened in its own
    sub-agent tab so the loaded view mirrors the live execution
    layout.

    A sub-agent row is identified by its ``parent_task_id`` column
    matching *parent_task_id* — the dedicated column written for a
    run submitted with ``parentTaskId`` (a ``run_agent`` sub-task; the
    ``extra`` payload's ``subagent`` object is synthesized back from
    this column by :func:`_row_to_extra_json`).

    Args:
        parent_task_id: Primary key (32-hex UUID TEXT id) of the
            parent ``task_history`` row.
        with_events: ``False`` skips the events table (every row's
            ``events`` is then ``[]``): the sub-agent tabs are being
            announced, their transcripts deferred.

    Returns:
        List of dicts ordered by ``rowid`` ASC (the order in which
        the parent enqueued sub-agents).  Each dict has ``task_id``
        (str, the row's UUID TEXT id), ``task`` (str), ``chat_id``
        (str), ``events`` (list of event dicts), and ``extra`` (str,
        JSON metadata synthesized by :func:`_row_to_extra_json`).
        Empty list when no sub-agent rows exist.
    """
    if not isinstance(parent_task_id, str) or not parent_task_id:
        return []
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            _HISTORY_SELECT
            + "WHERE parent_task_id = ? ORDER BY rowid ASC",
            (parent_task_id,),
        ).fetchall()
        sessions = [
            (r, _select_event_rows(db, str(r["id"])) if with_events else [])
            for r in rows
        ]
    return [_events_session_dict(r, event_rows) for r, event_rows in sessions]


def _get_adjacent_task_by_chat_id(
    chat_id: str, current_task_id: str | None, direction: str
) -> dict[str, object] | None:
    """Return the adjacent task within a chat session, relative to *current_task_id*.

    Args:
        chat_id: The string chat session identifier.
        current_task_id: The DB row id of the current task used to find
            the reference timestamp within the chat.  Using the row id
            (rather than the task description string) ensures that
            duplicate task texts within the same chat are navigated
            unambiguously.
        direction: ``"prev"`` for the earlier task, ``"next"`` for the
            later task in the same chat session.

    Returns:
        A dict with ``task`` (str), ``task_id`` (str) and ``events``
        (list of event dicts), or ``None`` if no adjacent task exists.
    """
    if not chat_id or current_task_id is None:
        return None
    db = _get_db()
    with _rw_lock.read_lock():
        row = db.execute(
            "SELECT rowid, id, timestamp FROM task_history "
            "WHERE id = ? AND chat_id = ?",
            (current_task_id, chat_id),
        ).fetchone()
        if not row:
            return None
        ts = row["timestamp"]
        cur_rowid = row["rowid"]
        cmp, order = ("<", "DESC") if direction == "prev" else (">", "ASC")
        adj = db.execute(
            _HISTORY_SELECT
            + "WHERE chat_id = ? "
            f"AND (timestamp {cmp} ? OR (timestamp = ? AND rowid {cmp} ?)) "
            f"AND {NOT_SUBAGENT} "
            f"ORDER BY timestamp {order}, rowid {order} LIMIT 1",
            (chat_id, ts, ts, cur_rowid),
        ).fetchone()
        if not adj:
            return None
        event_rows = _select_event_rows(db, str(adj["id"]))
    # The synthesized extra lets the replay reply builder attach
    # a task_settings event for rows that predate the event.
    return _events_session_dict(adj, event_rows)


def _load_chat_context(chat_id: str) -> list[_HistoryEntry]:
    """Load all tasks and results for a chat session in chronological order.

    Sub-agent rows (those with a non-empty ``parent_task_id`` column —
    set on the task row of every run submitted with ``parentTaskId``,
    a ``run_agent`` sub-task) are filtered out via the shared
    ``NOT_SUBAGENT`` SQL predicate.  Sub-agent
    tasks/results are an internal implementation detail of the
    parent's ``run_parallel`` tool call; surfacing them in the chat
    context would (a) pollute the LLM's "Previous tasks and results"
    augmentation built by
    :meth:`ChatSorcarAgent.build_chat_prompt` with N copies of every
    fan-out task, and (b) cause the parent tab's history panel to
    list the sub-agent rows alongside the parent task that already
    summarises them.

    Args:
        chat_id: The string chat session identifier.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered by
        timestamp ascending (oldest first), excluding sub-agent rows.
    """
    if not chat_id:
        return []
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            "SELECT task, result FROM task_history "
            f"WHERE chat_id = ? AND {NOT_SUBAGENT} "
            "ORDER BY timestamp ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
    return [{"task": r["task"], "result": r["result"]} for r in rows]


def _load_task_chain_context(task_id: str) -> list[_HistoryEntry]:
    """Load the tasks and results along a task's parent chain.

    Starting at *task_id*, follows ``parent_task_id`` links upward
    (task → parent → grandparent → …) and returns the collected
    entries in chronological order (root ancestor first, the *task_id*
    row last) — the same shape :func:`_load_chat_context` returns, so
    :meth:`ChatSorcarAgent.build_chat_prompt` can consume either
    interchangeably.  Used when a tab is opened by a specific task id
    and the first task subsequently issued in that tab needs the
    opened task's lineage — rather than its whole chat — as context.

    Traversal is cycle-safe (each row id is visited at most once) and
    stops silently at a missing row or an empty ``parent_task_id``.

    Args:
        task_id: The ``task_history.id`` to start the traversal at.

    Returns:
        List of dicts with ``task`` and ``result`` keys, ordered
        oldest ancestor first.  Empty when *task_id* is empty or has
        no persisted row.
    """
    if not task_id:
        return []
    db = _get_db()
    rows: list[sqlite3.Row] = []
    seen: set[str] = set()
    current = task_id
    with _rw_lock.read_lock():
        while current and current not in seen:
            seen.add(current)
            row = db.execute(
                "SELECT task, result, parent_task_id FROM task_history "
                "WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            rows.append(row)
            current = _safe_str(row["parent_task_id"])
    return [{"task": r["task"], "result": r["result"]} for r in reversed(rows)]


def _load_chat_context_text(chat_id: str) -> str:
    """Return the joined task+result text for *chat_id* with caching.

    Concatenates the ``task`` and ``result`` strings of every entry
    returned by :func:`_load_chat_context` with newline separators.
    The joined string is cached in ``_chat_context_text_cache`` and
    automatically invalidated by :func:`_add_task` and
    :func:`_save_task_result` so callers (notably the ghost-text
    autocomplete, which calls this on every keystroke) never re-run
    the SQL or rejoin the text while the chat context is unchanged.

    Args:
        chat_id: The string chat session identifier.  Empty string
            short-circuits to ``""``.

    Returns:
        Newline-joined concatenation of every prior task and result
        in the session, or ``""`` when *chat_id* is empty or no prior
        rows exist.
    """
    if not chat_id:
        return ""
    current_path = _current_db_path()
    _maybe_reset_caches(current_path, _db_file_identity(current_path))
    cached, snapshot_gen = locked_call(_chat_context_cache_lock, _cached_chat_context, chat_id)
    if cached is not None:
        return cached
    parts: list[str] = []
    for entry in _load_chat_context(chat_id):
        task = entry.get("task")
        result = entry.get("result")
        if isinstance(task, str):
            parts.append(task)
        if isinstance(result, str):
            parts.append(result)
    text = "\n".join(parts)
    locked_call(_chat_context_cache_lock, _store_chat_context, chat_id, text, snapshot_gen)
    return text


def _cached_chat_context(chat_id: str) -> tuple[str | None, int]:
    """Return the cached text of *chat_id* (or None) and the cache generation."""
    return _chat_context_text_cache.get(chat_id), _chat_context_cache_gen


def _store_chat_context(chat_id: str, text: str, snapshot_gen: int) -> None:
    """Cache *text* for *chat_id* unless the cache changed since *snapshot_gen*."""
    if _chat_context_cache_gen == snapshot_gen:
        _chat_context_text_cache[chat_id] = text


def _load_model_usage() -> dict[str, int]:
    """Return model usage counts as ``{model_name: count}``."""
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute("SELECT model, count FROM model_usage").fetchall()
        return {r["model"]: r["count"] for r in rows}


def _load_last_model() -> str:
    """Return the name of the most recently selected model, or ``""``.

    The last-selected model is a persistent **user preference** stored
    in the brand home's ``config.json`` (see ``kiss.core.user_config``,
    under the ``last_model`` key) — *not*
    in the SQLite ``model_usage`` table, which now tracks only per-model
    usage counts.
    """
    from kiss.core.user_config import load_config

    return str(load_config().get("last_model", "") or "")


def _save_last_model(model: str) -> None:
    """Persist the selected model name as a user preference.

    Writes the ``last_model`` key to the brand home's ``config.json`` (atomic).
    Does **not** touch the SQLite usage counters.

    Passes *only* the changed key to :func:`save_config` — whose
    locked read-merge-write overlays every DEFAULTS key present in its
    argument onto a fresh re-read of the file — so a stale full-config
    snapshot taken here can never clobber a concurrent update to an
    unrelated key (e.g. a settings toggle saved from another thread).

    Args:
        model: The model name to save as the last-selected model.
    """
    from kiss.core.user_config import save_config

    save_config({"last_model": model})


def _record_model_usage(model: str) -> None:
    """Increment a model's usage counter and mark it as last-used.

    The usage ``count`` lives in the SQLite ``model_usage`` table; the
    "last selected" pointer is persisted separately to
    ``config.json`` via :func:`_save_last_model`.
    """
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute(
            "INSERT INTO model_usage (model, count) VALUES (?, 1) "
            "ON CONFLICT(model) DO UPDATE SET count = count + 1",
            (model,),
        )
    _save_last_model(model)


def _load_file_usage() -> dict[str, int]:
    """Return file usage counts ordered oldest-first (by last_used).

    The returned dict preserves insertion order so that callers can
    derive recency from key position.
    """
    db = _get_db()
    with _rw_lock.read_lock():
        rows = db.execute(
            "SELECT path, count FROM file_usage ORDER BY last_used ASC"
        ).fetchall()
        return {r["path"]: r["count"] for r in rows}


def _record_file_usage(path: str) -> None:
    """Increment the access count for a file path atomically.

    The table is capped at ``_MAX_FILE_USAGE_ENTRIES`` rows: a new
    path at the cap evicts the least recently used one first (see
    :func:`_evict_for_new_key`).
    """
    now = time.time()
    _capped_upsert(
        "file_usage", "path", path, _MAX_FILE_USAGE_ENTRIES, "last_used ASC",
        "INSERT INTO file_usage (path, count, last_used) VALUES (?, 1, ?) "
        "ON CONFLICT(path) DO UPDATE SET count = count + 1, last_used = ?",
        (path, now, now),
    )


def _record_frequent_task(task: str) -> None:
    """Increment the run-count of *task* and refresh its timestamp.

    Upserts a row in the ``frequent_tasks`` table so that subsequent
    calls with the same *task* increment its ``count`` and update its
    ``timestamp`` to ``time.time()``.

    The table is capped at ``_MAX_FREQUENT_TASKS`` rows.  When inserting
    a brand-new task would exceed the cap, the row with the lowest
    ``count`` (and, on a count tie, the oldest ``timestamp``) is
    evicted before the insert completes (see :func:`_evict_for_new_key`).

    Args:
        task: The task description string.  Empty strings are ignored.
    """
    if not task:
        return
    now = time.time()
    _capped_upsert(
        "frequent_tasks", "task", task, _MAX_FREQUENT_TASKS,
        "count ASC, timestamp ASC",
        "INSERT INTO frequent_tasks (task, count, timestamp) VALUES (?, 1, ?) "
        "ON CONFLICT(task) DO UPDATE SET count = count + 1, timestamp = ?",
        (task, now, now),
    )
