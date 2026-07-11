# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (batch A / A3): stale per-tab merge lock guards the wrong review.

``_handle_web_merge_action`` holds the per-tab :class:`asyncio.Lock`
across all of ``_apply_web_merge_action`` — including the final
``await _run_cmd("all-done")``.  The completing holder H pops BOTH the
merge state and the lock-dict entry (``_pop_merge_state``) and then
stays suspended in ``all-done`` while a waiter W is still queued on
the now-popped lock object.  If, during that window, the backend
finishes and a NEW ``merge_data`` for the SAME tab is registered
(``_register_merge_state`` never mints a lock), W eventually acquires
the STALE lock object, re-fetches the state, finds the NEW review, and
proceeds — while any fresh ``mergeAction`` mints a FRESH lock via
``_merge_action_lock``.  Two coroutines then mutate the same
``_WebMergeState`` (and rewrite the same files) under two DIFFERENT
lock objects: both read the same ``current()`` hunk, both reject it,
and the other hunk's resolution is lost.

The fix: after acquiring the per-tab lock, re-verify (under
``_merge_states_lock``) that it is still the registered lock for the
tab and retry with the current one when it rotated.

The test uses a real server, real files and a REAL executor — a
:class:`ThreadPoolExecutor` subclass that only adds test-controlled
synchronization (holding the ``all-done`` backend call until the new
review is registered, and a rendezvous so the two racing rejects
overlap), following the ``_CoordExecutor`` precedent of
``test_merge_action_concurrency_race.py``.  No mocks.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.web_server import RemoteAccessServer


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a per-test directory."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class _GateExecutor(ThreadPoolExecutor):
    """Real executor with test-controlled synchronization points.

    * ``all_done_gate``: every backend ``all-done`` dispatch waits for
      this event, keeping the completing holder suspended (lock held,
      state already popped) while the test registers the next review.
    * ``first_reject_gate``: consumed on the FIRST hunk-reject
      dispatch — H's reject blocks here until the test proves W has
      queued on the per-tab lock, so H cannot race ahead, pop the
      state and finish before W ever enqueues.  Subsequent rejects do
      not wait (the attribute is cleared on consumption).
    * ``reject_barrier``: when set, hunk-reject work rendezvouses here
      so two racing rejects provably overlap; a lone reject times out
      after 2 s and proceeds (the barrier then stays broken and all
      later waits return immediately).
    """

    def __init__(self) -> None:
        super().__init__(max_workers=8)
        self.all_done_gate = threading.Event()
        self.first_reject_gate: threading.Event | None = None
        self.reject_barrier: threading.Barrier | None = None

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Submit *fn*, first waiting on any gate that applies to it."""
        is_all_done = (
            getattr(fn, "__name__", "") == "_handle_command"
            and args
            and isinstance(args[0], dict)
            and args[0].get("action") == "all-done"
        )
        inner = fn.func if isinstance(fn, partial) else fn
        is_reject = getattr(inner, "__name__", "").startswith("_reject_")
        gate = self.all_done_gate if is_all_done else None
        first_gate: threading.Event | None = None
        if is_reject and self.first_reject_gate is not None:
            first_gate = self.first_reject_gate
            self.first_reject_gate = None
        barrier = self.reject_barrier if is_reject else None

        def _wrapped() -> object:
            if gate is not None:
                gate.wait(timeout=10)
            if first_gate is not None:
                first_gate.wait(timeout=10)
            if barrier is not None:
                try:
                    barrier.wait(timeout=2.0)
                except threading.BrokenBarrierError:
                    pass
            return fn(*args, **kwargs)

        return super().submit(_wrapped)


def _one_hunk_data(work: Path, stem: str) -> dict[str, Any]:
    """Single-file, single-hunk merge data (the first, completing review)."""
    current = work / f"{stem}.txt"
    base = work / f"{stem}_base.txt"
    lines = "".join(f"line{i}\n" for i in range(10))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    return {
        "work_dir": str(work),
        "files": [
            {
                "name": f"{stem}.txt",
                "current": str(current),
                "base": str(base),
                "target": str(current),
                "hunks": [{"bs": 2, "bc": 1, "cs": 2, "cc": 1}],
            }
        ],
    }


def _two_hunk_data(work: Path, stem: str) -> dict[str, Any]:
    """Single-file, two-hunk merge data (the second, racing review)."""
    current = work / f"{stem}.txt"
    base = work / f"{stem}_base.txt"
    lines = "".join(f"line{i}\n" for i in range(20))
    current.write_text(
        lines.replace("line2\n", "edit2\n").replace("line10\n", "edit10\n"),
    )
    base.write_text(lines)
    return {
        "work_dir": str(work),
        "files": [
            {
                "name": f"{stem}.txt",
                "current": str(current),
                "base": str(base),
                "target": str(current),
                "hunks": [
                    {"bs": 2, "bc": 1, "cs": 2, "cc": 1},
                    {"bs": 10, "bc": 1, "cs": 10, "cc": 1},
                ],
            }
        ],
    }


class TestStaleMergeLock(IsolatedAsyncioTestCase):
    """A waiter on a rotated lock must not act on the NEW review."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bh9-a3-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.work = Path(self.tmpdir) / "work"
        self.work.mkdir()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=str(self.work),
        )
        loop = asyncio.get_running_loop()
        self.server._loop = loop
        self.executor = _GateExecutor()
        loop.set_default_executor(self.executor)

    async def asyncTearDown(self) -> None:
        self.executor.all_done_gate.set()
        if self.executor.first_reject_gate is not None:
            self.executor.first_reject_gate.set()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_waiter_on_rotated_lock_does_not_race_new_review(
        self,
    ) -> None:
        """The queued waiter must serialise with the new review's lock."""
        tab_id = "tab-a3"
        server = self.server
        server._register_merge_state(tab_id, _one_hunk_data(self.work, "r1"))
        # The lock object H (and the queued waiter W) will use.
        lock1 = server._merge_action_lock(tab_id)

        cmd = {"type": "mergeAction", "action": "reject", "tabId": tab_id}
        # Install a gate that blocks H's reject-executor work until we
        # can prove W has queued on ``lock1``.  Without it, H's single
        # ungated reject can complete, pop the state and dispatch
        # all-done before W ever runs its membership check — W would
        # then legitimately return early and never enter the race we
        # want to exercise.  The gate is consumed on H's dispatch so
        # W's later reject (in the second phase) is not blocked.
        first_reject_gate = threading.Event()
        self.executor.first_reject_gate = first_reject_gate
        # H completes review 1 (single hunk): pops state+lock entry,
        # then suspends in the gated all-done dispatch, HOLDING lock1.
        h_task = asyncio.ensure_future(
            server._handle_web_merge_action(dict(cmd)),
        )
        # W passes the membership check while review 1 still exists and
        # queues on lock1 (H is inside its reject executor await).
        w_task = asyncio.ensure_future(
            server._handle_web_merge_action(dict(cmd)),
        )

        def _has_pending_waiter() -> bool:
            waiters = getattr(lock1, "_waiters", None) or ()
            return any(not w.done() for w in waiters)

        # Wait until W has queued on lock1 (H is parked in reject).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if lock1.locked() and _has_pending_waiter():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(
            _has_pending_waiter(),
            "W must have queued on lock1 before H proceeds",
        )
        # Release H's reject; H now finishes review 1's on-disk work,
        # pops the state and parks in the gated all-done dispatch
        # while still holding lock1.
        first_reject_gate.set()

        # Wait until H popped the state (it is now parked in all-done).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with server._merge_states_lock:
                popped = tab_id not in server._merge_states
            if popped and lock1.locked():
                break
            await asyncio.sleep(0.01)
        with server._merge_states_lock:
            self.assertNotIn(tab_id, server._merge_states)
        self.assertTrue(lock1.locked(), "H must still hold the old lock")
        self.assertFalse(w_task.done(), "W must still be queued on lock1")

        # The agent starts a NEW review for the same tab while H is
        # still parked in all-done (real registration path).
        server._register_merge_state(tab_id, _two_hunk_data(self.work, "r2"))
        with server._merge_states_lock:
            state2 = server._merge_states[tab_id]

        # X arrives via the normal path and mints a FRESH lock.
        self.executor.reject_barrier = threading.Barrier(2)
        x_task = asyncio.ensure_future(
            server._handle_web_merge_action(dict(cmd)),
        )
        await asyncio.sleep(0.05)

        # Release H: it finishes all-done and releases lock1; W wakes
        # on the STALE lock object.
        self.executor.all_done_gate.set()
        await asyncio.wait_for(
            asyncio.gather(h_task, w_task, x_task), timeout=15,
        )

        # Serialised correctly, W and X reject DISTINCT hunks; under
        # the stale-lock race both reject hunk (0, 0) and hunk (0, 1)
        # is lost forever.
        self.assertTrue(state2.is_resolved(0, 0))
        self.assertTrue(
            state2.is_resolved(0, 1),
            "BUG: the waiter acquired the STALE (rotated) per-tab lock "
            "and raced the new review's fresh lock — the second hunk's "
            "rejection was lost",
        )
        self.assertEqual(state2.remaining, 0)
        with server._merge_states_lock:
            self.assertNotIn(
                tab_id, server._merge_states,
                "the completed second review must have been popped",
            )


if __name__ == "__main__":
    unittest.main()
